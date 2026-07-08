"""
MMPA-DRP Stage 2 & 3 Training (Program-aware Fusion)
====================================================
Drop-in replacement for finetune.fusion_taining.
Uses ProgramFusionMLP instead of Fusion_MLP.

Cleaned version supports:
  - CDAN domain adversarial (always on)
  - Optional sample-level OT alignment
  - Optional domain separation (Phase 18)
  - Optional Fragment Program Interaction (FPI)
  - Standard contrastive domain isolation (Stage 3)
"""

import os
import torch
import numpy as np
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset

from mmpa_drp.models import ProgramFusionMLP
from mmpa_drp.ot_utils import DrugAdaptiveOTLoss
from finetune import (
    fusion_model_train_step,
    fusion_eval_epoch,
    get_featuers,
    get_gene_feat_sim,
)
from myloss import contrastive_loss


def _fusion_model_train_step_with_fpi(
    model, batch, drug_feat, device, optimizer, loss_fn,
    frag_fp=None, frag_mask=None, frag_smiles=None,
    lambda_fpi_gamma_l1=0.001,
    lambda_fpi_entropy=0.0,
    alpha=1.0,
    ot_loss_module=None,
    current_epoch=None,
    drug_names=None,
):
    """
    Training step with optional FPI residual branch and optional sample-level OT.
    FPI does not change h_pool, so OT can safely use gene_feat.
    """
    model.train()
    optimizer.zero_grad()

    s_x = batch[0][0].to(device)
    s_y = batch[0][1].to(device)
    s_mask = batch[0][2].to(device)
    s_mask = (s_mask > 0)
    s_tissue = batch[0][3].to(device)

    s_outputs = model(
        s_x, drug_feat, s_tissue, alpha=alpha,
        return_dict=True,
        frag_fp=frag_fp, frag_mask=frag_mask, frag_smiles=frag_smiles,
        current_epoch=current_epoch,
        drug_names=drug_names,
    )
    s_yp = s_outputs["pred"]
    s_cdan_domain_output = s_outputs["cdan_domain_output"]
    source_gene_feat = s_outputs["gene_feat"]

    s_yp = s_yp.reshape(s_y.shape)
    s_loss_mat = loss_fn(s_yp, s_y.double())
    s_loss_mat = torch.where(
        s_mask, s_loss_mat,
        torch.zeros(s_loss_mat.shape).to(device).to(s_loss_mat.dtype))
    s_loss = torch.sum(s_loss_mat) / torch.sum(s_mask)

    t_x = batch[1][0].to(device)
    t_tissue = batch[1][3].to(device)

    t_outputs = model(
        t_x, drug_feat, t_tissue, alpha=alpha,
        return_dict=True,
        frag_fp=frag_fp, frag_mask=frag_mask, frag_smiles=frag_smiles,
        current_epoch=current_epoch,
        drug_names=drug_names,
    )
    t_cdan_domain_output = t_outputs["cdan_domain_output"]
    target_gene_feat = t_outputs["gene_feat"]

    # CDAN
    if s_cdan_domain_output is not None and t_cdan_domain_output is not None:
        s_domain_labels = torch.zeros(s_cdan_domain_output.size(0), 1).to(device)
        t_domain_labels = torch.ones(t_cdan_domain_output.size(0), 1).to(device)
        domain_outputs = torch.cat([s_cdan_domain_output, t_cdan_domain_output], dim=0)
        domain_labels = torch.cat([s_domain_labels, t_domain_labels], dim=0)
        domain_criterion = torch.nn.BCEWithLogitsLoss()
        cdan_domain_loss = domain_criterion(domain_outputs, domain_labels)
    else:
        cdan_domain_loss = torch.tensor(0.0).to(device)

    # Sample-level OT alignment loss
    sample_ot_loss = torch.tensor(0.0, device=device)
    if ot_loss_module is not None:
        sample_ot_loss, *_ = ot_loss_module(
            source_feat=source_gene_feat,
            target_feat=target_gene_feat,
            drug_names=drug_names,
        )

    # FPI regularization losses
    fpi_loss = torch.tensor(0.0, device=device)
    fpi_diag = {}

    fpi_out = s_outputs.get("fpi_out", None)
    if fpi_out is not None:
        gamma = fpi_out["gamma"]
        attn = fpi_out["attn"]
        residual_logits = fpi_out["residual_logits"]

        if lambda_fpi_gamma_l1 > 0:
            gamma_l1 = lambda_fpi_gamma_l1 * gamma.abs().mean()
            fpi_loss = fpi_loss + gamma_l1
            fpi_diag["gamma_l1"] = gamma_l1.item()

        if lambda_fpi_entropy > 0 and attn is not None:
            eps = 1e-8
            entropy = -(attn * torch.log(attn + eps)).sum(dim=(2, 3)).mean()
            fpi_loss = fpi_loss + lambda_fpi_entropy * entropy
            fpi_diag["entropy"] = entropy.item()

        fpi_diag["gamma_mean"] = gamma.mean().item()
        fpi_diag["gamma_max"] = gamma.max().item()
        fpi_diag["residual_logits_norm"] = residual_logits.abs().mean().item()
        fpi_diag["attention_entropy"] = (-(attn * torch.log(attn + 1e-8)).sum(dim=(2, 3))).mean().item()
        fpi_diag["top_pair_attention_mean"] = attn.reshape(attn.size(0), attn.size(1), -1).max(dim=-1)[0].mean().item()

    loss = s_loss + cdan_domain_loss + sample_ot_loss + fpi_loss
    loss.backward()
    optimizer.step()

    fpi_diag["sample_ot_loss"] = sample_ot_loss.item()
    fpi_diag["cdan_loss"] = cdan_domain_loss.item()
    fpi_diag["response_loss"] = s_loss.item()
    return loss, fpi_diag


def _fusion_model_train_step_with_contrastive(
    model, batch, drug_feat, device, optimizer, all_target_fusion_feat, loss_fn,
    beta=0.1,
):
    """
    Local contrastive train step with configurable beta.
    """
    model.train()
    optimizer.zero_grad()

    s_x = batch[0][0].to(device)
    s_y = batch[0][1].to(device)
    s_mask = batch[0][2].to(device)
    s_mask = (s_mask > 0)
    s_tissue = batch[0][3].to(device)
    s_topk_idx = batch[0][5].to(device)
    s_topk_sim = batch[0][4].to(device) if batch[0][4] is not None else None

    s_outputs = model(s_x, drug_feat, s_tissue)
    s_yp = s_outputs[0]
    s_fusion_feat = s_outputs[2]

    s_yp = s_yp.reshape(s_y.shape)
    s_loss_mat = loss_fn(s_yp, s_y.double())
    s_loss_mat = torch.where(
        s_mask, s_loss_mat,
        torch.zeros(s_loss_mat.shape).to(device).to(s_loss_mat.dtype))
    cls_loss = torch.sum(s_loss_mat) / torch.sum(s_mask)

    # Contrastive loss
    B = s_y.size(0)
    drug_num = s_y.size(1)
    D = s_fusion_feat.size(-1)

    fusion_feat = s_fusion_feat.view(B, drug_num, D)
    anchor = fusion_feat.reshape(-1, D)

    B, drug_num, K = s_topk_idx.shape
    M, _, D = all_target_fusion_feat.shape

    topk_idx_flat = s_topk_idx.reshape(-1, K)
    drug_ids = torch.arange(drug_num, device=s_topk_idx.device).unsqueeze(0).expand(B, drug_num)
    drug_ids_flat = drug_ids.reshape(-1, 1).expand(-1, K)

    anchor_z = model.encode_contrastive_feature(anchor, sample=True)["z"]

    t_flat = all_target_fusion_feat.reshape(M * drug_num, D)
    all_target_z = model.encode_contrastive_feature(t_flat, sample=True)["z"].reshape(M, drug_num, -1)

    pos_samples = all_target_z[topk_idx_flat, drug_ids_flat, :]
    neg_samples = all_target_z.reshape(-1, all_target_z.size(-1))

    contrastive_loss_val = contrastive_loss(anchor_z, pos_samples, neg_samples)

    loss = cls_loss + beta * contrastive_loss_val
    loss.backward()
    optimizer.step()
    return loss, {"contrastive_loss": contrastive_loss_val.item()}


def fusion_training_program(
    encoder,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    drugs,
    gene_num,
    drug_feat,
    fold=0,
    method_name="unknown",
    **kwargs,
):
    """
    Program-aware Stage 2 & 3 training.
    """
    drug_feat_out_dim = 64
    bilinear_dim = 128

    # Training hyperparameters
    alpha = kwargs.get('alpha', kwargs.get('alph', 1.0))
    beta = kwargs.get('beta', kwargs.get('bet', 0.1))
    topk_contrastive = kwargs.get('topk_contrastive', kwargs.get('K', 7))

    # OT alignment parameters
    use_ot_alignment = kwargs.get('use_ot_alignment', False)
    lambda_ot = kwargs.get('lambda_ot', 0.01)
    ot_reg = kwargs.get('ot_reg', 0.1)
    ot_max_iter = kwargs.get('ot_max_iter', 50)
    ot_cost_type = kwargs.get('ot_cost_type', 'cosine')
    ot_marginal_type = kwargs.get('ot_marginal_type', 'uniform')

    # FPI parameters
    use_fpi = kwargs.get('use_fpi', False)
    fpi_config = kwargs.get('fpi_config', None)
    frag_fp = kwargs.get('frag_fp', None)
    frag_mask = kwargs.get('frag_mask', None)
    frag_smiles = kwargs.get('frag_smiles', None)
    lambda_fpi_gamma_l1 = kwargs.get('lambda_fpi_gamma_l1', 0.001)
    lambda_fpi_entropy = kwargs.get('lambda_fpi_entropy', 0.0)

    # Domain separation (only affects validation domain_idx)
    use_domain_sep = kwargs.get('use_domain_sep', False)

    print("=" * 60)
    print("Effective training parameters:")
    print(f"  alpha = {alpha}")
    print(f"  beta = {beta}")
    print(f"  topk_contrastive = {topk_contrastive}")
    print(f"  use_ot_alignment = {use_ot_alignment}")
    print(f"  lambda_ot = {lambda_ot}")
    print(f"  use_fpi = {use_fpi}")
    print(f"  lambda_fpi_gamma_l1 = {lambda_fpi_gamma_l1}")
    print(f"  use_domain_sep = {use_domain_sep}")
    print("=" * 60)

    # Instantiate OT loss if needed
    ot_loss_module = None
    if use_ot_alignment:
        ot_loss_module = DrugAdaptiveOTLoss(
            mode='global',
            lambda_global=lambda_ot,
            reg=ot_reg,
            max_iter=ot_max_iter,
            cost_type=ot_cost_type,
            marginal_type=ot_marginal_type,
        ).to(kwargs["device"])

    frag_fp_tensor = frag_fp.to(kwargs["device"]) if frag_fp is not None else None
    frag_mask_tensor = frag_mask.to(kwargs["device"]) if frag_mask is not None else None

    input_dim = kwargs["encoder_hidden_dims"][-1] + drug_feat_out_dim + bilinear_dim

    fusion_model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=input_dim,
        output_dim=1,
        hidden_dims=[256, 128, 64],
        gene_num=gene_num,
        bilinear_dim=bilinear_dim,
        gene_feat_dim=kwargs["encoder_hidden_dims"][-1],
        drug_feat_dim=drug_feat_out_dim,
        drug_num=len(drugs),
        drop=0.1,
        use_fpi=use_fpi,
        fpi_config=fpi_config,
        frag_fp=frag_fp_tensor,
        frag_mask=frag_mask_tensor,
    ).to(kwargs["device"])

    fusion_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, fusion_model.parameters()),
        lr=kwargs["lr"]
    )
    classifier_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    print(
        f"fusion_training_program (MMPA Stage 2, "
        f"use_ot={use_ot_alignment}, use_fpi={use_fpi}, use_domain_sep={use_domain_sep})"
    )
    batch_size = kwargs["batch_size"]
    cls_best_auc = 0

    fpi_diag_records = []
    fpi_diag_csv = os.path.join(kwargs.get("output_dir", "outputs"), "fpi_diagnostics.csv")

    # Stage 2: Classification + CDAN (+ optional OT + optional FPI)
    for cls_epoch in range(kwargs["train_num_epochs"]):
        train_loss = 0
        epoch_sample_ot_loss = 0.0
        epoch_cdan_loss = 0.0
        epoch_response_loss = 0.0
        fusion_model.train()
        for step, batch in enumerate(zip(train_dataloader, cycle(test_dataloader))):
            if use_fpi:
                loss_val, fpi_diag = _fusion_model_train_step_with_fpi(
                    model=fusion_model,
                    batch=batch,
                    drug_feat=drug_feat,
                    device=kwargs["device"],
                    optimizer=fusion_optimizer,
                    loss_fn=classifier_loss,
                    frag_fp=frag_fp_tensor,
                    frag_mask=frag_mask_tensor,
                    frag_smiles=frag_smiles,
                    lambda_fpi_gamma_l1=lambda_fpi_gamma_l1,
                    lambda_fpi_entropy=lambda_fpi_entropy,
                    alpha=alpha,
                    ot_loss_module=ot_loss_module,
                    current_epoch=cls_epoch,
                    drug_names=drugs,
                )
                step_diags = {"fpi_diag": fpi_diag}
            else:
                loss_val = fusion_model_train_step(
                    model=fusion_model,
                    batch=batch,
                    drug_feat=drug_feat,
                    device=kwargs["device"],
                    optimizer=fusion_optimizer,
                    loss_fn=classifier_loss,
                )
                step_diags = {}

            train_loss += loss_val.cpu().detach().item()

            fpi_diag = step_diags.get("fpi_diag", {})
            epoch_sample_ot_loss += fpi_diag.get("sample_ot_loss", 0.0)
            epoch_cdan_loss += fpi_diag.get("cdan_loss", 0.0)
            epoch_response_loss += fpi_diag.get("response_loss", 0.0)

            if fpi_diag:
                fpi_record = {
                    "epoch": cls_epoch,
                    "step": step,
                    "gamma_mean": fpi_diag.get("gamma_mean", 0.0),
                    "gamma_max": fpi_diag.get("gamma_max", 0.0),
                    "residual_logits_norm": fpi_diag.get("residual_logits_norm", 0.0),
                    "attention_entropy": fpi_diag.get("attention_entropy", 0.0),
                    "top_pair_attention_mean": fpi_diag.get("top_pair_attention_mean", 0.0),
                    "gamma_l1": fpi_diag.get("gamma_l1", 0.0),
                    "entropy": fpi_diag.get("entropy", 0.0),
                }
                fpi_diag_records.append(fpi_record)

        num_steps = step + 1
        if use_ot_alignment:
            print(
                f"  Epoch {cls_epoch}: response={epoch_response_loss / num_steps:.4f}, "
                f"cdan={epoch_cdan_loss / num_steps:.4f}, "
                f"sample_ot={epoch_sample_ot_loss / num_steps:.4f}"
            )

        fusion_model.eval()
        val_domain_idx = 0 if getattr(fusion_model.encoder, 'use_domain_sep', False) else None
        metrics, valid_loss = fusion_eval_epoch(
            model=fusion_model,
            loader=val_dataloader,
            drug_feat=drug_feat,
            loss_fn=classifier_loss,
            device=kwargs["device"],
            domain_idx=val_domain_idx,
        )

        if metrics[0].mean() > cls_best_auc:
            torch.save(
                fusion_model.state_dict(),
                os.path.join(kwargs["model_save_folder"], f"fusion_model_fold{fold}.pt"),
            )
            cls_best_auc = metrics[0].mean()

    if fpi_diag_records:
        try:
            import pandas as pd
            pd.DataFrame(fpi_diag_records).to_csv(fpi_diag_csv, index=False)
        except Exception:
            pass

    # Stage 3: Contrastive domain isolation
    con_best_auc = 0
    K = topk_contrastive

    for con_epoch in range(kwargs["contrastive_num_epochs"]):
        (
            all_source_gene_feat,
            all_source_fusion_feat,
            all_target_gene_feat,
            all_target_fusion_feat,
        ) = get_featuers(
            fusion_model, train_dataloader, test_dataloader, drugs, drug_feat, kwargs["device"]
        )
        topk_sim, topk_idx = get_gene_feat_sim(
            all_source_fusion_feat, all_target_fusion_feat, len(drugs), K
        )

        datasets = train_dataloader.dataset.tensors
        inputs, label, mask, tissue = datasets[0], datasets[1], datasets[2], datasets[3]
        sim_data = TensorDataset(inputs, label, mask, tissue, topk_sim, topk_idx)
        train_dataloader = DataLoader(sim_data, batch_size=batch_size, shuffle=False)

        train_loss = 0
        fusion_model.train()
        for step, batch in enumerate(zip(train_dataloader, cycle(test_dataloader))):
            loss, step_diag = _fusion_model_train_step_with_contrastive(
                model=fusion_model,
                batch=batch,
                drug_feat=drug_feat,
                device=kwargs["device"],
                optimizer=fusion_optimizer,
                loss_fn=classifier_loss,
                all_target_fusion_feat=all_target_fusion_feat,
                beta=beta,
            )
            train_loss += loss.cpu().detach().item()

        num_steps = step + 1

        fusion_model.eval()
        val_domain_idx = 0 if getattr(fusion_model.encoder, 'use_domain_sep', False) else None
        metrics, valid_loss = fusion_eval_epoch(
            model=fusion_model,
            loader=val_dataloader,
            drug_feat=drug_feat,
            loss_fn=classifier_loss,
            device=kwargs["device"],
            domain_idx=val_domain_idx,
        )

        if metrics[0].mean() > con_best_auc:
            torch.save(
                fusion_model.state_dict(),
                os.path.join(kwargs["model_save_folder"], f"fusion_model_fold{fold}.pt"),
            )
            con_best_auc = metrics[0].mean()

    return fusion_model
