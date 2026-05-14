"""
MMPA-DRP Stage 2 & 3 Training (Program-aware Fusion)
====================================================
Drop-in replacement for finetune.fusion_taining.
Uses ProgramFusionMLP instead of Fusion_MLP.
All train/eval step functions remain compatible because
ProgramFusionMLP.forward returns the same tuple signature as Fusion_MLP.
"""

import os
import torch
import numpy as np
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset

from mmpa_drp.models import ProgramFusionMLP
from finetune import (
    fusion_model_train_step,
    fusion_model_train_step_with_contrastive,
    fusion_eval_epoch,
    get_featuers,
    get_gene_feat_sim,
)


from mmpa_drp.losses import (
    entropy_floor_loss, gate_load_balance_loss, gate_diversity_loss,
    drug_attention_diversity_loss, drug_attention_entropy_loss,
    program_usage_balance_loss, drug_gram_consistency_loss,
)
from mmpa_drp.ot_utils import ot_alignment_loss, DrugAdaptiveOTLoss, DrugConditionedOTLoss, ProgramOTLoss, q_align_loss
from mmpa_drp.grad_diagnostics import diagnose_program_ot_gradients


def _compute_gate_losses(model, lambda_gate_entropy_floor, gate_min_entropy,
                         lambda_gate_balance, lambda_gate_diversity):
    """
    Compute all gate regularization losses from the model's current gate state.
    Returns dict of losses.
    """
    losses = {}
    if not getattr(model, 'use_drug_gate', False):
        return losses
    g = getattr(model, 'last_gate_probs', None)
    if g is None:
        return losses

    if lambda_gate_entropy_floor > 0:
        losses['entropy_floor'] = lambda_gate_entropy_floor * entropy_floor_loss(g, min_entropy=gate_min_entropy)
    if lambda_gate_balance > 0:
        losses['load_balance'] = lambda_gate_balance * gate_load_balance_loss(g)
    if lambda_gate_diversity > 0:
        losses['diversity'] = lambda_gate_diversity * gate_diversity_loss(g)
    return losses


def _fusion_model_train_step_with_gate(
    model, batch, drug_feat, device, optimizer, loss_fn,
    lambda_gate_entropy_floor=0.0, gate_min_entropy=1.2,
    lambda_gate_balance=0.0, lambda_gate_diversity=0.0,
    alpha=1.0,
):
    """
    Extended fusion train step that supports Phase 3B-R gate losses.
    """
    model.train()
    optimizer.zero_grad()

    s_x = batch[0][0].to(device)
    s_y = batch[0][1].to(device)
    s_mask = batch[0][2].to(device)
    s_mask = (s_mask > 0)
    s_tissue = batch[0][3].to(device)

    s_outputs = model(s_x, drug_feat, s_tissue, alpha=alpha)
    s_yp = s_outputs[0]
    s_cdan_domain_output = s_outputs[4]

    s_yp = s_yp.reshape(s_y.shape)
    s_loss_mat = loss_fn(s_yp, s_y.double())
    s_loss_mat = torch.where(
        s_mask, s_loss_mat,
        torch.zeros(s_loss_mat.shape).to(device).to(s_loss_mat.dtype))
    s_loss = torch.sum(s_loss_mat) / torch.sum(s_mask)

    t_x = batch[1][0].to(device)
    t_tissue = batch[1][3].to(device)

    t_outputs = model(t_x, drug_feat, s_tissue, alpha=alpha)
    t_cdan_domain_output = t_outputs[4]

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

    # Gate regularization losses
    gate_losses = _compute_gate_losses(
        model, lambda_gate_entropy_floor, gate_min_entropy,
        lambda_gate_balance, lambda_gate_diversity,
    )
    gate_loss = sum(gate_losses.values()) if gate_losses else torch.tensor(0.0).to(device)

    loss = s_loss + cdan_domain_loss + gate_loss
    loss.backward()
    optimizer.step()
    return loss


def _fusion_model_train_step_with_ot(
    model, batch, drug_feat, device, optimizer, loss_fn,
    ot_loss_module=None,
    program_ot_loss_module=None,
    dcot_loss_module=None,
    lambda_ot=0.01, ot_reg=0.1, ot_max_iter=50,
    ot_cost_type="cosine", ot_marginal_type="uniform",
    use_cdan=True,
    drugs=None,
    lambda_program_ot=0.01,
    q_align_enabled=False,
    lambda_q_align=0.001,
    enable_grad_diagnostics=False,
    grad_diag_every=1,
    step=0,
    alpha=1.0,
    drug_conditioned_ot_config=None,
):
    """
    Training step with OT alignment loss (training-only, no inference change).
    Supports optional CDAN disable for ablation.
    Phase 4: supports DrugAdaptiveOTLoss.
    Phase 7: supports DrugConditionedOTLoss.
    """
    model.train()
    optimizer.zero_grad()

    s_x = batch[0][0].to(device)
    s_y = batch[0][1].to(device)
    s_mask = batch[0][2].to(device)
    s_mask = (s_mask > 0)
    s_tissue = batch[0][3].to(device)

    need_encoder_out = (program_ot_loss_module is not None) or \
                       (dcot_loss_module is not None and drug_conditioned_ot_config is not None and drug_conditioned_ot_config.get("enabled", False))

    s_outputs = model(s_x, drug_feat, s_tissue, alpha=alpha, return_encoder_out=need_encoder_out, return_fused_feature=(dcot_loss_module is not None))
    if isinstance(s_outputs, dict):
        s_yp = s_outputs["pred"]
        s_cdan_domain_output = s_outputs["cdan_domain_output"]
        s_encoder_out = s_outputs.get("encoder_out", None)
        source_gene_feat = s_outputs["gene_feat"]
        source_fused = s_outputs.get("fused_feature", None)
    else:
        s_yp = s_outputs[0]
        s_cdan_domain_output = s_outputs[4]
        s_encoder_out = s_outputs[6] if need_encoder_out else None
        source_gene_feat = s_outputs[1]
        source_fused = None

    s_yp = s_yp.reshape(s_y.shape)
    s_loss_mat = loss_fn(s_yp, s_y.double())
    s_loss_mat = torch.where(
        s_mask, s_loss_mat,
        torch.zeros(s_loss_mat.shape).to(device).to(s_loss_mat.dtype))
    s_loss = torch.sum(s_loss_mat) / torch.sum(s_mask)

    t_x = batch[1][0].to(device)
    t_tissue = batch[1][3].to(device)

    t_outputs = model(t_x, drug_feat, t_tissue, alpha=alpha, return_encoder_out=need_encoder_out, return_fused_feature=(dcot_loss_module is not None))
    if isinstance(t_outputs, dict):
        t_cdan_domain_output = t_outputs["cdan_domain_output"]
        t_encoder_out = t_outputs.get("encoder_out", None)
        target_gene_feat = t_outputs["gene_feat"]
        target_fused = t_outputs.get("fused_feature", None)
    else:
        t_cdan_domain_output = t_outputs[4]
        t_encoder_out = t_outputs[6] if need_encoder_out else None
        target_gene_feat = t_outputs[1]
        target_fused = None

    # CDAN (optional)
    if use_cdan and s_cdan_domain_output is not None and t_cdan_domain_output is not None:
        s_domain_labels = torch.zeros(s_cdan_domain_output.size(0), 1).to(device)
        t_domain_labels = torch.ones(t_cdan_domain_output.size(0), 1).to(device)
        domain_outputs = torch.cat([s_cdan_domain_output, t_cdan_domain_output], dim=0)
        domain_labels = torch.cat([s_domain_labels, t_domain_labels], dim=0)
        domain_criterion = torch.nn.BCEWithLogitsLoss()
        cdan_domain_loss = domain_criterion(domain_outputs, domain_labels)
    else:
        cdan_domain_loss = torch.tensor(0.0).to(device)

    # Sample-level OT alignment loss (training-only)
    sample_ot_loss = torch.tensor(0.0, device=device)
    if ot_loss_module is not None:
        # Skip sample OT if drug_conditioned_ot replaces it
        skip_sample_ot = False
        if drug_conditioned_ot_config is not None and drug_conditioned_ot_config.get("enabled", False) and drug_conditioned_ot_config.get("replace_sample_ot", False):
            skip_sample_ot = True
        if not skip_sample_ot:
            sample_ot_loss, _, _, _ = ot_loss_module(
                source_feat=source_gene_feat,
                target_feat=target_gene_feat,
                drug_names=drugs,
            )
    else:
        # Fallback to legacy global OT
        sample_ot_loss = ot_alignment_loss(
            source_gene_feat, target_gene_feat,
            reg=ot_reg, max_iter=ot_max_iter,
            cost_type=ot_cost_type, marginal_type=ot_marginal_type,
        )

    # Drug-conditioned OT alignment loss (Phase 7)
    dcot_loss = torch.tensor(0.0, device=device)
    dcot_info = {}
    if dcot_loss_module is not None and source_fused is not None and target_fused is not None:
        dcot_loss, dcot_info = dcot_loss_module(
            source_fused=source_fused,
            target_fused=target_fused,
            drug_names=drugs,
        )

    # Program-level OT alignment loss (training-only)
    program_ot_loss = torch.tensor(0.0, device=device)
    program_ot_diag = {}
    if program_ot_loss_module is not None and s_encoder_out is not None and t_encoder_out is not None:
        source_z = s_encoder_out["z"]  # [B_s, K, d]
        source_q = s_encoder_out["q"]  # [B_s, K]
        target_z = t_encoder_out["z"]  # [B_t, K, d]
        target_q = t_encoder_out["q"]  # [B_t, K]
        program_ot_loss, program_ot_diag = program_ot_loss_module(
            source_z, source_q, target_z, target_q
        )

    # q-distribution alignment (optional)
    q_align_loss_val = torch.tensor(0.0, device=device)
    q_align_diag = {}
    if q_align_enabled and s_encoder_out is not None and t_encoder_out is not None:
        source_q = s_encoder_out["q"]
        target_q = t_encoder_out["q"]
        q_align_loss_raw, q_align_diag = q_align_loss(source_q, target_q)
        q_align_loss_val = lambda_q_align * q_align_loss_raw

    loss = s_loss + cdan_domain_loss + sample_ot_loss + dcot_loss + program_ot_loss + q_align_loss_val

    # Gradient diagnostics (optional, only when enabled to avoid slowdown)
    grad_diag = None
    if enable_grad_diagnostics and step % grad_diag_every == 0:
        encoder_params = list(model.encoder.parameters())
        grad_diag = diagnose_program_ot_gradients(
            s_loss, cdan_domain_loss, sample_ot_loss, program_ot_loss,
            encoder_params, device,
        )

    loss.backward()
    optimizer.step()
    return loss, {
        "program_ot_diag": program_ot_diag,
        "q_align_diag": q_align_diag,
        "grad_diag": grad_diag,
        "dcot_info": dcot_info,
        "dcot_loss": dcot_loss.item(),
        "sample_ot_loss": sample_ot_loss.item(),
        "cdan_loss": cdan_domain_loss.item(),
        "response_loss": s_loss.item(),
    }


def _fusion_model_train_step_with_contrastive_and_gate(
    model, batch, drug_feat, device, optimizer, all_target_fusion_feat, loss_fn,
    lambda_gate_entropy_floor=0.0, gate_min_entropy=1.2,
    lambda_gate_balance=0.0, lambda_gate_diversity=0.0,
    beta=0.1,
    alpha=1.0,
):
    """
    Extended contrastive train step that supports Phase 3B-R gate losses.
    """
    model.train()
    optimizer.zero_grad()

    s_x = batch[0][0].to(device)
    s_y = batch[0][1].to(device)
    s_mask = batch[0][2].to(device)
    s_mask = (s_mask > 0)
    s_tissue = batch[0][3].to(device)
    s_topk_idx = batch[0][5].to(device)

    s_outputs = model(s_x, drug_feat, s_tissue, alpha=alpha)
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

    pos_samples = all_target_fusion_feat[topk_idx_flat, drug_ids_flat, :]
    neg_samples = all_target_fusion_feat.reshape(-1, D)

    from myloss import contrastive_loss
    contrastive_loss_val = contrastive_loss(anchor, pos_samples, neg_samples)

    # Gate regularization losses
    gate_losses = _compute_gate_losses(
        model, lambda_gate_entropy_floor, gate_min_entropy,
        lambda_gate_balance, lambda_gate_diversity,
    )
    gate_loss = sum(gate_losses.values()) if gate_losses else torch.tensor(0.0).to(device)

    loss = cls_loss + beta * contrastive_loss_val + gate_loss
    loss.backward()
    optimizer.step()
    return loss



def _fusion_model_train_step_with_contrastive(
    model, batch, drug_feat, device, optimizer, all_target_fusion_feat, loss_fn,
    beta=0.1,
):
    """
    Local contrastive train step with configurable beta (replaces finetune version).
    """
    model.train()
    optimizer.zero_grad()

    s_x = batch[0][0].to(device)
    s_y = batch[0][1].to(device)
    s_mask = batch[0][2].to(device)
    s_mask = (s_mask > 0)
    s_tissue = batch[0][3].to(device)
    s_topk_idx = batch[0][5].to(device)

    s_outputs = model(s_x, drug_feat, s_tissue)
    s_yp = s_outputs[0]
    s_fusion_feat = s_outputs[2]

    s_yp = s_yp.reshape(s_y.shape)
    s_loss_mat = loss_fn(s_yp, s_y.double())
    s_loss_mat = torch.where(
        s_mask, s_loss_mat,
        torch.zeros(s_loss_mat.shape).to(device).to(s_loss_mat.dtype))
    cls_loss = torch.sum(s_loss_mat) / torch.sum(s_mask)

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

    pos_samples = all_target_fusion_feat[topk_idx_flat, drug_ids_flat, :]
    neg_samples = all_target_fusion_feat.reshape(-1, D)

    from myloss import contrastive_loss
    contrastive_loss_val = contrastive_loss(anchor, pos_samples, neg_samples)

    loss = cls_loss + beta * contrastive_loss_val
    loss.backward()
    optimizer.step()
    return loss


# =============================================================================
# Phase 8: DSPI Training Step
# =============================================================================

def _fusion_model_train_step_with_dspi(
    model, batch, drug_feat, device, optimizer, loss_fn,
    lambda_dspi_attn_diversity=0.0,
    lambda_dspi_attn_entropy=0.0,
    lambda_dspi_program_balance=0.0,
    lambda_dspi_gram_consistency=0.0,
    alpha=1.0,
):
    """
    Training step with DSPI regularization losses.
    """
    model.train()
    optimizer.zero_grad()

    s_x = batch[0][0].to(device)
    s_y = batch[0][1].to(device)
    s_mask = batch[0][2].to(device)
    s_mask = (s_mask > 0)
    s_tissue = batch[0][3].to(device)

    s_outputs = model(s_x, drug_feat, s_tissue, alpha=alpha, return_fused_feature=True)
    if isinstance(s_outputs, dict):
        s_yp = s_outputs["pred"]
        s_cdan_domain_output = s_outputs["cdan_domain_output"]
        source_gene_feat = s_outputs["gene_feat"]
        source_fused = s_outputs.get("fused_feature", None)
    else:
        s_yp = s_outputs[0]
        s_cdan_domain_output = s_outputs[4]
        source_gene_feat = s_outputs[1]
        source_fused = None

    s_yp = s_yp.reshape(s_y.shape)
    s_loss_mat = loss_fn(s_yp, s_y.double())
    s_loss_mat = torch.where(
        s_mask, s_loss_mat,
        torch.zeros(s_loss_mat.shape).to(device).to(s_loss_mat.dtype))
    s_loss = torch.sum(s_loss_mat) / torch.sum(s_mask)

    t_x = batch[1][0].to(device)
    t_tissue = batch[1][3].to(device)

    t_outputs = model(t_x, drug_feat, t_tissue, alpha=alpha, return_fused_feature=True)
    if isinstance(t_outputs, dict):
        t_cdan_domain_output = t_outputs["cdan_domain_output"]
        target_gene_feat = t_outputs["gene_feat"]
        target_fused = t_outputs.get("fused_feature", None)
    else:
        t_cdan_domain_output = t_outputs[4]
        target_gene_feat = t_outputs[1]
        target_fused = None

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

    # DSPI regularization losses
    dspi_loss = torch.tensor(0.0, device=device)
    dspi_diag = {}

    # Get DSPI attention from model (stored as attribute during forward)
    dspi_attn = getattr(model, 'last_dspi_attn', None)
    dspi_h_drug = getattr(model, 'last_dspi_h_drug', None)

    if dspi_attn is not None:
        if lambda_dspi_attn_diversity > 0:
            div_loss = lambda_dspi_attn_diversity * drug_attention_diversity_loss(dspi_attn)
            dspi_loss = dspi_loss + div_loss
            dspi_diag['attn_diversity'] = div_loss.item()

        if lambda_dspi_attn_entropy > 0:
            ent_loss = lambda_dspi_attn_entropy * drug_attention_entropy_loss(dspi_attn)
            dspi_loss = dspi_loss + ent_loss
            dspi_diag['attn_entropy'] = ent_loss.item()

        if lambda_dspi_program_balance > 0:
            bal_loss = lambda_dspi_program_balance * program_usage_balance_loss(dspi_attn)
            dspi_loss = dspi_loss + bal_loss
            dspi_diag['program_balance'] = bal_loss.item()

    if lambda_dspi_gram_consistency > 0 and dspi_h_drug is not None and source_fused is not None and target_fused is not None:
        # Use h_drug from DSPI for gram consistency
        source_h_drug = getattr(model, 'last_dspi_h_drug_source', None)
        target_h_drug = getattr(model, 'last_dspi_h_drug_target', None)
        if source_h_drug is not None and target_h_drug is not None:
            gram_loss = lambda_dspi_gram_consistency * drug_gram_consistency_loss(source_h_drug, target_h_drug)
            dspi_loss = dspi_loss + gram_loss
            dspi_diag['gram_consistency'] = gram_loss.item()

    loss = s_loss + cdan_domain_loss + dspi_loss
    loss.backward()
    optimizer.step()
    return loss, dspi_diag


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
    Mirrors finetune.fusion_taining but instantiates ProgramFusionMLP.
    Phase 3B: supports drug-gated program aggregation.
    Phase 7: supports DrugConditionedOTLoss with drug-adaptive lambda.
    Phase 8: supports DrugSelectiveProgramInteraction (DSPI).
    """
    drug_feat_out_dim = 64
    bilinear_dim = 128
    use_drug_gate = kwargs.get('use_drug_gate', False)
    shared_gate = kwargs.get('shared_gate', False)
    num_programs = kwargs.get('num_programs', 8)
    gate_mode = kwargs.get('gate_mode', 'additive_bias')
    gate_temperature = kwargs.get('gate_temperature', 5.0)
    rho_init = kwargs.get('rho_init', 0.0)
    beta_init = kwargs.get('beta_init', 0.0)
    lambda_gate_entropy_floor = kwargs.get('lambda_gate_entropy_floor', 0.0)
    gate_min_entropy = kwargs.get('gate_min_entropy', 1.2)
    lambda_gate_balance = kwargs.get('lambda_gate_balance', 0.0)
    lambda_gate_diversity = kwargs.get('lambda_gate_diversity', 0.0)

    # Phase 8: DSPI parameters
    use_dspi = kwargs.get('use_dspi', False)
    dspi_config = kwargs.get('dspi_config', None)
    lambda_dspi_attn_diversity = kwargs.get('lambda_dspi_attn_diversity', 0.0)
    lambda_dspi_attn_entropy = kwargs.get('lambda_dspi_attn_entropy', 0.0)
    lambda_dspi_program_balance = kwargs.get('lambda_dspi_program_balance', 0.0)
    lambda_dspi_gram_consistency = kwargs.get('lambda_dspi_gram_consistency', 0.0)

    # OT alignment parameters
    use_ot_alignment = kwargs.get('use_ot_alignment', False)
    lambda_ot = kwargs.get('lambda_ot', 0.01)
    ot_reg = kwargs.get('ot_reg', 0.1)
    ot_max_iter = kwargs.get('ot_max_iter', 50)
    ot_cost_type = kwargs.get('ot_cost_type', 'cosine')
    ot_marginal_type = kwargs.get('ot_marginal_type', 'uniform')
    use_cdan = kwargs.get('use_cdan', True)
    ot_mode = kwargs.get('ot_mode', 'global')
    ot_use_drug_conditioned_feature = kwargs.get('ot_use_drug_conditioned_feature', False)
    ot_per_drug_lambda = kwargs.get('ot_per_drug_lambda', None)

    # Program OT parameters
    use_program_ot = kwargs.get('use_program_ot', False)
    program_ot_mode = kwargs.get('program_ot_mode', 'same_program_uniform')
    lambda_program_ot = kwargs.get('lambda_program_ot', 0.01)
    program_ot_reg = kwargs.get('program_ot_reg', 0.1)
    program_ot_max_iter = kwargs.get('program_ot_max_iter', 50)
    program_ot_cost_type = kwargs.get('program_ot_cost_type', 'cosine')
    program_ot_marginal_type = kwargs.get('program_ot_marginal_type', 'uniform')
    program_ot_detach_q = kwargs.get('program_ot_detach_q_marginal', True)
    program_ot_q_temp = kwargs.get('program_ot_q_temperature', 1.0)
    program_ot_q_power = kwargs.get('program_ot_q_power', 1.0)
    program_ot_min_mass = kwargs.get('program_ot_min_mass', 1e-6)

    # q-align parameters
    q_align_enabled = kwargs.get('q_align_enabled', False)
    lambda_q_align = kwargs.get('lambda_q_align', 0.001)

    # Gradient diagnostics
    enable_grad_diagnostics = kwargs.get('enable_grad_diagnostics', False)
    grad_diag_every = kwargs.get('grad_diag_every', 50)

    # Phase 7: alpha / beta / topk_contrastive parameter fix
    alpha = kwargs.get('alpha', None)
    if alpha is None:
        alpha = kwargs.get('alph', 1.0)
    beta = kwargs.get('beta', None)
    if beta is None:
        beta = kwargs.get('bet', 0.1)
    topk_contrastive = kwargs.get('topk_contrastive', None)
    if topk_contrastive is None:
        topk_contrastive = kwargs.get('K', 7)

    # Phase 7: drug-conditioned OT config
    drug_conditioned_ot_config = kwargs.get('drug_conditioned_ot', None)
    if drug_conditioned_ot_config is None:
        drug_conditioned_ot_config = {
            "enabled": False,
            "mode": "global",
            "replace_sample_ot": False,
            "global_lambda": lambda_ot,
            "reg": ot_reg,
            "cost": ot_cost_type,
        }

    # Print effective parameters (smoke test requirement)
    effective_alpha = alpha
    effective_beta = beta
    effective_topk_contrastive = topk_contrastive
    effective_lambda_ot = lambda_ot
    effective_ot_mode = ot_mode
    print("=" * 60)
    print("Effective training parameters:")
    print(f"  effective_alpha = {effective_alpha}")
    print(f"  effective_beta = {effective_beta}")
    print(f"  effective_topk_contrastive = {effective_topk_contrastive}")
    print(f"  effective_lambda_ot = {effective_lambda_ot}")
    print(f"  effective_ot_mode = {effective_ot_mode}")
    print(f"  drug_conditioned_ot_enabled = {drug_conditioned_ot_config.get('enabled', False)}")
    print(f"  drug_conditioned_ot_mode = {drug_conditioned_ot_config.get('mode', 'none')}")
    print(f"  drug_conditioned_ot_replace_sample_ot = {drug_conditioned_ot_config.get('replace_sample_ot', False)}")
    print("=" * 60)

    # Instantiate ProgramOTLoss if needed
    program_ot_loss_module = None
    if use_program_ot:
        program_ot_loss_module = ProgramOTLoss(
            mode=program_ot_mode,
            lambda_program_ot=lambda_program_ot,
            reg=program_ot_reg,
            max_iter=program_ot_max_iter,
            cost_type=program_ot_cost_type,
            marginal_type=program_ot_marginal_type,
            detach_q_marginal=program_ot_detach_q,
            q_temperature=program_ot_q_temp,
            q_power=program_ot_q_power,
            min_mass=program_ot_min_mass,
        ).to(kwargs["device"])

    # Instantiate DrugAdaptiveOTLoss if needed (legacy sample OT)
    ot_loss_module = None
    effective_use_ot = use_ot_alignment and (not use_program_ot or program_ot_mode != "program_only")
    if effective_use_ot:
        ot_loss_module = DrugAdaptiveOTLoss(
            mode=ot_mode,
            lambda_global=lambda_ot,
            reg=ot_reg,
            max_iter=ot_max_iter,
            cost_type=ot_cost_type,
            marginal_type=ot_marginal_type,
            use_drug_conditioned_feature=ot_use_drug_conditioned_feature,
            per_drug_lambda=ot_per_drug_lambda,
        ).to(kwargs["device"])

    # Instantiate DrugConditionedOTLoss if enabled
    dcot_loss_module = None
    if drug_conditioned_ot_config.get("enabled", False):
        dcot_cfg = drug_conditioned_ot_config
        discrepancy_cfg = dcot_cfg.get("discrepancy", {})
        fixed_lambda = dcot_cfg.get("fixed_lambda", {})
        dcot_loss_module = DrugConditionedOTLoss(
            mode=dcot_cfg.get("mode", "global"),
            global_lambda=dcot_cfg.get("global_lambda", lambda_ot),
            reg=dcot_cfg.get("reg", ot_reg),
            max_iter=ot_max_iter,
            cost=dcot_cfg.get("cost", ot_cost_type),
            marginal=ot_marginal_type,
            fixed_lambda_dict=fixed_lambda if dcot_cfg.get("mode") == "fixed" else None,
            discrepancy_config=discrepancy_cfg if dcot_cfg.get("mode") == "discrepancy" else None,
        ).to(kwargs["device"])

    fusion_model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=kwargs["encoder_hidden_dims"][-1] + drug_feat_out_dim + bilinear_dim,
        output_dim=1,
        hidden_dims=[256, 128, 64],
        gene_num=gene_num,
        bilinear_dim=bilinear_dim,
        gene_feat_dim=kwargs["encoder_hidden_dims"][-1],
        drug_feat_dim=drug_feat_out_dim,
        drug_num=len(drugs),
        drop=0.1,
        use_drug_gate=use_drug_gate,
        shared_gate=shared_gate,
        num_programs=num_programs,
        gate_mode=gate_mode,
        gate_temperature=gate_temperature,
        rho_init=rho_init,
        beta_init=beta_init,
        # Phase 8: DSPI
        use_dspi=use_dspi,
        dspi_config=dspi_config,
    ).to(kwargs["device"])

    fusion_optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=kwargs["lr"])
    classifier_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    print(f"fusion_training_program (MMPA Stage 2, use_drug_gate={use_drug_gate}, gate_mode={gate_mode}, use_ot={use_ot_alignment})")
    batch_size = kwargs["batch_size"]
    cls_best_auc = 0

    use_gate_step = use_drug_gate and (
        lambda_gate_entropy_floor > 0 or lambda_gate_balance > 0 or lambda_gate_diversity > 0
    )

    # Gate diagnostics logging
    gate_diag_records = []
    diag_csv_path = os.path.join(kwargs.get("output_dir", "outputs"), "gate_diagnostics.csv")

    # Phase 8: DSPI diagnostics logging
    dspi_diag_records = []
    dspi_diag_csv = os.path.join(kwargs.get("output_dir", "outputs"), "dspi_diagnostics.csv")

    # Program OT diagnostics logging
    program_ot_diag_records = []
    program_ot_diag_csv = os.path.join(kwargs.get("output_dir", "outputs"), "program_ot_repair_diagnostics.csv")

    # Gradient diagnostics logging
    grad_diag_records = []
    grad_diag_csv = os.path.join(kwargs.get("output_dir", "outputs"), "program_ot_gradient_diagnostics.csv")

    # Phase 7: DCOT diagnostics logging
    dcot_diag_records = []
    dcot_diag_csv = os.path.join("outputs", "drug_conditioned_ot_diagnostics.csv")

    # Stage 2: Classification + CDAN (+ optional OT)
    for cls_epoch in range(kwargs["train_num_epochs"]):
        train_loss = 0
        epoch_dcot_loss = 0.0
        epoch_sample_ot_loss = 0.0
        epoch_cdan_loss = 0.0
        epoch_response_loss = 0.0
        fusion_model.train()
        for step, batch in enumerate(zip(train_dataloader, cycle(test_dataloader))):
            if use_dspi:
                loss_val, dspi_diag = _fusion_model_train_step_with_dspi(
                    model=fusion_model,
                    batch=batch,
                    drug_feat=drug_feat,
                    device=kwargs["device"],
                    optimizer=fusion_optimizer,
                    loss_fn=classifier_loss,
                    lambda_dspi_attn_diversity=lambda_dspi_attn_diversity,
                    lambda_dspi_attn_entropy=lambda_dspi_attn_entropy,
                    lambda_dspi_program_balance=lambda_dspi_program_balance,
                    lambda_dspi_gram_consistency=lambda_dspi_gram_consistency,
                    alpha=alpha,
                )
                step_diags = {"dspi_diag": dspi_diag}
            elif (use_ot_alignment or use_program_ot or drug_conditioned_ot_config.get("enabled", False)) and not use_gate_step:
                loss_val, step_diags = _fusion_model_train_step_with_ot(
                    model=fusion_model,
                    batch=batch,
                    drug_feat=drug_feat,
                    device=kwargs["device"],
                    optimizer=fusion_optimizer,
                    loss_fn=classifier_loss,
                    ot_loss_module=ot_loss_module,
                    program_ot_loss_module=program_ot_loss_module,
                    dcot_loss_module=dcot_loss_module,
                    lambda_ot=lambda_ot,
                    ot_reg=ot_reg,
                    ot_max_iter=ot_max_iter,
                    ot_cost_type=ot_cost_type,
                    ot_marginal_type=ot_marginal_type,
                    use_cdan=use_cdan,
                    drugs=drugs,
                    lambda_program_ot=lambda_program_ot,
                    q_align_enabled=q_align_enabled,
                    lambda_q_align=lambda_q_align,
                    enable_grad_diagnostics=enable_grad_diagnostics,
                    grad_diag_every=grad_diag_every,
                    step=step,
                    alpha=alpha,
                    drug_conditioned_ot_config=drug_conditioned_ot_config,
                )
            elif use_gate_step:
                loss_val = _fusion_model_train_step_with_gate(
                    model=fusion_model,
                    batch=batch,
                    drug_feat=drug_feat,
                    device=kwargs["device"],
                    optimizer=fusion_optimizer,
                    loss_fn=classifier_loss,
                    lambda_gate_entropy_floor=lambda_gate_entropy_floor,
                    gate_min_entropy=gate_min_entropy,
                    lambda_gate_balance=lambda_gate_balance,
                    lambda_gate_diversity=lambda_gate_diversity,
                    alpha=alpha,
                )
                step_diags = {}
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

            epoch_dcot_loss += step_diags.get("dcot_loss", 0.0)
            epoch_sample_ot_loss += step_diags.get("sample_ot_loss", 0.0)
            epoch_cdan_loss += step_diags.get("cdan_loss", 0.0)
            epoch_response_loss += step_diags.get("response_loss", 0.0)

            # Accumulate program OT diagnostics
            pot_diag = step_diags.get("program_ot_diag", {})
            q_diag = step_diags.get("q_align_diag", {})
            if pot_diag or q_diag:
                record = {
                    "epoch": cls_epoch,
                    "stage": "cls",
                    "step": step,
                    "program_ot_mode": program_ot_mode if use_program_ot else "none",
                    "lambda_program_ot": lambda_program_ot if use_program_ot else 0.0,
                    "loss_program_ot_raw": pot_diag.get("ot_dist", 0.0),
                    "loss_program_ot_weighted": pot_diag.get("ot_dist", 0.0) * lambda_program_ot,
                    "loss_q_align_raw": q_diag.get("q_gap_l1", 0.0),
                    "loss_q_align_weighted": q_diag.get("q_gap_l1", 0.0) * lambda_q_align,
                    "transport_cost": pot_diag.get("transport_cost", 0.0),
                    "transport_entropy": pot_diag.get("transport_entropy", 0.0),
                    "cost_mean": pot_diag.get("cost_mean", 0.0),
                    "cost_std": pot_diag.get("cost_std", 0.0),
                    "P_max": pot_diag.get("P_max", 0.0),
                    "P_min": pot_diag.get("P_min", 0.0),
                    "marginal_a_entropy": pot_diag.get("marginal_a_entropy", 0.0),
                    "marginal_b_entropy": pot_diag.get("marginal_b_entropy", 0.0),
                    "marginal_a_min": pot_diag.get("marginal_a_min", 0.0),
                    "marginal_a_max": pot_diag.get("marginal_a_max", 0.0),
                    "marginal_b_min": pot_diag.get("marginal_b_min", 0.0),
                    "marginal_b_max": pot_diag.get("marginal_b_max", 0.0),
                    "q_entropy_source": q_diag.get("q_entropy_source", 0.0),
                    "q_entropy_target": q_diag.get("q_entropy_target", 0.0),
                    "q_gap_l1": q_diag.get("q_gap_l1", 0.0),
                    "q_gap_l2": q_diag.get("q_gap_l2", 0.0),
                }
                program_ot_diag_records.append(record)
            g_diag = step_diags.get("grad_diag")
            if g_diag is not None:
                g_record = {
                    "epoch": cls_epoch,
                    "stage": "cls",
                    "step": step,
                    "grad_norm_response": g_diag.get("grad_norm_response", 0.0),
                    "grad_norm_cdan": g_diag.get("grad_norm_cdan", 0.0),
                    "grad_norm_sample_ot": g_diag.get("grad_norm_sample_ot", 0.0),
                    "grad_norm_program_ot_raw": g_diag.get("grad_norm_program_ot_raw", 0.0),
                    "grad_norm_program_ot_weighted": g_diag.get("grad_norm_program_ot_weighted", 0.0),
                    "grad_norm_total": g_diag.get("grad_norm_total", 0.0),
                    "ratio_program_raw_to_total": g_diag.get("ratio_program_raw_to_total", 0.0),
                    "ratio_program_weighted_to_total": g_diag.get("ratio_program_weighted_to_total", 0.0),
                    "ratio_sample_ot_to_total": g_diag.get("ratio_sample_ot_to_total", 0.0),
                }
                grad_diag_records.append(g_record)

            # Phase 8: DSPI diagnostics accumulation
            dspi_diag = step_diags.get("dspi_diag", {})
            if dspi_diag:
                dspi_record = {
                    "epoch": cls_epoch,
                    "step": step,
                    "attn_diversity": dspi_diag.get("attn_diversity", 0.0),
                    "attn_entropy": dspi_diag.get("attn_entropy", 0.0),
                    "program_balance": dspi_diag.get("program_balance", 0.0),
                    "gram_consistency": dspi_diag.get("gram_consistency", 0.0),
                }
                dspi_diag_records.append(dspi_record)

            # Phase 7: DCOT per-drug diagnostics
            dcot_info = step_diags.get("dcot_info", {})
            if dcot_info and drug_conditioned_ot_config.get("enabled", False):
                loss_per_drug = dcot_info.get("loss_per_drug", None)
                lambda_per_drug = dcot_info.get("lambda_per_drug", None)
                gap_per_drug = dcot_info.get("gap_per_drug", None)
                diag_per_drug = dcot_info.get("diag_per_drug", [])
                source_fused_norm = dcot_info.get("source_fused_norm", None)
                target_fused_norm = dcot_info.get("target_fused_norm", None)
                if loss_per_drug is not None and lambda_per_drug is not None:
                    for d_idx, drug_name in enumerate(drugs):
                        d_diag = diag_per_drug[d_idx] if d_idx < len(diag_per_drug) else {}
                        dcot_diag_records.append({
                            "method": method_name,
                            "fold": fold,
                            "epoch": cls_epoch,
                            "drug_name": drug_name,
                            "gap_d": gap_per_drug[d_idx].item() if gap_per_drug is not None and d_idx < len(gap_per_drug) else 0.0,
                            "lambda_d": lambda_per_drug[d_idx].item() if lambda_per_drug is not None and d_idx < len(lambda_per_drug) else 0.0,
                            "loss_ot_d": loss_per_drug[d_idx].item() if loss_per_drug is not None and d_idx < len(loss_per_drug) else 0.0,
                            "transport_cost": d_diag.get("transport_cost", 0.0),
                            "transport_entropy": d_diag.get("transport_entropy", 0.0),
                            "cost_mean": d_diag.get("cost_mean", 0.0),
                            "cost_std": d_diag.get("cost_std", 0.0),
                            "P_max": d_diag.get("P_max", 0.0),
                            "P_min": d_diag.get("P_min", 0.0),
                            "source_fused_norm": source_fused_norm[d_idx].item() if source_fused_norm is not None and d_idx < len(source_fused_norm) else 0.0,
                            "target_fused_norm": target_fused_norm[d_idx].item() if target_fused_norm is not None and d_idx < len(target_fused_norm) else 0.0,
                        })

        # Log gate diagnostics if available
        if getattr(fusion_model, 'last_gate_diagnostics', None) is not None:
            diag = fusion_model.last_gate_diagnostics
            record = {
                'epoch': cls_epoch,
                'stage': 'cls',
                'rho': diag.get('rho', 0.0).item() if hasattr(diag.get('rho', 0.0), 'item') else float(diag.get('rho', 0.0)),
                'beta': diag.get('beta', 0.0).item() if hasattr(diag.get('beta', 0.0), 'item') else float(diag.get('beta', 0.0)),
                'mean_entropy': diag.get('gate_entropy', torch.tensor(0.0)).mean().item(),
                'min_entropy': diag.get('gate_entropy', torch.tensor(0.0)).min().item(),
                'max_entropy': diag.get('gate_entropy', torch.tensor(0.0)).max().item(),
                'mean_effective_programs': diag.get('effective_programs', torch.tensor(0.0)).mean().item(),
                'max_prob_mean': diag.get('max_prob_per_drug', torch.tensor(0.0)).mean().item(),
            }
            gate_diag_records.append(record)
            if len(gate_diag_records) % 10 == 0 or cls_epoch == kwargs["train_num_epochs"] - 1:
                try:
                    import pandas as pd
                    pd.DataFrame(gate_diag_records).to_csv(diag_csv_path, index=False)
                except Exception:
                    pass

        # Phase 8: Log DSPI diagnostics
        if use_dspi and dspi_diag_records:
            dspi_epoch_records = [r for r in dspi_diag_records if r.get("epoch") == cls_epoch]
            if dspi_epoch_records:
                avg_div = np.mean([r.get("attn_diversity", 0.0) for r in dspi_epoch_records])
                avg_ent = np.mean([r.get("attn_entropy", 0.0) for r in dspi_epoch_records])
                avg_bal = np.mean([r.get("program_balance", 0.0) for r in dspi_epoch_records])
                avg_gram = np.mean([r.get("gram_consistency", 0.0) for r in dspi_epoch_records])
                print(f"  DSPI: div={avg_div:.6f}, ent={avg_ent:.6f}, bal={avg_bal:.6f}, gram={avg_gram:.6f}")

        # Phase 7: Log epoch-level DCOT diagnostics
        num_steps = step + 1
        if drug_conditioned_ot_config.get("enabled", False):
            print(f"  Epoch {cls_epoch}: response={epoch_response_loss/num_steps:.4f}, cdan={epoch_cdan_loss/num_steps:.4f}, sample_ot={epoch_sample_ot_loss/num_steps:.4f}, dcot={epoch_dcot_loss/num_steps:.4f}")

        fusion_model.eval()
        metrics, valid_loss = fusion_eval_epoch(
            model=fusion_model,
            loader=val_dataloader,
            drug_feat=drug_feat,
            loss_fn=classifier_loss,
            device=kwargs["device"],
        )

        if metrics[0].mean() > cls_best_auc:
            torch.save(
                fusion_model.state_dict(),
                os.path.join(kwargs["model_save_folder"], "fusion_model.pt"),
            )
            cls_best_auc = metrics[0].mean()

    # Flush diagnostics
    if program_ot_diag_records:
        try:
            import pandas as pd
            pd.DataFrame(program_ot_diag_records).to_csv(program_ot_diag_csv, index=False)
        except Exception:
            pass
    if grad_diag_records:
        try:
            import pandas as pd
            pd.DataFrame(grad_diag_records).to_csv(grad_diag_csv, index=False)
        except Exception:
            pass
    if dcot_diag_records:
        try:
            import pandas as pd
            pd.DataFrame(dcot_diag_records).to_csv(dcot_diag_csv, index=False)
        except Exception:
            pass
    if dspi_diag_records:
        try:
            import pandas as pd
            pd.DataFrame(dspi_diag_records).to_csv(dspi_diag_csv, index=False)
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
            all_source_gene_feat, all_target_gene_feat, len(drugs), K
        )
        datasets = train_dataloader.dataset.tensors
        inputs, label, mask, tissue = datasets[0], datasets[1], datasets[2], datasets[3]
        sim_data = TensorDataset(inputs, label, mask, tissue, topk_sim, topk_idx)
        train_dataloader = DataLoader(sim_data, batch_size=batch_size, shuffle=False)

        train_loss = 0
        fusion_model.train()
        for step, batch in enumerate(zip(train_dataloader, cycle(test_dataloader))):
            if use_gate_step:
                loss = _fusion_model_train_step_with_contrastive_and_gate(
                    model=fusion_model,
                    batch=batch,
                    drug_feat=drug_feat,
                    device=kwargs["device"],
                    optimizer=fusion_optimizer,
                    loss_fn=classifier_loss,
                    all_target_fusion_feat=all_target_fusion_feat,
                    lambda_gate_entropy_floor=lambda_gate_entropy_floor,
                    gate_min_entropy=gate_min_entropy,
                    lambda_gate_balance=lambda_gate_balance,
                    lambda_gate_diversity=lambda_gate_diversity,
                    beta=beta,
                    alpha=alpha,
                )
            else:
                loss = _fusion_model_train_step_with_contrastive(
                    model=fusion_model,
                    batch=batch,
                    drug_feat=drug_feat,
                    device=kwargs["device"],
                    optimizer=fusion_optimizer,
                    loss_fn=classifier_loss,
                    all_target_fusion_feat=all_target_fusion_feat,
                    beta=beta,
                )
            train_loss += (loss.cpu().detach().item() if isinstance(loss, torch.Tensor) else loss)

        fusion_model.eval()
        metrics, valid_loss = fusion_eval_epoch(
            model=fusion_model,
            loader=val_dataloader,
            drug_feat=drug_feat,
            loss_fn=classifier_loss,
            device=kwargs["device"],
        )

        if metrics[0].mean() > con_best_auc:
            torch.save(
                fusion_model.state_dict(),
                os.path.join(kwargs["model_save_folder"], "fusion_model.pt"),
            )
            con_best_auc = metrics[0].mean()

    return fusion_model
