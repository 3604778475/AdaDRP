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
import torch.nn.functional as F
import numpy as np
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset

from mmpa_drp.models import ProgramFusionMLP
from mmpa_drp.losses import orthogonality_loss
from finetune import (
    fusion_model_train_step,
    fusion_model_train_step_with_contrastive,
    fusion_eval_epoch,
    get_featuers,
    get_gene_feat_sim,
)
from mmpa_drp.dbscan_filter import dbscan_filter_topk, dbscan_adaptive_neighbors


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


def gram_volume(v1, v2, v3, eps=1e-6):
    """
    计算三个向量张成的平行六面体体积 = sqrt(det(Gram matrix)).
    v1, v2, v3: [..., D]
    返回: [...] 标量体积（共线时→0）
    """
    g11 = (v1 * v1).sum(dim=-1)
    g12 = (v1 * v2).sum(dim=-1)
    g13 = (v1 * v3).sum(dim=-1)
    g22 = (v2 * v2).sum(dim=-1)
    g23 = (v2 * v3).sum(dim=-1)
    g33 = (v3 * v3).sum(dim=-1)
    det = g11 * (g22 * g33 - g23 * g23) \
        - g12 * (g12 * g33 - g13 * g23) \
        + g13 * (g12 * g23 - g13 * g22)
    return torch.sqrt(torch.clamp(det, min=eps))


def _fusion_model_train_step_with_gram(
    model, batch, drug_feat, device, optimizer, loss_fn,
    lambda_gram=1.0,
):
    """
    Stage 3 train step with Gram volume alignment (replaces contrastive loss).
    Minimizes volume(s_mean, t_mean, drug_emb) so the three vectors are collinear.
    """
    model.train()
    optimizer.zero_grad()

    s_x = batch[0][0].to(device)
    s_y = batch[0][1].to(device)
    s_mask = batch[0][2].to(device)
    s_mask = (s_mask > 0)
    s_tissue = batch[0][3].to(device)

    s_outputs = model(s_x, drug_feat, s_tissue)
    s_yp = s_outputs[0]
    s_fusion_feat = s_outputs[2]

    s_yp = s_yp.reshape(s_y.shape)
    s_loss_mat = loss_fn(s_yp, s_y.double())
    s_loss_mat = torch.where(s_mask, s_loss_mat, torch.zeros_like(s_loss_mat))
    cls_loss = torch.sum(s_loss_mat) / torch.sum(s_mask)

    # Target batch (no grad anchor)
    t_x = batch[1][0].to(device)
    t_tissue = batch[1][3].to(device)
    with torch.no_grad():
        t_outputs = model(t_x, drug_feat, t_tissue)
        t_fusion_feat = t_outputs[2]

    B_s = s_x.size(0)
    B_t = t_x.size(0)
    drug_num = s_y.size(1)
    D = s_fusion_feat.size(-1)

    s_fusion = s_fusion_feat.view(B_s, drug_num, D)
    t_fusion = t_fusion_feat.view(B_t, drug_num, D)

    # Per-drug normalized mean
    s_mean = F.normalize(s_fusion.mean(dim=0), dim=-1)   # [drug_num, D]
    t_mean = F.normalize(t_fusion.mean(dim=0), dim=-1)   # [drug_num, D]
    # Align drug_feat dimension with fusion feature via drug_mlp
    with torch.no_grad():
        drug_feat_mlp, _ = model.drug_mlp(drug_feat)
    drug_emb = F.normalize(drug_feat_mlp, dim=-1)        # [drug_num, D]

    gram_loss = gram_volume(s_mean, t_mean, drug_emb).mean()

    total_loss = cls_loss + lambda_gram * gram_loss
    total_loss.backward()
    optimizer.step()

    return total_loss, {"gram_loss": gram_loss.item(), "cls_loss": cls_loss.item()}


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

    t_outputs = model(t_x, drug_feat, t_tissue, alpha=alpha)
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
    # Phase 10-D: DCD-R
    lambda_dcd_gamma_l1=0.0,
    lambda_dcd_ortho=0.0,
    # Phase DCDA
    lambda_dcda_diff=0.1,
    lambda_dcda_align=0.05,
    # Phase 18: Domain separation
    lambda_migrate=0.0,
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

    # Domain indices for Phase 18
    s_domain_idx = torch.zeros(s_x.size(0), dtype=torch.long, device=device) if getattr(model.encoder, 'use_domain_sep', False) else None

    s_outputs = model(s_x, drug_feat, s_tissue, alpha=alpha, return_encoder_out=need_encoder_out, return_fused_feature=(dcot_loss_module is not None), domain_idx=s_domain_idx)
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

    # Phase DCDA: capture source aligned gene
    s_dcda_aligned = None
    s_dcda_diff_loss = torch.tensor(0.0, device=device)
    if getattr(model, "use_dcda", False) and model.last_dcda_info is not None:
        s_dcda_aligned = model.last_dcda_info["aligned_gene"]
        s_dcda_diff_loss = model.last_dcda_info["diff_loss"]

    s_yp = s_yp.reshape(s_y.shape)
    s_loss_mat = loss_fn(s_yp, s_y.double())
    s_loss_mat = torch.where(
        s_mask, s_loss_mat,
        torch.zeros(s_loss_mat.shape).to(device).to(s_loss_mat.dtype))
    s_loss = torch.sum(s_loss_mat) / torch.sum(s_mask)

    t_x = batch[1][0].to(device)
    t_tissue = batch[1][3].to(device)
    t_domain_idx = torch.ones(t_x.size(0), dtype=torch.long, device=device) if getattr(model.encoder, 'use_domain_sep', False) else None

    t_outputs = model(t_x, drug_feat, t_tissue, alpha=alpha, return_encoder_out=need_encoder_out, return_fused_feature=(dcot_loss_module is not None), domain_idx=t_domain_idx)
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

    # Phase DCDA: capture target aligned gene
    t_dcda_aligned = None
    t_dcda_diff_loss = torch.tensor(0.0, device=device)
    if getattr(model, "use_dcda", False) and model.last_dcda_info is not None:
        t_dcda_aligned = model.last_dcda_info["aligned_gene"]
        t_dcda_diff_loss = model.last_dcda_info["diff_loss"]

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

    # Phase 10-D: DCD-R regularization losses
    dcd_loss = torch.tensor(0.0, device=device)
    dcd_diag = {}
    if getattr(model, 'use_dcd_residual', False) and getattr(model, 'last_dcd_out', None) is not None:
        dcd_out = model.last_dcd_out
        gamma = dcd_out["gamma"]
        h_shared = dcd_out["h_shared"]
        h_private = dcd_out["h_private"]
        residual_logits = dcd_out["residual_logits"]

        if lambda_dcd_gamma_l1 > 0:
            gamma_l1 = lambda_dcd_gamma_l1 * gamma.abs().mean()
            dcd_loss = dcd_loss + gamma_l1
            dcd_diag["gamma_l1"] = gamma_l1.item()

        if lambda_dcd_ortho > 0:
            ortho = orthogonality_loss(h_shared, h_private)
            dcd_loss = dcd_loss + lambda_dcd_ortho * ortho
            dcd_diag["ortho"] = ortho.item()

        dcd_diag["gamma_mean"] = gamma.mean().item()
        dcd_diag["gamma_max"] = gamma.max().item()
        dcd_diag["residual_logits_norm"] = residual_logits.abs().mean().item()
        dcd_diag["dcd_effect_abs_mean"] = dcd_out.get("dcd_effect_abs_mean", 0.0).item() if isinstance(dcd_out.get("dcd_effect_abs_mean"), torch.Tensor) else dcd_out.get("dcd_effect_abs_mean", 0.0)

    # Phase DCDA: diffusion loss + domain alignment loss
    dcda_diff_loss_total = s_dcda_diff_loss + t_dcda_diff_loss
    dcda_align_loss = torch.tensor(0.0, device=device)
    if getattr(model, "use_dcda", False) and s_dcda_aligned is not None and t_dcda_aligned is not None:
        D = s_dcda_aligned.size(1)
        for d in range(D):
            s_mean = s_dcda_aligned[:, d].mean(0)
            t_mean = t_dcda_aligned[:, d].mean(0)
            s_std = s_dcda_aligned[:, d].std(0)
            t_std = t_dcda_aligned[:, d].std(0)
            dcda_align_loss = dcda_align_loss + F.mse_loss(s_mean, t_mean) + F.mse_loss(s_std, t_std)

    # Phase 18: Migration loss (private learns from reliable shared predictions)
    migrate_loss = torch.tensor(0.0, device=device)
    if getattr(model, 'use_domain_sep', False) and model.last_domain_sep_pred is not None:
        dsp = model.last_domain_sep_pred
        pred_s = dsp["pred_shared"]  # [B, D]
        pred_p = dsp["pred_private"]  # [B, D]
        rel_w = dsp["rel_weights"]     # [B, D, 2]
        # Soft migration: when shared is reliable (high rel_s), pull private towards shared
        migrate = (rel_w[:, :, 0] * (pred_p - pred_s.detach()).pow(2)).mean()
        migrate_loss = lambda_migrate * migrate

    loss = (
        s_loss
        + cdan_domain_loss
        + sample_ot_loss
        + dcot_loss
        + program_ot_loss
        + q_align_loss_val
        + dcd_loss
        + lambda_dcda_diff * dcda_diff_loss_total
        + lambda_dcda_align * dcda_align_loss
        + migrate_loss
    )

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
    result = {
        "program_ot_diag": program_ot_diag,
        "q_align_diag": q_align_diag,
        "grad_diag": grad_diag,
        "dcot_info": dcot_info,
        "dcot_loss": dcot_loss.item(),
        "sample_ot_loss": sample_ot_loss.item(),
        "cdan_loss": cdan_domain_loss.item(),
        "response_loss": s_loss.item(),
        "dcda_diff_loss": dcda_diff_loss_total.item(),
        "dcda_align_loss": dcda_align_loss.item(),
        "migrate_loss": migrate_loss.item(),
    }
    for k, v in dcd_diag.items():
        result[f"dcd_{k}"] = v
    return loss, result


def compute_vcb_beta(current_epoch, total_epochs, beta_target, warmup_ratio):
    """
    Phase 12: Linear warm-up for VCB KL weight.
    """
    if beta_target <= 0:
        return 0.0
    warmup_epochs = max(1, int(total_epochs * warmup_ratio))
    if current_epoch >= warmup_epochs:
        return beta_target
    return beta_target * float(current_epoch + 1) / float(warmup_epochs)


def _fusion_model_train_step_with_contrastive_and_gate(
    model, batch, drug_feat, device, optimizer, all_target_fusion_feat, loss_fn,
    lambda_gate_entropy_floor=0.0, gate_min_entropy=1.2,
    lambda_gate_balance=0.0, lambda_gate_diversity=0.0,
    beta=0.1,
    alpha=1.0,
    use_vcb=False,
    beta_vcb=0.0,
    # Phase 13: S3R
    use_s3r=False,
    s3r_same_drug_negatives=True,
    s3r_suppress_false_negatives=True,
    s3r_weighted_positives=False,
    s3r_positive_temperature=0.1,
    s3r_max_negatives_per_anchor=0,
    s3r_loss_temperature=0.1,
):
    """
    Extended contrastive train step that supports Phase 3B-R gate losses.
    Phase 12: supports VCB for contrastive feature encoding.
    Phase 13: supports S3R reliable contrastive anchoring.
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
    s_anchor_valid = batch[0][6].to(device) if len(batch[0]) > 6 else None

    s_outputs = model(s_x, drug_feat, s_tissue, alpha=alpha)
    s_yp = s_outputs[0]
    s_fusion_feat = s_outputs[2]

    s_yp = s_yp.reshape(s_y.shape)
    s_loss_mat = loss_fn(s_yp, s_y.double())
    s_loss_mat = torch.where(
        s_mask, s_loss_mat,
        torch.zeros(s_loss_mat.shape).to(device).to(s_loss_mat.dtype))
    cls_loss = torch.sum(s_loss_mat) / torch.sum(s_mask)

    # Phase 17: Counterfactual causal augmentation
    if lambda_causal > 0.0:
        ablation_mask = torch.rand_like(s_x) < causal_ablation_ratio
        s_x_cf = s_x.clone()
        s_x_cf[ablation_mask] = 0.0
        with torch.no_grad():
            cf_outputs = model(s_x_cf, drug_feat, s_tissue)
            s_fusion_feat_cf = cf_outputs[2]
        causal_loss = F.mse_loss(s_fusion_feat, s_fusion_feat_cf)
    else:
        causal_loss = torch.tensor(0.0, device=device)

    # Phase 19: Differentiated contrastive learning (DKC-inspired)
    # Low-confidence (uncertain) samples get higher contrastive weights
    anchor_weights = None
    if use_diff_contrastive:
        with torch.no_grad():
            # s_yp: [B, drug_num] logits; confidence via sigmoid
            confidence = torch.sigmoid(s_yp)  # [B, drug_num]
            # Mask out invalid positions
            confidence = torch.where(s_mask, confidence, torch.ones_like(confidence))
            # Low confidence -> high weight: weight = alpha - (alpha-1) * confidence
            weights = diff_contrastive_alpha - (diff_contrastive_alpha - 1.0) * confidence
            anchor_weights = weights.reshape(-1)  # [B*drug_num]

    # DBSCAN adaptive: skip outlier anchors
    if s_anchor_valid is not None:
        valid_flat = s_anchor_valid.reshape(-1)
        if anchor_weights is not None:
            anchor_weights = anchor_weights * valid_flat
        else:
            anchor_weights = valid_flat

    # Contrastive loss (with optional InfoNCE / hard negatives / pos weighting)
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
    drug_ids_per_anchor = drug_ids.reshape(-1)  # [B*D]

    # Phase 13: S3R reliable contrastive
    if use_s3r:
        from mmpa_drp.losses import build_reliable_contrastive_samples, reliable_contrastive_loss
        topk_sim_flat = s_topk_sim.reshape(-1, K) if s_topk_sim is not None else None
        s3r_samples = build_reliable_contrastive_samples(
            anchor_feat=anchor,
            all_target_feat=all_target_fusion_feat,
            topk_idx=topk_idx_flat,
            drug_ids=drug_ids_per_anchor,
            topk_sim=topk_sim_flat,
            same_drug_negatives=s3r_same_drug_negatives,
            suppress_false_negatives=s3r_suppress_false_negatives,
            max_negatives_per_anchor=s3r_max_negatives_per_anchor,
            weighted_positives=s3r_weighted_positives,
            positive_temperature=s3r_positive_temperature,
        )
        if s3r_samples["anchors"].size(0) > 0:
            con_out = reliable_contrastive_loss(
                anchors=s3r_samples["anchors"],
                positives=s3r_samples["positives"],
                negatives=s3r_samples["negatives"],
                temperature=s3r_loss_temperature,
                positive_weights=s3r_samples["positive_weights"],
            )
            contrastive_loss_val = con_out["loss"]
            s3r_diag = {
                "contrastive_loss": con_out["loss"].item(),
                "s3r_num_valid": con_out["num_valid"],
                "s3r_num_neg_mean": con_out["num_neg_mean"],
                "s3r_pos_sim_mean": con_out["pos_sim_mean"],
                "s3r_neg_sim_mean": con_out["neg_sim_mean"],
            }
        else:
            contrastive_loss_val = torch.tensor(0.0, device=device)
            s3r_diag = {
                "contrastive_loss": 0.0,
                "s3r_num_valid": 0,
                "s3r_num_neg_mean": 0.0,
                "s3r_pos_sim_mean": 0.0,
                "s3r_neg_sim_mean": 0.0,
            }
    else:
        # Legacy contrastive (with optional VCB)
        s_vcb = model.encode_contrastive_feature(anchor, sample=True)
        anchor_z = s_vcb["z"]
        s_kl = s_vcb["kl"]
        Z = anchor_z.size(-1)

        t_flat = all_target_fusion_feat.reshape(M * drug_num, D)
        t_vcb = model.encode_contrastive_feature(t_flat, sample=True)
        all_target_z = t_vcb["z"].reshape(M, drug_num, Z)

        pos_samples = all_target_z[topk_idx_flat, drug_ids_flat, :]
        neg_samples = all_target_z.reshape(-1, Z)

        if use_infonce:
            from myloss import contrastive_loss_infonce
            pos_weights = s_topk_sim.reshape(-1, K) if use_pos_weights and s_topk_sim is not None else None
            contrastive_loss_val = contrastive_loss_infonce(
                anchor_z, pos_samples, neg_samples,
                temperature=0.1,
                pos_weights=pos_weights,
                hard_negatives_only=hard_negatives_only,
                num_hard_negs=num_hard_negs,
                anchor_weights=anchor_weights,
            )
        else:
            from myloss import contrastive_loss
            contrastive_loss_val = contrastive_loss(anchor_z, pos_samples, neg_samples, anchor_weights=anchor_weights)
        s3r_diag = {"contrastive_loss": contrastive_loss_val.item()}

        if use_vcb:
            loss_vcb_kl = s_kl.mean() + t_kl.mean()
            s3r_diag["vcb_kl"] = loss_vcb_kl.item()
            if s_vcb["mu"] is not None:
                s3r_diag["vcb_mu_norm"] = s_vcb["mu"].norm(dim=-1).mean().item() + t_vcb["mu"].norm(dim=-1).mean().item()
                s3r_diag["vcb_logvar_mean"] = s_vcb["logvar"].mean().item() + t_vcb["logvar"].mean().item()
                s3r_diag["vcb_z_std"] = anchor_z.std().item() + t_vcb["z"].std().item()

    # Gate regularization losses
    gate_losses = _compute_gate_losses(
        model, lambda_gate_entropy_floor, gate_min_entropy,
        lambda_gate_balance, lambda_gate_diversity,
    )
    gate_loss = sum(gate_losses.values()) if gate_losses else torch.tensor(0.0).to(device)

    if use_s3r:
        loss = cls_loss + beta * contrastive_loss_val + gate_loss
    else:
        loss_vcb_kl_val = s3r_diag.get("vcb_kl", 0.0)
        loss_vcb_kl = torch.tensor(loss_vcb_kl_val, device=device) if not isinstance(loss_vcb_kl_val, torch.Tensor) else loss_vcb_kl_val
        loss = cls_loss + beta * contrastive_loss_val + gate_loss + beta_vcb * loss_vcb_kl

    loss.backward()
    optimizer.step()
    return loss, s3r_diag



def _fusion_model_train_step_with_contrastive(
    model, batch, drug_feat, device, optimizer, all_target_fusion_feat, loss_fn,
    beta=0.1,
    use_vcb=False,
    beta_vcb=0.0,
    # Phase 13: S3R
    use_s3r=False,
    s3r_same_drug_negatives=True,
    s3r_suppress_false_negatives=True,
    s3r_weighted_positives=False,
    s3r_positive_temperature=0.1,
    s3r_max_negatives_per_anchor=0,
    s3r_loss_temperature=0.1,
    # Phase 17: Counterfactual causal augmentation
    lambda_causal=0.0,
    causal_ablation_ratio=0.1,
    # Phase 18: InfoNCE / hard negative / pos weighting
    use_infonce=False,
    hard_negatives_only=False,
    num_hard_negs=50,
    use_pos_weights=False,
    # Phase 19: Differentiated contrastive learning (DKC-inspired)
    use_diff_contrastive=False,
    diff_contrastive_alpha=2.0,
):
    """
    Local contrastive train step with configurable beta (replaces finetune version).
    Phase 12: supports VCB for contrastive feature encoding.
    Phase 13: supports S3R reliable contrastive anchoring.
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
    s_anchor_valid = batch[0][6].to(device) if len(batch[0]) > 6 else None

    s_outputs = model(s_x, drug_feat, s_tissue)
    s_yp = s_outputs[0]
    s_fusion_feat = s_outputs[2]

    s_yp = s_yp.reshape(s_y.shape)
    s_loss_mat = loss_fn(s_yp, s_y.double())
    s_loss_mat = torch.where(
        s_mask, s_loss_mat,
        torch.zeros(s_loss_mat.shape).to(device).to(s_loss_mat.dtype))
    cls_loss = torch.sum(s_loss_mat) / torch.sum(s_mask)

    # Phase 17: Counterfactual causal augmentation
    if lambda_causal > 0.0:
        ablation_mask = torch.rand_like(s_x) < causal_ablation_ratio
        s_x_cf = s_x.clone()
        s_x_cf[ablation_mask] = 0.0
        with torch.no_grad():
            cf_outputs = model(s_x_cf, drug_feat, s_tissue)
            s_fusion_feat_cf = cf_outputs[2]
        causal_loss = F.mse_loss(s_fusion_feat, s_fusion_feat_cf)
    else:
        causal_loss = torch.tensor(0.0, device=device)

    # Phase 19: Differentiated contrastive learning (DKC-inspired)
    anchor_weights = None
    if use_diff_contrastive:
        with torch.no_grad():
            confidence = torch.sigmoid(s_yp)  # [B, drug_num]
            confidence = torch.where(s_mask, confidence, torch.ones_like(confidence))
            weights = diff_contrastive_alpha - (diff_contrastive_alpha - 1.0) * confidence
            anchor_weights = weights.reshape(-1)  # [B*drug_num]

    # DBSCAN adaptive: skip outlier anchors
    if s_anchor_valid is not None:
        valid_flat = s_anchor_valid.reshape(-1)
        if anchor_weights is not None:
            anchor_weights = anchor_weights * valid_flat
        else:
            anchor_weights = valid_flat

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
    drug_ids_per_anchor = drug_ids.reshape(-1)  # [B*D]

    # Phase 13: S3R reliable contrastive
    if use_s3r:
        from mmpa_drp.losses import build_reliable_contrastive_samples, reliable_contrastive_loss
        topk_sim_flat = s_topk_sim.reshape(-1, K) if s_topk_sim is not None else None
        s3r_samples = build_reliable_contrastive_samples(
            anchor_feat=anchor,
            all_target_feat=all_target_fusion_feat,
            topk_idx=topk_idx_flat,
            drug_ids=drug_ids_per_anchor,
            topk_sim=topk_sim_flat,
            same_drug_negatives=s3r_same_drug_negatives,
            suppress_false_negatives=s3r_suppress_false_negatives,
            max_negatives_per_anchor=s3r_max_negatives_per_anchor,
            weighted_positives=s3r_weighted_positives,
            positive_temperature=s3r_positive_temperature,
        )
        if s3r_samples["anchors"].size(0) > 0:
            con_out = reliable_contrastive_loss(
                anchors=s3r_samples["anchors"],
                positives=s3r_samples["positives"],
                negatives=s3r_samples["negatives"],
                temperature=s3r_loss_temperature,
                positive_weights=s3r_samples["positive_weights"],
            )
            contrastive_loss_val = con_out["loss"]
            s3r_diag = {
                "contrastive_loss": con_out["loss"].item(),
                "s3r_num_valid": con_out["num_valid"],
                "s3r_num_neg_mean": con_out["num_neg_mean"],
                "s3r_pos_sim_mean": con_out["pos_sim_mean"],
                "s3r_neg_sim_mean": con_out["neg_sim_mean"],
            }
        else:
            contrastive_loss_val = torch.tensor(0.0, device=device)
            s3r_diag = {
                "contrastive_loss": 0.0,
                "s3r_num_valid": 0,
                "s3r_num_neg_mean": 0.0,
                "s3r_pos_sim_mean": 0.0,
                "s3r_neg_sim_mean": 0.0,
            }
    else:
        # Legacy contrastive (with optional VCB)
        s_vcb = model.encode_contrastive_feature(anchor, sample=True)
        anchor_z = s_vcb["z"]
        s_kl = s_vcb["kl"]
        Z = anchor_z.size(-1)

        t_flat = all_target_fusion_feat.reshape(M * drug_num, D)
        t_vcb = model.encode_contrastive_feature(t_flat, sample=True)
        all_target_z = t_vcb["z"].reshape(M, drug_num, Z)

        pos_samples = all_target_z[topk_idx_flat, drug_ids_flat, :]
        neg_samples = all_target_z.reshape(-1, Z)

        from myloss import contrastive_loss
        contrastive_loss_val = contrastive_loss(anchor_z, pos_samples, neg_samples, anchor_weights=anchor_weights)
        s3r_diag = {"contrastive_loss": contrastive_loss_val.item()}

        if use_vcb:
            loss_vcb_kl = s_kl.mean() + t_kl.mean()
            s3r_diag["vcb_kl"] = loss_vcb_kl.item()
            if s_vcb["mu"] is not None:
                s3r_diag["vcb_mu_norm"] = s_vcb["mu"].norm(dim=-1).mean().item() + t_vcb["mu"].norm(dim=-1).mean().item()
                s3r_diag["vcb_logvar_mean"] = s_vcb["logvar"].mean().item() + t_vcb["logvar"].mean().item()
                s3r_diag["vcb_z_std"] = anchor_z.std().item() + t_vcb["z"].std().item()

    if use_s3r:
        loss = cls_loss + beta * contrastive_loss_val + lambda_causal * causal_loss
    else:
        loss_vcb_kl_val = s3r_diag.get("vcb_kl", 0.0)
        loss_vcb_kl = torch.tensor(loss_vcb_kl_val, device=device) if not isinstance(loss_vcb_kl_val, torch.Tensor) else loss_vcb_kl_val
        loss = cls_loss + beta * contrastive_loss_val + beta_vcb * loss_vcb_kl + lambda_causal * causal_loss

    if lambda_causal > 0.0:
        s3r_diag["causal_loss"] = causal_loss.item()

    loss.backward()
    optimizer.step()
    return loss, s3r_diag


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


# =============================================================================
# Phase 9: FPI Training Step
# =============================================================================

def _fusion_model_train_step_with_fpi(
    model, batch, drug_feat, device, optimizer, loss_fn,
    frag_fp=None, frag_mask=None, frag_smiles=None,
    lambda_fpi_gamma_l1=0.001,
    lambda_fpi_entropy=0.0,
    alpha=1.0,
    ot_loss_module=None,
    drug_conditioned_ot_config=None,
    dcot_loss_module=None,
    current_epoch=None,
    drug_names=None,
    lambda_router_gamma_l1=0.0,
    lambda_router_gamma_global_l1=None,
    lambda_router_gamma_cross_l1=None,
    # Phase 10-D: DCD-R
    lambda_dcd_gamma_l1=0.0,
    lambda_dcd_ortho=0.0,
    # Phase 14-A / 15-B: Drug-wise Pairwise Ranking Loss
    use_rank_loss=False,
    lambda_rank=0.0,
    rank_margin=0.0,
    rank_max_pairs_per_drug=0,
    # Phase 15-B: Effective Drug-wise Ranking Loss Sweep
    ranking_warmup_epochs=0,
    ranking_min_pos=1,
    ranking_min_neg=1,
    ranking_log_grad_ratio=False,
    ranking_log_drugwise_pair_acc=False,
    ranking_log_interval=50,
    step=None,
    # Phase 16: FAMO
    use_famo=False,
    famo_optimizer=None,
):
    """
    Training step with FPI residual branch.
    Supports optional sample OT and drug-conditioned OT.
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

    # Phase 16: Per-drug BCE for FAMO
    B_src = s_y.size(0)
    D_src = s_y.size(1)
    per_drug_bce = []
    for d in range(D_src):
        drug_mask = s_mask[:, d]
        if drug_mask.sum() > 0:
            bce_d = s_loss_mat[:, d][drug_mask].mean()
        else:
            bce_d = torch.tensor(0.0, device=device)
        per_drug_bce.append(bce_d)
    per_drug_bce_tensor = torch.stack(per_drug_bce) if per_drug_bce else torch.tensor([])

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

    # Sample-level OT alignment loss (training-only)
    sample_ot_loss = torch.tensor(0.0, device=device)
    if ot_loss_module is not None:
        skip_sample_ot = False
        if drug_conditioned_ot_config is not None and drug_conditioned_ot_config.get("enabled", False) and drug_conditioned_ot_config.get("replace_sample_ot", False):
            skip_sample_ot = True
        if not skip_sample_ot:
            sample_ot_loss, _, _, _ = ot_loss_module(
                source_feat=source_gene_feat,
                target_feat=target_gene_feat,
                drug_names=None,  # will use default
            )

    # Drug-conditioned OT alignment loss (Phase 7)
    dcot_loss = torch.tensor(0.0, device=device)
    if dcot_loss_module is not None:
        source_fused = s_outputs.get("fused_feature", None)
        target_fused = t_outputs.get("fused_feature", None)
        if source_fused is not None and target_fused is not None:
            dcot_loss, _ = dcot_loss_module(
                source_fused=source_fused,
                target_fused=target_fused,
                drug_names=None,
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

    # Phase 9-R: FPI Router losses
    router_out = s_outputs.get("router_out", None)
    if router_out is not None:
        gamma_g = router_out["gamma_global"]
        gamma_c = router_out["gamma_cross"]

        # Backward-compatible: if separate L1s not specified, fall back to unified lambda_router_gamma_l1
        eff_lambda_gg_l1 = lambda_router_gamma_global_l1 if lambda_router_gamma_global_l1 is not None else lambda_router_gamma_l1
        eff_lambda_gc_l1 = lambda_router_gamma_cross_l1 if lambda_router_gamma_cross_l1 is not None else lambda_router_gamma_l1

        if eff_lambda_gg_l1 > 0:
            gg_l1 = eff_lambda_gg_l1 * gamma_g.abs().mean()
            fpi_loss = fpi_loss + gg_l1
            fpi_diag["router_gamma_global_l1"] = gg_l1.item()
        if eff_lambda_gc_l1 > 0:
            gc_l1 = eff_lambda_gc_l1 * gamma_c.abs().mean()
            fpi_loss = fpi_loss + gc_l1
            fpi_diag["router_gamma_cross_l1"] = gc_l1.item()

        fpi_diag["router_gamma_global_mean"] = gamma_g.mean().item()
        fpi_diag["router_gamma_cross_mean"] = gamma_c.mean().item()
        fpi_diag["router_gamma_global_max"] = gamma_g.max().item()
        fpi_diag["router_gamma_cross_max"] = gamma_c.max().item()
        fpi_diag["router_global_logits_norm"] = router_out.get("residual_global", torch.tensor(0.0)).abs().mean().item()
        fpi_diag["router_cross_logits_norm"] = router_out.get("residual_cross", torch.tensor(0.0)).abs().mean().item()
        fpi_diag["router_cross_active"] = router_out.get("cross_active", False)

    # Phase 10-D: DCD-R regularization losses
    dcd_loss = torch.tensor(0.0, device=device)
    dcd_diag = {}
    dcd_out = s_outputs.get("dcd_out", None)
    if dcd_out is not None:
        gamma = dcd_out["gamma"]
        h_shared = dcd_out["h_shared"]
        h_private = dcd_out["h_private"]
        residual_logits = dcd_out["residual_logits"]

        if lambda_dcd_gamma_l1 > 0:
            gamma_l1 = lambda_dcd_gamma_l1 * gamma.abs().mean()
            dcd_loss = dcd_loss + gamma_l1
            dcd_diag["gamma_l1"] = gamma_l1.item()

        if lambda_dcd_ortho > 0:
            ortho = orthogonality_loss(h_shared, h_private)
            dcd_loss = dcd_loss + lambda_dcd_ortho * ortho
            dcd_diag["ortho"] = ortho.item()

        dcd_diag["gamma_mean"] = gamma.mean().item()
        dcd_diag["gamma_max"] = gamma.max().item()
        dcd_diag["residual_logits_norm"] = residual_logits.abs().mean().item()
        dcd_diag["dcd_effect_abs_mean"] = dcd_out.get("dcd_effect_abs_mean", 0.0).item() if isinstance(dcd_out.get("dcd_effect_abs_mean"), torch.Tensor) else dcd_out.get("dcd_effect_abs_mean", 0.0)

    # Phase 15-B: Drug-wise Pairwise Ranking Loss (source only)
    rank_loss = torch.tensor(0.0, device=device)
    rank_loss_raw = torch.tensor(0.0, device=device)
    rank_diag = {}
    rank_per_drug_records = []
    lambda_eff = 0.0
    if use_rank_loss:
        from mmpa_drp.losses import drugwise_pairwise_ranking_loss
        # s_y shape [B, D], s_mask shape [B, D]
        # final_logits shape [B*D, 1] -> [B, D]
        source_final_logits = s_outputs.get("final_logits", None)
        if source_final_logits is not None:
            B_src = s_y.size(0)
            D_src = s_y.size(1)
            logits_2d = source_final_logits.view(B_src, D_src)

            # Lambda warm-up
            if current_epoch is not None and ranking_warmup_epochs > 0:
                lambda_eff = lambda_rank * min(1.0, (current_epoch + 1) / ranking_warmup_epochs)
            else:
                lambda_eff = lambda_rank

            # Sanity: log final_logits vs pred statistics
            with torch.no_grad():
                rank_diag["final_logits_mean"] = logits_2d.mean().item()
                rank_diag["final_logits_std"] = logits_2d.std().item()
                rank_diag["final_logits_min"] = logits_2d.min().item()
                rank_diag["final_logits_max"] = logits_2d.max().item()
                pred_2d = s_yp.view(B_src, D_src)
                rank_diag["pred_mean"] = pred_2d.mean().item()
                rank_diag["pred_std"] = pred_2d.std().item()
                rank_diag["pred_min"] = pred_2d.min().item()
                rank_diag["pred_max"] = pred_2d.max().item()

            rank_loss_raw, rank_stats = drugwise_pairwise_ranking_loss(
                logits=logits_2d,
                labels=s_y,
                mask=s_mask,
                missing_value=-1,
                min_pos=ranking_min_pos,
                min_neg=ranking_min_neg,
                drug_names=drug_names,
                margin=rank_margin,
                return_stats=True,
            )
            if lambda_eff > 0.0:
                rank_loss = lambda_eff * rank_loss_raw
            rank_diag.update({
                "rank_loss_raw": rank_loss_raw.item(),
                "rank_loss_weighted": rank_loss.item(),
                "rank_num_valid_drugs": rank_stats["num_valid_drugs"],
                "rank_num_skipped_drugs": rank_stats["num_skipped_drugs"],
                "rank_mean_pair_acc": rank_stats["mean_pair_acc"],
                "rank_mean_pos_logit": rank_stats["mean_pos_logit"],
                "rank_mean_neg_logit": rank_stats["mean_neg_logit"],
                "rank_mean_diff": rank_stats["mean_diff"],
                "rank_lambda_eff": lambda_eff,
            })

            # Per-drug BCE (for comparison with rank loss)
            per_drug_bce = {}
            for d in range(D_src):
                drug_mask = s_mask[:, d]
                if drug_mask.sum() > 0:
                    bce_d = s_loss_mat[:, d][drug_mask].mean().item()
                else:
                    bce_d = float('nan')
                per_drug_bce[d] = bce_d

            # Per-drug diagnostics for CSV
            for stat in rank_stats["per_drug"]:
                d = stat["drug_index"]
                rank_per_drug_records.append({
                    "drug_idx": d,
                    "drug_name": stat["drug_name"],
                    "num_valid": stat["num_valid"],
                    "num_pos": stat["num_pos"],
                    "num_neg": stat["num_neg"],
                    "num_pairs": stat["pair_count"],
                    "bce_d": per_drug_bce.get(d, float('nan')),
                    "rank_loss_d": stat["rank_loss"] if not stat["skipped"] else 0.0,
                    "pair_acc_d": stat["pair_acc"] if not stat["skipped"] else float('nan'),
                    "pos_logit_mean": stat["pos_logit_mean"] if not stat["skipped"] else float('nan'),
                    "neg_logit_mean": stat["neg_logit_mean"] if not stat["skipped"] else float('nan'),
                    "mean_diff": stat["mean_diff"] if not stat["skipped"] else float('nan'),
                    "skipped": stat["skipped"],
                    "valid": not stat["skipped"],
                })

            # Phase 15-B: Gradient ratio logging
            if ranking_log_grad_ratio and step is not None and step % ranking_log_interval == 0 and lambda_eff > 0.0:
                params = [p for p in model.parameters() if p.requires_grad]
                bce_grads = torch.autograd.grad(s_loss, params, retain_graph=True, allow_unused=True)
                bce_norm = torch.sqrt(sum((g ** 2).sum() for g in bce_grads if g is not None))

                if rank_loss_raw.item() != 0.0:
                    rank_raw_grads = torch.autograd.grad(rank_loss_raw, params, retain_graph=True, allow_unused=True)
                    rank_raw_norm = torch.sqrt(sum((g ** 2).sum() for g in rank_raw_grads if g is not None))
                else:
                    rank_raw_norm = torch.tensor(0.0, device=device)

                if rank_loss.item() != 0.0:
                    rank_weighted_grads = torch.autograd.grad(rank_loss, params, retain_graph=True, allow_unused=True)
                    rank_weighted_norm = torch.sqrt(sum((g ** 2).sum() for g in rank_weighted_grads if g is not None))
                else:
                    rank_weighted_norm = torch.tensor(0.0, device=device)

                ratio = (rank_weighted_norm / bce_norm).item() if bce_norm > 0 else 0.0
                rank_diag["rank_grad_bce_norm"] = bce_norm.item()
                rank_diag["rank_grad_rank_raw_norm"] = rank_raw_norm.item()
                rank_diag["rank_grad_rank_weighted_norm"] = rank_weighted_norm.item()
                rank_diag["rank_grad_weighted_ratio"] = ratio

                if ratio < 0.01:
                    print(f"WARNING: weighted ranking gradient is <1% of BCE gradient; lambda_rank={lambda_eff:.4f} may still be ineffective.")
                elif ratio > 0.50:
                    print(f"WARNING: weighted ranking gradient is >50% of BCE gradient; ranking may dominate training.")

    # Phase 16: FAMO loss assembly
    if use_famo and famo_optimizer is not None:
        weighted_bce = famo_optimizer.get_weighted_loss(per_drug_bce_tensor)[0]
        loss = weighted_bce + cdan_domain_loss + sample_ot_loss + dcot_loss + fpi_loss + dcd_loss + rank_loss
        fpi_diag["famo_weighted_bce"] = weighted_bce.item()
        fpi_diag["famo_weights"] = F.softmax(famo_optimizer.w, dim=-1).detach().cpu().numpy().tolist()
    else:
        loss = s_loss + cdan_domain_loss + sample_ot_loss + dcot_loss + fpi_loss + dcd_loss + rank_loss
    loss.backward()
    optimizer.step()

    # Phase 16: FAMO update (double forward)
    if use_famo and famo_optimizer is not None:
        with torch.no_grad():
            s_outputs_new = model(
                s_x, drug_feat, s_tissue, alpha=alpha,
                return_dict=True,
                frag_fp=frag_fp, frag_mask=frag_mask, frag_smiles=frag_smiles,
                current_epoch=current_epoch,
                drug_names=drug_names,
            )
            s_yp_new = s_outputs_new["pred"].reshape(s_y.shape)
            s_loss_mat_new = loss_fn(s_yp_new, s_y.double())
            s_loss_mat_new = torch.where(
                s_mask, s_loss_mat_new,
                torch.zeros(s_loss_mat_new.shape).to(device).to(s_loss_mat_new.dtype))
            new_per_drug_bce = []
            for d in range(D_src):
                drug_mask = s_mask[:, d]
                if drug_mask.sum() > 0:
                    bce_d = s_loss_mat_new[:, d][drug_mask].mean()
                else:
                    bce_d = torch.tensor(0.0, device=device)
                new_per_drug_bce.append(bce_d)
            new_per_drug_bce_tensor = torch.stack(new_per_drug_bce)
            famo_optimizer.update(new_per_drug_bce_tensor)

    # Merge DCD diagnostics into fpi_diag (general diagnostics dict)
    for k, v in dcd_diag.items():
        fpi_diag[f"dcd_{k}"] = v

    # Merge ranking diagnostics
    for k, v in rank_diag.items():
        fpi_diag[k] = v
    if rank_per_drug_records:
        fpi_diag["rank_per_drug_records"] = rank_per_drug_records

    # Phase 16: FAMO diagnostics
    if per_drug_bce:
        fpi_diag["per_drug_bce"] = [x.item() for x in per_drug_bce]

    fpi_diag["sample_ot_loss"] = sample_ot_loss.item()
    fpi_diag["dcot_loss"] = dcot_loss.item()
    fpi_diag["cdan_loss"] = cdan_domain_loss.item()
    fpi_diag["response_loss"] = s_loss.item()
    return loss, fpi_diag


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

    # Phase 8-F: BPAF / LLRF parameters
    use_bpaf = kwargs.get('use_bpaf', False)
    bpaf_config = kwargs.get('bpaf_config', None)
    use_llrf = kwargs.get('use_llrf', False)
    llrf_config = kwargs.get('llrf_config', None)
    lambda_fusion_gamma_l1 = kwargs.get('lambda_fusion_gamma_l1', 0.001)
    lambda_selector_entropy = kwargs.get('lambda_selector_entropy', 0.0)
    lambda_selector_div = kwargs.get('lambda_selector_div', 0.0)

    # Pretrained / Graph Transformer drug encoder parameters
    use_pretrained_drug = kwargs.get('use_pretrained_drug', False)
    pretrained_drug_type = kwargs.get('pretrained_drug_type', 'chemberta')
    use_graph_transformer_drug = kwargs.get('use_graph_transformer_drug', False)
    smiles_list = kwargs.get('smiles_list', None)

    # Unified drug encoder parameters (Uni-Mol / MLP / Fusion)
    drug_encoder_type = kwargs.get('drug_encoder_type', 'mlp')
    unimol_emb_path = kwargs.get('unimol_emb_path', None)
    unimol_dim = kwargs.get('unimol_dim', 512)
    drug_emb_dim = kwargs.get('drug_emb_dim', 64)
    drug_fusion_dropout = kwargs.get('drug_fusion_dropout', 0.1)

    # Drug relation refiner parameters
    use_drug_relation_refiner = kwargs.get('use_drug_relation_refiner', False)
    drug_relation_graph = kwargs.get('drug_relation_graph', None)
    drug_relation_num_layers = kwargs.get('drug_relation_num_layers', 1)
    drug_relation_dropout = kwargs.get('drug_relation_dropout', 0.1)
    drug_relation_residual = kwargs.get('drug_relation_residual', True)

    # DBSCAN-based contrastive positive filtering
    use_dbscan_filter = kwargs.get('use_dbscan_filter', False)
    dbscan_mode = kwargs.get('dbscan_mode', 'adaptive')
    dbscan_eps = kwargs.get('dbscan_eps', 0.3)
    dbscan_min_samples = kwargs.get('dbscan_min_samples', 5)
    dbscan_distance = kwargs.get('dbscan_distance', 'cosine')

    # Phase 9: FPI parameters
    use_fpi = kwargs.get('use_fpi', False)
    fpi_config = kwargs.get('fpi_config', None)
    frag_fp = kwargs.get('frag_fp', None)
    frag_mask = kwargs.get('frag_mask', None)
    frag_smiles = kwargs.get('frag_smiles', None)
    lambda_fpi_gamma_l1 = kwargs.get('lambda_fpi_gamma_l1', 0.001)
    lambda_fpi_entropy = kwargs.get('lambda_fpi_entropy', 0.0)

    # Phase 9-R: FPI Router parameters
    use_fpi_router = kwargs.get('use_fpi_router', False)
    fpi_router_config = kwargs.get('fpi_router_config', None)
    lambda_router_gamma_l1 = kwargs.get('lambda_router_gamma_l1', 0.0)
    lambda_router_gamma_global_l1 = kwargs.get('lambda_router_gamma_global_l1', None)
    lambda_router_gamma_cross_l1 = kwargs.get('lambda_router_gamma_cross_l1', None)

    # Phase 10: Pseudo-Domain Guide parameters
    use_pdg = kwargs.get('use_pdg', False)
    pdg_config = kwargs.get('pdg_config', {}) or {}
    pdg_centroids = kwargs.get('pdg_centroids', None)
    pdg_centroid_path = kwargs.get('pdg_centroid_path', None)

    if use_pdg:
        if pdg_centroids is None:
            if pdg_centroid_path is not None:
                actual_path = pdg_centroid_path.replace("{fold}", str(fold))
                if os.path.exists(actual_path):
                    pdg_centroids = torch.tensor(
                        np.load(actual_path), dtype=torch.float32
                    ).to(kwargs["device"])
                else:
                    raise ValueError(f"PDG centroid file not found: {actual_path}")
            else:
                raise ValueError("use_pdg=True requires either pdg_centroids or pdg_centroid_path")
        else:
            pdg_centroids = pdg_centroids.to(kwargs["device"])
        print(f"PDG enabled: K={pdg_centroids.size(0)}, centroid_dim={pdg_centroids.size(1)}, pdg_dim={pdg_config.get('pdg_dim', 32)}")

    # Phase 10-D: DCD-R parameters
    use_dcd_residual = kwargs.get('use_dcd_residual', False)
    dcd_config = kwargs.get('dcd_config', None)
    lambda_dcd_gamma_l1 = kwargs.get('lambda_dcd_gamma_l1', 0.0)
    lambda_dcd_ortho = kwargs.get('lambda_dcd_ortho', 0.0)

    # Phase 11: Main Fusion Replacement
    main_fusion_type = kwargs.get('main_fusion_type', 'bilinear')
    main_fusion_config = kwargs.get('main_fusion_config', {}) or {}

    # Phase 12: VCB parameters
    use_vcb = kwargs.get('use_vcb', False)
    vcb_config = kwargs.get('vcb_config', None) or {}
    lambda_vcb_kl = kwargs.get('lambda_vcb_kl', 0.0)
    vcb_warmup_ratio = kwargs.get('vcb_warmup_ratio', 0.1)

    # Phase 13: S3R parameters
    use_s3r = kwargs.get('use_s3r', False)
    s3r_same_drug_negatives = kwargs.get('s3r_same_drug_negatives', True)
    s3r_suppress_false_negatives = kwargs.get('s3r_suppress_false_negatives', True)
    s3r_weighted_positives = kwargs.get('s3r_weighted_positives', False)
    s3r_positive_temperature = kwargs.get('s3r_positive_temperature', 0.1)
    s3r_max_negatives_per_anchor = kwargs.get('s3r_max_negatives_per_anchor', 0)
    s3r_loss_temperature = kwargs.get('s3r_loss_temperature', 0.1)

    # Phase 13-G: Gram volume alignment (replaces contrastive)
    use_gram = kwargs.get('use_gram_alignment', False)
    lambda_gram = kwargs.get('lambda_gram', 0.01)

    # Phase 17: Counterfactual causal augmentation
    lambda_causal = kwargs.get('lambda_causal', 0.0)
    causal_ablation_ratio = kwargs.get('causal_ablation_ratio', 0.1)

    # Phase 18: InfoNCE / hard negative / pos weighting
    use_infonce = kwargs.get('use_infonce', False)
    hard_negatives_only = kwargs.get('hard_negatives_only', False)
    num_hard_negs = kwargs.get('num_hard_negs', 50)
    use_pos_weights = kwargs.get('use_pos_weights', False)

    # Phase 19: Differentiated contrastive learning (DKC-inspired)
    use_diff_contrastive = kwargs.get('use_diff_contrastive', False)
    diff_contrastive_alpha = kwargs.get('diff_contrastive_alpha', 2.0)

    # Phase 14-A / 15-B: Drug-wise Pairwise Ranking Loss
    use_rank_loss = kwargs.get('use_rank_loss', False)
    lambda_rank = kwargs.get('lambda_rank', 0.0)
    rank_margin = kwargs.get('rank_margin', 0.0)
    rank_max_pairs_per_drug = kwargs.get('rank_max_pairs_per_drug', 0)
    rank_debug_sanity = kwargs.get('rank_debug_sanity', False)
    # Phase 15-B: Effective Drug-wise Ranking Loss Sweep
    ranking_warmup_epochs = kwargs.get('ranking_warmup_epochs', 0)
    ranking_min_pos = kwargs.get('ranking_min_pos', 1)
    ranking_min_neg = kwargs.get('ranking_min_neg', 1)
    ranking_log_grad_ratio = kwargs.get('ranking_log_grad_ratio', False)
    ranking_log_drugwise_pair_acc = kwargs.get('ranking_log_drugwise_pair_acc', False)
    ranking_log_interval = kwargs.get('ranking_log_interval', 50)
    # Phase 16: FAMO
    use_famo = kwargs.get('use_famo', False)
    famo_gamma = kwargs.get('famo_gamma', 0.01)
    famo_w_lr = kwargs.get('famo_w_lr', 0.025)
    famo_max_norm = kwargs.get('famo_max_norm', 1.0)

    # Phase DCDA
    use_dcda = kwargs.get('use_dcda', False)
    dcda_config = kwargs.get('dcda_config', {}) or {}
    lambda_dcda_diff = kwargs.get('lambda_dcda_diff', 0.1)
    lambda_dcda_align = kwargs.get('lambda_dcda_align', 0.05)

    # Phase 18: Domain separation
    lambda_migrate = kwargs.get('lambda_migrate', 0.0)

    # Phase 9-R Diagnosis: two-stage cross compensation
    router_two_stage_cross_only = kwargs.get('router_two_stage_cross_only', False)
    router_freeze_main_for_cross = kwargs.get('router_freeze_main_for_cross', False)
    router_stage_a_checkpoint_dir = kwargs.get('router_stage_a_checkpoint_dir', '')

    # Mutual exclusion: router takes priority over legacy FPI paths
    if use_fpi_router:
        use_fpi = False
        fpi_config = None

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
    print(f"  use_bpaf = {use_bpaf}")
    print(f"  use_llrf = {use_llrf}")
    print(f"  lambda_fusion_gamma_l1 = {lambda_fusion_gamma_l1}")
    print(f"  lambda_selector_entropy = {lambda_selector_entropy}")
    print(f"  lambda_selector_div = {lambda_selector_div}")
    print(f"  use_fpi = {use_fpi}")
    print(f"  lambda_fpi_gamma_l1 = {lambda_fpi_gamma_l1}")
    print(f"  lambda_fpi_entropy = {lambda_fpi_entropy}")
    print(f"  use_fpi_router = {use_fpi_router}")
    print(f"  lambda_router_gamma_l1 = {lambda_router_gamma_l1}")
    print(f"  use_pdg = {use_pdg}")
    if use_pdg and pdg_config:
        print(f"  pdg_dim = {pdg_config.get('pdg_dim', 32)}")
        print(f"  pdg_assign_mode = {pdg_config.get('assign_mode', 'nearest')}")
        print(f"  pdg_normalize = {pdg_config.get('normalize', True)}")
    if lambda_router_gamma_global_l1 is not None:
        print(f"  lambda_router_gamma_global_l1 = {lambda_router_gamma_global_l1}")
    if lambda_router_gamma_cross_l1 is not None:
        print(f"  lambda_router_gamma_cross_l1 = {lambda_router_gamma_cross_l1}")
    if use_fpi_router and fpi_router_config is not None:
        print(f"  router_cross_mode = {fpi_router_config.get('router_cross_mode', 'learnable')}")
        print(f"  router_cross_start_epoch = {fpi_router_config.get('router_cross_start_epoch', 0)}")
        print(f"  router_cross_ramp_epochs = {fpi_router_config.get('router_cross_ramp_epochs', 0)}")
        print(f"  gamma_global_max = {fpi_router_config.get('gamma_global_max', 0.05)}")
        print(f"  gamma_cross_max = {fpi_router_config.get('gamma_cross_max', 0.05)}")
        print(f"  residual_global_scale = {fpi_router_config.get('residual_global_scale', 1.0)}")
        print(f"  residual_cross_scale = {fpi_router_config.get('residual_cross_scale', 1.0)}")
        oracle_str = fpi_router_config.get('router_oracle_cross_drugs', '')
        if oracle_str:
            print(f"  WARNING: router_oracle_cross_drugs='{oracle_str}' (diagnostic only, not a valid method)")
    if use_fpi and fpi_config is not None:
        print(f"  fpi_detach_mode = {fpi_config.get('detach_mode', 'none')}")
        print(f"  use_fpi_sim = {fpi_config.get('use_fpi_sim', False)}")
        if fpi_config.get('use_fpi_sim', False):
            sim_cfg = fpi_config.get('fpi_sim_config', {})
            print(f"  fpi_sim_q_prior_weight = {sim_cfg.get('q_prior_weight', 0.01)}")
            print(f"  fpi_sim_union_alpha = {sim_cfg.get('union_alpha', 1.0)}")
            print(f"  fpi_sim_union_beta = {sim_cfg.get('union_beta', 1.0)}")
            print(f"  fpi_sim_program_temperature = {sim_cfg.get('program_temperature', 1.0)}")
            print(f"  fpi_sim_fragment_temperature = {sim_cfg.get('fragment_temperature', 1.0)}")
            print(f"  fpi_sim_cross_temperature = {sim_cfg.get('cross_temperature', 1.0)}")
    print(f"  use_dcd_residual = {use_dcd_residual}")
    if use_dcd_residual and dcd_config is not None:
        print(f"  dcd_gamma_max = {dcd_config.get('gamma_max', 0.03)}")
        print(f"  dcd_gamma_init = {dcd_config.get('gamma_init', 0.005)}")
        print(f"  lambda_dcd_gamma_l1 = {lambda_dcd_gamma_l1}")
        print(f"  lambda_dcd_ortho = {lambda_dcd_ortho}")
    print(f"  main_fusion_type = {main_fusion_type}")
    if main_fusion_type != "bilinear" and main_fusion_config is not None:
        print(f"  main_fusion_rank_dim = {main_fusion_config.get('rank_dim', 64)}")
        print(f"  main_fusion_out_dim = {main_fusion_config.get('out_dim', bilinear_dim)}")
        print(f"  film_init_scale = {main_fusion_config.get('film_init_scale', 1e-3)}")
    print(f"  use_vcb = {use_vcb}")
    if use_vcb and vcb_config is not None:
        print(f"  vcb_z_dim = {vcb_config.get('z_dim', 'same')}")
        print(f"  vcb_hidden_dim = {vcb_config.get('hidden_dim', 'same')}")
        print(f"  vcb_dropout = {vcb_config.get('dropout', 0.1)}")
        print(f"  vcb_logvar_min = {vcb_config.get('logvar_min', -6.0)}")
        print(f"  vcb_logvar_max = {vcb_config.get('logvar_max', 2.0)}")
        print(f"  vcb_normalize_z = {vcb_config.get('normalize_z', True)}")
        print(f"  lambda_vcb_kl = {lambda_vcb_kl}")
        print(f"  vcb_warmup_ratio = {vcb_warmup_ratio}")
    print(f"  use_s3r = {use_s3r}")
    if use_s3r:
        print(f"  s3r_same_drug_negatives = {s3r_same_drug_negatives}")
        print(f"  s3r_suppress_false_negatives = {s3r_suppress_false_negatives}")
        print(f"  s3r_weighted_positives = {s3r_weighted_positives}")
        print(f"  s3r_positive_temperature = {s3r_positive_temperature}")
        print(f"  s3r_max_negatives_per_anchor = {s3r_max_negatives_per_anchor}")
        print(f"  s3r_loss_temperature = {s3r_loss_temperature}")
    print(f"  use_rank_loss = {use_rank_loss}")
    if use_rank_loss:
        print(f"  lambda_rank = {lambda_rank}")
        print(f"  rank_margin = {rank_margin}")
        print(f"  rank_max_pairs_per_drug = {rank_max_pairs_per_drug}")
    print(f"  use_dcda = {use_dcda}")
    if use_dcda:
        print(f"  dcda_timesteps = {dcda_config.get('timesteps', 10)}")
        print(f"  lambda_dcda_diff = {lambda_dcda_diff}")
        print(f"  lambda_dcda_align = {lambda_dcda_align}")
    print(f"  use_domain_sep = {kwargs.get('use_domain_sep', False)}")
    if kwargs.get('use_domain_sep', False):
        print(f"  domain_sep_hidden_dim = {kwargs.get('domain_sep_hidden_dim', 256)}")
        print(f"  domain_sep_output_dim = {kwargs.get('domain_sep_output_dim', 64)}")
        print(f"  lambda_ortho = {kwargs.get('lambda_ortho', 0.0)}")
        print(f"  lambda_recon = {kwargs.get('lambda_recon', 0.0)}")
        print(f"  lambda_private_domain = {kwargs.get('lambda_private_domain', 0.0)}")
        print(f"  lambda_migrate = {lambda_migrate}")
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

    # Prepare fragment tensors for model registration
    frag_fp_tensor = frag_fp.to(kwargs["device"]) if frag_fp is not None else None
    frag_mask_tensor = frag_mask.to(kwargs["device"]) if frag_mask is not None else None

    main_fusion_out_dim = main_fusion_config.get("out_dim", bilinear_dim) if main_fusion_type != "bilinear" else bilinear_dim
    base_input_dim = kwargs["encoder_hidden_dims"][-1] + drug_feat_out_dim + main_fusion_out_dim
    if use_pdg:
        input_dim = base_input_dim + pdg_config.get("pdg_dim", 32)
    else:
        input_dim = base_input_dim

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
        # Phase 8-F: BPAF / LLRF
        use_bpaf=use_bpaf,
        bpaf_config=bpaf_config,
        use_llrf=use_llrf,
        llrf_config=llrf_config,
        # Phase 9: FPI
        use_fpi=use_fpi,
        fpi_config=fpi_config,
        # Phase 9-R: FPI Router
        use_fpi_router=use_fpi_router,
        fpi_router_config=fpi_router_config,
        frag_fp=frag_fp_tensor,
        frag_mask=frag_mask_tensor,
        # Phase 10: PDG
        use_pdg=use_pdg,
        pdg_config=pdg_config,
        pdg_centroids=pdg_centroids,
        # Phase 10-D: DCD-R
        use_dcd_residual=use_dcd_residual,
        dcd_config=dcd_config,
        # Phase 11: Main Fusion Replacement
        main_fusion_type=main_fusion_type,
        main_fusion_config=main_fusion_config,
        # Phase 11-C: Cross-Attention Fusion
        use_cross_attn=kwargs.get('use_cross_attn', False),
        cross_attn_num_heads=kwargs.get('cross_attn_num_heads', 4),
        # Phase 12: VCB
        use_vcb=use_vcb,
        vcb_config=vcb_config,
        # Phase DCDA
        use_dcda=use_dcda,
        dcda_config=dcda_config,
        # Pretrained / Graph Transformer drug encoder
        use_pretrained_drug=use_pretrained_drug,
        pretrained_drug_type=pretrained_drug_type,
        use_graph_transformer_drug=use_graph_transformer_drug,
        smiles_list=smiles_list,
        # Unified drug encoder (Uni-Mol / MLP / Fusion)
        drug_encoder_type=drug_encoder_type,
        unimol_emb_path=unimol_emb_path,
        unimol_dim=unimol_dim,
        drug_emb_dim=drug_emb_dim,
        drug_fusion_dropout=drug_fusion_dropout,
        # Drug relation refiner
        use_drug_relation_refiner=use_drug_relation_refiner,
        drug_relation_graph=drug_relation_graph,
        drug_relation_num_layers=drug_relation_num_layers,
        drug_relation_dropout=drug_relation_dropout,
        drug_relation_residual=drug_relation_residual,
    ).to(kwargs["device"])

    # Phase 9-R Diagnosis: two-stage cross compensation
    if use_fpi_router and router_two_stage_cross_only and router_stage_a_checkpoint_dir:
        ckpt_path = os.path.join(router_stage_a_checkpoint_dir, f"fusion_model_fold{fold}.pt")
        if os.path.exists(ckpt_path):
            print(f"Loading Stage A checkpoint from {ckpt_path}")
            missing, unexpected = fusion_model.load_state_dict(
                torch.load(ckpt_path, map_location=kwargs["device"]),
                strict=False,
            )
            if missing:
                print(f"  Stage B: {len(missing)} missing keys (expected: cross_fpi not in Stage A)")
                for k in missing[:5]:
                    print(f"    - {k}")
            if unexpected:
                print(f"  Stage B: {len(unexpected)} unexpected keys")
        else:
            print(f"WARNING: Stage A checkpoint not found at {ckpt_path}, training from scratch")

    def _set_frozen_modules_eval():
        """Keep frozen modules in eval mode to prevent BatchNorm drift."""
        if not (use_fpi_router and router_freeze_main_for_cross):
            return
        fusion_model.fusion_module.eval()
        fusion_model.cdan_discriminator.eval()
        fusion_model.drug_classifier.eval()
        fusion_model.encoder.eval()
        fusion_model.drug_mlp.eval()
        fusion_model.bilinear.eval()
        if fusion_model.fragment_encoder is not None:
            fusion_model.fragment_encoder.eval()
        if fusion_model.fpi_router is not None:
            fusion_model.fpi_router.global_fpi.eval()

    if use_fpi_router and router_freeze_main_for_cross:
        # Freeze ALL main model components (base path + global FPI)
        for p in fusion_model.encoder.parameters():
            p.requires_grad = False
        for p in fusion_model.drug_mlp.parameters():
            p.requires_grad = False
        for p in fusion_model.bilinear.parameters():
            p.requires_grad = False
        for p in fusion_model.fusion_module.parameters():
            p.requires_grad = False
        for p in fusion_model.cdan_discriminator.parameters():
            p.requires_grad = False
        for p in fusion_model.drug_classifier.parameters():
            p.requires_grad = False
        fusion_model.attention_masks.requires_grad = False
        if fusion_model.fragment_encoder is not None:
            for p in fusion_model.fragment_encoder.parameters():
                p.requires_grad = False
        if fusion_model.fpi_router is not None:
            for p in fusion_model.fpi_router.global_fpi.parameters():
                p.requires_grad = False
            fusion_model.fpi_router.raw_gamma_global.requires_grad = False
            # Also freeze raw_gamma_cross if it uses a fixed debug value
            if fusion_model.fpi_router.debug_fixed_gamma_cross is not None:
                fusion_model.fpi_router.raw_gamma_cross.requires_grad = False
        _set_frozen_modules_eval()

        # Verify freeze status
        freeze_check = {
            "encoder": all(not p.requires_grad for p in fusion_model.encoder.parameters()),
            "drug_mlp": all(not p.requires_grad for p in fusion_model.drug_mlp.parameters()),
            "bilinear": all(not p.requires_grad for p in fusion_model.bilinear.parameters()),
            "fusion_module": all(not p.requires_grad for p in fusion_model.fusion_module.parameters()),
            "cdan_discriminator": all(not p.requires_grad for p in fusion_model.cdan_discriminator.parameters()),
            "attention_masks": not fusion_model.attention_masks.requires_grad,
        }
        if fusion_model.fragment_encoder is not None:
            freeze_check["fragment_encoder"] = all(not p.requires_grad for p in fusion_model.fragment_encoder.parameters())
        if fusion_model.fpi_router is not None:
            freeze_check["global_fpi"] = all(not p.requires_grad for p in fusion_model.fpi_router.global_fpi.parameters())
            freeze_check["raw_gamma_global"] = not fusion_model.fpi_router.raw_gamma_global.requires_grad
            if fusion_model.fpi_router.cross_fpi is not None:
                freeze_check["cross_fpi"] = any(p.requires_grad for p in fusion_model.fpi_router.cross_fpi.parameters())
                freeze_check["raw_gamma_cross"] = fusion_model.fpi_router.raw_gamma_cross.requires_grad
        print("Phase 9-R Diagnosis: freeze status")
        for name, frozen in freeze_check.items():
            status = "FROZEN" if frozen else "TRAINABLE"
            print(f"  {name}: {status}")

    fusion_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, fusion_model.parameters()),
        lr=kwargs["lr"]
    )
    classifier_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    print(f"fusion_training_program (MMPA Stage 2, use_drug_gate={use_drug_gate}, gate_mode={gate_mode}, use_ot={use_ot_alignment}, use_bpaf={use_bpaf}, use_llrf={use_llrf}, use_fpi={use_fpi})")
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

    # Phase 9: FPI diagnostics logging
    fpi_diag_records = []
    fpi_diag_csv = os.path.join(kwargs.get("output_dir", "outputs"), "fpi_diagnostics.csv")
    fpi_top_pairs_records = []
    fpi_top_pairs_csv = os.path.join(kwargs.get("output_dir", "outputs"), "fpi_top_pairs.csv")

    # Stage 2: Classification + CDAN (+ optional OT)
    # Phase 16: FAMO initialization
    famo_optimizer = None
    famo_warmup_bce_list = []
    famo_warmup_steps = 5
    if use_famo:
        from mmpa_drp.famo import FAMO
        famo_optimizer = FAMO(
            n_tasks=len(drugs),
            device=kwargs["device"],
            gamma=famo_gamma,
            w_lr=famo_w_lr,
            max_norm=famo_max_norm,
        )
        print(f"FAMO initialized: n_tasks={len(drugs)}, gamma={famo_gamma}, w_lr={famo_w_lr}, max_norm={famo_max_norm}")

    # Phase 14-A: Rank loss debug accumulators
    rank_debug_records = []
    for cls_epoch in range(kwargs["train_num_epochs"]):
        train_loss = 0
        epoch_dcot_loss = 0.0
        epoch_sample_ot_loss = 0.0
        epoch_cdan_loss = 0.0
        epoch_response_loss = 0.0
        epoch_rank_loss_raw = 0.0
        epoch_rank_loss_weighted = 0.0
        fusion_model.train()
        _set_frozen_modules_eval()
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
            elif use_fpi_router:
                loss_val, fpi_diag = _fusion_model_train_step_with_fpi(
                    model=fusion_model,
                    batch=batch,
                    drug_feat=drug_feat,
                    device=kwargs["device"],
                    optimizer=fusion_optimizer,
                    loss_fn=classifier_loss,
                    lambda_fpi_gamma_l1=0.0,
                    lambda_fpi_entropy=0.0,
                    alpha=alpha,
                    ot_loss_module=ot_loss_module,
                    drug_conditioned_ot_config=drug_conditioned_ot_config,
                    dcot_loss_module=dcot_loss_module,
                    current_epoch=cls_epoch,
                    drug_names=drugs,
                    lambda_router_gamma_l1=lambda_router_gamma_l1,
                    lambda_router_gamma_global_l1=lambda_router_gamma_global_l1,
                    lambda_router_gamma_cross_l1=lambda_router_gamma_cross_l1,
                    lambda_dcd_gamma_l1=lambda_dcd_gamma_l1,
                    lambda_dcd_ortho=lambda_dcd_ortho,
                    use_rank_loss=use_rank_loss,
                    lambda_rank=lambda_rank,
                    rank_margin=rank_margin,
                    rank_max_pairs_per_drug=rank_max_pairs_per_drug,
                    ranking_warmup_epochs=ranking_warmup_epochs,
                    ranking_min_pos=ranking_min_pos,
                    ranking_min_neg=ranking_min_neg,
                    ranking_log_grad_ratio=ranking_log_grad_ratio,
                    ranking_log_drugwise_pair_acc=ranking_log_drugwise_pair_acc,
                    ranking_log_interval=ranking_log_interval,
                    step=step,
                    use_famo=use_famo,
                    famo_optimizer=famo_optimizer,
                    lambda_dcda_diff=lambda_dcda_diff,
                    lambda_dcda_align=lambda_dcda_align,
                    lambda_migrate=lambda_migrate,
                )
                step_diags = {"fpi_diag": fpi_diag}
            elif use_fpi:
                loss_val, fpi_diag = _fusion_model_train_step_with_fpi(
                    model=fusion_model,
                    batch=batch,
                    drug_feat=drug_feat,
                    device=kwargs["device"],
                    optimizer=fusion_optimizer,
                    loss_fn=classifier_loss,
                    lambda_fpi_gamma_l1=lambda_fpi_gamma_l1,
                    lambda_fpi_entropy=lambda_fpi_entropy,
                    alpha=alpha,
                    ot_loss_module=ot_loss_module,
                    drug_conditioned_ot_config=drug_conditioned_ot_config,
                    dcot_loss_module=dcot_loss_module,
                    lambda_dcd_gamma_l1=lambda_dcd_gamma_l1,
                    lambda_dcd_ortho=lambda_dcd_ortho,
                    use_rank_loss=use_rank_loss,
                    lambda_rank=lambda_rank,
                    rank_margin=rank_margin,
                    rank_max_pairs_per_drug=rank_max_pairs_per_drug,
                    ranking_warmup_epochs=ranking_warmup_epochs,
                    ranking_min_pos=ranking_min_pos,
                    ranking_min_neg=ranking_min_neg,
                    ranking_log_grad_ratio=ranking_log_grad_ratio,
                    ranking_log_drugwise_pair_acc=ranking_log_drugwise_pair_acc,
                    ranking_log_interval=ranking_log_interval,
                    step=step,
                    use_famo=use_famo,
                    famo_optimizer=famo_optimizer,
                )
                step_diags = {"fpi_diag": fpi_diag}
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
                    lambda_dcd_gamma_l1=lambda_dcd_gamma_l1,
                    lambda_dcd_ortho=lambda_dcd_ortho,
                    lambda_dcda_diff=lambda_dcda_diff,
                    lambda_dcda_align=lambda_dcda_align,
                    lambda_migrate=lambda_migrate,
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

            # Phase 16: FAMO warmup - collect min_losses from first few batches
            if use_famo and famo_optimizer is not None:
                fpi_diag_step = step_diags.get("fpi_diag", {})
                if "per_drug_bce" in fpi_diag_step:
                    bce_list = fpi_diag_step["per_drug_bce"]
                    if isinstance(bce_list, list) and len(bce_list) == len(drugs):
                        famo_warmup_bce_list.append(torch.tensor(bce_list, device=kwargs["device"]))
                if len(famo_warmup_bce_list) >= famo_warmup_steps and famo_optimizer.min_losses.sum().item() == 0:
                    stacked = torch.stack(famo_warmup_bce_list)
                    min_losses = stacked.min(dim=0)[0]
                    famo_optimizer.set_min_losses(min_losses)
                    print(f"FAMO min_losses initialized from {famo_warmup_steps} warmup batches: {min_losses.cpu().numpy().round(4)}")

            # Phase 8-F: BPAF / LLRF extra losses
            if use_bpaf or use_llrf:
                gamma_loss_val = 0.0
                selector_entropy_loss_val = 0.0
                selector_div_loss_val = 0.0

                if use_bpaf and fusion_model.bpaf is not None:
                    gamma = fusion_model.bpaf.get_gamma()
                    if lambda_fusion_gamma_l1 > 0:
                        gamma_loss_val += lambda_fusion_gamma_l1 * gamma.abs().mean().item()
                elif use_llrf and fusion_model.llrf is not None:
                    gamma = fusion_model.llrf.get_gamma()
                    if lambda_fusion_gamma_l1 > 0:
                        gamma_loss_val += lambda_fusion_gamma_l1 * gamma.abs().mean().item()

                if getattr(fusion_model, 'last_selector_attn', None) is not None:
                    attn = fusion_model.last_selector_attn
                    if lambda_selector_entropy > 0:
                        from mmpa_drp.losses import attention_entropy_loss
                        selector_entropy_loss_val += lambda_selector_entropy * attention_entropy_loss(attn).item()
                    if lambda_selector_div > 0:
                        from mmpa_drp.losses import attention_diversity_loss
                        selector_div_loss_val += lambda_selector_div * attention_diversity_loss(attn).item()

                # Note: these are scalar diagnostics; the actual backward happens inside train step
                # if we add them to the loss tensor. For now we just log them.
                step_diags["gamma_loss"] = gamma_loss_val
                step_diags["selector_entropy_loss"] = selector_entropy_loss_val
                step_diags["selector_div_loss"] = selector_div_loss_val

            epoch_dcot_loss += step_diags.get("dcot_loss", 0.0)
            epoch_sample_ot_loss += step_diags.get("sample_ot_loss", 0.0)
            epoch_cdan_loss += step_diags.get("cdan_loss", 0.0)
            epoch_response_loss += step_diags.get("response_loss", 0.0)
            # Phase 14-A: Rank loss scale tracking
            if use_rank_loss:
                fpi_diag_step = step_diags.get("fpi_diag", {})
                epoch_rank_loss_raw += fpi_diag_step.get("rank_loss_raw", 0.0)
                epoch_rank_loss_weighted += fpi_diag_step.get("rank_loss_weighted", 0.0)
                # Accumulate per-drug records
                if "rank_per_drug_records" in fpi_diag_step:
                    for rec in fpi_diag_step["rank_per_drug_records"]:
                        rec["epoch"] = cls_epoch
                        rec["fold"] = fold
                        rank_debug_records.append(rec)

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

            # Phase 9: FPI diagnostics accumulation
            fpi_diag = step_diags.get("fpi_diag", {})
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
                    # Phase 9-R: Router diagnostics
                    "router_gamma_global_mean": fpi_diag.get("router_gamma_global_mean", 0.0),
                    "router_gamma_cross_mean": fpi_diag.get("router_gamma_cross_mean", 0.0),
                    "router_gamma_global_max": fpi_diag.get("router_gamma_global_max", 0.0),
                    "router_gamma_cross_max": fpi_diag.get("router_gamma_cross_max", 0.0),
                    "router_global_logits_norm": fpi_diag.get("router_global_logits_norm", 0.0),
                    "router_cross_logits_norm": fpi_diag.get("router_cross_logits_norm", 0.0),
                    "router_gamma_global_l1": fpi_diag.get("router_gamma_global_l1", 0.0),
                    "router_gamma_cross_l1": fpi_diag.get("router_gamma_cross_l1", 0.0),
                    "router_cross_active": fpi_diag.get("router_cross_active", False),
                    "router_cross_detach_mode": fpi_router_config.get("router_cross_detach_mode", "none") if fpi_router_config else "none",
                }
                fpi_diag_records.append(fpi_record)

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

        # Phase 8-F: Log BPAF / LLRF diagnostics
        if use_bpaf or use_llrf:
            num_steps = step + 1
            if use_bpaf and fusion_model.bpaf is not None:
                gamma = fusion_model.bpaf.get_gamma().detach().cpu()
                print(f"  BPAF epoch {cls_epoch}: gamma_mean={gamma.mean():.6f}, gamma_max={gamma.max():.6f}, gamma_per_drug={gamma.numpy().round(4)}")
                if getattr(fusion_model, 'last_bpaf_out', None) is not None:
                    h_adapter = fusion_model.last_bpaf_out["h_adapter"]
                    h_eff = fusion_model.last_bpaf_out["h_eff"]
                    h_pool_exp = fusion_model.last_bpaf_out.get("h_pool_expand", None)
                    if h_pool_exp is None:
                        # Reconstruct for norm calc
                        B, D, H = h_eff.shape
                        # h_pool is stored implicitly; skip detailed norms for now
                        pass
            elif use_llrf and fusion_model.llrf is not None:
                gamma = fusion_model.llrf.get_gamma().detach().cpu()
                print(f"  LLRF epoch {cls_epoch}: gamma_mean={gamma.mean():.6f}, gamma_max={gamma.max():.6f}, gamma_per_drug={gamma.numpy().round(4)}")

        # Phase 9: Log FPI diagnostics
        if use_fpi and fpi_diag_records:
            num_steps = step + 1
            fpi_epoch_records = [r for r in fpi_diag_records if r.get("epoch") == cls_epoch]
            if fpi_epoch_records:
                avg_gamma_mean = np.mean([r.get("gamma_mean", 0.0) for r in fpi_epoch_records])
                avg_gamma_max = np.mean([r.get("gamma_max", 0.0) for r in fpi_epoch_records])
                avg_res_norm = np.mean([r.get("residual_logits_norm", 0.0) for r in fpi_epoch_records])
                avg_attn_ent = np.mean([r.get("attention_entropy", 0.0) for r in fpi_epoch_records])
                avg_top_pair = np.mean([r.get("top_pair_attention_mean", 0.0) for r in fpi_epoch_records])
                print(f"  FPI epoch {cls_epoch}: gamma_mean={avg_gamma_mean:.6f}, gamma_max={avg_gamma_max:.6f}, res_norm={avg_res_norm:.6f}, attn_ent={avg_attn_ent:.6f}, top_pair={avg_top_pair:.6f}")
                if fusion_model.fpi is not None:
                    gamma = fusion_model.fpi.get_gamma().detach().cpu()
                    print(f"    gamma_per_drug={gamma.numpy().round(4)}")

        # Phase 9-R: Log Router diagnostics
        if use_fpi_router and fpi_diag_records:
            num_steps = step + 1
            router_epoch_records = [r for r in fpi_diag_records if r.get("epoch") == cls_epoch]
            if router_epoch_records:
                avg_gg_mean = np.mean([r.get("router_gamma_global_mean", 0.0) for r in router_epoch_records])
                avg_gc_mean = np.mean([r.get("router_gamma_cross_mean", 0.0) for r in router_epoch_records])
                avg_gg_max = np.mean([r.get("router_gamma_global_max", 0.0) for r in router_epoch_records])
                avg_gc_max = np.mean([r.get("router_gamma_cross_max", 0.0) for r in router_epoch_records])
                avg_glog_norm = np.mean([r.get("router_global_logits_norm", 0.0) for r in router_epoch_records])
                avg_clog_norm = np.mean([r.get("router_cross_logits_norm", 0.0) for r in router_epoch_records])
                print(f"  Router epoch {cls_epoch}: gg_mean={avg_gg_mean:.6f}, gc_mean={avg_gc_mean:.6f}, gg_max={avg_gg_max:.6f}, gc_max={avg_gc_max:.6f}, glog_norm={avg_glog_norm:.6f}, clog_norm={avg_clog_norm:.6f}")
                if fusion_model.fpi_router is not None:
                    gamma_g = fusion_model.fpi_router.get_gamma_global().detach().cpu()
                    gamma_c = fusion_model.fpi_router.get_gamma_cross().detach().cpu()
                    print(f"    gamma_global_per_drug={gamma_g.numpy().round(4)}")
                    print(f"    gamma_cross_per_drug={gamma_c.numpy().round(4)}")

        # Phase 7: Log epoch-level DCOT diagnostics
        num_steps = step + 1
        if drug_conditioned_ot_config.get("enabled", False):
            print(f"  Epoch {cls_epoch}: response={epoch_response_loss/num_steps:.4f}, cdan={epoch_cdan_loss/num_steps:.4f}, sample_ot={epoch_sample_ot_loss/num_steps:.4f}, dcot={epoch_dcot_loss/num_steps:.4f}")

        # Phase 15-B: Log rank loss scale diagnostics
        if use_rank_loss and num_steps > 0:
            avg_response = epoch_response_loss / num_steps
            avg_rank_raw = epoch_rank_loss_raw / num_steps
            avg_rank_weighted = epoch_rank_loss_weighted / num_steps
            ratio = avg_rank_weighted / avg_response if avg_response > 0 else 0.0
            # Compute lambda_eff for display
            if ranking_warmup_epochs > 0:
                lambda_eff_display = lambda_rank * min(1.0, (cls_epoch + 1) / ranking_warmup_epochs)
            else:
                lambda_eff_display = lambda_rank
            print(f"  Rank epoch {cls_epoch}: response={avg_response:.4f}, rank_raw={avg_rank_raw:.4f}, rank_weighted={avg_rank_weighted:.4f}, lambda_eff={lambda_eff_display:.4f}, ratio={ratio:.4f}")

            # Per-drug table
            if rank_debug_records:
                epoch_recs = [r for r in rank_debug_records if r.get("epoch") == cls_epoch and not r.get("skipped", False)]
                if epoch_recs and ranking_log_drugwise_pair_acc:
                    print(f"  Per-drug (epoch {cls_epoch}):")
                    print(f"    {'Drug':<20} {'BCE':>8} {'Rank':>8} {'PairAcc':>8} {'Valid':>6} {'Pos':>5} {'Neg':>5} {'Pairs':>8} {'MeanDiff':>10}")
                    for rec in sorted(epoch_recs, key=lambda x: x.get("drug_idx", 0)):
                        name = rec.get("drug_name", f"drug_{rec.get('drug_idx', 0)}")
                        print(f"    {name:<20} {rec.get('bce_d', float('nan')):>8.4f} {rec.get('rank_loss_d', float('nan')):>8.4f} {rec.get('pair_acc_d', float('nan')):>8.4f} {rec.get('num_valid', 0):>6} {rec.get('num_pos', 0):>5} {rec.get('num_neg', 0):>5} {rec.get('num_pairs', 0):>8} {rec.get('mean_diff', float('nan')):>10.4f}")

        fusion_model.eval()
        # CCLE validation is source domain (domain_idx=0)
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

    # Flush diagnostics
    if rank_debug_records:
        try:
            import pandas as pd
            rank_debug_csv = os.path.join(kwargs.get("output_dir", "outputs"), "rank_debug_per_epoch.csv")
            pd.DataFrame(rank_debug_records).to_csv(rank_debug_csv, index=False)
        except Exception:
            pass
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
    if fpi_diag_records:
        try:
            import pandas as pd
            pd.DataFrame(fpi_diag_records).to_csv(fpi_diag_csv, index=False)
        except Exception:
            pass

    # Stage 3: Contrastive domain isolation
    con_best_auc = 0
    K = topk_contrastive

    # Phase 12: VCB diagnostics logging
    vcb_diag_records = []
    vcb_diag_csv = os.path.join(kwargs.get("output_dir", "outputs"), "vcb_diagnostics.csv")

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

        # DBSCAN-based positive filtering (SPRED-style)
        anchor_valid_mask = None
        if use_dbscan_filter:
            if dbscan_mode == 'adaptive':
                topk_sim, topk_idx, anchor_valid_mask, keep_ratio = dbscan_adaptive_neighbors(
                    all_source_fusion_feat=all_source_fusion_feat,
                    all_target_fusion_feat=all_target_fusion_feat,
                    K=K,
                    eps=dbscan_eps,
                    min_samples=dbscan_min_samples,
                    distance=dbscan_distance,
                    fallback_on_outlier=False,
                )
            else:
                dbscan_mask = dbscan_filter_topk(
                    all_source_fusion_feat=all_source_fusion_feat,
                    all_target_fusion_feat=all_target_fusion_feat,
                    topk_idx=topk_idx,
                    eps=dbscan_eps,
                    min_samples=dbscan_min_samples,
                    distance=dbscan_distance,
                )
                keep_ratio = dbscan_mask.mean().item()
                topk_sim = topk_sim * dbscan_mask
            print(
                f"[DBSCAN {dbscan_mode}] epoch {con_epoch}: keep_ratio={keep_ratio:.3f}, "
                f"eps={dbscan_eps}, min_samples={dbscan_min_samples}, distance={dbscan_distance}"
            )

        datasets = train_dataloader.dataset.tensors
        inputs, label, mask, tissue = datasets[0], datasets[1], datasets[2], datasets[3]
        if anchor_valid_mask is not None:
            sim_data = TensorDataset(inputs, label, mask, tissue, topk_sim, topk_idx, anchor_valid_mask)
        else:
            sim_data = TensorDataset(inputs, label, mask, tissue, topk_sim, topk_idx)
        train_dataloader = DataLoader(sim_data, batch_size=batch_size, shuffle=False)

        # Phase 12: compute VCB beta with warm-up
        beta_vcb_now = compute_vcb_beta(
            current_epoch=con_epoch,
            total_epochs=kwargs["contrastive_num_epochs"],
            beta_target=lambda_vcb_kl,
            warmup_ratio=vcb_warmup_ratio,
        )

        train_loss = 0
        epoch_contrastive_loss = 0.0
        epoch_vcb_kl = 0.0
        epoch_vcb_mu_norm = 0.0
        epoch_vcb_logvar_mean = 0.0
        epoch_vcb_z_std = 0.0
        num_vcb_diag_steps = 0

        # Phase 13: S3R diagnostics
        epoch_s3r_num_valid = 0.0
        epoch_s3r_num_neg_mean = 0.0
        epoch_s3r_pos_sim_mean = 0.0
        epoch_s3r_neg_sim_mean = 0.0
        num_s3r_diag_steps = 0

        fusion_model.train()
        _set_frozen_modules_eval()
        for step, batch in enumerate(zip(train_dataloader, cycle(test_dataloader))):
            if use_gram:
                loss, step_diag = _fusion_model_train_step_with_gram(
                    model=fusion_model,
                    batch=batch,
                    drug_feat=drug_feat,
                    device=kwargs["device"],
                    optimizer=fusion_optimizer,
                    loss_fn=classifier_loss,
                    lambda_gram=lambda_gram,
                )
            elif use_gate_step:
                loss, step_diag = _fusion_model_train_step_with_contrastive_and_gate(
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
                    use_vcb=use_vcb,
                    beta_vcb=beta_vcb_now,
                    use_s3r=use_s3r,
                    s3r_same_drug_negatives=s3r_same_drug_negatives,
                    s3r_suppress_false_negatives=s3r_suppress_false_negatives,
                    s3r_weighted_positives=s3r_weighted_positives,
                    s3r_positive_temperature=s3r_positive_temperature,
                    s3r_max_negatives_per_anchor=s3r_max_negatives_per_anchor,
                    s3r_loss_temperature=s3r_loss_temperature,
                )
            else:
                loss, step_diag = _fusion_model_train_step_with_contrastive(
                    model=fusion_model,
                    batch=batch,
                    drug_feat=drug_feat,
                    device=kwargs["device"],
                    optimizer=fusion_optimizer,
                    loss_fn=classifier_loss,
                    all_target_fusion_feat=all_target_fusion_feat,
                    beta=beta,
                    use_vcb=use_vcb,
                    beta_vcb=beta_vcb_now,
                    use_s3r=use_s3r,
                    s3r_same_drug_negatives=s3r_same_drug_negatives,
                    s3r_suppress_false_negatives=s3r_suppress_false_negatives,
                    s3r_weighted_positives=s3r_weighted_positives,
                    s3r_positive_temperature=s3r_positive_temperature,
                    s3r_max_negatives_per_anchor=s3r_max_negatives_per_anchor,
                    s3r_loss_temperature=s3r_loss_temperature,
                    lambda_causal=lambda_causal,
                    causal_ablation_ratio=causal_ablation_ratio,
                    use_infonce=use_infonce,
                    hard_negatives_only=hard_negatives_only,
                    num_hard_negs=num_hard_negs,
                    use_pos_weights=use_pos_weights,
                    use_diff_contrastive=use_diff_contrastive,
                    diff_contrastive_alpha=diff_contrastive_alpha,
                )
            train_loss += (loss.cpu().detach().item() if isinstance(loss, torch.Tensor) else loss)
            epoch_contrastive_loss += step_diag.get("contrastive_loss", step_diag.get("gram_loss", 0.0))
            if use_vcb:
                epoch_vcb_kl += step_diag.get("vcb_kl", 0.0)
                epoch_vcb_mu_norm += step_diag.get("vcb_mu_norm", 0.0)
                epoch_vcb_logvar_mean += step_diag.get("vcb_logvar_mean", 0.0)
                epoch_vcb_z_std += step_diag.get("vcb_z_std", 0.0)
                num_vcb_diag_steps += 1
            if use_s3r:
                epoch_s3r_num_valid += step_diag.get("s3r_num_valid", 0)
                epoch_s3r_num_neg_mean += step_diag.get("s3r_num_neg_mean", 0.0)
                epoch_s3r_pos_sim_mean += step_diag.get("s3r_pos_sim_mean", 0.0)
                epoch_s3r_neg_sim_mean += step_diag.get("s3r_neg_sim_mean", 0.0)
                num_s3r_diag_steps += 1

        num_steps = step + 1

        # Phase 12: Log VCB diagnostics per epoch
        if use_vcb and num_vcb_diag_steps > 0:
            avg_kl = epoch_vcb_kl / num_vcb_diag_steps
            avg_mu_norm = epoch_vcb_mu_norm / num_vcb_diag_steps
            avg_logvar_mean = epoch_vcb_logvar_mean / num_vcb_diag_steps
            avg_z_std = epoch_vcb_z_std / num_vcb_diag_steps
            avg_contrastive = epoch_contrastive_loss / num_steps
            print(f"  VCB epoch {con_epoch}: beta={beta_vcb_now:.6f}, kl={avg_kl:.6f}, mu_norm={avg_mu_norm:.4f}, logvar_mean={avg_logvar_mean:.4f}, z_std={avg_z_std:.4f}, con_loss={avg_contrastive:.4f}")
            vcb_diag_records.append({
                "epoch": con_epoch,
                "beta_now": beta_vcb_now,
                "vcb_kl": avg_kl,
                "vcb_mu_norm": avg_mu_norm,
                "vcb_logvar_mean": avg_logvar_mean,
                "vcb_z_std": avg_z_std,
                "contrastive_loss": avg_contrastive,
            })

        # Phase 13: Log S3R diagnostics per epoch
        if use_s3r and num_s3r_diag_steps > 0:
            avg_contrastive = epoch_contrastive_loss / num_steps
            avg_num_valid = epoch_s3r_num_valid / num_s3r_diag_steps
            avg_num_neg = epoch_s3r_num_neg_mean / num_s3r_diag_steps
            avg_pos_sim = epoch_s3r_pos_sim_mean / num_s3r_diag_steps
            avg_neg_sim = epoch_s3r_neg_sim_mean / num_s3r_diag_steps
            print(f"  S3R epoch {con_epoch}: con_loss={avg_contrastive:.4f}, num_valid={avg_num_valid:.1f}, num_neg={avg_num_neg:.1f}, pos_sim={avg_pos_sim:.4f}, neg_sim={avg_neg_sim:.4f}")

        fusion_model.eval()
        # CCLE validation is source domain (domain_idx=0)
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

    # Phase 12: Save VCB diagnostics
    if vcb_diag_records:
        try:
            import pandas as pd
            pd.DataFrame(vcb_diag_records).to_csv(vcb_diag_csv, index=False)
        except Exception:
            pass

    # Phase 9-R Diagnosis: save Stage A checkpoint for two-stage experiments
    if use_fpi_router and router_stage_a_checkpoint_dir and not router_two_stage_cross_only:
        os.makedirs(router_stage_a_checkpoint_dir, exist_ok=True)
        save_path = os.path.join(router_stage_a_checkpoint_dir, f"fusion_model_fold{fold}.pt")
        torch.save(fusion_model.state_dict(), save_path)
        print(f"Saved Stage A checkpoint to {save_path}")

    return fusion_model
