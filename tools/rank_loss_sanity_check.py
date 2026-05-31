#!/usr/bin/env python3
"""
Rank Loss Sanity Check Tool for MACB-DRP / MMPA-DRP
====================================================
Diagnostic script to investigate drug-wise AUC ranking loss implementation.
Does NOT modify training logic or model weights.

Usage:
    python tools/rank_loss_sanity_check.py \
        --checkpoint path/to/checkpoint.pt \
        --num_batches 3 \
        --device cuda \
        --output outputs/rank_sanity_report.json \
        [other training args]
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from pretraining import train_tissue_classifier
from mmpa_drp.models import ProgramFusionMLP
from mmpa_drp.losses import drug_wise_pairwise_ranking_loss
from utility import safe_make_dir, set_seed


# =============================================================================
# 1. 参数解析
# =============================================================================

def build_parser():
    parser = argparse.ArgumentParser(description="Rank Loss Sanity Check")

    # Data & training basics (mirror train_mmpa_program.py)
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='Z_SCORE',
                        choices=['Z_SCORE', 'LN_IC50', 'AUC'])
    parser.add_argument('--thres_gdsc', dest='thres_g', nargs='?', type=float, default=0.0)
    parser.add_argument('--thres_label', dest='thres_s', nargs='?', type=float, default=0.1)
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)
    parser.add_argument('--alph', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=0.3)

    # Model architecture
    parser.add_argument('--backbone_type', type=str, default='convnext1d',
                        choices=['mlp', 'grn_mlp', 'convnext1d'])
    parser.add_argument('--convnext_depth', type=int, default=1)
    parser.add_argument('--convnext_kernel_size', type=int, default=7)
    parser.add_argument('--num_programs', type=int, default=8)
    parser.add_argument('--program_dim', type=int, default=64)
    parser.add_argument('--num_gene_tokens', type=int, default=64)
    parser.add_argument('--token_dim', type=int, default=64)
    parser.add_argument('--convnext_expansion', type=int, default=4)

    # Stage 1 pretraining
    parser.add_argument('--pretrain_num_epochs', type=int, default=300)
    parser.add_argument('--lambda_div', type=float, default=0.01)
    parser.add_argument('--lambda_sparse', type=float, default=0.001)

    # Stage 2/3
    parser.add_argument('--train_num_epochs', type=int, default=100)
    parser.add_argument('--contrastive_num_epochs', type=int, default=200)
    parser.add_argument('--K', type=int, default=3)

    # OT / CDAN
    parser.add_argument('--use_ot_alignment', action='store_true')
    parser.add_argument('--lambda_ot', type=float, default=0.01)
    parser.add_argument('--ot_reg', type=float, default=0.1)
    parser.add_argument('--ot_cost_type', type=str, default='cosine')
    parser.add_argument('--ot_marginal_type', type=str, default='uniform')
    parser.add_argument('--use_cdan', action='store_true', default=True)
    parser.add_argument('--no_cdan', dest='use_cdan', action='store_false')
    parser.add_argument('--ot_mode', type=str, default='global')

    # FPI
    parser.add_argument('--use_fpi', action='store_true')
    parser.add_argument('--fpi_fragment_npz', type=str,
                        default='data/drug_fragments_brics_m16_fp1024.npz')
    parser.add_argument('--fpi_gamma_max', type=float, default=0.05)
    parser.add_argument('--fpi_gamma_init', type=float, default=0.0)
    parser.add_argument('--fpi_q_prior_weight', type=float, default=0.01)
    parser.add_argument('--fpi_dropout', type=float, default=0.1)
    parser.add_argument('--fpi_m_max', type=int, default=16)
    parser.add_argument('--fpi_fp_dim', type=int, default=1024)
    parser.add_argument('--lambda_fpi_gamma_l1', type=float, default=0.0)
    parser.add_argument('--lambda_fpi_entropy', type=float, default=0.0)
    parser.add_argument('--fpi_detach_mode', type=str, default='none')

    # Rank loss (for diagnostic comparison)
    parser.add_argument('--use_rank_loss', action='store_true')
    parser.add_argument('--lambda_rank', type=float, default=0.0)
    parser.add_argument('--rank_margin', type=float, default=0.0)
    parser.add_argument('--rank_max_pairs_per_drug', type=int, default=0)

    # Sanity-check specific args
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (fusion_model or stage1+fusion)')
    parser.add_argument('--num_batches', type=int, default=3,
                        help='Number of source batches to diagnose')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='outputs/rank_sanity_report.json')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--skip_pretrain', action='store_true',
                        help='Skip Stage 1 pretraining (use random init encoder)')
    parser.add_argument('--lambda_probe', type=float, default=1.0,
                        help='Probe lambda for gradient conflict analysis')

    return parser


# =============================================================================
# 2. 诊断核心函数
# =============================================================================

def compute_drugwise_rank_sanity(logits, labels, mask=None, drug_names=None,
                                  missing_value=-1, rank_margin=0.0):
    """
    Compute drug-wise BCE and Rank Loss diagnostics.

    Args:
        logits: [B, D] raw final logits (before sigmoid)
        labels: [B, D] binary labels (0=resistant, 1=sensitive)
        mask:   [B, D] bool or float mask (True/1 = valid)
        drug_names: list of str, length D
        missing_value: not actively used if mask is provided; kept for compatibility
        rank_margin: float, ranking margin

    Returns:
        dict with per-drug diagnostics and global aggregates
    """
    B, D = logits.shape
    device = logits.device

    # Ensure mask is bool
    if mask is None:
        mask = torch.ones(B, D, dtype=torch.bool, device=device)
    else:
        if mask.dtype != torch.bool:
            mask = mask > 0

    # BCE loss (per-element)
    bce_fn = nn.BCEWithLogitsLoss(reduction='none')
    bce_mat = bce_fn(logits, labels.double())
    bce_mat = torch.where(mask, bce_mat, torch.zeros_like(bce_mat))

    per_drug = []
    rank_losses_mean = []
    rank_losses_sum = []
    bce_losses = []

    global_num_pos = 0
    global_num_neg = 0
    global_num_pairs = 0
    global_correct_pairs = 0

    for d in range(D):
        valid = mask[:, d]
        y_d = labels[:, d]
        s_d = logits[:, d]

        pos_mask = valid & (y_d > 0.5)
        neg_mask = valid & (y_d <= 0.5)

        pos_logits = s_d[pos_mask]
        neg_logits = s_d[neg_mask]
        num_pos = pos_logits.numel()
        num_neg = neg_logits.numel()
        pair_count = num_pos * num_neg

        # BCE for this drug (mean over valid samples)
        bce_d = bce_mat[:, d][valid].mean().item() if valid.any() else float('nan')

        # Rank loss
        if num_pos == 0 or num_neg == 0:
            rank_mean_d = float('nan')
            rank_sum_d = float('nan')
            pos_mean = float('nan')
            neg_mean = float('nan')
            pos_std = float('nan')
            neg_std = float('nan')
            mean_diff = float('nan')
            frac_correct = float('nan')
            direction_status = "NO_POS_OR_NEG"
            grad_norm_d = 0.0
        else:
            diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)  # [P, N]
            rank_pair_loss = F.softplus(-(diff - rank_margin))
            rank_mean_d = rank_pair_loss.mean().item()
            rank_sum_d = rank_pair_loss.sum().item()

            pos_mean = pos_logits.mean().item()
            neg_mean = neg_logits.mean().item()
            pos_std = pos_logits.std().item() if num_pos > 1 else 0.0
            neg_std = neg_logits.std().item() if num_neg > 1 else 0.0
            mean_diff = pos_mean - neg_mean

            frac_correct = (diff > 0).float().mean().item()

            if mean_diff > 0:
                direction_status = "OK"
            else:
                direction_status = "REVERSED_OR_UNTRAINED"

            global_num_pos += num_pos
            global_num_neg += num_neg
            global_num_pairs += pair_count
            global_correct_pairs += int((diff > 0).sum().item())

            rank_losses_mean.append(rank_mean_d)
            rank_losses_sum.append(rank_sum_d)

        bce_losses.append(bce_d)

        record = {
            "drug_index": d,
            "drug_name": drug_names[d] if drug_names and d < len(drug_names) else f"drug_{d}",
            "num_valid": int(valid.sum().item()),
            "num_pos": num_pos,
            "num_neg": num_neg,
            "pair_count": pair_count,
            "bce_loss": bce_d,
            "rank_loss_mean": rank_mean_d,
            "rank_loss_sum": rank_sum_d,
            "rank_sum_over_mean_ratio": (rank_sum_d / rank_mean_d) if rank_mean_d and rank_mean_d > 0 else None,
            "pos_logit_mean": pos_mean,
            "neg_logit_mean": neg_mean,
            "pos_logit_std": pos_std,
            "neg_logit_std": neg_std,
            "mean_diff": mean_diff,
            "frac_correct_pairs": frac_correct,
            "direction_status": direction_status,
        }
        per_drug.append(record)

    # Global aggregates
    valid_rank_mean = [r for r in rank_losses_mean if not np.isnan(r)]
    valid_rank_sum = [r for r in rank_losses_sum if not np.isnan(r)]
    valid_bce = [r for r in bce_losses if not np.isnan(r)]

    global_report = {
        "logits_shape_raw": list(logits.shape),
        "labels_shape_raw": list(labels.shape),
        "mask_shape_raw": list(mask.shape),
        "num_drugs": D,
        "num_batches": B,
        "missing_value": missing_value,
        "global_num_pos": global_num_pos,
        "global_num_neg": global_num_neg,
        "global_num_pairs": global_num_pairs,
        "global_frac_correct_pairs": (global_correct_pairs / global_num_pairs) if global_num_pairs > 0 else None,
        "bce_total": np.mean(valid_bce) if valid_bce else None,
        "rank_total_mean": np.mean(valid_rank_mean) if valid_rank_mean else None,
        "rank_total_sum": np.sum(valid_rank_sum) if valid_rank_sum else None,
    }

    if global_report["bce_total"] and global_report["rank_total_mean"]:
        global_report["rank_over_bce_loss_ratio"] = global_report["rank_total_mean"] / global_report["bce_total"]
    else:
        global_report["rank_over_bce_loss_ratio"] = None

    return {
        "global": global_report,
        "per_drug": per_drug,
    }


def compute_grad_norms(model, logits, labels, mask, lambda_probe=1.0, lambda_rank=0.0):
    """
    Compute gradient norms for BCE loss and Rank loss separately.
    Does NOT call optimizer.step().
    """
    params = [p for p in model.parameters() if p.requires_grad]

    # BCE grad norm (masked mean)
    bce_fn_none = nn.BCEWithLogitsLoss(reduction='none')
    bce_mat = bce_fn_none(logits, labels.double())
    bce_mat = torch.where(mask, bce_mat, torch.zeros_like(bce_mat))
    bce_loss = bce_mat.sum() / mask.sum()

    bce_grads = torch.autograd.grad(bce_loss, params, allow_unused=True, retain_graph=True)
    bce_grad_norm = sum(g.norm().item() for g in bce_grads if g is not None)

    # Rank grad norm (full model, unscaled by lambda_rank)
    B, D = logits.shape
    device = logits.device
    rank_loss = torch.tensor(0.0, device=device)
    count = 0
    for d in range(D):
        valid = mask[:, d]
        y_d = labels[:, d]
        s_d = logits[:, d]
        pos = s_d[valid & (y_d > 0.5)]
        neg = s_d[valid & (y_d <= 0.5)]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        loss_d = F.softplus(-diff).mean()
        rank_loss = rank_loss + loss_d
        count += 1
    if count > 0:
        rank_loss = rank_loss / count

    rank_grads = torch.autograd.grad(rank_loss, params, allow_unused=True, retain_graph=True)
    rank_grad_norm = sum(g.norm().item() for g in rank_grads if g is not None)

    # Adjusted rank grad norm (as used in training with lambda_rank)
    rank_grad_norm_adjusted = rank_grad_norm * lambda_rank if lambda_rank > 0 else rank_grad_norm

    # Per-drug grad norms (for conflict matrix)
    drug_grads = []
    for d in range(D):
        valid = mask[:, d]
        y_d = labels[:, d]
        s_d = logits[:, d]
        pos = s_d[valid & (y_d > 0.5)]
        neg = s_d[valid & (y_d <= 0.5)]

        bce_d = bce_mat[:, d][valid].mean() if valid.any() else torch.tensor(0.0, device=device)
        if pos.numel() > 0 and neg.numel() > 0:
            diff = pos.unsqueeze(1) - neg.unsqueeze(0)
            rank_d = F.softplus(-diff).mean()
            loss_d = bce_d + lambda_probe * rank_d
        else:
            loss_d = bce_d

        grads = torch.autograd.grad(loss_d, params, allow_unused=True, retain_graph=True)
        grad_vec = torch.cat([g.flatten() for g in grads if g is not None])
        drug_grads.append(grad_vec)

    # Cosine conflict matrix
    D_actual = len(drug_grads)
    cosine_matrix = np.eye(D_actual)
    num_negative = 0
    total_pairs = 0
    most_conflicting = []
    for i in range(D_actual):
        for j in range(i + 1, D_actual):
            gi = drug_grads[i]
            gj = drug_grads[j]
            if gi.norm() > 1e-12 and gj.norm() > 1e-12:
                cos = F.cosine_similarity(gi.unsqueeze(0), gj.unsqueeze(0)).item()
            else:
                cos = 1.0
            cosine_matrix[i, j] = cos
            cosine_matrix[j, i] = cos
            if cos < 0:
                num_negative += 1
                most_conflicting.append({
                    "drug_i": i,
                    "drug_j": j,
                    "cosine": cos,
                })
            total_pairs += 1

    most_conflicting = sorted(most_conflicting, key=lambda x: x["cosine"])[:10]
    negative_ratio = num_negative / total_pairs if total_pairs > 0 else 0.0

    return {
        "bce_grad_norm": bce_grad_norm,
        "rank_grad_norm": rank_grad_norm,
        "rank_grad_norm_adjusted": rank_grad_norm_adjusted,
        "rank_over_bce_grad_ratio": (rank_grad_norm / bce_grad_norm) if bce_grad_norm > 0 else None,
        "rank_adjusted_over_bce_grad_ratio": (rank_grad_norm_adjusted / bce_grad_norm) if bce_grad_norm > 0 else None,
        "cosine_matrix": cosine_matrix.tolist(),
        "num_negative_pairs": num_negative,
        "negative_pair_ratio": negative_ratio,
        "most_conflicting_pairs": most_conflicting,
    }


def old_rank_loss_static_check():
    """
    Static analysis of existing drug_wise_pairwise_ranking_loss implementation.
    """
    import inspect
    source = inspect.getsource(drug_wise_pairwise_ranking_loss)

    checks = {
        "found_old_rank_loss": True,
        "uses_sum": "sum()" in source and "mean()" in source,
        "uses_flatten_across_drugs": "view(-1)" in source or "flatten" in source,
        "uses_sigmoid_probability": "sigmoid" in source,
        "masks_missing_labels": "mask" in source,
        "notes": [],
    }

    if checks["uses_sum"]:
        checks["notes"].append("WARNING: Old implementation contains .sum() path via reduction='sum'.")
    if checks["uses_flatten_across_drugs"]:
        checks["notes"].append("WARNING: Old implementation may flatten across drugs.")
    if checks["uses_sigmoid_probability"]:
        checks["notes"].append("WARNING: Old implementation uses sigmoid (probability) instead of raw logits.")
    if not checks["masks_missing_labels"]:
        checks["notes"].append("WARNING: Old implementation may not mask missing labels.")

    # Additional fine-grained checks
    if "for d in range(D):" in source:
        checks["notes"].append("OK: Iterates per-drug (not flattened).")
    if "pos.unsqueeze(1) - neg.unsqueeze(0)" in source:
        checks["notes"].append("OK: Computes pairwise diff within drug.")
    if "F.softplus" in source:
        checks["notes"].append("OK: Uses softplus ranking (not hard margin).")
    if "torch.stack(losses).mean()" in source:
        checks["notes"].append("OK: Cross-drug reduction is mean (not sum).")

    return checks


# =============================================================================
# 3. 主函数
# =============================================================================

def build_data_and_model(args):
    """Replicate data/model construction from train_mmpa_program.py."""
    device = args.device
    set_seed(args.seed)

    # Load gene expression features
    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)

    # Get drug list (hard-coded 9 drugs from project, must match CSV case)
    drugs = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
             "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"]

    # Drug features
    from dataload import get_drug_feat
    drug_feat = torch.from_numpy(get_drug_feat(drugs).astype('float32')).to(device)

    # Unlabeled dataloaders (for Stage 1)
    from dataload import get_unlabeled_dataloaders
    s_dataloaders, t_dataloaders = get_unlabeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=args.seed,
        test_ratio=0.25,
        batch_size=64,
    )

    # Stage 1 model
    from mmpa_drp.models import ProgramTissueClassifier
    pretrain_model = ProgramTissueClassifier(
        input_dim=gex_features_df.shape[1] - 1,
        num_tissues=len(config.tissue_map),
        num_programs=args.num_programs,
        program_dim=args.program_dim,
        hidden_dims=[512, 256, 128, 64],
        drop=0.2,
        backbone_type=args.backbone_type,
        num_gene_tokens=args.num_gene_tokens,
        token_dim=args.token_dim,
        convnext_depth=args.convnext_depth,
        convnext_kernel_size=args.convnext_kernel_size,
        convnext_expansion=args.convnext_expansion,
    ).to(device)

    # Stage 1 training (or skip)
    if not args.skip_pretrain:
        print("[SanityCheck] Stage 1 pretraining...")
        trained_model = train_tissue_classifier(
            model=pretrain_model,
            s_dataloaders=s_dataloaders,
            t_dataloaders=t_dataloaders,
            device=device,
            num_epochs=args.pretrain_num_epochs,
            lr=0.001,
            alpha=1.0,
            weight_decay=1e-4,
        )
    else:
        print("[SanityCheck] Skipping Stage 1 pretraining (random init).")
        trained_model = pretrain_model

    # Labeled dataloader
    from dataload import get_ccle_multi_labeled_dataloader
    labeled_dataloader = get_ccle_multi_labeled_dataloader(
        gex_features_df=gex_features_df,
        batch_size=args.batch_size,
        drug=drugs,
        seed=args.seed,
        measurement=args.measurement,
        threshold_gdsc=args.thres_g,
        threshold_label=args.thres_s,
        n_splits=args.n,
    )

    # Load fragments if FPI enabled
    frag_fp = frag_mask = frag_smiles = None
    if args.use_fpi:
        from mmpa_drp.fragment_utils import load_drug_fragments
        frag_data = load_drug_fragments(args.fpi_fragment_npz, expected_drug_names=drugs)
        frag_fp = frag_data["frag_fp"].to(device)
        frag_mask = frag_data["frag_mask"].to(device)
        frag_smiles = frag_data["frag_smiles"]

    # Build fusion model config (mirror fusion_training_program)
    drug_feat_out_dim = 64
    bilinear_dim = 128
    encoder_hidden_dims = [512, 256, 64]  # from train_params.json
    input_dim = encoder_hidden_dims[-1] + drug_feat_out_dim + bilinear_dim

    frag_fp_tensor = frag_fp.to(device) if frag_fp is not None else None
    frag_mask_tensor = frag_mask.to(device) if frag_mask is not None else None

    fusion_model = ProgramFusionMLP(
        encoder=trained_model.program_encoder,
        input_dim=input_dim,
        output_dim=1,
        hidden_dims=[256, 128, 64],
        gene_num=gex_features_df.shape[1] - 1,
        bilinear_dim=bilinear_dim,
        gene_feat_dim=encoder_hidden_dims[-1],
        drug_feat_dim=drug_feat_out_dim,
        drug_num=len(drugs),
        drop=0.1,
        use_fpi=args.use_fpi,
        fpi_config={
            'program_dim': args.program_dim,
            'drug_dim': 64,
            'fragment_dim': args.program_dim,
            'num_programs': args.num_programs,
            'num_drugs': len(drugs),
            'max_fragments': args.fpi_m_max,
            'fp_dim': args.fpi_fp_dim,
            'q_prior_weight': args.fpi_q_prior_weight,
            'gamma_max': args.fpi_gamma_max,
            'gamma_init': args.fpi_gamma_init,
            'dropout': args.fpi_dropout,
            'detach_mode': args.fpi_detach_mode,
        },
        frag_fp=frag_fp_tensor,
        frag_mask=frag_mask_tensor,
    ).to(device)

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[SanityCheck] Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        fusion_model.load_state_dict(state, strict=False)
    else:
        print("[SanityCheck] No checkpoint loaded (using current init).")

    fusion_model.eval()

    return fusion_model, labeled_dataloader, drugs, drug_feat, frag_fp, frag_mask, device


def diagnose_batch(model, batch, drug_feat, frag_fp, frag_mask, drugs, device, args):
    """Diagnose a single source batch."""
    # batch is (train_loader, test_loader) from labeled_dataloader
    train_loader, test_loader = batch

    # We only need ONE source batch from train_loader
    try:
        s_batch = next(iter(train_loader))
    except StopIteration:
        return None

    s_x = s_batch[0].to(device)
    s_y = s_batch[1].to(device)
    s_mask_raw = s_batch[2].to(device)
    s_mask = (s_mask_raw > 0)
    s_tissue = s_batch[3].to(device)

    B_src = s_y.size(0)
    D_src = s_y.size(1)

    with torch.set_grad_enabled(True):
        model.eval()  # use eval for deterministic diagnostics, but keep grad enabled
        s_outputs = model(
            s_x, drug_feat, s_tissue, alpha=args.alph,
            return_dict=True,
            frag_fp=frag_fp, frag_mask=frag_mask,
            drug_names=drugs,
        )

        # Extract logits
        final_logits = s_outputs["final_logits"]  # [B*D, 1] before sigmoid
        pred = s_outputs["pred"]  # [B*D, 1] after sigmoid

        # Reshape to [B, D]
        logits_2d = final_logits.view(B_src, D_src)
        pred_2d = pred.view(B_src, D_src)

        # Run diagnostics
        rank_report = compute_drugwise_rank_sanity(
            logits=logits_2d,
            labels=s_y,
            mask=s_mask,
            drug_names=drugs,
            rank_margin=args.rank_margin,
        )

        # Gradient analysis (only if we have valid pairs)
        has_valid_pairs = any(
            not np.isnan(d["rank_loss_mean"]) for d in rank_report["per_drug"]
        )
        if has_valid_pairs:
            grad_report = compute_grad_norms(
                model, logits_2d, s_y, s_mask, lambda_probe=args.lambda_probe, lambda_rank=args.lambda_rank
            )
        else:
            grad_report = {
                "bce_grad_norm": None,
                "rank_grad_norm": None,
                "rank_over_bce_grad_ratio": None,
                "cosine_matrix": None,
                "num_negative_pairs": None,
                "negative_pair_ratio": None,
                "most_conflicting_pairs": [],
            }

    return {
        "batch_size": B_src,
        "logits_shape_raw": list(final_logits.shape),
        "logits_shape_used": list(logits_2d.shape),
        "labels_shape_raw": list(s_y.shape),
        "labels_shape_used": list(s_y.shape),
        "mask_shape_raw": list(s_mask_raw.shape),
        "mask_shape_used": list(s_mask.shape),
        "pred_shape_raw": list(pred.shape),
        "pred_shape_used": list(pred_2d.shape),
        "logits_min": logits_2d.min().item(),
        "logits_max": logits_2d.max().item(),
        "logits_mean": logits_2d.mean().item(),
        "logits_std": logits_2d.std().item(),
        "pred_min": pred_2d.min().item(),
        "pred_max": pred_2d.max().item(),
        "pred_mean": pred_2d.mean().item(),
        **rank_report,
        "gradient": grad_report,
    }


def print_report(report):
    """Pretty-print the sanity report to terminal."""
    print("\n" + "=" * 100)
    print("RANK LOSS SANITY CHECK REPORT")
    print("=" * 100)

    # Global
    g = report.get("global", {})
    print("\n[Global Stats]")
    for k, v in g.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.6f}")
        else:
            print(f"  {k:30s}: {v}")

    # Per-drug table
    print("\n[Per-Drug Diagnostics]")
    header = f"{'Drug':<20s} {'valid':>6s} {'pos':>5s} {'neg':>5s} {'pairs':>8s} {'BCE':>8s} {'RankMean':>10s} {'Sum/Mean':>10s} {'pos_logit':>10s} {'neg_logit':>10s} {'diff':>8s} {'pair_acc':>8s} {'status':>20s}"
    print(header)
    print("-" * len(header))
    for d in report.get("per_drug", []):
        sm_ratio = d.get("rank_sum_over_mean_ratio")
        sm_str = f"{sm_ratio:.1f}" if sm_ratio is not None else "N/A"
        print(f"{d['drug_name']:<20s} {d['num_valid']:>6d} {d['num_pos']:>5d} {d['num_neg']:>5d} {d['pair_count']:>8d} "
              f"{d['bce_loss']:>8.4f} {d['rank_loss_mean']:>10.4f} {sm_str:>10s} "
              f"{d['pos_logit_mean']:>10.4f} {d['neg_logit_mean']:>10.4f} {d['mean_diff']:>8.4f} "
              f"{d['frac_correct_pairs']:>8.4f} {d['direction_status']:>20s}")

    # Gradient
    grad = report.get("gradient", {})
    print("\n[Gradient Analysis]")
    for k in ["bce_grad_norm", "rank_grad_norm", "rank_grad_norm_adjusted", "rank_over_bce_grad_ratio", "rank_adjusted_over_bce_grad_ratio"]:
        v = grad.get(k)
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.6f}")
        else:
            print(f"  {k:30s}: {v}")

    ratio = grad.get("rank_adjusted_over_bce_grad_ratio")
    if ratio is not None:
        if ratio > 5.0:
            print("  ⚠️  WARNING: Adjusted ranking gradient is much larger than BCE gradient. lambda_rank may need to be much smaller.")
        elif ratio < 0.01:
            print("  ⚠️  WARNING: Adjusted ranking gradient is too small. lambda_rank may have no effect.")

    neg_ratio = grad.get("negative_pair_ratio")
    if neg_ratio is not None:
        print(f"\n  Negative gradient pair ratio: {neg_ratio:.4f}")
        if neg_ratio > 0.2:
            print("  ⚠️  WARNING: High drug-wise gradient conflict detected. Consider PCGrad / FAMO.")

    # Old rank loss static check
    old_check = report.get("old_rank_loss_static_check", {})
    print("\n[Old Rank Loss Static Check]")
    for k, v in old_check.items():
        if k == "notes":
            for note in v:
                print(f"  {note}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 100)


def main():
    parser = build_parser()
    args = parser.parse_args()
    safe_make_dir(os.path.dirname(args.output) if os.path.dirname(args.output) else ".")

    print("[SanityCheck] Building data and model...")
    model, labeled_dataloader, drugs, drug_feat, frag_fp, frag_mask, device = build_data_and_model(args)

    print(f"[SanityCheck] Running diagnosis over {args.num_batches} batches...")
    batch_reports = []
    batch_count = 0

    for fold_data in labeled_dataloader:
        if batch_count >= args.num_batches:
            break
        report = diagnose_batch(model, fold_data, drug_feat, frag_fp, frag_mask, drugs, device, args)
        if report is not None:
            batch_reports.append(report)
            print(f"[SanityCheck] Batch {batch_count + 1}/{args.num_batches} done.")
            batch_count += 1

    if not batch_reports:
        print("[SanityCheck] No valid batches found.")
        return

    # Aggregate across batches
    aggregated = {
        "config": vars(args),
        "num_batches_diagnosed": len(batch_reports),
        "batch_reports": batch_reports,
        "old_rank_loss_static_check": old_rank_loss_static_check(),
    }

    # Compute mean per-drug stats across batches
    mean_per_drug = defaultdict(lambda: defaultdict(list))
    for r in batch_reports:
        for d in r.get("per_drug", []):
            name = d["drug_name"]
            for k, v in d.items():
                if isinstance(v, (int, float)) and k not in ("drug_index", "drug_name"):
                    mean_per_drug[name][k].append(v)

    aggregated["per_drug_mean"] = {}
    for name, stats in mean_per_drug.items():
        aggregated["per_drug_mean"][name] = {
            k: (np.nanmean(v) if any(not np.isnan(x) for x in v) else None)
            for k, v in stats.items()
        }

    # Print first batch report as example
    print_report(batch_reports[0])

    # Save JSON
    with open(args.output, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(f"[SanityCheck] Full report saved to: {args.output}")


if __name__ == '__main__':
    main()
