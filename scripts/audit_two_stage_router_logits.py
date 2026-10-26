#!/usr/bin/env python3
"""
Phase 9-F: Two-stage Router Logit Sanity Check

Audit script to verify:
1. Non-target drug logits are invariant between Global-only and Two-stage
2. Cross mask only affects the specified target drug (default: Temozolomide)
3. Checkpoint consistency between Stage A and Stage B
4. Root cause of any non-target logit drift

Usage:
    python scripts/audit_two_stage_router_logits.py \
      --global_checkpoint_dir checkpoints/router_global_only \
      --two_stage_checkpoint_dir logs/<timestamp>_param_str \
      --output_dir outputs/router_logit_audit \
      --drug_name_for_cross Temozolomide \
      --folds 0,1,2,3,4 \
      --tolerance 1e-6 \
      --save_logits \
      --recompute_metrics
"""

import argparse
import os
import sys
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataload import get_multi_labeled_dataloader, get_drug_feat
from adadrp.models import MolecularProgramEncoder, ProgramFusionMLP
from adadrp.fragment_utils import load_drug_fragments

warnings.filterwarnings('ignore')

# =============================================================================
# 0. Constants matching train_adadrp_program.py defaults
# =============================================================================

DRUGS = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
         "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"]
DRUG_NAME_TO_IDX = {d: i for i, d in enumerate(DRUGS)}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Two-stage Router logits")
    parser.add_argument('--global_checkpoint_dir', type=str, required=True,
                        help='Directory containing fusion_model_fold{fold}.pt (Stage A)')
    parser.add_argument('--two_stage_checkpoint_dir', type=str, required=True,
                        help='Directory containing fold-{fold}/fusion_model.pt (Stage B)')
    parser.add_argument('--output_dir', type=str, default='outputs/router_logit_audit')
    parser.add_argument('--drug_name_for_cross', type=str, default='Temozolomide')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4')
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--save_logits', action='store_true')
    parser.add_argument('--recompute_metrics', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fpi_fragment_npz', type=str, default='data/drug_fragments_brics_m16_fp1024.npz')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--measurement', type=str, default='Z_SCORE')
    parser.add_argument('--thres_gdsc', type=float, default=0.0)
    parser.add_argument('--thres_label', type=float, default=0.1)
    parser.add_argument('--n_splits', type=int, default=5)
    # Model architecture defaults (must match training config)
    parser.add_argument('--backbone_type', type=str, default='convnext1d')
    parser.add_argument('--convnext_depth', type=int, default=1)
    parser.add_argument('--convnext_kernel_size', type=int, default=7)
    parser.add_argument('--num_programs', type=int, default=8)
    parser.add_argument('--program_dim', type=int, default=64)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--lambda_ot', type=float, default=0.005)
    # Router config defaults (must match training config)
    parser.add_argument('--router_gamma_global_max', type=float, default=0.1)
    parser.add_argument('--router_gamma_global_init', type=float, default=0.05)
    parser.add_argument('--router_gamma_cross_max', type=float, default=0.05)
    parser.add_argument('--router_gamma_cross_init', type=float, default=0.0)
    parser.add_argument('--lambda_router_gamma_l1', type=float, default=0.0)
    parser.add_argument('--router_residual_global_scale', type=float, default=1.0)
    parser.add_argument('--router_residual_cross_scale', type=float, default=1.0)
    parser.add_argument('--router_cross_mode', type=str, default='oracle_mask')
    parser.add_argument('--router_oracle_cross_drugs', type=str, default='Temozolomide')
    parser.add_argument('--router_cross_start_epoch', type=int, default=0)
    parser.add_argument('--router_cross_ramp_epochs', type=int, default=0)
    parser.add_argument('--router_cross_detach_mode', type=str, default='none')
    parser.add_argument('--fpi_m_max', type=int, default=16)
    parser.add_argument('--fpi_fp_dim', type=int, default=1024)
    parser.add_argument('--fpi_q_prior_weight', type=float, default=0.01)
    parser.add_argument('--fpi_dropout', type=float, default=0.1)
    return parser.parse_args()


# =============================================================================
# 1. Model construction
# =============================================================================

def build_router_config(args):
    """Build fpi_router_config dict matching training script."""
    return {
        'program_dim': args.program_dim,
        'drug_dim': 64,
        'fragment_dim': args.program_dim,
        'num_programs': args.num_programs,
        'num_drugs': len(DRUGS),
        'max_fragments': args.fpi_m_max,
        'q_prior_weight': args.fpi_q_prior_weight,
        'gamma_global_max': args.router_gamma_global_max,
        'gamma_global_init': args.router_gamma_global_init,
        'gamma_cross_max': args.router_gamma_cross_max,
        'gamma_cross_init': args.router_gamma_cross_init,
        'dropout': args.fpi_dropout,
        'router_cross_start_epoch': args.router_cross_start_epoch,
        'router_cross_ramp_epochs': args.router_cross_ramp_epochs,
        'router_cross_mode': args.router_cross_mode,
        'router_oracle_cross_drugs': args.router_oracle_cross_drugs,
        'router_residual_global_scale': args.router_residual_global_scale,
        'router_residual_cross_scale': args.router_residual_cross_scale,
        'router_branch_effect_threshold': 1e-4,
        'router_cross_detach_mode': args.router_cross_detach_mode,
        'debug_fixed_gamma_global': None,
    }


def build_model(device, frag_fp, frag_mask, router_config, use_fpi_router=True):
    """Construct ProgramFusionMLP with exact training configuration."""
    encoder = MolecularProgramEncoder(
        input_dim=1306,
        num_programs=router_config['num_programs'],
        program_dim=router_config['program_dim'],
        backbone_type='convnext1d',
        convnext_depth=1,
        convnext_kernel_size=7,
        convnext_expansion=4,
    )
    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=256,
        output_dim=1,
        gene_num=1306,
        drug_num=len(DRUGS),
        bilinear_dim=128,
        gene_feat_dim=64,
        drug_feat_dim=64,
        num_programs=router_config['num_programs'],
        program_dim=router_config['program_dim'],
        drop=0.1,
        use_fpi_router=use_fpi_router,
        fpi_router_config=router_config,
        frag_fp=frag_fp,
        frag_mask=frag_mask,
        return_dict=True,
    ).to(device)
    return model


# =============================================================================
# 2. Checkpoint loading
# =============================================================================

def load_checkpoint(model, checkpoint_path, device, desc=""):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [{desc}] Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  [{desc}] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    return model


# =============================================================================
# 3. Data loading
# =============================================================================

def build_test_dataloader(args):
    """Build TCGA test dataloader (same as training script)."""
    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)
    labeled_dataloader = get_multi_labeled_dataloader(
        gex_features_df=gex_features_df,
        drug=DRUGS,
        seed=args.seed,
        batch_size=args.batch_size,
        ccle_measurement=args.measurement,
        threshold_gdsc=args.thres_gdsc,
        threshold_label=args.thres_label,
        n_splits=args.n_splits,
    )
    # TCGA dataloader is the same for all folds; extract once
    _, _, tcga_dataloader = next(labeled_dataloader)
    return tcga_dataloader


# =============================================================================
# 4. Forward pass with full extraction
# =============================================================================

@torch.no_grad()
def forward_extract(model, dataloader, drug_feat, device, drug_names_list, current_epoch=9999):
    """
    Run model on dataloader and extract all router-related tensors.
    
    Returns dict with numpy arrays:
        final_logits, residual_global, residual_cross,
        gamma_global, gamma_cross, effective_gamma_cross, cross_active_mask,
        labels, masks, base_logits (inferred)
    """
    model.eval()
    
    all_final_logits = []
    all_residual_global = []
    all_residual_cross = []
    all_labels = []
    all_masks = []
    
    # Gamma and mask are constant across batches (same model params)
    gamma_global_ref = None
    gamma_cross_ref = None
    effective_gamma_cross_ref = None
    cross_active_mask_ref = None
    
    scale_global = model.fpi_router.router_residual_global_scale
    scale_cross = model.fpi_router.router_residual_cross_scale
    
    for batch_idx, (x, y, mask, tissue) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        tissue = tissue.to(device)
        
        result = model(
            x, drug_feat, tissue,
            return_dict=True,
            current_epoch=current_epoch,
            drug_names=drug_names_list,
        )
        
        router_out = result.get('router_out', None)
        if router_out is None:
            raise RuntimeError(
                "router_out not found. Ensure use_fpi_router=True and model has router."
            )
        
        all_final_logits.append(router_out['final_logits'].cpu().numpy())
        all_residual_global.append(router_out['residual_global'].cpu().numpy())
        all_residual_cross.append(router_out['residual_cross'].cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_masks.append(mask.cpu().numpy())
        
        # Capture gamma/mask from first batch (should be identical across batches)
        if gamma_global_ref is None:
            gamma_global_ref = router_out['gamma_global'].cpu().numpy()
            gamma_cross_ref = router_out['gamma_cross'].cpu().numpy()
            effective_gamma_cross_ref = router_out['effective_gamma_cross'].cpu().numpy()
            cross_active_mask_ref = router_out['cross_active_mask'].cpu().numpy()
    
    final_logits = np.concatenate(all_final_logits, axis=0)      # [N, D]
    residual_global = np.concatenate(all_residual_global, axis=0)  # [N, D]
    residual_cross = np.concatenate(all_residual_cross, axis=0)    # [N, D]
    labels = np.concatenate(all_labels, axis=0)                   # [N, D]
    masks = np.concatenate(all_masks, axis=0)                     # [N, D]
    
    # Inferred base_logits = final_logits - gamma_global * scale_global * residual_global - effective_gamma_cross * scale_cross * residual_cross
    gamma_global_b = gamma_global_ref[np.newaxis, :]  # [1, D]
    eff_gamma_cross_b = effective_gamma_cross_ref[np.newaxis, :]  # [1, D]
    
    base_logits = (
        final_logits
        - gamma_global_b * scale_global * residual_global
        - eff_gamma_cross_b * scale_cross * residual_cross
    )
    
    return {
        'final_logits': final_logits,
        'base_logits': base_logits,
        'residual_global': residual_global,
        'residual_cross': residual_cross,
        'gamma_global': gamma_global_ref,
        'gamma_cross': gamma_cross_ref,
        'effective_gamma_cross': effective_gamma_cross_ref,
        'cross_active_mask': cross_active_mask_ref,
        'labels': labels,
        'masks': masks,
    }


# =============================================================================
# 5. Parameter hash & comparison
# =============================================================================

def compute_module_hash(module):
    """Compute SHA256 hash of all parameters and buffers in a module."""
    hasher = hashlib.sha256()
    for name, param in module.named_parameters():
        hasher.update(param.detach().cpu().numpy().tobytes())
    for name, buf in module.named_buffers():
        hasher.update(buf.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def compare_module_params(global_model, two_stage_model, module_path, tolerance=1e-6):
    """
    Compare a specific submodule between global-only and two-stage models.
    Returns list of dicts with diff info.
    """
    diffs = []
    
    def get_module(model, path):
        parts = path.split('.')
        mod = model
        for p in parts:
            mod = getattr(mod, p, None)
            if mod is None:
                return None
        return mod
    
    mod_g = get_module(global_model, module_path)
    mod_t = get_module(two_stage_model, module_path)
    
    if mod_g is None or mod_t is None:
        return diffs
    
    # Compare parameters
    params_g = dict(mod_g.named_parameters())
    params_t = dict(mod_t.named_parameters())
    for p_name in params_g:
        if p_name in params_t:
            diff = (params_g[p_name] - params_t[p_name]).abs().max().item()
            mean_diff = (params_g[p_name] - params_t[p_name]).abs().mean().item()
            if diff > tolerance:
                diffs.append({
                    'module_name': module_path,
                    'param_or_buffer_name': p_name,
                    'max_abs_diff': diff,
                    'mean_abs_diff': mean_diff,
                })
    
    # Compare buffers (especially BatchNorm running_mean/var)
    bufs_g = dict(mod_g.named_buffers())
    bufs_t = dict(mod_t.named_buffers())
    for b_name in bufs_g:
        if b_name in bufs_t:
            diff = (bufs_g[b_name] - bufs_t[b_name]).abs().max().item()
            mean_diff = (bufs_g[b_name] - bufs_t[b_name]).abs().mean().item()
            if diff > tolerance:
                diffs.append({
                    'module_name': module_path,
                    'param_or_buffer_name': b_name,
                    'max_abs_diff': diff,
                    'mean_abs_diff': mean_diff,
                })
    
    return diffs


# =============================================================================
# 6. Recompute AUC from logits
# =============================================================================

def recompute_auc_from_logits(logits, labels, masks):
    """
    logits, labels, masks: [N, D]
    Returns per-drug AUC list.
    """
    auc_list = []
    for d in range(logits.shape[1]):
        valid = masks[:, d] > 0
        y_true_d = labels[valid, d]
        y_pred_d = logits[valid, d]
        if len(np.unique(y_true_d)) > 1:
            auc = roc_auc_score(y_true_d, y_pred_d)
        else:
            auc = np.nan
        auc_list.append(auc)
    return auc_list


# =============================================================================
# 7. Main audit logic
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    folds = [int(f.strip()) for f in args.folds.split(',')]
    target_drug_idx = DRUG_NAME_TO_IDX[args.drug_name_for_cross]
    
    # Load fragment data
    print(f"Loading fragments from {args.fpi_fragment_npz}...")
    frag_data = load_drug_fragments(args.fpi_fragment_npz, expected_drug_names=DRUGS)
    frag_fp = frag_data['frag_fp'].to(device)
    frag_mask = frag_data['frag_mask'].to(device)
    
    # Build router config
    router_config = build_router_config(args)
    
    # Build test dataloader (TCGA, same for all folds)
    print("Building TCGA test dataloader...")
    tcga_dataloader = build_test_dataloader(args)
    print(f"  TCGA batches: {len(tcga_dataloader)}")
    
    # Load drug features (same as training)
    drug_feat = get_drug_feat(DRUGS)
    drug_feat = torch.from_numpy(drug_feat).float().to(device)
    
    # Prepare result collectors
    per_fold_records = []
    per_fold_mask_checks = []
    per_fold_checkpoint_records = []
    per_fold_param_diffs = []
    
    for fold in folds:
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")
        
        # ------------------------------------------------------------------
        # 7.1 Load checkpoints
        # ------------------------------------------------------------------
        global_ckpt_path = os.path.join(args.global_checkpoint_dir, f'fusion_model_fold{fold}.pt')
        
        # Two-stage checkpoint may be directly in dir or in fold-{fold}/ subdir
        two_stage_ckpt_path = os.path.join(args.two_stage_checkpoint_dir, f'fold-{fold}', 'fusion_model.pt')
        if not os.path.exists(two_stage_ckpt_path):
            two_stage_ckpt_path = os.path.join(args.two_stage_checkpoint_dir, f'fusion_model_fold{fold}.pt')
        
        print(f"Global ckpt:  {global_ckpt_path}")
        print(f"Two-stage ckpt: {two_stage_ckpt_path}")
        
        global_exists = os.path.exists(global_ckpt_path)
        two_stage_exists = os.path.exists(two_stage_ckpt_path)
        
        per_fold_checkpoint_records.append({
            'fold': fold,
            'global_checkpoint_path': global_ckpt_path,
            'two_stage_checkpoint_path': two_stage_ckpt_path,
            'global_checkpoint_exists': global_exists,
            'two_stage_checkpoint_exists': two_stage_exists,
        })
        
        if not global_exists or not two_stage_exists:
            print(f"  SKIPPING fold {fold}: checkpoint missing")
            continue
        
        # ------------------------------------------------------------------
        # 7.2 Build models
        # ------------------------------------------------------------------
        global_model = build_model(device, frag_fp, frag_mask, router_config, use_fpi_router=True)
        two_stage_model = build_model(device, frag_fp, frag_mask, router_config, use_fpi_router=True)
        
        load_checkpoint(global_model, global_ckpt_path, device, desc=f"fold{fold}_global")
        load_checkpoint(two_stage_model, two_stage_ckpt_path, device, desc=f"fold{fold}_two_stage")
        
        # ------------------------------------------------------------------
        # 7.3 Parameter hash check
        # ------------------------------------------------------------------
        main_modules = [
            'encoder',
            'drug_mlp',
            'bilinear',
            'fusion_module',
            'cdan_discriminator',
            'drug_classifier',
            'fragment_encoder',
            'fpi_router.global_fpi',
            'fpi_router.raw_gamma_global',
        ]
        
        global_hash = compute_module_hash(global_model)
        two_stage_hash = compute_module_hash(two_stage_model)
        
        print(f"  Global model hash:   {global_hash[:16]}...")
        print(f"  Two-stage model hash: {two_stage_hash[:16]}...")
        
        # Compare individual main modules
        all_module_diffs = []
        for mod_path in main_modules:
            diffs = compare_module_params(global_model, two_stage_model, mod_path, tolerance=args.tolerance)
            all_module_diffs.extend(diffs)
            if diffs:
                print(f"  MODULE DIFF: {mod_path} has {len(diffs)} differing params/buffers")
        
        # Update checkpoint record
        per_fold_checkpoint_records[-1]['global_param_hash'] = global_hash
        per_fold_checkpoint_records[-1]['two_stage_main_param_hash'] = two_stage_hash
        per_fold_checkpoint_records[-1]['main_params_match'] = len(all_module_diffs) == 0
        
        per_fold_param_diffs.extend([{**d, 'fold': fold} for d in all_module_diffs])
        
        # ------------------------------------------------------------------
        # 7.4 Forward both models
        # ------------------------------------------------------------------
        print("  Forwarding Global-only model...")
        global_out = forward_extract(global_model, tcga_dataloader, drug_feat, device, DRUGS, current_epoch=9999)
        
        print("  Forwarding Two-stage model...")
        two_stage_out = forward_extract(two_stage_model, tcga_dataloader, drug_feat, device, DRUGS, current_epoch=9999)
        
        # Verify sample count
        assert global_out['final_logits'].shape == two_stage_out['final_logits'].shape, \
            f"Shape mismatch: {global_out['final_logits'].shape} vs {two_stage_out['final_logits'].shape}"
        N, D = global_out['final_logits'].shape
        print(f"  Samples: {N}, Drugs: {D}")
        
        # ------------------------------------------------------------------
        # 7.5 Per-drug logit diff analysis
        # ------------------------------------------------------------------
        for d_idx, drug_name in enumerate(DRUGS):
            is_target = (d_idx == target_drug_idx)
            
            # Gamma values
            gg_global = global_out['gamma_global'][d_idx]
            gg_two_stage = two_stage_out['gamma_global'][d_idx]
            gg_diff = abs(gg_two_stage - gg_global)
            
            gc_two_stage = two_stage_out['gamma_cross'][d_idx]
            eff_gc_two_stage = two_stage_out['effective_gamma_cross'][d_idx]
            cam_two_stage = two_stage_out['cross_active_mask'][d_idx]
            
            # Logit diffs
            base_max_diff = np.abs(two_stage_out['base_logits'][:, d_idx] - global_out['base_logits'][:, d_idx]).max()
            base_mean_diff = np.abs(two_stage_out['base_logits'][:, d_idx] - global_out['base_logits'][:, d_idx]).mean()
            
            res_g_max_diff = np.abs(two_stage_out['residual_global'][:, d_idx] - global_out['residual_global'][:, d_idx]).max()
            res_g_mean_diff = np.abs(two_stage_out['residual_global'][:, d_idx] - global_out['residual_global'][:, d_idx]).mean()
            
            final_max_diff = np.abs(two_stage_out['final_logits'][:, d_idx] - global_out['final_logits'][:, d_idx]).max()
            final_mean_diff = np.abs(two_stage_out['final_logits'][:, d_idx] - global_out['final_logits'][:, d_idx]).mean()
            
            # Cross effect on this drug
            cross_effect_mean = np.abs(
                two_stage_out['effective_gamma_cross'][d_idx] * two_stage_out['residual_cross'][:, d_idx]
            ).mean()
            
            # Pass/fail for non-target drugs
            if not is_target:
                passed = (
                    abs(cam_two_stage) < args.tolerance
                    and abs(eff_gc_two_stage) < args.tolerance
                    and final_max_diff < args.tolerance
                )
            else:
                passed = (abs(cam_two_stage - 1.0) < args.tolerance)
            
            per_fold_records.append({
                'fold': fold,
                'drug_name': drug_name,
                'is_cross_target': is_target,
                'cross_active_mask': float(cam_two_stage),
                'gamma_global_global': float(gg_global),
                'gamma_global_two_stage': float(gg_two_stage),
                'gamma_global_abs_diff': float(gg_diff),
                'gamma_cross_two_stage': float(gc_two_stage),
                'effective_gamma_cross_two_stage': float(eff_gc_two_stage),
                'base_max_abs_diff': float(base_max_diff),
                'base_mean_abs_diff': float(base_mean_diff),
                'residual_global_max_abs_diff': float(res_g_max_diff),
                'residual_global_mean_abs_diff': float(res_g_mean_diff),
                'final_max_abs_diff': float(final_max_diff),
                'final_mean_abs_diff': float(final_mean_diff),
                'cross_effect_abs_mean': float(cross_effect_mean),
                'passed_non_target_invariance': passed,
            })
            
            # Mask check
            expected_mask = 1.0 if is_target else 0.0
            per_fold_mask_checks.append({
                'fold': fold,
                'drug_name': drug_name,
                'cross_active_mask': float(cam_two_stage),
                'effective_gamma_cross': float(eff_gc_two_stage),
                'expected_mask': expected_mask,
                'mask_correct': abs(cam_two_stage - expected_mask) < args.tolerance,
            })
        
        # ------------------------------------------------------------------
        # 7.6 Save logits if requested
        # ------------------------------------------------------------------
        if args.save_logits:
            npz_path = os.path.join(args.output_dir, f'logits_fold{fold}.npz')
            np.savez(
                npz_path,
                final_logits_global=global_out['final_logits'],
                final_logits_two_stage=two_stage_out['final_logits'],
                base_logits_global=global_out['base_logits'],
                base_logits_two_stage=two_stage_out['base_logits'],
                residual_global_global=global_out['residual_global'],
                residual_global_two_stage=two_stage_out['residual_global'],
                residual_cross_two_stage=two_stage_out['residual_cross'],
                labels=global_out['labels'],
                masks=global_out['masks'],
                drug_names=np.array(DRUGS),
            )
            print(f"  Saved logits to {npz_path}")
        
        # ------------------------------------------------------------------
        # 7.7 Recompute metrics if requested
        # ------------------------------------------------------------------
        if args.recompute_metrics:
            print("  Recomputing AUC from saved logits...")
            auc_global = recompute_auc_from_logits(global_out['final_logits'], global_out['labels'], global_out['masks'])
            auc_two_stage = recompute_auc_from_logits(two_stage_out['final_logits'], two_stage_out['labels'], two_stage_out['masks'])
            
            recompute_records = []
            for d_idx, drug_name in enumerate(DRUGS):
                final_max_diff = np.abs(two_stage_out['final_logits'][:, d_idx] - global_out['final_logits'][:, d_idx]).max()
                recompute_records.append({
                    'fold': fold,
                    'drug_name': drug_name,
                    'auc_global': auc_global[d_idx],
                    'auc_two_stage': auc_two_stage[d_idx],
                    'delta_auc': auc_two_stage[d_idx] - auc_global[d_idx],
                    'final_max_abs_diff': final_max_diff,
                })
            
            recompute_df = pd.DataFrame(recompute_records)
            recompute_csv = os.path.join(args.output_dir, 'recomputed_auc.csv')
            if os.path.exists(recompute_csv):
                recompute_df.to_csv(recompute_csv, mode='a', header=False, index=False)
            else:
                recompute_df.to_csv(recompute_csv, index=False)
            print(f"  Saved recomputed AUC to {recompute_csv}")
    
    # =====================================================================
    # 8. Aggregate and save results
    # =====================================================================
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")
    
    # 8.1 Per-fold diff CSV
    diff_df = pd.DataFrame(per_fold_records)
    diff_csv = os.path.join(args.output_dir, 'logit_diff_per_fold.csv')
    diff_df.to_csv(diff_csv, index=False)
    print(f"Saved: {diff_csv}")
    
    # 8.2 Mask check CSV
    mask_df = pd.DataFrame(per_fold_mask_checks)
    mask_csv = os.path.join(args.output_dir, 'cross_mask_check.csv')
    mask_df.to_csv(mask_csv, index=False)
    print(f"Saved: {mask_csv}")
    
    # 8.3 Checkpoint pairing CSV
    ckpt_df = pd.DataFrame(per_fold_checkpoint_records)
    ckpt_csv = os.path.join(args.output_dir, 'checkpoint_pairing.csv')
    ckpt_df.to_csv(ckpt_csv, index=False)
    print(f"Saved: {ckpt_csv}")
    
    # 8.4 Main param diff CSV
    if per_fold_param_diffs:
        param_diff_df = pd.DataFrame(per_fold_param_diffs)
        param_diff_csv = os.path.join(args.output_dir, 'main_param_diff.csv')
        param_diff_df.to_csv(param_diff_csv, index=False)
        print(f"Saved: {param_diff_csv}")
    
    # 8.5 Summary CSV (aggregated across folds)
    summary_records = []
    for drug_name in DRUGS:
        drug_df = diff_df[diff_df['drug_name'] == drug_name]
        if len(drug_df) == 0:
            continue
        is_target = drug_name == args.drug_name_for_cross
        all_passed = drug_df['passed_non_target_invariance'].all()
        
        summary_records.append({
            'drug_name': drug_name,
            'is_cross_target': is_target,
            'mean_final_max_abs_diff': drug_df['final_max_abs_diff'].mean(),
            'max_final_max_abs_diff': drug_df['final_max_abs_diff'].max(),
            'mean_final_mean_abs_diff': drug_df['final_mean_abs_diff'].mean(),
            'mean_base_max_abs_diff': drug_df['base_max_abs_diff'].mean(),
            'mean_residual_global_max_abs_diff': drug_df['residual_global_max_abs_diff'].mean(),
            'mean_gamma_global_abs_diff': drug_df['gamma_global_abs_diff'].mean(),
            'mean_cross_effect_abs_mean': drug_df['cross_effect_abs_mean'].mean(),
            'all_folds_passed': all_passed,
        })
    
    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(args.output_dir, 'logit_diff_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")
    
    # =====================================================================
    # 9. Diagnosis & Pass/Fail Report
    # =====================================================================
    print(f"\n{'='*60}")
    print("DIAGNOSIS REPORT")
    print(f"{'='*60}")
    
    # Non-target drugs
    non_target_df = diff_df[~diff_df['is_cross_target']]
    non_target_failed = non_target_df[~non_target_df['passed_non_target_invariance']]
    
    if len(non_target_failed) == 0:
        print("\n✅ PASS: All non-target drugs passed invariance check.")
        print("   Two-stage is strict target-only compensation.")
    else:
        print(f"\n❌ FAIL: {len(non_target_failed)} non-target drug-fold pairs failed invariance.")
        print("\nFailed pairs:")
        for _, row in non_target_failed.iterrows():
            print(f"  Fold {int(row['fold'])}, {row['drug_name']}: "
                  f"final_max_diff={row['final_max_abs_diff']:.2e}, "
                  f"eff_gamma_cross={row['effective_gamma_cross_two_stage']:.2e}, "
                  f"base_diff={row['base_max_abs_diff']:.2e}, "
                  f"res_g_diff={row['residual_global_max_abs_diff']:.2e}, "
                  f"gamma_g_diff={row['gamma_global_abs_diff']:.2e}")
        
        # Auto-diagnose root cause
        print("\n--- Root Cause Analysis ---")
        
        # Case A: effective_gamma_cross != 0 for non-target
        case_a = non_target_failed[non_target_failed['effective_gamma_cross_two_stage'].abs() >= args.tolerance]
        if len(case_a) > 0:
            print("Case A (MASK BUG): Non-target effective_gamma_cross != 0")
            print("  -> cross_active_mask or broadcasting is incorrect.")
            print(f"  Affected: {case_a[['fold','drug_name','effective_gamma_cross_two_stage']].to_string(index=False)}")
        
        # Case B: base_logits diff > tolerance
        case_b = non_target_failed[
            (non_target_failed['effective_gamma_cross_two_stage'].abs() < args.tolerance) &
            (non_target_failed['base_max_abs_diff'] >= args.tolerance)
        ]
        if len(case_b) > 0:
            print("Case B (BASE PATH CHANGED): base_logits differ but cross is zero")
            print("  -> Checkpoint mismatch, incomplete freeze, or BatchNorm buffer drift.")
            print(f"  Affected: {case_b[['fold','drug_name','base_max_abs_diff']].to_string(index=False)}")
        
        # Case C: residual_global diff > tolerance
        case_c = non_target_failed[
            (non_target_failed['effective_gamma_cross_two_stage'].abs() < args.tolerance) &
            (non_target_failed['base_max_abs_diff'] < args.tolerance) &
            (non_target_failed['residual_global_max_abs_diff'] >= args.tolerance)
        ]
        if len(case_c) > 0:
            print("Case C (GLOBAL BRANCH CHANGED): global FPI branch differs")
            print("  -> global_fpi not frozen or raw_gamma_global changed.")
            print(f"  Affected: {case_c[['fold','drug_name','residual_global_max_abs_diff']].to_string(index=False)}")
        
        # Case D: gamma_global diff > tolerance
        case_d = non_target_failed[
            (non_target_failed['effective_gamma_cross_two_stage'].abs() < args.tolerance) &
            (non_target_failed['base_max_abs_diff'] < args.tolerance) &
            (non_target_failed['residual_global_max_abs_diff'] < args.tolerance) &
            (non_target_failed['gamma_global_abs_diff'] >= args.tolerance)
        ]
        if len(case_d) > 0:
            print("Case D (GAMMA CHANGED): gamma_global differs")
            print("  -> raw_gamma_global not frozen.")
            print(f"  Affected: {case_d[['fold','drug_name','gamma_global_abs_diff']].to_string(index=False)}")
        
        # Case E: all logits component same but final differ
        case_e = non_target_failed[
            (non_target_failed['effective_gamma_cross_two_stage'].abs() < args.tolerance) &
            (non_target_failed['base_max_abs_diff'] < args.tolerance) &
            (non_target_failed['residual_global_max_abs_diff'] < args.tolerance) &
            (non_target_failed['gamma_global_abs_diff'] < args.tolerance) &
            (non_target_failed['final_max_abs_diff'] >= args.tolerance)
        ]
        if len(case_e) > 0:
            print("Case E (FINAL LOGIT BUG): All components match but final_logits differ")
            print("  -> Addition or broadcasting bug in final_logits computation.")
            print(f"  Affected: {case_e[['fold','drug_name','final_max_abs_diff']].to_string(index=False)}")
    
    # Target drug report
    target_df = diff_df[diff_df['is_cross_target']]
    print(f"\n--- Target Drug ({args.drug_name_for_cross}) Report ---")
    print(f"  cross_active_mask: {target_df['cross_active_mask'].iloc[0]:.6f} (expected 1.0)")
    print(f"  mean effective_gamma_cross: {target_df['effective_gamma_cross_two_stage'].mean():.6f}")
    print(f"  mean cross_effect_abs_mean: {target_df['cross_effect_abs_mean'].mean():.6f}")
    print(f"  mean final_max_abs_diff: {target_df['final_max_abs_diff'].mean():.6f}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("AUDIT COMPLETE")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
