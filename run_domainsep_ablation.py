#!/usr/bin/env python3
"""
DomainSep Mechanism Ablation Script

Runs 4 variants to dissect the DomainSep module:
1. Full              - Original code (target forced shared-only)
2. w/o Domain Sep    - Disable domain separation entirely
3. shared only       - Force BOTH source and target to shared-only
4. w/o target s-only - Remove target shared-only enforcement (target uses private too)

Optional:
5. private only      - Force BOTH source and target to private-only (appendix)

For each variant, the script:
  1. Patches models.py
  2. Runs train_adadrp_program.py
  3. Restores models.py
  4. Records AUC

Usage:
    python run_domainsep_ablation.py

Results saved to: outputs/domainsep_ablation_results.csv
"""

import subprocess
import os
import glob
import re
import csv
import time
import shutil
from collections import defaultdict
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
BASE_CMD = [
    "python", "train_adadrp_program.py",
    "--measurement", "Z_SCORE",
    "--thres_gdsc", "0.0",
    "--thres_label", "0.1",
    "--n", "5",
    "--train",
    "--backbone_type", "convnext1d",
    "--convnext_depth", "1",
    "--convnext_kernel_size", "7",
    "--num_programs", "8",
    "--program_dim", "64",
    "--K", "3",
    "--use_ot_alignment",
    "--lambda_ot", "0.005",
    "--ot_reg", "0.1",
    "--ot_cost_type", "cosine",
    "--ot_marginal_type", "uniform",
    "--use_cdan",
    "--use_domain_sep",
    "--domain_sep_hidden_dim", "256",
    "--domain_sep_output_dim", "64",
    "--lambda_ortho", "0.001",
    "--lambda_recon", "0.005",
    "--lambda_private_domain", "0.005",
    "--lambda_migrate", "0.01",
    "--alph", "1.4",
    "--beta", "0.1",
    "--use_fpi",
    "--fpi_gamma_max", "0.1",
    "--fpi_gamma_init", "0.05",
    "--fpi_q_prior_weight", "0.01",
    "--fpi_dropout", "0.1",
    "--fpi_m_max", "16",
    "--fpi_fp_dim", "1024",
    "--lambda_fpi_gamma_l1", "0.001",
    "--fpi_detach_mode", "none",
    "--pretrain_num_epochs", "100",
    "--train_num_epochs", "100",
    "--contrastive_num_epochs", "100",
]

MODELS_PATH = "adadrp/models.py"
MODELS_BACKUP = "adadrp/models.py.domainsep_ablation_backup"

EXPERIMENT_TIMEOUT = 7200  # 2 hours
LOGDIR_MAX_WAIT = 300  # 5 minutes


# ============================================================================
# Variant definitions: (name, description, patch_func, extra_args)
# ============================================================================

def patch_full(content):
    """No patch needed. Target shared-only enforcement stays."""
    return content, "original"


def patch_no_domain_sep(content):
    """Disable domain separation: replace --use_domain_sep with --no_domain_sep in args,
    but since we build cmd manually, we just note this variant uses extra_args."""
    return content, "no_domain_sep"


def patch_shared_only(content):
    """Force BOTH source and target to use shared-only."""
    old_block = """        rel_weights = self.bml_router(
            pred_s_logits.view(-1),
            pred_p_logits.view(-1),
        )  # [B*D, 2]
        rel_weights = rel_weights.view(batch_size, self.drug_num, 2)

        # CRITICAL FIX: Target (TCGA, domain=1) should NOT use private branch
        # because private_TCGA has never seen drug response labels.
        if domain_idx is not None and (domain_idx == 1).all():
            # Target: force shared-only
            rel_weights = torch.zeros_like(rel_weights)
            rel_weights[:, :, 0] = 1.0  # shared weight = 1
            rel_weights[:, :, 1] = 0.0  # private weight = 0

        # Fused logits
        final_logits = rel_weights[:, :, 0] * pred_s_logits + rel_weights[:, :, 1] * pred_p_logits"""

    new_block = """        rel_weights = self.bml_router(
            pred_s_logits.view(-1),
            pred_p_logits.view(-1),
        )  # [B*D, 2]
        rel_weights = rel_weights.view(batch_size, self.drug_num, 2)

        # ABLATION: shared-only for BOTH source and target
        rel_weights = torch.zeros_like(rel_weights)
        rel_weights[:, :, 0] = 1.0  # shared weight = 1
        rel_weights[:, :, 1] = 0.0  # private weight = 0

        # Fused logits
        final_logits = rel_weights[:, :, 0] * pred_s_logits + rel_weights[:, :, 1] * pred_p_logits"""

    if old_block not in content:
        raise ValueError("Could not find target shared-only block in models.py")
    return content.replace(old_block, new_block), "shared_only"


def patch_no_target_shared_only(content):
    """Remove target shared-only enforcement. Target also uses private."""
    old_block = """        # CRITICAL FIX: Target (TCGA, domain=1) should NOT use private branch
        # because private_TCGA has never seen drug response labels.
        if domain_idx is not None and (domain_idx == 1).all():
            # Target: force shared-only
            rel_weights = torch.zeros_like(rel_weights)
            rel_weights[:, :, 0] = 1.0  # shared weight = 1
            rel_weights[:, :, 1] = 0.0  # private weight = 0

        # Fused logits"""

    new_block = """        # ABLATION: removed target shared-only enforcement
        # Target domain now uses private branch too

        # Fused logits"""

    if old_block not in content:
        raise ValueError("Could not find target shared-only block in models.py")
    return content.replace(old_block, new_block), "no_target_shared_only"


def patch_private_only(content):
    """Force BOTH source and target to use private-only."""
    old_block = """        rel_weights = self.bml_router(
            pred_s_logits.view(-1),
            pred_p_logits.view(-1),
        )  # [B*D, 2]
        rel_weights = rel_weights.view(batch_size, self.drug_num, 2)

        # CRITICAL FIX: Target (TCGA, domain=1) should NOT use private branch
        # because private_TCGA has never seen drug response labels.
        if domain_idx is not None and (domain_idx == 1).all():
            # Target: force shared-only
            rel_weights = torch.zeros_like(rel_weights)
            rel_weights[:, :, 0] = 1.0  # shared weight = 1
            rel_weights[:, :, 1] = 0.0  # private weight = 0

        # Fused logits
        final_logits = rel_weights[:, :, 0] * pred_s_logits + rel_weights[:, :, 1] * pred_p_logits"""

    new_block = """        rel_weights = self.bml_router(
            pred_s_logits.view(-1),
            pred_p_logits.view(-1),
        )  # [B*D, 2]
        rel_weights = rel_weights.view(batch_size, self.drug_num, 2)

        # ABLATION: private-only for BOTH source and target
        rel_weights = torch.zeros_like(rel_weights)
        rel_weights[:, :, 0] = 0.0  # shared weight = 0
        rel_weights[:, :, 1] = 1.0  # private weight = 1

        # Fused logits
        final_logits = rel_weights[:, :, 0] * pred_s_logits + rel_weights[:, :, 1] * pred_p_logits"""

    if old_block not in content:
        raise ValueError("Could not find target shared-only block in models.py")
    return content.replace(old_block, new_block), "private_only"


VARIANTS = [
    ("Full", "Original: target forced shared-only", patch_full, []),
    ("w/o Domain Sep", "Disable domain separation", patch_no_domain_sep, ["--no_domain_sep"]),
    ("shared only", "Both source and target forced shared-only", patch_shared_only, []),
    ("w/o target s-only", "Remove target shared-only enforcement", patch_no_target_shared_only, []),
    ("private only", "Both source and target forced private-only (appendix)", patch_private_only, []),
]


# ============================================================================
# Helper functions (copied from run_hyperparam_tuning.py)
# ============================================================================

def get_existing_log_dirs():
    return set(glob.glob('logs/2026-*'))


def find_new_log_dir(existing_dirs, max_wait=LOGDIR_MAX_WAIT):
    interval = 2
    for _ in range(max_wait // interval):
        current_dirs = set(glob.glob('logs/2026-*'))
        new_dirs = current_dirs - existing_dirs
        if new_dirs:
            return list(new_dirs)[0]
        time.sleep(interval)
    return None


def parse_auc_from_log_dir(log_dir):
    brief_log = os.path.join(log_dir, 'brief_logs.txt')
    if not os.path.exists(brief_log):
        return None
    with open(brief_log, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r'auc[：:]\s*(\d+\.\d+)', content)
    if match:
        return float(match.group(1))
    return None


def save_results(results, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['variant', 'auc', 'elapsed_min', 'note'])
        writer.writeheader()
        for r in results:
            writer.writerow({
                'variant': r['variant'],
                'auc': f"{r['auc']:.4f}" if r['auc'] is not None else 'N/A',
                'elapsed_min': f"{r['elapsed_min']:.1f}",
                'note': r.get('note', ''),
            })


# ============================================================================
# Main execution
# ============================================================================

def run_variant(name, desc, patch_func, extra_args):
    """Run a single variant."""
    print(f"\n{'='*70}")
    print(f"  Variant: {name}")
    print(f"  Description: {desc}")
    print(f"{'='*70}")

    # Read original models.py
    with open(MODELS_PATH, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # Apply patch
    try:
        patched_content, patch_tag = patch_func(original_content)
    except ValueError as e:
        print(f"  [ERROR] Patch failed: {e}")
        return None, 0

    # Write patched models.py
    if patched_content != original_content:
        with open(MODELS_PATH, 'w', encoding='utf-8') as f:
            f.write(patched_content)
        print(f"  [PATCH] Applied: {patch_tag}")
        
        # Verify patch by re-reading
        with open(MODELS_PATH, 'r', encoding='utf-8') as f:
            verify = f.read()
        if verify != patched_content:
            print(f"  [WARNING] File write verification failed!")
        else:
            # Quick sanity check: ensure target force is removed for no_target_s_only
            if patch_tag == "no_target_shared_only":
                if "domain_idx == 1" in verify:
                    print(f"  [WARNING] Patch did not remove target force!")
                else:
                    print(f"  [VERIFY] Target force successfully removed")
            elif patch_tag == "shared_only":
                if "rel_weights[:, :, 0] = 1.0" in verify and "rel_weights[:, :, 1] = 0.0" in verify:
                    print(f"  [VERIFY] Shared-only enforcement confirmed")
            elif patch_tag == "private_only":
                if "rel_weights[:, :, 0] = 0.0" in verify and "rel_weights[:, :, 1] = 1.0" in verify:
                    print(f"  [VERIFY] Private-only enforcement confirmed")
    else:
        print(f"  [PATCH] No change (baseline)")

    # Build command
    cmd = BASE_CMD.copy()
    if extra_args:
        # For w/o Domain Sep: remove --use_domain_sep and all related args
        # Since use_domain_sep defaults to False, simply removing the flag is enough
        if "--no_domain_sep" in extra_args:
            try:
                sep_idx = cmd.index("--use_domain_sep")
                # Remove --use_domain_sep and the next 8 args
                # (--domain_sep_hidden_dim 256 --domain_sep_output_dim 64
                #  --lambda_ortho 0.001 --lambda_recon 0.005
                #  --lambda_private_domain 0.005 --lambda_migrate 0.01)
                del cmd[sep_idx:sep_idx + 9]
            except ValueError:
                pass
        else:
            cmd.extend(extra_args)

    print(f"  Command: {' '.join(cmd[:6])} ...")

    # Run experiment
    existing_dirs = get_existing_log_dirs()
    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=EXPERIMENT_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Timed out after {EXPERIMENT_TIMEOUT/3600:.1f} hours")
        # Restore original code
        with open(MODELS_PATH, 'w', encoding='utf-8') as f:
            f.write(original_content)
        return None, time.time() - start_time

    elapsed = time.time() - start_time

    # Check return code
    if result.returncode != 0:
        print(f"  [ERROR] Failed with code {result.returncode}")
        stderr_lines = result.stderr.strip().split('\n')[-10:] if result.stderr else []
        for line in stderr_lines:
            print(f"    {line}")

    # Parse AUC
    log_dir = find_new_log_dir(existing_dirs)
    auc = None
    if log_dir:
        print(f"  Log dir: {os.path.basename(log_dir)}")
        auc = parse_auc_from_log_dir(log_dir)
        if auc is not None:
            print(f"  Result: AUC = {auc:.4f}  (time: {elapsed/60:.1f} min)")
        else:
            print(f"  [WARNING] Could not parse AUC")
    else:
        print(f"  [WARNING] Could not find log dir")

    # Restore original models.py
    with open(MODELS_PATH, 'w', encoding='utf-8') as f:
        f.write(original_content)
    print(f"  [RESTORE] models.py restored to original")

    return auc, elapsed


def main():
    output_csv = 'outputs/domainsep_ablation_results.csv'
    results = []

    print("="*70)
    print("  DomainSep Mechanism Ablation")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Variants: {len(VARIANTS)}")
    print(f"  Estimated: ~{len(VARIANTS) * 20} minutes")
    print("="*70)

    # Backup models.py at start
    shutil.copy2(MODELS_PATH, MODELS_BACKUP)
    print(f"\n[BACKUP] {MODELS_PATH} -> {MODELS_BACKUP}")

    for name, desc, patch_func, extra_args in VARIANTS:
        auc, elapsed = run_variant(name, desc, patch_func, extra_args)
        results.append({
            'variant': name,
            'auc': auc,
            'elapsed_min': elapsed / 60,
            'note': 'failed' if auc is None else '',
        })
        save_results(results, output_csv)

    # Final summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    full_auc = None
    for r in results:
        if r['variant'] == 'Full':
            full_auc = r['auc']
            break

    for r in results:
        auc_str = f"{r['auc']:.4f}" if r['auc'] is not None else 'N/A'
        delta_str = ''
        if r['auc'] is not None and full_auc is not None and r['variant'] != 'Full':
            delta = r['auc'] - full_auc
            delta_str = f" (Δ={delta:+.4f})"
        print(f"  {r['variant']:20s}: AUC = {auc_str}{delta_str}")

    # Interpretation guide
    print("\n" + "="*70)
    print("  INTERPRETATION")
    print("="*70)
    print("""
  Full vs w/o Domain Sep      → DomainSep decomposition overall value
  Full vs shared only         → Private branch value (source-specific signal)
  Full vs w/o target s-only   → Target shared-only necessity
  Full vs private only        → Shared branch value (domain-invariant signal)
    """)

    print(f"\n  Results saved to: {output_csv}")
    print(f"  End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Clean up backup
    if os.path.exists(MODELS_BACKUP):
        os.remove(MODELS_BACKUP)
        print(f"[CLEANUP] Removed {MODELS_BACKUP}")


if __name__ == '__main__':
    main()
