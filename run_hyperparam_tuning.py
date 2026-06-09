#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for MACB-DRP (S3-S6 Only)

Main text (S3-S4): fpi_gamma_max, num_programs
Appendix (S5-S6): lambda_ot, fpi_m_max

Excluded (not core innovations):
- S1 (alph) and S2 (beta): justified in paper text, no grid search needed.

Note: num_programs=1 and lambda_ot=0 are EXCLUDED because they overlap with
completed ablation experiments (Exp7: single program, Exp5: w/o OT).

Usage:
    python run_hyperparam_tuning.py

Results saved to: outputs/hyperparam_tuning/S3_S6_results.csv
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
# Baseline configuration (Experiment 2: Current Best, AUC=0.6220)
# ============================================================================
BASE_PARAMS = {
    'measurement': 'Z_SCORE',
    'thres_gdsc': 0.0,
    'thres_label': 0.1,
    'n': 2,
    'backbone_type': 'convnext1d',
    'convnext_depth': 1,
    'convnext_kernel_size': 7,
    'num_programs': 8,
    'program_dim': 64,
    'K': 3,
    'use_ot_alignment': True,
    'lambda_ot': 0.005,
    'ot_reg': 0.1,
    'ot_cost_type': 'cosine',
    'ot_marginal_type': 'uniform',
    'use_cdan': True,
    'use_domain_sep': True,
    'domain_sep_hidden_dim': 256,
    'domain_sep_output_dim': 64,
    'lambda_ortho': 0.001,
    'lambda_recon': 0.005,
    'lambda_private_domain': 0.005,
    'lambda_migrate': 0.01,
    'alph': 1.4,
    'beta': 0.1,
    'use_fpi': True,
    'fpi_gamma_max': 0.1,
    'fpi_gamma_init': 0.05,
    'fpi_q_prior_weight': 0.01,
    'fpi_dropout': 0.1,
    'fpi_m_max': 16,
    'fpi_fp_dim': 1024,
    'lambda_fpi_gamma_l1': 0.001,
    'fpi_detach_mode': 'none',
    'pretrain_num_epochs': 100,
    'train_num_epochs': 100,
    'contrastive_num_epochs': 100,
}

# ============================================================================
# Search spaces (S3-S6 only)
# ============================================================================
MAIN_TEXT_SPACES = {
    # S3: FPI gamma max (core FPI module parameter)
    'fpi_gamma_max': [0.0, 0.05, 0.1, 0.2, 0.3],
    # S4: Number of programs (structural parameter, excluding 1 which is done in ablation)
    'num_programs': [4, 8, 12, 16],
}

APPENDIX_SPACES = {
    # S5: OT alignment strength (excluding 0 which is done in ablation Exp5)
    'lambda_ot': [0.001, 0.005, 0.01, 0.02],
    # S6: FPI fragment count (detail parameter)
    'fpi_m_max': [4, 8, 16, 32],
}

# Timeout for each experiment (seconds)
EXPERIMENT_TIMEOUT = 7200  # 2 hours
# Wait time for log directory to appear (seconds)
LOGDIR_MAX_WAIT = 300  # 5 minutes


def build_param_str(params):
    """Reconstruct param_str for validation (must match train_mmpa_program.py logic)."""
    ps = f"pretrain_num_epochs_{params['pretrain_num_epochs']}"
    ps += f"_train_num_epochs_{params['train_num_epochs']}"
    ps += f"_contrastive_num_epochs_{params['contrastive_num_epochs']}"
    ps += f"_K_{params['K']}"
    
    if params.get('use_domain_sep'):
        ps += f"_domsep_h{params['domain_sep_hidden_dim']}_o{params['domain_sep_output_dim']}"
        ps += f"_ortho{params['lambda_ortho']}_recon{params['lambda_recon']}"
        ps += f"_privdom{params['lambda_private_domain']}_migrate{params['lambda_migrate']}"
    
    if params.get('use_fpi'):
        ps += f"_fpi_gmax{params['fpi_gamma_max']}_ginit{params['fpi_gamma_init']}"
        ps += f"_qprior{params['fpi_q_prior_weight']}"
        if params.get('lambda_fpi_gamma_l1', 0) > 0:
            ps += f"_fgl1{params['lambda_fpi_gamma_l1']}"
        ps += f"_detach{params['fpi_detach_mode']}"
        ps += "_global"
        ps += f"_m{params['fpi_m_max']}_fp{params['fpi_fp_dim']}"
    
    return ps


def build_cmd(params):
    """Build command-line argument list from parameter dict."""
    cmd = ['python', 'train_mmpa_program.py', '--train']
    for key, val in params.items():
        if isinstance(val, bool):
            if val:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(val)])
    return cmd


def get_existing_log_dirs():
    """Get set of existing log directories before running experiment."""
    return set(glob.glob('logs/2026-*'))


def find_new_log_dir(existing_dirs, expected_param_str):
    """Find the newly created log directory after experiment.
    
    Validates by checking brief_logs.txt content matches expected_param_str.
    """
    interval = 2
    max_attempts = LOGDIR_MAX_WAIT // interval
    
    for _ in range(max_attempts):
        current_dirs = set(glob.glob('logs/2026-*'))
        new_dirs = current_dirs - existing_dirs
        
        for d in new_dirs:
            brief_log = os.path.join(d, 'brief_logs.txt')
            if os.path.exists(brief_log):
                with open(brief_log, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    second_line = f.readline().strip()
                    content = first_line + second_line
                    if expected_param_str in content:
                        return d
        
        time.sleep(interval)
    
    # Fallback: return any new directory if validation fails
    current_dirs = set(glob.glob('logs/2026-*'))
    new_dirs = current_dirs - existing_dirs
    if new_dirs:
        fallback = list(new_dirs)[0]
        print(f"  [WARNING] Param_str validation failed, using fallback: {os.path.basename(fallback)}")
        return fallback
    return None


def parse_auc_from_log_dir(log_dir):
    """Parse mean AUC from brief_logs.txt in log directory."""
    brief_log = os.path.join(log_dir, 'brief_logs.txt')
    if not os.path.exists(brief_log):
        return None
    
    with open(brief_log, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Match "auc：X.XXXX/X.XXXX" (full-width colon used in logs)
    match = re.search(r'auc[：:]\s*(\d+\.\d+)', content)
    if match:
        return float(match.group(1))
    return None


def backup_results(expected_param_str):
    """Backup results files that would be overwritten by same param_str."""
    results_backup_dir = 'outputs/hyperparam_tuning/results_backup'
    os.makedirs(results_backup_dir, exist_ok=True)
    
    for fold_dir in glob.glob('results/fold-*'):
        for csv_file in glob.glob(os.path.join(fold_dir, f'{expected_param_str}_mmpa.csv')):
            if os.path.exists(csv_file):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f"{os.path.basename(csv_file)}.{timestamp}"
                backup_path = os.path.join(results_backup_dir, backup_name)
                shutil.copy2(csv_file, backup_path)
                print(f"  [BACKUP] {csv_file} -> {backup_path}")


def run_single_experiment(param_name, param_value, params, section_label):
    """Run one experiment and return AUC and elapsed time."""
    expected_param_str = build_param_str(params)
    
    print(f"\n{'='*70}")
    print(f"  [{section_label}] Experiment: {param_name} = {param_value}")
    print(f"  Expected param_str: {expected_param_str[:80]}...")
    print(f"{'='*70}")
    
    existing_dirs = get_existing_log_dirs()
    cmd = build_cmd(params)
    
    print(f"  Command: python train_mmpa_program.py --train ... --{param_name} {param_value}")
    
    # Backup existing results to prevent overwrite
    backup_results(expected_param_str)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=EXPERIMENT_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Experiment timed out after {EXPERIMENT_TIMEOUT/3600:.1f} hours!")
        return None, time.time() - start_time
    
    elapsed = time.time() - start_time
    
    # Check return code
    if result.returncode != 0:
        print(f"  [ERROR] Experiment failed with return code {result.returncode}!")
        stderr_lines = result.stderr.strip().split('\n')[-20:] if result.stderr else []
        if stderr_lines:
            print("  Last stderr lines:")
            for line in stderr_lines:
                print(f"    {line}")
        return None, elapsed
    
    # Find the new log directory
    log_dir = find_new_log_dir(existing_dirs, expected_param_str)
    if not log_dir:
        print(f"  [ERROR] Could not find new log directory after {LOGDIR_MAX_WAIT}s!")
        return None, elapsed
    
    print(f"  Log dir: {os.path.basename(log_dir)}")
    
    # Parse AUC
    auc = parse_auc_from_log_dir(log_dir)
    if auc is not None:
        print(f"  Result: AUC = {auc:.4f}  (time: {elapsed/60:.1f} min)")
    else:
        print(f"  [WARNING] Could not parse AUC from log!")
    
    return auc, elapsed


def save_results(results, filepath):
    """Save results incrementally to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['section', 'param_name', 'param_value', 'auc', 'elapsed_min']
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                'section': r['section'],
                'param_name': r['param_name'],
                'param_value': r['param_value'],
                'auc': f"{r['auc']:.4f}" if r['auc'] is not None else 'N/A',
                'elapsed_min': f"{r['elapsed_min']:.1f}",
            })


def print_summary(results):
    """Print formatted summary."""
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    
    grouped = defaultdict(list)
    for r in results:
        grouped[r['param_name']].append(r)
    
    # Baseline values
    baseline = {
        'fpi_gamma_max': 0.1,
        'num_programs': 8,
        'lambda_ot': 0.005,
        'fpi_m_max': 16,
    }
    
    # Print results per parameter
    for param_name in ['fpi_gamma_max', 'num_programs', 'lambda_ot', 'fpi_m_max']:
        if param_name not in grouped:
            continue
        
        param_results = grouped[param_name]
        valid_results = [r for r in param_results if r['auc'] is not None]
        if not valid_results:
            continue
        
        best_auc = max(r['auc'] for r in valid_results)
        section_label = valid_results[0]['section']
        
        print(f"\n  --- [{section_label}] {param_name} ---")
        for r in sorted(valid_results, key=lambda x: x['param_value']):
            marker = "  <-- BEST" if r['auc'] == best_auc else ""
            baseline_marker = "  [BASELINE]" if r['param_value'] == baseline.get(param_name) else ""
            print(f"    {param_name:15s} = {str(r['param_value']):6s}  ->  AUC = {r['auc']:.4f}{marker}{baseline_marker}")
    
    # Check if baseline is still best for main-text parameters
    print("\n" + "="*70)
    print("  BASELINE VALIDATION (Main Text Only)")
    print("="*70)
    
    main_text_params = ['fpi_gamma_max', 'num_programs']
    baseline_valid = True
    
    for param_name in main_text_params:
        if param_name not in grouped:
            continue
        param_results = grouped[param_name]
        valid_results = [r for r in param_results if r['auc'] is not None]
        if not valid_results:
            continue
        
        baseline_result = next(
            (r for r in valid_results if r['param_value'] == baseline[param_name]),
            None
        )
        best_result = max(valid_results, key=lambda x: x['auc'])
        
        if baseline_result and best_result:
            if best_result['param_value'] == baseline[param_name]:
                print(f"  ✓ {param_name}: baseline={baseline[param_name]} is BEST ({baseline_result['auc']:.4f})")
            else:
                diff = best_result['auc'] - baseline_result['auc']
                print(f"  ! {param_name}: baseline={baseline[param_name]} ({baseline_result['auc']:.4f})")
                print(f"    best={best_result['param_value']} ({best_result['auc']:.4f}), diff={diff:+.4f}")
                if diff > 0.005:
                    baseline_valid = False
    
    if baseline_valid:
        print("\n  Current baseline configuration remains optimal or near-optimal.")
        print("  No ablation rerun needed.")
    else:
        print("\n  WARNING: Some main-text parameters have significantly better values.")
        print("  Consider updating baseline and rerunning relevant ablation experiments.")
    
    # Overall best
    all_valid = [r for r in results if r['auc'] is not None]
    if all_valid:
        overall_best = max(all_valid, key=lambda x: x['auc'])
        print(f"\n  Overall best single-parameter experiment:")
        print(f"    {overall_best['param_name']}={overall_best['param_value']}, AUC={overall_best['auc']:.4f}")
    print(f"\n  Current baseline: AUC=0.6220")


def main():
    """Main execution flow."""
    results = []
    output_csv = 'outputs/hyperparam_tuning/S3_S6_results.csv'
    
    total_experiments = (
        sum(len(v) for v in MAIN_TEXT_SPACES.values()) +
        sum(len(v) for v in APPENDIX_SPACES.values())
    )
    
    print("="*70)
    print("  Hyperparameter Tuning for MACB-DRP (S3-S6 Only)")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Main text experiments (S3-S4): {sum(len(v) for v in MAIN_TEXT_SPACES.values())}")
    print(f"  Appendix experiments (S5-S6):  {sum(len(v) for v in APPENDIX_SPACES.values())}")
    print(f"  Total: {total_experiments} experiments")
    print(f"  Estimated time: ~{total_experiments * 20} minutes (~{total_experiments * 20 / 60:.1f} hours)")
    print("  Note: S1 (alph) and S2 (beta) excluded — not core innovations.")
    print("="*70)
    
    # ========================================================================
    # Main Text: S3-S4
    # ========================================================================
    print("\n" + "="*70)
    print("  MAIN TEXT: S3-S4 (fpi_gamma_max, num_programs)")
    print("="*70)
    
    for param_name, values in MAIN_TEXT_SPACES.items():
        for val in values:
            params = BASE_PARAMS.copy()
            params[param_name] = val
            
            auc, elapsed = run_single_experiment(param_name, val, params, 'Main')
            results.append({
                'section': 'Main',
                'param_name': param_name,
                'param_value': val,
                'auc': auc,
                'elapsed_min': elapsed / 60,
            })
            save_results(results, output_csv)
    
    # ========================================================================
    # Appendix: S5-S6
    # ========================================================================
    print("\n" + "="*70)
    print("  APPENDIX: S5-S6 (lambda_ot, fpi_m_max)")
    print("="*70)
    
    for param_name, values in APPENDIX_SPACES.items():
        for val in values:
            params = BASE_PARAMS.copy()
            params[param_name] = val
            
            auc, elapsed = run_single_experiment(param_name, val, params, 'Appendix')
            results.append({
                'section': 'Appendix',
                'param_name': param_name,
                'param_value': val,
                'auc': auc,
                'elapsed_min': elapsed / 60,
            })
            save_results(results, output_csv)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_summary(results)
    
    print(f"\n  Results saved to: {output_csv}")
    print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Done!")


if __name__ == '__main__':
    main()
