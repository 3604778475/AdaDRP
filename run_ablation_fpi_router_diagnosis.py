#!/usr/bin/env python3
"""
Phase 9-FPI-Router Diagnosis Ablation Runner.

Runs controlled experiments to diagnose why cross branch hurts global AUC.

Experiments:
1. B0_current (baseline)
2. Router_global_only_confirm (verified good)
3. Router_oracle_temo_g005_match_global (strict global config match)
4. Router_oracle_temo_g005_match_global_cross_detach_zq_drug
5. Router_oracle_temo_g0025_match_global
6. Router_two_stage_oracle_temo_g005

Usage:
    python run_ablation_fpi_router_diagnosis.py
"""

import subprocess
import sys
import os


def run_cmd(cmd_list, desc=""):
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {' '.join(cmd_list)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd_list, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"WARNING: Command failed with return code {result.returncode}")
    return result.returncode


BASE = [
    sys.executable, "train_mmpa_program.py",
    "--measurement", "Z_SCORE",
    "--thres_gdsc", "0.0",
    "--thres_label", "0.1",
    "--n", "5",
    "--alph", "0.3",
    "--beta", "0.3",
    "--train",
    "--backbone_type", "convnext1d",
    "--convnext_depth", "1",
    "--convnext_kernel_size", "7",
    "--num_programs", "8",
    "--program_dim", "64",
    "--K", "3",
    "--lambda_ot", "0.005",
    "--use_cdan",
    "--use_ot_alignment",
]


def b0_current():
    return BASE + [], "B0 Current (no FPI, no router)"


def router_global_only_confirm():
    return BASE + [
        "--use_fpi_router",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--router_gamma_global_max", "0.1",
        "--router_gamma_global_init", "0.05",
        "--router_gamma_cross_max", "0.0",
        "--router_gamma_cross_init", "0.0",
        "--lambda_router_gamma_global_l1", "0.0",
        "--lambda_router_gamma_cross_l1", "0.0",
    ], "Router Global-only Confirm (cross disabled, matches FPI2_warmup)"


def router_oracle_temo_g005_match_global():
    return BASE + [
        "--use_fpi_router",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--router_gamma_global_max", "0.1",
        "--router_gamma_global_init", "0.05",
        "--router_gamma_cross_max", "0.05",
        "--router_gamma_cross_init", "0.0",
        "--router_cross_mode", "oracle_mask",
        "--router_oracle_cross_drugs", "Temozolomide",
        "--lambda_router_gamma_global_l1", "0.0",
        "--lambda_router_gamma_cross_l1", "0.0",
    ], "Router Oracle Temo g005 match global (ggmax=0.1, strict control)"


def router_oracle_temo_g005_match_global_cross_detach_zq_drug():
    return BASE + [
        "--use_fpi_router",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--router_gamma_global_max", "0.1",
        "--router_gamma_global_init", "0.05",
        "--router_gamma_cross_max", "0.05",
        "--router_gamma_cross_init", "0.0",
        "--router_cross_mode", "oracle_mask",
        "--router_oracle_cross_drugs", "Temozolomide",
        "--router_cross_detach_mode", "zq_drug",
        "--lambda_router_gamma_global_l1", "0.0",
        "--lambda_router_gamma_cross_l1", "0.0",
    ], "Router Oracle Temo g005 match global + cross detach zq_drug"


def router_oracle_temo_g0025_match_global():
    return BASE + [
        "--use_fpi_router",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--router_gamma_global_max", "0.1",
        "--router_gamma_global_init", "0.05",
        "--router_gamma_cross_max", "0.025",
        "--router_gamma_cross_init", "0.0",
        "--router_cross_mode", "oracle_mask",
        "--router_oracle_cross_drugs", "Temozolomide",
        "--lambda_router_gamma_global_l1", "0.0",
        "--lambda_router_gamma_cross_l1", "0.0",
    ], "Router Oracle Temo g0025 match global (weaker cross)"


def router_two_stage_oracle_temo_g005():
    return BASE + [
        "--use_fpi_router",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--router_gamma_global_max", "0.1",
        "--router_gamma_global_init", "0.05",
        "--router_gamma_cross_max", "0.05",
        "--router_gamma_cross_init", "0.0",
        "--router_cross_mode", "oracle_mask",
        "--router_oracle_cross_drugs", "Temozolomide",
        "--router_two_stage_cross_only",
        "--router_freeze_main_for_cross",
        "--router_stage_a_checkpoint_dir", "checkpoints/router_global_only",
        "--lambda_router_gamma_global_l1", "0.0",
        "--lambda_router_gamma_cross_l1", "0.0",
    ], "Router Two-stage Oracle Temo g005 (freeze main, train cross only)"


EXPERIMENTS = [
    b0_current,
    router_global_only_confirm,
    router_oracle_temo_g005_match_global,
    router_oracle_temo_g005_match_global_cross_detach_zq_drug,
    router_oracle_temo_g0025_match_global,
    router_two_stage_oracle_temo_g005,
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, default="all", choices=["batch1", "all"])
    args = parser.parse_args()

    # batch1: first 3 experiments (controls)
    # batch2: remaining experiments (only run if batch1 shows promise or for diagnosis)
    for idx, exp_fn in enumerate(EXPERIMENTS):
        if args.batch == "batch1" and idx >= 3:
            continue
        cmd, desc = exp_fn()
        run_cmd(cmd, desc)

    print("\n" + "="*60)
    print("Phase 9-FPI-Router Diagnosis ablation completed.")
    print("="*60)


if __name__ == "__main__":
    main()
