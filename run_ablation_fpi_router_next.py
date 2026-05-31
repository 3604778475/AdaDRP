#!/usr/bin/env python3
"""
Phase 9-FPI-Router Next Ablation Runner.

Runs the full experiment suite for DualFPIResidualRouter with cross branch.

Experiments:
1. B0_current
2. FPI_global_warmup
3. Router_global_only_confirm
4. Router_oracle_temo_g005
5. Router_oracle_temo_g0025
6. Router_learnable_l1_005
7. Router_learnable_l1_01
8. Router_learnable_g0025_l1_005
9. Router_learnable_delayed

Priority batch 1: 1-6
Priority batch 2: 7-9 (only if oracle_temo shows promise)

Usage:
    python run_ablation_fpi_router_next.py
"""

import subprocess
import sys
import os
import csv
import json


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


def fpi_global_warmup():
    return BASE + [
        "--use_fpi",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--fpi_gamma_max", "0.1",
        "--fpi_gamma_init", "0.05",
        "--fpi_q_prior_weight", "0.01",
        "--lambda_fpi_gamma_l1", "0.0",
        ], "FPI Global Warm-up (gamma_max=0.1, gamma_init=0.05, no L1) - Best known config"


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


def router_oracle_temo_g005():
    return BASE + [
        "--use_fpi_router",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--router_gamma_global_max", "0.05",
        "--router_gamma_global_init", "0.05",
        "--router_gamma_cross_max", "0.05",
        "--router_gamma_cross_init", "0.0",
        "--router_cross_mode", "oracle_mask",
        "--router_oracle_cross_drugs", "Temozolomide",
        "--lambda_router_gamma_global_l1", "0.0",
        "--lambda_router_gamma_cross_l1", "0.0",
        ], "Router Oracle Temozolomide g005 (diagnostic upper bound)"


def router_oracle_temo_g0025():
    return BASE + [
        "--use_fpi_router",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--router_gamma_global_max", "0.05",
        "--router_gamma_global_init", "0.05",
        "--router_gamma_cross_max", "0.025",
        "--router_gamma_cross_init", "0.0",
        "--router_cross_mode", "oracle_mask",
        "--router_oracle_cross_drugs", "Temozolomide",
        "--lambda_router_gamma_global_l1", "0.0",
        "--lambda_router_gamma_cross_l1", "0.0",
        ], "Router Oracle Temozolomide g0025 (diagnostic upper bound)"


def router_learnable_l1_005():
    return BASE + [
        "--use_fpi_router",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--router_gamma_global_max", "0.05",
        "--router_gamma_global_init", "0.05",
        "--router_gamma_cross_max", "0.05",
        "--router_gamma_cross_init", "0.0",
        "--router_cross_mode", "learnable",
        "--lambda_router_gamma_global_l1", "0.0",
        "--lambda_router_gamma_cross_l1", "0.005",
        ], "Router Learnable L1=0.005 (cross L1 sparse)"


def router_learnable_l1_01():
    return BASE + [
        "--use_fpi_router",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--router_gamma_global_max", "0.05",
        "--router_gamma_global_init", "0.05",
        "--router_gamma_cross_max", "0.05",
        "--router_gamma_cross_init", "0.0",
        "--router_cross_mode", "learnable",
        "--lambda_router_gamma_global_l1", "0.0",
        "--lambda_router_gamma_cross_l1", "0.01",
        ], "Router Learnable L1=0.01 (stronger cross sparse)"


def router_learnable_g0025_l1_005():
    return BASE + [
        "--use_fpi_router",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--router_gamma_global_max", "0.05",
        "--router_gamma_global_init", "0.05",
        "--router_gamma_cross_max", "0.025",
        "--router_gamma_cross_init", "0.0",
        "--router_cross_mode", "learnable",
        "--lambda_router_gamma_global_l1", "0.0",
        "--lambda_router_gamma_cross_l1", "0.005",
        ], "Router Learnable g0025 L1=0.005 (weaker cross amplitude)"


def router_learnable_delayed():
    return BASE + [
        "--use_fpi_router",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--router_gamma_global_max", "0.05",
        "--router_gamma_global_init", "0.05",
        "--router_gamma_cross_max", "0.05",
        "--router_gamma_cross_init", "0.0",
        "--router_cross_mode", "learnable",
        "--router_cross_start_epoch", "30",
        "--router_cross_ramp_epochs", "20",
        "--lambda_router_gamma_global_l1", "0.0",
        "--lambda_router_gamma_cross_l1", "0.005",
        ], "Router Learnable Delayed (cross starts at epoch 30, ramps 20 epochs)"


EXPERIMENTS = [
    (b0_current, "batch1"),
    (fpi_global_warmup, "batch1"),
    (router_global_only_confirm, "batch1"),
    (router_oracle_temo_g005, "batch1"),
    (router_oracle_temo_g0025, "batch1"),
    (router_learnable_l1_005, "batch1"),
    (router_learnable_l1_01, "batch2"),
    (router_learnable_g0025_l1_005, "batch2"),
    (router_learnable_delayed, "batch2"),
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, default="batch1", choices=["batch1", "batch2", "all"])
    args = parser.parse_args()

    for exp_fn, batch in EXPERIMENTS:
        if args.batch != "all" and batch != args.batch:
            continue
        cmd, desc = exp_fn()
        run_cmd(cmd, desc)

    print("\n" + "="*60)
    print("Phase 9-FPI-Router Next ablation completed.")
    print("="*60)


if __name__ == "__main__":
    main()
