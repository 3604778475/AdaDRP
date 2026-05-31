#!/usr/bin/env python
"""
Phase 8-F Fusion-Repair Ablation Script
=======================================
Runs BPAF and LLRF ablation experiments against B0 baseline.

Usage:
    python run_ablation_fusion_repair.py
"""

import subprocess
import sys
import time

# Base config shared by all experiments
BASE_CMD = [
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
    "--topk_contrastive", "3",
    "--lambda_ot", "0.005",
]

EXPERIMENTS = [
    {
        "name": "B0_current",
        "extra_args": [],
    },
    {
        "name": "BPAF0_sanity",
        "extra_args": [
            "--use_bpaf",
            "--bpaf_gamma_max", "0.0",
            "--bpaf_gamma_init", "0.0",
        ],
    },
    {
        "name": "BPAF1_g005",
        "extra_args": [
            "--use_bpaf",
            "--bpaf_gamma_max", "0.05",
            "--bpaf_gamma_init", "0.02",
            "--lambda_fusion_gamma_l1", "0.001",
        ],
    },
    {
        "name": "BPAF2_g01",
        "extra_args": [
            "--use_bpaf",
            "--bpaf_gamma_max", "0.1",
            "--bpaf_gamma_init", "0.02",
            "--lambda_fusion_gamma_l1", "0.001",
        ],
    },
    {
        "name": "LLRF1_g01",
        "extra_args": [
            "--use_llrf",
            "--llrf_gamma_max", "0.1",
            "--llrf_gamma_init", "0.02",
            "--lambda_fusion_gamma_l1", "0.001",
        ],
    },
]


def run_experiment(name, extra_args):
    cmd = BASE_CMD + extra_args
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    print("Command:", " ".join(cmd))
    start = time.time()
    result = subprocess.run(cmd, cwd="/home/waas/MACB-DRP-main")
    elapsed = time.time() - start
    print(f"Completed {name} in {elapsed/60:.1f} min (exit={result.returncode})")
    return result.returncode


def main():
    results = []
    for exp in EXPERIMENTS:
        code = run_experiment(exp["name"], exp["extra_args"])
        results.append((exp["name"], code))

    print(f"\n{'='*60}")
    print("ABLAUTION SUMMARY")
    print(f"{'='*60}")
    for name, code in results:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"  {name:20s}: {status}")


if __name__ == "__main__":
    main()
