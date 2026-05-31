#!/usr/bin/env python3
"""
Phase 8-F Ablation: BPAF vs LLRF vs Baseline
==============================================
Runs a grid of BPAF and LLRF configurations to evaluate fusion repair strategies.

Usage:
    python run_phase8f_ablation.py [--gpu 0] [--dry_run]

Experiments:
    1. baseline (no BPAF, no LLRF)
    2. bpaf_default (BPAF with default gamma_max=0.05)
    3. bpaf_strong (BPAF with gamma_max=0.1)
    4. bpaf_weak (BPAF with gamma_max=0.01)
    5. bpaf_regularized (BPAF + attn_entropy + attn_diversity + gamma_l1)
    6. llrf_default (LLRF with default gamma_max=0.1)
    7. llrf_strong (LLRF with gamma_max=0.2)
    8. llrf_weak (LLRF with gamma_max=0.05)
    9. llrf_regularized (LLRF + attn_entropy + attn_diversity + gamma_l1)
"""

import subprocess
import sys
import os
import time
import argparse

BASE_CMD = [
    "python", "train_mmpa_program.py",
    "--alph", "0.3",
    "--beta", "0.3",
    "--num_programs", "8",
    "--program_dim", "64",
    "--lambda_div", "0.01",
    "--lambda_sparse", "0.001",
    "--K", "7",
]

EXPERIMENTS = {
    "baseline": {
        "desc": "Baseline (no BPAF, no LLRF)",
        "args": [],
    },
    "bpaf_default": {
        "desc": "BPAF default (gamma_max=0.05)",
        "args": ["--use_bpaf"],
    },
    "bpaf_strong": {
        "desc": "BPAF strong (gamma_max=0.1)",
        "args": ["--use_bpaf", "--bpaf_gamma_max", "0.1"],
    },
    "bpaf_weak": {
        "desc": "BPAF weak (gamma_max=0.01)",
        "args": ["--use_bpaf", "--bpaf_gamma_max", "0.01"],
    },
    "bpaf_regularized": {
        "desc": "BPAF + attn_entropy=0.01 + attn_diversity=0.01 + gamma_l1=0.001",
        "args": [
            "--use_bpaf",
            "--lambda_attn_entropy", "0.01",
            "--lambda_attn_diversity", "0.01",
            "--lambda_gamma_l1", "0.001",
        ],
    },
    "llrf_default": {
        "desc": "LLRF default (gamma_max=0.1)",
        "args": ["--use_llrf"],
    },
    "llrf_strong": {
        "desc": "LLRF strong (gamma_max=0.2)",
        "args": ["--use_llrf", "--llrf_gamma_max", "0.2"],
    },
    "llrf_weak": {
        "desc": "LLRF weak (gamma_max=0.05)",
        "args": ["--use_llrf", "--llrf_gamma_max", "0.05"],
    },
    "llrf_regularized": {
        "desc": "LLRF + attn_entropy=0.01 + attn_diversity=0.01 + gamma_l1=0.001",
        "args": [
            "--use_llrf",
            "--lambda_attn_entropy", "0.01",
            "--lambda_attn_diversity", "0.01",
            "--lambda_gamma_l1", "0.001",
        ],
    },
}


def main():
    parser = argparse.ArgumentParser("Phase 8-F Ablation")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--experiments", type=str, nargs="+", default=None,
                        help="Specific experiments to run (default: all)")
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiments_to_run = args.experiments if args.experiments else list(EXPERIMENTS.keys())

    for exp_name in experiments_to_run:
        if exp_name not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_name}. Available: {list(EXPERIMENTS.keys())}")
            continue

        exp = EXPERIMENTS[exp_name]
        cmd = BASE_CMD + exp["args"]
        log_file = f"phase8f_{exp_name}.log"

        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"Description: {exp['desc']}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Log file: {log_file}")
        print(f"{'='*60}\n")

        if args.dry_run:
            continue

        with open(log_file, "w") as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Description: {exp['desc']}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            f.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                universal_newlines=True,
                bufsize=1,
            )

            for line in process.stdout:
                f.write(line)
                f.flush()
                print(line, end="")

            process.wait()
            f.write(f"\nExit code: {process.returncode}\n")
            f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"\nExperiment {exp_name} finished with exit code {process.returncode}")

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
