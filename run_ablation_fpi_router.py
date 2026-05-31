#!/usr/bin/env python3
"""
Phase 9-R: DualFPIResidualRouter Ablation Runner.

Runs a sweep of router configurations to find the optimal combination
of global and cross branch gamma gates.

Key configurations to test:
1. B0 baseline (no FPI, no router)
2. Router: global only (cross_init=0, cross_max=0 or start_epoch=inf)
3. Router: global + learnable cross (gamma_global_max=0.05, gamma_cross_max=0.05)
4. Router: global + delayed cross ramp
5. Router: oracle cross mask (diagnostic upper bound)

Usage:
    python run_ablation_fpi_router.py
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


def main():
    base = [
        sys.executable, "train_mmpa_program.py",
        "--train",
        "--num_programs", "8",
        "--program_dim", "64",
        "--alph", "1.0",
        "--beta", "0.1",
        "--K", "7",
        "--use_cdan",
        "--measurement", "AUC",
        "--backbone_type", "mlp",
    ]

    experiments = []

    # ------------------------------------------------------------------
    # 0. B0 Baseline (no FPI, no router)
    # ------------------------------------------------------------------
    experiments.append((
        base + ["--exp_name", "phase9r_B0_baseline"],
        "B0 Baseline (no FPI)"
    ))

    # ------------------------------------------------------------------
    # 1. Router: Global branch only (cross disabled)
    #    Equivalent to FPI2 warm-up best config (AUC 0.6031)
    # ------------------------------------------------------------------
    experiments.append((
        base + [
            "--use_fpi_router",
            "--router_gamma_global_max", "0.05",
            "--router_gamma_global_init", "0.05",
            "--router_gamma_cross_max", "0.0",
            "--router_gamma_cross_init", "0.0",
            "--lambda_router_gamma_l1", "0.0",
            "--router_residual_global_scale", "1.0",
            "--router_residual_cross_scale", "1.0",
            "--fpi_m_max", "16",
            "--fpi_fp_dim", "1024",
            "--exp_name", "phase9r_global_only",
        ],
        "Router: Global branch only (cross disabled)"
    ))

    # ------------------------------------------------------------------
    # 2. Router: Global + Learnable Cross (both active from start)
    #    Key test: can cross branch help without hurting global?
    # ------------------------------------------------------------------
    experiments.append((
        base + [
            "--use_fpi_router",
            "--router_gamma_global_max", "0.05",
            "--router_gamma_global_init", "0.05",
            "--router_gamma_cross_max", "0.05",
            "--router_gamma_cross_init", "0.0",
            "--lambda_router_gamma_l1", "0.0",
            "--router_residual_global_scale", "1.0",
            "--router_residual_cross_scale", "1.0",
            "--fpi_m_max", "16",
            "--fpi_fp_dim", "1024",
            "--exp_name", "phase9r_global_learnable_cross",
        ],
        "Router: Global + Learnable Cross (from start)"
    ))

    # ------------------------------------------------------------------
    # 3. Router: Global + Delayed Cross Ramp
    #    Cross branch starts at epoch 10, ramps over 10 epochs
    # ------------------------------------------------------------------
    experiments.append((
        base + [
            "--use_fpi_router",
            "--router_gamma_global_max", "0.05",
            "--router_gamma_global_init", "0.05",
            "--router_gamma_cross_max", "0.05",
            "--router_gamma_cross_init", "0.0",
            "--lambda_router_gamma_l1", "0.0",
            "--router_cross_start_epoch", "10",
            "--router_cross_ramp_epochs", "10",
            "--router_residual_global_scale", "1.0",
            "--router_residual_cross_scale", "1.0",
            "--fpi_m_max", "16",
            "--fpi_fp_dim", "1024",
            "--exp_name", "phase9r_global_cross_ramp10",
        ],
        "Router: Global + Delayed Cross Ramp (start=10, ramp=10)"
    ))

    # ------------------------------------------------------------------
    # 4. Router: Oracle Cross Mask (diagnostic only)
    #    Force cross branch only for Temozolomide
    # ------------------------------------------------------------------
    experiments.append((
        base + [
            "--use_fpi_router",
            "--router_gamma_global_max", "0.05",
            "--router_gamma_global_init", "0.05",
            "--router_gamma_cross_max", "0.05",
            "--router_gamma_cross_init", "0.0",
            "--lambda_router_gamma_l1", "0.0",
            "--router_cross_mode", "oracle_mask",
            "--router_oracle_cross_drugs", "Temozolomide",
            "--router_residual_global_scale", "1.0",
            "--router_residual_cross_scale", "1.0",
            "--fpi_m_max", "16",
            "--fpi_fp_dim", "1024",
            "--exp_name", "phase9r_oracle_cross_temozolomide",
        ],
        "Router: Oracle Cross Mask for Temozolomide (diagnostic)"
    ))

    # ------------------------------------------------------------------
    # 5. Router: Global + Learnable Cross with L1 regularization
    #    Encourage cross branch to stay at 0 unless needed
    # ------------------------------------------------------------------
    experiments.append((
        base + [
            "--use_fpi_router",
            "--router_gamma_global_max", "0.05",
            "--router_gamma_global_init", "0.05",
            "--router_gamma_cross_max", "0.05",
            "--router_gamma_cross_init", "0.0",
            "--lambda_router_gamma_l1", "0.001",
            "--router_residual_global_scale", "1.0",
            "--router_residual_cross_scale", "1.0",
            "--fpi_m_max", "16",
            "--fpi_fp_dim", "1024",
            "--exp_name", "phase9r_global_cross_l1",
        ],
        "Router: Global + Learnable Cross with L1=0.001"
    ))

    # ------------------------------------------------------------------
    # 6. Router: Lower global max, higher cross max
    #    Test if a weaker global allows cross to shine
    # ------------------------------------------------------------------
    experiments.append((
        base + [
            "--use_fpi_router",
            "--router_gamma_global_max", "0.03",
            "--router_gamma_global_init", "0.03",
            "--router_gamma_cross_max", "0.08",
            "--router_gamma_cross_init", "0.0",
            "--lambda_router_gamma_l1", "0.0",
            "--router_residual_global_scale", "1.0",
            "--router_residual_cross_scale", "1.0",
            "--fpi_m_max", "16",
            "--fpi_fp_dim", "1024",
            "--exp_name", "phase9r_lowglobal_highcross",
        ],
        "Router: Lower global (0.03) + Higher cross (0.08)"
    ))

    # Run all experiments
    for cmd, desc in experiments:
        run_cmd(cmd, desc)

    print("\n" + "="*60)
    print("All Phase 9-R ablation experiments completed.")
    print("="*60)


if __name__ == "__main__":
    main()
