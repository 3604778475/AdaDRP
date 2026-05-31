#!/usr/bin/env python
"""Wrapper to run Phase 14-A experiments"""
import subprocess
import sys

def run_exp(lambda_rank):
    cmd = [
        sys.executable, "train_mmpa_program.py",
        "--measurement", "Z_SCORE",
        "--thres_gdsc", "0.0",
        "--thres_label", "0.1",
        "--n", "5",
        "--alph", "0.3",
        "--beta", "0.3",
        "--train",
        "--num_programs", "8",
        "--program_dim", "64",
        "--lambda_div", "0.01",
        "--lambda_sparse", "0.001",
        "--use_ot_alignment",
        "--lambda_ot", "0.005",
        "--ot_reg", "0.1",
        "--ot_cost_type", "cosine",
        "--ot_marginal_type", "uniform",
        "--backbone_type", "convnext1d",
        "--convnext_depth", "1",
        "--convnext_kernel_size", "7",
        "--K", "3",
        "--use_fpi",
        "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
        "--fpi_gamma_max", "0.05",
        "--fpi_gamma_init", "0.05",
        "--fpi_q_prior_weight", "0.01",
        "--lambda_fpi_gamma_l1", "0.0",
        "--use_rank_loss",
        "--lambda_rank", str(lambda_rank),
        "--rank_margin", "0.0",
        "--rank_max_pairs_per_drug", "0",
        "--pretrain_num_epochs", "10",
        "--train_num_epochs", "10",
        "--contrastive_num_epochs", "10",
    ]
    print(f"\n{'='*60}")
    print(f"Running Rank-{lambda_rank}")
    print(f"{'='*60}\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    for lr in [0.005, 0.01, 0.05]:
        run_exp(lr)
