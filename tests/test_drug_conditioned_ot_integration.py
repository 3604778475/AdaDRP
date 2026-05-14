"""
Quick integration test: run 1 epoch of each stage with DCOT enabled.
This verifies the end-to-end pipeline without taking too long.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

from train_mmpa_program import main as program_main

PARAMS = {
    "pretrain_num_epochs": 1,
    "train_num_epochs": 1,
    "contrastive_num_epochs": 1,
    "K": 3,
}

args = argparse.Namespace(
    measurement="Z_SCORE",
    thres_g=0.0,
    thres_s=0.1,
    n=5,
    alph=0.3,
    beta=0.3,
    retrain_flag=True,
    norm_flag=True,
    metric="auroc",
    num_programs=8,
    program_dim=64,
    lambda_div=0.01,
    lambda_sparse=0.001,
    use_drug_gate=False,
    shared_gate=False,
    gate_mode="additive_bias",
    gate_temperature=5.0,
    rho_init=0.0,
    beta_init=0.0,
    lambda_gate_entropy_floor=0.0,
    gate_min_entropy=1.2,
    lambda_gate_balance=0.0,
    lambda_gate_diversity=0.0,
    use_ot_alignment=True,
    lambda_ot=0.005,
    ot_reg=0.1,
    ot_max_iter=50,
    ot_cost_type="cosine",
    ot_marginal_type="uniform",
    use_cdan=True,
    ot_mode="global",
    ot_use_drug_conditioned_feature=False,
    ot_per_drug_lambda=None,
    use_program_ot=False,
    program_ot_mode="same_program_uniform",
    lambda_program_ot=0.01,
    program_ot_reg=0.1,
    program_ot_max_iter=50,
    program_ot_cost_type="cosine",
    program_ot_marginal_type="uniform",
    program_ot_detach_q_marginal=True,
    program_ot_q_temperature=1.0,
    program_ot_q_power=1.0,
    program_ot_min_mass=1e-6,
    q_align_enabled=False,
    lambda_q_align=0.001,
    enable_grad_diagnostics=False,
    grad_diag_every=50,
    backbone_type="convnext1d",
    norm_type="layernorm",
    num_gene_tokens=64,
    token_dim=64,
    convnext_depth=1,
    convnext_kernel_size=7,
    convnext_expansion=4,
    K=3,
    # Phase 7
    drug_conditioned_ot_enabled=True,
    drug_conditioned_ot_mode="discrepancy",
    drug_conditioned_ot_replace_sample_ot=False,
    drug_conditioned_ot_global_lambda=0.005,
    drug_conditioned_ot_reg=0.1,
    drug_conditioned_ot_cost="cosine",
    drug_conditioned_ot_lambda_min=0.0,
    drug_conditioned_ot_lambda_max=0.01,
    drug_conditioned_ot_temperature=5.0,
    drug_conditioned_ot_threshold="mean",
    drug_conditioned_ot_detach_gap=True,
    drug_conditioned_ot_normalize_gap=True,
    drug_conditioned_ot_ema_gap=True,
    drug_conditioned_ot_ema_momentum=0.9,
    drug_conditioned_ot_fixed_lambda=None,
)

if __name__ == "__main__":
    now_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + "_dcot_integration_test"
    print("Starting integration test with DCOT enabled...")
    try:
        program_main(args, PARAMS, now_time)
        print("\nIntegration test PASSED: training completed without errors.")
    except Exception as e:
        print(f"\nIntegration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
