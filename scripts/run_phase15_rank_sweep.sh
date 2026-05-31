#!/usr/bin/env bash
# Phase 15-B: Effective Drug-wise Ranking Loss Sweep
# Runs MACB + Global FPI baseline and multiple lambda_rank values.
# Baseline config reproduces ~0.6031 AUC.
#
# Usage:
#   bash scripts/run_phase15_rank_sweep.sh
#
# Each experiment writes to outputs/phase15/rank_lam<XXX>/

set -e

# Baseline config that reproduces 0.6031 AUC
PRETRAIN_EPOCHS=300
TRAIN_EPOCHS=100
CONTRASTIVE_EPOCHS=200
K=3
BACKBONE="convnext1d"

# Base flags for MACB + Global FPI + CDAN + OT (0.6031 baseline)
BASE_FLAGS="--use_fpi --use_cdan --use_ot_alignment --backbone_type ${BACKBONE} --convnext_depth 1 --convnext_kernel_size 7 --num_programs 8 --program_dim 64 --pretrain_num_epochs ${PRETRAIN_EPOCHS} --train_num_epochs ${TRAIN_EPOCHS} --contrastive_num_epochs ${CONTRASTIVE_EPOCHS} --K ${K} --lambda_ot 0.005 --fpi_gamma_max 0.1 --fpi_gamma_init 0.05 --fpi_q_prior_weight 0.01 --lambda_fpi_gamma_l1 0.0"

echo "========================================"
echo "Phase 15-B: Rank Loss Lambda Sweep"
echo "========================================"

# 1. Baseline (no rank loss code path)
echo "[1/7] Running BASELINE (no rank loss)..."
python train_mmpa_program.py ${BASE_FLAGS} \
    --output_dir outputs/phase15/baseline

# 2. Control run: rank loss enabled but lambda=0
echo "[2/7] Running CONTROL (rank enabled, lambda=0)..."
python train_mmpa_program.py ${BASE_FLAGS} \
    --use_rank_loss --lambda_rank 0.0 \
    --ranking_log_grad_ratio --ranking_log_drugwise_pair_acc \
    --output_dir outputs/phase15/rank_lam0

# 3. lambda_rank = 0.05
echo "[3/7] Running lambda_rank=0.05..."
python train_mmpa_program.py ${BASE_FLAGS} \
    --use_rank_loss --lambda_rank 0.05 --ranking_warmup_epochs 20 \
    --ranking_log_grad_ratio --ranking_log_drugwise_pair_acc \
    --output_dir outputs/phase15/rank_lam005

# 4. lambda_rank = 0.10
echo "[4/7] Running lambda_rank=0.10..."
python train_mmpa_program.py ${BASE_FLAGS} \
    --use_rank_loss --lambda_rank 0.10 --ranking_warmup_epochs 20 \
    --ranking_log_grad_ratio --ranking_log_drugwise_pair_acc \
    --output_dir outputs/phase15/rank_lam010

# 5. lambda_rank = 0.20
echo "[5/7] Running lambda_rank=0.20..."
python train_mmpa_program.py ${BASE_FLAGS} \
    --use_rank_loss --lambda_rank 0.20 --ranking_warmup_epochs 20 \
    --ranking_log_grad_ratio --ranking_log_drugwise_pair_acc \
    --output_dir outputs/phase15/rank_lam020

# 6. lambda_rank = 0.30
echo "[6/7] Running lambda_rank=0.30..."
python train_mmpa_program.py ${BASE_FLAGS} \
    --use_rank_loss --lambda_rank 0.30 --ranking_warmup_epochs 20 \
    --ranking_log_grad_ratio --ranking_log_drugwise_pair_acc \
    --output_dir outputs/phase15/rank_lam030

# 7. lambda_rank = 0.50
echo "[7/7] Running lambda_rank=0.50..."
python train_mmpa_program.py ${BASE_FLAGS} \
    --use_rank_loss --lambda_rank 0.50 --ranking_warmup_epochs 20 \
    --ranking_log_grad_ratio --ranking_log_drugwise_pair_acc \
    --output_dir outputs/phase15/rank_lam050

echo "========================================"
echo "Sweep complete. Collecting results..."
echo "========================================"

python tools/collect_phase15_rank_results.py \
    --results_dir results \
    --output_dir outputs/phase15 \
    --output_csv outputs/phase15/rank_sweep_summary.csv

echo "Summary saved to outputs/phase15/rank_sweep_summary.csv"
