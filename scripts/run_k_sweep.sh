#!/bin/bash
# =============================================================================
# Hyperparameter sweep for number of program tokens (K = --num_programs)
# =============================================================================
# Usage:
#   bash scripts/run_k_sweep.sh
#
# This script runs the baseline MMPA-DRP configuration with different values
# of K (num_programs) and saves each experiment to a separate output directory.
#
# Fixed parameters are taken from the current best configuration (AUC ~0.6351).
# Only --num_programs is varied.
# =============================================================================

set -e

# Values of K to test
K_VALUES=(4 6 8 12 16)

# Common baseline configuration
COMMON_FLAGS="\
--measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 --train \
--backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
--program_dim 64 --K 3 \
--use_ot_alignment --lambda_ot 0.005 --ot_reg 0.1 \
--ot_cost_type cosine --ot_marginal_type uniform \
--use_cdan \
--use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64 \
--lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005 --lambda_migrate 0.01 \
--alph 1.4 --beta 0.066 \
--use_fpi --fpi_gamma_max 0.1 --fpi_gamma_init 0.05 --fpi_q_prior_weight 0.01 \
--fpi_dropout 0.1 --fpi_m_max 16 --fpi_fp_dim 1024 --lambda_fpi_gamma_l1 0.001 \
--fpi_detach_mode none \
--pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100"

BASE_OUTPUT_DIR="outputs/k_sweep"
mkdir -p ${BASE_OUTPUT_DIR}

echo "========================================"
echo "K-Sweep: testing K in ${K_VALUES[@]}"
echo "========================================"

for K in "${K_VALUES[@]}"; do
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running K=${K} ..."
    echo ""

    python train_mmpa_program.py ${COMMON_FLAGS} \
        --num_programs ${K} \
        --output_dir ${BASE_OUTPUT_DIR}/k${K}

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] K=${K} completed."
done

echo ""
echo "========================================"
echo "K-Sweep complete. Results saved to:"
echo "  ${BASE_OUTPUT_DIR}/"
echo ""
echo "Subdirectories:"
for K in "${K_VALUES[@]}"; do
    echo "  ${BASE_OUTPUT_DIR}/k${K}/"
done
echo "========================================"
