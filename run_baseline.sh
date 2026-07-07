#!/bin/bash
# Baseline: 100/100/100, domain_sep + bilinear + FPI

cd /d/药物反应预测/MACB-DRP-main
source /d/anaconda/etc/profile.d/conda.sh
conda activate macb_drp

python train_mmpa_program.py \
  --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 \
  --train --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
  --num_programs 8 --program_dim 64 --K 3 \
  --use_ot_alignment --lambda_ot 0.005 --ot_reg 0.1 --ot_cost_type cosine --ot_marginal_type uniform \
  --use_cdan \
  --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64 \
  --lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005 --lambda_migrate 0.01 \
  --alph 1.4 --beta 0.066 \
  --use_fpi --fpi_gamma_max 0.1 --fpi_gamma_init 0.05 --fpi_q_prior_weight 0.01 --fpi_dropout 0.1 \
  --fpi_m_max 16 --fpi_fp_dim 1024 --lambda_fpi_gamma_l1 0.001 --fpi_detach_mode none \
  --pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100
