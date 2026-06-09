#!/bin/bash
# Ablation Study Script for MACB-DRP
# 8 experiments: Original MACB, Current Best (0.6188), 6 Ablations
# Run on Linux server (waas) with CUDA

set -e

# Create log directory
ABLA_LOG_DIR="ablation_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ABLA_LOG_DIR"

echo "========================================"
echo "Ablation Study Started at $(date)"
echo "Logs saved to: $ABLA_LOG_DIR"
echo "========================================"

# ============================================================
# Experiment 1: Original MACB Baseline (original epochs: 300/100/200)
# ============================================================
echo ""
echo "[1/8] Running Original MACB Baseline (300/100/200 epochs)..."
python main.py \
    --train \
    --measurement Z_SCORE \
    --thres_gdsc 0.0 \
    --thres_label 0.1 \
    --n 5 \
    --alph 0.3 \
    --beta 0.3 \
    > "$ABLA_LOG_DIR/exp01_original_macb.log" 2>&1

echo "[1/8] Original MACB completed at $(date)"

# ============================================================
# Experiment 2: Current Best (AUC 0.6188)
# ============================================================
echo ""
echo "[2/8] Running Current Best (AUC 0.6188)..."
python train_mmpa_program.py \
    --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 --train \
    --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
    --num_programs 8 --program_dim 64 --K 3 \
    --use_ot_alignment --lambda_ot 0.005 --ot_reg 0.1 \
    --ot_cost_type cosine --ot_marginal_type uniform \
    --use_cdan \
    --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64 \
    --lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005 --lambda_migrate 0.01 \
    --alph 1.4 --beta 0.1 \
    --use_fpi --fpi_gamma_max 0.1 --fpi_gamma_init 0.05 --fpi_q_prior_weight 0.01 \
    --fpi_dropout 0.1 --fpi_m_max 16 --fpi_fp_dim 1024 --lambda_fpi_gamma_l1 0.001 \
    --fpi_detach_mode none \
    --pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100 \
    > "$ABLA_LOG_DIR/exp02_current_best.log" 2>&1

echo "[2/8] Current Best completed at $(date)"

# ============================================================
# Experiment 3: Ablation - Remove FPI
# ============================================================
echo ""
echo "[3/8] Ablation: Remove FPI..."
python train_mmpa_program.py \
    --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 --train \
    --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
    --num_programs 8 --program_dim 64 --K 3 \
    --use_ot_alignment --lambda_ot 0.005 --ot_reg 0.1 \
    --ot_cost_type cosine --ot_marginal_type uniform \
    --use_cdan \
    --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64 \
    --lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005 --lambda_migrate 0.01 \
    --alph 1.4 --beta 0.1 \
    --pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100 \
    > "$ABLA_LOG_DIR/exp03_no_fpi.log" 2>&1

echo "[3/8] No FPI completed at $(date)"

# ============================================================
# Experiment 4: Ablation - Remove DomainSep
# ============================================================
echo ""
echo "[4/8] Ablation: Remove DomainSep..."
python train_mmpa_program.py \
    --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 --train \
    --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
    --num_programs 8 --program_dim 64 --K 3 \
    --use_ot_alignment --lambda_ot 0.005 --ot_reg 0.1 \
    --ot_cost_type cosine --ot_marginal_type uniform \
    --use_cdan \
    --alph 1.4 --beta 0.1 \
    --use_fpi --fpi_gamma_max 0.1 --fpi_gamma_init 0.05 --fpi_q_prior_weight 0.01 \
    --fpi_dropout 0.1 --fpi_m_max 16 --fpi_fp_dim 1024 --lambda_fpi_gamma_l1 0.001 \
    --fpi_detach_mode none \
    --pretrain_num_epochs 300 --train_num_epochs 100 --contrastive_num_epochs 200 \
    > "$ABLA_LOG_DIR/exp04_no_domain_sep.log" 2>&1

echo "[4/8] No DomainSep completed at $(date)"

# ============================================================
# Experiment 5: Ablation - Remove OT Alignment
# ============================================================
echo ""
echo "[5/8] Ablation: Remove OT Alignment..."
python train_mmpa_program.py \
    --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 --train \
    --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
    --num_programs 8 --program_dim 64 --K 3 \
    --use_cdan \
    --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64 \
    --lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005 --lambda_migrate 0.01 \
    --alph 1.4 --beta 0.1 \
    --use_fpi --fpi_gamma_max 0.1 --fpi_gamma_init 0.05 --fpi_q_prior_weight 0.01 \
    --fpi_dropout 0.1 --fpi_m_max 16 --fpi_fp_dim 1024 --lambda_fpi_gamma_l1 0.001 \
    --fpi_detach_mode none \
    --pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100 \
    > "$ABLA_LOG_DIR/exp05_no_ot.log" 2>&1

echo "[5/8] No OT completed at $(date)"

# ============================================================
# Experiment 6: Ablation - Replace ConvNeXt1D with original MACB ResNet backbone
# ============================================================
echo ""
echo "[6/8] Ablation: Replace ConvNeXt1D with original MACB ResNet backbone..."
python train_mmpa_program.py \
    --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 --train \
    --backbone_type resnet \
    --num_programs 8 --program_dim 64 --K 3 \
    --use_ot_alignment --lambda_ot 0.005 --ot_reg 0.1 \
    --ot_cost_type cosine --ot_marginal_type uniform \
    --use_cdan \
    --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64 \
    --lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005 --lambda_migrate 0.01 \
    --alph 1.4 --beta 0.1 \
    --use_fpi --fpi_gamma_max 0.1 --fpi_gamma_init 0.05 --fpi_q_prior_weight 0.01 \
    --fpi_dropout 0.1 --fpi_m_max 16 --fpi_fp_dim 1024 --lambda_fpi_gamma_l1 0.001 \
    --fpi_detach_mode none \
    --pretrain_num_epochs 300 --train_num_epochs 100 --contrastive_num_epochs 200 \
    > "$ABLA_LOG_DIR/exp06_resnet_backbone.log" 2>&1

echo "[6/8] ResNet Backbone completed at $(date)"

# ============================================================
# Experiment 7: Ablation - Program Encoder degenerate (num_programs=1)
# ============================================================
echo ""
echo "[7/8] Ablation: Program Encoder degenerate (num_programs=1)..."
python train_mmpa_program.py \
    --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 --train \
    --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
    --num_programs 1 --program_dim 64 --K 3 \
    --use_ot_alignment --lambda_ot 0.005 --ot_reg 0.1 \
    --ot_cost_type cosine --ot_marginal_type uniform \
    --use_cdan \
    --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64 \
    --lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005 --lambda_migrate 0.01 \
    --alph 1.4 --beta 0.1 \
    --use_fpi --fpi_gamma_max 0.1 --fpi_gamma_init 0.05 --fpi_q_prior_weight 0.01 \
    --fpi_dropout 0.1 --fpi_m_max 16 --fpi_fp_dim 1024 --lambda_fpi_gamma_l1 0.001 \
    --fpi_detach_mode none \
    --pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100 \
    > "$ABLA_LOG_DIR/exp07_single_program.log" 2>&1

echo "[7/8] Single Program completed at $(date)"

# ============================================================
# Experiment 8: Ablation - Global KNN (remove per-drug mechanism)
# ============================================================
echo ""
echo "[8/8] Ablation: Global KNN (remove per-drug mechanism)..."

# Backup original finetune.py
cp finetune.py finetune.py.bak.perdrug

# Patch get_gene_feat_sim to global version
python3 << 'PYEOF'
import re

with open('finetune.py', 'r') as f:
    content = f.read()

old_func = '''def get_gene_feat_sim(all_source_fusion_feat, all_target_fusion_feat, drug_num, K):
    """
    Per-drug KNN: compute similarity in each drug's fusion feature space independently.
    Args:
        all_source_fusion_feat: [N, drug_num, D]
        all_target_fusion_feat: [M, drug_num, D]
        drug_num: number of drugs
        K: top-K neighbors
    Returns:
        topk_sim: [N, drug_num, K]
        topk_idx: [N, drug_num, K]
    """
    N = all_source_fusion_feat.size(0)
    device = all_source_fusion_feat.device

    topk_sim = torch.zeros(N, drug_num, K, device=device)
    topk_idx = torch.zeros(N, drug_num, K, dtype=torch.long, device=device)

    for d in range(drug_num):
        source_d = all_source_fusion_feat[:, d, :]  # [N, D]
        target_d = all_target_fusion_feat[:, d, :]  # [M, D]

        sim = F.cosine_similarity(source_d.unsqueeze(1), target_d.unsqueeze(0), dim=-1)  # [N, M]
        sim_d, idx_d = torch.topk(sim, K, dim=1)  # [N, K]
        topk_sim[:, d, :] = sim_d
        topk_idx[:, d, :] = idx_d

    return topk_sim, topk_idx'''

new_func = '''def get_gene_feat_sim(all_source_fusion_feat, all_target_fusion_feat, drug_num, K):
    """
    Global KNN (ablation): compute similarity across all drugs jointly.
    Per-drug mechanism removed: all drugs share the same neighbor set.
    """
    N = all_source_fusion_feat.size(0)
    M = all_target_fusion_feat.size(0)
    device = all_source_fusion_feat.device

    # Flatten drug dimension: [N, drug_num * D] and [M, drug_num * D]
    source_flat = all_source_fusion_feat.reshape(N, -1)
    target_flat = all_target_fusion_feat.reshape(M, -1)

    # Global cosine similarity across all drugs
    sim = F.cosine_similarity(source_flat.unsqueeze(1), target_flat.unsqueeze(0), dim=-1)  # [N, M]
    sim_d, idx_d = torch.topk(sim, K, dim=1)  # [N, K]

    # Broadcast same global neighbors to all drugs
    topk_sim = sim_d.unsqueeze(1).expand(-1, drug_num, -1)  # [N, drug_num, K]
    topk_idx = idx_d.unsqueeze(1).expand(-1, drug_num, -1)  # [N, drug_num, K]

    return topk_sim, topk_idx'''

if old_func in content:
    content = content.replace(old_func, new_func)
    with open('finetune.py', 'w') as f:
        f.write(content)
    print("[PATCH] get_gene_feat_sim replaced with global KNN version")
else:
    print("[WARNING] Could not find exact function match for patching")
PYEOF

python train_mmpa_program.py \
    --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 --train \
    --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
    --num_programs 8 --program_dim 64 --K 3 \
    --use_ot_alignment --lambda_ot 0.005 --ot_reg 0.1 \
    --ot_cost_type cosine --ot_marginal_type uniform \
    --use_cdan \
    --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64 \
    --lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005 --lambda_migrate 0.01 \
    --alph 1.4 --beta 0.1 \
    --use_fpi --fpi_gamma_max 0.1 --fpi_gamma_init 0.05 --fpi_q_prior_weight 0.01 \
    --fpi_dropout 0.1 --fpi_m_max 16 --fpi_fp_dim 1024 --lambda_fpi_gamma_l1 0.001 \
    --fpi_detach_mode none \
    --pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100 \
    > "$ABLA_LOG_DIR/exp08_global_knn.log" 2>&1

# Restore original finetune.py
mv finetune.py.bak.perdrug finetune.py
echo "[PATCH] finetune.py restored"

echo "[8/8] Global KNN completed at $(date)"

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo "All 8 experiments completed at $(date)"
echo "Logs: $ABLA_LOG_DIR"
echo "========================================"

# Extract AUC results from logs
echo ""
echo "AUC Summary (grep from logs):"
for log in "$ABLA_LOG_DIR"/*.log; do
    exp_name=$(basename "$log" .log)
    auc_line=$(grep -o "auc.*[0-9]\{4\}" "$log" | tail -1 || true)
    if [ -n "$auc_line" ]; then
        echo "  $exp_name: $auc_line"
    fi
done
