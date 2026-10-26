#!/bin/bash
# 快速测试3个种子

SEEDS=(2024 2025 2026)

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Testing seed: $SEED"
    echo "=========================================="
    
    # 修改种子
    sed -i "s/seed = 2025/seed = $SEED/" train_adadrp_program.py
    
    # 运行实验
    python train_adadrp_program.py \
        --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 \
        --alph 0.3 --beta 0.3 --train \
        --num_programs 8 --program_dim 64 \
        --lambda_div 0.01 --lambda_sparse 0.001 \
        --use_ot_alignment --lambda_ot 0.005 --ot_reg 0.1 \
        --ot_cost_type cosine --ot_marginal_type uniform \
        --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
        --K 3 2>&1 | tail -50
    
    echo ""
    echo "Seed $SEED completed"
    echo ""
done

# 恢复原始种子
sed -i "s/seed = [0-9]*/seed = 2025/" train_adadrp_program.py
