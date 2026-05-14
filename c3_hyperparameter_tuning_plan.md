# C3 超参数调优计划

## 当前 C3 最佳配置

```yaml
backbone:
  backbone_type: "convnext1d"
  convnext_depth: 1
  convnext_kernel_size: 7
  convnext_expansion: 4
  num_gene_tokens: 64
  token_dim: 64

encoder:
  num_programs: 8
  program_dim: 64
  hidden_dims: [512, 256, 128, 64]
  drop: 0.2

sample_ot:
  enabled: true
  lambda_sample_ot: 0.01
  reg: 0.1
  cost_type: "cosine"
  marginal_type: "uniform"

cdan:
  enabled: true
  alpha: 0.3

contrastive:
  enabled: true
  K: 7
  beta: 0.3

training:
  pretrain_num_epochs: 300
  train_num_epochs: 100
  contrastive_num_epochs: 200
  lr: 0.001
  stage2_drop: 0.1

best_auc: 0.5924
```

---

## 调优策略

**两阶段筛选**：
1. **快速筛选 (n=2)**：每个参数组合用 2-fold CV 快速排除明显差的配置
2. **最终确认 (n=5)**：快速筛选出的 Top-3 配置用 5-fold CV 确认

---

## P0: 高优先级（最可能提升 AUC）

### 1. lambda_sample_ot (OT 权重)

当前：0.01（Phase 4 确定的最佳值）

| 候选值 | 理由 |
|:---|:---|
| 0.005 | 减轻 OT 约束，给主任务更多空间 |
| **0.01** | 当前最佳 |
| 0.02 | 增强 OT 对齐强度 |
| 0.05 | 强 OT 约束（stress test）|

搜索空间：`[0.005, 0.01, 0.02]`
预计实验数：3 × n=2 = ~1 小时

---

### 2. K (Contrastive Top-K)

当前：7

| 候选值 | 理由 |
|:---|:---|
| 3 | 只取最相似的 target 样本，更严格 |
| 5 | 中等严格度 |
| **7** | 当前最佳 |
| 9 | 更宽松的匹配窗口 |

搜索空间：`[3, 5, 7, 9]`
预计实验数：4 × n=2 = ~1.5 小时

---

### 3. alpha (CDAN 权重)

当前：0.3

| 候选值 | 理由 |
|:---|:---|
| 0.1 | 减轻 domain adaptation，专注分类 |
| **0.3** | 当前最佳 |
| 0.5 | 更强 domain adaptation |

搜索空间：`[0.1, 0.3, 0.5]`
预计实验数：3 × n=2 = ~1 小时

---

### 4. beta (Contrastive 权重)

当前：0.3

| 候选值 | 理由 |
|:---|:---|
| 0.1 | 减轻 contrastive 约束 |
| **0.3** | 当前最佳 |
| 0.5 | 更强 domain isolation |

搜索空间：`[0.1, 0.3, 0.5]`
预计实验数：3 × n=2 = ~1 小时

---

## P1: 中优先级（探索 encoder 结构）

### 5. program_dim (Program Token 维度)

当前：64

| 候选值 | 理由 |
|:---|:---|
| 32 | 更紧凑的表示，减少过拟合 |
| **64** | 当前最佳 |
| 128 | 更丰富的表达能力 |

搜索空间：`[32, 64, 128]`
预计实验数：3 × n=2 = ~1 小时

---

### 6. num_programs (Program 数量 K)

当前：8

| 候选值 | 理由 |
|:---|:---|
| 4 | 更少但更专用的 program |
| **8** | 当前最佳 |
| 16 | 更细粒度的 program 分解 |

搜索空间：`[4, 8, 16]`
预计实验数：3 × n=2 = ~1 小时

---

### 7. ot_reg (Sinkhorn 熵正则化)

当前：0.1

| 候选值 | 理由 |
|:---|:---|
| 0.05 | 更紧的 transport plan（ sharper alignment）|
| **0.1** | 当前最佳 |
| 0.2 | 更平滑的 transport plan |

搜索空间：`[0.05, 0.1, 0.2]`
预计实验数：3 × n=2 = ~1 小时

---

### 8. Stage 2 Dropout

当前：0.1

| 候选值 | 理由 |
|:---|:---|
| 0.05 | 更少的正则化 |
| **0.1** | 当前最佳 |
| 0.2 | 更强的正则化（可能改善 F1/ACC）|

搜索空间：`[0.05, 0.1, 0.2]`
预计实验数：3 × n=2 = ~1 小时

---

## P2: 低优先级（训练动态）

### 9. Learning Rate

当前：0.001

| 候选值 | 理由 |
|:---|:---|
| 0.0005 | 更保守的优化 |
| **0.001** | 当前最佳 |
| 0.002 | 更快收敛 |

搜索空间：`[0.0005, 0.001, 0.002]`
预计实验数：3 × n=2 = ~1 小时

---

### 10. Training Epochs

当前：train=100, contrastive=200

| 候选值 | 理由 |
|:---|:---|
| train=50, con=100 | 快速训练，防止过拟合 |
| **train=100, con=200** | 当前最佳 |
| train=150, con=300 | 更充分训练 |

搜索空间：`[(50,100), (100,200), (150,300)]`
预计实验数：3 × n=2 = ~1 小时

---

## 推荐执行计划

### Phase A: P0 快速筛选（n=2）

```bash
# 1. lambda_ot 筛选（3 个实验）
python train_mmpa_program.py ... --lambda_ot 0.005 --n 2
python train_mmpa_program.py ... --lambda_ot 0.01  --n 2  # baseline
python train_mmpa_program.py ... --lambda_ot 0.02  --n 2

# 2. K 筛选（4 个实验）
python train_mmpa_program.py ... --beta 0.3 --K 3 --n 2
python train_mmpa_program.py ... --beta 0.3 --K 5 --n 2
python train_mmpa_program.py ... --beta 0.3 --K 7 --n 2  # baseline
python train_mmpa_program.py ... --beta 0.3 --K 9 --n 2

# 3. alpha 筛选（3 个实验）
python train_mmpa_program.py ... --alph 0.1 --n 2
python train_mmpa_program.py ... --alph 0.3 --n 2  # baseline
python train_mmpa_program.py ... --alph 0.5 --n 2

# 4. beta 筛选（3 个实验）
python train_mmpa_program.py ... --beta 0.1 --K 7 --n 2
python train_mmpa_program.py ... --beta 0.3 --K 7 --n 2  # baseline
python train_mmpa_program.py ... --beta 0.5 --K 7 --n 2
```

**总计：13 个实验 × ~20 分钟 = ~4.5 小时**

### Phase B: P0 交叉验证（n=5）

取 Phase A 中 Top-3 配置，用 n=5 确认：

**总计：3 个实验 × ~1.5 小时 = ~4.5 小时**

### Phase C: P1 筛选（可选）

如果 Phase B 未突破 0.5924，继续 P1：

**总计：5 个参数 × 3 个值 × ~20 分钟 = ~5 小时（n=2）**

---

## 预期结果

| 场景 | 预期 AUC | 概率 |
|:---|:---|:---|
| 找到比 C3 更好的配置 | > 0.5924 | 30% |
| C3 仍是最佳 | = 0.5924 | 60% |
| 所有配置都更差 | < 0.5924 | 10% |

**保守估计**：最多找到 AUC 0.5950-0.6000 的配置（+0.003-0.008 提升）。

---

## 脚本准备

文件：`run_c3_hyperparam_tuning.py`

- 自动运行 Phase A 全部 13 个实验
- 每个实验 n=2
- 输出 `outputs/c3_tuning_results.csv`
- 自动排序并推荐 Top-3 配置

是否需要我生成并运行这个脚本？
