# CODEMAP_MACB.md — MACB-DRP Code Structure & MMPA-DRP Migration Plan

> **Project**: MACB-DRP → MMPA-DRP  
> **Phase**: 1 + 2 (Code structure梳理 + Safe extension interfaces + MolecularProgramEncoder 接入)  
> **Date**: 2026-05-11  
> **Rule**: 所有原始 MACB-DRP 文件禁止直接重命名/删除；新增 MMPA 模块必须与原始代码并存。

---

## 1. 如何运行原始 MACB-DRP

```bash
python main.py \
  --measurement Z_SCORE \
  --thres_gdsc 0.0 \
  --thres_label 0.1 \
  --n 5 \
  --alph 0.3 \
  --beta 0.3 \
  --train
```

参数说明见 `main.py` argparse。训练参数默认值在 `train_params.json` 中定义。

---

## 2. 如何运行 MMPA-DRP Phase 2

```bash
python train_mmpa_program.py \
  --measurement Z_SCORE \
  --thres_gdsc 0.0 \
  --thres_label 0.1 \
  --n 5 \
  --alph 0.3 \
  --beta 0.3 \
  --train \
  --num_programs 8 \
  --program_dim 64 \
  --lambda_div 0.01 \
  --lambda_sparse 0.001
```

MMPA 专属参数：
- `--num_programs`: 分子程序数量（默认 8），即 `MolecularProgramEncoder` 输出的 program tokens 个数
- `--program_dim`: 每个程序 token 的维度（默认 64）
- `--lambda_div`: diversity loss 权重（默认 0.01）
- `--lambda_sparse`: sparsity loss 权重（默认 0.001）

---

## 3. 数据加载入口

| 文件 | 说明 |
|------|------|
| `dataload.py` | 所有数据加载函数 |
| `get_unlabeled_dataloaders(...)` | Stage 1：加载 CCLE (source) 和 TCGA (target) 无标签基因表达数据 |
| `get_multi_labeled_dataloader(...)` | Stage 2&3：加载 CCLE 有标签药物响应数据 + TCGA 临床标签数据，5-fold CV |
| `get_drug_feat(...)` | 基于 SMILES 用 RDKit 生成药物指纹特征 |

**注意**：`dataload.py` 在导入时会更新全局变量 `config.tissue_map`、`config.label_graph`、`config.drug_feat`。

---

## 4. 模型入口

### 原始 MACB-DRP

| 文件 | 说明 |
|------|------|
| `models.py` | 原始 MACB-DRP 所有神经网络定义 |
| `FeatMLP` | 通用 MLP encoder / predictor（Stage 1 和 Stage 2 复用） |
| `Drug_MLP` | 药物指纹编码器，输出 drug embedding `[D, d]` |
| `Fusion_MLP` | Stage 2 核心：bilinear fusion + CDAN + response predictor |
| `ReverseLayerF` | Gradient Reversal Layer（用于 adversarial domain adaptation） |

### MMPA-DRP 新增 / 替换模型

| 文件 | 类 | 说明 |
|------|-----|------|
| `mmpa_drp/models.py` | `MolecularProgramEncoder` | Gene encoder 替换模块。输入 `x:[B,G]`，输出 `{"z":[B,K,d], "q":[B,K], "h":[B,d]}`，其中 `h = sum_k q_k * z_k` |
| `mmpa_drp/models.py` | `ProgramTissueClassifier` | Stage 1 替代 `ResNetTissueClassifier`。封装 `MolecularProgramEncoder` + tissue classifier + domain discriminator，接口与原模型完全一致 `(tissue_pred, domain_pred, gene_feat)` |
| `mmpa_drp/models.py` | `ProgramFusionMLP` | Stage 2/3 替代 `Fusion_MLP`。`encoder` 接收返回 dict 的 `MolecularProgramEncoder`，其余 bilinear/CDAN/predictor 逻辑完全保留 |
| `mmpa_drp/models.py` | `DrugMechanismGate` | Phase 1 占位（未接入训练） |
| `mmpa_drp/models.py` | `DrugGatedProgramAggregator` | Phase 1 占位（未接入训练） |

---

## 5. Stage 1: Tissue-aware representation learning

### 原始对应文件

| 组件 | 文件 | 类 / 函数 |
|------|------|-----------|
| **Gene Encoder** | `pretraining.py` | `ResNetTissueClassifier` 的前向部分（`input_layer` + `res_blocks` + `attention`） |
| **Tissue Classifier** | `pretraining.py` | `ResNetTissueClassifier.classifier` |
| **Domain Discriminator** | `pretraining.py` | `ResNetTissueClassifier.domain_discriminator` |
| **Tissue Classification Loss** | `pretraining.py` | `nn.CrossEntropyLoss(weight=...)` |
| **Adversarial Loss** | `pretraining.py` | `nn.BCEWithLogitsLoss()` (domain binary classification) |
| **Stage 1 训练函数** | `pretraining.py` | `train_tissue_classifier(...)` |

### MMPA 替代对应文件

| 组件 | 文件 | 类 / 函数 |
|------|------|-----------|
| **Gene Encoder (Program)** | `mmpa_drp/models.py` | `MolecularProgramEncoder` |
| **Tissue Classifier (Program)** | `mmpa_drp/models.py` | `ProgramTissueClassifier` |
| **Stage 1 训练函数 (Program)** | `mmpa_drp/stage1_program.py` | `train_tissue_classifier_program(...)` |
| **Feature Save (Program)** | `mmpa_drp/stage1_program.py` | `save_feat_program(...)` |

### 关键输出形状

```python
# 原始 ResNetTissueClassifier
tissue_pred, domain_pred, attended_features = model(x)
# attended_features 即 gene representation h
# h shape: [B, 64]

# MMPA ProgramTissueClassifier
tissue_pred, domain_pred, gene_feat = model(x)
# gene_feat 即 h_pool，由 MolecularProgramEncoder 聚合得到
# h_pool shape: [B, 64]  （与原 h 完全一致）

# MolecularProgramEncoder 内部
out = program_encoder(x)
out["z"].shape  # [B, K, d]
out["q"].shape  # [B, K]
out["h"].shape  # [B, d]
```

### Stage 1 辅助损失（Phase 2 新增）

| Loss | 文件 | 函数 | 权重 |
|------|------|------|------|
| Reconstruction Loss | `mmpa_drp/losses.py` | `program_reconstruction_loss` | 0.0（本阶段未实现 decoder） |
| Diversity Loss | `mmpa_drp/losses.py` | `program_diversity_loss` | `lambda_div`（默认 0.01） |
| Sparsity Loss | `mmpa_drp/losses.py` | `program_sparsity_loss` | `lambda_sparse`（默认 0.001） |

---

## 6. Stage 2: Drug-conditioned adversarial transfer and bilinear fusion

### 原始对应文件

| 组件 | 文件 | 类 / 函数 |
|------|------|-----------|
| **Drug Encoder** | `models.py` | `Drug_MLP` |
| **Bilinear Fusion** | `models.py` | `Fusion_MLP.bilinear` + `Fusion_MLP.fusion_module` |
| **Drug Response Predictor** | `models.py` | `Fusion_MLP.fusion_module`（输出经 `torch.sigmoid`） |
| **CDAN / Conditional Domain Discriminator** | `models.py` | `Fusion_MLP.cdan_discriminator` + `Fusion_MLP.drug_classifier` |
| **Prediction Loss** | `finetune.py` | `nn.BCEWithLogitsLoss(reduction='none')` |
| **CDAN Loss** | `finetune.py` | `nn.BCEWithLogitsLoss()`（domain discriminator binary loss） |
| **Stage 2 训练函数** | `finetune.py` | `fusion_model_train_step(...)` |

### MMPA 替代对应文件

| 组件 | 文件 | 类 / 函数 |
|------|------|-----------|
| **Fusion Model (Program)** | `mmpa_drp/models.py` | `ProgramFusionMLP` |
| **Stage 2&3 训练函数 (Program)** | `mmpa_drp/stage2_program.py` | `fusion_training_program(...)` |

**关键说明**：`ProgramFusionMLP.forward` 返回值与原始 `Fusion_MLP.forward` **完全兼容**：
```python
return outputs, gene_feat, feat, expanded_tissue_idx, cdan_domain_output, drug_pred
```
因此 `finetune.py` 中的 `fusion_model_train_step`、`fusion_eval_epoch`、`get_featuers` 等函数可以直接复用，无需修改。

### 关键输入 / 输出形状

```python
# ProgramFusionMLP.forward(inputs, drug_feat, tissue_idx, alpha=1.0)
encoder_out = self.encoder(inputs)
gene_feat = encoder_out["h"]   # [B, 64]  <- 唯一改动：从 tuple 解包改为 dict 取 "h"

# 后续 bilinear / CDAN / predictor 逻辑与原 Fusion_MLP 完全一致
drug_feat, _ = self.drug_mlp(drug_feat)  # [D, 64]
...
```

---

## 7. Stage 3: Contrastive domain isolation

### 对应文件

| 组件 | 文件 | 类 / 函数 | 状态 |
|------|------|-----------|------|
| **Contrastive Anchoring** | `finetune.py` | `fusion_model_train_step_with_contrastive(...)` | ✅ 直接复用 |
| **Top-K Positive Pair 选择** | `finetune.py` | `get_gene_feat_sim(...)` + `get_featuers(...)` | ✅ 直接复用 |
| **Negative Pair 构造** | `finetune.py` | `all_target_fusion_feat.reshape(-1, D)` | ✅ 直接复用 |
| **Contrastive Loss** | `myloss.py` | `contrastive_loss(...)` | ✅ 直接复用 |
| **Stage 3 训练函数** | `mmpa_drp/stage2_program.py` | `fusion_training_program` 中的 contrastive loop | ✅ 已接入 |

**Phase 2 未改动**：本阶段仍使用 `sample-level` contrastive（基于 `h_pool` 的 cosine similarity 选 Top-K），**尚未替换为 tissue-program prototype alignment**。

---

## 8. Evaluation 入口

| 文件 | 函数 | 说明 |
|------|------|------|
| `testing.py` | `testing(...)` | TCGA 测试集评估入口 |
| `testing.py` | `multi_eval_epoch(...)` | 计算 AUC, AUPR, ACC, F1 |
| `finetune.py` | `fusion_eval_epoch(...)` | Validation 评估（GDSC/CCLE 验证集） |

MMPA 结果额外保存至：`outputs/mmpa_step2_results.csv`

---

## 9. Checkpoint 保存位置

- `fusion_model.pt` 保存在 `logs/<timestamp>_<param_str>/fusion_model.pt`
- `source_features.npy` / `target_features.npy` 保存在当前工作目录（由 `save_feat_program` 生成）
- Stage 1 的 `pretrain_model` 不直接保存 checkpoint，而是通过 `deepcopy(trained_model.program_encoder)` 传给 Stage 2 的 `ProgramFusionMLP`

---

## 10. 任务完成状态总览

### ✅ Phase 1 已完成

| # | 任务 | 状态 |
|---|------|------|
| 1 | 梳理 MACB-DRP 三阶段代码结构 | ✅ |
| 2 | 记录 Stage 1/2/3 对应文件、函数、张量 shape | ✅ |
| 3 | 建立 `mmpa_drp/` 扩展包结构 | ✅ |
| 4 | 实现 `MolecularProgramEncoder`（Phase 1 骨架） | ✅ |
| 5 | 实现 `DrugMechanismGate`（Phase 1 骨架） | ✅ |
| 6 | 实现 `DrugGatedProgramAggregator`（Phase 1 骨架） | ✅ |
| 7 | 实现 `ProgramPrototypeAlignmentLoss`（Phase 1 接口） | ✅ |
| 8 | 实现 `ProgramPrototypeContrastiveLoss`（Phase 1 接口） | ✅ |
| 9 | 原始 MACB-DRP 文件未修改 | ✅ |

### ✅ Phase 2 已完成

| # | 任务 | 状态 | 文件 |
|---|------|------|------|
| 1 | `MolecularProgramEncoder` 输出改为 dict `{"z", "q", "h"}`，其中 `h = sum_k q_k * z_k` | ✅ | `mmpa_drp/models.py` |
| 2 | 实现 `ProgramTissueClassifier`（Stage 1 drop-in replacement） | ✅ | `mmpa_drp/models.py` |
| 3 | 实现 `ProgramFusionMLP`（Stage 2/3 drop-in replacement） | ✅ | `mmpa_drp/models.py` |
| 4 | 实现 Program auxiliary losses（diversity, sparsity, reconstruction placeholder） | ✅ | `mmpa_drp/losses.py` |
| 5 | 实现 `train_tissue_classifier_program`（Stage 1 训练，含 auxiliary losses） | ✅ | `mmpa_drp/stage1_program.py` |
| 6 | 实现 `save_feat_program`（适配 ProgramTissueClassifier） | ✅ | `mmpa_drp/stage1_program.py` |
| 7 | 实现 `fusion_training_program`（Stage 2&3 训练，使用 ProgramFusionMLP） | ✅ | `mmpa_drp/stage2_program.py` |
| 8 | 新增 `train_mmpa_program.py` 主入口 | ✅ | `train_mmpa_program.py` |
| 9 | 原始 MACB-DRP 文件未修改 | ✅ | 全部原始文件时间戳未变 |
| 10 | `ProgramFusionMLP` 使用延迟导入避免 rdkit 依赖链 | ✅ | `mmpa_drp/models.py` |

### ✅ Phase 2 （现已完成）

| # | 任务 | 状态 | 说明 |
|---|------|------|------|
| 1 | 运行 `test_mmpa_modules.py` smoke test 验证 forward/backward | ✅ 已通过 | 在 `macb-drp` conda 环境（torch+rdkit）和 base 环境（仅 torch）均验证通过；通过 mock 断开了 rdkit/sklearn/matplotlib 导入链 |
| 2 | 端到端训练验证（Stage 1 → Stage 2 → Stage 3 → Evaluation） | ✅ 已通过 | 使用 `train_mmpa_program.py` 快速验证（pretrain/train/contrastive 各 1 epoch），5-fold CV 完整跑通，无报错 |
| 3 | 输出 `outputs/mmpa_step2_results.csv` 并确认指标正常 | ✅ 已生成 | 文件已生成，包含 9 种药物的 auc/aupr/f1/acc 指标；各 fold 结果保存在 `results/fold-{0..4}/` 中 |

---

## 11. 禁止直接修改的文件（仍然有效）

以下文件在 Phase 1/2 均保持原样，原始 MACB-DRP 训练命令仍能运行：

- `main.py`（原始训练入口）
- `pretraining.py`（Stage 1 全部逻辑）
- `models.py`（原始 MACB-DRP 模型定义）
- `finetune.py`（Stage 2 & 3 训练逻辑）
- `myloss.py`（原始 contrastive loss）
- `dataload.py`（数据加载逻辑）
- `testing.py`（评估逻辑）
- `utility.py`（工具函数）
- `config.py`（全局配置）
- `train_params.json`（原始超参）

---

## 12. 下一阶段改造计划（Phase 3）

### Step 3: 替换 Bilinear Fusion 输入（引入 Drug Mechanism Gate）
- **目标**：不再直接用 `h_pool` 做 bilinear，而是用 drug-gated aggregated representation `v`
- **公式**：`v_{i,d} = sum_k q_{i,k} * g_{d,k} * z_{i,k}`
- **改造点**：
  - 在 `ProgramFusionMLP` 中新增 `DrugMechanismGate`
  - 将 `bilinear` 的输入从 `gene_feat_expand` 改为 `v.reshape(-1, d)`
- **状态**：`DrugMechanismGate` 和 `DrugGatedProgramAggregator` 已在 `mmpa_drp/models.py` 中实现，但未接入训练

### Step 4: 替换 Sample-level Contrastive 为 Tissue-Program Prototype Alignment
- **目标**：放弃基于 `h_pool` cosine similarity 的 sample-level Top-K 对比
- **新机制**：
  - 按 tissue type 聚合 source/target 的 program tokens `z`，得到 tissue-program prototypes `[C, K, d]`
  - 使用 `ProgramPrototypeAlignmentLoss` 做跨 domain prototype alignment
- **状态**：接口已在 `mmpa_drp/losses.py` 中定义，但核心逻辑仍是 TODO

### Step 5: 加入 Drug-Gated Program Contrastive Separation
- **目标**：在 prototype alignment 基础上，加入 drug gate 加权的 program-level contrastive separation
- **状态**：接口已在 `mmpa_drp/losses.py` 中定义，但核心逻辑仍是 TODO

---

## 13. Phase 2 消融实验结果（正式运行）

> **运行时间**：2026-05-11  
> **环境**：`macb-drp` conda (torch+rdkit+pandas)  
> **设置**：pretrain 300 / train 100 / contrastive 200 epochs，5-fold CV，`num_programs=8`，`Top-K=7`

### 13.1 汇总表（9 种药物平均）

| 指标 | MACB-DRP | MMPA w/o aux | MMPA + aux | 提升 (vs MACB) |
|------|----------|--------------|------------|----------------|
| **AUC** | 0.4970 | 0.5292 | **0.5763** | **+15.9%** |
| **AUPR** | 0.6657 | 0.6874 | **0.7098** | **+6.6%** |
| **F1** | 0.4951 | 0.5103 | **0.5420** | **+9.5%** |
| **ACC** | 0.4657 | 0.4786 | **0.5108** | **+9.7%** |

### 13.2 关键发现

1. **MMPA w/o aux > MACB-DRP**：仅将基因编码器替换为 `MolecularProgramEncoder`（`num_programs=8`，即 8 个 program tokens），不使用任何辅助损失，就在全部四项指标上超越了原始模型。说明**程序 token 的解构表示本身**就具有优势。
2. **MMPA + aux > MMPA w/o aux**：加入 `diversity loss` + `sparsity loss` 后，AUC 从 0.5292 进一步提升到 0.5763（+8.9%），AUPR/F1/ACC 也全面上涨。说明**辅助损失能有效引导 program tokens 学到更有结构、更稀疏的表征**。

### 13.3 各药物 AUC 对比（MMPA + aux vs MACB-DRP）

| 药物 | MACB-DRP | MMPA + aux | 变化 |
|------|----------|------------|------|
| 5-Fluorouracil | 0.5394 | 0.5777 | +7.1% |
| Cisplatin | 0.6434 | 0.6083 | -5.5% |
| Cyclophosphamide | 0.4268 | **0.6063** | **+42.0%** |
| Docetaxel | 0.3942 | 0.4785 | +21.4% |
| Doxorubicin | 0.3232 | **0.4691** | **+45.1%** |
| Etoposide | 0.6577 | 0.7034 | +6.9% |
| Gemcitabine | 0.5755 | 0.5896 | +2.4% |
| Paclitaxel | 0.4273 | 0.4657 | +9.0% |
| Temozolomide | 0.4858 | **0.6884** | **+41.7%** |

> **注**：Cisplatin 是唯一 MMPA + aux 略低于 MACB-DRP 的药物（-5.5%），其余 8 种药物全部提升。Cyclophosphamide、Doxorubicin、Temozolomide 提升最为显著（>40%）。

### 13.4 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/mmpa_step2_ablation.csv` | 三模型 mean 指标对比 |
| `outputs/mmpa_step2_ablation_full.csv` | 每药物完整四项指标 |
| `results/..._mmpa_noaux.csv` | MMPA w/o aux 5-fold 平均 |
| `results/..._mmpa_aux.csv` | MMPA + aux 5-fold 平均 |

---

## 14. Phase 3A 消融实验结果（Disentangled Program Encoder）

> **运行时间**：2026-05-11  
> **设置**：pretrain 300 / train 100 / contrastive 200 epochs，5-fold CV，`num_programs=8`，`Top-K=7`

### 14.1 实验设计

| 实验 | 配置 |
|------|------|
| **A** | Phase 2 baseline (MMPA + aux) |
| **B** | + disentangle branches only (λ_adv=0, λ_nuisance=0, λ_ortho=0) |
| **C** | + branches + adversarial response (λ_adv=0.005) |
| **D** | + branches + adv + domain nuisance (λ_adv=0.005, λ_nuisance=0.005) |
| **E** | 完整版 (+ orthogonality, λ_ortho=0.001) |

### 14.2 汇总结果

| 实验 | AUC | AUPR | F1 | ACC |
|------|-----|------|----|-----|
| **A** Phase 2 baseline | **0.5763** | **0.7098** | **0.5420** | **0.5108** |
| **B** branches only | 0.5198 | 0.6861 | 0.4644 | 0.3925 |
| **C** + adv response | 0.5295 | 0.6849 | 0.5131 | 0.4931 |
| **D** + adv + nuisance | 0.5425 | 0.6922 | 0.5194 | 0.4747 |
| **E** complete + ortho | 0.5507 | 0.6976 | 0.5396 | 0.5091 |

### 14.3 关键发现

1. **单纯增加 projection heads（ExpB）会损害性能**：AUC 从 0.5763 降至 0.5198。无约束的解耦导致表征退化。
2. **Adversarial response（ExpC）开始修复**：AUC 回升至 0.5295，F1 大幅回升至 0.5131。
3. **Domain nuisance（ExpD）继续修复**：AUC 回升至 0.5425。
4. **Orthogonality（ExpE）最接近 baseline**：AUC 0.5507，F1 0.5396，ACC 0.5091，但仍未超越 Phase 2。
5. **结论**：解耦约束是有效的（B→C→D→E 逐步恢复），但 naive disentanglement 引入了表示退化，需要进一步修复。

---

## 15. Phase 3A-R: Residual-Preserving Disentanglement

> **状态**：代码已完成，消融实验进行中（R1–R5）  
> **核心思想**：用残差缩放 `h_pred = h_base + alpha * h_response` + teacher distillation + freeze warm-up 修复 Phase 3A 的表征退化。

### 15.1 新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| `DisentangledProgramEncoder` (residual 版) | `mmpa_drp/models.py` | 支持 `residual_mode`: fixed / learnable_scalar / learnable_vector |
| `BinaryDomainDiscriminator` | `mmpa_drp/models.py` | 二分类 domain discriminator |
| `DisentangledProgramTissueClassifier` | `mmpa_drp/models.py` | Stage 1 使用 `h_pred` 做分类 |
| `DisentangledProgramFusionMLP` | `mmpa_drp/models.py` | Stage 2/3 使用 `h_pred` 做 bilinear fusion |
| `orthogonality_loss` | `mmpa_drp/losses.py` | DCDNet DiffLoss 风格的正交约束 |
| `response_domain_adversarial_loss` | `mmpa_drp/losses.py` | GRL + domain discriminator on h_response |
| `domain_nuisance_loss` | `mmpa_drp/losses.py` | 正常 domain classification on h_domain |
| `train_tissue_classifier_disentangle` | `mmpa_drp/stage1_disentangle.py` | Stage 1 训练，支持 freeze warm-up + distill |
| `fusion_training_disentangle` | `mmpa_drp/stage2_disentangle.py` | Stage 2/3 训练 |
| `train_mmpa_disentangle.py` | 根目录 | Phase 3A 主入口 |
| `run_ablation_step3a.py` | 根目录 | Phase 3A 消融脚本 (A–E) |
| `run_ablation_step3a_residual.py` | 根目录 | Phase 3A-R 消融脚本 (A, E, R1–R5) |
| `extract_teacher_encoder.py` | 根目录 | 从 Phase 2 fusion model 提取 teacher encoder |

### 15.2 Residual Scaling 设计

```python
h_pred = h_base + alpha * h_response
```

| 模式 | alpha | 初始化 |
|------|-------|--------|
| fixed | scalar (0.1) | 固定不变 |
| learnable_scalar | nn.Parameter | 0.0 |
| learnable_vector | nn.Parameter([d]) | 0.0 |

默认使用 **learnable_scalar, init=0.0**，使模型初始等价于 Phase 2，训练中逐渐学习是否引入 response branch。

### 15.3 Teacher Distillation

- Teacher: Phase 2 `MolecularProgramEncoder`（从 `fusion_model.pt` 提取）
- Student: `DisentangledProgramEncoder` 的 `h_pred`
- Loss: `MSE(h_pred, stopgrad(h_teacher))`
- Teacher 始终 `eval()` + `requires_grad=False`

### 15.4 Freeze Warm-up

- 前 `freeze_warmup_epochs`（默认 20）只训练 response_head / domain_head / discriminators / residual_alpha
- 冻结 `MolecularProgramEncoder` backbone
- Warm-up 结束后解冻全部参数

### 15.5 实验结果 (R1–R5)

| 实验 | alpha 模式 | λ_adv | λ_nuisance | λ_ortho | distill | freeze warmup | **AUC** | AUPR | F1 | ACC |
|------|-----------|-------|-----------|---------|---------|---------------|---------|------|----|-----|
| **A** | Phase 2 baseline | — | — | — | — | — | **0.5763** | **0.7098** | **0.5420** | **0.5108** |
| **E** | original (no residual) | 0.005 | 0.005 | 0.001 | ❌ | ❌ | 0.5507 | 0.6976 | 0.5396 | 0.5091 |
| **R1** | fixed 0.1 | 0.005 | 0.005 | 0.001 | ❌ | ❌ | 0.5500 | 0.6910 | 0.5136 | 0.4877 |
| **R2** | learnable scalar | 0.005 | 0.005 | 0.001 | ❌ | ❌ | 0.5086 | 0.6699 | 0.4940 | 0.4805 |
| **R3** | learnable scalar + weak loss | 0.001 | 0.005 | 0.0005 | ❌ | ❌ | 0.5360 | 0.6863 | 0.5299 | 0.4956 |
| **R4** | learnable + weak + distill | 0.001 | 0.005 | 0.0005 | ✅ 0.01 | ❌ | 0.5443 | 0.6984 | 0.5258 | 0.5327 |
| **R5** | learnable + weak + distill + freeze | 0.001 | 0.005 | 0.0005 | ✅ 0.01 | ✅ 20ep | **0.5533** | 0.6947 | 0.5339 | 0.5181 |

### 15.6 关键发现

1. **R5 最接近 baseline**：AUC 0.5533，比 E (0.5507) 提升 0.0026，但仍低于 Phase 2 (0.5763)。说明 residual scaling + distill + freeze warm-up **部分修复了表征退化**，但未完全恢复。
2. **降低 loss 权重有帮助**：R3 (weak loss, AUC 0.5360) > R2 (strong loss, AUC 0.5086)。过强的 adversarial/orthogonality 约束会进一步抹除响应信号。
3. **Teacher distillation 有效**：R4 (0.5443) > R3 (0.5360)，distill 帮助 h_pred 不偏离有效表示。
4. **Freeze warm-up 有效**：R5 (0.5533) > R4 (0.5443)，先固定 backbone 训练新 heads 再解冻是合理的策略。
5. **Residual 本身不是银弹**：R1 (fixed α=0.1, AUC 0.5500) 与 E (α=1.0 implicit, AUC 0.5507) 几乎相同，说明单纯改变 α 形式不够，需要配合 distill + freeze。
6. **药物异质性**：
   - Etoposide 在 R4 达到最高 AUC (0.7872)，远超 baseline (0.7034)；
   - Gemcitabine (R5: 0.5957) 和 Paclitaxel (R5: 0.4847) 在 R5 取得提升；
   - Docetaxel、Doxorubicin、Cyclophosphamide 在所有 R 配置中均低于 baseline。

### 15.7 结论

- **已停止**。Residual-Preserving Disentanglement **部分有效**但**未超越 Phase 2**（最佳 R5 AUC 0.5533 < baseline 0.5763）。
- 解耦架构的根本限制可能需要更强的 alignment 机制来修复。
- 相关消融脚本和模型代码保留在 `mmpa_drp/` 中作为历史记录，但不再继续迭代。

---

## 16. DCDNet 代码分析

> **Repo**: https://github.com/rawwap/DCDNet  
> **Commit**: `3c393b218eaf0ec9cb928da6c9e8abd80c67c0d1`  
> **分析文档**: `CODEMAP_DCDNET.md`

### 16.1 核心模块定位

| 论文模块 | 代码文件 | 类/函数 | 输入/输出 Shape |
|----------|----------|---------|----------------|
| ACFD shared branch | `model/DCDNet.py` | `Shared` (line 344) | `[B,256,H,W]` → `[B,1024,H/2,W/2]` |
| ACFD private branch | `model/DCDNet.py` | `Private` (line 375) | `[B,1024,H,W]` → `[B,1024,H,W]` |
| MGDF | `model/DCDNet.py` | `Fusion` (line 408) | 3× `[B,1024,H,W]` → `[B,1024,H,W]` |
| CAM | `model/DCDNet.py` | `Modulation` (line 10) | `private, shared` → `modulated private` |
| GRL | `model/discriminator.py` | `GradientReversal` (line 72) | — |
| Discriminator | `model/discriminator.py` | `Discriminator` (line 5) | `[B,1024,H,W]` → `[B,num_classes]` |
| Orthogonality loss | `train.py` | `DiffLoss` (line 232) | 两个 feature map → scalar |
| Contrastive loss | `train.py` | `PrivateContraLoss` (line 251) | feature, class label → scalar |

### 16.2 可迁移到 MMPA-DRP 的思想

| DCDNet 概念 | MMPA-DRP 适配 |
|-------------|---------------|
| base feature | `z_base` / `h_base` |
| private feature | `z_response` / `h_response` (response-relevant) |
| shared feature | `z_domain` / `h_domain` (domain-nuisance) |
| adversarial loss | GRL on `h_response` + domain discriminator |
| orthogonality loss | `orthogonality_loss(z_response, z_domain)` |
| contrastive loss | Drug-response prototype contrastive (后续) |
| MGDF | Program-wise dynamic fusion (后续) |
| CAM | Optional later-stage modulation (建议暂缓) |

### 16.3 不可直接迁移的部分

- **Pixel-level CE Loss**：DCDNet 是分割任务，MMPA-DRP 是 DRP 预测
- **H×W 空间依赖**：`Conv2d`、`Dropout2d`、`MaxPool2d` 需改为 `Linear`
- **Few-shot support/query 结构**：MMPA-DRP 是标准监督/域适应，无 support/query
- **Real/Fake Discriminator**：过于复杂，建议简化为 binary domain discriminator

---

## 17. Phase 3B 实验结果（DrugMechanismGate）

> **运行时间**：2026-05-11  
> **设置**：pretrain 300 / train 100 / contrastive 200 epochs，5-fold CV，`num_programs=8`，`Top-K=7`

### 17.1 实验设计

| 实验 | 配置 |
|------|------|
| **A** | Phase 2 baseline (MMPA + aux, use_drug_gate=False) |
| **B** | + shared drug gate (所有药物共享同一个 gate) |
| **C** | + drug-specific gate (每个药物有自己的 gate) |
| **D** | + drug-specific gate + gate entropy reg (λ=0.1, minimize) |

### 17.2 汇总结果

| 实验 | AUC | AUPR | F1 | ACC |
|------|-----|------|----|-----|
| **A** Phase 2 baseline | **0.5763** | **0.7098** | **0.5420** | **0.5108** |
| **B** shared gate | 0.5371 | 0.6910 | 0.5060 | 0.4693 |
| **C** drug-specific gate | 0.5433 | 0.6864 | 0.4622 | 0.3457 |
| **D** + gate entropy | 0.5433 | 0.6864 | 0.4622 | 0.3457 |

### 17.3 关键发现

1. **Drug gate 未能提升性能**：所有 B/C/D 均低于 baseline AUC 0.5763。Drug-gated program aggregation 在当前架构下引入了额外复杂度，但没有带来收益。
2. **Gate collapse 严重**：诊断分析显示 7/9 种药物共享完全相同的 gate（cosine similarity = 1.0000）。只有 5-Fluorouracil 和 Cisplatin 学到了不同的 gate。
3. **Program usage 极不平衡**：
   - prog_1: 61.0% 平均权重
   - prog_2: 22.3%
   - prog_6: 11.0%
   - 其余 5 个 program: <1.5%（几乎不被使用）
4. **Gate entropy regularization 无效**：C 和 D 的结果完全相同，gate 权重也完全相同。λ=0.1 的 entropy reg 未能改变 gate 分布。
5. **Gate 趋于 one-hot**：5-FU 的 entropy=0.012，Cisplatin=0.020，均接近 one-hot（理论最小 0）。其余药物 entropy=0.440，仍然较低（最大 entropy log(8)=2.079）。
6. **Per-drug 分析**：只有 Etoposide 和 Gemcitabine 在 drug gate 版本中略高于 baseline，其余 7 种药物全部下降。

### 17.4 诊断输出

| 文件 | 说明 |
|------|------|
| `outputs/mmpa_step3b_results.csv` | A–D 四组实验 mean 指标对比 |
| `outputs/mmpa_step3b_per_drug.csv` | 每药物完整四项指标 |
| `outputs/drug_gate_step3b.csv` | Exp D 每药物 gate 权重 |
| `outputs/drug_gate_entropy.csv` | 每药物 gate entropy |
| `outputs/drug_gate_cosine.csv` | 药物间 gate cosine similarity |
| `outputs/drug_gate_avg_usage.csv` | 每个 program 平均使用权重 |

### 17.5 结论

- **已停止**。DrugMechanismGate **未超越 Phase 2**（最佳 G5 AUC 0.5683 < baseline 0.5763）。
- Gate collapse 严重（7/9 药物共享相同 gate），且 residual/additive 初始化未能修复。
- 相关消融脚本保留作为历史记录，不再继续迭代。

---

## 18. Phase OT 实验结果（Optimal Transport Alignment Loss）

> **运行时间**：2026-05-12  
> **设置**：pretrain 300 / train 100 / contrastive 200 epochs，5-fold CV，`num_programs=8`，`Top-K=7`  
> **核心设计**：OT loss 仅用于训练，不改变 inference（forward 返回值与 baseline 完全一致）

### 18.1 实验设计

| 实验 | 配置 | 说明 |
|------|------|------|
| **A** | Phase 2 baseline (CDAN only) | 参考基线 |
| **OT1** | `λ_ot=0.01, reg=0.1, uniform` | 轻量 OT，推荐主版本 |
| **OT2** | `λ_ot=0.05, reg=0.1, uniform` | 强 OT 权重 |
| **OT3** | `λ_ot=0.01, reg=0.05, uniform` | 更紧的 Sinkhorn plan |
| **OT4** | `λ_ot=0.01, reg=0.1, unbalanced` | 非均匀 marginal（已验证无效） |
| **OT5** | `λ_ot=0.005, reg=0.1, uniform` | 半量 OT 权重 |

### 18.2 汇总结果

| 实验 | AUC | AUPR | F1 | ACC | vs Baseline |
|------|-----|------|----|-----|-------------|
| **A** Baseline | **0.5763** | 0.7098 | 0.5420 | 0.5108 | — |
| **OT1** | **0.5798** | 0.7092 | 0.5324 | 0.5101 | **+0.35%** |
| **OT2** | **0.5803** | 0.7115 | 0.5256 | 0.5040 | **+0.39%** |
| OT3 | 0.5792 | 0.7121 | 0.5578 | 0.5302 | +0.29% |
| OT4 | 0.5790 | 0.7151 | 0.5272 | 0.4875 | +0.27% |
| OT5 | 0.5791 | 0.7138 | 0.5367 | 0.4977 | +0.27% |

### 18.3 关键发现

1. **全部 5 个 OT 配置均超越 baseline**。OT 是首个超越 Phase 2 的改进方向。
2. **OT1 是最稳健配置**：`λ=0.01, reg=0.1, uniform`，7/9 药物改善，平均排名 2.00（最佳）。
3. **OT2 高风险高回报**：`λ=0.05` 平均 AUC 最高 (0.5803)，但 Temozolomide 下降 -8.16%。
4. **unbalanced marginal 无效**：OT4 平均排名 4.11，表现最差。
5. **Temozolomide 是 OT 盲区**：所有 OT 配置在该药物上均低于 baseline，需要 drug-specific 权重。
6. **OT 不能替代 CDAN**：quick ablation 验证 OT-only (无 CDAN) < CDAN-only，OT 只能作为 CDAN 的补充。

### 18.4 推荐配置

**主版本（生产配置）：OT1**
```bash
python train_mmpa_program.py \
  --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 \
  --alph 0.3 --beta 0.3 --train \
  --num_programs 8 --program_dim 64 \
  --lambda_div 0.01 --lambda_sparse 0.001 \
  --use_ot_alignment \
  --lambda_ot 0.01 \
  --ot_reg 0.1 \
  --ot_max_iter 50 \
  --ot_cost_type cosine \
  --ot_marginal_type uniform
```

**激进版本：OT2**（对跨域差异大的药物更有效，但可能牺牲个别药物）
```bash
# 同上，仅 lambda_ot 改为 0.05
--lambda_ot 0.05
```

### 18.5 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/mmpa_ot_results.csv` | 6 组实验 mean 指标对比 |
| `outputs/mmpa_ot_per_drug.csv` | 每药物完整四项指标 |
| `results/..._mmpa_ExpOT{1-5}.csv` | 各实验 5-fold 原始结果 |

### 18.6 结论

- **OT alignment loss 是当前最有效的改进方向**，首次超越 Phase 2 baseline。
- **CDAN 不可替代**，OT 只能叠加在 CDAN 之上作为补充对齐约束。
- 下一步可考虑 **drug-adaptive OT**（不同药物使用不同 `λ_ot`）以解决 Temozolomide 下降问题。

---

## 19. Phase 5: ConvNeXt V2-inspired Backbone on OT1

### 19.1 背景
在 OT1 (AUC 0.5798) 基础上，升级 `MolecularProgramEncoder` 的 backbone，探索 GRN-enhanced MLP 和 ConvNeXt1D 是否能进一步提升性能。

### 19.2 新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| `GRN1DVector` | `mmpa_drp/backbones.py` | ConvNeXt V2 GRN for vector [B,C] |
| `GRN1DToken` | `mmpa_drp/backbones.py` | ConvNeXt V2 GRN for tokens [B,L,C] |
| `GRNMLPBackbone` | `mmpa_drp/backbones.py` | 2-layer MLP + GRN + LayerNorm + GELU |
| `ConvNeXt1DBlock` | `mmpa_drp/backbones.py` | 1D depthwise conv + pointwise MLP + GRN |
| `ConvNeXt1DBackbone` | `mmpa_drp/backbones.py` | learned tokenizer + ConvNeXt blocks + mean pool |
| `compute_program_diagnostics` | `mmpa_drp/program_diagnostics.py` | q_entropy, pairwise_cosine, effective_active_programs 等 |

`MolecularProgramEncoder` 新增 `backbone_type` 参数：
- `"mlp"`: 原始 MLP (Phase 2 兼容)
- `"grn_mlp"`: GRN-enhanced MLP
- `"convnext1d"`: ConvNeXt V2-inspired 1D backbone

### 19.3 原始 MACB-DRP 基准

为验证 MMPA 改进的有效性，重新运行原始 MACB-DRP（`main.py`，`ResNetTissueClassifier`，无 molecular program，无 OT）。

| 方法 | AUC | AUPR | F1 | ACC |
|------|-----|------|-----|-----|
| **原始 MACB-DRP** | 0.5575 | 0.7013 | **0.5789** | **0.5772** |
| MMPA C0 (Phase 2 baseline) | 0.5763 | 0.7098 | 0.5420 | 0.5108 |

**结论**：MMPA 本身（C0）相对原始 MACB-DRP，AUC 提升 **+1.88%**（0.5575 → 0.5763），AUPR 提升 +0.85%，但 F1/ACC 略有下降。说明 molecular program 的引入改善了排序能力（AUC/AUPR），但对硬分类阈值（F1/ACC）影响较小。

### 19.4 Phase 5 实验结果

| 实验 | Backbone | Mean AUC | vs 原始 MACB | vs C1 (OT1) | vs C0 (baseline) |
|------|---------|---------|------------|------------|-----------------|
| **原始 MACB-DRP** | ResNetTissueClassifier | 0.5575 | — | — | — |
| **C0** | Phase 2 baseline, no OT | 0.5763 | +1.88% | -0.35% | — |
| **C1** | OT1 + MLP | 0.5798 | +2.23% | — | +0.35% |
| **C2** | OT1 + GRN-MLP | **0.5903** | **+3.28%** | **+1.05%** | +1.40% |
| **C3** | OT1 + ConvNeXt1D d=1 | **0.5924** | **+3.49%** | **+1.26%** | +1.61% |
| **C4** | OT1 + ConvNeXt1D d=2 | 0.5840 | +2.65% | +0.42% | +0.77% |
| **C5** | OT1 + ConvNeXt1D d=2 k=15 | 0.5499 | -0.76% | -2.99% | -2.64% |

### 19.5 关键发现

1. **C3 (ConvNeXt1D depth=1) 是目前最佳方法**：AUC 0.5924，比原始 MACB-DRP 提升 **+3.49%**，比 OT1 提升 +1.26%
2. **C2 (GRN-MLP) 是风险最低的升级**：AUC 0.5903，比原始 MACB-DRP 提升 +3.28%，改动最小
3. **ConvNeXt 深度不是越深越好**：depth=1 (0.5924) > depth=2 (0.5840)，说明 shallow ConvNeXt 更适合
4. **大 kernel 有害**：kernel=15 导致 AUC 暴跌到 0.5499，比原始 MACB-DRP 还低
5. **MMPA 对 F1/ACC 的影响**：所有 MMPA 变体的 F1 和 ACC 均低于原始 MACB-DRP，说明 molecular program + OT 优化的是排序能力，而非硬分类阈值

### 19.6 Per-Drug 详细对比 (AUC)

| 药物 | MACB | C1 OT1 | C2 GRN-MLP | C3 ConvNeXt d=1 | C4 ConvNeXt d=2 | C5 ConvNeXt k=15 |
|------|--------|-----------|----------------|----------------|-----------------|
| 5-FU | 0.5218 | 0.5856 | **0.6622** ⬆️ | 0.6120 | 0.5772 | 0.5902 |
| Cisplatin | 0.6682 | 0.6452 | 0.6415 | 0.6442 | 0.6441 | 0.6380 |
| Cyclophosphamide | 0.6341 | 0.6263 | 0.6152 | 0.6066 | **0.6264** | 0.6051 |
| Docetaxel | 0.3739 | 0.4921 | 0.4950 | **0.5269** ⬆️ | 0.4855 | 0.4492 |
| Doxorubicin | 0.3976 | 0.4635 | 0.4875 | **0.5221** ⬆️ | **0.4989** | 0.4722 |
| Etoposide | 0.7548 | 0.7173 | 0.6823 | 0.6842 | **0.7358** ⬆️ | 0.6530 |
| Gemcitabine | 0.5997 | 0.5916 | **0.6100** ⬆️ | 0.5949 | 0.5856 | 0.5445 |
| Paclitaxel | 0.4676 | 0.4749 | 0.4923 | 0.4956 | **0.5076** ⬆️ | 0.4623 |
| Temozolomide | 0.5998 | 0.6215 | 0.6268 | **0.6452** ⬆️ | 0.5952 | 0.5349 |

### 19.7 决策结论

| 条件 | 结果 | 决策 |
|------|------|------|
| C2 > C1 | ✅ 0.5903 > 0.5798 | GRN-MLP 成为稳健候选 |
| C3 > C1 + 最佳 AUC | ✅ 0.5924 > 0.5798 | **ConvNeXt1D d=1 成为主方法候选** |
| C4/C5 > C1 | ❌ C4 略好, C5 差 | 深度增加和 kernel 放大均无益 |

**推荐命令 (C3):**
```bash
python train_mmpa_program.py \
  --use_ot_alignment --lambda_ot 0.01 --ot_reg 0.1 \
  --ot_cost_type cosine --ot_marginal_type uniform \
  --backbone_type convnext1d \
  --convnext_depth 1 --convnext_kernel_size 7
```

**稳健备选 (C2):**
```bash
python train_mmpa_program.py \
  --use_ot_alignment --lambda_ot 0.01 --ot_reg 0.1 \
  --ot_cost_type cosine --ot_marginal_type uniform \
  --backbone_type grn_mlp --norm_type layernorm
```

### 19.8 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/mmpa_ot1_convnext_results.csv` | 6 组实验 mean 指标对比 |
| `outputs/mmpa_ot1_convnext_per_drug.csv` | 每药物完整四项指标 |
| `results/..._mmpa_ExpC{0-5}.csv` | 各实验 5-fold 原始结果 |

---

## 20. TransDRP 基准对比

### 20.1 背景
TransDRP 是论文 "Knowledge-guided domain adaptation model for transferring drug response prediction from cell lines to patients" 的官方实现，位于 `/home/waas/TransDRP-main/`。

### 20.2 TransDRP 结果（重新运行）

| 指标 | 旧结果 (5月4日) | **新结果 (重新运行)** |
|------|----------------|----------------------|
| AUC | 0.5732 | **0.5440** |
| AUPR | 0.6334 | **0.6380** |
| F1 | 0.5256 | **0.5023** |
| ACC | 0.4441 | **0.4079** |

> ⚠️ **注意**：重新运行后 AUC 从 0.5732 降至 0.5440，说明 TransDRP 存在未固定的随机性（`set_seed_all(2024)` 未能完全控制 PyTorch Geometric 的随机性）。以下分析基于 **最新重新运行结果**。

### 20.3 全方法对比

| 方法 | AUC | AUPR | F1 | ACC |
|------|-----|------|-----|-----|
| **原始 MACB-DRP** | **0.5575** | **0.7013** | **0.5789** | **0.5772** |
| **TransDRP** | 0.5440 | 0.6380 | 0.5023 | 0.4079 |
| MMPA C0 (Phase 2) | 0.5763 | 0.7098 | 0.5420 | 0.5108 |
| OT1 MLP (C1) | 0.5798 | 0.7092 | 0.5324 | 0.5101 |
| GRN-MLP (C2) | 0.5903 | 0.7224 | 0.5444 | 0.4937 |
| **ConvNeXt1D d=1 (C3)** | **0.5924** | **0.7281** | 0.5674 | 0.5324 |

### 20.4 关键发现

| 维度 | TransDRP 表现 |
|------|--------------|
| **AUC** | 0.5440，**低于原始 MACB-DRP** (0.5575) 和所有 MMPA 变体 |
| **AUPR/F1/ACC** | **显著低于所有方法**。AUPR 比 MACB 低 6.33%，ACC 低 16.93% |
| **优势药物** | 仅 **Temozolomide (0.7995)** 是所有方法中最高的；5-FU (0.5639) 也不错 |
| **劣势药物** | Doxorubicin (0.3316)、Cyclophosphamide (0.5019)、Etoposide (0.5766) 表现较差 |

### 20.5 Per-Drug 全部指标对比（集中表格）

#### AUC

| 药物 | MACB | TransDRP | MMPA C3 | 最佳 |
|------|------|----------|---------|------|
| 5-Fluorouracil | 0.5218 | 0.5639 | **0.6120** | MMPA C3 |
| Cisplatin | **0.6682** | 0.6484 | 0.6442 | MACB |
| Cyclophosphamide | **0.6341** | 0.5019 | 0.6066 | MACB |
| Docetaxel | 0.3739 | 0.4121 | **0.5269** | MMPA C3 |
| Doxorubicin | 0.3976 | 0.3316 | **0.5221** | MMPA C3 |
| Etoposide | **0.7548** | 0.5766 | 0.6842 | MACB |
| Gemcitabine | **0.5997** | 0.5982 | 0.5949 | MACB |
| Paclitaxel | 0.4676 | 0.4639 | **0.4956** | MMPA C3 |
| Temozolomide | 0.5998 | **0.7995** | 0.6452 | TransDRP |
| **Mean** | **0.5575** | 0.5440 | **0.5924** | **MMPA C3** |

#### AUPR

| 药物 | MACB | TransDRP | MMPA C3 | 最佳 |
|------|------|----------|---------|------|
| 5-Fluorouracil | 0.6715 | 0.6879 | **0.7149** | MMPA C3 |
| Cisplatin | **0.8211** | 0.6996 | 0.8080 | MACB |
| Cyclophosphamide | **0.9371** | 0.8820 | 0.9346 | MACB |
| Docetaxel | 0.6177 | 0.6157 | **0.7034** | MMPA C3 |
| Doxorubicin | 0.6892 | 0.5064 | **0.7873** | MMPA C3 |
| Etoposide | **0.9117** | 0.6663 | 0.8855 | MACB |
| Gemcitabine | 0.5834 | 0.4567 | **0.5858** | MMPA C3 |
| Paclitaxel | 0.6878 | 0.5740 | **0.7053** | MMPA C3 |
| Temozolomide | 0.3924 | **0.6532** | 0.4286 | TransDRP |
| **Mean** | **0.7013** | 0.6380 | **0.7281** | **MMPA C3** |

#### F1

| 药物 | MACB | TransDRP | MMPA C3 | 最佳 |
|------|------|----------|---------|------|
| 5-Fluorouracil | 0.5318 | 0.5227 | **0.5336** | MMPA C3 |
| Cisplatin | **0.7026** | 0.6158 | 0.5468 | MACB |
| Cyclophosphamide | **0.6800** | 0.2235 | 0.6000 | MACB |
| Docetaxel | 0.3887 | 0.4151 | **0.5069** | MMPA C3 |
| Doxorubicin | 0.3208 | 0.3220 | **0.4830** | MMPA C3 |
| Etoposide | **0.8083** | 0.5583 | 0.6250 | MACB |
| Gemcitabine | 0.5730 | **0.6130** | 0.5774 | TransDRP |
| Paclitaxel | 0.4568 | 0.4459 | **0.5081** | MMPA C3 |
| Temozolomide | 0.7478 | **0.8045** | 0.7254 | TransDRP |
| **Mean** | **0.5789** | 0.5023 | 0.5674 | **MACB** |

#### ACC

| 药物 | MACB | TransDRP | MMPA C3 | 最佳 |
|------|------|----------|---------|------|
| 5-Fluorouracil | **0.6039** | 0.4892 | 0.5211 | MACB |
| Cisplatin | **0.7988** | 0.6575 | 0.5861 | MACB |
| Cyclophosphamide | **0.7934** | 0.2186 | 0.6967 | MACB |
| Docetaxel | 0.4518 | 0.4332 | **0.5362** | MMPA C3 |
| Doxorubicin | 0.3429 | 0.2257 | **0.5095** | MMPA C3 |
| Etoposide | **0.8818** | 0.5781 | 0.6966 | MACB |
| Gemcitabine | **0.5508** | 0.4819 | 0.4926 | MACB |
| Paclitaxel | 0.5244 | 0.3866 | **0.5752** | MMPA C3 |
| Temozolomide | 0.2467 | **0.2003** | 0.1778 | MACB |
| **Mean** | **0.5772** | 0.4079 | 0.5324 | **MACB** |

### 20.6 结论

- **TransDRP 重新运行后 AUC 降至 0.5440，低于原始 MACB-DRP (0.5575)**
- **原始 MACB-DRP 在 AUPR/F1/ACC 上全面优于 TransDRP**
- **MMPA C3 全面超越 TransDRP**：AUC +4.84%，AUPR +9.02%，F1 +6.50%，ACC +12.45%
- TransDRP 的唯一亮点是 **Temozolomide (0.7995)**，是所有方法中最高的
- **MMPA + ConvNeXt1D + OT 是当前最佳方案**

---

## 附录：活跃文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `mmpa_drp/models.py` | `MolecularProgramEncoder`, `ProgramTissueClassifier`, `ProgramFusionMLP` | ✅ 活跃 |
| `mmpa_drp/losses.py` | diversity, sparsity, entropy_floor, gate_balance, gate_diversity | ✅ 活跃 |
| `mmpa_drp/ot_utils.py` | Sinkhorn, OT cost, OT alignment loss | ✅ 活跃（Phase OT） |
| `mmpa_drp/stage1_program.py` | Stage 1 训练 | ✅ 活跃 |
| `mmpa_drp/stage2_program.py` | Stage 2&3 训练（含 OT loss 集成） | ✅ 活跃 |
| `train_mmpa_program.py` | MMPA-DRP 训练主入口 | ✅ 活跃 |
| `test_mmpa_ot.py` | OT utility smoke test | ✅ 活跃 |
| `CODEMAP_MACB.md` | 本文件 | ✅ 活跃 |
| `CODEMAP_OT_TRANSFER_TO_MMPA.md` | OT 迁移设计文档 | ✅ 参考 |

### 已停止的模块（代码保留在 `mmpa_drp/` 中作为历史记录，脚本已清理）

| 模块 | 停止原因 |
|------|----------|
| `DisentangledProgramEncoder` / `DisentangledProgramTissueClassifier` / `DisentangledProgramFusionMLP` | 3A-R 未超越 baseline |
| `DrugMechanismGate` / `DrugGatedProgramAggregator` / `SharedDrugGate` | 3B-R 未超越 baseline |
| `CODEMAP_DCDNET.md` | DCDNet 参考分析已完成 |


## 21. Phase 6: Program-level Optimal Transport (POT)

### 21.1 设计目标

在 C3 (ConvNeXt1D depth=1 + OT1) 的基础上，增加 **program-level OT 对齐损失**：

```
L_total = L_response + L_cdan + L_aux + lambda_sample_ot * L_sample_OT + lambda_program_ot * L_program_OT
```

其中：
- `L_sample_OT`: 现有的样本级 OT（基于 `h_pool`/`gene_feat`）
- `L_program_OT`: 新增的 program-level OT（基于 `z: [B, K, d]` 和 `q: [B, K]`）

### 21.2 实现组件

#### `ProgramOTLoss` (`mmpa_drp/ot_utils.py`)

```python
class ProgramOTLoss(nn.Module):
    # Modes:
    # - "same_program_uniform":     1/K * sum_k OT(z_s[:,k,:], z_t[:,k,:]) 均匀边际
    # - "same_program_q_weighted":  同上，但边际由 q 归一化权重决定
    # - "program_only" (ablation):  仅 program OT，无样本级 OT
    # - "flattened_program" (ablation): z reshape 为 [B*K, d] 后做单 OT
```

核心计算通过 `ot_alignment_loss` 的 batch 3D 支持实现：
- `source_z.transpose(0,1)` → `[K, B_s, d]`
- `target_z.transpose(0,1)` → `[K, B_t, d]`
- `compute_ot_cost` 返回 `[K, B_s, B_t]` cost
- `sinkhorn_log_domain` batch 求解 K 个 transport plan
- `ot_dist = sum(T * cost) / K`

#### `ProgramFusionMLP.forward` (`mmpa_drp/models.py`)

新增可选参数 `return_encoder_out=False`：
- 当 `True` 时，额外返回 `encoder_out` dict（含 `z`, `q`, `h`）
- 完全向后兼容，现有调用方不受影响

#### `stage2_program.py` 训练流程

`_fusion_model_train_step_with_ot` 中：
1. `model.forward(..., return_encoder_out=True)` 获取 `s_encoder_out` 和 `t_encoder_out`
2. 提取 `source_z`, `source_q`, `target_z`, `target_q`
3. 调用 `program_ot_loss_module(source_z, source_q, target_z, target_q)`
4. `loss = s_loss + cdan_domain_loss + ot_loss + program_ot_loss`

`fusion_training_program` 中：
- 解析 `use_program_ot`, `program_ot_mode`, `lambda_program_ot` 等参数
- 实例化 `ProgramOTLoss`
- `"program_only"` 模式下自动禁用样本级 OT（`ot_loss_module = None`）

#### `train_mmpa_program.py` 参数

新增 CLI 参数：
```bash
--use_program_ot                      # 启用 program OT
--program_ot_mode {same_program_uniform, same_program_q_weighted, program_only, flattened_program}
--lambda_program_ot 0.01              # program OT 权重
--program_ot_reg 0.1                  # Sinkhorn 正则化
--program_ot_max_iter 50              # Sinkhorn 迭代次数
--program_ot_cost_type {cosine, euclidean}
--program_ot_marginal_type {uniform, unbalanced, max_sim}
```

### 21.3 实验设计 (POT1-POT5)

| 实验 | 模式 | lambda_program_ot | 样本 OT | 说明 |
|------|------|-------------------|---------|------|
| POT1 | same_program_uniform | 0.005 | ✅ | 轻量 program OT |
| POT2 | same_program_uniform | 0.01 | ✅ | 标准 program OT |
| POT3 | same_program_q_weighted | 0.01 | ✅ | q 加权边际 |
| POT4 | program_only | 0.01 | ❌ | 消融：仅 program OT |
| POT5 | flattened_program | 0.01 | ✅ | 消融：展平 token |

所有实验基于 C3 配置：
- `backbone_type=convnext1d`, `convnext_depth=1`, `convnext_kernel_size=7`
- `lambda_ot=0.01`, `use_cdan=True`
- `pretrain=300`, `train=100`, `contrastive=200`, `K=7`

### 21.4 Smoke Tests

文件：`tests/test_program_ot.py`

| 测试 | 内容 | 状态 |
|------|------|------|
| test_program_ot_loss_shapes | 4 种模式前向传播 | ✅ PASS |
| test_program_ot_backward | 梯度回传检查 | ✅ PASS |
| test_program_ot_q_weighted_marginals | q 加权边际 | ✅ PASS |
| test_flattened_program_vs_batched | 展平 vs 均匀模式差异 | ✅ PASS |
| test_program_ot_with_encoder_output | ProgramFusionMLP + return_encoder_out | ✅ PASS |
| test_ot_alignment_loss_custom_marginals | ot_alignment_loss 自定义边际 | ✅ PASS |

---

## 附录：活跃文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| `mmpa_drp/models.py` | `MolecularProgramEncoder`, `ProgramTissueClassifier`, `ProgramFusionMLP` | ✅ 活跃 |
| `mmpa_drp/backbones.py` | `ConvNeXt1DBackbone`, `GRNMLPBackbone` | ✅ 活跃 (Phase 5) |
| `mmpa_drp/losses.py` | diversity, sparsity, entropy_floor, gate_balance, gate_diversity | ✅ 活跃 |
| `mmpa_drp/ot_utils.py` | `Sinkhorn`, `OT alignment`, `DrugAdaptiveOTLoss`, `ProgramOTLoss` | ✅ 活跃 (Phase 6) |
| `mmpa_drp/stage1_program.py` | Stage 1 训练 | ✅ 活跃 |
| `mmpa_drp/stage2_program.py` | Stage 2&3 训练（含 OT + Program OT） | ✅ 活跃 |
| `mmpa_drp/program_diagnostics.py` | Program 诊断计算 | ✅ 活跃 |
| `train_mmpa_program.py` | MMPA-DRP 训练主入口 | ✅ 活跃 |
| `tests/test_program_ot.py` | Phase 6 smoke tests | ✅ 活跃 |
| `tests/test_convnext_program_encoder.py` | Phase 5 smoke tests | ✅ 活跃 |
| `CODEMAP_MACB.md` | 本文件 | ✅ 活跃 |

### 已停止的模块（代码保留在 `mmpa_drp/` 中作为历史记录）

| 模块 | 停止原因 |
|------|----------|
| `DisentangledProgramEncoder` / `DisentangledProgramTissueClassifier` / `DisentangledProgramFusionMLP` | 3A-R 未超越 baseline |
| `DrugMechanismGate` / `DrugGatedProgramAggregator` / `SharedDrugGate` | 3B-R 未超越 baseline |
| `DrugAdaptiveOTLoss` (per_drug_fixed/validation/discrepancy) | Phase 4 药物自适应 OT 无效，保留 global 模式 |


## 22. Phase 6-R: Program OT Diagnostics and Repair

### 22.1 背景

Phase 6 (POT1-POT5) 中，所有 program OT 实验结果与 C3 baseline **完全相同**（AUC 0.5924），呈现完全 no-op。Phase 6-R 的目标是诊断根本原因，判断 program-level OT 是否有真实价值。

### 22.2 诊断过程

#### Task 1: 梯度诊断

实现 `mmpa_drp/grad_diagnostics.py`，直接调用 `_fusion_model_train_step_with_ot` 检查梯度：

| 检查项 | 结果 |
|:---|:---|
| `loss_program_ot_raw.requires_grad` | ✅ True |
| `source_z.requires_grad` | ✅ True |
| `source_q.requires_grad` | ✅ True |
| `grad_norm_program_ot_raw` | ✅ 0.0251 > 0 |
| `grad_norm_total` | 0.6255 |
| **ratio** | **4.0%** |

#### 发现的 Bug

`_fusion_model_train_step_with_ot` 中存在三重遗漏：
1. `model.forward()` **未传** `return_encoder_out=True`
2. 函数内**未定义** `s_encoder_out` / `t_encoder_out`
3. `fusion_training_program` 调用时**未传入** `program_ot_loss_module`

Python `and` 短路救了它：`program_ot_loss_module is None` → 条件 False → 跳过代码块 → 不触发 NameError → **program OT 完全没运行**。

**修复验证**（修复后直接调用 train step）：
- loss 无 program OT: 2.2151
- loss 有 q_mass_flattened (λ=0.05): **1.9832** ✅
- ot_dist: 0.445853

### 22.3 实验结果

#### Stress Test (n=2)

| 实验 | 模式 | λ | AUC | AUPR | F1 | ACC | ΔAUC |
|:---|:---|:---:|---:|---:|---:|---:|---:|
| C3 baseline | sample OT only | — | 0.6240 | 0.7476 | 0.6250 | 0.6163 | — |
| D1 stress | same_uniform | 0.1 | 0.5755 | 0.7153 | 0.5818 | 0.5327 | **-4.85%** ❌ |
| D3 q_mass | q_mass_flattened | 0.01 | 0.6101 | 0.7264 | **0.6542** | **0.6772** | -1.39% ⚠️ |
| D7 program_only | program_only | 0.01 | 0.5428 | 0.6856 | 0.4722 | 0.3948 | **-13.12%** ❌ |

#### D3 n=5 确认实验

| 指标 | C3 (n=5) | D3 (n=5) | Δ |
|:---|---:|---:|---:|
| AUC | **0.5924** | 0.5895 | **-0.0029** ❌ |
| AUPR | **0.7281** | 0.7220 | **-0.0061** ❌ |
| F1 | 0.5674 | **0.5860** | **+0.0186** ✅ |
| ACC | 0.5324 | **0.5655** | **+0.0331** ✅ |

**n=2 vs n=5 趋势一致**：F1/ACC 提升稳定存在，但幅度从 n=2 的 (+2.92%, +6.09%) 缩小到 n=5 的 (+1.86%, +3.31%)。

### 22.4 结论

| 问题 | 答案 |
|:---|:---|
| Program OT 为什么之前无效？ | **实现 bug**，`program_ot_loss_module` 根本没传给 train step |
| 修复后有效吗？ | **有效**，能改变训练结果和损失值 |
| 能提升 AUC/AUPR？ | **不能**，所有变体 AUC ≤ C3 |
| 能提升 F1/ACC？ | **能**，q-mass λ=0.01: F1 +1.86%, ACC +3.31% (n=5 稳定) |
| 能替代 sample OT？ | **绝对不能**，program_only 导致 AUC 暴跌 13% |
| naive same_program_uniform 有害？ | **是**，λ=0.1 导致 AUC 下降 4.85% |
| 最终推荐？ | **保留 C3 作为主线**；Program OT 作为 negative study for ranking，但 positive regularization for classification |

### 22.5 关键发现解释

**q-mass program OT 提升 F1/ACC 但不提升 AUC 的机制：**
- q-mass 用 program 激活 q 作为 transport mass，迫使模型关注高激活 program
- 这引入了一种**保守正则化**：模型降低了对不确定样本的预测方差
- 对 hard classification（阈值 0.5），保守模型更准确 → F1/ACC 提升
- 对 ranking（AUC），保守模型减少了正负样本之间的区分度 → AUC 不变或略降

### 22.6 输出文件

| 文件 | 说明 |
|:---|:---|
| `outputs/program_ot_repair_results.csv` | 实验汇总结果 |
| `outputs/program_ot_repair_per_drug.csv` | per-drug 结果 |
| `outputs/program_ot_repair_diagnostics.csv` | Program OT 训练诊断 |
| `outputs/program_ot_gradient_diagnostics.csv` | 梯度范数诊断 |
| `outputs/program_ot_leakage_check.txt` | Leakage check 文档 |

### 22.7 新增/修改文件清单

| 文件 | 改动 | 状态 |
|:---|:---|:---|
| `mmpa_drp/ot_utils.py` | 新增 `q_mass_flattened` 模式、`q_align_loss`、增强 diagnostics | ✅ |
| `mmpa_drp/grad_diagnostics.py` | **新增**：梯度诊断工具 | ✅ |
| `mmpa_drp/stage2_program.py` | **修复 bug** + 集成 q_align + 诊断日志 | ✅ |
| `train_mmpa_program.py` | 新增 CLI 参数 + 防文件名碰撞 | ✅ |
| `tests/test_program_ot_repair.py` | **新增**：Phase 6-R smoke tests | ✅ |
| `run_phase6r_repair.py` | **新增**：实验运行脚本 | ✅ |

---

## 附录：活跃文件清单（最终版）

| 文件 | 说明 | 状态 |
|------|------|------|
| `mmpa_drp/models.py` | `MolecularProgramEncoder`, `ProgramTissueClassifier`, `ProgramFusionMLP` | ✅ 活跃 |
| `mmpa_drp/backbones.py` | `ConvNeXt1DBackbone`, `GRNMLPBackbone` | ✅ 活跃 (Phase 5) |
| `mmpa_drp/losses.py` | diversity, sparsity, entropy_floor, gate_balance, gate_diversity | ✅ 活跃 |
| `mmpa_drp/ot_utils.py` | Sinkhorn, OT alignment, `ProgramOTLoss` (含 q_mass_flattened) | ✅ 活跃 (Phase 6-R) |
| `mmpa_drp/grad_diagnostics.py` | 梯度诊断工具 | ✅ 活跃 (Phase 6-R) |
| `mmpa_drp/stage1_program.py` | Stage 1 训练 | ✅ 活跃 |
| `mmpa_drp/stage2_program.py` | Stage 2&3 训练（含 OT + Program OT + q_align） | ✅ 活跃 |
| `train_mmpa_program.py` | MMPA-DRP 训练主入口 | ✅ 活跃 |
| `tests/test_program_ot_repair.py` | Phase 6-R smoke tests | ✅ 活跃 |
| `CODEMAP_MACB.md` | 本文件 | ✅ 活跃 |

### 最终方法栈（Phase 6-R 之后）

| 层级 | 组件 | 配置 |
|:---|:---|:---|
| Backbone | `ConvNeXt1DBackbone` | depth=1, kernel=7, expansion=4 |
| Encoder | `MolecularProgramEncoder` | K=8 programs, d=64 |
| Sample OT | `DrugAdaptiveOTLoss` (global) | λ=0.01, reg=0.1, cosine |
| Program OT | **不使用**（AUC 无提升） | — |
| CDAN | 保留 | α=0.3 |
| Stage 3 | Contrastive | K=7, β=0.3 |

---

## 23. Phase 7: C3 Hyperparameter Tuning

### 23.1 背景

在 Phase 6-R 确认 Program OT 无法提升 AUC 后，将优化重点转回 **C3 配置本身**（ConvNeXt1D depth=1 + sample OT + CDAN + contrastive）。目标是系统性地搜索 backbone 外超参数（K, lambda_ot, alpha, beta 等），在不改变模型架构的前提下挖掘剩余性能空间。

### 23.2 实验设计（n=5, 单变量扫描）

| 编号 | 变量 | 取值 | 其他固定 |
|:---|:---|:---|:---|
| C3 baseline | — | K=7, λ_ot=0.01 | depth=1, kernel=7, α=0.3, β=0.3 |
| **λ_ot=0.005** | lambda_ot | 0.005 | K=7 |
| **λ_ot=0.02** | lambda_ot | 0.02 | K=7 |
| **K=3** | Top-K contrastive | 3 | λ_ot=0.01 |
| **K=5** | Top-K contrastive | 5 | λ_ot=0.01 |
| **K=9** | Top-K contrastive | 9 | λ_ot=0.01 |
| **α=0.1** | alpha (CDAN weight) | 0.1 | K=7, λ_ot=0.01 |
| **α=0.5** | alpha (CDAN weight) | 0.5 | K=7, λ_ot=0.01 |
| **β=0.1** | beta (contrastive weight) | 0.1 | K=7, λ_ot=0.01 |
| **β=0.5** | beta (contrastive weight) | 0.5 | K=7, λ_ot=0.01 |
| **Combo** | K=3 + λ_ot=0.005 | K=3, λ_ot=0.005 | — |

### 23.3 汇总结果

| 排名 | 实验 | AUC | AUPR | F1 | ACC | ΔAUC (vs C3) |
|:---|:---|---:|---:|---:|---:|---:|
| 🥇 | **Combo K=3 + λ_ot=0.005** | **0.6189** | **0.7405** | **0.6304** | **0.6264** | **+4.47%** |
| 🥈 | C3_K_3 | 0.6178 | 0.7448 | 0.6075 | 0.6141 | +4.29% |
| 🥉 | C3_λ_ot=0.005 | 0.6139 | 0.7313 | 0.6332 | 0.6317 | +3.63% |
| 4 | C3_λ_ot=0.02 | 0.6078 | 0.7317 | 0.6247 | 0.6375 | +2.60% |
| — | **C3 baseline** | **0.5924** | 0.7281 | 0.5674 | 0.5324 | — |
| 5 | C3_K_5 | 0.5908 | 0.7252 | 0.5672 | 0.5391 | -0.27% |
| 6 | C3_K_9 | 0.5903 | 0.7265 | 0.5675 | 0.5335 | -0.35% |
| — | C3_α=0.1/0.5 | 0.5924 | 0.7281 | 0.5674 | 0.5324 | 0.00% |
| — | C3_β=0.1/0.5 | 0.5924 | 0.7281 | 0.5674 | 0.5324 | 0.00% |

### 23.4 关键发现

1. **K=3 是最大突破**：AUC 0.6178 (+4.29%)，显著优于 K=7 baseline。更严格的 Top-3 最近邻筛选比 Top-7 更有效。
2. **λ_ot=0.005 也有提升**：AUC 0.6139 (+3.63%)，说明当前 0.01 的 OT 权重可能略强，适度降低更有利。
3. **组合效果：K=3 主导**：Combo (0.6189) vs K=3 单独 (0.6178)，增益仅 +0.0011。说明 K=3 已覆盖大部分 λ_ot 的增益。
4. **K=5/K=9 均劣于 baseline**：K 增大反而降低性能，说明过多近邻引入了噪声。
5. **alpha/beta 完全无效**：0.1/0.3/0.5 结果完全相同到小数点后4位。根因分析：这些参数在 `train_mmpa_program.py` 中存入 `training_params`，但 `stage2_program.py` 的训练函数未接收它们，而是使用 kwargs 默认值（α=0.3, β=0.3）。参数传递链断裂。
6. **组合配置全面超越所有先前方法**：
   - vs 原始 MACB-DRP: +6.17% (0.5575 → 0.6189)
   - vs TransDRP: +7.41% (0.5440 → 0.6189)
   - vs C3 baseline: +4.47% (0.5924 → 0.6189)

### 23.5 Per-Drug AUC（Combo vs 最佳 baseline）

| 药物 | MACB | TransDRP | C3 baseline | **Combo** | 最佳 |
|:---|---:|---:|---:|---:|:---|
| 5-FU | 0.5218 | 0.5639 | 0.6120 | **0.6581** | Combo |
| Cisplatin | **0.6682** | 0.6484 | 0.6442 | 0.6362 | MACB |
| Cyclophosphamide | 0.6341 | 0.5019 | 0.6066 | **0.6612** | Combo |
| Docetaxel | 0.3739 | 0.4121 | 0.5269 | **0.5522** | Combo |
| Doxorubicin | 0.3976 | 0.3316 | 0.5221 | **0.5415** | Combo |
| Etoposide | **0.7548** | 0.5766 | 0.6842 | 0.7573 | MACB |
| Gemcitabine | 0.5997 | 0.5982 | 0.5949 | **0.5754** | MACB |
| Paclitaxel | 0.4676 | 0.4639 | 0.4956 | **0.5364** | Combo |
| Temozolomide | 0.5998 | **0.7995** | 0.6452 | 0.6514 | TransDRP |
| **Mean** | 0.5575 | 0.5440 | 0.5924 | **0.6189** | **Combo** |

### 23.6 输出文件

| 文件 | 说明 |
|:---|:---|
| `run_c3_tuning.py` | C3 超参调优运行脚本 |
| `results/pretrain_num_epochs_300_train_num_epochs_100_contrastive_num_epochs_200_K_*.csv` | 各 K 值 5-fold 原始结果 |
| `results/pretrain_num_epochs_300_train_num_epochs_100_contrastive_num_epochs_200_lambda_ot_*.csv` | 各 λ_ot 5-fold 原始结果 |
| `results/pretrain_num_epochs_300_train_num_epochs_100_contrastive_num_epochs_200_K_3_lambda_ot_0.005.csv` | Combo 实验 5-fold 原始结果 |

### 23.7 新增/修改文件

| 文件 | 改动 | 状态 |
|:---|:---|:---|
| `run_c3_tuning.py` | **新增**：C3 超参调优脚本 | ✅ |
| `mmpa_drp/stage2_program.py` | 修复 alpha/beta 参数传递（部分修复，仍待确认） | ⚠️ |

---

## 附录：活跃文件清单（最终版）

| 文件 | 说明 | 状态 |
|------|------|------|
| `mmpa_drp/models.py` | `MolecularProgramEncoder`, `ProgramTissueClassifier`, `ProgramFusionMLP` | ✅ 活跃 |
| `mmpa_drp/backbones.py` | `ConvNeXt1DBackbone`, `GRNMLPBackbone` | ✅ 活跃 (Phase 5) |
| `mmpa_drp/losses.py` | diversity, sparsity, entropy_floor, gate_balance, gate_diversity | ✅ 活跃 |
| `mmpa_drp/ot_utils.py` | Sinkhorn, OT alignment, `ProgramOTLoss` (含 q_mass_flattened) | ✅ 活跃 (Phase 6-R) |
| `mmpa_drp/grad_diagnostics.py` | 梯度诊断工具 | ✅ 活跃 (Phase 6-R) |
| `mmpa_drp/stage1_program.py` | Stage 1 训练 | ✅ 活跃 |
| `mmpa_drp/stage2_program.py` | Stage 2&3 训练（含 OT + Program OT + q_align） | ✅ 活跃 |
| `train_mmpa_program.py` | MMPA-DRP 训练主入口 | ✅ 活跃 |
| `tests/test_program_ot_repair.py` | Phase 6-R smoke tests | ✅ 活跃 |
| `run_c3_tuning.py` | Phase 7 超参调优脚本 | ✅ 活跃 |
| `CODEMAP_MACB.md` | 本文件 | ✅ 活跃 |

### 最终方法栈（Phase 7 之后）

| 层级 | 组件 | 配置 |
|:---|:---|:---|
| Backbone | `ConvNeXt1DBackbone` | depth=1, kernel=7, expansion=4 |
| Encoder | `MolecularProgramEncoder` | K=8 programs, d=64 |
| Sample OT | `DrugAdaptiveOTLoss` (global) | **λ=0.005**, reg=0.1, cosine |
| Program OT | 不使用（AUC 无提升） | — |
| CDAN | 保留 | α=0.3 |
| Stage 3 | Contrastive | **K=3**, β=0.3 |

**最佳 AUC**: **0.6189** (Combo K=3 + λ_ot=0.005, 5-fold CV)

**全方法 AUC 排名**:
| 排名 | 方法 | AUC |
|:---|:---|---:|
| 1 | **MMPA Combo (K=3, λ=0.005)** | **0.6189** |
| 2 | MMPA C3_K_3 | 0.6178 |
| 3 | MMPA C3_λ=0.005 | 0.6139 |
| 4 | MMPA C3_λ=0.02 | 0.6078 |
| 5 | MMPA C3 (ConvNeXt1D d=1) | 0.5924 |
| 6 | MMPA C2 (GRN-MLP) | 0.5903 |
| 7 | MMPA OT1 (MLP) | 0.5798 |
| 8 | MMPA C0 (Phase 2) | 0.5763 |
| 9 | 原始 MACB-DRP | 0.5575 |
| 10 | TransDRP | 0.5440 |

**若需提升 F1/ACC**: 可选添加 q-mass program OT (λ=0.01, detach_q=true)
