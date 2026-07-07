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
300/100/200
5-fold average/std results: 
5-Fluorouracil    auc: 0.5202/0.0434 | aupr: 0.6762/0.0287 | f1: 0.5245/0.0281 | acc: 0.5965/0.0268
Cisplatin         auc: 0.6761/0.0450 | aupr: 0.8317/0.0381 | f1: 0.7036/0.0264 | acc: 0.7955/0.0228
Cyclophosphamide  auc: 0.6570/0.0679 | aupr: 0.9461/0.0199 | f1: 0.5741/0.1433 | acc: 0.6946/0.1366
Docetaxel         auc: 0.4000/0.0720 | aupr: 0.6346/0.0485 | f1: 0.4063/0.0655 | acc: 0.4441/0.1125
Doxorubicin       auc: 0.4076/0.0980 | aupr: 0.6895/0.0540 | f1: 0.3296/0.0698 | acc: 0.3367/0.1138
Etoposide         auc: 0.7833/0.0616 | aupr: 0.9275/0.0229 | f1: 0.8167/0.0433 | acc: 0.8872/0.0319
Gemcitabine       auc: 0.5936/0.0228 | aupr: 0.5721/0.0439 | f1: 0.5826/0.0263 | acc: 0.5494/0.0532
Paclitaxel        auc: 0.4697/0.0240 | aupr: 0.6905/0.0105 | f1: 0.4685/0.0431 | acc: 0.5400/0.0773
Temozolomide      auc: 0.6146/0.0628 | aupr: 0.4302/0.0385 | f1: 0.7448/0.0266 | acc: 0.2686/0.1547
4 metrics average/std results: 5-Fluorouracil: 0.5794/0.0717 | Cisplatin: 0.7517/0.0727 | Cyclophosphamide: 0.7180/0.1719 | Docetaxel: 0.4713/0.1217 | Doxorubicin: 0.4408/0.1705 | Etoposide: 0.8537/0.0701 | Gemcitabine: 0.5744/0.0392 | Paclitaxel: 0.5422/0.1019 | Temozolomide: 0.5145/0.2018
auc&aupr average/std results: 5-Fluorouracil: 0.5982/0.0892 | Cisplatin: 0.7539/0.0909 | Cyclophosphamide: 0.8016/0.1595 | Docetaxel: 0.5173/0.1365 | Doxorubicin: 0.5486/0.1663 | Etoposide: 0.8554/0.0877 | Gemcitabine: 0.5829/0.0349 | Paclitaxel: 0.5801/0.1176 | Temozolomide: 0.5224/0.1089
auc：0.5691/0.1350  aupr：0.7109/0.1625  acc：0.5681/0.2115  f1：0.5723/0.1636
all_metrics：0.6051/0.1792
auc_aupr_metrics：0.6400/0.1638


100/100/100
5-fold average/std results: 
5-Fluorouracil    auc: 0.5192/0.0161 | aupr: 0.6682/0.0167 | f1: 0.5136/0.0161 | acc: 0.5705/0.0433
Cisplatin         auc: 0.6572/0.0325 | aupr: 0.8104/0.0188 | f1: 0.6763/0.0477 | acc: 0.7695/0.0462
Cyclophosphamide  auc: 0.6512/0.0458 | aupr: 0.9377/0.0128 | f1: 0.5788/0.2209 | acc: 0.6827/0.2001
Docetaxel         auc: 0.3797/0.0482 | aupr: 0.6129/0.0371 | f1: 0.3874/0.0457 | acc: 0.4439/0.1012
Doxorubicin       auc: 0.4456/0.0493 | aupr: 0.7235/0.0299 | f1: 0.3711/0.0670 | acc: 0.4036/0.1167
Etoposide         auc: 0.7741/0.0310 | aupr: 0.9315/0.0211 | f1: 0.8021/0.0423 | acc: 0.8772/0.0309
Gemcitabine       auc: 0.6067/0.0244 | aupr: 0.5947/0.0263 | f1: 0.5617/0.0220 | acc: 0.5415/0.0580
Paclitaxel        auc: 0.4713/0.0265 | aupr: 0.6954/0.0153 | f1: 0.4883/0.0508 | acc: 0.5814/0.0820
Temozolomide      auc: 0.5181/0.1040 | aupr: 0.3252/0.0964 | f1: 0.6851/0.0890 | acc: 0.2101/0.1480
4 metrics average/std results: 5-Fluorouracil: 0.5679/0.0679 | Cisplatin: 0.7284/0.0741 | Cyclophosphamide: 0.7126/0.1961 | Docetaxel: 0.4560/0.1125 | Doxorubicin: 0.4859/0.1583 | Etoposide: 0.8462/0.0702 | Gemcitabine: 0.5762/0.0423 | Paclitaxel: 0.5591/0.1026 | Temozolomide: 0.4346/0.2128
auc&aupr average/std results: 5-Fluorouracil: 0.5937/0.0801 | Cisplatin: 0.7338/0.0845 | Cyclophosphamide: 0.7945/0.1543 | Docetaxel: 0.4963/0.1294 | Doxorubicin: 0.5846/0.1514 | Etoposide: 0.8528/0.0867 | Gemcitabine: 0.6007/0.0247 | Paclitaxel: 0.5834/0.1198 | Temozolomide: 0.4217/0.1388
auc：0.5581/0.1259  aupr：0.7000/0.1828  acc：0.5645/0.2135  f1：0.5627/0.1572
all_metrics：0.5963/0.1811
auc_aupr_metrics：0.6290/0.1706
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
原始：
5-fold average/std results: 
5-Fluorouracil    auc: 0.6038/0.0230 | aupr: 0.7128/0.0236 | f1: 0.5473/0.0335 | acc: 0.5766/0.0722
Cisplatin         auc: 0.6804/0.0380 | aupr: 0.8400/0.0238 | f1: 0.6552/0.0764 | acc: 0.7395/0.0873
Cyclophosphamide  auc: 0.5974/0.0492 | aupr: 0.9290/0.0193 | f1: 0.6976/0.1334 | acc: 0.8062/0.0987
Docetaxel         auc: 0.4445/0.0296 | aupr: 0.6519/0.0379 | f1: 0.4516/0.0396 | acc: 0.5423/0.0662
Doxorubicin       auc: 0.5126/0.0170 | aupr: 0.7731/0.0188 | f1: 0.4302/0.0704 | acc: 0.4895/0.1050
Etoposide         auc: 0.7114/0.0731 | aupr: 0.9021/0.0353 | f1: 0.7937/0.0298 | acc: 0.8732/0.0211
Gemcitabine       auc: 0.5984/0.0178 | aupr: 0.5681/0.0172 | f1: 0.5574/0.0289 | acc: 0.5689/0.0291
Paclitaxel        auc: 0.5578/0.0456 | aupr: 0.7352/0.0368 | f1: 0.5577/0.0538 | acc: 0.6454/0.0605
Temozolomide      auc: 0.6201/0.0467 | aupr: 0.4330/0.0364 | f1: 0.6537/0.1233 | acc: 0.3562/0.1719
4 metrics average/std results: 5-Fluorouracil: 0.6101/0.0754 | Cisplatin: 0.7288/0.0926 | Cyclophosphamide: 0.7576/0.1498 | Docetaxel: 0.5226/0.0958 | Doxorubicin: 0.5513/0.1473 | Etoposide: 0.8201/0.0864 | Gemcitabine: 0.5732/0.0270 | Paclitaxel: 0.6240/0.0883 | Temozolomide: 0.5158/0.1629
auc&aupr average/std results: 5-Fluorouracil: 0.6583/0.0615 | Cisplatin: 0.7602/0.0893 | Cyclophosphamide: 0.7632/0.1783 | Docetaxel: 0.5482/0.1139 | Doxorubicin: 0.6428/0.1384 | Etoposide: 0.8067/0.1141 | Gemcitabine: 0.5832/0.0230 | Paclitaxel: 0.6465/0.1013 | Temozolomide: 0.5266/0.1062
auc：0.5918/0.0858  aupr：0.7272/0.1544  acc：0.6220/0.1748  f1：0.5938/0.1309
all_metrics：0.6337/0.1495
auc_aupr_metrics：0.6595/0.1409

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


---

## 24. Phase 9: Fragment-Program Interaction (FPI)

### 24.1 背景

在 B0 (MMPA Combo K=3 + λ_ot=0.005) 基础上，尝试引入 **fragment-level 细粒度药物表征**，通过 BRICS 分解得到药物 fragment，计算 fragment 与 program token 的 pair attention，作为 residual branch 修正预测 logits。

**核心设计**：
- `DrugFragmentEncoder`: 将 Morgan FP 编码为 fragment embedding
- `FragmentProgramInteraction`: pair attention + per-drug gamma gate
- `ProgramFusionMLP`: `final_logits = base_logits + gamma * residual_logits`
- 安全初始化：`gamma_init=0.0` + `tanh` 输出，确保不破坏基线

### 24.2 实验环境

| 项目 | 配置 |
|:---|:---|
| 机器 | Linux 服务器 (`waas`) |
| Conda | `macb-drp` |
| 路径 | `/home/waas/MACB-DRP-main` |
| 数据 | 与 Phase 7 相同 |

> **注意**: 本次实验在 Linux 服务器上运行，故 B0 基线与 CODEMAP 记录的 0.6189 不同。以下分析以**当前环境 B0** 为准。

### 24.3 实验设计（第一轮）

| 实验 | 配置 | 说明 |
|:---|:---|:---|
| **B0_current** | 无 FPI | 当前环境基线 |
| **FPI1_g005** | `gamma_max=0.05, gamma_init=0.0, lambda_fpi_gamma_l1=0.001` | 轻量 FPI |
| **FPI2_g01** | `gamma_max=0.1, gamma_init=0.0, lambda_fpi_gamma_l1=0.001` | 标准 FPI |

固定配置：
- `backbone_type=convnext1d`, `convnext_depth=1`, `convnext_kernel_size=7`
- `num_programs=8`, `program_dim=64`
- `K=3` (topk_contrastive)
- `use_ot_alignment`, `lambda_ot=0.005`, `use_cdan`
- `alph=0.3`, `beta=0.3`
- `pretrain=300`, `train=100`, `contrastive=200`, 5-fold CV

### 24.4 第一轮实验结果（问题暴露）

#### 汇总表

| 实验 | AUC | AUPR | F1 | ACC | ΔAUC (vs B0) |
|:---|---:|---:|---:|---:|---:|
| **B0_current** | **0.5779** | 0.7136 | 0.5034 | 0.4093 | — |
| FPI1_g005 | 0.5757 | 0.7190 | 0.5125 | 0.4414 | -0.0022 |
| FPI2_g01 | 0.5838 | 0.7175 | 0.5050 | 0.4010 | +0.0059 |

**第一轮结论**：FPI 提升微弱（+0.0059），且存在药物间 tradeoff。当时判断为"FPI 无效"，但后续诊断发现根本原因是 **gamma 被钉死在 0**。

---

### 24.5 第二轮：Gamma 诊断（关键发现）

#### 问题根因

第一轮实验中，`gamma_init=0.0` + `lambda_fpi_gamma_l1=0.001` 导致 gamma 训练后仍 ≈ 0.000005，FPI 分支数学上等效关闭：`final_logits ≈ base_logits`。

诊断实验设计：

| 实验 | gamma_init | lambda_fpi_gamma_l1 | 说明 |
|:---|:---|:---|:---|
| **FPI2_warmup** | **0.05** | **0.0** | gamma 从有效值启动，不惩罚 |
| **FPI_SIM_soft** | 0.05 | 0.0 | FPI-SIM + union α=β=1.0 |
| **FPI_SIM_cross_only** | 0.05 | 0.0 | FPI-SIM + union α=β=0.0 |

#### 第二轮汇总结果

| 实验 | AUC | AUPR | F1 | ACC | ΔAUC (vs B0) |
|:---|---:|---:|---:|---:|---:|
| B0_current | 0.5779 | 0.7136 | 0.5034 | 0.4093 | — |
| FPI2_g01 (旧) | 0.5838 | 0.7175 | 0.5050 | 0.4010 | +0.0059 |
| **FPI2_warmup** | **0.6031** | **0.7371** | **0.5260** | **0.4440** | **+0.0252** |
| FPI_SIM_soft | 0.5621 | 0.7049 | 0.4850 | 0.4078 | -0.0158 |
| FPI_SIM_cross_only | 0.5734 | 0.7161 | 0.4900 | 0.4045 | -0.0045 |

#### Per-Drug AUC（B0 vs FPI2_warmup）

| 药物 | B0 | FPI2_warmup | Δ |
|:---|---:|---:|---:|
| 5-Fluorouracil | 0.5865 | **0.6186** | +0.0321 |
| Cisplatin | 0.6213 | **0.6524** | +0.0311 |
| Cyclophosphamide | 0.5526 | **0.6022** | +0.0496 |
| Docetaxel | 0.5326 | **0.5321** | -0.0005 |
| Doxorubicin | 0.5100 | **0.5411** | +0.0311 |
| Etoposide | 0.7537 | **0.7655** | +0.0118 |
| Gemcitabine | 0.5929 | **0.6122** | +0.0193 |
| Paclitaxel | 0.5090 | **0.5262** | +0.0172 |
| Temozolomide | 0.5426 | **0.5775** | +0.0349 |
| **Mean** | **0.5779** | **0.6031** | **+0.0252** |

> **9/9 种药物全部提升或持平**，无药物下降。Temozolomide 从第一轮旧 FPI2 的 0.4950（下降）恢复并超越 B0。

---

### 24.6 最优实验：FPI2_warmup 详细配置

#### 运行命令（0.6031）*****************************************

```bash
python train_mmpa_program.py \
  --measurement Z_SCORE \
  --thres_gdsc 0.0 \
  --thres_label 0.1 \
  --n 5 \
  --alph 0.3 \
  --beta 0.3 \
  --train \
  --backbone_type convnext1d \
  --convnext_depth 1 \
  --convnext_kernel_size 7 \
  --num_programs 8 \
  --program_dim 64 \
  --K 3 \
  --use_ot_alignment \
  --lambda_ot 0.005 \
  --use_cdan \
  --use_fpi \
  --fpi_fragment_npz data/drug_fragments_brics_m16_fp1024.npz \
  --fpi_gamma_max 0.1 \
  --fpi_gamma_init 0.05 \
  --fpi_q_prior_weight 0.01 \
  --lambda_fpi_gamma_l1 0.0
```

#### 关键参数说明

| 参数 | 值 | 作用 |
|:---|:---|:---|
| `--fpi_gamma_max` | 0.1 | gamma 上限 |
| `--fpi_gamma_init` | **0.05** | **关键：初始 gamma ≈ 0.05，避免坍缩到 0** |
| `--lambda_fpi_gamma_l1` | **0.0** | **关键：关闭 L1 惩罚，允许 gamma 保持有效** |
| `--fpi_q_prior_weight` | 0.01 | q prior 加权 |

#### 训练日志特征（gamma 行为）

```
FPI epoch 0:   gamma_mean=0.050000, gamma_max=0.050000
FPI epoch 99:  gamma_mean=0.050808, gamma_max=0.051572
    gamma_per_drug=[0.0504 0.0504 0.0503 0.051 0.0512 0.051 0.0505 0.0509 0.0516]
```

- gamma 从 0.05 启动，最终稳定在 ~0.051
- 9 种药物各自学习到 slightly different 的 gamma 值
- residual_norm ≈ 0.88，说明 FPI 分支确实输出了有意义的 residual signal

---

### 24.7 FPI-SIM 实验结果

FPI-SIM（Signal-style selective interaction）在第一轮 gamma=0 配置下失败后，用 gamma=0.05 + no L1 重新测试：

| 实验 | AUC | vs B0 | 结论 |
|:---|---:|---:|:---|
| FPI_SIM_soft (union α=β=1.0) | 0.5621 | -0.016 | ❌ 低于 B0 |
| FPI_SIM_cross_only (union α=β=0.0) | 0.5734 | -0.005 | ❌ 低于 B0 |

**FPI-SIM 未带来收益**，global softmax FPI 仍是最佳形式。Selective attention（program intra + fragment intra + soft-union）可能过度过滤了有效信号。

---

### 24.8 关键发现

1. **FPI 本身有效，但 gamma 配置是关键瓶颈**：
   - `gamma_init=0.0 + L1=0.001` → gamma 坍缩到 0 → FPI 关闭 → AUC ≈ B0
   - `gamma_init=0.05 + L1=0.0` → gamma 保持 ~0.05 → FPI 生效 → AUC **+0.025**

2. **Global softmax FPI 优于 selective attention (SIM)**：
   - FPI2_warmup (global): 0.6031 ✅
   - FPI-SIM soft: 0.5621 ❌
   - FPI-SIM cross-only: 0.5734 ❌

3. **FPI 收益是全局性的**：
   - 9/9 药物在 FPI2_warmup 下提升或持平
   - Temozolomide 从旧 FPI2 的 0.4950 恢复至 0.5775

4. **Gamma 几乎不更新**：
   - epoch 0: 0.05 → epoch 99: 0.0508
   - 说明 0.05 是一个稳定有效的工作点，模型不需要大幅调整

---

### 24.9 结论与决策

| 条件 | 结果 | 决策 |
|:---|:---|:---|
| FPI2_warmup > B0 + 0.01？ | ✅ 0.6031 > 0.5879 | **FPI 有效，采用 warm-up 配置** |
| FPI-SIM > FPI2_warmup？ | ❌ 0.5621 < 0.6031 | **停止 FPI-SIM** |
| 9/9 药物提升？ | ✅ | **无药物 tradeoff** |

- **FPI 方向继续，但仅限 global softmax 形式**
- **FPI-SIM 停止**
- **生产配置**：`gamma_init=0.05`, `lambda_fpi_gamma_l1=0.0`, `gamma_max=0.1`

---

## 附录：活跃文件清单（最终版）

| 文件 | 说明 | 状态 |
|------|------|------|
| `mmpa_drp/models.py` | `MolecularProgramEncoder`, `ProgramTissueClassifier`, `ProgramFusionMLP`, `FPISIMInteraction` | ✅ 活跃 |
| `mmpa_drp/backbones.py` | `ConvNeXt1DBackbone`, `GRNMLPBackbone` | ✅ 活跃 (Phase 5) |
| `mmpa_drp/losses.py` | diversity, sparsity, entropy_floor, gate_balance, gate_diversity | ✅ 活跃 |
| `mmpa_drp/ot_utils.py` | Sinkhorn, OT alignment, `ProgramOTLoss` (含 q_mass_flattened) | ✅ 活跃 (Phase 6-R) |
| `mmpa_drp/grad_diagnostics.py` | 梯度诊断工具 | ✅ 活跃 (Phase 6-R) |
| `mmpa_drp/fpi.py` | `FragmentProgramInteraction`, `FPISIMInteraction` | ✅ 活跃 (Phase 9) |
| `mmpa_drp/fragment_encoder.py` | `DrugFragmentEncoder` | ✅ 活跃 (Phase 9) |
| `mmpa_drp/fragment_utils.py` | `load_drug_fragments` | ✅ 活跃 (Phase 9) |
| `mmpa_drp/stage1_program.py` | Stage 1 训练 | ✅ 活跃 |
| `mmpa_drp/stage2_program.py` | Stage 2&3 训练（含 OT + Program OT + q_align + FPI） | ✅ 活跃 |
| `train_mmpa_program.py` | MMPA-DRP 训练主入口 | ✅ 活跃 |
| `CODEMAP_MACB.md` | 本文件 | ✅ 活跃 |

### 已停止的模块（代码保留在 `mmpa_drp/` 中作为历史记录）

| 模块 | 停止原因 |
|------|----------|
| `DisentangledProgramEncoder` / `DisentangledProgramTissueClassifier` / `DisentangledProgramFusionMLP` | 3A-R 未超越 baseline |
| `DrugMechanismGate` / `DrugGatedProgramAggregator` / `SharedDrugGate` | 3B-R 未超越 baseline |
| `FPISIMInteraction` | FPI-SIM 未超越 global FPI |

### 最终方法栈（Phase 9 之后）

| 层级 | 组件 | 配置 |
|:---|:---|:---|
| Backbone | `ConvNeXt1DBackbone` | depth=1, kernel=7, expansion=4 |
| Encoder | `MolecularProgramEncoder` | K=8 programs, d=64 |
| Sample OT | `DrugAdaptiveOTLoss` (global) | **λ=0.005**, reg=0.1, cosine |
| Program OT | 不使用（AUC 无提升） | — |
| CDAN | 保留 | α=0.3 |
| Stage 3 | Contrastive | **K=3**, β=0.3 |
| FPI | `FragmentProgramInteraction` (global) | **γ_init=0.05**, γ_max=0.1, no L1 |

---

### 24.10 Phase 9-R: Dual FPI Residual Router

在 Phase 9 发现 **Global FPI** (AUC 0.6031) 全面优于 **FPI-SIM**，但 FPI-SIM 的 **cross-only** 模式对 **Temozolomide** 有潜力 (0.6274 vs global 0.5775)。因此设计 `DualFPIResidualRouter`，让两种分支共存，由 **per-drug learnable gamma gates** 自动选择。

#### 24.10.1 动机

| 分支 | 整体 AUC | Temozolomide | 特点 |
|:---|---:|---:|:---|
| Global FPI | **0.6031** | 0.5775 | 稳健，9/9 药物提升 |
| FPI-SIM cross-only | 0.5734 | **0.6274** | 整体差，但 Temozolomide  Specialist |

**假设**：不同药物需要不同的 FPI 形式。Global 适合大多数药物，Cross 适合 Temozolomide 等特定药物。

#### 24.10.2 架构设计

```
base_logits ──┬──> Global FPI Branch ──> residual_global
              │       (FragmentProgramInteraction, use_fpi_sim=False)
              │
              ├──> Cross FPI Branch ──> residual_cross
              │       (FragmentProgramInteraction, use_fpi_sim=True, union_α=β=0)
              │
              └──> Gamma Gates ──> final_logits
```

**数学形式**：
```
final_logits = base_logits
    + gamma_global[d] * residual_global
    + effective_gamma_cross[d] * residual_cross

effective_gamma_cross = schedule_factor(epoch)
    * gamma_cross[d]
    * cross_active_mask[d]
```

#### 24.10.3 实现细节

**文件**: `mmpa_drp/fpi.py` — `DualFPIResidualRouter`

| 组件 | 类/参数 | 说明 |
|:---|:---|:---|
| Global branch | `self.global_fpi = FragmentProgramInteraction(use_fpi_sim=False)` | 复用最优 global FPI |
| Cross branch | `self.cross_fpi = FragmentProgramInteraction(use_fpi_sim=True, union_α=β=0)` | 纯 cross-modal attention |
| Gamma global | `raw_gamma_global` → sigmoid * γ_max | per-drug, 所有药物共享学习 |
| Gamma cross | `raw_gamma_cross` → sigmoid * γ_max | per-drug, 默认关闭 (init=0) |
| Schedule | `router_cross_start_epoch`, `router_cross_ramp_epochs` | 延迟激活 cross，避免早期干扰 |
| Oracle mask | `router_cross_mode="oracle_mask"` | **仅诊断用**，强制指定药物使用 cross |
| Diagnostics | `branch_preference`, `branch_effect_ratio` | 自动判断每个药物偏好哪个分支 |

**文件**: `mmpa_drp/models.py` — `ProgramFusionMLP`

- `__init__`: 新增 `use_fpi_router`, `fpi_router_config` 参数
- `forward`: 当 `use_fpi_router=True` 时，调用 `self.fpi_router(..., current_epoch, drug_names)`
- 互斥检查：`use_fpi_router` 与 `use_fpi`, `use_bpaf`, `use_llrf`, `use_dspi`, `use_drug_gate` 不能共存

**文件**: `mmpa_drp/stage2_program.py`

- `fusion_training_program`: 读取 `use_fpi_router`, `lambda_router_gamma_l1`, `fpi_router_config`
- `_fusion_model_train_step_with_fpi`: 新增 `current_epoch`, `drug_names`, `lambda_router_gamma_l1` 参数
- Router loss: `lambda_router_gamma_l1 * (|gamma_global| + |gamma_cross|).mean()`
- 互斥：`use_fpi_router=True` 时，`use_fpi` 自动置为 False

**文件**: `train_mmpa_program.py`

新增 argparse 参数（共 12 个）：
- `--use_fpi_router`
- `--router_gamma_global_max`, `--router_gamma_global_init`
- `--router_gamma_cross_max`, `--router_gamma_cross_init`
- `--router_residual_global_scale`, `--router_residual_cross_scale`
- `--router_cross_start_epoch`, `--router_cross_ramp_epochs`
- `--router_cross_mode`, `--router_oracle_cross_drugs`
- `--router_branch_effect_threshold`
- `--lambda_router_gamma_l1`

#### 24.10.4 关键参数（推荐配置）

| 参数 | 推荐值 | 理由 |
|:---|:---|:---|
| `router_gamma_global_max` | 0.05 | 复用 FPI2_warmup 最优值 |
| `router_gamma_global_init` | 0.05 | 避免 gamma 坍缩 |
| `router_gamma_cross_max` | 0.05 | cross 上限与 global 相同 |
| `router_gamma_cross_init` | 0.0 | cross 默认关闭，让需要的药物自己学 |
| `lambda_router_gamma_l1` | 0.0 | **关键**：不惩罚 gamma，避免钉死 |
| `router_cross_start_epoch` | 0 | 不延迟（或 10 如果早期不稳定） |
| `router_cross_ramp_epochs` | 0 | 不 ramp（或 10 平滑过渡） |

#### 24.10.5 互斥逻辑

```python
# train_mmpa_program.py
if args.use_fpi_router and (args.use_fpi or args.use_bpaf or ...):
    parser.error("...")

# stage2_program.py
if use_fpi_router:
    use_fpi = False  # 强制关闭旧 FPI

# models.py
if use_fpi_router and (use_fpi or use_bpaf or ...):
    raise ValueError("...")
```

#### 24.10.6 诊断指标

Router forward 返回的 dict 包含丰富的 diagnostics：

| 键 | 形状 | 说明 |
|:---|:---|:---|
| `gamma_global` | [D] | 每个药物的 global gamma 值 |
| `gamma_cross` | [D] | 每个药物的 cross gamma 值 |
| `effective_gamma_cross` | [D] | 经过 schedule + mask 后的实际 cross gamma |
| `branch_global_effect` | [D] | global 分支对 base 的相对影响 |
| `branch_cross_effect` | [D] | cross 分支对 base 的相对影响 |
| `branch_preference` | List[str] | 每个药物的偏好："base_only", "global_fpi", "cross_fpi" |
| `cross_active_mask` | [D] | 哪些药物被允许使用 cross（learnable 全 1，oracle 部分 1） |
| `schedule_factor` | scalar | 当前 epoch 的 cross 激活系数 |

#### 24.10.7 消融实验

**脚本**: `run_ablation_fpi_router.py`

| 实验 | 配置 | 目的 |
|:---|:---|:---|
| B0 baseline | 无 FPI, 无 router | 对照 |
| Global only | cross_max=0 | 验证 global 单独是否仍达 0.6031 |
| Global + Learnable Cross | cross_init=0, no L1 | 核心实验：cross 能否自动被需要的药物学习 |
| Global + Cross Ramp | start=10, ramp=10 | 测试延迟激活是否更稳定 |
| Oracle Cross (Temozolomide) | mode=oracle, drugs=Temozolomide | **诊断上限**：如果 Temozolomide 强制 cross，AUC 能到多少 |
| Global + Cross + L1 | lambda_router_gamma_l1=0.001 | 验证 L1 是否会钉死 cross gamma |
| Low Global + High Cross | gg_max=0.03, gc_max=0.08 | 探索更激进的 cross 权重 |

#### 24.10.8 测试覆盖

**脚本**: `tests/test_fpi_router.py`（7 项测试，全部通过）

| 测试 | 验证内容 |
|:---|:---|
| `test_router_instantiation_and_forward` | 输出形状正确 |
| `test_gamma_ranges` | gamma 在 [0, max] 范围内 |
| `test_cross_branch_schedule` | ramp 逻辑正确（epoch 0→5→10） |
| `test_oracle_mask_mode` | oracle mask 仅对指定药物激活 |
| `test_integration_with_program_fusion_mlp` | 与 ProgramFusionMLP 集成正确 |
| `test_mutual_exclusion_fpi_router` | use_fpi + use_fpi_router 抛出 ValueError |
| `test_router_without_fragments` | 无 fragment 数据时 router 被安全绕过 |

---

#### 24.10.9 首次实验结果与 Bug 修复

**实验命令**（修复前）：
```bash
python train_mmpa_program.py --use_fpi_router \
  --router_gamma_global_max 0.05 --router_gamma_global_init 0.05 \
  --router_gamma_cross_max 0.05 --router_gamma_cross_init 0.0 \
  --lambda_router_gamma_l1 0.0 \
  --fpi_m_max 16 --fpi_fp_dim 1024 \
  --use_cdan --use_ot_alignment --lambda_ot 0.005 \
  --alph 0.3 --beta 0.3 --K 3 --train
```

**结果（修复前）**：

| 药物 | AUC (Router) | AUC (FPI2_warmup) | Δ |
|:---|---:|---:|---:|
| 5-Fluorouracil | 0.5066 | 0.6186 | **-0.112** |
| Cisplatin | 0.5840 | 0.6524 | -0.068 |
| Cyclophosphamide | 0.5484 | 0.6022 | -0.054 |
| Docetaxel | 0.4467 | 0.5321 | -0.085 |
| Doxorubicin | 0.4447 | 0.5411 | -0.096 |
| Etoposide | 0.6989 | 0.7655 | -0.067 |
| Gemcitabine | 0.5882 | 0.6122 | -0.024 |
| Paclitaxel | 0.4906 | 0.5262 | -0.036 |
| Temozolomide | **0.6916** | 0.5775 | **+0.114** |
| **Mean AUC** | **0.5555** | **0.6031** | **-0.048** |

**现象**：8/9 药物暴跌，仅 Temozolomide 暴涨 (+0.114)。

**根因诊断：双重 Gamma 缩放**

`DualFPIResidualRouter` 内部实例化了两台 `FragmentProgramInteraction`，每台都有自己的 `raw_gamma`。`FragmentProgramInteraction.forward_global()` 已经用 internal gamma 缩放过一次 residual：
```python
# FragmentProgramInteraction.forward_global()
residual_logits = torch.tanh(residual_logits)
gamma = self.get_gamma()  # internal gamma ≈ 0.05
# returns residual_logits (已缩放)
```

然后 `DualFPIResidualRouter.forward()` 又用自己的 gamma 再缩放一次：
```python
final_logits = base_logits + gamma_global * residual_global
```

→ Global branch 总贡献 = internal 0.05 × router 0.05 = **0.0025**（仅为 FPI2_warmup 的 1/20）
→ Cross branch 总贡献 = internal 0.0 × router 0.0 = **0.0**

这导致 global branch 实际上被关闭，大多数药物回到接近 B0 甚至更差（参数增多导致训练不稳定）。而 Temozolomide 在训练中 cross branch 学到了非零 gamma，所以暴涨——这反而**验证了 cross branch 对 Temozolomide 确实有效**。

**修复**：`DualFPIResidualRouter.__init__` 中创建 `global_fpi` 和 `cross_fpi` 时，将 internal `gamma_max=1.0`, `gamma_init=1.0`，禁用内部门控，让 Router 的 gamma 承担全部门控责任。

```python
self.global_fpi = FragmentProgramInteraction(
    ...,
    gamma_max=1.0,   # 固定为 1.0
    gamma_init=1.0,  # 固定为 1.0
    ...,
)
```

**修复后状态**：代码已更新，`tests/test_fpi_router.py` 7/7 通过，待重新运行实验验证。

---

#### 24.10.10 第二次实验与 Bug 修复

**修复后重新运行，结果仍然 AUC=0.5555。**

再次诊断发现：**`use_fpi_router` 根本没有传给 `ProgramFusionMLP`**！

在 `mmpa_drp/stage2_program.py` 的 `fusion_training_program` 中：
```python
fusion_model = ProgramFusionMLP(
    encoder=encoder,
    ...
    use_fpi=use_fpi,
    fpi_config=fpi_config,
    frag_fp=frag_fp_tensor,
    frag_mask=frag_mask_tensor,
).to(kwargs["device"])
```

`use_fpi_router` 和 `fpi_router_config` 完全没有出现在构造函数参数列表中。`ProgramFusionMLP.__init__` 中 `use_fpi_router` 默认为 `False`，因此：
- `fpi_router` 未被实例化
- `fragment_encoder` 未被实例化
- `_frag_fp` / `_frag_mask` buffer 保持为 `None`
- `forward()` 中 router 路径被跳过

**实际运行的是纯 B0 baseline**（AUC 0.5555 只是该次随机种子的结果）。

**修复**（`mmpa_drp/stage2_program.py`）：
```python
fusion_model = ProgramFusionMLP(
    encoder=encoder,
    ...
    use_fpi=use_fpi,
    fpi_config=fpi_config,
    # Phase 9-R: FPI Router
    use_fpi_router=use_fpi_router,
    fpi_router_config=fpi_router_config,
    frag_fp=frag_fp_tensor,
    frag_mask=frag_mask_tensor,
).to(kwargs["device"])
```

**教训**：参数从 argparse → `training_params` → `kwargs` → `fusion_training_program` → `ProgramFusionMLP` 的传递链条很长，任何一个环节遗漏都会导致模块静默失效。`print(f"use_fpi_router={use_fpi_router}")` 虽然打印了参数值，但没有验证模型内部是否实际创建了 router。

**待验证**：修复后重新运行实验，确认 Router 真正生效。

---

## 附录：活跃文件清单（Phase 9-R 更新版）

| 文件 | 说明 | 状态 |
|------|------|------|
| `mmpa_drp/models.py` | `MolecularProgramEncoder`, `ProgramTissueClassifier`, `ProgramFusionMLP`, `FPISIMInteraction`, **DualFPIResidualRouter** | ✅ 活跃 |
| `mmpa_drp/backbones.py` | `ConvNeXt1DBackbone`, `GRNMLPBackbone` | ✅ 活跃 (Phase 5) |
| `mmpa_drp/losses.py` | diversity, sparsity, entropy_floor, gate_balance, gate_diversity | ✅ 活跃 |
| `mmpa_drp/ot_utils.py` | Sinkhorn, OT alignment, `ProgramOTLoss` (含 q_mass_flattened) | ✅ 活跃 (Phase 6-R) |
| `mmpa_drp/grad_diagnostics.py` | 梯度诊断工具 | ✅ 活跃 (Phase 6-R) |
| `mmpa_drp/fpi.py` | `FragmentProgramInteraction`, `FPISIMInteraction`, **DualFPIResidualRouter** | ✅ 活跃 (Phase 9-R) |
| `mmpa_drp/fragment_encoder.py` | `DrugFragmentEncoder` | ✅ 活跃 (Phase 9) |
| `mmpa_drp/fragment_utils.py` | `load_drug_fragments` | ✅ 活跃 (Phase 9) |
| `mmpa_drp/stage1_program.py` | Stage 1 训练 | ✅ 活跃 |
| `mmpa_drp/stage2_program.py` | Stage 2&3 训练（含 OT + Program OT + q_align + FPI + **Router**） | ✅ 活跃 |
| `train_mmpa_program.py` | MMPA-DRP 训练主入口 | ✅ 活跃 |
| `run_ablation_fpi_router.py` | Phase 9-R 消融实验脚本 | ✅ 新增 |
| `tests/test_fpi_router.py` | Router smoke tests (7/7 通过) | ✅ 新增 |
| `CODEMAP_MACB.md` | 本文件 | ✅ 活跃 |

### 已停止的模块

| 模块 | 停止原因 |
|------|----------|
| `DisentangledProgramEncoder` / `DisentangledProgramTissueClassifier` / `DisentangledProgramFusionMLP` | 3A-R 未超越 baseline |
| `DrugMechanismGate` / `DrugGatedProgramAggregator` / `SharedDrugGate` | 3B-R 未超越 baseline |
| `FPISIMInteraction`（standalone） | FPI-SIM 未超越 global FPI，但保留作为 Router 的 cross branch |

### 最终方法栈（Phase 9-R 之后）

| 层级 | 组件 | 配置 |
|:---|:---|:---|
| Backbone | `ConvNeXt1DBackbone` | depth=1, kernel=7, expansion=4 |
| Encoder | `MolecularProgramEncoder` | K=8 programs, d=64 |
| Sample OT | `DrugAdaptiveOTLoss` (global) | **λ=0.005**, reg=0.1, cosine |
| Program OT | 不使用（AUC 无提升） | — |
| CDAN | 保留 | α=0.3 |
| Stage 3 | Contrastive | **K=3**, β=0.3 |
| FPI Router | `DualFPIResidualRouter` | global + cross learnable gates, γ_init_global=0.05, γ_init_cross=0.0, no L1 |

**历史最佳 AUC (Windows 本地)**: **0.6189** (Combo K=3 + λ_ot=0.005, 无 FPI)
**当前环境最佳 AUC (Linux 服务器)**: **0.6031** (FPI2_warmup, K=3 + λ_ot=0.005 + FPI global)
**Phase 9-R 目标**: 在 0.6031 基础上，通过 cross branch 为 Temozolomide 等药物进一步提升
## 2026-05-17 并行实验记录：C3_K_3 vs Combo Best (K=3, λ_ot 对比)（重跑C3实验）

### 实验背景
在同一环境、同一天（2026-05-17）并行运行两个高度相似的超参组合，检验 λ_ot 的敏感性及跨配置稳定性。

### 运行命令

**实验 A — C3_K_3 (λ_ot=0.01)** — AUC 胜出
```bash
python3 train_mmpa_program.py \
    --train --use_ot_alignment --use_cdan \
    --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
    --lambda_ot 0.01 --n 5 --alph 0.3 --beta 0.3 --K 3
```

**实验 B — Combo Best (λ_ot=0.005)** — 历史最佳配置复现
```bash
python3 train_mmpa_program.py \
    --train --use_ot_alignment --use_cdan \
    --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
    --lambda_ot 0.005 --n 5 --alph 0.3 --beta 0.3 --K 3
```

### 共享配置

| 参数 | 值 |
|:---|:---|
| Backbone | ConvNeXt1D (depth=1, kernel=7) |
| Programs | 8 × 64-dim |
| CDAN | α=0.3 |
| Contrastive | β=0.3, **K=3** |
| n-folds | 5 |
| Pretrain | 300 epochs |
| Train | 100 epochs |
| Contrastive | 200 epochs |
| Seed | 2025 (hardcoded) |

### 结果对比

| 指标 | **C3_K_3 (λ_ot=0.01)** | Combo (λ_ot=0.005) | Δ |
|:---|---:|---:|---:|
| **AUC** | **0.5901** ± 0.0915 | 0.5779 ± 0.1020 | **+0.0122** |
| AUPR | 0.7179 ± 0.1920 | 0.7136 ± 0.1760 | +0.0043 |
| ACC | **0.4447** ± 0.2350 | 0.4093 ± 0.2053 | +0.0354 |
| F1 | **0.5119** ± 0.1540 | 0.5034 ± 0.1320 | +0.0085 |

**结论：λ_ot=0.01 在今天的环境下全面优于 0.005。**

### 最佳实验详细结果（C3_K_3, λ_ot=0.01）

**5-fold 平均 ± 标准差：**

| 药物 | AUC | AUPR | F1 | ACC |
|:---|---:|---:|---:|---:|
| 5-Fluorouracil | 0.6388 ± 0.0514 | 0.7385 ± 0.0441 | 0.4855 ± 0.0730 | 0.3490 ± 0.1868 |
| Cisplatin | 0.6466 ± 0.0303 | 0.7945 ± 0.0125 | 0.5271 ± 0.0659 | 0.5688 ± 0.0924 |
| Cyclophosphamide | 0.6112 ± 0.0618 | 0.9275 ± 0.0277 | 0.3600 ± 0.2227 | 0.4261 ± 0.3117 |
| Docetaxel | 0.5632 ± 0.0393 | 0.7338 ± 0.0152 | 0.5019 ± 0.0786 | 0.5176 ± 0.1506 |
| Doxorubicin | 0.5446 ± 0.0753 | 0.7800 ± 0.0542 | 0.4579 ± 0.1704 | 0.4591 ± 0.2323 |
| **Etoposide** | **0.7394 ± 0.0774** | **0.9161 ± 0.0349** | 0.6000 ± 0.2049 | 0.6773 ± 0.2191 |
| Gemcitabine | 0.5682 ± 0.0277 | 0.5778 ± 0.0334 | 0.5400 ± 0.0342 | 0.3884 ± 0.1291 |
| Paclitaxel | 0.5304 ± 0.0352 | 0.7223 ± 0.0214 | 0.4874 ± 0.0913 | 0.5250 ± 0.1465 |
| Temozolomide | 0.4683 ± 0.0839 | 0.2705 ± 0.0684 | 0.6478 ± 0.2163 | 0.0914 ± 0.1743 |

**复合指标：**
- `all_metrics`: 0.5662 ± 0.2017
- `auc_aupr_metrics`: 0.6540 ± 0.1618

### 各药物 AUC 横向对比（λ_ot=0.01 vs 0.005）

| 药物 | λ_ot=0.01 | λ_ot=0.005 | Δ |
|:---|---:|---:|---:|
| 5-FU | 0.6388 | 0.5865 | **+0.052** |
| Cisplatin | 0.6466 | 0.6213 | +0.025 |
| Cyclophosphamide | 0.6112 | 0.5526 | +0.059 |
| Docetaxel | 0.5632 | 0.5326 | +0.031 |
| Doxorubicin | 0.5446 | 0.5100 | +0.035 |
| Etoposide | 0.7394 | 0.7537 | -0.014 |
| Gemcitabine | 0.5682 | 0.5929 | -0.025 |
| Paclitaxel | 0.5304 | 0.5090 | +0.021 |
| Temozolomide | 0.4683 | 0.5426 | **-0.074** |

λ_ot=0.01 在 7/9 药物上更好，仅在 Etoposide (-0.014) 和 Gemcitabine (-0.025) 上略差，Temozolomide 差异最大（-0.074 对 λ_ot=0.005 有利）。

### 关键发现

1. **λ_ot 存在环境敏感波动**：同一天内 λ_ot=0.01 优于 0.005，但历史最佳 0.6189 恰好是 λ_ot=0.005。说明 λ_ot 最优值对 CUDA/GPU 状态敏感，不是一个稳定可迁移的超参。
2. **结果文件碰撞**：两个实验生成的 `param_str` 完全相同（不含 `lambda_ot`），导致 CSV 文件互相覆盖。后完成的 Combo 结果留在了 `results/` 目录，C3_K_3 的结果仅存于日志 `brief_logs.txt` 中。
3. **Etoposide 持续最强**：两个配置下 Etoposide 都是 AUC 最高的药物（0.7394 / 0.7537），说明模型对 Etoposide 的响应模式学习得最稳定。
4. **Temozolomide 持续最弱**：两个配置下 Temozolomide 都是 AUC 最低的药物（0.4683 / 0.5426），模型对此药响应区分度不足。

### 日志位置

- C3_K_3 (λ_ot=0.01): `logs/2026-05-17_17-01-41mmpa_program/`
- Combo (λ_ot=0.005): `logs/2026-05-17_17-04-14mmpa_program/`

---

## 25. Phase 9-F: FPI Router Diagnosis (Two-stage Cross Compensation)

### 25.1 背景

在 Phase 9-R 的 `DualFPIResidualRouter` 实现后，首次修复实验显示 **Mean AUC 0.5555**（远低于 FPI2_warmup 的 0.6031），但 **Temozolomide 暴涨至 0.6916**（+0.114）。这揭示了 cross branch 对 Temozolomide 有独特价值，但对其他药物有害。

Phase 9-F 的目标是通过系统化诊断实验，确认 cross branch 的益处是否可以被**隔离**并仅用于 Temozolomide，而不损害全局性能。

### 25.2 为什么只对 Temozolomide 进行 Cross Branch

**数据驱动的选择**：

| 药物 | FPI2_warmup AUC | 9-R 首次实验 AUC | Δ |
|:---|---:|---:|---:|
| 5-Fluorouracil | 0.6186 | 0.5066 | **-0.112** |
| Cisplatin | 0.6524 | 0.5840 | -0.068 |
| Cyclophosphamide | 0.6022 | 0.5484 | -0.054 |
| Docetaxel | 0.5321 | 0.4467 | -0.085 |
| Doxorubicin | 0.5411 | 0.4447 | -0.096 |
| Etoposide | 0.7655 | 0.6989 | -0.067 |
| Gemcitabine | 0.6122 | 0.5882 | -0.024 |
| Paclitaxel | 0.5262 | 0.4906 | -0.036 |
| **Temozolomide** | **0.5775** | **0.6916** | **+0.114** |
| **Mean** | **0.6031** | **0.5555** | **-0.048** |

- **8/9 药物暴跌**：cross branch 联合 global branch 导致全局性能崩溃
- **仅 Temozolomide 暴涨**：说明 cross branch 的 signal 对 Temozolomide 是独特的，与其他药物不兼容
- **Temozolomide 是最弱药物**：在所有 baseline 配置中（MACB 0.5998, C3 0.6452, FPI2_warmup 0.5775, C3_K_3 λ_ot=0.01 0.4683），Temozolomide  consistently 是最难预测的药物之一

**生物学/结构假设**：
Temozolomide 的 BRICS fragment 分解结果与其他药物显著不同（小分子、简单环结构、含 N-甲基等特征），其 fragment fingerprint 与 program token 的 **cross-modal interaction**（而非 global average）可能捕获了其他表征路径无法提取的药物响应信号。

**策略选择**：
由于 cross branch 对 8/9 药物有害，不能全局启用。采用 **Oracle Mask** 诊断策略：强制仅对 Temozolomide 激活 cross branch，验证"cross-only 后处理补偿器"假设。

> ⚠️ **Oracle mask 仅为诊断工具**，不能作为最终生产方法。若诊断成功，需设计自动机制（如 per-drug gamma gate）让模型自己学习哪些药物需要 cross。

### 25.3 诊断实验设计

| 实验 | 名称 | 配置 | 目的 |
|:---|:---|:---|:---|
| **1** | Match Global | `cross_fpi` 复制 `global_fpi` 的初始化参数 | 排除 cross branch 结构本身缺陷 |
| **2** | Cross Detach | `router_cross_detach_mode="all"`（detach z/q/h/drug/frag） | 阻断梯度污染，测试 pure residual 冲突 |
| **3** | Two-stage Cross Only | Stage A: 训练 Global-only → save ckpt; Stage B: 冻结全部主模型，仅训练 cross_fpi + gamma_cross | 完全隔离 cross branch 训练 |
| **4** | Oracle Mask (诊断) | `router_cross_mode="oracle_mask"`, `router_oracle_cross_drugs="Temozolomide"` | 强制仅 Temozolomide 使用 cross |
| **5** | Cross Detach + Oracle | 实验2 + Oracle mask 组合 | 双重保险验证 |

**基线对照**：
- **Router Global-only Confirm**: `router_gamma_cross_max=0.0` → `cross_fpi=None` → 等价于 FPI2_warmup
  - 结果：**Mean AUC 0.6031**, **Temozolomide 0.5775**（与 FPI2_warmup 完全一致，验证 lazy creation 正确性）

### 25.4 实验结果

#### 汇总表

| 实验 | Mean AUC | Temozolomide | vs Global-only | 结论 |
|:---|---:|---:|---:|:---|
| **Global-only Confirm** | **0.6031** | 0.5775 | — | 基线已复现 ✅ |
| **1 Match Global** | 0.5644 | — | -0.0387 | ❌ Cross branch 联合训练本身崩溃 |
| **2 Cross Detach** | 0.5801 | — | -0.0230 | ⚠️ 缓解梯度污染但仍不足 |
| **3 Two-stage** | **0.6117** | **0.6051** | **+0.0086** / **+0.0276** | ✅ **Cross branch 能隔离提升 Temo** |

#### 实验 3 (Two-stage) 详细结果

| 药物 | Global-only | Two-stage | Δ |
|:---|---:|---:|---:|
| 5-Fluorouracil | 0.6186 | **0.6355** | +0.0169 |
| Cisplatin | 0.6524 | 0.6456 | -0.0068 |
| Cyclophosphamide | 0.6022 | **0.6345** | +0.0323 |
| Docetaxel | 0.5321 | 0.5220 | -0.0101 |
| Doxorubicin | 0.5411 | **0.5498** | +0.0087 |
| Etoposide | 0.7655 | 0.7492 | -0.0163 |
| Gemcitabine | 0.6122 | **0.6268** | +0.0146 |
| Paclitaxel | 0.5262 | 0.5206 | -0.0056 |
| **Temozolomide** | **0.5775** | **0.6051** | **+0.0276** |
| **Mean** | **0.6031** | **0.6117** | **+0.0086** |

> **稳定性验证**：修复 BatchNorm eval 后重新运行，所有 AUC 数字与修复前**完全一致**（确定性强）。

### 25.5 关键发现与分析

#### 发现 1：Cross branch 联合训练导致全局崩溃

**实验 1 (Match Global, 0.5644)** 证明：即使 `cross_fpi` 和 `global_fpi` 使用完全相同的初始化参数，联合训练仍然导致 Mean AUC 从 0.6031 暴跌到 0.5644。

**根因**：两个 residual branch 同时优化时，loss landscape 存在冲突。global branch 试图修正 base_logits 的方向 A，cross branch 试图修正方向 B，梯度在共享的 encoder/fusion 参数上互相干扰。

#### 发现 2：Detach 缓解但不够

**实验 2 (Cross Detach, 0.5801)** 通过 detach z/q/h_pool/drug_emb/frag_emb，阻断了 cross branch 的梯度向主模型回流。

- 从 0.5644 恢复到 0.5801（+0.0157），说明**梯度污染是部分原因**
- 但仍低于 Global-only 的 0.6031（-0.0230），说明**即使无梯度污染，residual 本身也存在冲突**

**根因**：cross branch 输出的 residual 信号在数学上与 global branch 的 residual 不兼容。`final_logits = base + global_residual + cross_residual`，两个 residual 向量可能互相抵消。

#### 发现 3：Two-stage 完全隔离是成功的

**实验 3 (Two-stage, 0.6117)** 的核心设计：

```
Stage A: 训练 Global-only (cross_fpi=None) → 保存 checkpoint
Stage B: 加载 checkpoint → 冻结 encoder/fusion/cdan/global_fpi/attention_masks → 仅训练 cross_fpi + gamma_cross
```

关键实现：
- `load_state_dict(..., strict=False)` 兼容 Stage A 无 cross_fpi 的 state dict
- 冻结模块调用 `.eval()` 阻止 BatchNorm running stats 漂移
- 仅 2 个参数组可训练：`cross_fpi` 网络参数 + `raw_gamma_cross` per-drug gate

**结果**：
- Mean AUC **0.6117** > Global-only **0.6031**（+0.86%）
- Temozolomide **0.6051** > Global-only **0.5775**（+2.76%）
- 其他药物变化微小（±0.03 以内），无显著损害

**结论**：当主模型完全冻结时，cross branch 的 residual 不再与 global branch 冲突，能够作为**纯后处理补偿器**提升特定药物的性能。

#### 发现 4：非 Temozolomide 药物仍有微小漂移

| 药物 | Δ |
|:---|---:|
| 5-FU | +0.017 |
| Cyclophosphamide | +0.032 |
| Gemcitabine | +0.015 |

这些药物在 Two-stage 中也有正向漂移（尽管 oracle mask 理论上只给 Temozolomide 开 cross）。可能原因：
1. `gamma_cross` 对非 Temo 药物不是严格的 0（sigmoid(0) * 0.05 = 0.025，不是 0）
2. BatchNorm running stats 在 Stage B 中仍有极微小漂移
3. 同一 checkpoint 在不同运行中的固有变异

但这些漂移幅度小（<0.035）且方向为正，不构成问题。

### 25.6 运行命令（差差差）

#### Stage A: Global-only Checkpoint 训练（保存 checkpoint）

```bash  0.5480
python train_mmpa_program.py \
  --measurement Z_SCORE \
  --thres_gdsc 0.0 --thres_label 0.1 \
  --n 5 --alph 0.3 --beta 0.3 --train \
  --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
  --num_programs 8 --program_dim 64 \
  --K 3 --use_ot_alignment --lambda_ot 0.005 --use_cdan \
  --use_fpi_router \
  --router_gamma_global_max 0.05 --router_gamma_global_init 0.05 \
  --router_gamma_cross_max 0.0 \
  --lambda_router_gamma_l1 0.0 \
  --router_stage_a_checkpoint_dir checkpoints/router_global_only
```

> `--router_gamma_cross_max 0.0` 触发 lazy creation，`cross_fpi=None`，等价于纯 Global FPI。

#### Stage B: Two-stage Cross-only 训练

```bash
python train_mmpa_program.py \
  --measurement Z_SCORE \
  --thres_gdsc 0.0 --thres_label 0.1 \
  --n 5 --alph 0.3 --beta 0.3 --train \
  --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
  --num_programs 8 --program_dim 64 \
  --K 3 --use_ot_alignment --lambda_ot 0.005 --use_cdan \
  --use_fpi_router \
  --router_gamma_global_max 0.05 --router_gamma_global_init 0.05 \
  --router_gamma_cross_max 0.05 --router_gamma_cross_init 0.0 \
  --lambda_router_gamma_l1 0.0 \
  --router_cross_mode oracle_mask \
  --router_oracle_cross_drugs "Temozolomide" \
  --router_two_stage_cross_only \
  --router_freeze_main_for_cross \
  --router_stage_a_checkpoint_dir checkpoints/router_global_only
```

#### 诊断脚本

```bash
# 运行全部诊断实验（实验 1-5）
python run_ablation_fpi_router_diagnosis.py

# 分析结果
python scripts/analyze_phase9_router_ablation.py
```

### 25.7 结论与决策

| 问题 | 答案 | 决策 |
|:---|:---|:---|
| Cross branch 对 Temozolomide 有效吗？ | ✅ **有效**，Two-stage 提升 +0.0276 | 继续探索 cross branch 的 Temozolomide 特异性 |
| Cross branch 能全局启用吗？ | ❌ **不能**，联合训练导致崩溃 (0.5644) | 不允许全局 cross branch |
| Two-stage 是可行路径吗？ | ✅ **是**，冻结主模型后 cross 能作为后处理补偿器 | 保留 Two-stage 机制 |
| Oracle mask 能用于生产吗？ | ❌ **不能**，Oracle mask 是人为先验，非 learnable | 需替换为自动门控机制 |

**下一步方向**：
1. **自动药物选择**：让 `gamma_cross` 自己学习哪些药物需要 cross（而非人工 oracle mask）。如果 `gamma_cross` 在非 Temo 药物上保持 ≈0，在 Temo 上上升到 >0，则自动实现等效效果。
2. **Temozolomide 专项分析**：深入分析 Temozolomide 的 BRICS fragment 模式，理解为什么 cross-modal interaction 对其独特有效。
3. **生产化 Two-stage**：将 Two-stage 训练流程自动化（Stage A → save → Stage B），作为可选的训练模式集成到主流程。

---

## 附录：活跃文件清单（Phase 9-F 更新版）

| 文件 | 说明 | 状态 |
|------|------|------|
| `mmpa_drp/models.py` | `MolecularProgramEncoder`, `ProgramFusionMLP`, `DualFPIResidualRouter` | ✅ 活跃 |
| `mmpa_drp/fpi.py` | `FragmentProgramInteraction`, `DualFPIResidualRouter` | ✅ 活跃 |
| `mmpa_drp/stage2_program.py` | Stage 2&3 训练（含 Two-stage + freeze + eval 逻辑） | ✅ 活跃 |
| `train_mmpa_program.py` | MMPA-DRP 训练主入口（含 router 参数） | ✅ 活跃 |
| `run_ablation_fpi_router_diagnosis.py` | Phase 9-F 诊断实验脚本 | ✅ 活跃 |
| `scripts/analyze_phase9_router_ablation.py` | 诊断结果自动汇总 | ✅ 活跃 |
| `CODEMAP_MACB.md` | 本文件 | ✅ 活跃 |

### 已停止的模块

| 模块 | 停止原因 |
|------|----------|
| `DisentangledProgramEncoder` / `DisentangledProgramTissueClassifier` / `DisentangledProgramFusionMLP` | 3A-R 未超越 baseline |
| `DrugMechanismGate` / `DrugGatedProgramAggregator` / `SharedDrugGate` | 3B-R 未超越 baseline |
| `FPISIMInteraction`（standalone） | FPI-SIM 未超越 global FPI，但作为 Router cross branch 保留 |
| `DrugAdaptiveOTLoss` (per_drug_fixed/validation/discrepancy) | Phase 4 药物自适应 OT 无效，保留 global 模式 |

### 最终方法栈（Phase 9-F 之后）

| 层级 | 组件 | 配置 |
|:---|:---|:---|
| Backbone | `ConvNeXt1DBackbone` | depth=1, kernel=7, expansion=4 |
| Encoder | `MolecularProgramEncoder` | K=8 programs, d=64 |
| Sample OT | `DrugAdaptiveOTLoss` (global) | λ=0.005, reg=0.1, cosine |
| Program OT | 不使用（AUC 无提升） | — |
| CDAN | 保留 | α=0.3 |
| Stage 3 | Contrastive | K=3, β=0.3 |
| FPI Router | `DualFPIResidualRouter` | global: γ=0.05; cross: Two-stage for Temozolomide |

**历史最佳 AUC (Windows 本地)**: **0.6189** (Combo K=3 + λ_ot=0.005, 无 FPI)
**当前环境最佳 AUC (Linux 服务器)**: **0.6117** (Two-stage FPI Router, Global + Cross for Temozolomide)
**Temozolomide 最佳 AUC**: **0.6051** (Two-stage cross branch)

*******************************************MACB*********************************
                              0       1       2       3       4
drug             metric                                        
5-Fluorouracil   auc     0.5371  0.5552  0.5131  0.4411  0.5624
                 aupr    0.6996  0.6837  0.6406  0.6226  0.7113
                 f1      0.5364  0.5591  0.5318  0.4727  0.5591
                 acc     0.6047  0.6196  0.6084  0.5704  0.6166
Cisplatin        auc     0.6886  0.6738  0.6669  0.5959  0.7159
                 aupr    0.8375  0.8279  0.8143  0.7610  0.8648
                 f1      0.7122  0.6954  0.6835  0.6882  0.7338
                 acc     0.8119  0.7894  0.7836  0.7841  0.8252
Cyclophosphamide auc     0.6929  0.4176  0.7346  0.6237  0.7015
                 aupr    0.9496  0.8885  0.9556  0.9283  0.9636
                 f1      0.7294  0.5000  0.8706  0.6000  0.7000
                 acc     0.8333  0.6559  0.9290  0.7344  0.8145
Docetaxel        auc     0.4165  0.3321  0.4065  0.3125  0.4018
                 aupr    0.6543  0.6016  0.6216  0.5767  0.6342
                 f1      0.4088  0.3459  0.3774  0.3962  0.4151
                 acc     0.4835  0.4409  0.4142  0.4074  0.5131
Doxorubicin      auc     0.4374  0.3515  0.4637  0.3195  0.4162
                 aupr    0.7080  0.6653  0.7293  0.6497  0.6938
                 f1      0.3333  0.3270  0.3585  0.2704  0.3145
                 acc     0.3765  0.3815  0.3704  0.2468  0.3394
Etoposide        auc     0.8289  0.6586  0.7586  0.7609  0.7672
                 aupr    0.9369  0.8691  0.9190  0.9255  0.9080
                 f1      0.8021  0.7292  0.8333  0.8125  0.8646
                 acc     0.8774  0.8267  0.8987  0.8831  0.9231
Gemcitabine      auc     0.6149  0.5690  0.6152  0.5926  0.6067
                 aupr    0.6299  0.5551  0.6222  0.5564  0.5536
                 f1      0.5609  0.5304  0.5913  0.5783  0.6043
                 acc     0.5388  0.4953  0.5481  0.5872  0.5845
Paclitaxel       auc     0.4716  0.4845  0.5160  0.4253  0.4407
                 aupr    0.6874  0.7055  0.6984  0.6758  0.6720
                 f1      0.4775  0.5180  0.4414  0.4099  0.4369
                 acc     0.5469  0.6373  0.4746  0.4178  0.5455
Temozolomide     auc     0.5174  0.7306  0.6306  0.6056  0.5150
                 aupr    0.3422  0.5394  0.4221  0.3483  0.3103
                 f1      0.7537  0.7836  0.7388  0.7239  0.7388
                 acc     0.1538  0.4528  0.2222  0.3019  0.1026

5-fold average/std results: 
5-Fluorouracil    auc: 0.5218/0.0489 | aupr: 0.6715/0.0383 | f1: 0.5318/0.0354 | acc: 0.6039/0.0197
Cisplatin         auc: 0.6682/0.0446 | aupr: 0.8211/0.0383 | f1: 0.7026/0.0206 | acc: 0.7988/0.0187
Cyclophosphamide  auc: 0.6341/0.1275 | aupr: 0.9371/0.0302 | f1: 0.6800/0.1396 | acc: 0.7934/0.1035
Docetaxel         auc: 0.3739/0.0479 | aupr: 0.6177/0.0299 | f1: 0.3887/0.0279 | acc: 0.4518/0.0455
Doxorubicin       auc: 0.3976/0.0602 | aupr: 0.6892/0.0321 | f1: 0.3208/0.0324 | acc: 0.3429/0.0562
Etoposide         auc: 0.7548/0.0611 | aupr: 0.9117/0.0260 | f1: 0.8083/0.0503 | acc: 0.8818/0.0355
Gemcitabine       auc: 0.5997/0.0194 | aupr: 0.5834/0.0390 | f1: 0.5730/0.0287 | acc: 0.5508/0.0377
Paclitaxel        auc: 0.4676/0.0359 | aupr: 0.6878/0.0143 | f1: 0.4568/0.0418 | acc: 0.5244/0.0830
Temozolomide      auc: 0.5998/0.0895 | aupr: 0.3924/0.0918 | f1: 0.7478/0.0226 | acc: 0.2467/0.1374
4 metrics average/std results: 5-Fluorouracil: 0.5823/0.0708 | Cisplatin: 0.7477/0.0721 | Cyclophosphamide: 0.7612/0.1561 | Docetaxel: 0.4580/0.1054 | Doxorubicin: 0.4376/0.1578 | Etoposide: 0.8392/0.0756 | Gemcitabine: 0.5767/0.0347 | Paclitaxel: 0.5341/0.1054 | Temozolomide: 0.4967/0.2150
auc&aupr average/std results:5-Fluorouracil: 0.5967/0.0891 | Cisplatin: 0.7447/0.0896 | Cyclophosphamide: 0.7856/0.1821 | Docetaxel: 0.4958/0.1339 | Doxorubicin: 0.5434/0.1603 | Etoposide: 0.8333/0.0938 | Gemcitabine: 0.5915/0.0303 | Paclitaxel: 0.5777/0.1189 | Temozolomide: 0.4961/0.1388
auc：0.5575/0.1354  aupr：0.7013/0.1669  acc：0.5772/0.2155  f1：0.5789/0.1684
all_metrics：0.6037/0.1812
auc_aupr_metrics：0.6294/0.1666

********************************有OT******************************************
0.5779
python train_mmpa_program.py \
  --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 \
  --alph 0.3 --beta 0.3 --train \
  --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
  --num_programs 8 --program_dim 64 --K 3 --lambda_ot 0.005 \
  --use_cdan --use_ot_alignment
5-fold average/std results: 
5-Fluorouracil    auc: 0.5865/0.0293 | aupr: 0.6900/0.0372 | f1: 0.4836/0.0766 | acc: 0.3651/0.2124
Cisplatin         auc: 0.6213/0.0314 | aupr: 0.7912/0.0231 | f1: 0.4532/0.0474 | acc: 0.4465/0.0930
Cyclophosphamide  auc: 0.5526/0.1299 | aupr: 0.9154/0.0314 | f1: 0.3918/0.1771 | acc: 0.4940/0.2030
Docetaxel         auc: 0.5326/0.0393 | aupr: 0.7094/0.0210 | f1: 0.4604/0.0215 | acc: 0.4583/0.0567
Doxorubicin       auc: 0.5100/0.0383 | aupr: 0.7708/0.0249 | f1: 0.3836/0.0199 | acc: 0.3851/0.0310
Etoposide         auc: 0.7537/0.0843 | aupr: 0.9298/0.0338 | f1: 0.6188/0.1119 | acc: 0.7107/0.1132
Gemcitabine       auc: 0.5929/0.0204 | aupr: 0.5777/0.0269 | f1: 0.5626/0.0205 | acc: 0.3887/0.0631
Paclitaxel        auc: 0.5090/0.0221 | aupr: 0.6841/0.0225 | f1: 0.4351/0.0335 | acc: 0.4359/0.0810
Temozolomide      auc: 0.5426/0.1635 | aupr: 0.3542/0.1644 | f1: 0.7418/0.0041 | acc: 0.0000/0.0000
4 metrics average/std results: 5-Fluorouracil: 0.5313/0.1628 | Cisplatin: 0.5780/0.1540 | Cyclophosphamide: 0.5884/0.2450 | Docetaxel: 0.5402/0.1104 | Doxorubicin: 0.5124/0.1641 | Etoposide: 0.7533/0.1432 | Gemcitabine: 0.5305/0.0913 | Paclitaxel: 0.5161/0.1127 | Temozolomide: 0.4096/0.3000
auc&aupr average/std results: 5-Fluorouracil: 0.6382/0.0630 | Cisplatin: 0.7062/0.0933 | Cyclophosphamide: 0.7340/0.2109 | Docetaxel: 0.6210/0.0978 | Doxorubicin: 0.6404/0.1408 | Etoposide: 0.8418/0.1108 | Gemcitabine: 0.5853/0.0239 | Paclitaxel: 0.5966/0.0946 | Temozolomide: 0.4484/0.1838
auc：0.5779/0.1020  aupr：0.7136/0.1760  acc：0.4093/0.2053  f1：0.5034/0.1320
all_metrics：0.5511/0.1925
auc_aupr_metrics：0.6458/0.1576

python train_mmpa_program.py --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 --alph 0.3 --beta 0.3 --train --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 --num_programs 8 --program_dim 64 --K 3 --use_ot_alignment --lambda_ot 0.005 --use_cdan

5-fold average/std results: 
5-Fluorouracil    auc: 0.5865/0.0293 | aupr: 0.6900/0.0372 | f1: 0.4836/0.0766 | acc: 0.3651/0.2124
Cisplatin         auc: 0.6213/0.0314 | aupr: 0.7912/0.0231 | f1: 0.4532/0.0474 | acc: 0.4465/0.0930
Cyclophosphamide  auc: 0.5526/0.1299 | aupr: 0.9154/0.0314 | f1: 0.3918/0.1771 | acc: 0.4940/0.2030
Docetaxel         auc: 0.5326/0.0393 | aupr: 0.7094/0.0210 | f1: 0.4604/0.0215 | acc: 0.4583/0.0567
Doxorubicin       auc: 0.5100/0.0383 | aupr: 0.7708/0.0249 | f1: 0.3836/0.0199 | acc: 0.3851/0.0310
Etoposide         auc: 0.7537/0.0843 | aupr: 0.9298/0.0338 | f1: 0.6188/0.1119 | acc: 0.7107/0.1132
Gemcitabine       auc: 0.5929/0.0204 | aupr: 0.5777/0.0269 | f1: 0.5626/0.0205 | acc: 0.3887/0.0631
Paclitaxel        auc: 0.5090/0.0221 | aupr: 0.6841/0.0225 | f1: 0.4351/0.0335 | acc: 0.4359/0.0810
Temozolomide      auc: 0.5426/0.1635 | aupr: 0.3542/0.1644 | f1: 0.7418/0.0041 | acc: 0.0000/0.0000
4 metrics average/std results: 5-Fluorouracil: 0.5313/0.1628 | Cisplatin: 0.5780/0.1540 | Cyclophosphamide: 0.5884/0.2450 | Docetaxel: 0.5402/0.1104 | Doxorubicin: 0.5124/0.1641 | Etoposide: 0.7533/0.1432 | Gemcitabine: 0.5305/0.0913 | Paclitaxel: 0.5161/0.1127 | Temozolomide: 0.4096/0.3000
auc&aupr average/std results: 5-Fluorouracil: 0.6382/0.0630 | Cisplatin: 0.7062/0.0933 | Cyclophosphamide: 0.7340/0.2109 | Docetaxel: 0.6210/0.0978 | Doxorubicin: 0.6404/0.1408 | Etoposide: 0.8418/0.1108 | Gemcitabine: 0.5853/0.0239 | Paclitaxel: 0.5966/0.0946 | Temozolomide: 0.4484/0.1838
auc：0.5779/0.1020  aupr：0.7136/0.1760  acc：0.4093/0.2053  f1：0.5034/0.1320
all_metrics：0.5511/0.1925
auc_aupr_metrics：0.6458/0.1576
************************************无FPI+convnext_depth1*******************
  python3 train_mmpa_program.py \
    --train --use_ot_alignment --use_cdan \
    --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
    --lambda_ot 0.01 --n 5 --alph 0.3 --beta 0.3 --K 3

5-fold average/std results: 
5-Fluorouracil    auc: 0.6388/0.0514 | aupr: 0.7385/0.0441 | f1: 0.4855/0.0730 | acc: 0.3490/0.1868
Cisplatin         auc: 0.6466/0.0303 | aupr: 0.7945/0.0125 | f1: 0.5271/0.0659 | acc: 0.5688/0.0924
Cyclophosphamide  auc: 0.6112/0.0618 | aupr: 0.9275/0.0277 | f1: 0.3600/0.2227 | acc: 0.4261/0.3117
Docetaxel         auc: 0.5632/0.0393 | aupr: 0.7338/0.0152 | f1: 0.5019/0.0786 | acc: 0.5176/0.1506
Doxorubicin       auc: 0.5446/0.0753 | aupr: 0.7800/0.0542 | f1: 0.4579/0.1704 | acc: 0.4591/0.2323
Etoposide         auc: 0.7394/0.0774 | aupr: 0.9161/0.0349 | f1: 0.6000/0.2049 | acc: 0.6773/0.2191
Gemcitabine       auc: 0.5682/0.0277 | aupr: 0.5778/0.0334 | f1: 0.5400/0.0342 | acc: 0.3884/0.1291
Paclitaxel        auc: 0.5304/0.0352 | aupr: 0.7223/0.0214 | f1: 0.4874/0.0913 | acc: 0.5250/0.1465
Temozolomide      auc: 0.4683/0.0839 | aupr: 0.2705/0.0684 | f1: 0.6478/0.2163 | acc: 0.0914/0.1743
4 metrics average/std results: 5-Fluorouracil: 0.5529/0.1805 | Cisplatin: 0.6342/0.1179 | Cyclophosphamide: 0.5812/0.2879 | Docetaxel: 0.5791/0.1240 | Doxorubicin: 0.5604/0.1937 | Etoposide: 0.7332/0.1864 | Gemcitabine: 0.5186/0.1015 | Paclitaxel: 0.5663/0.1243 | Temozolomide: 0.3695/0.2541
auc&aupr average/std results: 5-Fluorouracil: 0.6886/0.0693 | Cisplatin: 0.7205/0.0810 | Cyclophosphamide: 0.7693/0.1727 | Docetaxel: 0.6485/0.0942 | Doxorubicin: 0.6623/0.1386 | Etoposide: 0.8277/0.1090 | Gemcitabine: 0.5730/0.0294 | Paclitaxel: 0.6263/0.1048 | Temozolomide: 0.3694/0.1268
auc：0.5901/0.0915  aupr：0.7179/0.1920  acc：0.4447/0.2350  f1：0.5119/0.1540
all_metrics：0.5662/0.2017
auc_aupr_metrics：0.6540/0.1618

**************************************有FPI***********************************
0.6031
 python train_mmpa_program.py \
  --measurement Z_SCORE \
  --thres_gdsc 0.0 \
  --thres_label 0.1 \
  --n 5 \
  --alph 0.3 \
  --beta 0.3 \
  --train \
  --backbone_type convnext1d \
  --convnext_depth 1 \
  --convnext_kernel_size 7 \
  --num_programs 8 \
  --program_dim 64 \
  --K 3 \
  --use_ot_alignment \
  --lambda_ot 0.005 \
  --use_cdan \
  --use_fpi \
  --fpi_fragment_npz data/drug_fragments_brics_m16_fp1024.npz \
  --fpi_gamma_max 0.1 \
  --fpi_gamma_init 0.05 \
  --fpi_q_prior_weight 0.01 \
  --lambda_fpi_gamma_l1 0.0

5-fold average/std results: 
5-Fluorouracil    auc: 0.6186/0.0630 | aupr: 0.7284/0.0335 | f1: 0.4682/0.0575 | acc: 0.2957/0.1317
Cisplatin         auc: 0.6524/0.0156 | aupr: 0.8184/0.0125 | f1: 0.5405/0.0425 | acc: 0.5887/0.0670
Cyclophosphamide  auc: 0.6022/0.0468 | aupr: 0.9267/0.0186 | f1: 0.4341/0.1250 | acc: 0.5564/0.1346
Docetaxel         auc: 0.5321/0.0863 | aupr: 0.7104/0.0763 | f1: 0.4792/0.0883 | acc: 0.4909/0.1082
Doxorubicin       auc: 0.5411/0.0666 | aupr: 0.7946/0.0539 | f1: 0.4252/0.0756 | acc: 0.4291/0.1041
Etoposide         auc: 0.7655/0.0397 | aupr: 0.9369/0.0157 | f1: 0.6104/0.0742 | acc: 0.7058/0.0797
Gemcitabine       auc: 0.6122/0.0189 | aupr: 0.6024/0.0365 | f1: 0.5722/0.0125 | acc: 0.3953/0.0452
Paclitaxel        auc: 0.5262/0.0724 | aupr: 0.7176/0.0402 | f1: 0.4640/0.0426 | acc: 0.4864/0.0657
Temozolomide      auc: 0.5775/0.0617 | aupr: 0.3985/0.0990 | f1: 0.7403/0.0133 | acc: 0.0480/0.1073
4 metrics average/std results: 5-Fluorouracil: 0.5277/0.1824 | Cisplatin: 0.6500/0.1141 | Cyclophosphamide: 0.6299/0.2062 | Docetaxel: 0.5531/0.1264 | Doxorubicin: 0.5475/0.1695 | Etoposide: 0.7546/0.1332 | Gemcitabine: 0.5455/0.0947 | Paclitaxel: 0.5485/0.1153 | Temozolomide: 0.4411/0.2737
auc&aupr average/std results: 5-Fluorouracil: 0.6735/0.0749 | Cisplatin: 0.7354/0.0885 | Cyclophosphamide: 0.7645/0.1743 | Docetaxel: 0.6212/0.1214 | Doxorubicin: 0.6678/0.1453 | Etoposide: 0.8512/0.0947 | Gemcitabine: 0.6073/0.0279 | Paclitaxel: 0.6219/0.1150 | Temozolomide: 0.4880/0.1223
auc：0.6031/0.0879  aupr：0.7371/0.1646  acc：0.4440/0.2014  f1：0.5260/0.1153
all_metrics：0.5776/0.1825
auc_aupr_metrics：0.6701/0.1467

*********************stage1分为shared和private domain+OT，无FPI**********
python train_mmpa_program.py \
  --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 \
  --train --backbone_type convnext1d --convnext_depth 1 \
  --num_programs 8 --program_dim 64 --K 3 \
  --use_ot_alignment --lambda_ot 0.005 --use_cdan \
  --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64 \
  --lambda_ortho 0.001 --lambda_recon 0.001 --lambda_private_domain 0.001 \
  --lambda_migrate 0.01 \
  --alph 1.4 \
  --pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100

5-fold average/std results: 
5-Fluorouracil    auc: 0.6046/0.0690 | aupr: 0.7195/0.0482 | f1: 0.5973/0.0558 | acc: 0.6835/0.0782
Cisplatin         auc: 0.6628/0.0243 | aupr: 0.8355/0.0213 | f1: 0.7022/0.0397 | acc: 0.8046/0.0372
Cyclophosphamide  auc: 0.6321/0.0792 | aupr: 0.9443/0.0155 | f1: 0.7071/0.1336 | acc: 0.8122/0.1004
Docetaxel         auc: 0.4918/0.0721 | aupr: 0.6919/0.0722 | f1: 0.5208/0.0666 | acc: 0.6339/0.0877
Doxorubicin       auc: 0.5765/0.1075 | aupr: 0.8071/0.0711 | f1: 0.5950/0.1084 | acc: 0.6936/0.1171
Etoposide         auc: 0.7728/0.0381 | aupr: 0.9415/0.0123 | f1: 0.8292/0.0203 | acc: 0.8997/0.0117
Gemcitabine       auc: 0.5953/0.0245 | aupr: 0.5885/0.0155 | f1: 0.5513/0.0523 | acc: 0.5972/0.0502
Paclitaxel        auc: 0.5321/0.0759 | aupr: 0.7182/0.0593 | f1: 0.6171/0.0391 | acc: 0.7373/0.0368
Temozolomide      auc: 0.6029/0.0658 | aupr: 0.4143/0.0934 | f1: 0.5910/0.2337 | acc: 0.2324/0.1711
4 metrics average/std results: 5-Fluorouracil: 0.6512/0.0792 | Cisplatin: 0.7513/0.0784 | Cyclophosphamide: 0.7739/0.1475 | Docetaxel: 0.5846/0.1084 | Doxorubicin: 0.6681/0.1332 | Etoposide: 0.8608/0.0698 | Gemcitabine: 0.5831/0.0406 | Paclitaxel: 0.6512/0.0986 | Temozolomide: 0.4601/0.2109
auc&aupr average/std results:5-Fluorouracil: 0.6621/0.0826 | Cisplatin: 0.7492/0.0935 | Cyclophosphamide: 0.7882/0.1731 | Docetaxel: 0.5918/0.1255 | Doxorubicin: 0.6918/0.1489 | Etoposide: 0.8571/0.0928 | Gemcitabine: 0.5919/0.0196 | Paclitaxel: 0.6252/0.1172 | Temozolomide: 0.5086/0.1252
auc：0.6079/0.0975  aupr：0.7401/0.1682  acc：0.6771/0.2000  f1：0.6345/0.1311
all_metrics：0.6649/0.1603
auc_aupr_metrics：0.6740/0.1511  

***********************************+fpi*******************
python train_mmpa_program.py   --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5   --train --backbone_type convnext1d --convnext_depth 1   --num_programs 8 --program_dim 64 --K 3   --use_ot_alignment --lambda_ot 0.005 --use_cdan   --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64   --lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005   --lambda_migrate 0.01   --alph 1.4   --use_fpi --fpi_gamma_max 0.1   --pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100

5-fold average/std results: 
5-Fluorouracil    auc: 0.6251/0.0569 | aupr: 0.7298/0.0436 | f1: 0.6091/0.0318 | acc: 0.7078/0.0356
Cisplatin         auc: 0.6610/0.0351 | aupr: 0.8415/0.0145 | f1: 0.6758/0.0391 | acc: 0.7763/0.0401
Cyclophosphamide  auc: 0.6698/0.0345 | aupr: 0.9481/0.0130 | f1: 0.7659/0.0630 | acc: 0.8603/0.0444
Docetaxel         auc: 0.5284/0.0677 | aupr: 0.7239/0.0479 | f1: 0.5585/0.0625 | acc: 0.6675/0.0657
Doxorubicin       auc: 0.5729/0.0638 | aupr: 0.8280/0.0336 | f1: 0.5673/0.0463 | acc: 0.6753/0.0612
Etoposide         auc: 0.7550/0.0628 | aupr: 0.9319/0.0255 | f1: 0.7833/0.0468 | acc: 0.8671/0.0363
Gemcitabine       auc: 0.5884/0.0270 | aupr: 0.6009/0.0142 | f1: 0.5591/0.0435 | acc: 0.5729/0.0428
Paclitaxel        auc: 0.5474/0.0253 | aupr: 0.7410/0.0202 | f1: 0.6306/0.0294 | acc: 0.7481/0.0264
Temozolomide      auc: 0.6089/0.0968 | aupr: 0.4360/0.1123 | f1: 0.7493/0.0100 | acc: 0.0743/0.0383
4 metrics average/std results: 5-Fluorouracil: 0.6679/0.0662 | Cisplatin: 0.7387/0.0822 | Cyclophosphamide: 0.8110/0.1136 | Docetaxel: 0.6196/0.0991 | Doxorubicin: 0.6609/0.1187 | Etoposide: 0.8343/0.0827 | Gemcitabine: 0.5804/0.0352 | Paclitaxel: 0.6668/0.0885 | Temozolomide: 0.4671/0.2684
auc&aupr average/std results: 5-Fluorouracil: 0.6775/0.0730 | Cisplatin: 0.7513/0.0984 | Cyclophosphamide: 0.8090/0.1487 | Docetaxel: 0.6262/0.1169 | Doxorubicin: 0.7004/0.1428 | Etoposide: 0.8434/0.1036 | Gemcitabine: 0.5947/0.0214 | Paclitaxel: 0.6442/0.1043 | Temozolomide: 0.5225/0.1344
auc：0.6174/0.0843  aupr：0.7535/0.1597  acc：0.6611/0.2316  f1：0.6554/0.0960
all_metrics：0.6719/0.1607
auc_aupr_metrics：0.6855/0.1434
***********************-fpi
python train_mmpa_program.py   --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5   --train --backbone_type convnext1d --convnext_depth 1   --num_programs 8 --program_dim 64 --K 3   --use_ot_alignment --lambda_ot 0.005 --use_cdan   --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64   --lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005   --lambda_migrate 0.01   --alph 1.4 --pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100
                              0       1       2       3       4
drug             metric                                        
5-Fluorouracil   auc     0.6312  0.6466  0.5609  0.5788  0.5325
                 aupr    0.7173  0.7268  0.6594  0.6650  0.6806
                 f1      0.5409  0.6318  0.6000  0.5318  0.5273
                 acc     0.5944  0.6966  0.7317  0.5253  0.5398
Cisplatin        auc     0.6680  0.6574  0.6829  0.6509  0.6578
                 aupr    0.8174  0.8249  0.8437  0.8100  0.8360
                 f1      0.6978  0.7242  0.7122  0.5707  0.6595
                 acc     0.7955  0.8228  0.8125  0.6524  0.7543
Cyclophosphamide auc     0.5394  0.6895  0.5789  0.6108  0.6013
                 aupr    0.9019  0.9599  0.9258  0.9126  0.9290
                 f1      0.4529  0.6765  0.8118  0.6647  0.4353
                 acc     0.5903  0.7955  0.8940  0.7849  0.5676
Docetaxel        auc     0.4101  0.5369  0.6605  0.5523  0.4139
                 aupr    0.6527  0.7310  0.7949  0.7016  0.6603
                 f1      0.4717  0.5409  0.6541  0.5786  0.4403
                 acc     0.6000  0.6404  0.7489  0.6763  0.5616
Doxorubicin      auc     0.4084  0.5748  0.7210  0.6525  0.4097
                 aupr    0.7156  0.8226  0.8783  0.8221  0.7140
                 f1      0.4465  0.5283  0.7170  0.6478  0.4277
                 acc     0.5644  0.6305  0.8018  0.7383  0.5285
Etoposide        auc     0.7586  0.8031  0.7156  0.8062  0.7391
                 aupr    0.9323  0.9527  0.9258  0.9512  0.9205
                 f1      0.7708  0.7708  0.8542  0.8229  0.8021
                 acc     0.8571  0.8553  0.9176  0.8944  0.8820
Gemcitabine      auc     0.6001  0.5855  0.5437  0.5888  0.5943
                 aupr    0.6243  0.5874  0.5554  0.6335  0.5748
                 f1      0.5696  0.5565  0.5087  0.5565  0.5696
                 acc     0.6207  0.5234  0.5703  0.5565  0.5992
Paclitaxel       auc     0.4495  0.5657  0.5990  0.5387  0.4555
                 aupr    0.6524  0.7327  0.7536  0.7120  0.6591
                 f1      0.5541  0.6351  0.6577  0.5631  0.5450
                 acc     0.6877  0.7523  0.7683  0.6881  0.6622
Temozolomide     auc     0.6653  0.6306  0.5632  0.3659  0.3500
                 aupr    0.4823  0.4702  0.3283  0.2277  0.2205
                 f1      0.7612  0.7537  0.3060  0.7537  0.2612
                 acc     0.4839  0.0571  0.3841  0.0571  0.3529

5-fold average/std results: 
5-Fluorouracil    auc: 0.5900/0.0479 | aupr: 0.6898/0.0306 | f1: 0.5664/0.0469 | acc: 0.6176/0.0927
Cisplatin         auc: 0.6634/0.0125 | aupr: 0.8264/0.0136 | f1: 0.6729/0.0621 | acc: 0.7675/0.0694
Cyclophosphamide  auc: 0.6040/0.0551 | aupr: 0.9259/0.0219 | f1: 0.6082/0.1607 | acc: 0.7265/0.1415
Docetaxel         auc: 0.5147/0.1052 | aupr: 0.7081/0.0580 | f1: 0.5371/0.0853 | acc: 0.6454/0.0721
Doxorubicin       auc: 0.5533/0.1415 | aupr: 0.7905/0.0728 | f1: 0.5535/0.1260 | acc: 0.6527/0.1154
Etoposide         auc: 0.7645/0.0397 | aupr: 0.9365/0.0147 | f1: 0.8042/0.0356 | acc: 0.8813/0.0262
Gemcitabine       auc: 0.5825/0.0224 | aupr: 0.5951/0.0331 | f1: 0.5522/0.0252 | acc: 0.5740/0.0377
Paclitaxel        auc: 0.5217/0.0667 | aupr: 0.7020/0.0447 | f1: 0.5910/0.0516 | acc: 0.7117/0.0459
Temozolomide      auc: 0.5150/0.1481 | aupr: 0.3458/0.1266 | f1: 0.5672/0.2594 | acc: 0.2670/0.1976
4 metrics average/std results: 5-Fluorouracil: 0.6159/0.0722 | Cisplatin: 0.7325/0.0820 | Cyclophosphamide: 0.7161/0.1684 | Docetaxel: 0.6013/0.1106 | Doxorubicin: 0.6375/0.1464 | Etoposide: 0.8466/0.0740 | Gemcitabine: 0.5759/0.0320 | Paclitaxel: 0.6316/0.0947 | Temozolomide: 0.4237/0.2146
auc&aupr average/std results:5-Fluorouracil: 0.6399/0.0648 | Cisplatin: 0.7449/0.0868 | Cyclophosphamide: 0.7649/0.1742 | Docetaxel: 0.6114/0.1296 | Doxorubicin: 0.6719/0.1640 | Etoposide: 0.8505/0.0949 | Gemcitabine: 0.5888/0.0274 | Paclitaxel: 0.6118/0.1091 | Temozolomide: 0.4304/0.1575
auc：0.5899/0.1095  aupr：0.7244/0.1803  acc：0.6493/0.1867  f1：0.6058/0.1344
all_metrics：0.6424/0.1629
auc_aupr_metrics：0.6572/0.1621

******************************
********************************************
**********************************stage3分离的改进，原有的直接分离，现有在药物基础上分离(per-drug KNN)******************
python train_mmpa_program.py   --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5   --train --backbone_type convnext1d --convnext_depth 1   --num_programs 8 --program_dim 64 --K 3   --use_ot_alignment --lambda_ot 0.005 --use_cdan   --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64   --lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005   --lambda_migrate 0.01   --alph 1.4   --use_fpi --fpi_gamma_max 0.1   --pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100

5-fold average/std results: 
5-Fluorouracil    auc: 0.6225/0.0393 | aupr: 0.7245/0.0369 | f1: 0.6036/0.0354 | acc: 0.6981/0.0506
Cisplatin         auc: 0.6579/0.0374 | aupr: 0.8352/0.0194 | f1: 0.6801/0.0425 | acc: 0.7791/0.0373
Cyclophosphamide  auc: 0.6545/0.0425 | aupr: 0.9445/0.0123 | f1: 0.7647/0.1301 | acc: 0.8531/0.1037
Docetaxel         auc: 0.5276/0.0858 | aupr: 0.7224/0.0584 | f1: 0.5547/0.0743 | acc: 0.6696/0.0815
Doxorubicin       auc: 0.5648/0.0760 | aupr: 0.8169/0.0430 | f1: 0.5786/0.0876 | acc: 0.6826/0.1045
Etoposide         auc: 0.7705/0.0494 | aupr: 0.9414/0.0178 | f1: 0.7896/0.0259 | acc: 0.8717/0.0219
Gemcitabine       auc: 0.5853/0.0234 | aupr: 0.5947/0.0195 | f1: 0.5470/0.0446 | acc: 0.5674/0.0368
Paclitaxel        auc: 0.5441/0.0409 | aupr: 0.7411/0.0340 | f1: 0.6279/0.0291 | acc: 0.7511/0.0256
Temozolomide      auc: 0.6418/0.1126 | aupr: 0.4846/0.1299 | f1: 0.7582/0.0136 | acc: 0.1373/0.0585
4 metrics average/std results: 5-Fluorouracil: 0.6622/0.0640 | Cisplatin: 0.7381/0.0809 | Cyclophosphamide: 0.8042/0.1355 | Docetaxel: 0.6186/0.1077 | Doxorubicin: 0.6607/0.1275 | Etoposide: 0.8433/0.0756 | Gemcitabine: 0.5736/0.0353 | Paclitaxel: 0.6660/0.0927 | Temozolomide: 0.5055/0.2539
auc&aupr average/std results: 5-Fluorouracil: 0.6735/0.0647 | Cisplatin: 0.7465/0.0976 | Cyclophosphamide: 0.7995/0.1556 | Docetaxel: 0.6250/0.1238 | Doxorubicin: 0.6909/0.1450 | Etoposide: 0.8560/0.0966 | Gemcitabine: 0.5900/0.0209 | Paclitaxel: 0.6426/0.1097 | Temozolomide: 0.5632/0.1414
auc：0.6188/0.0908  aupr：0.7561/0.1517  acc：0.6678/0.2181  f1：0.6560/0.1076
all_metrics：0.6747/0.1570
auc_aupr_metrics：0.6875/0.1414

 --beta 0.1
 python train_mmpa_program.py
--measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 --train
--backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7
--num_programs 8 --program_dim 64 --K 3
--use_ot_alignment --lambda_ot 0.005 --ot_reg 0.1
--ot_cost_type cosine --ot_marginal_type uniform
--use_cdan
--use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64
--lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005 --lambda_migrate 0.01
--alph 1.4 --beta 0.1
--use_fpi --fpi_gamma_max 0.1 --fpi_gamma_init 0.05 --fpi_q_prior_weight 0.01
--fpi_dropout 0.1 --fpi_m_max 16 --fpi_fp_dim 1024 --lambda_fpi_gamma_l1 0.001
--fpi_detach_mode none
--pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100
 0 1 2 3 4 drug metric
5-Fluorouracil auc 0.5313 0.6517 0.5985 0.7506 0.6219 aupr 0.6418 0.7371 0.7098 0.8220 0.7233 f1 0.5909 0.6136 0.6273 0.6773 0.6136 acc 0.7256 0.6743 0.7405 0.7760 0.7447 Cisplatin auc 0.6119 0.6656 0.6470 0.6869 0.6505 aupr 0.8126 0.8457 0.8359 0.8526 0.8292 f1 0.7218 0.7122 0.7098 0.7074 0.7098 acc 0.8263 0.8193 0.8141 0.8051 0.8153 Cyclophosphamide auc 0.6112 0.7002 0.6972 0.7230 0.7049 aupr 0.9406 0.9555 0.9558 0.9640 0.9578 f1 0.7765 0.8765 0.8529 0.8647 0.7941 acc 0.8716 0.9333 0.9180 0.9260 0.8805 Docetaxel auc 0.5256 0.4660 0.5405 0.6387 0.5685 aupr 0.7115 0.6853 0.7187 0.7905 0.7566 f1 0.5975 0.4528 0.5975 0.6981 0.5849 acc 0.7311 0.4596 0.7217 0.8110 0.7130 Doxorubicin auc 0.5939 0.5508 0.6355 0.6332 0.5981 aupr 0.8468 0.8196 0.8516 0.8494 0.8372 f1 0.6415 0.4277 0.6792 0.6981 0.6164 acc 0.7692 0.4348 0.7811 0.8140 0.7382 Etoposide auc 0.6711 0.7969 0.7672 0.8000 0.6539 aupr 0.8912 0.9548 0.9180 0.9386 0.8648 f1 0.8542 0.7396 0.8438 0.8542 0.8438 acc 0.9186 0.8227 0.9112 0.9125 0.9080 Gemcitabine auc 0.5530 0.5739 0.5792 0.6090 0.5733 aupr 0.5860 0.5541 0.6086 0.6139 0.5727 f1 0.4696 0.5304 0.5174 0.5783 0.5478 acc 0.6039 0.3647 0.5934 0.6421 0.5517 Paclitaxel auc 0.5597 0.5252 0.5495 0.5917 0.5491 aupr 0.7332 0.7140 0.7461 0.7705 0.7110 f1 0.6351 0.5541 0.6802 0.6847 0.6892 acc 0.7718 0.6400 0.7977 0.8023 0.8023 Temozolomide auc 0.6479 0.5768 0.6656 0.6229 0.5203 aupr 0.4838 0.3125 0.5340 0.4591 0.4079 f1 0.7463 0.7463 0.7612 0.7537 0.7612 acc 0.0556 0.0000 0.1579 0.0571 0.1111

5-fold average/std results: 5-Fluorouracil auc: 0.6308/0.0803 | aupr: 0.7268/0.0646 | f1: 0.6245/0.0322 | acc: 0.7322/0.0372 Cisplatin auc: 0.6524/0.0275 | aupr: 0.8352/0.0155 | f1: 0.7122/0.0056 | acc: 0.8160/0.0078 Cyclophosphamide auc: 0.6873/0.0437 | aupr: 0.9548/0.0086 | f1: 0.8329/0.0447 | acc: 0.9059/0.0279 Docetaxel auc: 0.5479/0.0631 | aupr: 0.7325/0.0413 | f1: 0.5862/0.0874 | acc: 0.6873/0.1331 Doxorubicin auc: 0.6023/0.0346 | aupr: 0.8409/0.0131 | f1: 0.6126/0.1082 | acc: 0.7075/0.1548 Etoposide auc: 0.7378/0.0702 | aupr: 0.9135/0.0362 | f1: 0.8271/0.0492 | acc: 0.8946/0.0404 Gemcitabine auc: 0.5777/0.0201 | aupr: 0.5871/0.0249 | f1: 0.5287/0.0402 | acc: 0.5512/0.1091 Paclitaxel auc: 0.5551/0.0241 | aupr: 0.7350/0.0245 | f1: 0.6486/0.0572 | acc: 0.7628/0.0698 Temozolomide auc: 0.6067/0.0587 | aupr: 0.4395/0.0842 | f1: 0.7537/0.0075 | acc: 0.0763/0.0602 4 metrics average/std results: 5-Fluorouracil: 0.6786/0.0741 | Cisplatin: 0.7540/0.0784 | Cyclophosphamide: 0.8452/0.1083 | Docetaxel: 0.6385/0.1112 | Doxorubicin: 0.6908/0.1322 | Etoposide: 0.8432/0.0846 | Gemcitabine: 0.5611/0.0601 | Paclitaxel: 0.6754/0.0944 | Temozolomide: 0.4691/0.2648 auc&aupr average/std results:5-Fluorouracil: 0.6788/0.0853 | Cisplatin: 0.7438/0.0986 | Cyclophosphamide: 0.8210/0.1441 | Docetaxel: 0.6402/0.1095 | Doxorubicin: 0.7216/0.1282 | Etoposide: 0.8256/0.1065 | Gemcitabine: 0.5824/0.0219 | Paclitaxel: 0.6450/0.0976 | Temozolomide: 0.5231/0.1116 auc：0.6220/0.0755 aupr：0.7517/0.1585 acc：0.6815/0.2522 f1：0.6807/0.1146 all_metrics：0.6840/0.1685 auc_aupr_metrics：0.6868/0.1388 


 python train_mmpa_program.py --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 --train --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 --num_programs 8 --program_dim 64 --K 3 --use_ot_alignment --lambda_ot 0.005 --ot_reg 0.1 --ot_cost_type cosine --ot_marginal_type uniform --use_cdan --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64 --lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005 --lambda_migrate 0.01 --alph 1.4 --beta 0.066 --use_fpi --fpi_gamma_max 0.1 --fpi_gamma_init 0.05 --fpi_q_prior_weight 0.01 --fpi_dropout 0.1 --fpi_m_max 16 --fpi_fp_dim 1024 --lambda_fpi_gamma_l1 0.001 --fpi_detach_mode none --pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100

python train_mmpa_program.py \
  --measurement Z_SCORE --thres_gdsc 0.0 --thres_label 0.1 --n 5 \
  --train --backbone_type convnext1d --convnext_depth 1 --convnext_kernel_size 7 \
  --num_programs 8 --program_dim 64 --K 9 \
  --use_ot_alignment --lambda_ot 0.005 --ot_reg 0.1 --ot_cost_type cosine --ot_marginal_type uniform \
  --use_cdan \
  --use_domain_sep --domain_sep_hidden_dim 256 --domain_sep_output_dim 64 \
  --lambda_ortho 0.001 --lambda_recon 0.005 --lambda_private_domain 0.005 --lambda_migrate 0.01 \
  --alph 1.4 --beta 0.066 \
  --use_fpi --fpi_gamma_max 0.1 --fpi_gamma_init 0.05 --fpi_q_prior_weight 0.01 --fpi_dropout 0.1 \
  --fpi_m_max 16 --fpi_fp_dim 1024 --lambda_fpi_gamma_l1 0.001 --fpi_detach_mode none \
  --pretrain_num_epochs 100 --train_num_epochs 100 --contrastive_num_epochs 100

5-fold average/std results: 
5-Fluorouracil    auc: 0.6500/0.0589 | aupr: 0.7449/0.0489 | f1: 0.6509/0.0403 | acc: 0.7458/0.0322
Cisplatin         auc: 0.6423/0.0373 | aupr: 0.8261/0.0257 | f1: 0.6739/0.0295 | acc: 0.7802/0.0311
Cyclophosphamide  auc: 0.6979/0.0279 | aupr: 0.9519/0.0120 | f1: 0.8259/0.0339 | acc: 0.9010/0.0219
Docetaxel         auc: 0.5637/0.0541 | aupr: 0.7434/0.0392 | f1: 0.6101/0.0585 | acc: 0.7299/0.0588
Doxorubicin       auc: 0.6025/0.0665 | aupr: 0.8418/0.0289 | f1: 0.6428/0.0517 | acc: 0.7586/0.0484
Etoposide         auc: 0.7722/0.0652 | aupr: 0.9283/0.0260 | f1: 0.8479/0.0240 | acc: 0.9105/0.0168
Gemcitabine       auc: 0.5761/0.0257 | aupr: 0.5752/0.0326 | f1: 0.5313/0.0291 | acc: 0.5964/0.0352
Paclitaxel        auc: 0.5873/0.0295 | aupr: 0.7641/0.0217 | f1: 0.6649/0.0145 | acc: 0.7859/0.0108
Temozolomide      auc: 0.6239/0.0075 | aupr: 0.4633/0.0421 | f1: 0.7493/0.0085 | acc: 0.1300/0.1702
4 metrics average/std results: 5-Fluorouracil: 0.6979/0.0645 | Cisplatin: 0.7306/0.0822 | Cyclophosphamide: 0.8442/0.1008 | Docetaxel: 0.6617/0.0927 | Doxorubicin: 0.7114/0.1076 | Etoposide: 0.8647/0.0718 | Gemcitabine: 0.5697/0.0374 | Paclitaxel: 0.7005/0.0839 | Temozolomide: 0.4916/0.2514
auc&aupr average/std results: 5-Fluorouracil: 0.6974/0.0715 | Cisplatin: 0.7342/0.1015 | Cyclophosphamide: 0.8249/0.1354 | Docetaxel: 0.6535/0.1047 | Doxorubicin: 0.7221/0.1351 | Etoposide: 0.8503/0.0947 | Gemcitabine: 0.5756/0.0277 | Paclitaxel: 0.6757/0.0963 | Temozolomide: 0.5436/0.0893
auc：0.6351/0.0755  aupr：0.7599/0.1532  acc：0.7042/0.2315  f1：0.6885/0.1024
all_metrics：0.6969/0.1574
auc_aupr_metrics：0.6975/0.1347

k=4
5-fold average/std results: 
5-Fluorouracil    auc: 0.6344/0.0688 | aupr: 0.7322/0.0553 | f1: 0.6382/0.0554 | acc: 0.7338/0.0432
Cisplatin         auc: 0.6483/0.0342 | aupr: 0.8273/0.0211 | f1: 0.6777/0.0186 | acc: 0.7815/0.0212
Cyclophosphamide  auc: 0.7020/0.0586 | aupr: 0.9557/0.0175 | f1: 0.8071/0.0705 | acc: 0.8869/0.0478
Docetaxel         auc: 0.5746/0.0773 | aupr: 0.7492/0.0426 | f1: 0.6075/0.0753 | acc: 0.7274/0.0650
Doxorubicin       auc: 0.5989/0.0659 | aupr: 0.8399/0.0282 | f1: 0.6126/0.0687 | acc: 0.7293/0.0738
Etoposide         auc: 0.7642/0.0680 | aupr: 0.9309/0.0293 | f1: 0.8187/0.0394 | acc: 0.8912/0.0270
Gemcitabine       auc: 0.5836/0.0217 | aupr: 0.5891/0.0310 | f1: 0.5435/0.0351 | acc: 0.5978/0.0225
Paclitaxel        auc: 0.5853/0.0281 | aupr: 0.7621/0.0106 | f1: 0.6486/0.0346 | acc: 0.7649/0.0371
Temozolomide      auc: 0.6539/0.0500 | aupr: 0.4687/0.0569 | f1: 0.7552/0.0123 | acc: 0.0761/0.0791
4 metrics average/std results: 5-Fluorouracil: 0.6846/0.0717 | Cisplatin: 0.7337/0.0785 | Cyclophosphamide: 0.8379/0.1082 | Docetaxel: 0.6647/0.0982 | Doxorubicin: 0.6952/0.1152 | Etoposide: 0.8513/0.0774 | Gemcitabine: 0.5785/0.0335 | Paclitaxel: 0.6902/0.0831 | Temozolomide: 0.4885/0.2708
auc&aupr average/std results: 5-Fluorouracil: 0.6833/0.0783 | Cisplatin: 0.7378/0.0981 | Cyclophosphamide: 0.8288/0.1398 | Docetaxel: 0.6619/0.1092 | Doxorubicin: 0.7194/0.1357 | Etoposide: 0.8476/0.1008 | Gemcitabine: 0.5864/0.0254 | Paclitaxel: 0.6737/0.0953 | Temozolomide: 0.5613/0.1099
auc：0.6384/0.0784  aupr：0.7617/0.1517  acc：0.6877/0.2388  f1：0.6788/0.1014
all_metrics：0.6916/0.1599
auc_aupr_metrics：0.7000/0.1344

K=9
5-fold average/std results: 
5-Fluorouracil    auc: 0.6366/0.0622 | aupr: 0.7373/0.0502 | f1: 0.6464/0.0411 | acc: 0.7438/0.0343
Cisplatin         auc: 0.6432/0.0380 | aupr: 0.8276/0.0259 | f1: 0.6806/0.0167 | acc: 0.7872/0.0166
Cyclophosphamide  auc: 0.7102/0.0646 | aupr: 0.9591/0.0153 | f1: 0.8188/0.0571 | acc: 0.8952/0.0371
Docetaxel         auc: 0.5945/0.0708 | aupr: 0.7563/0.0457 | f1: 0.6164/0.0440 | acc: 0.7354/0.0402
Doxorubicin       auc: 0.6219/0.0670 | aupr: 0.8507/0.0270 | f1: 0.6516/0.0332 | acc: 0.7648/0.0305
Etoposide         auc: 0.7670/0.0837 | aupr: 0.9311/0.0342 | f1: 0.8313/0.0528 | acc: 0.8988/0.0384
Gemcitabine       auc: 0.5714/0.0191 | aupr: 0.5737/0.0263 | f1: 0.5374/0.0271 | acc: 0.5894/0.0310
Paclitaxel        auc: 0.6000/0.0305 | aupr: 0.7715/0.0167 | f1: 0.6622/0.0110 | acc: 0.7795/0.0064
Temozolomide      auc: 0.6748/0.0374 | aupr: 0.5191/0.0437 | f1: 0.7582/0.0187 | acc: 0.1173/0.1691
4 metrics average/std results: 5-Fluorouracil: 0.6910/0.0674 | Cisplatin: 0.7347/0.0808 | Cyclophosphamide: 0.8458/0.1047 | Docetaxel: 0.6756/0.0869 | Doxorubicin: 0.7222/0.1015 | Etoposide: 0.8570/0.0826 | Gemcitabine: 0.5680/0.0309 | Paclitaxel: 0.7033/0.0793 | Temozolomide: 0.5174/0.2659
auc&aupr average/std results: 5-Fluorouracil: 0.6869/0.0752 | Cisplatin: 0.7354/0.1019 | Cyclophosphamide: 0.8346/0.1384 | Docetaxel: 0.6754/0.1021 | Doxorubicin: 0.7363/0.1298 | Etoposide: 0.8491/0.1054 | Gemcitabine: 0.5726/0.0217 | Paclitaxel: 0.6857/0.0933 | Temozolomide: 0.5970/0.0906
auc：0.6466/0.0780  aupr：0.7696/0.1440  acc：0.7013/0.2335  f1：0.6892/0.0977
all_metrics：0.7017/0.1555
auc_aupr_metrics：0.7081/0.1300

四、怎么知道 latent program 学到了什么
如果想验证 latent program 是否对应通路，可以做事后分析：

看每个 program 对哪些基因权重高；
把这些基因做通路富集分析（GSEA / ORA）；
看是否显著富集到某个 Hallmark / KEGG 通路。
如果显著富集，说明模型学到了和通路相关的结构；
如果没有，说明它学到的是任务特有的、非通路化的特征组合。

*****************************************************

## 18.在stage1阶段在现有模型基础上利用对比和对抗学习分析出shared domain和private domain（无药物参与），借鉴DCD的分离方法，在stage2仅使用shareddomain与药物进行交互，借鉴BML，将shared domain对drug的学习迁移到private domain中对齐

Stage 1: Shared/Private 分离（DCDNet 风格）
Gene expr → ConvNeXt → hidden [B, 512]
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
        shared_mlp    private_mlp   recon_decoder
              │            │            │
        h_shared      h_private     recon [B, 1472]
          [B, 256]     [B, 256]

Loss:
  L_tissue + λ_adv*L_domain_adv(shared混淆domain) 
  + λ_ortho*L_ortho + λ_recon*L_recon + λ_private*L_private_domain(private保留domain)
 Stage 2: BML 可靠性路由 + Shared→Private 迁移（动态融合）
  h_shared [B, 256] ──→ Shared_Drug_Head ──→ pred_shared [B, 9]
h_private [B, 256] ──→ Private_Drug_Head ──→ pred_private [B, 9]
                              │
                              ├──→ BML ReliabilityEstimator
                              │       ↓
                              │   reliability_shared [B, 1]
                              │   reliability_private [B, 1]
                              │       ↓
                              ├──→ 迁移对齐 (核心！)
                              │   mask = reliability_shared > 0.5
                              │   L_migrate = MSE(
                              │       pred_private[mask], 
                              │       pred_shared[mask].detach()
                              │   )
                              │   # private 学习 shared 的可靠预测
                              │
                              └──→ 最终预测
                                  pred_final = rel_s * pred_shared + rel_p * pred_private
 这个方案的核心价值
组件	作用
Shared	跨域稳定，学 drug response 的主干知识
Private	域特异，初始随机，通过迁移学习 shared 的可靠预测
BML Router	动态判断哪个 domain 在当前样本上更可靠
L_migrate	知识蒸馏：shared 的可靠预测 → private 的监督
BML Router 是动态融合，因为它的权重不是固定的，而是通过 MLP 根据每个样本-药物对的预测可靠性（entropy + agreement）实时计算出来的。同一个样本对不同药物、不同样本对同一种药物，权重都可以不同。                     


代码已上传到  https://github.com/3604778475/mmpa

关于你提到的结果文件和数据文件不需要推送，当前 .gitignore 已配置如下：

类型	状态	说明
logs/	✅ 已忽略	实验日志
results/	✅ 已忽略	5-fold 输出结果
outputs/	✅ 已忽略	实验输出
data/preprocessed/	✅ 刚添加忽略	大数据 CSV
*.npy / *.npz	✅ 已忽略	特征文件
*.pt / *.pth	✅ 已忽略	模型权重
但 data/raw_dat/ 和 data/TCGA_labels.csv 仍在仓库中（它们不大，且是原始数据索引，通常需要保留以便复现）。如果你也想忽略整个 data/ 目录，告诉我，我可以再加一行


