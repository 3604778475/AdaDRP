# CODEMAP_DCDNET.md — DCDNet Code Structure & MMPA-DRP Transferability Analysis

> **Repo**: https://github.com/rawwap/DCDNet  
> **Commit**: `3c393b218eaf0ec9cb928da6c9e8abd80c67c0d1`  
> **Analysis Date**: 2026-05-11  
> **Scope**: 静态代码阅读 + 模块定位 + 迁移可行性分析（不运行完整训练，不修改原始代码）

---

## 1. Repository Information

### Top-level Structure

```
DCDNet/
├── train.py              # 主训练入口（ACFD + MGDF + losses）
├── finetuning.py         # Fine-tuning 入口（启用 CAM / refine）
├── test.py               # 测试入口
├── model/
│   ├── DCDNet.py         # 主模型：Shared / Private / Fusion / Modulation
│   ├── discriminator.py  # Discriminator + GradientReversal
│   └── resnet.py         # ResNet50/101 backbone
├── data/                 # 数据集加载（FSS / DeepGlobe / ISIC / Lung）
├── util/
│   └── utils.py          # count_params / set_seed / mIOU
└── requirements.txt
```

### Environment Dependencies
- `torch`, `torchvision`
- `tqdm`, `prettytable`, `thop`, `einops`
- 分割任务依赖：mask / image 处理（对 MMPA-DRP 无直接迁移价值）

---

## 2. Paper-to-Code Mapping

| Paper Module | Code File | Class / Function | Purpose | Input Shape | Output Shape |
|-------------|-----------|------------------|---------|-------------|--------------|
| ACFD shared branch | `model/DCDNet.py` | `Shared` (line 344) | 从 ResNet layer1 生成 domain-agnostic / class-agnostic shared feature | `[B, 256, H, W]` | `[B, 1024, H/2, W/2]` |
| ACFD private branch | `model/DCDNet.py` | `Private` (line 375) | 从 ResNet layer3 生成 class-specific private feature | `[B, 1024, H, W]` | `[B, 1024, H, W]` |
| MGDF (Fusion) | `model/DCDNet.py` | `Fusion` (line 408) | 动态加权融合 base + shared + private | 3× `[B, 1024, H, W]` | `[B, 1024, H, W]` |
| CAM (Modulation) | `model/DCDNet.py` | `Modulation` (line 10) | Fine-tuning 阶段用 shared 调制 private | `private [B,1024,H,W]`, `shared [B,1024,H,W]` | `[B, 1024, H, W]` |
| GRL | `model/discriminator.py` | `GradientReversal` (line 72) | 梯度反转层 | — | — |
| Discriminator | `model/discriminator.py` | `Discriminator` (line 5) | 对 shared feature 做 class classification + real/fake 判别 | `[B, 1024, H, W]` | `[B, num_classes]` |
| Segmentation CE Loss | `train.py` | `CrossEntropyLoss` | 主分割任务 | pixel-level logits / mask | scalar |
| Adversarial Loss | `train.py` | `adv_loss` (line 446-450) | 主模型通过 GRL 迷惑 discriminator | shared feature logits | scalar |
| Orthogonality Loss | `train.py` | `DiffLoss` (line 232) | 约束 shared 与 private 正交 | shared `[B,C,H,W]`, private `[B,C,H,W]` | scalar |
| Contrastive Loss | `train.py` | `PrivateContraLoss` (line 251) | Prototype-based contrastive on private branch | private feature, class label | scalar |

---

## 3. ACFD Code Analysis

### 3.1 Shared Branch (`class Shared`)

**位置**: `model/DCDNet.py:344`

```python
class Shared(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=[512, 1024], ...)
```

**逻辑**:
1. `conv_block1`: `Conv2d(256→512, 3×3)` + GroupNorm + ReLU + Dropout2d
2. `conv_block2`: `Conv2d(512→1024, 3×3)` + GroupNorm + ReLU + Dropout2d + `MaxPool2d(2)`
3. `adaptor`: `Conv2d(1024→1024, 1×1)` + GroupNorm + **Sigmoid** → 生成空间注意力
4. 输出: `x * attn`（逐元素乘法， gated feature）

**输入**: ResNet `layer1` 的输出 `[B, 256, H, W]`（如 100×100）  
**输出**: `[B, 1024, H/2, W/2]`（如 50×50）

**关键观察**: Shared branch 使用 **Sigmoid adaptor** 做自注意力加权，目的是 suppress 与任务无关的信息，保留 domain-invariant / class-agnostic 的通用表征。

---

### 3.2 Private Branch (`class Private`)

**位置**: `model/DCDNet.py:375`

```python
class Private(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=[512, 1024], ...)
```

**逻辑**:
1. `conv_block1`: `Conv2d(1024→512, 3×3)` + GroupNorm + **GELU** + Dropout2d
2. `conv_block2`: `Conv2d(512→1024, 3×3)` + GroupNorm + GELU + Dropout2d
3. `domain_attn`: `AdaptiveAvgPool2d(1)` → `Conv2d(1024→128, 1×1)` → ReLU → `Conv2d(128→1024, 1×1)` → Sigmoid → channel attention
4. 残差连接: `x * domain_attn + identity`

**输入**: ResNet `layer3` 的输出 `[B, 1024, H, W]`（如 50×50）  
**输出**: `[B, 1024, H, W]`

**关键观察**: Private branch 使用 **channel-wise Sigmoid attention**（squeeze-excitation 风格）+ 残差连接。GELU 激活函数（而非 Shared 的 ReLU）暗示该分支承担更复杂的非线性映射。

---

### 3.3 Adversarial Loss

**位置**: `train.py:446-450`

```python
dis_s_out_training = discriminator(output["S_shared"])
s_adv_loss = adversarial_loss(dis_s_out_training, class_sample.to(device))
dis_q_out_training = discriminator(output["Q_shared"])
q_adv_loss = adversarial_loss(dis_q_out_training, class_sample.to(device))
adv_loss = s_adv_loss + q_adv_loss
```

**机制**:
- `Discriminator` 内部包含 `GradientReversal(args.lambda_)`（`model/discriminator.py:9`）
- Discriminator 试图将 `S_shared` / `Q_shared` **按类别分类**（输出 `[B, num_classes]`）
- 同时 discriminator 还做 real/fake 判别：真实 shared feature 用 class label，随机噪声 `z_fake` 用 label 0
- **GRL 使得主模型梯度方向与 discriminator 相反**：主模型希望生成让 discriminator 难以分类的 shared feature，从而将类别特异性信息“驱赶”到 private branch

**注意**: 这里的 adversarial loss 不是传统的 domain classification（源域/目标域二分类），而是 **class-level adversarial**（多类别分类 + GRL）。迁移到 MMPA-DRP 时需注意此差异。

---

### 3.4 Contrastive Loss (`PrivateContraLoss`)

**位置**: `train.py:251-313`

```python
class PrivateContraLoss(nn.Module):
    def __init__(self, num_classes=20, feature_dim=1024, temperature=0.07, momentum=0.99):
        self.register_buffer("prototypes", init_scale * torch.randn(num_classes, feature_dim))
```

**逻辑**:
1. `gmp`: `AdaptiveMaxPool2d((1,1))` 将 private feature `[B,1024,H,W]` → `[B,1024]`
2. `_update_prototypes`: 按 class label 聚合 feature，动量更新 prototype bank
3. `_contrastive_loss`: 计算 normalized feature 与 prototype 的 cosine similarity / temperature，做 smoothed cross-entropy（label smoothing: 0.9 + 0.1/num_classes）

**关键观察**: 这是 **prototype-based supervised contrastive learning**，不是 SimCLR/InfoNCE 式的 instance-level contrastive。迁移到 MMPA-DRP 时，可直接复用 prototype bank 机制，但需将 `AdaptiveMaxPool2d` 替换为 program-token 的 pooling。

---

### 3.5 Orthogonality Loss (`DiffLoss`)

**位置**: `train.py:232-248`

```python
class DiffLoss(torch.nn.Module):
    def forward(self, D1, D2):
        D1 = D1.view(D1.size(0), -1)
        D1_norm = torch.norm(D1, p=2, dim=1, keepdim=True).detach()
        D1_norm = D1.div(D1_norm.expand_as(D1) + 1e-6)
        # 同理 D2_norm
        return torch.mean((D1_norm.mm(D2_norm.t()).pow(2)))
```

**逻辑**: 将 shared 和 private feature 分别 flatten + L2 normalize，计算 batch 内所有样本对的余弦相似度平方，取均值。

**目标**: 最小化 shared 与 private 之间的相关性，迫使两者捕获互补（不重叠）的信息。

**直接可迁移**: 该 loss 对 tensor shape 无特殊要求（只要 view 后同维度即可），可直接用于 MMPA-DRP 的 program-token 解耦。

---

## 4. MGDF Code Analysis

### 4.1 模块定位

**位置**: `model/DCDNet.py:408` (`class Fusion`)

**逻辑**:
```python
class Fusion(nn.Module):
    def forward(self, base_feat, share_feat, private_feat, finetuning):
        combined = torch.cat([base_feat, share_feat, private_feat], dim=1)  # [B, 3072, H, W]
        reduced = self.channel_reduce(combined)                              # [B, 64, H, W]  (1×1 conv)
        logits = self.weight_generator(reduced)                              # [B, 3, H, W]   (3×3 conv)
        weights = F.softmax(logits / self.temperature, dim=1)               # temperature-softmax
        weights = torch.clamp(weights, min=self.min_weight)                 # floor=0.05
        weights = weights / weights.sum(dim=1, keepdim=True)                # renormalize
        w_base, w_share, w_private = weights[:,0:1], weights[:,1:2], weights[:,2:3]
        fused = w_base * base_feat + w_share * share_feat + w_private * private_feat
        return self.enhance(fused) + fused                                  # residual enhancement
```

### 4.2 关键设计

| 组件 | 实现 | 作用 |
|------|------|------|
| Channel Reduce | `Conv2d(3072→64, 1×1)` | 降维，减少后续 3×3 conv 计算量 |
| Weight Generator | `Conv2d(64→3, 3×3)` | 生成三个分支的空间权重图 |
| Temperature Softmax | `softmax(logits / 0.7)` | 控制权重分布的 sharpness |
| Min Weight Clamp | `clamp(min=0.05)` | 防止某个分支权重坍缩到 0 |
| Residual Enhance | `Conv2d(1024→1024, 1×1)` + GN + ReLU | 对融合结果做轻量增强 |

### 4.3 空间维度依赖性

MGDF 的 `weight_generator` 输出 `[B, 3, H, W]`，意味着 **每个空间位置 (h,w) 有独立的融合权重**。这强烈依赖于图像的 H×W 空间结构。

**对 MMPA-DRP 的启示**:
- Gene expression / program tokens 是 **1D 向量**（无 H×W），不能直接使用空间权重矩阵
- 可改造为 **channel-wise / program-wise 权重**: `Conv1d` 或 `Linear` 生成 `[B, 3, K]` 或 `[B, 3, d]` 权重
- 或简化为 **sample-level 标量权重**: 每个样本一个 `(w_base, w_share, w_private)` 三元组

---

## 5. CAM Code Analysis

### 5.1 模块定位

**位置**: `model/DCDNet.py:10` (`class Modulation`)

**逻辑**:
```python
class Modulation(nn.Module):
    def forward(self, private, shared):
        combined = torch.cat([private, shared], dim=1)          # [B, 2048, H, W]
        interaction = self.interaction(combined)                # [B, 128, H, W]  (3×3 conv)
        params = self.param_generator(interaction)              # [B, 2048, H, W] (1×1 conv + Tanh)
        gamma, beta = torch.chunk(params, 2, dim=1)             # 各 [B, 1024, H, W]
        gamma = 0.1 * gamma                                     # 训练稳定性缩放
        beta = 0.1 * beta
        modulated = private * (1 + gamma) + beta
        return modulated
```

### 5.2 关键设计

- **Affine Transform**: `private * (1 + gamma) + beta`，类似 BN 的 scale/shift，但由 shared feature 动态生成
- **Tanh + 0.1 缩放**: 限制 modulation 幅度，防止早期训练不稳定
- **Spatial-aware**: gamma/beta 都是 `[B, 1024, H, W]`，每个空间位置独立调制

### 5.3 使用阶段

**仅在 `finetuning=True` 时启用** (`DCDNet.forward` line 70-73, 89-93):

```python
if self.fintuning == False:
    output['S_private'] = self.private(s_3)
else:
    output['S_private'] = self.modulator(self.private(s_3), output['S_shared'])
```

- `train.py`: `fintuning=False`（默认），**不使用 CAM**
- `finetuning.py`: `fintuning=True`，**使用 CAM**

### 5.4 对 MMPA-DRP 的建议

**建议暂缓使用 CAM**，原因：
1. CAM 依赖 H×W 空间维度，program tokens 是 1D 向量，直接迁移需重新设计
2. CAM 在 fine-tuning 阶段才启用，说明它是一个**锦上添花**的模块，不是核心解耦机制
3. 过早引入 CAM 可能污染 response-relevant representation（shared 直接调制 private）
4. 建议先验证 ACFD（shared/private + orthogonality + adversarial）的有效性，再在后续阶段考虑是否加入 program-wise modulation

---

## 6. Loss and Optimization Analysis

### 6.1 Loss 定义与权重

| Loss | 类/函数 | 权重 (默认) | 输入 | 说明 |
|------|---------|------------|------|------|
| Segmentation CE | `CrossEntropyLoss` | `ce_loss_reg=1.0` | pixel-level logits & mask | 主任务分割损失 |
| Adversarial | `CrossEntropyLoss` + GRL | `adv_loss_reg=0.005` | shared feature, class label | 迷惑 discriminator |
| Orthogonality | `DiffLoss` | `diff_loss_reg=0.1` | shared & private features | 正交约束 |
| Contrastive | `PrivateContraLoss` | `contr_loss_reg=0.05` | private feature, class label | Prototype-based contrastive |
| Discriminator Real | `CrossEntropyLoss` | — | shared feature (detached), class label | 判别真实 shared feature |
| Discriminator Fake | `CrossEntropyLoss` | — | random noise, label=0 | 判别 fake |

### 6.2 Loss 组合公式（主模型）

```python
loss = 1.0 * ce_loss + 0.005 * adv_loss + 0.1 * diff_loss + 0.05 * contr_loss
```

其中:
```python
ce_loss = CE(Q_out, mask_q) + CE(self_out, mask_q) + 0.2*CE(S_out, mask_s) + 0.4*CE(S_out_1, mask_s)
adv_loss = CE(disc(S_shared), class) + CE(disc(Q_shared), class)   # GRL 反向
```

### 6.3 Optimizer 配置

| 组件 | Optimizer | LR | 其他参数 |
|------|-----------|-----|----------|
| 主模型 (DCDNet) | `SGD` | `lr=0.001` | momentum=0.9, weight_decay=5e-4 |
| Discriminator | `Adam` | `d_lr=0.0001` | weight_decay=0.01 |

### 6.4 是否交替训练

**是，交替优化** (`train.py:433-492`):

```python
# Main model steps (s_steps=1)
for _ in range(args.s_steps):
    output = model(...)
    loss = ...  # 组合 ce + adv + diff + contr
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Discriminator steps (d_steps=1)
for _ in range(args.d_steps):
    d_optimizer.zero_grad()
    discriminator.zero_grad()
    # Real loss
    dis_q_real_out = discriminator(output['S_shared'].detach())
    dis_q_real_loss = CE(dis_q_real_out, class_sample)
    dis_q_real_loss.backward(retain_graph=True)
    # Fake loss
    z_fake = torch.randn(...)
    dis_fake_out = discriminator(z_fake)
    dis_fake_loss = CE(dis_fake_out, zeros)
    dis_fake_loss.backward()
    d_optimizer.step()
```

**关键细节**:
- Discriminator 使用 `.detach()` 隔离主模型梯度
- `retain_graph=True` 用于 real loss（因为 fake loss 还要用同一 batch）
- Fake data 是随机高斯噪声，label=0 表示 fake class

### 6.5 Fine-tuning 阶段 Loss

`finetuning.py:279-281`:
```python
loss = CE(Q_out, mask_q) + CE(out_refine, mask_q) + CE(self_out, mask_q) + \
       CE(S_out, mask_s)*0.2 + CE(S_out_1, mask_s)*0.4 + \
       CE(Q_out_2, mask_q)*0.1 + CE(S_out_2, mask_s)*0.1 + \
       CE(Q_out_3, mask_q)*0.1 + CE(S_out_3, mask_s)*0.1
```

- **仅 segmentation CE**，无 adversarial / diff / contrastive
- 多轮 refinement 输出（`out_refine`, `Q_out_2/3`, `S_out_2/3`）都有监督

---

## 7. Transferable Components to MMPA-DRP

### 7.1 概念映射表

| DCDNet Concept | DCDNet 实现 | MMPA-DRP Adaptation | 可行性 |
|---------------|-------------|---------------------|--------|
| Base feature (ResNet layer3) | `s_3 / q_3` | `z_base` from `MolecularProgramEncoder` | ✅ 直接对应 |
| Shared feature | `Shared(s_1)` | `z_domain` 或 `h_domain` (domain-nuisance) | ✅ 需新增 projection head |
| Private feature | `Private(s_3)` | `z_response` 或 `h_response` (response-relevant) | ✅ 需新增 projection head |
| GRL | `GradientReversal` | 复用 MMPA 已有的 `GradientReversal` | ✅ 已存在 |
| Domain/Class Discriminator | `Discriminator` | Source/Target domain classifier (binary) | ⚠️ 需改造为多任务或二分类 |
| Adversarial loss | `CE(disc(shared), label)` with GRL | `BCEWithLogitsLoss` on response branch | ✅ 直接迁移 |
| Orthogonality loss | `DiffLoss` | 复用到 `z_response` 与 `z_domain` | ✅ 直接可用 |
| Contrastive loss | `PrivateContraLoss` (prototype-based) | Drug-response prototype contrastive | ✅ 需改 label 为 response |
| MGDF | `Fusion` (spatial weights) | Program-wise / feature-wise dynamic fusion | ⚠️ 需去除空间维度 |
| CAM | `Modulation` (spatial affine) | Program-wise modulation | ⚠️ 建议暂缓 |

### 7.2 可直接迁移的 Loss

1. **`DiffLoss` (Orthogonality)**
   - 代码完全通用，只需保证两个输入 tensor 的 batch size 和 feature dim 一致
   - 在 MMPA 中: `DiffLoss(z_response, z_domain)` 或 `DiffLoss(h_response, h_domain)`

2. **`GradientReversal`**
   - MMPA-DRP 已有类似实现 (`mmpa_drp/models.py:GradientReversal`)
   - 可直接复用

3. **Adversarial Loss 思想**
   - 对 `h_response` 加 GRL + domain discriminator，使其难以区分 source/target
   - 对 `h_domain` 加正常 domain classifier，使其吸收 domain 差异

### 7.3 需改造后迁移的模块

1. **`PrivateContraLoss`**
   - 需将 `AdaptiveMaxPool2d` 替换为 program-token pooling (e.g., `z.mean(dim=1)` 或 `q-weighted sum`)
   - Prototype bank 机制可直接复用，但 class label 需改为 drug response label 或 tissue label

2. **`Fusion` (MGDF)**
   - 去除空间维度 H×W
   - 改为 `Linear(3*d → d//reduction)` → `Linear(d//reduction → 3)` → softmax → weighted sum
   - 或更简化为 scalar weight per sample: `Linear(3*d → 3)`

3. **`Discriminator`**
   - DCDNet 的 discriminator 输出 `num_classes` 个 logits（class classification）
   - MMPA-DRP 需要 domain discriminator（binary: source/target），结构更简单

---

## 8. Recommended MMPA Implementation Plan

### Step 3A-1: Disentangled Program Encoder

在 `MolecularProgramEncoder` 之后（或内部）新增两个 projection heads：

```
x: [B, G]
  └── backbone ──→ hidden: [B, H]
       ├── program_proj ──→ z: [B, K, d], q: [B, K], h_pool: [B, d]
       │
       ├── response_head(z) ──→ z_response: [B, K, d]  (response-relevant)
       │
       └── domain_head(z)  ──→ z_domain:  [B, K, d]  (domain-nuisance)
```

- `response_head` 和 `domain_head` 各为 `Linear(d → d)` 或轻量 MLP
- `h_response = sum_k q_k * z_response_k`
- `h_domain  = sum_k q_k * z_domain_k`

### Step 3A-2: Domain-adversarial Response Branch

对 `h_response` 施加 GRL + domain discriminator：
- GRL: 使 `h_response` 的梯度反向，主模型学习**去除** domain 信息
- Domain Discriminator: 二分类（source=0 / target=1），试图从 `h_response` 中判别 domain
- Loss: `BCEWithLogitsLoss(disc(GRL(h_response)), domain_label)`

### Step 3A-3: Domain-nuisance Branch

对 `h_domain` 施加正常的 domain classifier（无 GRL）：
- 目标: 让 `h_domain` **保留** domain 信息
- Loss: `BCEWithLogitsLoss(disc(h_domain), domain_label)`
- 这样 domain-specific variance 被显式吸收到 `h_domain`，不干扰 `h_response`

### Step 3A-4: Orthogonality Loss

```python
L_ortho = DiffLoss(z_response, z_domain)
```
- 最小化两个 program-token 集合的相关性
- 迫使 response-relevant 和 domain-nuisance 信息互不重叠

### Step 3A-5: Optional Response Contrastive Loss

在 source domain (GDSC/CCLE) 上：
- 根据 drug response label（敏感/耐药）构造 prototype bank
- `L_contrast = PrivateContraLoss(z_response, response_label)`
- 注意: 需将 `AdaptiveMaxPool2d` 替换为适用于 1D program tokens 的 pooling

### Step 3A-6: Optional Program-wise Dynamic Fusion

将 `h_pool` (原始)、`h_response`、`h_domain` 做轻量 fusion：
- 可借鉴 MGDF 思想，但去除空间维度
- 生成 sample-level 或 program-level 权重
- **初期建议**: 不要让 `h_domain` 直接输入 prediction，只用 `h_response` 做 drug response prediction，避免 domain 信息泄漏

---

## 9. Risks and Non-transferable Parts

### 9.1 明确不可迁移的部分

1. **Pixel-level Segmentation Loss**
   - DCDNet 是图像分割，CE loss 作用于每个像素 `[B, C, H, W]` vs `[B, H, W]` mask
   - MMPA-DRP 是 drug response prediction，输出是 `[B, drug_num, 1]` 的 sigmoid/BCE
   - 分割 mask 相关的所有逻辑（`masked_average_pooling`, `SSP_func`, `iter_BFP`）**不可直接迁移**

2. **空间维度依赖**
   - `Shared` / `Private` 的 `Conv2d`、`Dropout2d`、`MaxPool2d` 依赖 H×W
   - `Fusion` 的 `Conv2d(3×3)` 生成空间权重图，依赖 H×W
   - `Modulation` (CAM) 的 gamma/beta 是 `[B, C, H, W]`，依赖 H×W
   - 迁移时必须将 `Conv2d` 替换为 `Linear` 或 `Conv1d`

3. **Few-shot Support/Query 结构**
   - DCDNet 的 `forward` 接收 `img_s_list` (support) 和 `img_q` (query)，是 few-shot learning 框架
   - MMPA-DRP 是标准的 supervised/domain adaptation，无 support/query 概念
   - `masked_average_pooling`、`FP/BP prototype` 等 few-shot 逻辑 **需彻底移除**

4. **Real/Fake Discriminator 设计**
   - DCDNet discriminator 同时做 class classification + real/fake 判别
   - 对 MMPA-DRP 过于复杂，建议简化为纯 binary domain discriminator

### 9.2 方法论风险

1. **不应声称严格因果识别**
   - DCDNet 的 "causal disentanglement" 是启发式的（heuristic），基于 adversarial + orthogonality + contrastive
   - 没有 do-calculus、IV、后门准则等严格因果推断
   - 在 MMPA-DRP 中应称为 **causal-inspired disentanglement** 或 **response-domain feature decomposition**

2. **Domain-nuisance branch 可能吸收有用信息**
   - 正交约束 + adversarial 可能无法完全阻止 domain 信息泄漏到 response branch
   - 需通过消融实验验证：去掉 domain branch 后指标是否下降

3. **CAM 可能污染 response representation**
   - 若过早引入 CAM（shared 调制 private），domain 信息可能通过 modulation 泄漏
   - 建议先验证无 CAM 的 ACFD 基线，再决定是否引入

---

## 10. Quick Reference: File Locations

| 要找的内容 | 文件 | 行号 |
|-----------|------|------|
| Shared branch | `model/DCDNet.py` | 344-372 |
| Private branch | `model/DCDNet.py` | 375-405 |
| Fusion (MGDF) | `model/DCDNet.py` | 408-444 |
| Modulation (CAM) | `model/DCDNet.py` | 10-37 |
| GRL | `model/discriminator.py` | 48-78 |
| Discriminator | `model/discriminator.py` | 5-32 |
| DiffLoss (Ortho) | `train.py` | 232-248 |
| PrivateContraLoss | `train.py` | 251-313 |
| 主模型 loss 组合 | `train.py` | 458-459 |
| Discriminator 训练 | `train.py` | 469-492 |
| Fine-tuning loss | `finetuning.py` | 279-281 |
| Fine-tuning CAM 开关 | `model/DCDNet.py` | 70-73, 89-93 |
