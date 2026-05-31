# Stage 3 基线方法深度分析与改进方案

## 一、现有方法完整流程

```
Epoch 开始
  ├── get_featuers() 提取全量 source [N,9,D] 和 target [M,9,D]
  ├── get_gene_feat_sim() 每药物 cosine KNN → topk_idx [N,9,K], topk_sim [N,9,K]
  └── 重建 dataloader（绑定 KNN 信息到每个 sample）

每个 batch
  ├── Forward source → fusion_feat [B*9, D]
  ├── encode_contrastive_feature() → anchor_z（VCB=False 时直接原 feature）
  ├── pos = target_z[topk_idx]              [B*9, K, Z]
  ├── neg = target_z.reshape(-1, Z)         [M*9, Z]
  └── contrastive_loss(anchor_z, pos, neg, temperature=0.1)
```

---

## 二、核心问题分析

### 问题 1：Loss 形式错误 —— 用的是 multi-label BCE，不是标准 InfoNCE

**当前代码** (`myloss.py`):
```python
pos_sim = torch.bmm(pos_samples, anchor.unsqueeze(2)).squeeze(2) / temperature  # [B, K]
neg_sim = torch.matmul(anchor, neg_samples.T) / temperature                      # [B, M]
logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, K+M]
labels = torch.zeros_like(logits)
labels[:, :K] = 1
loss = F.binary_cross_entropy_with_logits(logits, labels)
```

**数学本质**：把 K+M 个样本当作 K+M 个独立的二分类。
```
L = -Σ_k log(σ(pos_k/τ)) - Σ_m log(1-σ(neg_m/τ))
```

**为什么这是问题**：
- 正样本之间没有竞争（每个正样本独立优化）
- 负样本之间没有竞争
- 对 easy negatives 梯度饱和（σ→0 时梯度≈0）
- 没有利用"哪个是正类"的结构性信息

**标准 InfoNCE 应该是**：
```python
# 所有正样本和负样本在一个 softmax 中竞争
pos_sim = torch.bmm(pos_samples, anchor.unsqueeze(2)).squeeze(2) / tau  # [B, K]
neg_sim = torch.matmul(anchor, neg_samples.T) / tau                      # [B, M]

# 对每个 anchor，在 K+M 个候选中，前 K 个是正类
logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, K+M]
# 标准做法：取正样本的平均或最大
pos_sim_pooled = pos_sim.mean(dim=1)  # [B]  或 logsumexp
neg_sim_logsumexp = torch.logsumexp(neg_sim, dim=1)  # [B]

# InfoNCE = -log( exp(pos/τ) / (exp(pos/τ) + Σ_neg exp(neg/τ)) )
loss = -pos_sim_pooled + torch.log(torch.exp(pos_sim_pooled) + torch.sum(torch.exp(neg_sim), dim=1))
```

标准 InfoNCE 的优势：
- 所有样本在一个 softmax 中竞争
- 正样本的相似度会相互增强
- 困难负样本自动获得更高的权重
- 对 easy negatives 不会梯度饱和

---

### 问题 2：正样本污染（False Negatives）

```python
neg_samples = all_target_z.reshape(-1, Z)  # [M*9, Z]
```

`neg_samples` 包含了**所有 target 的所有药物**，其中必然包含：
- 没进 top-K 但实际上很相似的 target 细胞（同药物假负样本）
- 其他药物下和 anchor 结构相似的 target 细胞（跨药物假负样本）

后果：模型被迫推远"本应相似"的样本，表示空间被扭曲。

---

### 问题 3：负样本过于简单

所有 M*9 个 target 都作为负样本，其中大部分和 anchor 明显不相似。
模型很容易区分，学习信号弱。

对比学习的关键在于**困难负样本**——那些"看起来像正样本"的负样本。推远这些样本才能学到好的决策边界。

---

### 问题 4：Temperature 固定

所有 anchor 共享 temperature=0.1。但实际上：
- **简单 anchor**（正样本明显比负样本相似）：需要更高 τ 来增加困难负样本的梯度
- **困难 anchor**（正样本和负样本接近）：需要更低 τ 来 sharpen 分布

---

### 问题 5：Top-K 相似度分数未利用

`get_gene_feat_sim()` 计算了具体的 cosine 分数 `topk_sim`，但 legacy contrastive loss 完全没用它。

相似度更高的正样本应该对 loss 有更大的贡献（因为它们更"确定"是邻居）。

---

### 问题 6：KNN 更新频率低

每 epoch 更新一次 KNN，但模型参数在训练过程中不断变化。
- Epoch 早期：模型变化快，KNN 很快过时
- Epoch 后期：模型变化慢，KNN 较稳定

这种"全局静态 KNN + 局部动态更新"的错位导致优化不稳定。

---

### 问题 7：K 值固定

不同药物的 target 分布密度不同：
- 某些药物（如 Etoposide）可能有很多相似的 target 细胞 → K=3 太少，丢失信息
- 某些药物（如 Temozolomide）可能只有少数相似 → K=3 太多，引入噪声

---

### 问题 8：跨药物信息未利用

9 种药物完全独立做 KNN。但：
- Docetaxel 和 Paclitaxel 都是 Taxane，结构高度相似
- 如果模型对一种 Taxane 学到了好的表示，应该能帮助另一种

---

## 三、改进方案（按优先级排序）

### 方案 A：标准 InfoNCE 替换 BCE（最基础，预期提升最大）

将 `myloss.py` 的 `contrastive_loss` 改为标准 InfoNCE。

### 方案 B：困难负样本挖掘（Hard Negative Mining）

不把所有 target 作为负样本，只选最困难的 N 个。

### 方案 C：正样本相似度加权

利用 `topk_sim` 对正样本加权。

### 方案 D：动态 Temperature

根据 anchor 难度自适应调整 temperature。

---

## 四、建议实施顺序

1. **先做方案 A（InfoNCE）** —— 最基础，可能带来 0.005-0.01 提升
2. **方案 A + 方案 B（Hard Negative）** —— 组合使用
3. 如果前两个有效，再加方案 C（加权）

方案 D（动态 temperature）效果通常不如前三个明显，可以最后尝试。
