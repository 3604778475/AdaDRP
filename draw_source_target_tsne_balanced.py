"""
对 source_features.npy 与 target_features.npy 做“每个组织每个域最多 100 个样本”的均衡采样，
然后画 T-SNE 域分布图。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 必须先导入 utility 避免循环导入
from utility import *
from dataload import get_unlabeled_dataloaders
import config
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

SRC_PATH = "source_features.npy"
TGT_PATH = "target_features.npy"
SAVE_DIR = "outputs/figures"
SAVE_PATH = os.path.join(SAVE_DIR, "source_target_tsne_balanced.png")
MAX_PER_TISSUE_DOMAIN = 100


def subsample(features, labels, max_per_group):
    """按标签分层采样，每类最多 max_per_group 个。"""
    sampled_idx = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if len(idx) > max_per_group:
            idx = np.random.choice(idx, size=max_per_group, replace=False)
        sampled_idx.append(idx)
    sampled_idx = np.concatenate(sampled_idx)
    return features[sampled_idx], labels[sampled_idx], sampled_idx


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.random.seed(42)

    # 1. 加载已保存的特征
    src = np.load(SRC_PATH)
    tgt = np.load(TGT_PATH)
    print(f"Source: {src.shape}, Target: {tgt.shape}")

    # 2. 重建 tissue label（与 save_feat_program 使用的 s_train/t_train 顺序一致）
    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)
    s_loaders, t_loaders = get_unlabeled_dataloaders(
        gex_features_df=gex_features_df, seed=2025, test_ratio=0.25, batch_size=128
    )

    def collect_labels(loader):
        labels = []
        for batch in loader:
            labels.append(batch[1].numpy())
        return np.concatenate(labels)

    src_labels = collect_labels(s_loaders[0])
    tgt_labels = collect_labels(t_loaders[0])
    print(f"Source labels: {len(src_labels)}, Target labels: {len(tgt_labels)}")

    # 3. 每个组织每个域最多采样 100 个
    src_sub, src_lbl_sub, src_idx = subsample(src, src_labels, MAX_PER_TISSUE_DOMAIN)
    tgt_sub, tgt_lbl_sub, tgt_idx = subsample(tgt, tgt_labels, MAX_PER_TISSUE_DOMAIN)

    features = np.vstack([src_sub, tgt_sub])
    tissue_labels = np.concatenate([src_lbl_sub, tgt_lbl_sub])
    domains = np.array([0] * len(src_sub) + [1] * len(tgt_sub))
    domain_names = np.array(["Source" if d == 0 else "Target" for d in domains])

    print(f"Subsampling: Source={len(src_sub)}, Target={len(tgt_sub)}, Total={len(features)}")

    # 4. T-SNE
    print("正在计算 T-SNE ...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                max_iter=1000, random_state=42)
    emb = tsne.fit_transform(features)

    # 5. 绘图：左图按域着色，右图按组织着色
    reverse_map = {v: k for k, v in config.tissue_map.items()}
    tissue_names = np.array([reverse_map.get(l, str(l)) for l in tissue_labels])

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # 左图：Domain
    ax = axes[0]
    sns.scatterplot(
        x=emb[:, 0], y=emb[:, 1], hue=domain_names,
        palette={"Source": "#3498db", "Target": "#e74c3c"},
        alpha=0.7, s=35, ax=ax
    )
    ax.set_title("Source vs Target (balanced: ≤100 per tissue/domain)", fontsize=15)
    ax.set_xlabel("T-SNE 1", fontsize=12)
    ax.set_ylabel("T-SNE 2", fontsize=12)
    ax.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc='upper left')

    # 右图：Tissue（只显示同时在两个域出现的组织，避免颜色过多）
    ax = axes[1]
    source_tissues = set(src_lbl_sub)
    target_tissues = set(tgt_lbl_sub)
    common_tissues = sorted(source_tissues & target_tissues)
    print(f"共同组织数: {len(common_tissues)}")

    # 只给共同组织上色，其他显示为灰色
    palette = sns.color_palette("tab20", len(common_tissues))
    color_map = {tissue_names[i]: palette[common_tissues.index(tissue_labels[i])]
                 if tissue_labels[i] in common_tissues else (0.7, 0.7, 0.7, 1.0)
                 for i in range(len(tissue_labels))}
    point_colors = [color_map[name] for name in tissue_names]

    sns.scatterplot(
        x=emb[:, 0], y=emb[:, 1], hue=tissue_names,
        palette=color_map, alpha=0.7, s=35, ax=ax, legend=False
    )
    # 手动添加 legend，只显示共同组织
    for idx, tissue_id in enumerate(common_tissues):
        ax.scatter([], [], c=[palette[idx]], label=reverse_map[tissue_id], s=60)
    ax.set_title("Colored by Tissue (common tissues only)", fontsize=15)
    ax.set_xlabel("T-SNE 1", fontsize=12)
    ax.set_ylabel("T-SNE 2", fontsize=12)
    ax.legend(title="Tissue", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {SAVE_PATH}")


if __name__ == "__main__":
    main()
