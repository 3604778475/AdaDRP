"""
对 source_features.npy 与 target_features.npy 画 T-SNE 对比图。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

SRC_PATH = "source_features.npy"
TGT_PATH = "target_features.npy"
SAVE_DIR = "outputs/figures"
SAVE_PATH = os.path.join(SAVE_DIR, "source_target_tsne.png")

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    src = np.load(SRC_PATH)
    tgt = np.load(TGT_PATH)
    print(f"Source: {src.shape}, Target: {tgt.shape}")

    features = np.vstack([src, tgt])
    labels = np.array([0] * len(src) + [1] * len(tgt))
    label_names = ["Source" if l == 0 else "Target" for l in labels]

    print("正在计算 T-SNE ...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                max_iter=1000, random_state=42)
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=emb[:, 0], y=emb[:, 1], hue=label_names,
        palette={"Source": "#3498db", "Target": "#e74c3c"},
        alpha=0.6, s=20
    )
    plt.title("Source vs Target Feature T-SNE", fontsize=16)
    plt.xlabel("T-SNE 1", fontsize=12)
    plt.ylabel("T-SNE 2", fontsize=12)
    plt.legend(title="Domain", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {SAVE_PATH}")

if __name__ == "__main__":
    main()
