"""
T-SNE 可视化脚本
用于可视化 MACB-DRP/MMPA-DRP 模型提取的特征
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_tsne(features, labels, title="T-SNE", save_path="tsne.png",
              palette="tab10", alpha=0.7, s=20):
    """
    绘制 T-SNE 图

    Args:
        features: [N, D] numpy array
        labels: [N] list or array of int/str
        title: 图标题
        save_path: 保存路径
        palette: seaborn 颜色方案
        alpha: 点透明度
        s: 点大小
    """
    # T-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                max_iter=1000, random_state=42)
    embeddings = tsne.fit_transform(features)

    # 绘图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1],
                    hue=labels, palette=palette, alpha=alpha, s=s)
    plt.title(title, fontsize=16)
    plt.xlabel("T-SNE 1", fontsize=12)
    plt.ylabel("T-SNE 2", fontsize=12)
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


def extract_and_plot_tsne(model, source_loader, target_loader, device,
                          feature_type="gene_feat", save_dir="outputs/tsne"):
    """
    从模型中提取特征并画 T-SNE

    feature_type 可选:
        - "h_pool": encoder 输出的 h_pool
        - "h_shared": domain separation 的 shared 表示
        - "h_private": domain separation 的 private 表示
        - "gene_feat": Stage2/3 中用于 OT 的基因特征
        - "fused_feature": 融合特征 [B, D, H]，会 reshape 后使用
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    all_feats = []
    all_domains = []  # 0=source, 1=target
    all_tissues = []
    all_drugs = []    # 用于 fused_feature

    with torch.no_grad():
        # 源域
        for batch in source_loader:
            x = batch[0].to(device)
            tissue = batch[3].to(device) if len(batch) > 3 else None
            outputs = model(x, drug_feat=None, tissue_idx=tissue,
                            return_dict=True, return_encoder_out=True,
                            return_fused_feature=True)

            feats = _get_features(outputs, feature_type, drug_idx=0)
            all_feats.append(feats.cpu().numpy())
            all_domains.extend([0] * feats.size(0))
            if tissue is not None:
                all_tissues.extend(tissue.cpu().tolist())

        # 目标域
        for batch in target_loader:
            x = batch[0].to(device)
            tissue = batch[3].to(device) if len(batch) > 3 else None
            outputs = model(x, drug_feat=None, tissue_idx=tissue,
                            return_dict=True, return_encoder_out=True,
                            return_fused_feature=True)

            feats = _get_features(outputs, feature_type, drug_idx=0)
            all_feats.append(feats.cpu().numpy())
            all_domains.extend([1] * feats.size(0))
            if tissue is not None:
                all_tissues.extend(tissue.cpu().tolist())

    all_feats = np.concatenate(all_feats, axis=0)

    # 1. 按域（源域/目标域）画 T-SNE
    domain_labels = ["Source" if d == 0 else "Target" for d in all_domains]
    plot_tsne(all_feats, domain_labels,
              title=f"T-SNE by Domain ({feature_type})",
              save_path=os.path.join(save_dir, f"tsne_{feature_type}_domain.png"),
              palette={"Source": "#3498db", "Target": "#e74c3c"})

    # 2. 按组织类型画 T-SNE
    if len(all_tissues) == len(all_feats):
        # 假设有 tissue_map 可以映射 tissue id 到名称
        plot_tsne(all_feats, all_tissues,
                  title=f"T-SNE by Tissue ({feature_type})",
                  save_path=os.path.join(save_dir, f"tsne_{feature_type}_tissue.png"),
                  palette="tab20")


def _get_features(outputs, feature_type, drug_idx=0):
    """从模型输出中提取指定特征"""
    if feature_type == "h_pool":
        return outputs["encoder_out"]["h"]
    elif feature_type == "h_shared":
        return outputs["encoder_out"]["h_shared"]
    elif feature_type == "h_private":
        return outputs["encoder_out"]["h_private"]
    elif feature_type == "gene_feat":
        return outputs["gene_feat"]
    elif feature_type == "fused_feature":
        # [B, D, H] -> 取第 drug_idx 种药物
        feat = outputs["fused_feature"]
        return feat[:, drug_idx, :]
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default=None,
                        help="预存特征 .npz 路径（可选）")
    parser.add_argument("--feature_type", type=str, default="h_shared",
                        choices=["h_pool", "h_shared", "h_private", "gene_feat", "fused_feature"],
                        help="要可视化的特征类型")
    parser.add_argument("--save_dir", type=str, default="outputs/tsne")
    parser.add_argument("--perplexity", type=float, default=30)
    args = parser.parse_args()

    # 如果提供了 .npz 文件，直接加载
    if args.feature_path is not None:
        data = np.load(args.feature_path)
        features = data["features"]
        labels = data["labels"]
        title = os.path.basename(args.feature_path).replace(".npz", "")
        plot_tsne(features, labels, title=title,
                  save_path=os.path.join(args.save_dir, f"tsne_{title}.png"))
    else:
        print("请提供 --feature_path，或修改脚本直接传入 model 和 dataloader")


if __name__ == "__main__":
    main()
