"""
实验完成后效果图绘制脚本（稳定版）
支持：
  1. Stage1 特征的 T-SNE 可视化
  2. 药物级性能柱状图（AUC/AUPR/F1/ACC）
  3. Stage2/3 预存特征的 T-SNE 可视化

用法：
    /d/anaconda/envs/macb_drp/python.exe visualize_results.py
"""
import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ================= 路径配置 =================
STAGE1_CKPT = "checkpoints/stage1/stage1_model.pt"
STAGE2_FEATURE_PATTERN = "outputs/saved_features/stage2_fold*_tcga.npz"  # 自动匹配所有 fold
RESULTS_CSV_PATTERN = "results/fold-*/*.csv"
SAVE_DIR = "outputs/figures"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# ===========================================

# 必须先导入 utility 避免循环导入
from utility import *
from dataload import get_unlabeled_dataloaders
from mmpa_drp.models import ProgramTissueClassifier
import config


def set_plot_style():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")


def plot_tsne(features, labels, title, save_path, palette="tab10"):
    """画 T-SNE 图"""
    print(f"  正在计算 T-SNE: {title}")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                max_iter=1000, random_state=42)
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    if len(np.unique(labels)) <= 2 and isinstance(labels[0], (int, np.integer)):
        label_names = ["Source" if l == 0 else "Target" for l in labels]
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=label_names,
                        palette={"Source": "#3498db", "Target": "#e74c3c"},
                        alpha=0.6, s=20)
    else:
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=labels,
                        palette=palette, alpha=0.6, s=20)

    plt.title(title, fontsize=16)
    plt.xlabel("T-SNE 1", fontsize=12)
    plt.ylabel("T-SNE 2", fontsize=12)
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def load_stage1_model(checkpoint_path, device):
    """加载 Stage1 模型"""
    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)
    input_dim = gex_features_df.shape[1] - 1

    s_loaders, t_loaders = get_unlabeled_dataloaders(
        gex_features_df=gex_features_df, seed=2025, test_ratio=0.25, batch_size=64
    )

    model = ProgramTissueClassifier(
        input_dim=input_dim,
        num_tissues=len(config.tissue_map),
        num_programs=8,
        program_dim=64,
        hidden_dims=[512, 256, 128, 64],
        drop=0.2,
        backbone_type="convnext1d",
        norm_type="layernorm",
        num_gene_tokens=64,
        token_dim=64,
        convnext_depth=1,
        convnext_kernel_size=7,
        convnext_expansion=4,
        use_domain_sep=True,
        domain_sep_hidden_dim=256,
        domain_sep_output_dim=64,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"已加载 Stage1: {checkpoint_path}")
    return model, s_loaders, t_loaders


def extract_stage1_features(model, s_loaders, t_loaders, device, max_samples=None):
    """提取 Stage1 特征"""
    all_h_shared = []
    all_h_pool = []
    all_domains = []
    count = 0

    with torch.no_grad():
        for loader in [s_loaders[0], s_loaders[1]]:
            for batch in loader:
                x = batch[0].to(device)
                domain_idx = torch.zeros(x.size(0), dtype=torch.long, device=device)
                out = model.program_encoder(x, domain_idx=domain_idx)
                all_h_shared.append(out["h_shared"].cpu().numpy())
                all_h_pool.append(out["h"].cpu().numpy())
                all_domains.extend([0] * x.size(0))
                count += x.size(0)
                if max_samples and count >= max_samples:
                    break
            if max_samples and count >= max_samples:
                break

        count = 0
        for loader in [t_loaders[0], t_loaders[1]]:
            for batch in loader:
                x = batch[0].to(device)
                domain_idx = torch.ones(x.size(0), dtype=torch.long, device=device)
                out = model.program_encoder(x, domain_idx=domain_idx)
                all_h_shared.append(out["h_shared"].cpu().numpy())
                all_h_pool.append(out["h"].cpu().numpy())
                all_domains.extend([1] * x.size(0))
                count += x.size(0)
                if max_samples and count >= max_samples:
                    break
            if max_samples and count >= max_samples:
                break

    return np.concatenate(all_h_shared), np.concatenate(all_h_pool), np.array(all_domains)


def plot_performance_bars(save_dir):
    """画药物级性能柱状图"""
    files = glob.glob(RESULTS_CSV_PATTERN)
    if not files:
        print("未找到结果 CSV 文件，跳过性能图")
        return

    print(f"\n找到 {len(files)} 个结果文件")
    all_results = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        all_results.append(df)

    mean_df = pd.concat(all_results).groupby(level=0).mean()
    std_df = pd.concat(all_results).groupby(level=0).std()

    metrics = ['auc', 'aupr', 'f1', 'acc']
    drugs = mean_df.columns

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = mean_df.loc[metric].values
        errs = std_df.loc[metric].values
        colors = sns.color_palette("husl", len(drugs))
        bars = ax.bar(range(len(drugs)), values, yerr=errs,
                      capsize=4, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(drugs)))
        ax.set_xticklabels(drugs, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=13)
        ax.set_title(f"{metric.upper()} per Drug (5-fold mean ± std)", fontsize=14)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)

        # 在柱子上方标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{val:.3f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "performance_per_drug.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")

    # 保存汇总表格
    mean_df.to_csv(os.path.join(save_dir, "mean_results.csv"))
    std_df.to_csv(os.path.join(save_dir, "std_results.csv"))
    print(f"已保存汇总表格: {save_dir}/mean_results.csv, {save_dir}/std_results.csv")


def plot_overall_performance(save_dir):
    """画总体性能柱状图"""
    files = glob.glob(RESULTS_CSV_PATTERN)
    if not files:
        return

    all_results = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        all_results.append(df)

    mean_df = pd.concat(all_results).groupby(level=0).mean()
    std_df = pd.concat(all_results).groupby(level=0).std()

    metrics = ['auc', 'aupr', 'f1', 'acc']
    means = [mean_df.loc[m].mean() for m in metrics]
    stds = [std_df.loc[m].mean() for m in metrics]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, means, yerr=stds, capsize=6,
                   color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'],
                   edgecolor='black', linewidth=1)
    plt.ylim([0, 1.05])
    plt.ylabel("Score", fontsize=13)
    plt.title("Overall Performance (averaged over drugs and folds)", fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha='center', va='bottom', fontsize=12, weight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, "overall_performance.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


def main():
    set_plot_style()
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ============== Stage1 T-SNE ==============
    if os.path.exists(STAGE1_CKPT):
        print("\n===== Stage1 特征可视化 =====")
        stage1_model, s_loaders, t_loaders = load_stage1_model(STAGE1_CKPT, DEVICE)
        h_shared, h_pool, domains = extract_stage1_features(
            stage1_model, s_loaders, t_loaders, DEVICE
        )
        print(f"  样本数: {len(domains)} (Source={(domains==0).sum()}, Target={(domains==1).sum()})")

        plot_tsne(h_shared, domains, "Stage1: h_shared (Domain Separation)",
                  os.path.join(SAVE_DIR, "tsne_stage1_hshared.png"))
        plot_tsne(h_pool, domains, "Stage1: h_pool (Pooled Representation)",
                  os.path.join(SAVE_DIR, "tsne_stage1_hpool.png"))
    else:
        print(f"跳过 Stage1: 找不到 {STAGE1_CKPT}")

    # ============== Stage2/3 预存特征 T-SNE ==============
    stage2_files = glob.glob(STAGE2_FEATURE_PATTERN)
    if stage2_files:
        print(f"\n===== Stage2/3 预存特征可视化 ({len(stage2_files)} folds) =====")
        for feat_path in sorted(stage2_files):
            fold_name = os.path.basename(feat_path).replace(".npz", "")
            data = np.load(feat_path)
            gene_feat = data["gene_feat"]
            fused = data["fused_feature"]
            labels = data.get("labels", np.zeros(len(gene_feat)))

            # gene_feat T-SNE
            plot_tsne(gene_feat, labels[:, 0] if labels.ndim > 1 else labels,
                      f"Stage2/3: {fold_name} gene_feat",
                      os.path.join(SAVE_DIR, f"tsne_{fold_name}_genefeat.png"))

            # fused_feature: 对 9 种药物取平均
            fused_mean = fused.mean(axis=1)
            plot_tsne(fused_mean, labels[:, 0] if labels.ndim > 1 else labels,
                      f"Stage2/3: {fold_name} fused_feature (mean over drugs)",
                      os.path.join(SAVE_DIR, f"tsne_{fold_name}_fused_mean.png"))

            # fused_feature: 第 0 种药物
            plot_tsne(fused[:, 0, :], labels[:, 0] if labels.ndim > 1 else labels,
                      f"Stage2/3: {fold_name} fused_feature (drug 0)",
                      os.path.join(SAVE_DIR, f"tsne_{fold_name}_fused_drug0.png"))
    else:
        print(f"\n跳过 Stage2/3: 找不到预存特征匹配 {STAGE2_FEATURE_PATTERN}")
        print("  请重新运行 train_mmpa_program.py，训练时会自动保存 Stage2/3 特征。")

    # ============== 性能图 ==============
    print("\n===== 药物性能可视化 =====")
    plot_performance_bars(SAVE_DIR)
    plot_overall_performance(SAVE_DIR)

    print(f"\n所有图片已保存到: {SAVE_DIR}")


if __name__ == "__main__":
    main()
