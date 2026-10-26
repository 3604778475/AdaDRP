"""
一键提取 Stage1 的 h_shared 特征并画 T-SNE
用法：
    /d/anaconda/envs/macb_drp/python.exe extract_tsne_stage1.py
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 必须先导入 utility，避免 dataload 循环导入
from utility import *
from dataload import get_unlabeled_dataloaders

# ========== 你只需要改这里 ==========
CHECKPOINT_PATH = "checkpoints/stage1/stage1_model.pt"  # Stage1 模型路径
SAVE_DIR = "outputs/tsne"                               # 保存目录
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# ===================================


def load_stage1_model(checkpoint_path, device):
    """加载 Stage1 模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型: {checkpoint_path}\n请先训练 Stage1 或检查路径")
    
    # 导入模型类
    from adadrp.models import ProgramTissueClassifier
    import config
    
    # 读取 gex 特征维度
    import pandas as pd
    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)
    input_dim = gex_features_df.shape[1] - 1
    
    # 构建模型（用你训练时的配置）
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
    
    # 加载权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"已加载模型: {checkpoint_path}")
    return model


def get_unlabeled_data(device):
    """加载 Stage1 的无标签数据"""
    import config
    import pandas as pd
    
    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)
    s_dataloaders, t_dataloaders = get_unlabeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=2025,
        test_ratio=0.25,
        batch_size=64
    )
    return s_dataloaders, t_dataloaders


def extract_features(model, s_loaders, t_loaders, device):
    """提取 h_shared 和 h_pool"""
    all_h_shared = []
    all_h_pool = []
    all_domains = []
    
    model.eval()
    with torch.no_grad():
        # 源域：用训练集和测试集
        for loader in [s_loaders[0], s_loaders[1]]:
            for batch in loader:
                x = batch[0].to(device)
                domain_idx = torch.zeros(x.size(0), dtype=torch.long, device=device)
                out = model.program_encoder(x, domain_idx=domain_idx)
                all_h_shared.append(out["h_shared"].cpu().numpy())
                all_h_pool.append(out["h"].cpu().numpy())
                all_domains.extend([0] * x.size(0))
        
        # 目标域：用训练集和测试集
        for loader in [t_loaders[0], t_loaders[1]]:
            for batch in loader:
                x = batch[0].to(device)
                domain_idx = torch.ones(x.size(0), dtype=torch.long, device=device)
                out = model.program_encoder(x, domain_idx=domain_idx)
                all_h_shared.append(out["h_shared"].cpu().numpy())
                all_h_pool.append(out["h"].cpu().numpy())
                all_domains.extend([1] * x.size(0))
    
    all_h_shared = np.concatenate(all_h_shared, axis=0)
    all_h_pool = np.concatenate(all_h_pool, axis=0)
    all_domains = np.array(all_domains)
    
    return all_h_shared, all_h_pool, all_domains


def plot_tsne(features, labels, title, save_path):
    """画 T-SNE"""
    print(f"正在计算 T-SNE: {title}")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                max_iter=1000, random_state=42)
    emb = tsne.fit_transform(features)
    
    label_names = ["Source(CCLE/GDSC)" if l == 0 else "Target(TCGA)" for l in labels]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=label_names,
                    palette={"Source(CCLE/GDSC)": "#3498db", "Target(TCGA)": "#e74c3c"},
                    alpha=0.7, s=30)
    plt.title(title, fontsize=16)
    plt.xlabel("T-SNE 1", fontsize=12)
    plt.ylabel("T-SNE 2", fontsize=12)
    plt.legend(title="Domain", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. 先加载数据（这会设置 config.tissue_map）
    s_loaders, t_loaders = get_unlabeled_data(DEVICE)
    
    # 2. 加载模型
    model = load_stage1_model(CHECKPOINT_PATH, DEVICE)
    
    # 3. 提取特征
    h_shared, h_pool, domains = extract_features(model, s_loaders, t_loaders, DEVICE)
    print(f"h_shared shape: {h_shared.shape}")
    print(f"h_pool shape: {h_pool.shape}")
    print(f"Source samples: {(domains == 0).sum()}")
    print(f"Target samples: {(domains == 1).sum()}")
    
    # 4. 保存 npz
    np.savez(os.path.join(SAVE_DIR, "stage1_hshared.npz"),
             features=h_shared, labels=domains)
    np.savez(os.path.join(SAVE_DIR, "stage1_hpool.npz"),
             features=h_pool, labels=domains)
    
    # 5. 画 T-SNE
    plot_tsne(h_shared, domains, "Stage1 h_shared (Domain Separation)",
              os.path.join(SAVE_DIR, "tsne_stage1_hshared.png"))
    plot_tsne(h_pool, domains, "Stage1 h_pool (Pooled Representation)",
              os.path.join(SAVE_DIR, "tsne_stage1_hpool.png"))
    
    print("\n全部完成！图片保存在:", SAVE_DIR)


if __name__ == "__main__":
    main()
