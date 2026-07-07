#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate publication-quality t-SNE visualizations.
Right panel: ONLY show tissues with GOOD cross-domain alignment (small cosine distance).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SELECTED_DRUGS = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel"]

# 更鲜明的颜色方案 - 使用ColorBrewer高质量配色
DRUG_COLORS = {
    "5-Fluorouracil": "#E41A1C",  # 红
    "Cisplatin": "#377EB8",       # 蓝
    "Cyclophosphamide": "#4DAF4A", # 绿
    "Docetaxel": "#984EA3",       # 紫
    "Paclitaxel": "#FF7F00",      # 橙
    "Carboplatin": "#FFFF33",     # 黄
    "Oxaliplatin": "#A65628",     # 棕
    "Gemcitabine": "#F781BF",     # 粉
}

# 8种高饱和度深色 - 色相环均匀分布，确保视觉区分度
VIVID_COLORS = {
    5:  "#B22222",   # Colon - 耐火砖红 (0°)
    11: "#DAA520",   # Head and Neck - 金菊黄 (45°)
    12: "#228B22",   # Kidney - 森林绿 (90°)
    14: "#008B8B",   # Liver - 深青 (135°)
    15: "#4169E1",   # Lung - 皇家蓝 (180°)
    21: "#9400D3",   # Ovarian - 深紫 (225°)
    23: "#FF1493",   # Prostate - 深粉红 (270°)
    31: "#FF8C00",   # Uterine - 深橙 (315°)
    # 备选颜色（用于未在top8中出现但被选中的组织）
    1:  "#8B4513",   # Bladder - 深鞍棕
    4:  "#20B2AA",   # Breast - 浅海绿
    25: "#800080",   # Skin - 紫色
    28: "#FF6347",   # Testis - 番茄红
    29: "#4682B4",   # Thymoma - 钢蓝
    30: "#9ACD32",   # Thyroid - 黄绿
    3:  "#D2691E",   # Brain - 巧克力色
    22: "#708090",   # Pancreas - 石板灰
}

# 用于分组边界的高亮颜色
HIGHLIGHT_COLORS = {
    "source": "#1a1a2e",
    "target": "#16213e",
}

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    # 左图参数
    parser.add_argument("--left-perplexity", type=int, default=30)
    parser.add_argument("--left-early-exaggeration", type=float, default=8.0)
    parser.add_argument("--left-max-iter", type=int, default=3000)
    parser.add_argument("--left-max-per-tissue", type=int, default=100)
    parser.add_argument("--left-scatter-size", type=int, default=25)
    parser.add_argument("--left-scatter-alpha", type=float, default=0.7)
    # 右图参数
    parser.add_argument("--right-perplexity", type=int, default=50)
    parser.add_argument("--right-early-exaggeration", type=float, default=12.0)
    parser.add_argument("--right-max-iter", type=int, default=3000)
    parser.add_argument("--right-max-per-tissue", type=int, default=40)
    parser.add_argument("--right-scatter-size", type=int, default=60)
    parser.add_argument("--right-scatter-alpha", type=float, default=0.85)
    parser.add_argument("--right-edgecolors", type=str, default="white")
    parser.add_argument("--right-linewidths", type=float, default=0.8)
    # 新增：右图对齐筛选参数
    parser.add_argument("--right-align-threshold", type=float, default=0.6,
                        help="Max cosine distance threshold for tissue alignment (smaller=better aligned).")
    parser.add_argument("--right-top-n-tissues", type=int, default=8,
                        help="Show top N best-aligned tissues (0=use threshold only).")
    # 通用
    parser.add_argument("--feature-scaling", type=str, default="l2", choices=["l2", "standard", "none"])
    parser.add_argument("--selected-drugs", type=str, default=",".join(DEFAULT_SELECTED_DRUGS))
    parser.add_argument("--gene-representation", type=str, default="auto")
    parser.add_argument("--include-unpaired-tissues", action="store_true")
    parser.add_argument("--metric-pca-dim", type=int, default=0)
    # 新增美观参数
    parser.add_argument("--fig-dpi", type=int, default=300)
    parser.add_argument("--fig-width", type=float, default=16)
    parser.add_argument("--fig-height", type=float, default=6.5)
    parser.add_argument("--title-fontsize", type=int, default=14)
    parser.add_argument("--label-fontsize", type=int, default=12)
    parser.add_argument("--legend-fontsize", type=int, default=10)
    parser.add_argument("--show-confidence-ellipse", action="store_true", 
                        help="Show confidence ellipses around each group")
    return parser.parse_args(argv)

def load_features(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Feature file not found: {npz_path}")
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}

def choose_gene_representation(data, key):
    if key != "auto":
        src_key, tgt_key = f"source_{key}", f"target_{key}"
        if src_key not in data or tgt_key not in data:
            raise KeyError(f"Requested representation '{key}' not found.")
        return data[src_key], data[tgt_key], key
    for k in ["gene_feat", "h_shared", "h_pool", "hidden"]:
        sk, tk = f"source_{k}", f"target_{k}"
        if sk in data and tk in data:
            return data[sk], data[tk], k
    raise KeyError("No suitable gene representation found.")

def _tissue_label(tissue, tissue_names):
    if tissue_names is None or int(tissue) >= len(tissue_names):
        return f"Tissue {tissue}"
    name = str(tissue_names[int(tissue)]).strip()
    return name if name else f"Tissue {tissue}"

def compute_joint_tsne(source_feat, target_feat, seed, perplexity,
                       early_exaggeration=12.0, max_iter=2000, feature_scaling="l2"):
    X = np.concatenate([source_feat, target_feat], axis=0)
    n_samples, feature_dim = X.shape

    if feature_scaling == "l2":
        X = normalize(X, norm="l2", axis=1)
    elif feature_scaling == "standard":
        X = StandardScaler().fit_transform(X)

    pca_dim = min(50, feature_dim, n_samples - 1)
    if pca_dim < feature_dim:
        X_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    else:
        X_pca = X

    eff_perp = float(min(perplexity, max(5, (n_samples - 1) // 3), n_samples - 1))

    tsne = TSNE(n_components=2, perplexity=eff_perp, early_exaggeration=early_exaggeration,
                learning_rate="auto", init="pca", max_iter=max_iter, metric="cosine", random_state=seed)
    try:
        coords = tsne.fit_transform(X_pca)
    except ValueError as exc:
        warnings.warn(f"TSNE init=pca failed ({exc}); using init=random")
        tsne = TSNE(n_components=2, perplexity=eff_perp, early_exaggeration=early_exaggeration,
                    learning_rate="auto", init="random", max_iter=max_iter, metric="cosine", random_state=seed)
        coords = tsne.fit_transform(X_pca)

    return coords, pca_dim, eff_perp

def _build_vivid_color_map(tissues):
    """直接从预定义字典获取颜色，确保每个组织固定颜色"""
    unique = sorted(np.unique(tissues).tolist())
    color_map = {}
    for t in unique:
        tissue_id = int(t)
        if tissue_id in VIVID_COLORS:
            color_map[tissue_id] = VIVID_COLORS[tissue_id]
        else:
            # 如果组织不在预定义字典中，使用默认颜色（深灰）
            color_map[tissue_id] = "#2F4F4F"
    return color_map

def compute_tissue_alignment(source_feat, target_feat, source_tissue, target_tissue, tissue_names=None):
    """
    计算每个paired tissue的跨域对齐质量（余弦距离）。
    返回：按对齐质量排序的tissue列表 [(tissue_id, cosine_distance, src_n, tgt_n, tissue_name), ...]
    """
    X_src = normalize(source_feat, norm="l2", axis=1)
    X_tgt = normalize(target_feat, norm="l2", axis=1)

    all_tissues = sorted(set(np.unique(source_tissue)) & set(np.unique(target_tissue)))
    results = []

    for tissue in all_tissues:
        s_mask = source_tissue == tissue
        t_mask = target_tissue == tissue
        src_n, tgt_n = s_mask.sum(), t_mask.sum()
        if src_n < 2 or tgt_n < 2:
            continue

        # 计算tissue质心的余弦距离
        c_src = normalize(X_src[s_mask].mean(axis=0, keepdims=True), norm="l2")
        c_tgt = normalize(X_tgt[t_mask].mean(axis=0, keepdims=True), norm="l2")
        cos_dist = float(1 - cosine_similarity(c_src, c_tgt)[0, 0])

        tissue_name = _tissue_label(int(tissue), tissue_names)
        results.append({
            "tissue": int(tissue),
            "tissue_name": tissue_name,
            "cosine_distance": cos_dist,
            "n_source": int(src_n),
            "n_target": int(tgt_n),
        })

    # 按cosine_distance升序排序（越小对齐越好）
    results_df = pd.DataFrame(results).sort_values("cosine_distance")
    return results_df

def sample_aligned_tissues(source_tissue, target_tissue, max_per_tissue, seed,
                            aligned_tissues_df, top_n=8, threshold=0.6):
    """
    只采样对齐质量好的tissue。
    aligned_tissues_df: 来自compute_tissue_alignment的排序结果
    """
    rng = np.random.RandomState(seed)
    src_mask = np.zeros(len(source_tissue), dtype=bool)
    tgt_mask = np.zeros(len(target_tissue), dtype=bool)

    # 选择tissue：优先top_n，且满足threshold
    selected = aligned_tissues_df.head(top_n) if top_n > 0 else aligned_tissues_df
    selected = selected[selected["cosine_distance"] <= threshold]

    records = []
    for _, row in selected.iterrows():
        tissue = row["tissue"]
        src_idx = np.where(source_tissue == tissue)[0]
        tgt_idx = np.where(target_tissue == tissue)[0]
        n = min(len(src_idx), len(tgt_idx), max_per_tissue)

        src_chosen = rng.choice(src_idx, size=n, replace=False)
        tgt_chosen = rng.choice(tgt_idx, size=n, replace=False)
        src_mask[src_chosen] = True
        tgt_mask[tgt_chosen] = True

        records.append({
            "tissue": int(tissue),
            "tissue_name": row["tissue_name"],
            "cosine_distance": row["cosine_distance"],
            "n_selected_per_domain": n,
            "source_available": row["n_source"],
            "target_available": row["n_target"],
        })

    df = pd.DataFrame(records)
    print(f"\nAligned tissue sampling: selected source={int(src_mask.sum())}, target={int(tgt_mask.sum())}")
    print(f"Selected {len(df)} tissues with cosine_distance <= {threshold}")
    print(df.to_string(index=False))
    return src_mask, tgt_mask, df

def plot_selected_drug_joint_tsne(
    source_fused, target_fused, source_tissue, target_tissue,
    drug_names, selected_drugs, output_dir, seed,
    feature_scaling="l2", tissue_names=None, include_unpaired=False,
    left_perplexity=30, left_early_exaggeration=8.0, left_max_iter=3000,
    left_max_per_tissue=100, left_scatter_size=25, left_scatter_alpha=0.7,
    right_perplexity=50, right_early_exaggeration=12.0, right_max_iter=3000,
    right_max_per_tissue=40, right_scatter_size=60, right_scatter_alpha=0.85,
    right_edgecolors="white", right_linewidths=0.8,
    right_align_threshold=0.6, right_top_n_tissues=8,
    fig_dpi=300, fig_width=16, fig_height=6.5,
    title_fontsize=14, label_fontsize=12, legend_fontsize=10,
    show_confidence_ellipse=False,
):
    # ========== 左图：Drug聚类 ==========
    print("\n=== Left Panel: Drug-wise t-SNE ===")
    rng = np.random.RandomState(seed)
    src_mask = np.zeros(len(source_tissue), dtype=bool)
    tgt_mask = np.zeros(len(target_tissue), dtype=bool)

    # 左图：简单采样，每种drug的tissue都采
    all_tissues = sorted(set(np.unique(source_tissue)) | set(np.unique(target_tissue)))
    for tissue in all_tissues:
        src_idx = np.where(source_tissue == tissue)[0]
        tgt_idx = np.where(target_tissue == tissue)[0]
        src_n, tgt_n = len(src_idx), len(tgt_idx)
        if src_n > 0 and tgt_n > 0:
            n = min(src_n, tgt_n, left_max_per_tissue)
            src_mask[rng.choice(src_idx, size=n, replace=False)] = True
            tgt_mask[rng.choice(tgt_idx, size=n, replace=False)] = True
        elif include_unpaired:
            available = src_idx if src_n > 0 else tgt_idx
            n = min(len(available), left_max_per_tissue)
            (src_mask if src_n > 0 else tgt_mask)[rng.choice(available, size=n, replace=False)] = True

    n_src_left, n_tgt_left = int(src_mask.sum()), int(tgt_mask.sum())

    drug_indices = []
    for dname in selected_drugs:
        matches = np.where(drug_names == dname)[0]
        if len(matches) == 0:
            raise ValueError(f"Drug '{dname}' not found")
        drug_indices.append(int(matches[0]))

    src_all = np.concatenate([source_fused[src_mask, d, :] for d in drug_indices], axis=0)
    tgt_all = np.concatenate([target_fused[tgt_mask, d, :] for d in drug_indices], axis=0)
    left_coords, left_pca_dim, left_eff_perp = compute_joint_tsne(
        src_all, tgt_all, seed, left_perplexity,
        early_exaggeration=left_early_exaggeration, max_iter=left_max_iter, feature_scaling=feature_scaling
    )

    n_src_all, n_tgt_all = src_all.shape[0], tgt_all.shape[0]
    left_domain_labels = np.concatenate([np.zeros(n_src_all, dtype=int), np.ones(n_tgt_all, dtype=int)])
    left_drug_labels = np.concatenate([
        np.repeat(np.arange(len(selected_drugs)), n_src_left),
        np.repeat(np.arange(len(selected_drugs)), n_tgt_left),
    ])

    # ========== 右图：Docetaxel，只显示对齐好的tissue ==========
    print("\n=== Right Panel: Docetaxel (aligned tissues only) ===")
    doc_name = "Docetaxel" if "Docetaxel" in selected_drugs else selected_drugs[0]
    doc_idx = drug_indices[selected_drugs.index(doc_name)]

    # 计算每个tissue的对齐质量
    print("\nComputing tissue alignment quality...")
    alignment_df = compute_tissue_alignment(
        source_fused[:, doc_idx, :], target_fused[:, doc_idx, :],
        source_tissue, target_tissue, tissue_names
    )
    print(alignment_df.to_string(index=False))

    # 只采样对齐好的tissue
    right_src_mask, right_tgt_mask, right_sampling_df = sample_aligned_tissues(
        source_tissue, target_tissue, right_max_per_tissue, seed + 1,
        alignment_df, top_n=right_top_n_tissues, threshold=right_align_threshold
    )

    n_src_right, n_tgt_right = int(right_src_mask.sum()), int(right_tgt_mask.sum())
    if n_src_right == 0 or n_tgt_right == 0:
        raise ValueError("No aligned tissues found! Try increasing --right-align-threshold.")

    right_src = source_fused[right_src_mask, doc_idx, :]
    right_tgt = target_fused[right_tgt_mask, doc_idx, :]
    right_coords, right_pca_dim, right_eff_perp = compute_joint_tsne(
        right_src, right_tgt, seed + 100, right_perplexity,
        early_exaggeration=right_early_exaggeration, max_iter=right_max_iter, feature_scaling=feature_scaling
    )

    right_tissues = np.concatenate([source_tissue[right_src_mask], target_tissue[right_tgt_mask]])
    right_domains = np.concatenate([np.zeros(n_src_right, dtype=int), np.ones(n_tgt_right, dtype=int)])

    # ========== 绘图 ==========
    # 使用更美观的样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=fig_dpi)
    
    # 设置背景颜色
    for ax in [ax1, ax2]:
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # ====== Left: drug coloring with improved aesthetics ======
    # 获取药物颜色
    drug_color_map = {d: DRUG_COLORS.get(d, plt.cm.tab10(i % 10)) 
                      for i, d in enumerate(selected_drugs)}
    
    # 为每个药物绘制，使用不同透明度区分domain
    for i, dname in enumerate(selected_drugs):
        # Source domain (圆点)
        sel_src = (left_drug_labels == i) & (left_domain_labels == 0)
        if sel_src.sum() > 0:
            ax1.scatter(left_coords[sel_src, 0], left_coords[sel_src, 1],
                       c=[drug_color_map[dname]], marker='o', 
                       s=left_scatter_size * 1.1, alpha=left_scatter_alpha * 0.8,
                       edgecolors='white', linewidths=0.3,
                       rasterized=True, label=f'{dname} (Source)' if i == 0 else "")
        
        # Target domain (三角形)
        sel_tgt = (left_drug_labels == i) & (left_domain_labels == 1)
        if sel_tgt.sum() > 0:
            ax1.scatter(left_coords[sel_tgt, 0], left_coords[sel_tgt, 1],
                       c=[drug_color_map[dname]], marker='^', 
                       s=left_scatter_size * 1.3, alpha=left_scatter_alpha,
                       edgecolors='white', linewidths=0.3,
                       rasterized=True, label=f'{dname} (Target)' if i == 0 else "")

    # 为左图添加置信椭圆（可选）
    if show_confidence_ellipse:
        from matplotlib.patches import Ellipse
        for i, dname in enumerate(selected_drugs):
            for domain_label, color in [(0, '#1a1a2e'), (1, '#16213e')]:
                mask = (left_drug_labels == i) & (left_domain_labels == domain_label)
                if mask.sum() > 2:
                    points = left_coords[mask]
                    if len(points) > 2:
                        # 计算协方差和置信椭圆
                        cov = np.cov(points.T)
                        if not np.isnan(cov).any():
                            try:
                                # 使用2倍标准差
                                v, w = np.linalg.eigh(cov)
                                v = np.sqrt(v) * 1.5  # 1.5 sigma ellipse
                                u = w[0] / np.linalg.norm(w[0])
                                angle = np.arctan2(u[1], u[0]) * 180 / np.pi
                                ell = Ellipse(xy=np.mean(points, axis=0),
                                            width=2 * v[0], height=2 * v[1],
                                            angle=angle, alpha=0.15,
                                            facecolor=color, edgecolor='none')
                                ax1.add_patch(ell)
                            except:
                                pass

    ax1.set_title("Cross-Domain Drug Representations", fontsize=title_fontsize, fontweight='bold', pad=10)
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=label_fontsize)
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=label_fontsize)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # 美化图例 - 只显示药物名称，不区分domain
    from matplotlib.lines import Line2D
    drug_handles = [Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=drug_color_map[d], 
                          markersize=8, label=d, markeredgecolor='white', markeredgewidth=0.5)
                   for d in selected_drugs]
    legend1 = ax1.legend(handles=drug_handles, title="Drugs", 
                        loc='upper right', frameon=True, 
                        fancybox=True, shadow=True, 
                        fontsize=legend_fontsize, title_fontsize=legend_fontsize + 1)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.9)

    # ====== Right: tissue coloring with enhanced aesthetics ======
    # 构建颜色映射
    color_map = _build_vivid_color_map(right_tissues)
    tissue_names_map = {int(t): _tissue_label(int(t), tissue_names) for t in np.unique(right_tissues)}
    
    # 为每个组织绘制，source用圆点，target用三角形
    unique_tissues = sorted(np.unique(right_tissues))
    
    # 先绘制所有点，按组织分组
    for tissue in unique_tissues:
        color = color_map[int(tissue)]
        tissue_label = tissue_names_map[int(tissue)]
        
        # Source domain (圆点)
        mask_src = (right_domains == 0) & (right_tissues == tissue)
        if mask_src.sum() > 0:
            ax2.scatter(right_coords[mask_src, 0], right_coords[mask_src, 1],
                       c=[color], marker='o', s=right_scatter_size,
                       alpha=right_scatter_alpha * 0.85,
                       edgecolors=right_edgecolors, linewidths=right_linewidths,
                       rasterized=True, label=f'{tissue_label} (Source)' if tissue == unique_tissues[0] else "")
        
        # Target domain (三角形)
        mask_tgt = (right_domains == 1) & (right_tissues == tissue)
        if mask_tgt.sum() > 0:
            ax2.scatter(right_coords[mask_tgt, 0], right_coords[mask_tgt, 1],
                       c=[color], marker='^', s=right_scatter_size * 1.2,
                       alpha=right_scatter_alpha,
                       edgecolors=right_edgecolors, linewidths=right_linewidths,
                       rasterized=True, label=f'{tissue_label} (Target)' if tissue == unique_tissues[0] else "")

    # 为右图添加置信椭圆
    if show_confidence_ellipse:
        from matplotlib.patches import Ellipse
        for tissue in unique_tissues:
            for domain_mask, color_prefix in [(0, '#2c3e50'), (1, '#34495e')]:
                mask = (right_tissues == tissue) & (right_domains == domain_mask)
                if mask.sum() > 2:
                    points = right_coords[mask]
                    if len(points) > 2:
                        cov = np.cov(points.T)
                        if not np.isnan(cov).any():
                            try:
                                v, w = np.linalg.eigh(cov)
                                v = np.sqrt(v) * 1.5
                                u = w[0] / np.linalg.norm(w[0])
                                angle = np.arctan2(u[1], u[0]) * 180 / np.pi
                                ell = Ellipse(xy=np.mean(points, axis=0),
                                            width=2 * v[0], height=2 * v[1],
                                            angle=angle, alpha=0.12,
                                            facecolor=color_map[int(tissue)], edgecolor='none')
                                ax2.add_patch(ell)
                            except:
                                pass

    ax2.set_title(f"{doc_name} - Tissue Alignment", fontsize=title_fontsize, fontweight='bold', pad=10)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=label_fontsize)
    ax2.set_ylabel("")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # 右图图例 - 只显示组织名称
    tissue_handles = [Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=color_map[int(t)],
                            markersize=8, label=tissue_names_map[int(t)],
                            markeredgecolor='white', markeredgewidth=0.5)
                     for t in unique_tissues]
    legend2 = ax2.legend(handles=tissue_handles, title="Tissues",
                        loc='upper right', frameon=True,
                        fancybox=True, shadow=True,
                        fontsize=legend_fontsize, title_fontsize=legend_fontsize + 1)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_alpha(0.9)

    # 添加子图标签 (a, b)
    ax1.text(-0.08, 1.02, 'a', transform=ax1.transAxes, 
            fontsize=title_fontsize + 4, fontweight='bold', va='bottom')
    ax2.text(-0.08, 1.02, 'b', transform=ax2.transAxes,
            fontsize=title_fontsize + 4, fontweight='bold', va='bottom')

    # 整体调整
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    # 保存
    base = os.path.join(output_dir, "selected_drug_joint_tsne")
    fig.savefig(base + ".png", dpi=fig_dpi, bbox_inches="tight", facecolor='white')
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor='white')
    plt.close(fig)

    # 保存数据
    pd.DataFrame({
        "x": left_coords[:, 0], "y": left_coords[:, 1],
        "domain": np.where(left_domain_labels == 0, "source", "target"),
        "drug": np.asarray(selected_drugs)[left_drug_labels],
    }).to_csv(os.path.join(output_dir, "left_panel_coordinates.csv"), index=False)

    pd.DataFrame({
        "x": right_coords[:, 0], "y": right_coords[:, 1],
        "domain": np.where(right_domains == 0, "source", "target"),
        "tissue": right_tissues,
        "tissue_name": [tissue_names_map[int(t)] for t in right_tissues],
    }).to_csv(os.path.join(output_dir, "right_panel_coordinates.csv"), index=False)

    alignment_df.to_csv(os.path.join(output_dir, "tissue_alignment_quality.csv"), index=False)
    right_sampling_df.to_csv(os.path.join(output_dir, "right_panel_sampling.csv"), index=False)

    print(f"\nSaved to {base}.png/.pdf")
    return {
        "left": {"pca_dim": left_pca_dim, "eff_perp": left_eff_perp},
        "right": {"pca_dim": right_pca_dim, "eff_perp": right_eff_perp,
                  "n_tissues": len(color_map), "align_threshold": right_align_threshold},
    }

def compute_representation_metrics(source_feat, target_feat, source_tissue, target_tissue, pca_dim=50):
    X_src = normalize(source_feat, norm="l2", axis=1)
    X_tgt = normalize(target_feat, norm="l2", axis=1)
    X = np.concatenate([X_src, X_tgt], axis=0)
    domains = np.concatenate([np.zeros(len(X_src)), np.ones(len(X_tgt))])
    tissues = np.concatenate([source_tissue, target_tissue])

    if 0 < pca_dim < X.shape[1]:
        pca = PCA(n_components=min(pca_dim, X.shape[0] - 1, X.shape[1]), random_state=2025)
        X = pca.fit_transform(X)
        X_src, X_tgt = X[:len(X_src)], X[len(X_src):]

    def _sil(labels):
        if len(np.unique(labels)) < 2: return np.nan
        try: return float(silhouette_score(X, labels))
        except ValueError as exc: return f"failed: {exc}"

    centroid_cos = float(1 - cosine_similarity(
        X_src.mean(axis=0, keepdims=True), X_tgt.mean(axis=0, keepdims=True)
    )[0, 0])

    try:
        from sklearn.metrics.pairwise import rbf_kernel
        n, m = len(X_src), len(X_tgt)
        Kxx, Kyy, Kxy = rbf_kernel(X_src, X_src), rbf_kernel(X_tgt, X_tgt), rbf_kernel(X_src, X_tgt)
        mmd2 = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean()
        mmd = float(np.sqrt(max(mmd2, 0.0)))
    except Exception as exc:
        mmd = f"failed: {exc}"

    records = [{
        "tissue": "ALL", "n_source": len(X_src), "n_target": len(X_tgt),
        "domain_silhouette": _sil(domains), "tissue_silhouette": _sil(tissues),
        "centroid_cosine_dist": centroid_cos, "mmd_rbf": mmd, "per_tissue_cos_dist": np.nan,
    }]

    for tissue in sorted(set(np.unique(source_tissue)) & set(np.unique(target_tissue))):
        s_mask, t_mask = source_tissue == tissue, target_tissue == tissue
        if s_mask.sum() < 1 or t_mask.sum() < 1: continue
        c_src = normalize(X_src[s_mask].mean(axis=0, keepdims=True), norm="l2")
        c_tgt = normalize(X_tgt[t_mask].mean(axis=0, keepdims=True), norm="l2")
        records.append({
            "tissue": int(tissue), "n_source": int(s_mask.sum()), "n_target": int(t_mask.sum()),
            "domain_silhouette": np.nan, "tissue_silhouette": np.nan,
            "centroid_cosine_dist": np.nan, "mmd_rbf": np.nan,
            "per_tissue_cos_dist": float(1 - cosine_similarity(c_src, c_tgt)[0, 0]),
        })

    return pd.DataFrame(records)

def save_tsne_config(output_dir, args, selected_drugs, drug_names, gene_meta, joint_meta, n_source, n_target, tissue_names):
    cfg = {
        "feature_file": args.feature_file, "seed": args.seed,
        "selected_drugs": selected_drugs, "drug_order": drug_names,
        "tissue_mapping": {int(i): str(name) for i, name in enumerate(tissue_names)},
        "include_unpaired": args.include_unpaired_tissues,
        "n_source": n_source, "n_target": n_target,
        "gene_representation": gene_meta, "joint_meta": joint_meta,
        "left": {k.replace("left_", ""): getattr(args, k) for k in dir(args) if k.startswith("left_")},
        "right": {k.replace("right_", ""): getattr(args, k) for k in dir(args) if k.startswith("right_")},
    }
    with open(os.path.join(output_dir, "tsne_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_features(args.feature_file)
    drug_names = data["drug_names"]
    selected_drugs = [d.strip() for d in args.selected_drugs.split(",") if d.strip()]
    for d in selected_drugs:
        if d not in drug_names:
            raise ValueError(f"Drug '{d}' not in drug_names")

    tissue_names = data.get("tissue_names", np.array([]))
    source_fused = data["source_fused_feat"]
    target_fused = data["target_fused_feat"]
    source_tissue = data["source_tissue"]
    target_tissue = data["target_tissue"]

    src_gene, tgt_gene, rep_name = choose_gene_representation(data, args.gene_representation)

    joint_meta = plot_selected_drug_joint_tsne(
        source_fused, target_fused, source_tissue, target_tissue,
        drug_names, selected_drugs, args.output_dir, args.seed,
        feature_scaling=args.feature_scaling, tissue_names=tissue_names,
        include_unpaired=args.include_unpaired_tissues,
        left_perplexity=args.left_perplexity, left_early_exaggeration=args.left_early_exaggeration,
        left_max_iter=args.left_max_iter, left_max_per_tissue=args.left_max_per_tissue,
        left_scatter_size=args.left_scatter_size, left_scatter_alpha=args.left_scatter_alpha,
        right_perplexity=args.right_perplexity, right_early_exaggeration=args.right_early_exaggeration,
        right_max_iter=args.right_max_iter, right_max_per_tissue=args.right_max_per_tissue,
        right_scatter_size=args.right_scatter_size, right_scatter_alpha=args.right_scatter_alpha,
        right_edgecolors=args.right_edgecolors, right_linewidths=args.right_linewidths,
        right_align_threshold=args.right_align_threshold, right_top_n_tissues=args.right_top_n_tissues,
        fig_dpi=args.fig_dpi, fig_width=args.fig_width, fig_height=args.fig_height,
        title_fontsize=args.title_fontsize, label_fontsize=args.label_fontsize,
        legend_fontsize=args.legend_fontsize, show_confidence_ellipse=args.show_confidence_ellipse,
    )

    metrics_df = compute_representation_metrics(src_gene, tgt_gene, source_tissue, target_tissue, pca_dim=args.metric_pca_dim)
    metrics_df.to_csv(os.path.join(args.output_dir, "tsne_representation_metrics.csv"), index=False)

    drug_metrics = []
    for d_name in selected_drugs:
        d_idx = int(np.where(drug_names == d_name)[0][0])
        df = compute_representation_metrics(
            source_fused[:, d_idx, :], target_fused[:, d_idx, :],
            source_tissue, target_tissue, pca_dim=args.metric_pca_dim
        )
        df.insert(0, "drug", d_name)
        drug_metrics.append(df)
    pd.concat(drug_metrics, ignore_index=True).to_csv(
        os.path.join(args.output_dir, "drugwise_representation_metrics.csv"), index=False
    )

    save_tsne_config(args.output_dir, args, selected_drugs, drug_names.tolist(),
                     {"representation": rep_name}, joint_meta,
                     source_fused.shape[0], target_fused.shape[0], tissue_names)

    print("\nAll done!")

if __name__ == "__main__":
    main()