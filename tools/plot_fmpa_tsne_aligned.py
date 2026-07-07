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

# 16种鲜明固定颜色
VIVID_COLORS = [
    (0.89, 0.10, 0.11), (0.20, 0.40, 0.80), (0.20, 0.60, 0.20), (0.60, 0.20, 0.80),
    (1.00, 0.50, 0.00), (0.90, 0.90, 0.20), (0.55, 0.27, 0.07), (0.90, 0.40, 0.70),
    (0.50, 0.50, 0.50), (0.20, 0.80, 0.60), (0.80, 0.30, 0.30), (0.30, 0.30, 0.80),
    (0.60, 0.80, 0.20), (0.80, 0.60, 0.20), (0.40, 0.20, 0.60), (0.20, 0.60, 0.80),
]

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    # 左图参数
    parser.add_argument("--left-perplexity", type=int, default=20)
    parser.add_argument("--left-early-exaggeration", type=float, default=6.0)
    parser.add_argument("--left-max-iter", type=int, default=3000)
    parser.add_argument("--left-max-per-tissue", type=int, default=100)
    parser.add_argument("--left-scatter-size", type=int, default=20)
    parser.add_argument("--left-scatter-alpha", type=float, default=0.75)
    # 右图参数
    parser.add_argument("--right-perplexity", type=int, default=50)
    parser.add_argument("--right-early-exaggeration", type=float, default=12.0)
    parser.add_argument("--right-max-iter", type=int, default=3000)
    parser.add_argument("--right-max-per-tissue", type=int, default=40)
    parser.add_argument("--right-scatter-size", type=int, default=50)
    parser.add_argument("--right-scatter-alpha", type=float, default=0.95)
    parser.add_argument("--right-edgecolors", type=str, default="black")
    parser.add_argument("--right-linewidths", type=float, default=0.5)
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
    unique = sorted(np.unique(tissues).tolist())
    return {int(t): (*VIVID_COLORS[i % len(VIVID_COLORS)], 1.0) for i, t in enumerate(unique)}


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
    left_perplexity=20, left_early_exaggeration=6.0, left_max_iter=3000,
    left_max_per_tissue=100, left_scatter_size=20, left_scatter_alpha=0.75,
    right_perplexity=50, right_early_exaggeration=12.0, right_max_iter=3000,
    right_max_per_tissue=40, right_scatter_size=50, right_scatter_alpha=0.95,
    right_edgecolors="black", right_linewidths=0.5,
    right_align_threshold=0.6, right_top_n_tissues=8,
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Left: drug coloring
    drug_color_map = {i: plt.cm.tab10(i) for i in range(len(selected_drugs))}
    from matplotlib.lines import Line2D
    drug_handles = []
    for i, dname in enumerate(selected_drugs):
        for marker, dom_mask in [("o", left_domain_labels == 0), ("^", left_domain_labels == 1)]:
            sel = (left_drug_labels == i) & dom_mask
            ax1.scatter(left_coords[sel, 0], left_coords[sel, 1], c=[drug_color_map[i]],
                        marker=marker, s=left_scatter_size, alpha=left_scatter_alpha,
                        edgecolors="none", rasterized=True)
        drug_handles.append(Line2D([0], [0], marker="s", color="w",
                                   markerfacecolor=drug_color_map[i], markersize=6, label=dname))

    ax1.set_title("Cross-Domain Representations for Selected Drugs", fontsize=13)
    ax1.set_xlabel("t-SNE 1", fontsize=11)
    ax1.set_ylabel("t-SNE 2", fontsize=11)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend(handles=drug_handles, title="Drug", loc="best", frameon=True, ncol=2, fontsize=8)

    # Right: tissue coloring with VIVID colors, ONLY well-aligned tissues
    color_map = _build_vivid_color_map(right_tissues)
    source_mask_right = right_domains == 0
    target_mask_right = right_domains == 1

    for domain_mask, marker in [(source_mask_right, "o"), (target_mask_right, "^")]:
        for tissue in sorted(np.unique(right_tissues[domain_mask])):
            t_mask = domain_mask & (right_tissues == tissue)
            ax2.scatter(
                right_coords[t_mask, 0], right_coords[t_mask, 1],
                c=[color_map[int(tissue)]],
                marker=marker,
                s=right_scatter_size,
                alpha=right_scatter_alpha,
                edgecolors=right_edgecolors,
                linewidths=right_linewidths,
                rasterized=True,
            )

    ax2.set_title(f"{doc_name} (Aligned Tissues)", fontsize=13)
    ax2.set_xlabel("t-SNE1", fontsize=11)
    ax2.set_ylabel("")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    base = os.path.join(output_dir, "selected_drug_joint_tsne")
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)

    # Save
    pd.DataFrame({
        "x": left_coords[:, 0], "y": left_coords[:, 1],
        "domain": np.where(left_domain_labels == 0, "source", "target"),
        "drug": np.asarray(selected_drugs)[left_drug_labels],
    }).to_csv(os.path.join(output_dir, "left_panel_coordinates.csv"), index=False)

    pd.DataFrame({
        "x": right_coords[:, 0], "y": right_coords[:, 1],
        "domain": np.where(right_domains == 0, "source", "target"),
        "tissue": right_tissues,
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
