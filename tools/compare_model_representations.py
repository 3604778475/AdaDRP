#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare t-SNE of different model representations for Docetaxel.

Panels:
  (a) fused_feat  perplexity=50 (original)
  (b) fused_feat  perplexity=15
  (c) h_shared    perplexity=30
  (d) gene_feat   perplexity=30

Uses the same tissue/sample subset for fair comparison.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIVID_COLORS = {
    5: "#B22222", 11: "#DAA520", 12: "#228B22", 14: "#008B8B",
    15: "#4169E1", 21: "#9400D3", 23: "#FF1493", 31: "#FF8C00",
    1: "#8B4513", 4: "#20B2AA", 25: "#800080", 28: "#FF6347",
    29: "#4682B4", 30: "#9ACD32", 3: "#D2691E", 22: "#708090",
    9: "#2E8B57",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str,
                        default=str(PROJECT_ROOT / "artifacts" / "tsne" / "fold_1" / "fmpa_tsne_features.npz"))
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "figures"))
    parser.add_argument("--drug", type=str, default="Docetaxel",
                        help="Drug name to visualize (must exist in npz drug_names).")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max-per-tissue", type=int, default=40)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def compute_joint_tsne(source_feat, target_feat, seed, perplexity=30):
    X = np.concatenate([source_feat, target_feat], axis=0)
    X = normalize(X, norm="l2", axis=1)
    n_samples, d = X.shape
    pca_dim = min(50, d, n_samples - 1)
    if pca_dim < d:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    eff_perp = float(min(perplexity, max(5, (n_samples - 1) // 3), n_samples - 1))
    coords = TSNE(n_components=2, perplexity=eff_perp, early_exaggeration=12.0,
                  learning_rate="auto", init="pca", max_iter=2000,
                  metric="cosine", random_state=seed).fit_transform(X)
    return coords


def alignment_quality(source_feat, target_feat, source_tissue, target_tissue):
    X_src = normalize(source_feat, norm="l2", axis=1)
    X_tgt = normalize(target_feat, norm="l2", axis=1)
    results = []
    for t in sorted(set(np.unique(source_tissue)) & set(np.unique(target_tissue))):
        s_mask = source_tissue == t
        t_mask = target_tissue == t
        if s_mask.sum() < 2 or t_mask.sum() < 2:
            continue
        c_src = normalize(X_src[s_mask].mean(axis=0, keepdims=True), norm="l2")
        c_tgt = normalize(X_tgt[t_mask].mean(axis=0, keepdims=True), norm="l2")
        dist = float(1 - cosine_similarity(c_src, c_tgt)[0, 0])
        results.append((t, dist, s_mask.sum(), t_mask.sum()))
    return sorted(results, key=lambda x: x[1])


def sample_tissue_points(source_tissue, target_tissue, selected_tissues, max_per_tissue, seed):
    rng = np.random.RandomState(seed)
    src_idx, tgt_idx = [], []
    for t in selected_tissues:
        s_idx = np.where(source_tissue == t)[0]
        t_idx = np.where(target_tissue == t)[0]
        n = min(len(s_idx), len(t_idx), max_per_tissue)
        if n == 0:
            continue
        src_idx.extend(rng.choice(s_idx, size=n, replace=False))
        tgt_idx.extend(rng.choice(t_idx, size=n, replace=False))
    return np.asarray(src_idx), np.asarray(tgt_idx)


def plot_panel(ax, coords, src_t, tgt_t, title):
    all_t = np.concatenate([src_t, tgt_t])
    domains = np.concatenate([np.zeros(len(src_t), dtype=int), np.ones(len(tgt_t), dtype=int)])
    for t in sorted(np.unique(all_t)):
        color = VIVID_COLORS.get(t, "#2F4F4F")
        for d, marker in [(0, "o"), (1, "^")]:
            mask = (all_t == t) & (domains == d)
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1], c=color, marker=marker,
                           s=60 if marker == "o" else 70, alpha=0.8,
                           edgecolors="white", linewidths=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data = np.load(args.npz)

    drug_names = data["drug_names"]
    if args.drug not in drug_names:
        raise ValueError(f"Drug '{args.drug}' not found. Available: {list(drug_names)}")
    drug_idx = int(np.where(drug_names == args.drug)[0][0])
    tissue_names = {i: str(name) for i, name in enumerate(data["tissue_names"])}

    # select top 8 model-aligned tissues based on fused_feat
    align = alignment_quality(
        data["source_fused_feat"][:, drug_idx, :],
        data["target_fused_feat"][:, drug_idx, :],
        data["source_tissue"], data["target_tissue"]
    )
    selected = [t for t, _, _, _ in align[:8]]
    print("Selected tissues:", [tissue_names[t] for t in selected])

    src_idx, tgt_idx = sample_tissue_points(
        data["source_tissue"], data["target_tissue"], selected, args.max_per_tissue, args.seed
    )

    def subset(key):
        return data[f"source_{key}"][src_idx], data[f"target_{key}"][tgt_idx]

    fused_src, fused_tgt = subset("fused_feat")
    # for fused use drug-specific slice
    fused_src = fused_src[:, drug_idx, :]
    fused_tgt = fused_tgt[:, drug_idx, :]

    shared_src, shared_tgt = subset("h_shared")
    gene_src, gene_tgt = subset("gene_feat")

    configs = [
        ("fused_feat p=50", fused_src, fused_tgt, 50),
        ("fused_feat p=15", fused_src, fused_tgt, 15),
        ("h_shared p=30", shared_src, shared_tgt, 30),
        ("gene_feat p=30", gene_src, gene_tgt, 30),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=args.dpi)
    src_t = data["source_tissue"][src_idx]
    tgt_t = data["target_tissue"][tgt_idx]

    for ax, (name, s_feat, t_feat, perp) in zip(axes.flat, configs):
        coords = compute_joint_tsne(s_feat, t_feat, args.seed, perp)
        plot_panel(ax, coords, src_t, tgt_t, name)

    from matplotlib.lines import Line2D
    tissue_handles = [Line2D([0], [0], marker="o", color="w",
                             markerfacecolor=VIVID_COLORS.get(t, "#2F4F4F"),
                             markeredgecolor="white", markersize=8,
                             label=tissue_names.get(t, f"Tissue {t}"))
                      for t in selected]
    domain_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=8, label="Source"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=8, label="Target"),
    ]
    fig.legend(handles=tissue_handles + domain_handles,
               labels=[h.get_label() for h in tissue_handles + domain_handles],
               loc="lower center", ncol=min(len(tissue_handles) + 2, 6),
               bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=9)

    fig.suptitle(f"{args.drug} t-SNE with Different Model Representations (same samples)",
                 fontsize=14, y=1.00)
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    base = os.path.join(args.output_dir, f"model_representation_comparison_{args.drug.lower()}")
    fig.savefig(base + ".png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {base}.png and {base}.pdf")


if __name__ == "__main__":
    main()
