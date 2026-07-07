#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Side-by-side comparison of the right panels (Docetaxel tissue alignment)
for raw input features vs. model-learned features.

Usage:
    D:/anaconda/envs/macb_drp/python.exe tools/compare_raw_model_tsne.py \
        --raw-csv artifacts/tsne_raw/fold_1/figures/right_panel_coordinates.csv \
        --model-csv artifacts/tsne/fold_1/figures_aligned/right_panel_coordinates.csv \
        --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIVID_COLORS = {
    5:  "#B22222",   # Colon
    11: "#DAA520",   # Head and Neck
    12: "#228B22",   # Kidney
    14: "#008B8B",   # Liver
    15: "#4169E1",   # Lung
    21: "#9400D3",   # Ovarian
    23: "#FF1493",   # Prostate
    31: "#FF8C00",   # Uterine
    1:  "#8B4513",   # Bladder
    4:  "#20B2AA",   # Breast
    25: "#800080",   # Skin
    28: "#FF6347",   # Testis
    29: "#4682B4",   # Thymoma
    30: "#9ACD32",   # Thyroid
    3:  "#D2691E",   # Brain
    22: "#708090",   # Pancreas
    9:  "#2E8B57",   # Gallbladder
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=str,
                        default=str(PROJECT_ROOT / "artifacts" / "tsne_raw" / "fold_1" / "figures" / "right_panel_coordinates.csv"))
    parser.add_argument("--model-csv", type=str,
                        default=str(PROJECT_ROOT / "artifacts" / "tsne" / "fold_1" / "figures_aligned" / "right_panel_coordinates.csv"))
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "figures"))
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_right_panel(csv_path: str):
    df = pd.read_csv(csv_path)
    df["tissue"] = df["tissue"].astype(int)
    return df


def plot_panel(ax, df: pd.DataFrame, title: str):
    tissues = sorted(df["tissue"].unique())
    color_map = {t: VIVID_COLORS.get(t, "#2F4F4F") for t in tissues}
    name_map = {t: df[df["tissue"] == t]["tissue_name"].iloc[0] for t in tissues}

    for tissue in tissues:
        color = color_map[tissue]
        name = name_map[tissue]
        src = df[(df["tissue"] == tissue) & (df["domain"] == "source")]
        tgt = df[(df["tissue"] == tissue) & (df["domain"] == "target")]
        if len(src) > 0:
            ax.scatter(src["x"], src["y"], c=color, marker="o", s=60,
                       alpha=0.8, edgecolors="white", linewidths=0.5,
                       label=name if len(tgt) == 0 else "")
        if len(tgt) > 0:
            ax.scatter(tgt["x"], tgt["y"], c=color, marker="^", s=70,
                       alpha=0.8, edgecolors="white", linewidths=0.5,
                       label=name if len(src) == 0 else "")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw_df = load_right_panel(args.raw_csv)
    model_df = load_right_panel(args.model_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=args.dpi)
    plot_panel(axes[0], raw_df, "(a) Raw Input")
    plot_panel(axes[1], model_df, "(b) After Model")

    # Common tissue legend
    all_tissues = sorted(set(raw_df["tissue"].unique()) | set(model_df["tissue"].unique()))
    color_map = {t: VIVID_COLORS.get(t, "#2F4F4F") for t in all_tissues}
    name_map = {}
    for t in all_tissues:
        for df in [raw_df, model_df]:
            sub = df[df["tissue"] == t]
            if len(sub) > 0:
                name_map[t] = sub["tissue_name"].iloc[0]
                break

    from matplotlib.lines import Line2D
    tissue_handles = [Line2D([0], [0], marker="o", color="w",
                             markerfacecolor=color_map[t], markeredgecolor="white",
                             markersize=9, label=name_map.get(t, f"Tissue {t}"))
                      for t in all_tissues]
    domain_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=9, label="Source"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markeredgecolor="white", markersize=9, label="Target"),
    ]

    fig.legend(handles=tissue_handles + domain_handles,
               labels=[h.get_label() for h in tissue_handles + domain_handles],
               loc="lower center", ncol=min(len(tissue_handles) + 2, 6),
               bbox_to_anchor=(0.5, -0.06), frameon=True, fontsize=10)

    fig.suptitle("Docetaxel Tissue Alignment: Raw Input vs. Model Output",
                 fontsize=15, y=1.02)
    fig.tight_layout(rect=[0, 0.07, 1, 1])

    base = os.path.join(args.output_dir, "raw_vs_model_tsne_right")
    fig.savefig(base + ".png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {base}.png and {base}.pdf")


if __name__ == "__main__":
    main()
