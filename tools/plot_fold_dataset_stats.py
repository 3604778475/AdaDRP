#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize per-fold source/target dataset distribution statistics.

Usage:
    source activate macb_drp
    python tools/plot_fold_dataset_stats.py \
        --stats-file path/to/fold_stats.csv \
        --output-dir outputs/fold_dataset_stats

Expected CSV columns:
    Fold, Source samples, Source tissues, Source pairs,
    Source pos, Source neg, Target samples, Target tissues,
    Target pairs, Target pos, Target neg
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats-file", type=str, default=None,
        help="CSV with per-fold dataset statistics."
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/fold_dataset_stats",
        help="Output directory for PNG/PDF."
    )
    return parser.parse_args()


def get_default_stats() -> pd.DataFrame:
    """Default fold statistics computed from the project dataloaders."""
    data = {
        "Fold": [0, 1, 2, 3, 4],
        "Source samples": [572, 572, 572, 572, 572],
        "Source tissues": [24, 26, 25, 26, 26],
        "Source pairs": [4859, 4870, 4881, 4864, 4850],
        "Source pos": [2586, 2572, 2530, 2506, 2590],
        "Source neg": [2273, 2298, 2351, 2358, 2260],
        "Target samples": [1186, 1186, 1186, 1186, 1186],
        "Target tissues": [20, 20, 20, 20, 20],
        "Target pairs": [1807, 1807, 1807, 1807, 1807],
        "Target pos": [1201, 1201, 1201, 1201, 1201],
        "Target neg": [606, 606, 606, 606, 606],
    }
    return pd.DataFrame(data)


def load_stats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "Fold", "Source samples", "Source tissues", "Source pairs",
        "Source pos", "Source neg",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    # Fill target columns with defaults if not provided
    defaults = {
        "Target samples": 1186,
        "Target tissues": 20,
        "Target pairs": 1807,
        "Target pos": 1201,
        "Target neg": 606,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df


def add_value_labels(ax, bars, fmt="{:d}"):
    """Annotate bar tops with their values."""
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                fmt.format(int(height)),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7,
            )


def plot_single_figure(df: pd.DataFrame, output_dir: str):
    """
    Draw all five statistics in a single figure with 5 sub-panels (2 rows x 3 cols).
    """
    os.makedirs(output_dir, exist_ok=True)
    folds = df["Fold"].astype(int).values
    x = np.arange(len(folds))
    width = 0.55

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    axes = axes.flatten()

    panels = [
        {
            "col": "Source samples",
            "target_col": "Target samples",
            "ylabel": "Number of samples",
            "title": "Number of Samples",
            "color": "#4C78A8",
        },
        {
            "col": "Source tissues",
            "target_col": "Target tissues",
            "ylabel": "Number of tissues",
            "title": "Number of Tissues",
            "color": "#54A24B",
        },
        {
            "col": "Source pairs",
            "target_col": "Target pairs",
            "ylabel": "Tissue--drug pairs",
            "title": "Tissue--Drug Pairs",
            "color": "#F58518",
        },
        {
            "col": "Source pos",
            "target_col": "Target pos",
            "ylabel": "Number of responses",
            "title": "Positive Responses",
            "color": "#72B7B2",
        },
        {
            "col": "Source neg",
            "target_col": "Target neg",
            "ylabel": "Number of responses",
            "title": "Negative Responses",
            "color": "#B279A2",
        },
    ]

    for idx, cfg in enumerate(panels):
        ax = axes[idx]
        bars = ax.bar(x, df[cfg["col"]], width, color=cfg["color"])
        target_val = df[cfg["target_col"]].iloc[0]
        ax.axhline(target_val, color="#E45756", linestyle="--", linewidth=2)
        # Annotate target value at the right end of the dashed line
        ax.text(len(folds) - 0.45, target_val, f"{target_val}",
                va="center", ha="left", fontsize=9, color="#E45756",
                fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {f}" for f in folds])
        ax.set_ylabel(cfg["ylabel"])
        ax.set_title(cfg["title"], fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Add headroom so value labels are not clipped
        y_max = max(df[cfg["col"]].max(), target_val)
        ax.set_ylim(0, y_max * 1.15)
        add_value_labels(ax, bars)

    # Use the unused 6th panel for a global legend + summary text
    axes[-1].axis("off")
    source_patch = mpatches.Patch(color="gray", label="Source (GDSC)")
    target_line = mlines.Line2D(
        [], [], color="#E45756", linestyle="--", linewidth=2,
        label="Target (TCGA)"
    )
    legend = axes[-1].legend(
        handles=[source_patch, target_line],
        loc="upper left",
        title="Legend",
        fontsize=11,
        title_fontsize=12,
        frameon=True,
    )
    # Place summary text below the legend
    summary_text = (
        "Dataset distribution across 5 folds\n"
        "(Source = GDSC/CCLE train split, Target = TCGA)\n\n"
        f"Source samples: {df['Source samples'].iloc[0]} per fold\n"
        f"Target samples: {df['Target samples'].iloc[0]}\n"
        f"Target tissues: {df['Target tissues'].iloc[0]}\n"
        f"Target pairs: {df['Target pairs'].iloc[0]}\n"
        f"Target pos/neg: {df['Target pos'].iloc[0]} / {df['Target neg'].iloc[0]}"
    )
    axes[-1].text(0.05, 0.55, summary_text, transform=axes[-1].transAxes,
                  fontsize=11, verticalalignment="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    fig.suptitle("Per-Fold Source--Target Dataset Distribution (Five Metrics)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    base = os.path.join(output_dir, "fold_dataset_distribution")
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {base}.png and {base}.pdf")


def main():
    args = parse_args()
    df = load_stats(args.stats_file) if args.stats_file else get_default_stats()
    plot_single_figure(df, args.output_dir)


if __name__ == "__main__":
    main()
