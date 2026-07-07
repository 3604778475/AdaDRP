#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse experiment logs and reproduce the MACB-DRP hyperparameter sensitivity figure
(Sensitivity to K / Sensitivity to alpha) with broken y-axes.

Usage:
    conda run -n macb_drp python tools/plot_hyperparam_sensitivity.py \
        --output-dir outputs/hyperparam_tuning/figures
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-dir", type=str, default=str(PROJECT_ROOT / "logs"),
        help="Directory containing experiment log folders."
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "hyperparam_tuning" / "figures"),
        help="Directory to save PNG/PDF outputs."
    )
    parser.add_argument(
        "--k-fixed-alpha", type=float, default=0.1,
        help="Alpha value at which the K-sensitivity curve is computed."
    )
    parser.add_argument(
        "--alpha-fixed-k", type=int, default=9,
        help="K value at which the alpha-sensitivity curve is computed."
    )
    return parser.parse_args()


def parse_brief_log(path: str) -> Dict[str, float] | None:
    """Extract K, alpha, and overall AUC/AUPR/ACC/F1 mean/std from a brief_logs.txt."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 15:
        return None

    config_line = lines[1].strip()
    m_k = re.search(r"K_(\d+)", config_line)
    m_alpha = re.search(r"fpi_gmax([0-9.]+)", config_line)
    if not m_k or not m_alpha:
        return None

    k = int(m_k.group(1))
    alpha = float(m_alpha.group(1))

    # The overall metric line looks like:
    # auc：0.6466/0.0780  aupr：0.7696/0.1440  acc：0.7013/0.2335  f1：0.6892/0.0977
    metric_line = None
    for line in lines:
        if line.strip().startswith("auc：") or line.strip().startswith("auc:"):
            metric_line = line.strip()
            break
    if metric_line is None:
        return None

    pattern = re.compile(r"(auc|aupr|acc|f1)[：:]\s*([0-9.]+)/([0-9.]+)")
    matches = pattern.findall(metric_line)
    if len(matches) != 4:
        return None

    record = {"K": k, "alpha": alpha}
    for metric, mean_s, std_s in matches:
        record[f"{metric}_mean"] = float(mean_s)
        record[f"{metric}_std"] = float(std_s)
    return record


def aggregate_logs(logs_dir: str) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for root, _, files in os.walk(logs_dir):
        if "brief_logs.txt" in files:
            rec = parse_brief_log(os.path.join(root, "brief_logs.txt"))
            if rec is not None:
                records.append(rec)
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid brief_logs.txt found under {logs_dir}")
    # Average across repeated runs with the same (K, alpha)
    grouped = df.groupby(["K", "alpha"], as_index=False).mean(numeric_only=True)
    return grouped


def build_sensitivity_series(
    df: pd.DataFrame,
    param: str,
    fixed_param: str,
    fixed_value: float,
) -> pd.DataFrame:
    """Filter df to rows where fixed_param == fixed_value and sort by param."""
    sub = df[df[fixed_param] == fixed_value].copy()
    sub = sub.sort_values(param)
    return sub


def compute_average(row: pd.Series) -> Tuple[float, float]:
    """Average of AUC/AUPR/ACC/F1 means and stds."""
    means = [row["auc_mean"], row["aupr_mean"], row["acc_mean"], row["f1_mean"]]
    stds = [row["auc_std"], row["aupr_std"], row["acc_std"], row["f1_std"]]
    return float(np.mean(means)), float(np.mean(stds))



def add_break_markers(ax_top, ax_bot, d=0.012):
    """Draw small diagonal lines on the left and right spines to indicate a y-axis break."""
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


def plot_sensitivity(
    k_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col", height_ratios=[2, 1])

    metric_config = {
        "auc":   {"color": "#4c78a8", "label": "AUC"},
        "aupr":  {"color": "#808080", "label": "AUPR"},
        "f1":    {"color": "#b279a2", "label": "F1"},
        "acc":   {"color": "#f58518", "label": "ACC"},
    }

    panels = [
        (k_df, "K", "Sensitivity to $K$", 0),
        (alpha_df, "alpha", r"Sensitivity to $\alpha$", 1),
    ]

    for df, xlabel, title, col in panels:
        x = df["K"].values if xlabel == "K" else df["alpha"].values
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        for key, cfg in metric_config.items():
            y_mean = df[f"{key}_mean"].values
            y_std = df[f"{key}_std"].values

            ax_top.plot(x, y_mean, marker="o", color=cfg["color"], label=cfg["label"], linestyle="-")
            ax_bot.plot(x, y_std, marker="o", color=cfg["color"], linestyle="--", alpha=0.7)

        ax_top.set_title(title, fontsize=13)
        ax_top.set_ylim(0.55, 0.78)
        ax_bot.set_ylim(0.05, 0.18)
        x_pad = 0.5 if xlabel == "K" else 0.02
        ax_top.set_xlim(x.min() - x_pad, x.max() + x_pad)
        ax_bot.set_xlim(x.min() - x_pad, x.max() + x_pad)

        # Broken-axis styling
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(labelbottom=False, bottom=False)
        add_break_markers(ax_top, ax_bot)

        # X-axis labels only on bottom
        ax_bot.set_xlabel("$K$" if xlabel == "K" else r"$\alpha$", fontsize=12)

        # Tick formatting
        if xlabel == "K":
            ax_bot.set_xticks(x)
            ax_bot.set_xticklabels([str(int(v)) for v in x], rotation=0)
        else:
            ax_bot.set_xticks(x)
            ax_bot.set_xticklabels([f"{v:.2f}" for v in x], rotation=0)

    # Y labels
    axes[0, 0].set_ylabel("Metric", fontsize=11)
    axes[1, 0].set_ylabel("Std", fontsize=11)

    # Shared legend below plots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    std_handles = [
        plt.Line2D([0], [0], color=metric_config[k]["color"], linestyle="--", marker="o", alpha=0.7,
                   label=metric_config[k]["label"] + "-std")
        for k in metric_config
    ]
    fig.legend(
        handles + std_handles,
        labels + [h.get_label() for h in std_handles],
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    base = os.path.join(output_dir, "hyperparam_sensitivity")
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {base}.png and {base}.pdf")


def main():
    args = parse_args()
    df = aggregate_logs(args.logs_dir)
    print(f"Aggregated {len(df)} unique (K, alpha) configurations from logs.")

    os.makedirs(args.output_dir, exist_ok=True)
    agg_path = os.path.join(args.output_dir, "aggregated_metrics.csv")
    df.round(4).to_csv(agg_path, index=False)
    print(f"Saved aggregated metrics to {agg_path}")

    k_df = build_sensitivity_series(df, "K", "alpha", args.k_fixed_alpha)
    alpha_df = build_sensitivity_series(df, "alpha", "K", args.alpha_fixed_k)

    print(f"K series (alpha={args.k_fixed_alpha}): K = {k_df['K'].tolist()}")
    print(f"Alpha series (K={args.alpha_fixed_k}): alpha = {alpha_df['alpha'].tolist()}")

    if k_df.empty or alpha_df.empty:
        raise ValueError("Could not build sensitivity series; check fixed K/alpha values.")

    plot_sensitivity(k_df, alpha_df, args.output_dir)


if __name__ == "__main__":
    main()
