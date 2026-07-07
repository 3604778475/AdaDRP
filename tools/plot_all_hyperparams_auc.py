#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive AUC-only hyperparameter sensitivity figure.

Uses:
  - logs/*/brief_logs.txt for K and fpi_gamma_max (alpha) with mean ± std.
  - outputs/hyperparam_tuning/S3_S8_results.csv for the remaining parameters
    (num_programs, beta, lambda_ot, fpi_m_max).

Output:
  outputs/hyperparam_tuning/figures/all_hyperparams_auc.png
  outputs/hyperparam_tuning/figures/all_hyperparams_auc.pdf
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_config_line(cfg: str) -> dict:
    tokens = cfg.split("_")
    rec = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        m = re.match(r"^([a-zA-Z]+)([0-9.]+|none|true|false)?$", t)
        if not m:
            i += 1
            continue
        key = m.group(1)
        val = m.group(2)
        if val is None:
            if i + 1 < len(tokens) and re.match(r"^[0-9.]+$", tokens[i + 1]):
                val = tokens[i + 1]
                i += 1
            else:
                val = ""
        rec[key] = val
        i += 1
    return rec


def parse_brief_log(path: str) -> dict | None:
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 15:
        return None
    rec = parse_config_line(lines[1].strip())
    metric_line = None
    for line in lines:
        s = line.strip()
        if s.startswith("auc：") or s.startswith("auc:"):
            metric_line = s
            break
    if metric_line is None:
        return None
    m = re.findall(r"(auc|aupr|acc|f1)[：:]\s*([0-9.]+)/([0-9.]+)", metric_line)
    if len(m) != 4:
        return None
    for metric, mean_s, std_s in m:
        rec[f"{metric}_mean"] = float(mean_s)
        rec[f"{metric}_std"] = float(std_s)
    return rec


def load_logs(logs_dir: str) -> pd.DataFrame:
    records = []
    for root, _, files in os.walk(logs_dir):
        if "brief_logs.txt" in files:
            rec = parse_brief_log(os.path.join(root, "brief_logs.txt"))
            if rec is not None:
                records.append(rec)
    df = pd.DataFrame(records)
    return df


def baseline_mask(df: pd.DataFrame, vary: str) -> pd.Series:
    """Return a boolean mask selecting runs where every other tuned parameter is at its default."""
    defaults = {
        "pt": "100", "tr": "100", "ct": "100", "K": "9", "gmax": "0.1", "m": "16",
        "migrate": "0.01", "ortho": "0.001", "fgl": "10.001", "fp": "1024",
        "ginit": "0.05", "privdom": "0.005", "recon": "0.005", "qprior": "0.01",
        "h": "256", "o": "64", "detachnone": "", "global": "", "domsep": "", "fpi": "",
    }
    mask = pd.Series(True, index=df.index)
    for k, v in defaults.items():
        if k == vary:
            continue
        if k in df.columns:
            mask &= df[k].fillna("").astype(str) == v
    return mask


def series_from_logs(df: pd.DataFrame, param: str) -> pd.DataFrame:
    sub = df[baseline_mask(df, param)].copy()
    if sub.empty:
        return sub
    sub = sub.groupby(param, as_index=False).agg(
        auc_mean=("auc_mean", "mean"),
        auc_std=("auc_std", "mean"),
    )
    sub = sub.rename(columns={param: "param_value"})
    sub["param_value"] = pd.to_numeric(sub["param_value"], errors="coerce")
    sub = sub.dropna(subset=["param_value"]).sort_values("param_value")
    return sub


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["auc"].notna()]
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df["param_value"] = pd.to_numeric(df["param_value"], errors="coerce")
    df = df.dropna(subset=["auc", "param_value"])
    return df


def plot_panel(ax, param_name: str, sub: pd.DataFrame, from_logs: bool):
    x = sub["param_value"].values
    y = sub["auc_mean"].values if from_logs else sub["auc"].values

    ax.plot(x, y, marker="o", color="#4c78a8", linestyle="-", linewidth=2,
            label="AUC" if from_logs else None, markersize=7)

    if from_logs and "auc_std" in sub.columns:
        std = sub["auc_std"].values
        ax.plot(x, std, marker="o", color="#4c78a8", linestyle="--",
                alpha=0.6, label="AUC-std", markersize=5)

    ax.set_title(param_name, fontsize=12)
    ax.set_xlabel(get_xlabel(param_name), fontsize=10)
    ax.set_ylabel("AUC", fontsize=10)
    ax.set_ylim(0.52, 0.68)
    ax.grid(True, alpha=0.3)

    # Format x-ticks nicely
    if param_name in ("K", "num_programs", "fpi_m_max"):
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x])
    elif param_name == r"$\alpha$":
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.2f}" for v in x])


def get_xlabel(name: str) -> str:
    labels = {
        "K": "$K$",
        r"$\alpha$": r"$\alpha$",
        "num_programs": "num programs",
        r"$\beta$": r"$\beta$",
        r"$\lambda_{\mathrm{OT}}$": r"$\lambda_{\mathrm{OT}}$",
        "fpi_m_max": "fpi_m_max",
    }
    return labels.get(name, name)


def main():
    logs_dir = PROJECT_ROOT / "logs"
    csv_path = PROJECT_ROOT / "outputs" / "hyperparam_tuning" / "S3_S8_results.csv"
    out_dir = PROJECT_ROOT / "outputs" / "hyperparam_tuning" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    logs_df = load_logs(str(logs_dir))
    csv_df = load_csv(str(csv_path))

    # Parameters to plot: (panel title, source, from_logs flag)
    k_series = series_from_logs(logs_df, "K")
    alpha_series = series_from_logs(logs_df, "gmax")

    panels = [
        ("num_programs", csv_df[csv_df["param_name"] == "num_programs"].copy(), False),
        (r"$\beta$", csv_df[csv_df["param_name"] == "beta"].copy(), False),
        (r"$\lambda_{\mathrm{OT}}$", csv_df[csv_df["param_name"] == "lambda_ot"].copy(), False),
        ("fpi_m_max", csv_df[csv_df["param_name"] == "fpi_m_max"].copy(), False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()

    for ax, (title, sub, from_logs) in zip(axes, panels):
        if sub.empty:
            ax.set_title(f"{title} (no data)")
            ax.set_ylabel("AUC")
            continue
        # For CSV panels, rename columns to uniform names
        if not from_logs:
            sub = sub.rename(columns={"param_value": "param_value", "auc": "auc"})
        plot_panel(ax, title, sub, from_logs)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle("Other Hyperparameter Sensitivities (AUC)", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    base = out_dir / "other_hyperparams_auc"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {base}.png and {base}.pdf")


if __name__ == "__main__":
    main()
