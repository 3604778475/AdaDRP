#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone 1x3 figure for the three cleanly scanned hyperparameters:
K, alpha (fpi_gamma_max), and fpi_m_max (m).

Output:
  outputs/hyperparam_tuning/figures/clean_hyperparams.png
  outputs/hyperparam_tuning/figures/clean_hyperparams.pdf
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS = {
    "auc":   {"color": "#4c78a8", "label": "AUC"},
    "aupr":  {"color": "#808080", "label": "AUPR"},
    "f1":    {"color": "#b279a2", "label": "F1"},
    "acc":   {"color": "#f58518", "label": "ACC"},
}


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
    return pd.DataFrame(records)


def filter_and_group(df: pd.DataFrame, param: str, fixed: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for k, v in fixed.items():
        if k in df.columns:
            mask &= df[k].fillna("").astype(str) == v
        else:
            if v != "":
                mask = pd.Series(False, index=df.index)
    sub = df[mask].copy()
    if sub.empty:
        return sub
    sub = sub.groupby(param, as_index=False).agg(
        auc_mean=("auc_mean", "mean"),
        auc_std=("auc_std", "mean"),
        aupr_mean=("aupr_mean", "mean"),
        aupr_std=("aupr_std", "mean"),
        f1_mean=("f1_mean", "mean"),
        f1_std=("f1_std", "mean"),
        acc_mean=("acc_mean", "mean"),
        acc_std=("acc_std", "mean"),
    )
    sub = sub.rename(columns={param: "param_value"})
    sub["param_value"] = pd.to_numeric(sub["param_value"], errors="coerce")
    sub = sub.dropna(subset=["param_value"]).sort_values("param_value")
    return sub


def plot_panel(ax, title: str, sub: pd.DataFrame, xlabel: str, xtick_fmt: str):
    x = sub["param_value"].values
    for key, cfg in METRICS.items():
        y_mean = sub[f"{key}_mean"].values
        y_std = sub[f"{key}_std"].values
        ax.plot(x, y_mean, marker="o", color=cfg["color"], linestyle="-",
                label=cfg["label"], linewidth=2.5, markersize=7)
        ax.fill_between(x, np.clip(y_mean - y_std, 0.0, 1.0),
                        np.clip(y_mean + y_std, 0.0, 1.0),
                        color=cfg["color"], alpha=0.15)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Metric", fontsize=12)
    ax.set_ylim(0.50, 0.78)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(x)
    if xtick_fmt == "int":
        ax.set_xticklabels([str(int(v)) for v in x])
    elif xtick_fmt == "float":
        ax.set_xticklabels([f"{v:.2f}" for v in x])


def main():
    logs_dir = PROJECT_ROOT / "logs"
    out_dir = PROJECT_ROOT / "outputs" / "hyperparam_tuning" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_logs(str(logs_dir))

    default_base = {
        "pt": "100", "tr": "100", "ct": "100",
        "K": "9", "gmax": "0.1", "m": "16",
        "migrate": "0.01", "ortho": "0.001",
        "fgl": "10.001", "fp": "1024", "ginit": "0.05",
        "privdom": "0.005", "recon": "0.005", "qprior": "0.01",
        "h": "256", "o": "64",
        "detachnone": "", "global": "", "domsep": "", "fpi": "",
    }

    def make_fixed(vary: str, extra: dict | None = None) -> dict:
        fixed = {k: v for k, v in default_base.items() if k != vary}
        if extra:
            fixed.update(extra)
        return fixed

    k_sub = filter_and_group(df, "K", make_fixed("K", extra={"gmax": "0.1"}))
    a_sub = filter_and_group(df, "gmax", make_fixed("gmax", extra={"K": "9"}))
    m_sub = filter_and_group(df, "m", make_fixed("m", extra={"K": "9", "gmax": "0.1"}))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

    plot_panel(axes[0], r"(a) Sensitivity to $K$ ($\alpha=0.1$)", k_sub, "$K$", "int")
    plot_panel(axes[1], r"(b) Sensitivity to $\alpha$ ($K=9$)", a_sub, r"$\alpha$", "float")
    plot_panel(axes[2], r"(c) Sensitivity to fpi_m_max ($K=9, \alpha=0.1$)",
               m_sub, "fpi_m_max", "int")

    handles, labels = axes[0].get_legend_handles_labels()
    std_patch = mpatches.Patch(color="gray", alpha=0.25, label=r"$\pm$std")
    fig.legend(handles + [std_patch], labels + [std_patch.get_label()],
               loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.10), frameon=True, fontsize=11)

    fig.suptitle("Hyperparameter Sensitivity (AUC/AUPR/F1/ACC with $\pm$std band)",
                 fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0.09, 1, 1])

    base = out_dir / "clean_hyperparams"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {base}.png and {base}.pdf")


if __name__ == "__main__":
    main()
