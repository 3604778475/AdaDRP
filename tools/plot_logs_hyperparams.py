#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive hyperparameter sensitivity using the detailed logs.

Plots AUC/AUPR/F1/ACC (mean as solid lines, std as dashed lines) for every
parameter that has non-default values in logs/*/brief_logs.txt.

Output:
  outputs/hyperparam_tuning/figures/logs_hyperparams_full.png
  outputs/hyperparam_tuning/figures/logs_hyperparams_full.pdf
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
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
    """Filter rows to those matching `fixed` and group/average by `param`."""
    mask = pd.Series(True, index=df.index)
    for k, v in fixed.items():
        if k in df.columns:
            mask &= df[k].fillna("").astype(str) == v
        else:
            # if key absent, only match when v is empty
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
    sub[param] = pd.to_numeric(sub[param], errors="coerce")
    sub = sub.dropna(subset=[param]).sort_values(param)
    return sub


def plot_panel(ax, title: str, sub: pd.DataFrame, param: str, xlabel: str):
    x = sub[param].values
    for key, cfg in METRICS.items():
        y_mean = sub[f"{key}_mean"].values
        y_std = sub[f"{key}_std"].values
        ax.plot(x, y_mean, marker="o", color=cfg["color"], linestyle="-",
                label=cfg["label"], linewidth=2, markersize=6)
        ax.fill_between(x, np.clip(y_mean - y_std, 0.0, 1.0),
                        np.clip(y_mean + y_std, 0.0, 1.0),
                        color=cfg["color"], alpha=0.12)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Metric", fontsize=10)
    ax.set_ylim(0.50, 0.78)
    ax.grid(True, alpha=0.3)

    if param in ("K", "m"):
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x])
    elif param == "gmax":
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.2f}" for v in x])


def main():
    logs_dir = PROJECT_ROOT / "logs"
    out_dir = PROJECT_ROOT / "outputs" / "hyperparam_tuning" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_logs(str(logs_dir))
    print(f"Loaded {len(df)} valid logs.")

    default_base = {
        "pt": "100", "tr": "100", "ct": "100",
        "K": "9", "gmax": "0.1", "m": "16",
        "migrate": "0.01", "ortho": "0.001",
        "fgl": "10.001", "fp": "1024", "ginit": "0.05",
        "privdom": "0.005", "recon": "0.005", "qprior": "0.01",
        "h": "256", "o": "64",
        "detachnone": "", "global": "", "domsep": "", "fpi": "",
    }

    # Define each parameter and the baseline used to isolate it.
    # default_base is copied and the varied parameter (plus any co-varied params)
    # is removed so it can vary.
    def make_fixed(vary: str, also_vary: tuple = (), extra: dict | None = None) -> dict:
        exclude = {vary, *also_vary}
        fixed = {k: v for k, v in default_base.items() if k not in exclude}
        if extra:
            fixed.update(extra)
        return fixed

    panels = [
        {
            "param": "K",
            "title": r"Sensitivity to $K$ ($\alpha=0.1$)",
            "xlabel": "$K$",
            "fixed": make_fixed("K", extra={"gmax": "0.1"}),
        },
        {
            "param": "gmax",
            "title": r"Sensitivity to $\alpha$ ($K=9$)",
            "xlabel": r"$\alpha$",
            "fixed": make_fixed("gmax", extra={"K": "9"}),
        },
        {
            "param": "m",
            "title": r"Sensitivity to fpi_m_max ($K=9, \alpha=0.1$)",
            "xlabel": "fpi_m_max",
            "fixed": make_fixed("m", extra={"K": "9", "gmax": "0.1"}),
        },
        {
            "param": "migrate",
            "title": r"Sensitivity to migrate ($K=4, \alpha=0.1$)",
            "xlabel": "migrate",
            "fixed": make_fixed("migrate", extra={"K": "4", "gmax": "0.1", "m": "16"}),
        },
        {
            "param": "ortho",
            "title": r"Sensitivity to ortho ($K=4, \alpha=0.1$, migrate free)",
            "xlabel": "ortho",
            "fixed": make_fixed("ortho", also_vary=("migrate",), extra={"K": "4", "gmax": "0.1", "m": "16"}),
        },
        {
            "param": "ct",
            "title": r"Sensitivity to ct ($K=9, \alpha=0.1$, pt free)",
            "xlabel": "ct epochs",
            "fixed": make_fixed("ct", also_vary=("pt",), extra={"K": "9", "gmax": "0.1", "m": "16"}),
        },
        {
            "param": "pt",
            "title": r"Sensitivity to pt ($K=9, \alpha=0.1$, ct free)",
            "xlabel": "pt epochs",
            "fixed": make_fixed("pt", also_vary=("ct",), extra={"K": "9", "gmax": "0.1", "m": "16"}),
        },
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharey=True)
    axes = axes.flatten()

    nonempty_panels = []
    for p in panels:
        sub = filter_and_group(df, p["param"], p["fixed"])
        if not sub.empty and len(sub) >= 2:
            nonempty_panels.append((p, sub))
        else:
            print(f"Skipping {p['param']}: not enough data under chosen baseline.")

    for ax, (p, sub) in zip(axes, nonempty_panels):
        plot_panel(ax, p["title"], sub, p["param"], p["xlabel"])

    # hide unused axes
    for ax in axes[len(nonempty_panels):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    # only keep metric labels (skip the dashed std duplicates)
    unique = [(h, l) for h, l in zip(handles, labels) if "std" not in l]
    fig.legend(*zip(*unique), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), frameon=True, fontsize=10)

    fig.suptitle(r"Hyperparameter Sensitivity from Logs (AUC/AUPR/F1/ACC with $\pm$std band)",
                 fontsize=14, y=1.00)
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    base = out_dir / "logs_hyperparams_full"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {base}.png and {base}.pdf")


if __name__ == "__main__":
    main()
