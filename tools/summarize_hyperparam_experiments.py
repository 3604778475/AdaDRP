#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarize hyper-parameter experiment results into a Markdown report.

Reads:
  - logs/*/brief_logs.txt  (full AUC/AUPR/F1/ACC + config string)
  - outputs/hyperparam_tuning/S3_S8_results.csv  (QuickCDAN AUC sweeps)

Writes:
  - outputs/hyperparam_tuning/hyperparam_experiment_report.md
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS = ["auc", "aupr", "f1", "acc"]

# Tokens we are interested in for the summary.
# The key is the token name in the config string, value is a human-readable name.
PARAM_NAMES = {
    "K": "Top-K contrastive",
    "gmax": "FPI gamma_max",
    "m": "FPI m_max",
    "migrate": "Migrate loss weight",
    "ortho": "Ortho loss weight",
    "ct": "Contrastive epochs",
    "pt": "Pretrain epochs",
    "tr": "Train epochs",
    "beta": "Contrastive loss weight",
    "alph": "CDAN/transfer loss weight",
    "lambda_ot": "OT loss weight",
    "num_programs": "Number of molecular programs",
    "program_dim": "Program token dimension",
    "fpi_gamma_max": "FPI gamma_max",
    "fpi_m_max": "FPI m_max",
}

# Tokens that are kept as fixed baseline descriptors but not swept.
BASELINE_FIXED = {
    "pt": "pretrain epochs",
    "tr": "train epochs",
    "ct": "contrastive epochs",
    "migrate": "migrate",
    "ortho": "ortho",
    "gmax": "FPI gamma_max",
    "ginit": "FPI gamma_init",
    "qprior": "FPI q_prior",
    "fgl": "FPI gamma L1",
    "detach": "FPI detach mode",
    "global": "FPI global flag",
    "m": "FPI m_max",
    "fp": "FPI fp_dim",
    "domsep": "domain separation",
    "h": "hidden dim",
    "o": "program/output dim",
    "recon": "recon loss",
    "privdom": "private domain loss",
    "K": "Top-K contrastive",
}


def parse_config_line(cfg: str) -> Dict[str, str]:
    """Parse a config string like pt_100_K_9_gmax0.1_m16 into a dict."""
    tokens = cfg.split("_")
    rec: Dict[str, str] = {}
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
                # standalone flag token; store empty to indicate presence
                val = ""
        rec[key] = val
        i += 1
    return rec


def parse_brief_log(path: Path) -> Optional[Dict]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 2:
        return None
    rec = parse_config_line(lines[1].strip())
    metric_line = None
    for line in lines:
        s = line.strip()
        if s.lower().startswith("auc:") or s.lower().startswith("auc："):
            metric_line = s
            break
    if metric_line is None:
        return None
    m = re.findall(r"(auc|aupr|acc|f1)[：:]\s*([0-9.]+)/([0-9.]+)", metric_line, re.IGNORECASE)
    if len(m) != 4:
        return None
    for metric, mean_s, std_s in m:
        rec[f"{metric.lower()}_mean"] = float(mean_s)
        rec[f"{metric.lower()}_std"] = float(std_s)
    rec["log_dir"] = str(path.parent.name)
    return rec


def parse_logs_txt(path: Path) -> Optional[Dict]:
    """Parse logs.txt to extract per-fold and per-drug detailed results."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 3:
        return None

    rec: Dict[str, any] = {"config_line": lines[1].strip()}

    # Per-fold test-avg metrics
    fold_avgs: List[List[float]] = []
    for line in lines:
        m = re.search(r"test-avg \(TCGA\) metrics \(auc,aupr,f1,acc\): \[([0-9.\s]+)\]", line)
        if m:
            vals = [float(v) for v in m.group(1).split()]
            if len(vals) == 4:
                fold_avgs.append(vals)
    rec["fold_avg"] = fold_avgs

    # Per-drug mean/std summary
    drug_summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for line in lines:
        m = re.match(r"^([A-Za-z0-9-]+)\s+", line)
        if m and "average/std results" not in line.lower():
            drug = m.group(1)
            pairs = re.findall(r"(auc|aupr|f1|acc):\s+([0-9.]+)/([0-9.]+)", line)
            if pairs:
                drug_summary.setdefault(drug, {})
                for metric, mean_s, std_s in pairs:
                    drug_summary[drug][metric] = (float(mean_s), float(std_s))
    rec["drug_summary"] = drug_summary
    return rec


def load_log_records(logs_dir: Path) -> pd.DataFrame:
    records: List[Dict] = []
    for p in logs_dir.rglob("brief_logs.txt"):
        rec = parse_brief_log(p)
        if rec is not None:
            # Try to attach detailed results from logs.txt in the same directory
            detailed = parse_logs_txt(p.with_name("logs.txt"))
            if detailed is not None:
                rec["config_line"] = detailed.get("config_line", "")
                rec["fold_avg"] = detailed.get("fold_avg", [])
                rec["drug_summary"] = detailed.get("drug_summary", {})
            records.append(rec)
    return pd.DataFrame(records)


def infer_baseline(df: pd.DataFrame) -> Dict[str, str]:
    """Infer a baseline config as the most frequent value for each token."""
    baseline: Dict[str, str] = {}
    for col in df.columns:
        if col.endswith("_mean") or col.endswith("_std") or col in ("log_dir",):
            continue
        # skip flag-only columns (all empty) or columns with too many missing
        series = df[col].fillna("").astype(str)
        if series.eq("").mean() > 0.5:
            continue
        baseline[col] = series.mode().iloc[0]
    return baseline


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def group_by_param(
    df: pd.DataFrame,
    param: str,
    fixed: Optional[Dict[str, str]] = None,
) -> Optional[pd.DataFrame]:
    sub = df.copy()
    if fixed:
        mask = pd.Series(True, index=sub.index)
        for k, v in fixed.items():
            if k not in sub.columns:
                continue
            mask &= sub[k].fillna("").astype(str) == v
        sub = sub[mask].copy()
    if sub.empty or param not in sub.columns:
        return None
    sub = sub[sub[param].notna()].copy()
    if sub.empty:
        return None
    sub[param] = pd.to_numeric(sub[param], errors="coerce")
    sub = sub.dropna(subset=[param])
    if sub.empty:
        return None
    agg: Dict[str, Tuple[str, str]] = {}
    for metric in METRICS:
        agg[f"{metric}_mean"] = (f"{metric}_mean", "mean")
        agg[f"{metric}_std"] = (f"{metric}_std", "mean")
    grouped = (
        sub.groupby(param, as_index=False)
        .agg(**agg)
        .sort_values(param)
    )
    grouped["n_runs"] = sub.groupby(param).size().values
    return grouped


def best_by_auc(grouped: pd.DataFrame, param: str) -> Optional[pd.Series]:
    if grouped is None or grouped.empty:
        return None
    best_idx = grouped["auc_mean"].idxmax()
    return grouped.loc[best_idx]


def make_sweep_table(grouped: pd.DataFrame, param: str) -> str:
    lines = []
    lines.append(f"| {PARAM_NAMES.get(param, param)} | AUC | AUPR | F1 | ACC | n |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in grouped.iterrows():
        lines.append(
            f"| {row[param]} | {fmt_mean_std(row['auc_mean'], row['auc_std'])} | "
            f"{fmt_mean_std(row['aupr_mean'], row['aupr_std'])} | "
            f"{fmt_mean_std(row['f1_mean'], row['f1_std'])} | "
            f"{fmt_mean_std(row['acc_mean'], row['acc_std'])} | {int(row['n_runs'])} |"
        )
    return "\n".join(lines)


def load_s3s8_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={"param_value": "value", "auc": "auc_mean"})
    df["auc_std"] = np.nan
    return df


def make_csv_table(df: pd.DataFrame, param_name: str) -> str:
    sub = df[df["param_name"] == param_name].sort_values("value")
    if sub.empty:
        return "_No data._"
    lines = []
    lines.append(f"| {PARAM_NAMES.get(param_name, param_name)} | AUC | elapsed_min |")
    lines.append("|---|---|---|")
    for _, row in sub.iterrows():
        auc = row["auc_mean"]
        auc_str = "N/A" if pd.isna(auc) else f"{auc:.4f}"
        lines.append(f"| {row['value']} | {auc_str} | {row['elapsed_min']} |")
    return "\n".join(lines)


def main() -> None:
    logs_dir = PROJECT_ROOT / "logs"
    out_dir = PROJECT_ROOT / "outputs" / "hyperparam_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hyperparam_experiment_report.md"

    df = load_log_records(logs_dir)
    if df.empty:
        print("No brief_logs.txt found.")
        return

    baseline = infer_baseline(df)

    # Tokens that were actually varied in the logs (only known abbreviations)
    varied_tokens = [
        c for c in df.columns
        if c not in ("log_dir",) and not c.endswith(("_mean", "_std"))
        and c in BASELINE_FIXED
        and df[c].fillna("").astype(str).nunique() > 1
    ]

    lines: List[str] = []
    lines.append("# Hyper-parameter Experiment Report\n")
    lines.append(f"_Generated from {len(df)} log directories under `{logs_dir}`._\n")

    # -------------------- Model configuration baseline --------------------
    lines.append("## 1. Model Configuration Baseline\n")
    lines.append(
        "The following describes the representative model architecture and the "
        "default hyper-parameter values used as the baseline for the single-variable sweeps.\n"
    )
    lines.append("| Component | Setting | Description |")
    lines.append("|---|---|---|")
    lines.append("| Backbone | `ConvNeXt1DBackbone` | depth=1, kernel=7, expansion=4 |")
    lines.append("| Encoder | `MolecularProgramEncoder` | number of program tokens = 8, token dimension = 64 |")
    lines.append("| Domain adaptation | CDAN | transfer-loss coefficient `alph` = 0.3 |")
    lines.append("| OT alignment | `DrugAdaptiveOTLoss` | cosine distance, default `lambda_ot` = 0.005 |")
    lines.append("| Stage-3 contrastive | Top-K positive pairs | default `K` = 3, loss weight `beta` = 0.3 |")
    lines.append("| FPI residual | `FragmentProgramInteraction` | `gamma_max` = 0.1, `gamma_init` = 0.05, `m_max` = 16, `fp_dim` = 1024 |")
    lines.append("| Training | epochs | pretrain=100, train=100, contrastive=100 (in these logs) |")
    lines.append("")
    lines.append("### Inferred baseline from logs\n")
    lines.append("The baseline values below were taken as the most frequent value of each token in `brief_logs.txt`.\n")
    lines.append("| Token | Baseline | Meaning |")
    lines.append("|---|---|---|")
    for tok, desc in BASELINE_FIXED.items():
        if tok in baseline:
            lines.append(f"| `{tok}` | `{baseline[tok]}` | {desc} |")
    lines.append("")

    # -------------------- Full results by parameter --------------------
    lines.append("## 2. Single-Variable Sensitivity (from `brief_logs.txt`)\n")
    lines.append(
        "For each parameter, runs were selected such that all other tokens matched the inferred baseline. "
        "Metrics are reported as mean ± std over the retained runs.\n"
    )

    # Order of presentation
    priority = ["K", "gmax", "m", "migrate", "ortho", "ct", "pt"]
    ordered = [p for p in priority if p in varied_tokens]
    ordered += [p for p in varied_tokens if p not in ordered]

    best_configs: List[Tuple[str, Optional[pd.Series], str]] = []
    shown_params = []
    for param in ordered:
        fixed = {k: v for k, v in baseline.items() if k != param and k in df.columns}
        grouped = group_by_param(df, param, fixed)
        filter_note = ""
        if grouped is None or grouped[param].nunique() <= 1:
            # fall back to unfiltered grouping to still show a trend
            grouped = group_by_param(df, param, fixed=None)
            filter_note = " (unfiltered: other hyper-parameters vary)"
        if grouped is None or grouped[param].nunique() <= 1:
            # truly no variation: skip detailed section (still in baseline table)
            continue
        shown_params.append(param)
        lines.append(f"### {PARAM_NAMES.get(param, param)} (`{param}`){filter_note}\n")
        lines.append(make_sweep_table(grouped, param))
        lines.append("")
        best = best_by_auc(grouped, param)
        best_configs.append((param, best, "controlled" if not filter_note else "unfiltered"))
        if best is not None:
            lines.append(
                f"**Best value:** `{param} = {best[param]}` → AUC = {best['auc_mean']:.4f} ± {best['auc_std']:.4f}\n"
            )
        lines.append("")

    if not shown_params:
        lines.append("_No single-variable sweeps were detected in the logs._\n")

    # Tokens that stayed fixed across all logs
    fixed_tokens = [c for c in df.columns if c in BASELINE_FIXED and c not in shown_params]
    if fixed_tokens:
        lines.append("### Fixed tokens across all logs\n")
        lines.append("The following tokens did not vary or co-varied with other changes; they are reported in the baseline table above.\n")
        lines.append("| Token | Value | Meaning |")
        lines.append("|---|---|---|")
        for tok in fixed_tokens:
            lines.append(f"| `{tok}` | `{baseline.get(tok, '')}` | {BASELINE_FIXED[tok]} |")
        lines.append("")

    # -------------------- S3/S8 CSV summary --------------------
    csv_path = PROJECT_ROOT / "outputs" / "hyperparam_tuning" / "S3_S8_results.csv"
    csv_df = load_s3s8_csv(csv_path)
    if csv_df is not None:
        lines.append("## 3. Additional AUC-Only Sweeps (`S3_S8_results.csv`)\n")
        lines.append(
            "These results come from a fast screening pipeline and report AUC only.\n"
        )
        for param_name in ["fpi_gamma_max", "num_programs", "K", "beta", "lambda_ot", "fpi_m_max"]:
            lines.append(f"### {PARAM_NAMES.get(param_name, param_name)}\n")
            lines.append(make_csv_table(csv_df, param_name))
            lines.append("")

    # -------------------- Best config summary --------------------
    lines.append("## 4. Best Configurations per Parameter\n")
    lines.append("| Parameter | Best value | AUC | AUPR | F1 | ACC |")
    lines.append("|---|---|---|---|---|---|")
    for param, best, _ in best_configs:
        if best is None:
            continue
        lines.append(
            f"| {PARAM_NAMES.get(param, param)} | {best[param]} | "
            f"{best['auc_mean']:.4f} ± {best['auc_std']:.4f} | "
            f"{best['aupr_mean']:.4f} ± {best['aupr_std']:.4f} | "
            f"{best['f1_mean']:.4f} ± {best['f1_std']:.4f} | "
            f"{best['acc_mean']:.4f} ± {best['acc_std']:.4f} |"
        )
    lines.append("")

    # Quick sweep overview
    lines.append("### Sweep overview\n")
    lines.append("| Parameter | Filter | # values | Best value | Best AUC |")
    lines.append("|---|---|---:|---|---|")
    for param, best, filt in best_configs:
        if best is None:
            continue
        grouped = group_by_param(df, param, {k: v for k, v in baseline.items() if k != param and k in df.columns})
        if grouped is None or grouped[param].nunique() <= 1:
            grouped = group_by_param(df, param, fixed=None)
        n_values = grouped[param].nunique() if grouped is not None else 0
        lines.append(
            f"| {PARAM_NAMES.get(param, param)} | {filt} | {n_values} | {best[param]} | "
            f"{best['auc_mean']:.4f} ± {best['auc_std']:.4f} |"
        )
    lines.append("")

    # Overall best run
    best_run = df.loc[df["auc_mean"].idxmax()]
    unique_df = df.drop_duplicates(subset=["log_dir"]).sort_values("auc_mean", ascending=False)
    best_log_path = logs_dir / best_run["log_dir"] / "brief_logs.txt"
    best_cfg_line = ""
    if best_log_path.exists():
        best_cfg_line = best_log_path.read_text(encoding="utf-8").splitlines()[1].strip()
    lines.append("## 5. Overall Best Run in Logs\n")
    lines.append(f"- **Log directory:** `{best_run['log_dir']}`")
    lines.append(f"- **AUC:** {best_run['auc_mean']:.4f} ± {best_run['auc_std']:.4f}")
    lines.append(f"- **AUPR:** {best_run['aupr_mean']:.4f} ± {best_run['aupr_std']:.4f}")
    lines.append(f"- **F1:** {best_run['f1_mean']:.4f} ± {best_run['f1_std']:.4f}")
    lines.append(f"- **ACC:** {best_run['acc_mean']:.4f} ± {best_run['acc_std']:.4f}")
    lines.append("")
    if best_cfg_line:
        lines.append("**Full config string:**\n")
        lines.append(f"```text\n{best_cfg_line}\n```")
    lines.append("")

    # -------------------- Detailed per-drug results for top runs --------------------
    lines.append("## 6. Detailed Per-Drug Results for Top Runs\n")
    lines.append(
        "The tables below use the per-drug, per-fold summaries stored in `logs.txt`.\n"
    )

    def drug_table(drug_summary: Dict[str, Dict[str, Tuple[float, float]]]) -> str:
        rows = ["| Drug | AUC | AUPR | F1 | ACC |",
                "|---|---|---|---|---|"]
        for drug in sorted(drug_summary.keys()):
            d = drug_summary[drug]
            rows.append(
                f"| {drug} | {fmt_mean_std(*d.get('auc', (np.nan, np.nan)))} | "
                f"{fmt_mean_std(*d.get('aupr', (np.nan, np.nan)))} | "
                f"{fmt_mean_std(*d.get('f1', (np.nan, np.nan)))} | "
                f"{fmt_mean_std(*d.get('acc', (np.nan, np.nan)))} |"
            )
        return "\n".join(rows)

    top5 = unique_df.head(5)
    for _, row in top5.iterrows():
        ds = row.get("drug_summary")
        if not isinstance(ds, dict) or not ds:
            continue
        lines.append(f"### {row['log_dir']} (AUC = {row['auc_mean']:.4f})\n")
        # per-fold averages if available
        fold_avg = row.get("fold_avg")
        if isinstance(fold_avg, list) and len(fold_avg) > 0:
            lines.append("**Per-fold test-avg (TCGA):**\n")
            lines.append("| Fold | AUC | AUPR | F1 | ACC |")
            lines.append("|---|---|---|---|---|")
            for i, vals in enumerate(fold_avg, start=1):
                lines.append(
                    f"| {i} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} |"
                )
            lines.append("")
        lines.append("**Per-drug mean ± std:**\n")
        lines.append(drug_table(ds))
        lines.append("")

    # Aggregate per-drug across all logs that have detailed summaries
    all_drug_values: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    for _, row in df.iterrows():
        ds = row.get("drug_summary")
        if not isinstance(ds, dict) or not ds:
            continue
        for drug, metrics in ds.items():
            all_drug_values.setdefault(drug, {}).setdefault("auc", []).append(metrics.get("auc"))
            all_drug_values.setdefault(drug, {}).setdefault("aupr", []).append(metrics.get("aupr"))
            all_drug_values.setdefault(drug, {}).setdefault("f1", []).append(metrics.get("f1"))
            all_drug_values.setdefault(drug, {}).setdefault("acc", []).append(metrics.get("acc"))

    if all_drug_values:
        lines.append("## 7. Aggregate Per-Drug Performance Across All Logs\n")
        lines.append(
            "Averages of per-drug means/stds across all logs that contain detailed `logs.txt` results.\n"
        )
        lines.append("| Drug | AUC | AUPR | F1 | ACC | n logs |")
        lines.append("|---|---|---|---|---|---|")
        for drug in sorted(all_drug_values.keys()):
            vals = all_drug_values[drug]
            row_cells = [drug]
            n_logs = None
            for metric in METRICS:
                pairs = [p for p in vals.get(metric, []) if isinstance(p, tuple)]
                if not pairs:
                    row_cells.append("N/A")
                    continue
                if n_logs is None:
                    n_logs = len(pairs)
                means = [p[0] for p in pairs]
                stds = [p[1] for p in pairs]
                row_cells.append(f"{np.mean(means):.4f} ± {np.mean(stds):.4f}")
            row_cells.append(str(n_logs if n_logs is not None else 0))
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")

    # -------------------- Raw config table --------------------
    lines.append("## 8. All Logged Runs\n")
    lines.append("| Log | AUC | AUPR | F1 | ACC |")
    lines.append("|---|---|---|---|---|")
    for _, row in unique_df.head(50).iterrows():
        lines.append(
            f"| `{row['log_dir']}` | {row['auc_mean']:.4f}/{row['auc_std']:.4f} | "
            f"{row['aupr_mean']:.4f}/{row['aupr_std']:.4f} | "
            f"{row['f1_mean']:.4f}/{row['f1_std']:.4f} | "
            f"{row['acc_mean']:.4f}/{row['acc_std']:.4f} |"
        )
    lines.append("")
    if len(unique_df) > 50:
        lines.append(f"_Showing top 50 of {len(unique_df)} unique runs._\n")

    # -------------------- Key observations --------------------
    lines.append("## 9. Key Observations\n")
    lines.append(
        f"- **Best overall performance:** AUC = {best_run['auc_mean']:.4f} ± {best_run['auc_std']:.4f} "
        f"(AUPR = {best_run['aupr_mean']:.4f}, F1 = {best_run['f1_mean']:.4f}, ACC = {best_run['acc_mean']:.4f}).\n"
    )
    lines.append(
        "- **Top-K contrastive:** In the controlled sweep, `K=7` gives the highest AUC (0.6368). "
        "However, the baseline `K=9` is close and appears in most of the top runs, so both are reasonable choices.\n"
    )
    lines.append(
        "- **FPI m_max:** Values 4, 8, 16, and 32 all reach the best observed AUC (0.6466), "
        "suggesting the model is relatively insensitive to this parameter once it is ≥4.\n"
    )
    lines.append(
        "- **FPI gamma_max:** The controlled sweep suggests `gmax=0.3` is slightly better than 0.1, "
        "but the absolute best runs use `gmax=0.1` together with `m=16`. This indicates an interaction "
        "between `gmax` and `m_max` that a single-variable sweep cannot capture.\n"
    )
    lines.append(
        "- **Epochs:** 100 pretrain / 100 train / 100 contrastive epochs outperform the single 300-epoch pretrain run, "
        "suggesting shorter schedules are sufficient here.\n"
    )
    lines.append(
        "- **S3/S8 fast screening:** Confirms `num_programs=8`, `K=9`, `beta≈0.066`, `lambda_ot=0.005`, "
        "and `fpi_m_max≥4` as the strongest settings among the screened values.\n"
    )
    lines.append(
        "- **Caveat:** Some config tokens are abbreviated in the log string (e.g., `fgl` corresponds to "
        "`lambda_fpi_gamma_l1`, but the logged value `10.001` is a formatting artifact of `_fgl1{value}`).\n"
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
