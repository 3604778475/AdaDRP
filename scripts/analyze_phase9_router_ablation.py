#!/usr/bin/env python3
"""
Phase 9-FPI-Router Ablation Result Analyzer

Scans results/ directory for batch1 experiment outputs and produces
a comparison table of mean AUC and per-drug AUC.

Usage:
    python scripts/analyze_phase9_router_ablation.py
"""

import os
import glob
import pandas as pd
import numpy as np

RESULTS_DIR = "results"
OUTPUT_CSV = "outputs/phase9_router_ablation_summary.csv"

# Map experiment config substrings to human-readable names
EXPERIMENT_PATTERNS = {
    "B0 (no FPI, no router)": "_K_3_mmpa.csv",
    "FPI Global Warmup": "_fpi_gmax0.1_ginit0.05_qprior0.01_fgl10.0_detachnone_global_m16_fp1024_mmpa.csv",
    "Router Global-only Confirm": "_router_ggmax0.1_gginit0.05_gcmax0.0_gcinit0.0_rgs1.0_rcs1.0_m16_fp1024_mmpa.csv",
    "Router Oracle Temo g005": "_router_ggmax0.05_gginit0.05_gcmax0.05_gcinit0.0_rgs1.0_rcs1.0_m16_fp1024_mmpa.csv",
    "Router Oracle Temo g0025": "_router_ggmax0.05_gginit0.05_gcmax0.025_gcinit0.0_rgs1.0_rcs1.0_m16_fp1024_mmpa.csv",
    "Router Learnable L1=0.005": "_router_ggmax0.05_gginit0.05_gcmax0.05_gcinit0.0_rggl10.0_rgcl10.005_rgs1.0_rcs1.0_m16_fp1024_mmpa.csv",
    "Router Learnable L1=0.01": "_router_ggmax0.05_gginit0.05_gcmax0.05_gcinit0.0_rggl10.0_rgcl10.01_rgs1.0_rcs1.0_m16_fp1024_mmpa.csv",
    "Router Learnable g0025 L1=0.005": "_router_ggmax0.05_gginit0.05_gcmax0.025_gcinit0.0_rggl10.0_rgcl10.005_rgs1.0_rcs1.0_m16_fp1024_mmpa.csv",
    "Router Learnable Delayed": "_router_ggmax0.05_gginit0.05_gcmax0.05_gcinit0.0_ramp30+20_rggl10.0_rgcl10.005_rgs1.0_rcs1.0_m16_fp1024_mmpa.csv",
}

DRUGS = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
         "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"]


def find_result_file(pattern_suffix):
    """Find the most recent result file matching the pattern."""
    candidates = glob.glob(os.path.join(RESULTS_DIR, f"*{pattern_suffix}"))
    if not candidates:
        return None
    # Sort by modification time, take the most recent
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_result(path):
    """Load a result CSV and return mean AUC and per-drug AUC dict."""
    df = pd.read_csv(path, index_col=0)
    mean_auc = df.loc["auc"].mean()
    per_drug = {drug: df.loc["auc", drug] for drug in DRUGS if drug in df.columns}
    return mean_auc, per_drug, df


def main():
    rows = []
    print("=" * 80)
    print("Phase 9-FPI-Router Ablation Results")
    print("=" * 80)

    for exp_name, pattern_suffix in EXPERIMENT_PATTERNS.items():
        path = find_result_file(pattern_suffix)
        if path is None:
            print(f"\n{exp_name}: NOT FOUND (waiting for results...)")
            continue

        mean_auc, per_drug, df = load_result(path)
        temo_auc = per_drug.get("Temozolomide", np.nan)

        print(f"\n{exp_name}")
        print(f"  File: {os.path.basename(path)}")
        print(f"  Mean AUC: {mean_auc:.4f}")
        print(f"  Temozolomide AUC: {temo_auc:.4f}")

        row = {"experiment": exp_name, "mean_auc": mean_auc, "Temozolomide": temo_auc}
        row.update(per_drug)
        rows.append(row)

    if rows:
        summary_df = pd.DataFrame(rows)
        os.makedirs("outputs", exist_ok=True)
        summary_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n{'=' * 80}")
        print("Summary saved to:", OUTPUT_CSV)
        print(summary_df.to_string(index=False))
    else:
        print("\nNo results found yet. Run experiments and re-run this script.")


if __name__ == "__main__":
    main()
