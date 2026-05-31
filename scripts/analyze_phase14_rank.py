#!/usr/bin/env python
"""
Phase 14-A: Drug-wise Pairwise Ranking Loss Analysis Script
============================================================
Reads result CSVs and outputs summary / per-drug / diagnostics.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np


def find_results_csv(log_dir_pattern="logs/*/pretrain_num_epochs_*_train_num_epochs_*_contrastive_num_epochs_*_K_*/results.csv"):
    """Find all result CSV files."""
    files = glob.glob(log_dir_pattern)
    return files


def parse_method_from_path(path):
    """Extract method name from path."""
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if "pretrain_num_epochs" in part:
            return part
    return "unknown"


def load_global_fpi_baseline():
    """Load Global FPI baseline results if available."""
    baseline_files = glob.glob("logs/*/pretrain_num_epochs_*_train_num_epochs_*_contrastive_num_epochs_*_K_*_fpi*/results.csv")
    baseline_files = [f for f in baseline_files if "rank" not in f.lower() and "vcb" not in f.lower() and "s3r" not in f.lower()]
    if not baseline_files:
        return None
    # Use the most recent one
    baseline_files.sort(key=os.path.getmtime, reverse=True)
    try:
        df = pd.read_csv(baseline_files[0])
        return df
    except Exception:
        return None


def analyze():
    os.makedirs("outputs", exist_ok=True)

    # Find all rank-related results
    rank_files = glob.glob("logs/*/pretrain_num_epochs_*_train_num_epochs_*_contrastive_num_epochs_*_K_*_rank*/results.csv")

    baseline_df = load_global_fpi_baseline()
    baseline_auc = {}
    if baseline_df is not None:
        for _, row in baseline_df.iterrows():
            drug = row.get("drug", row.get("drug_name", ""))
            baseline_auc[drug] = row.get("auc", np.nan)

    summary_records = []
    per_drug_records = []
    diag_records = []

    for fpath in rank_files:
        method = parse_method_from_path(fpath)
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Skip {fpath}: {e}")
            continue

        # Summary metrics
        mean_auc = df["auc"].mean()
        mean_aupr = df["aupr"].mean() if "aupr" in df.columns else np.nan
        mean_f1 = df["f1"].mean() if "f1" in df.columns else np.nan
        mean_acc = df["acc"].mean() if "acc" in df.columns else np.nan

        baseline_mean = np.nanmean([baseline_auc.get(d, np.nan) for d in df["drug"].values])
        delta = mean_auc - baseline_mean if not np.isnan(baseline_mean) else np.nan

        summary_records.append({
            "method": method,
            "AUC": mean_auc,
            "AUPR": mean_aupr,
            "F1": mean_f1,
            "ACC": mean_acc,
            "delta_vs_GlobalFPI": delta,
        })

        # Per-drug metrics
        for _, row in df.iterrows():
            drug = row.get("drug", row.get("drug_name", ""))
            base_auc = baseline_auc.get(drug, np.nan)
            per_drug_records.append({
                "method": method,
                "drug_name": drug,
                "auc": row.get("auc", np.nan),
                "aupr": row.get("aupr", np.nan),
                "f1": row.get("f1", np.nan),
                "acc": row.get("acc", np.nan),
                "delta_vs_GlobalFPI": row.get("auc", np.nan) - base_auc,
            })

    # Save outputs
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv("outputs/phase14_rank_summary.csv", index=False)
        print("Saved outputs/phase14_rank_summary.csv")
        print(summary_df.to_string(index=False))
    else:
        print("No rank results found.")

    if per_drug_records:
        per_drug_df = pd.DataFrame(per_drug_records)
        per_drug_df.to_csv("outputs/phase14_rank_per_drug.csv", index=False)
        print("Saved outputs/phase14_rank_per_drug.csv")

        # Extra statistics
        for method in per_drug_df["method"].unique():
            sub = per_drug_df[per_drug_df["method"] == method]
            improved = (sub["delta_vs_GlobalFPI"] > 0).sum()
            degraded = (sub["delta_vs_GlobalFPI"] < 0).sum()
            severe = (sub["delta_vs_GlobalFPI"] < -0.02).sum()
            print(f"\n{method}:")
            print(f"  improved: {improved}, degraded: {degraded}, severe_drop: {severe}")
            print(f"  per-drug delta AUC (sorted):")
            sorted_d = sub.sort_values("delta_vs_GlobalFPI", ascending=False)
            for _, r in sorted_d.iterrows():
                print(f"    {r['drug_name']}: {r['delta_vs_GlobalFPI']:+.4f}")


if __name__ == "__main__":
    analyze()
