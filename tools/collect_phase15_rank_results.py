#!/usr/bin/env python
"""
Phase 15-B: Collect rank sweep results and summarize.

Scans the results/ directory for CSV files matching the Phase 15-B param pattern,
and optionally reads rank_debug_per_epoch.csv from the experiment output dirs.

Usage:
    python tools/collect_phase15_rank_results.py \
        --results_dir results \
        --output_dir outputs/phase15 \
        --output_csv outputs/phase15/rank_sweep_summary.csv
"""

import os
import sys
import glob
import re
import argparse
import pandas as pd
import numpy as np

DRUG_NAMES = [
    "5-Fluorouracil",
    "Cisplatin",
    "Cyclophosphamide",
    "Docetaxel",
    "Doxorubicin",
    "Etoposide",
    "Gemcitabine",
    "Paclitaxel",
    "Temozolomide",
]


def safe_col_name(drug_name: str) -> str:
    """Convert drug name to safe CSV column suffix."""
    return drug_name.lower().replace("-", "").replace(" ", "_")


def extract_lambda_from_param_str(param_str: str):
    """Extract lambda_rank from filename like ..._rank_lrank0.05_..."""
    m = re.search(r"_rank_lrank([0-9.]+)_", param_str)
    if m:
        return float(m.group(1))
    return None


def find_baseline_file(results_dir: str):
    """Find baseline result (no _rank_lrank in filename) with same other params."""
    # Heuristic: find the most recent *_mmpa.csv without _rank_ that is not a sweep file
    files = glob.glob(os.path.join(results_dir, "*_mmpa.csv"))
    candidates = [f for f in files if "_rank_lrank" not in os.path.basename(f)]
    if not candidates:
        return None
    # Sort by mtime, take most recent
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def parse_results_csv(csv_path: str):
    """Read a result CSV and return DataFrame with index=['auc','aupr','f1','acc']."""
    df = pd.read_csv(csv_path, index_col=0)
    return df


def read_rank_diagnostics(output_dir: str, lambda_rank: float):
    """Try to read rank_debug_per_epoch.csv from the experiment output dir."""
    # Map lambda to directory name used in sweep
    lam_str = str(lambda_rank).replace(".", "")
    # Handle 0 -> lam0
    if lambda_rank == 0.0:
        lam_str = "0"
    rank_dir = os.path.join(output_dir, f"rank_lam{lam_str}")
    rank_csv = os.path.join(rank_dir, "rank_debug_per_epoch.csv")

    out = {
        "final_bce_loss": np.nan,
        "final_rank_loss": np.nan,
        "final_pair_acc_mean": np.nan,
        "final_weighted_rank_grad_ratio": np.nan,
    }

    if not os.path.exists(rank_csv):
        return out

    try:
        rank_df = pd.read_csv(rank_csv)
        if rank_df.empty:
            return out

        last_epoch = rank_df["epoch"].max()
        last_rows = rank_df[rank_df["epoch"] == last_epoch]

        if "bce_d" in last_rows.columns:
            out["final_bce_loss"] = last_rows["bce_d"].mean()
        if "rank_loss_d" in last_rows.columns:
            out["final_rank_loss"] = last_rows["rank_loss_d"].mean()
        if "pair_acc_d" in last_rows.columns:
            vals = last_rows["pair_acc_d"].dropna()
            if not vals.empty:
                out["final_pair_acc_mean"] = vals.mean()

        if "rank_grad_weighted_ratio" in rank_df.columns:
            valid = rank_df["rank_grad_weighted_ratio"].dropna()
            if not valid.empty:
                out["final_weighted_rank_grad_ratio"] = valid.iloc[-1]
    except Exception as e:
        print(f"  Warning: could not read {rank_csv}: {e}")

    return out


def build_summary(results_dir: str, output_dir: str):
    # Scan result files with _rank_lrank pattern
    result_files = glob.glob(os.path.join(results_dir, "*_mmpa.csv"))
    sweep_files = {}
    for f in result_files:
        basename = os.path.basename(f)
        lam = extract_lambda_from_param_str(basename)
        if lam is None:
            continue
        # Prefer most recent if duplicates
        if lam not in sweep_files or os.path.getmtime(f) > os.path.getmtime(sweep_files[lam]):
            sweep_files[lam] = f

    if not sweep_files:
        print("No sweep result files found in", results_dir)
        return pd.DataFrame()

    # Baseline for comparison
    baseline_path = find_baseline_file(results_dir)
    baseline_auc = None
    baseline_per_drug = {}
    if baseline_path:
        try:
            baseline_df = parse_results_csv(baseline_path)
            if "auc" in baseline_df.index:
                baseline_auc = baseline_df.loc["auc"]
                baseline_per_drug = {d: baseline_auc.get(d, np.nan) for d in DRUG_NAMES}
        except Exception:
            pass

    records = []
    for lam in sorted(sweep_files.keys()):
        f = sweep_files[lam]
        basename = os.path.basename(f)
        experiment_name = f"rank_lam{str(lam).replace('.', '')}"

        try:
            df = parse_results_csv(f)
        except Exception as e:
            print(f"  Warning: could not parse {f}: {e}")
            continue

        auc_row = df.loc["auc"] if "auc" in df.index else pd.Series(dtype=float)
        mean_auc = auc_row.mean() if len(auc_row) > 0 else np.nan
        per_drug_aucs = {d: auc_row.get(d, np.nan) for d in DRUG_NAMES}
        worst_drug_auc = auc_row.min() if len(auc_row) > 0 else np.nan
        auc_std = auc_row.std() if len(auc_row) > 0 else np.nan

        # Compare to baseline
        num_improved = 0
        num_collapsed = 0
        if baseline_auc is not None:
            for d in DRUG_NAMES:
                sweep_val = per_drug_aucs.get(d, np.nan)
                base_val = baseline_per_drug.get(d, np.nan)
                if np.isnan(sweep_val) or np.isnan(base_val):
                    continue
                if sweep_val > base_val + 0.005:
                    num_improved += 1
                elif sweep_val < base_val - 0.005:
                    num_collapsed += 1

        # Read rank diagnostics
        rank_diag = read_rank_diagnostics(output_dir, lam)

        record = {
            "experiment": experiment_name,
            "lambda_rank": lam,
            "warmup_epochs": 20 if lam > 0 else 0,
            "best_epoch": np.nan,  # Not easily available from CSV; log parsers can add later
            "mean_auc": mean_auc,
            **{f"auc_{safe_col_name(d)}": per_drug_aucs.get(d, np.nan) for d in DRUG_NAMES},
            "worst_drug_auc": worst_drug_auc,
            "auc_std": auc_std,
            "num_improved_drugs_vs_baseline": num_improved,
            "num_collapsed_drugs_vs_baseline": num_collapsed,
            "final_bce_loss": rank_diag["final_bce_loss"],
            "final_rank_loss": rank_diag["final_rank_loss"],
            "final_pair_acc_mean": rank_diag["final_pair_acc_mean"],
            "final_weighted_rank_grad_ratio": rank_diag["final_weighted_rank_grad_ratio"],
            "negative_gradient_pair_ratio_if_available": np.nan,
        }
        records.append(record)

    summary_df = pd.DataFrame(records)

    # Simple auto-judgement
    if not summary_df.empty and baseline_auc is not None:
        best_idx = summary_df["mean_auc"].idxmax()
        if not pd.isna(best_idx):
            best_row = summary_df.loc[best_idx]
            print("\n=== Phase 15-B Auto-Judgement ===")
            print(f"Baseline mean AUC: {baseline_auc.mean():.4f}")
            # Check control run
            control_rows = summary_df[summary_df["lambda_rank"] == 0.0]
            if not control_rows.empty:
                control_auc = control_rows.iloc[0]["mean_auc"]
                if abs(control_auc - baseline_auc.mean()) > 0.01:
                    print("WARNING: lambda=0 control run differs from baseline by >0.01.")
                    print("  => Rank loss code path may be changing training. Check code path.")
                else:
                    print("OK: lambda=0 control run is consistent with baseline.")

            # Check gradient ratio
            for _, row in summary_df.iterrows():
                ratio = row["final_weighted_rank_grad_ratio"]
                if not pd.isna(ratio) and ratio < 0.01 and row["lambda_rank"] > 0:
                    print(f"WARNING: lambda={row['lambda_rank']:.2f} has weighted_rank_grad_ratio < 1%.")
                    print("  => lambda_rank may still be too small.")

            print(f"Best mean AUC: {best_row['mean_auc']:.4f} at lambda_rank={best_row['lambda_rank']:.2f}")
            print(f"Worst-drug AUC at best lambda: {best_row['worst_drug_auc']:.4f}")
            print(f"Collapsed drugs at best lambda: {int(best_row['num_collapsed_drugs_vs_baseline'])}")
            print(f"Improved drugs at best lambda: {int(best_row['num_improved_drugs_vs_baseline'])}")
            print("==================================\n")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Collect Phase 15-B rank sweep results")
    parser.add_argument("--results_dir", default="results", help="Directory containing result CSVs")
    parser.add_argument("--output_dir", default="outputs/phase15", help="Directory containing experiment output subdirs")
    parser.add_argument("--output_csv", default="outputs/phase15/rank_sweep_summary.csv", help="Output summary CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    summary_df = build_summary(args.results_dir, args.output_dir)
    if not summary_df.empty:
        summary_df.to_csv(args.output_csv, index=False)
        print(f"Summary saved to {args.output_csv}")
    else:
        print("No data to save.")


if __name__ == "__main__":
    main()
