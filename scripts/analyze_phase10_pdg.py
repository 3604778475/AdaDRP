"""
Phase 10 PDG Analysis Script
============================
Reads result CSVs and produces summary tables.

Usage:
    python scripts/analyze_phase10_pdg.py
"""

import os
import glob
import numpy as np
import pandas as pd


def find_latest_results(pattern="results/*_mmpa.csv"):
    """Find result CSVs matching pattern."""
    files = glob.glob(pattern)
    return files


def read_result_csv(path):
    """Read a result CSV and return mean per-drug metrics."""
    df = pd.read_csv(path, index_col=0)
    # df index: ['auc', 'aupr', 'f1', 'acc'], columns: drugs
    return df


def compute_summary(files, baseline_b0_path=None, baseline_gfpi_path=None):
    """Compute summary from result files."""
    drugs = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
             "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"]

    baseline_b0 = None
    baseline_gfpi = None
    if baseline_b0_path and os.path.exists(baseline_b0_path):
        baseline_b0 = read_result_csv(baseline_b0_path)
    if baseline_gfpi_path and os.path.exists(baseline_gfpi_path):
        baseline_gfpi = read_result_csv(baseline_gfpi_path)

    summary_rows = []
    per_drug_rows = []

    for f in files:
        method = os.path.basename(f).replace("_mmpa.csv", "")
        df = read_result_csv(f)

        # Overall means
        auc_mean = df.loc["auc"].mean()
        aupr_mean = df.loc["aupr"].mean()
        f1_mean = df.loc["f1"].mean()
        acc_mean = df.loc["acc"].mean()

        delta_b0 = np.nan
        delta_gfpi = np.nan
        if baseline_b0 is not None:
            delta_b0 = auc_mean - baseline_b0.loc["auc"].mean()
        if baseline_gfpi is not None:
            delta_gfpi = auc_mean - baseline_gfpi.loc["auc"].mean()

        summary_rows.append({
            "method": method,
            "AUC": round(auc_mean, 4),
            "AUPR": round(aupr_mean, 4),
            "F1": round(f1_mean, 4),
            "ACC": round(acc_mean, 4),
            "delta_vs_B0": round(delta_b0, 4) if not np.isnan(delta_b0) else "",
            "delta_vs_GlobalFPI": round(delta_gfpi, 4) if not np.isnan(delta_gfpi) else "",
        })

        for drug in drugs:
            if drug not in df.columns:
                continue
            d_auc = df.loc["auc", drug]
            d_aupr = df.loc["aupr", drug]
            d_f1 = df.loc["f1", drug]
            d_acc = df.loc["acc", drug]

            d_delta_b0 = np.nan
            d_delta_gfpi = np.nan
            if baseline_b0 is not None and drug in baseline_b0.columns:
                d_delta_b0 = d_auc - baseline_b0.loc["auc", drug]
            if baseline_gfpi is not None and drug in baseline_gfpi.columns:
                d_delta_gfpi = d_auc - baseline_gfpi.loc["auc", drug]

            per_drug_rows.append({
                "method": method,
                "drug_name": drug,
                "auc": round(d_auc, 4),
                "aupr": round(d_aupr, 4),
                "f1": round(d_f1, 4),
                "acc": round(d_acc, 4),
                "delta_vs_B0": round(d_delta_b0, 4) if not np.isnan(d_delta_b0) else "",
                "delta_vs_GlobalFPI": round(d_delta_gfpi, 4) if not np.isnan(d_delta_gfpi) else "",
            })

    summary_df = pd.DataFrame(summary_rows)
    per_drug_df = pd.DataFrame(per_drug_rows)
    return summary_df, per_drug_df


def load_pdg_diagnostics(diag_dir="outputs/pseudo_domain"):
    """Load PDG diagnostics CSVs."""
    files = glob.glob(os.path.join(diag_dir, "fold*_K*_diagnostics.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def main():
    import argparse
    parser = argparse.ArgumentParser("Analyze Phase 10 PDG results")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--baseline_b0", type=str, default=None,
                        help="Path to B0 baseline CSV")
    parser.add_argument("--baseline_gfpi", type=str, default=None,
                        help="Path to Global FPI baseline CSV")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.results_dir, "*_mmpa.csv"))
    if not files:
        print(f"No result files found in {args.results_dir}")
        return

    summary_df, per_drug_df = compute_summary(
        files, baseline_b0_path=args.baseline_b0, baseline_gfpi_path=args.baseline_gfpi
    )

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "phase10_pdg_summary.csv")
    per_drug_path = os.path.join(args.output_dir, "phase10_pdg_per_drug.csv")
    summary_df.to_csv(summary_path, index=False)
    per_drug_df.to_csv(per_drug_path, index=False)

    diag_df = load_pdg_diagnostics()
    diag_path = os.path.join(args.output_dir, "phase10_pdg_diagnostics.csv")
    if not diag_df.empty:
        diag_df.to_csv(diag_path, index=False)

    print("=== Phase 10 PDG Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {per_drug_path}")
    if not diag_df.empty:
        print(f"Saved: {diag_path}")


if __name__ == "__main__":
    main()
