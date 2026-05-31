"""
Phase 12: VCB Results Analysis Script
=====================================
Analyze experimental results for VCB variants and produce summary CSVs.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DRUGS = [
    "5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
    "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"
]


def find_result_files(prefix_pattern):
    """Find result CSVs matching prefix_pattern across folds."""
    files = []
    for fold in range(5):
        pattern = f"results/fold-{fold}/{prefix_pattern}_mmpa.csv"
        matched = glob.glob(pattern)
        files.extend(matched)
    return files


def load_results(prefix_pattern):
    """Load and aggregate 5-fold results."""
    fold_results = []
    for fold in range(5):
        path = f"results/fold-{fold}/{prefix_pattern}_mmpa.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            fold_results.append(df)
    if not fold_results:
        return None
    avg = sum(fold_results) / len(fold_results)
    return avg


def analyze_method(method_name, prefix_pattern, baseline_auc=None):
    avg = load_results(prefix_pattern)
    if avg is None:
        print(f"  No results found for {method_name}")
        return None

    overall = avg.mean(axis=1)
    row = {
        "method": method_name,
        "AUC": overall.get("auc", np.nan),
        "AUPR": overall.get("aupr", np.nan),
        "F1": overall.get("f1", np.nan),
        "ACC": overall.get("acc", np.nan),
    }
    if baseline_auc is not None:
        row["delta_vs_GlobalFPI"] = row["AUC"] - baseline_auc
    else:
        row["delta_vs_GlobalFPI"] = np.nan
    return row, avg


def main():
    os.makedirs("outputs", exist_ok=True)

    # Baseline: Global FPI without VCB
    baseline_result = analyze_method("GlobalFPI_baseline", "*fpi*global*", baseline_auc=None)
    baseline_auc = baseline_result[0]["AUC"] if baseline_result else 0.6031
    print(f"Global FPI baseline AUC: {baseline_auc:.4f}")

    methods = [
        ("VCB-1e-4", "*vcb_vkl0.0001*"),
        ("VCB-5e-4", "*vcb_vkl0.0005*"),
        ("VCB-1e-3", "*vcb_vkl0.001*"),
    ]

    summary_rows = []
    per_drug_rows = []

    for method_name, pattern in methods:
        result = analyze_method(method_name, pattern, baseline_auc=baseline_auc)
        if result is None:
            continue
        row, avg = result
        summary_rows.append(row)
        for drug in DRUGS:
            per_drug_rows.append({
                "method": method_name,
                "drug_name": drug,
                "auc": avg.loc["auc", drug] if "auc" in avg.index and drug in avg.columns else np.nan,
                "aupr": avg.loc["aupr", drug] if "aupr" in avg.index and drug in avg.columns else np.nan,
                "f1": avg.loc["f1", drug] if "f1" in avg.index and drug in avg.columns else np.nan,
                "acc": avg.loc["acc", drug] if "acc" in avg.index and drug in avg.columns else np.nan,
                "delta_vs_GlobalFPI": (avg.loc["auc", drug] - baseline_auc) if "auc" in avg.index and drug in avg.columns else np.nan,
            })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = "outputs/phase12_vcb_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    print(summary_df.to_string(index=False))

    # Save per-drug
    per_drug_df = pd.DataFrame(per_drug_rows)
    per_drug_path = "outputs/phase12_vcb_per_drug.csv"
    per_drug_df.to_csv(per_drug_path, index=False)
    print(f"Saved per-drug to {per_drug_path}")

    # Save diagnostics summary
    diag_files = glob.glob("logs/*/outputs/vcb_diagnostics.csv")
    if diag_files:
        all_diags = []
        for f in diag_files:
            df = pd.read_csv(f)
            # Infer method from parent log dir
            method = os.path.basename(os.path.dirname(os.path.dirname(f)))
            df["method"] = method
            all_diags.append(df)
        diag_df = pd.concat(all_diags, ignore_index=True)
        diag_path = "outputs/phase12_vcb_diagnostics.csv"
        diag_df.to_csv(diag_path, index=False)
        print(f"Saved diagnostics to {diag_path}")
    else:
        print("No VCB diagnostics found.")


if __name__ == "__main__":
    main()
