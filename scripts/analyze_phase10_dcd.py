"""
Analyze Phase 10-DCD-R experimental results.

Reads result CSVs from results/ and produces summary tables.
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRUGS = [
    "5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
    "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"
]

METRICS = ["auc", "aupr", "f1", "acc"]


def find_result_csv(root="results"):
    """Find all *_mmpa.csv result files."""
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith("_mmpa.csv"):
                files.append(os.path.join(dirpath, f))
    return files


def parse_method_name(path):
    """Extract method descriptor from filename."""
    basename = os.path.basename(path).replace("_mmpa.csv", "")
    # Simplify known patterns
    if "dcdR" in basename:
        if "fpi" in basename.lower() or "_fpi_" in basename:
            if "dortho" in basename:
                return "DCD2_ortho+FPI"
            return "DCD1_plain+FPI"
        else:
            if "dortho" in basename:
                return "DCD2_ortho"
            return "DCD1_plain"
    elif "fpi" in basename.lower():
        return "Global_FPI"
    else:
        return "B0_current"


def load_results(files):
    rows = []
    for f in files:
        method = parse_method_name(f)
        try:
            df = pd.read_csv(f, index_col=0)
        except Exception as e:
            print(f"Skip {f}: {e}")
            continue
        for drug in DRUGS:
            for metric in METRICS:
                col = f"{drug}_{metric}"
                if col in df.columns:
                    vals = df[col].dropna().values
                    if len(vals) > 0:
                        rows.append({
                            "method": method,
                            "drug": drug,
                            "metric": metric,
                            "value": vals.mean(),
                            "std": vals.std(),
                            "folds": len(vals),
                        })
    return pd.DataFrame(rows)


def compute_summary(df):
    summary = []
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        row = {"method": method}
        for metric in METRICS:
            msub = sub[sub["metric"] == metric]
            row[metric.upper()] = msub["value"].mean()
        summary.append(row)
    return pd.DataFrame(summary)


def compute_per_drug(df):
    per_drug = []
    for method in df["method"].unique():
        for drug in DRUGS:
            sub = df[(df["method"] == method) & (df["drug"] == drug)]
            row = {"method": method, "drug_name": drug}
            for metric in METRICS:
                msub = sub[sub["metric"] == metric]
                if len(msub) > 0:
                    row[metric] = msub["value"].mean()
                else:
                    row[metric] = np.nan
            per_drug.append(row)
    return pd.DataFrame(per_drug)


def add_deltas(df, baseline="B0_current", fpi_ref="Global_FPI"):
    """Add delta columns versus baseline and FPI reference."""
    base_aucs = {}
    fpi_aucs = {}
    for drug in DRUGS:
        b = df[(df["method"] == baseline) & (df["drug_name"] == drug)]
        f = df[(df["method"] == fpi_ref) & (df["drug_name"] == drug)]
        if "auc" in b.columns and len(b) > 0:
            base_aucs[drug] = b["auc"].mean()
        if "auc" in f.columns and len(f) > 0:
            fpi_aucs[drug] = f["auc"].mean()

    df["delta_vs_B0"] = df.apply(
        lambda r: r["auc"] - base_aucs.get(r["drug_name"], np.nan), axis=1
    )
    df["delta_vs_GlobalFPI"] = df.apply(
        lambda r: r["auc"] - fpi_aucs.get(r["drug_name"], np.nan), axis=1
    )
    return df


def main():
    files = find_result_csv()
    if not files:
        print("No result CSVs found in results/")
        return

    print(f"Found {len(files)} result files")
    df = load_results(files)
    if df.empty:
        print("No data loaded")
        return

    summary = compute_summary(df)
    summary_path = os.path.join(OUTPUT_DIR, "phase10_dcd_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    print(summary.to_string(index=False))

    per_drug = compute_per_drug(df)
    per_drug = add_deltas(per_drug)
    per_drug_path = os.path.join(OUTPUT_DIR, "phase10_dcd_per_drug.csv")
    per_drug.to_csv(per_drug_path, index=False)
    print(f"\nSaved per-drug to {per_drug_path}")

    # Print key drugs
    key_drugs = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Etoposide", "Gemcitabine", "Temozolomide"]
    for method in per_drug["method"].unique():
        print(f"\n=== {method} ===")
        sub = per_drug[(per_drug["method"] == method) & (per_drug["drug_name"].isin(key_drugs))]
        print(sub[["drug_name", "auc", "aupr", "delta_vs_B0", "delta_vs_GlobalFPI"]].to_string(index=False))


if __name__ == "__main__":
    main()
