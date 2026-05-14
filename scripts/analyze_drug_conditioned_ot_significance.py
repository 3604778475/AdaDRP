"""
Significance analysis for Drug-Conditioned OT experiments.

Compares DCOT variants against B0 baseline on fold-drug level AUC and AUPR.
Outputs: outputs/drug_conditioned_ot_significance.csv
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results"
OUTPUT_DIR = "outputs"
METRICS = ["auc", "aupr"]


def load_fold_results(method_prefix):
    """
    Load per-fold, per-drug results for a given method prefix.
    Returns dict: metric -> list of fold-drug values.
    """
    pattern = os.path.join(RESULTS_DIR, "fold-*", f"{method_prefix}_mmpa.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"Warning: no files found for prefix {method_prefix}")
        return None

    data = {m: [] for m in METRICS}
    for f in files:
        df = pd.read_csv(f, index_col=0)
        for metric in METRICS:
            vals = df.loc[metric].values.flatten().tolist()
            data[metric].extend(vals)
    return data


def compute_stats(baseline_vals, method_vals):
    """Compute mean delta, std delta, and improved count."""
    baseline_vals = np.array(baseline_vals)
    method_vals = np.array(method_vals)
    min_len = min(len(baseline_vals), len(method_vals))
    baseline_vals = baseline_vals[:min_len]
    method_vals = method_vals[:min_len]
    deltas = method_vals - baseline_vals
    improved = (deltas > 0).sum()
    return {
        "mean_delta": deltas.mean(),
        "std_delta": deltas.std(),
        "improved_count": int(improved),
        "total_count": int(min_len),
        "baseline_mean": baseline_vals.mean(),
        "method_mean": method_vals.mean(),
    }


def try_statistical_tests(baseline_vals, method_vals):
    """Try paired t-test and Wilcoxon signed-rank test if scipy is available."""
    try:
        from scipy import stats
        baseline_vals = np.array(baseline_vals)
        method_vals = np.array(method_vals)
        min_len = min(len(baseline_vals), len(method_vals))
        baseline_vals = baseline_vals[:min_len]
        method_vals = method_vals[:min_len]
        t_stat, t_pval = stats.ttest_rel(method_vals, baseline_vals)
        w_stat, w_pval = stats.wilcoxon(method_vals, baseline_vals)
        return {"t_stat": t_stat, "t_pval": t_pval, "w_stat": w_stat, "w_pval": w_pval}
    except ImportError:
        return {}


def main():
    # Method prefixes to compare
    # B0: baseline combo (adjust prefix to match your actual run)
    # DCOT1..DCOT7: drug-conditioned OT variants
    methods = {
        "B0": None,  # will be set from baseline_prefix arg or default
        "DCOT1": None,
        "DCOT2": None,
        "DCOT3": None,
        "DCOT6": None,
        "DCOT7": None,
    }

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="pretrain_num_epochs_300_train_num_epochs_100_contrastive_num_epochs_200_K_3", help="Baseline method prefix")
    parser.add_argument("--dcot1", default=None, help="DCOT1 prefix")
    parser.add_argument("--dcot2", default=None, help="DCOT2 prefix")
    parser.add_argument("--dcot3", default=None, help="DCOT3 prefix")
    parser.add_argument("--dcot6", default=None, help="DCOT6 prefix")
    parser.add_argument("--dcot7", default=None, help="DCOT7 prefix")
    args = parser.parse_args()

    methods["B0"] = args.baseline
    methods["DCOT1"] = args.dcot1 or f"{args.baseline}_dcot_global_replace_l0.005"
    methods["DCOT2"] = args.dcot2 or f"{args.baseline}_dcot_global_add_l0.001"
    methods["DCOT3"] = args.dcot3 or f"{args.baseline}_dcot_global_add_l0.003"
    methods["DCOT6"] = args.dcot6 or f"{args.baseline}_dcot_discrepancy_add_l0.005"
    methods["DCOT7"] = args.dcot7 or f"{args.baseline}_dcot_discrepancy_add_l0.005"

    baseline_data = load_fold_results(methods["B0"])
    if baseline_data is None:
        print(f"Baseline not found: {methods['B0']}")
        return

    rows = []
    for method_name in ["DCOT1", "DCOT2", "DCOT3", "DCOT6", "DCOT7"]:
        method_data = load_fold_results(methods[method_name])
        if method_data is None:
            continue
        for metric in METRICS:
            stats_dict = compute_stats(baseline_data[metric], method_data[metric])
            test_dict = try_statistical_tests(baseline_data[metric], method_data[metric])
            row = {
                "comparison": f"{method_name}_vs_B0",
                "metric": metric,
                **stats_dict,
                **test_dict,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "drug_conditioned_ot_significance.csv")
    safe_make_dir(OUTPUT_DIR)
    df.to_csv(out_path, index=False)
    print(f"Saved significance analysis to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
