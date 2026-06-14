"""
scripts/collect_k_sweep_results.py
==================================
Collect results from a K-sweep experiment.

Usage:
    python scripts/collect_k_sweep_results.py --sweep_dir outputs/k_sweep --output_csv outputs/k_sweep_summary.csv
"""

import argparse
import os
import re
import pandas as pd


def parse_log_for_metrics(log_path):
    """Parse final metrics from a training log file."""
    if not os.path.exists(log_path):
        return {}

    metrics = {}
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        # Match: auc：0.6351/0.0755  aupr：0.7599/0.1532  acc：0.7042/0.2315  f1：0.6885/0.1024
        if line.strip().startswith("auc："):
            match = re.search(
                r"auc：(?P<auc>\d+\.\d+)/(?P<auc_std>\d+\.\d+)\s+"
                r"aupr：(?P<aupr>\d+\.\d+)/(?P<aupr_std>\d+\.\d+)\s+"
                r"acc：(?P<acc>\d+\.\d+)/(?P<acc_std>\d+\.\d+)\s+"
                r"f1：(?P<f1>\d+\.\d+)/(?P<f1_std>\d+\.\d+)",
                line,
            )
            if match:
                metrics.update(match.groupdict())
                metrics = {k: float(v) for k, v in metrics.items()}
                break

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Collect K-sweep results")
    parser.add_argument("--sweep_dir", type=str, default="outputs/k_sweep",
                        help="Directory containing k*/ subdirectories")
    parser.add_argument("--output_csv", type=str, default="outputs/k_sweep_summary.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    rows = []
    for subdir in sorted(os.listdir(args.sweep_dir)):
        match = re.match(r"k(\d+)", subdir)
        if not match:
            continue
        k = int(match.group(1))
        log_path = os.path.join(args.sweep_dir, subdir, "logs.txt")
        metrics = parse_log_for_metrics(log_path)
        if metrics:
            rows.append({"K": k, **metrics})
        else:
            print(f"[WARNING] No metrics found for K={k} ({log_path})")

    if not rows:
        print("[WARNING] No results found.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("K")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSummary saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
