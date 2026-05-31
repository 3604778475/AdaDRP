#!/usr/bin/env python
"""
Phase 14-A Debug: Rank Loss Sanity Report Generator
====================================================
Analyzes rank loss implementation and training outputs.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd


def check_implementation():
    """Check drug_wise_pairwise_ranking_loss implementation details."""
    from mmpa_drp.losses import drug_wise_pairwise_ranking_loss
    import inspect

    source = inspect.getsource(drug_wise_pairwise_ranking_loss)
    checks = {}

    # Check 1: uses mean, not sum
    checks["uses_mean_not_sum"] = ".mean()" in source and "loss = torch.stack(losses).mean()" in source

    # Check 2: uses softplus on -(diff - margin)
    checks["softplus_minus_diff"] = "F.softplus(-(diff - margin)).mean()" in source

    # Check 3: uses raw logits (not pred/prob)
    checks["function_name_hints_logits"] = "logits" in source

    # Check 4: per-drug loop
    checks["per_drug_loop"] = "for d in range(D):" in source

    # Check 5: skip if no pos or no neg
    checks["skip_invalid_drug"] = "pos.numel() == 0 or neg.numel() == 0" in source

    return checks


def analyze_training_log(log_dir_pattern="logs/*/logs.txt"):
    """Extract rank loss diagnostics from training logs."""
    files = sorted(glob.glob(log_dir_pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    log_file = files[0]

    records = []
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Rank epoch" in line:
                # Parse: "Rank epoch X: response=..., rank_raw=..., rank_weighted=..., ratio=..."
                parts = line.strip().split(": ")
                if len(parts) >= 2:
                    epoch_str = parts[0].replace("Rank epoch ", "").strip()
                    kv_str = parts[1]
                    kv = {}
                    for item in kv_str.split(", "):
                        if "=" in item:
                            k, v = item.split("=")
                            kv[k.strip()] = float(v.strip())
                    kv["epoch"] = int(epoch_str)
                    records.append(kv)
    return pd.DataFrame(records) if records else None


def analyze_label_direction(result_csv_pattern="logs/*/pretrain_num_epochs_*/results.csv"):
    """Check if AUC(logits) vs AUC(-logits) reveals label direction issues."""
    from sklearn.metrics import roc_auc_score

    files = sorted(glob.glob(result_csv_pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None

    # We need predictions and labels per drug. The results.csv only has summary metrics.
    # For a proper check, we need fold-level predictions. This is a simplified check.
    return None


def analyze_rank_debug_csv(csv_path="outputs/rank_debug_per_epoch.csv"):
    """Analyze per-drug rank debug CSV."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    summary = {}
    valid_df = df[df["valid"] == True]
    if valid_df.empty:
        return summary

    summary["total_records"] = len(df)
    summary["valid_records"] = len(valid_df)
    summary["avg_num_pos"] = valid_df["num_pos"].mean()
    summary["avg_num_neg"] = valid_df["num_neg"].mean()
    summary["avg_num_pairs"] = valid_df["num_pairs"].mean()
    summary["avg_rank_loss_d"] = valid_df["rank_loss_d"].mean()
    summary["avg_mean_pos_logit"] = valid_df["mean_pos_logit"].mean()
    summary["avg_mean_neg_logit"] = valid_df["mean_neg_logit"].mean()
    summary["avg_mean_pair_diff"] = valid_df["mean_pair_diff"].mean()

    # Per-drug summary
    per_drug = valid_df.groupby("drug_idx").agg({
        "num_pos": "mean",
        "num_neg": "mean",
        "num_pairs": "mean",
        "rank_loss_d": "mean",
        "mean_pos_logit": "mean",
        "mean_neg_logit": "mean",
        "mean_pair_diff": "mean",
    }).reset_index()

    return summary, per_drug


def generate_report(output_dir="outputs/rank_debug"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Implementation checks
    impl_checks = check_implementation()
    impl_df = pd.DataFrame([{"check_item": k, "status": "PASS" if v else "FAIL"} for k, v in impl_checks.items()])
    impl_df.to_csv(os.path.join(output_dir, "rank_debug_summary.csv"), index=False)
    print("=== Implementation Checks ===")
    for k, v in impl_checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    # 2. Training log scale analysis
    train_df = analyze_training_log()
    if train_df is not None and not train_df.empty:
        train_df.to_csv(os.path.join(output_dir, "rank_debug_loss_scale.csv"), index=False)
        print("\n=== Loss Scale (last epoch) ===")
        last = train_df.iloc[-1]
        print(f"  response={last.get('response', 'N/A'):.4f}")
        print(f"  rank_raw={last.get('rank_raw', 'N/A'):.4f}")
        print(f"  rank_weighted={last.get('rank_weighted', 'N/A'):.4f}")
        print(f"  ratio={last.get('ratio', 'N/A'):.4f}")
        if last.get("ratio", 0) > 1.0:
            print("  WARNING: weighted_rank_loss > source_bce!")
    else:
        print("\n=== Loss Scale: No training log found ===")

    # 3. Per-drug analysis
    per_drug_result = analyze_rank_debug_csv()
    if per_drug_result is not None:
        summary, per_drug = per_drug_result
        per_drug.to_csv(os.path.join(output_dir, "rank_debug_per_drug.csv"), index=False)
        print("\n=== Per-Drug Summary ===")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        print("\n=== Per-Drug: No rank_debug_per_epoch.csv found ===")

    print(f"\nReport saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/rank_debug")
    args = parser.parse_args()
    generate_report(args.output_dir)
