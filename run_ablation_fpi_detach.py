"""
Phase 9-FPI-R: Stop-Gradient Diagnosis Ablation Runner
========================================================
Runs FPI detach-mode ablation experiments.

Experiments:
- B0_current:        baseline without FPI
- FPI2_none:         FPI gamma_max=0.1, detach_mode="none"
- FPI2_detach_z:     FPI gamma_max=0.1, detach_mode="z"
- FPI2_detach_zq:    FPI gamma_max=0.1, detach_mode="zq"
- FPI2_detach_zq_drug: FPI gamma_max=0.1, detach_mode="zq_drug"

Outputs:
    outputs/fpi_detach_results.csv
    outputs/fpi_detach_per_drug.csv
    outputs/fpi_detach_per_fold.csv
"""

import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np
import time

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

EXPERIMENTS = [
    {
        "name": "B0_current",
        "use_fpi": False,
        "fpi_gamma_max": 0.0,
        "fpi_gamma_init": 0.0,
        "fpi_q_prior_weight": 0.01,
        "lambda_fpi_gamma_l1": 0.0,
        "lambda_fpi_entropy": 0.0,
        "fpi_detach_mode": "none",
    },
    {
        "name": "FPI2_none",
        "use_fpi": True,
        "fpi_gamma_max": 0.1,
        "fpi_gamma_init": 0.0,
        "fpi_q_prior_weight": 0.01,
        "lambda_fpi_gamma_l1": 0.001,
        "lambda_fpi_entropy": 0.0,
        "fpi_detach_mode": "none",
    },
    {
        "name": "FPI2_detach_z",
        "use_fpi": True,
        "fpi_gamma_max": 0.1,
        "fpi_gamma_init": 0.0,
        "fpi_q_prior_weight": 0.01,
        "lambda_fpi_gamma_l1": 0.001,
        "lambda_fpi_entropy": 0.0,
        "fpi_detach_mode": "z",
    },
    {
        "name": "FPI2_detach_zq",
        "use_fpi": True,
        "fpi_gamma_max": 0.1,
        "fpi_gamma_init": 0.0,
        "fpi_q_prior_weight": 0.01,
        "lambda_fpi_gamma_l1": 0.001,
        "lambda_fpi_entropy": 0.0,
        "fpi_detach_mode": "zq",
    },
    {
        "name": "FPI2_detach_zq_drug",
        "use_fpi": True,
        "fpi_gamma_max": 0.1,
        "fpi_gamma_init": 0.0,
        "fpi_q_prior_weight": 0.01,
        "lambda_fpi_gamma_l1": 0.001,
        "lambda_fpi_entropy": 0.0,
        "fpi_detach_mode": "zq_drug",
    },
]


def make_param_str(exp_config):
    """Reconstruct param_str exactly as train_mmpa_program.py builds it."""
    base = "pretrain_num_epochs_300_train_num_epochs_100_contrastive_num_epochs_200_K_3"
    if not exp_config["use_fpi"]:
        return base
    s = base
    s += f"_fpi_gmax{exp_config['fpi_gamma_max']}_ginit{exp_config['fpi_gamma_init']}_qprior{exp_config['fpi_q_prior_weight']}"
    if exp_config["lambda_fpi_gamma_l1"] > 0:
        s += f"_fgl1{exp_config['lambda_fpi_gamma_l1']}"
    if exp_config["lambda_fpi_entropy"] > 0:
        s += f"_fent{exp_config['lambda_fpi_entropy']}"
    s += f"_detach{exp_config['fpi_detach_mode']}"
    s += "_m16_fp1024"
    return s


# Fixed B0 / C3 Combo hyperparameters
COMMON_ARGS = [
    "--measurement", "Z_SCORE",
    "--thres_gdsc", "0.0",
    "--thres_label", "0.1",
    "--n", "5",
    "--alph", "0.3",
    "--beta", "0.3",
    "--train",
    "--backbone_type", "convnext1d",
    "--convnext_depth", "1",
    "--convnext_kernel_size", "7",
    "--num_programs", "8",
    "--program_dim", "64",
    "--K", "3",
    "--use_ot_alignment",
    "--lambda_ot", "0.005",
    "--use_cdan",
]


def run_experiment(exp_config):
    name = exp_config["name"]
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"{'='*60}")

    cmd = [sys.executable, "train_mmpa_program.py"] + COMMON_ARGS

    if exp_config["use_fpi"]:
        cmd.extend([
            "--use_fpi",
            "--fpi_fragment_npz", "data/drug_fragments_brics_m16_fp1024.npz",
            "--fpi_gamma_max", str(exp_config["fpi_gamma_max"]),
            "--fpi_gamma_init", str(exp_config["fpi_gamma_init"]),
            "--fpi_q_prior_weight", str(exp_config["fpi_q_prior_weight"]),
            "--lambda_fpi_gamma_l1", str(exp_config["lambda_fpi_gamma_l1"]),
            "--lambda_fpi_entropy", str(exp_config["lambda_fpi_entropy"]),
            "--fpi_detach_mode", exp_config["fpi_detach_mode"],
        ])

    log_file = f"outputs/fpi_detach_{name}.log"
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_file}")

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"Finished {name} with return code {result.returncode}")
    return result.returncode == 0


def collect_results():
    """Collect results from results/ directory."""
    drugs = [
        "5-Fluorouracil", "Cisplatin", "Cyclophosphamide",
        "Docetaxel", "Doxorubicin", "Etoposide",
        "Gemcitabine", "Paclitaxel", "Temozolomide",
    ]
    metrics = ['auc', 'aupr', 'f1', 'acc']

    results_rows = []
    per_drug_rows = []
    per_fold_rows = []

    b0_means = None

    for exp in EXPERIMENTS:
        name = exp["name"]
        param_str = make_param_str(exp)
        result_file = os.path.join("results", f"{param_str}_mmpa.csv")
        if not os.path.exists(result_file):
            print(f"Warning: no result file found for {name} at {result_file}")
            continue
        try:
            df = pd.read_csv(result_file, index_col=0)
        except Exception as e:
            print(f"Warning: failed to read {result_file}: {e}")
            continue

        overall_mean = df.mean(axis=1)
        results_rows.append({
            "method": name,
            "AUC": overall_mean['auc'],
            "AUPR": overall_mean['aupr'],
            "F1": overall_mean['f1'],
            "ACC": overall_mean['acc'],
        })

        if name == "B0_current":
            b0_means = {drug: df.loc[:, drug].values for drug in drugs}

        for drug in drugs:
            per_drug_rows.append({
                "method": name,
                "drug_name": drug,
                "auc": df.loc['auc', drug],
                "aupr": df.loc['aupr', drug],
                "f1": df.loc['f1', drug],
                "acc": df.loc['acc', drug],
                "delta_vs_B0": np.nan,
            })

        # Per-fold results
        for fold_idx in range(5):
            fold_file = os.path.join("results", f"fold-{fold_idx}", f"{param_str}_mmpa.csv")
            if os.path.exists(fold_file):
                try:
                    fold_df = pd.read_csv(fold_file, index_col=0)
                    fold_mean = fold_df.mean(axis=1)
                    per_fold_rows.append({
                        "method": name,
                        "fold": fold_idx,
                        "auc": fold_mean['auc'],
                        "aupr": fold_mean['aupr'],
                        "f1": fold_mean['f1'],
                        "acc": fold_mean['acc'],
                    })
                except Exception:
                    pass

    # Compute delta_vs_B0
    if b0_means is not None:
        for row in per_drug_rows:
            if row["method"] != "B0_current" and row["drug_name"] in b0_means:
                b0_auc = b0_means[row["drug_name"]][0]
                row["delta_vs_B0"] = row["auc"] - b0_auc

    pd.DataFrame(results_rows).to_csv("outputs/fpi_detach_results.csv", index=False)
    pd.DataFrame(per_drug_rows).to_csv("outputs/fpi_detach_per_drug.csv", index=False)
    pd.DataFrame(per_fold_rows).to_csv("outputs/fpi_detach_per_fold.csv", index=False)

    print("\nCollected results:")
    print(pd.DataFrame(results_rows).to_string(index=False))


def main():
    # Step 1: Precompute fragments if not exists
    if not os.path.exists("data/drug_fragments_brics_m16_fp1024.npz"):
        print("Precomputing drug fragments...")
        ret = subprocess.run([
            sys.executable, "scripts/precompute_drug_fragments.py",
            "--m_max", "16",
            "--fp_dim", "1024",
            "--radius", "2",
        ])
        if ret.returncode != 0:
            print("Fragment precomputation failed!")
            return

    # Step 2: Run experiments
    for exp in EXPERIMENTS:
        success = run_experiment(exp)
        if not success:
            print(f"Experiment {exp['name']} failed!")
            if exp["name"] == "FPI2_none":
                print("FPI2_none failed! Stopping ablation as baseline may be broken.")
                break

    # Step 3: Collect results
    collect_results()


if __name__ == "__main__":
    main()
