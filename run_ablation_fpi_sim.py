"""
Phase 9-FPI-SIM: Signal-style Selective Interaction Ablation Runner
=====================================================================
Runs FPI-SIM ablation experiments.

Experiments:
- B0_current:        baseline without FPI
- FPI2_global:       FPI with global softmax (current baseline)
- FPI_SIM_soft:      FPI-SIM with union_alpha=1.0, union_beta=1.0
- FPI_SIM_cross_only: FPI-SIM with union_alpha=0.0, union_beta=0.0

Outputs:
    outputs/fpi_sim_results.csv
    outputs/fpi_sim_per_drug.csv
    outputs/fpi_sim_per_fold.csv
"""

import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np

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
        "use_fpi_sim": False,
        "fpi_sim_union_alpha": 0.0,
        "fpi_sim_union_beta": 0.0,
        "fpi_sim_q_prior_weight": 0.01,
    },
    {
        "name": "FPI2_global",
        "use_fpi": True,
        "fpi_gamma_max": 0.1,
        "fpi_gamma_init": 0.0,
        "fpi_q_prior_weight": 0.01,
        "lambda_fpi_gamma_l1": 0.001,
        "lambda_fpi_entropy": 0.0,
        "use_fpi_sim": False,
        "fpi_sim_union_alpha": 0.0,
        "fpi_sim_union_beta": 0.0,
        "fpi_sim_q_prior_weight": 0.01,
    },
    {
        "name": "FPI_SIM_soft",
        "use_fpi": True,
        "fpi_gamma_max": 0.1,
        "fpi_gamma_init": 0.0,
        "fpi_q_prior_weight": 0.01,
        "lambda_fpi_gamma_l1": 0.001,
        "lambda_fpi_entropy": 0.0,
        "use_fpi_sim": True,
        "fpi_sim_union_alpha": 1.0,
        "fpi_sim_union_beta": 1.0,
        "fpi_sim_q_prior_weight": 0.01,
    },
    {
        "name": "FPI_SIM_cross_only",
        "use_fpi": True,
        "fpi_gamma_max": 0.1,
        "fpi_gamma_init": 0.0,
        "fpi_q_prior_weight": 0.01,
        "lambda_fpi_gamma_l1": 0.001,
        "lambda_fpi_entropy": 0.0,
        "use_fpi_sim": True,
        "fpi_sim_union_alpha": 0.0,
        "fpi_sim_union_beta": 0.0,
        "fpi_sim_q_prior_weight": 0.01,
    },
]


def make_param_str(exp_config):
    base = "pretrain_num_epochs_300_train_num_epochs_100_contrastive_num_epochs_200_K_3"
    if not exp_config["use_fpi"]:
        return base
    s = base
    s += f"_fpi_gmax{exp_config['fpi_gamma_max']}_ginit{exp_config['fpi_gamma_init']}_qprior{exp_config['fpi_q_prior_weight']}"
    if exp_config["lambda_fpi_gamma_l1"] > 0:
        s += f"_fgl1{exp_config['lambda_fpi_gamma_l1']}"
    if exp_config["lambda_fpi_entropy"] > 0:
        s += f"_fent{exp_config['lambda_fpi_entropy']}"
    if exp_config["use_fpi_sim"]:
        s += "_sim"
        s += f"_ua{exp_config['fpi_sim_union_alpha']}_ub{exp_config['fpi_sim_union_beta']}"
        s += "_pt1.0_ft1.0_ct1.0"
    else:
        s += "_global"
    s += "_m16_fp1024"
    return s


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
        ])
        if exp_config["use_fpi_sim"]:
            cmd.extend([
                "--use_fpi_sim",
                "--fpi_sim_q_prior_weight", str(exp_config["fpi_sim_q_prior_weight"]),
                "--fpi_sim_union_alpha", str(exp_config["fpi_sim_union_alpha"]),
                "--fpi_sim_union_beta", str(exp_config["fpi_sim_union_beta"]),
            ])

    log_file = f"outputs/fpi_sim_{name}.log"
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_file}")

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"Finished {name} with return code {result.returncode}")
    return result.returncode == 0


def collect_results():
    drugs = [
        "5-Fluorouracil", "Cisplatin", "Cyclophosphamide",
        "Docetaxel", "Doxorubicin", "Etoposide",
        "Gemcitabine", "Paclitaxel", "Temozolomide",
    ]

    results_rows = []
    per_drug_rows = []
    per_fold_rows = []

    b0_means = None
    fpi2_means = None

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
            "use_fpi_sim": exp["use_fpi_sim"],
            "AUC": overall_mean['auc'],
            "AUPR": overall_mean['aupr'],
            "F1": overall_mean['f1'],
            "ACC": overall_mean['acc'],
        })

        if name == "B0_current":
            b0_means = {drug: df.loc['auc', drug] for drug in drugs}
        if name == "FPI2_global":
            fpi2_means = {drug: df.loc['auc', drug] for drug in drugs}

        for drug in drugs:
            per_drug_rows.append({
                "method": name,
                "use_fpi_sim": exp["use_fpi_sim"],
                "drug_name": drug,
                "auc": df.loc['auc', drug],
                "aupr": df.loc['aupr', drug],
                "f1": df.loc['f1', drug],
                "acc": df.loc['acc', drug],
                "delta_vs_B0": np.nan,
                "delta_vs_FPI2_global": np.nan,
            })

        for fold_idx in range(5):
            fold_file = os.path.join("results", f"fold-{fold_idx}", f"{param_str}_mmpa.csv")
            if os.path.exists(fold_file):
                try:
                    fold_df = pd.read_csv(fold_file, index_col=0)
                    fold_mean = fold_df.mean(axis=1)
                    per_fold_rows.append({
                        "method": name,
                        "use_fpi_sim": exp["use_fpi_sim"],
                        "fold": fold_idx,
                        "auc": fold_mean['auc'],
                        "aupr": fold_mean['aupr'],
                        "f1": fold_mean['f1'],
                        "acc": fold_mean['acc'],
                    })
                except Exception:
                    pass

    if b0_means is not None:
        for row in per_drug_rows:
            if row["method"] != "B0_current" and row["drug_name"] in b0_means:
                row["delta_vs_B0"] = row["auc"] - b0_means[row["drug_name"]]

    if fpi2_means is not None:
        for row in per_drug_rows:
            if row["method"] not in ("B0_current", "FPI2_global") and row["drug_name"] in fpi2_means:
                row["delta_vs_FPI2_global"] = row["auc"] - fpi2_means[row["drug_name"]]

    pd.DataFrame(results_rows).to_csv("outputs/fpi_sim_results.csv", index=False)
    pd.DataFrame(per_drug_rows).to_csv("outputs/fpi_sim_per_drug.csv", index=False)
    pd.DataFrame(per_fold_rows).to_csv("outputs/fpi_sim_per_fold.csv", index=False)

    print("\nCollected results:")
    print(pd.DataFrame(results_rows).to_string(index=False))


def main():
    if not os.path.exists("data/drug_fragments_brics_m16_fp1024.npz"):
        print("Precomputing drug fragments...")
        ret = subprocess.run([
            sys.executable, "scripts/precompute_drug_fragments.py",
            "--m_max", "16", "--fp_dim", "1024", "--radius", "2",
        ])
        if ret.returncode != 0:
            print("Fragment precomputation failed!")
            return

    for exp in EXPERIMENTS:
        success = run_experiment(exp)
        if not success:
            print(f"Experiment {exp['name']} failed!")
            if exp["name"] == "FPI2_global":
                print("FPI2_global failed! Stopping ablation.")
                break

    collect_results()


if __name__ == "__main__":
    main()
