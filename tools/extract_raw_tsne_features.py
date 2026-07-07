#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract RAW input features (gene expression + drug fingerprint) for t-SNE.

No model is loaded; the features are the direct inputs fed to the model:
  - gene expression vector per sample
  - RDKit fingerprint (128-d) per drug
  - concatenation of the two for every sample-drug pair

Output has the same NPZ structure as tools/extract_fmpa_tsne_features.py,
so the same plotting scripts can be reused.

Example
-------
    D:/anaconda/envs/macb_drp/python.exe tools/extract_raw_tsne_features.py \
        --fold 1 --output-dir artifacts/tsne_raw/fold_1 --seed 2025
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config
from utility import set_seed
from dataload import get_drug_feat, get_multi_labeled_dataloader, get_unlabeled_dataloaders

DEFAULT_DRUGS = [
    "5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
    "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--measurement", type=str, default="Z_SCORE")
    parser.add_argument("--thres-gdsc", type=float, default=0.0)
    parser.add_argument("--thres-label", type=float, default=0.1)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--drug-names", type=str, default=",".join(DEFAULT_DRUGS))
    parser.add_argument("--include-val-source", action="store_true")
    return parser.parse_args(argv)


def build_dataloaders(args: argparse.Namespace, gex_features_df: pd.DataFrame):
    set_seed(args.seed)
    drugs = [d.strip() for d in args.drug_names.split(",") if d.strip()]
    labeled_iter = get_multi_labeled_dataloader(
        gex_features_df=gex_features_df,
        drug=drugs,
        seed=args.seed,
        batch_size=args.batch_size,
        ccle_measurement=args.measurement,
        threshold_gdsc=args.thres_gdsc,
        threshold_label=args.thres_label,
        n_splits=args.n_splits,
    )
    train_loader = val_loader = tcga_loader = None
    for fold_idx, (train_ccle, val_ccle, tcga) in enumerate(labeled_iter):
        if fold_idx == args.fold:
            train_loader, val_loader, tcga_loader = train_ccle, val_ccle, tcga
            break
    if train_loader is None:
        raise ValueError(f"Fold {args.fold} not found.")
    return train_loader, val_loader, tcga_loader, drugs


def extract_raw_features(loader, drug_feat: np.ndarray) -> Dict[str, np.ndarray]:
    """
    For every batch, build per-drug raw input = [gene_expr; drug_fp].
    Returns arrays shaped like the model-based extraction script.
    """
    gene_list, fused_list, tissue_list = [], [], []
    for batch in loader:
        x = batch[0].numpy()            # [B, gene_dim]
        tissue = batch[-1].numpy()      # [B,]
        n_drugs = drug_feat.shape[0]
        # tile gene expression: [B, n_drugs, gene_dim]
        x_tiled = np.tile(x[:, None, :], (1, n_drugs, 1))
        # tile drug fingerprints: [B, n_drugs, fp_dim]
        drug_tiled = np.tile(drug_feat[None, :, :], (x.shape[0], 1, 1))
        # concatenate -> [B, n_drugs, gene_dim + fp_dim]
        fused = np.concatenate([x_tiled, drug_tiled], axis=-1)
        gene_list.append(x)
        fused_list.append(fused)
        tissue_list.append(tissue)

    return {
        "gene_feat": np.concatenate(gene_list, axis=0),
        "fused_feature": np.concatenate(fused_list, axis=0),
        "tissue": np.concatenate(tissue_list, axis=0),
    }


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    gex_features_df = pd.read_csv(project_config.gex_feature_file, index_col=0)
    # populate config.tissue_map
    get_unlabeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=args.seed,
        test_ratio=0.25,
        batch_size=args.batch_size,
    )

    train_loader, val_loader, tcga_loader, drugs = build_dataloaders(args, gex_features_df)
    drug_feat = get_drug_feat(drugs).astype("float32")
    print(f"Drug features shape: {drug_feat.shape}")

    # Source domain
    source_parts = [extract_raw_features(train_loader, drug_feat)]
    if args.include_val_source and val_loader is not None:
        source_parts.append(extract_raw_features(val_loader, drug_feat))
    source: Dict[str, np.ndarray] = {
        k: np.concatenate([p[k] for p in source_parts], axis=0) for k in source_parts[0]
    }

    # Target domain
    target = extract_raw_features(tcga_loader, drug_feat)

    tissue_map = project_config.tissue_map
    max_label = max(int(v) for v in tissue_map.values())
    tissue_names = [""] * (max_label + 1)
    for name, idx in tissue_map.items():
        tissue_names[int(idx)] = str(name)

    output_path = os.path.join(args.output_dir, "fmpa_tsne_features.npz")
    np.savez_compressed(
        output_path,
        source_gene_feat=source["gene_feat"],
        target_gene_feat=target["gene_feat"],
        source_fused_feat=source["fused_feature"],
        target_fused_feat=target["fused_feature"],
        source_tissue=source["tissue"],
        target_tissue=target["tissue"],
        drug_names=np.asarray(drugs),
        tissue_names=np.asarray(tissue_names),
        fold=np.array(args.fold),
        checkpoint=np.array("raw_input_no_model"),
    )
    print(f"Saved raw features: {output_path}")
    print(f"  source: {source['gene_feat'].shape}, target: {target['gene_feat'].shape}")
    print(f"  fused:  {source['fused_feature'].shape}")


if __name__ == "__main__":
    main()
