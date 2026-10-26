"""
Build pseudo-domain centroids from trained Stage1 encoder.

Usage:
    python scripts/build_pseudo_domains.py \
        --checkpoint_path checkpoints/stage1/stage1_model.pt \
        --fold 0 \
        --k 8 \
        --output_dir outputs/pseudo_domain

If no checkpoint_path is provided, falls back to training a fresh Stage1 model
or using existing source_features.npy / target_features.npy.
"""

import os
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import torch
import gzip
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import config
from utility import set_seed, safe_make_dir
from dataload import get_unlabeled_dataloaders
from adadrp.models import ProgramTissueClassifier


def extract_h_pool_from_checkpoint(checkpoint_path, gex_features_df, device, seed=2025):
    """Load checkpoint and extract h_pool from source + target unlabeled data."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer model architecture from checkpoint keys
    # Try to read num_programs and program_dim from checkpoint
    num_programs = 8
    program_dim = 64
    backbone_type = "mlp"
    for k in checkpoint.keys():
        if "program_encoder.backbone" in k:
            if "convnext" in k:
                backbone_type = "convnext1d"
            elif "grn" in k:
                backbone_type = "grn_mlp"
    # Default architecture params
    model = ProgramTissueClassifier(
        input_dim=gex_features_df.shape[1] - 1,
        num_tissues=len(config.tissue_map),
        num_programs=num_programs,
        program_dim=program_dim,
        hidden_dims=[512, 256, 128, 64],
        drop=0.2,
        backbone_type=backbone_type,
    ).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    s_dataloaders, t_dataloaders = get_unlabeled_dataloaders(
        gex_features_df=gex_features_df, seed=seed, test_ratio=0.25, batch_size=256
    )

    source_features = []
    source_tissues = []
    target_features = []
    target_tissues = []

    with torch.no_grad():
        for batch in s_dataloaders[0]:
            inputs = batch[0].to(device)
            tissues = batch[1].to(device)
            out = model.program_encoder(inputs)
            source_features.append(out["h"].cpu().numpy())
            source_tissues.append(tissues.cpu().numpy())

        for batch in t_dataloaders[0]:
            inputs = batch[0].to(device)
            tissues = batch[1].to(device)
            out = model.program_encoder(inputs)
            target_features.append(out["h"].cpu().numpy())
            target_tissues.append(tissues.cpu().numpy())

    source_features = np.concatenate(source_features, axis=0)
    source_tissues = np.concatenate(source_tissues, axis=0)
    target_features = np.concatenate(target_features, axis=0)
    target_tissues = np.concatenate(target_tissues, axis=0)

    return source_features, source_tissues, target_features, target_tissues


def extract_h_pool_from_npy(source_npy="source_features.npy", target_npy="target_features.npy"):
    """Fallback: load pre-saved feature files."""
    source_features = np.load(source_npy)
    target_features = np.load(target_npy)
    # Tissue labels are not available in this fallback
    source_tissues = np.full(len(source_features), -1, dtype=np.int64)
    target_tissues = np.full(len(target_features), -1, dtype=np.int64)
    return source_features, source_tissues, target_features, target_tissues


def build_pseudo_domains(
    source_features,
    source_tissues,
    target_features,
    target_tissues,
    k=8,
    seed=42,
    use_source=True,
    use_target=True,
):
    """Run KMeans++ on L2-normalized h_pool."""
    parts = []
    split_labels = []
    tissue_list = []

    if use_source:
        parts.append(source_features)
        split_labels.extend(["source"] * len(source_features))
        tissue_list.extend(source_tissues.tolist())
    if use_target:
        parts.append(target_features)
        split_labels.extend(["target"] * len(target_features))
        tissue_list.extend(target_tissues.tolist())

    all_features = np.concatenate(parts, axis=0)  # [N, H]
    all_features_norm = normalize(all_features, norm="l2", axis=1)

    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=seed, n_init=10)
    assignments = kmeans.fit_predict(all_features_norm)
    centroids = kmeans.cluster_centers_  # [K, H]

    # NOTE: sklearn KMeans.cluster_centers_ are in the original feature space,
    # not normalized. For cosine similarity, we should normalize centroids too.
    centroids_norm = normalize(centroids, norm="l2", axis=1)

    return centroids_norm, assignments, split_labels, tissue_list, all_features


def save_diagnostics(
    output_dir, fold, k, assignments, split_labels, tissue_list, all_features, centroids
):
    """Save assignments.csv and diagnostics.csv."""
    safe_make_dir(output_dir)

    # Assignments CSV
    df_assign = pd.DataFrame({
        "fold": [fold] * len(assignments),
        "sample_index": range(len(assignments)),
        "split_type": split_labels,
        "tissue_idx": tissue_list,
        "pseudo_domain_id": assignments,
    })
    assign_path = os.path.join(output_dir, f"fold{fold}_K{k}_assignments.csv")
    df_assign.to_csv(assign_path, index=False)

    # Diagnostics CSV
    diag_rows = []
    for pid in range(k):
        mask = assignments == pid
        count_total = int(mask.sum())
        count_source = int(np.array([s == "source" for s in split_labels])[mask].sum())
        count_target = count_total - count_source
        source_ratio = count_source / count_total if count_total > 0 else 0.0

        # Tissue distribution
        tissues_in_cluster = np.array(tissue_list)[mask]
        tissues_in_cluster = tissues_in_cluster[tissues_in_cluster >= 0]
        if len(tissues_in_cluster) > 0:
            unique, counts = np.unique(tissues_in_cluster, return_counts=True)
            tissue_dist = {int(u): int(c) for u, c in zip(unique, counts)}
        else:
            tissue_dist = {}

        diag_rows.append({
            "fold": fold,
            "K": k,
            "pseudo_domain_id": pid,
            "count_total": count_total,
            "count_source": count_source,
            "count_target": count_target,
            "source_ratio": round(source_ratio, 4),
            "tissue_distribution_json": json.dumps(tissue_dist),
        })

    df_diag = pd.DataFrame(diag_rows)
    diag_path = os.path.join(output_dir, f"fold{fold}_K{k}_diagnostics.csv")
    df_diag.to_csv(diag_path, index=False)

    # Centroids NPY
    centroids_path = os.path.join(output_dir, f"fold{fold}_K{k}_centroids.npy")
    np.save(centroids_path, centroids.astype(np.float32))

    print(f"Saved:")
    print(f"  {centroids_path}")
    print(f"  {assign_path}")
    print(f"  {diag_path}")

    return df_diag


def main():
    parser = argparse.ArgumentParser("Build Pseudo-Domain Centroids")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to trained ProgramTissueClassifier checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/pseudo_domain")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--feature_key", type=str, default="h",
                        choices=["h", "h_pred"], help="Feature key to extract")
    parser.add_argument("--use_source", action="store_true", default=True,
                        help="Include source domain samples")
    parser.add_argument("--use_target", action="store_true", default=True,
                        help="Include target domain samples")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_npy", type=str, default="source_features.npy")
    parser.add_argument("--target_npy", type=str, default="target_features.npy")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)

    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        source_features, source_tissues, target_features, target_tissues = \
            extract_h_pool_from_checkpoint(args.checkpoint_path, gex_features_df, device, seed=args.seed)
    elif os.path.exists(args.source_npy) and os.path.exists(args.target_npy):
        print(f"No checkpoint found, falling back to {args.source_npy} and {args.target_npy}")
        source_features, source_tissues, target_features, target_tissues = \
            extract_h_pool_from_npy(args.source_npy, args.target_npy)
    else:
        raise FileNotFoundError(
            f"No checkpoint at {args.checkpoint_path} and no feature files found. "
            "Please train a Stage1 model first or provide feature files."
        )

    print(f"Source features: {source_features.shape}")
    print(f"Target features: {target_features.shape}")

    centroids, assignments, split_labels, tissue_list, all_features = build_pseudo_domains(
        source_features, source_tissues,
        target_features, target_tissues,
        k=args.k, seed=args.seed,
        use_source=args.use_source,
        use_target=args.use_target,
    )

    print(f"Centroids shape: {centroids.shape}")
    df_diag = save_diagnostics(
        args.output_dir, args.fold, args.k,
        assignments, split_labels, tissue_list, all_features, centroids,
    )
    print(df_diag.to_string(index=False))


if __name__ == "__main__":
    main()
