"""
Phase 17: Drug Gradient Diagnostics for Group Private Adapter
Runs a short baseline training to collect per-drug gradients, computes
cosine similarity matrix, and clusters drugs into groups via K-Means.

Output: data/drug_group_ids.json
"""

import os
import sys
import json
import argparse
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from mmpa_drp.models import ProgramFusionMLP, ProgramTissueClassifier
from copy import deepcopy

# Pre-import utility to resolve circular import with dataload
# (mirrors train_mmpa_program.py import order)
import utility


def diagnose_drug_groups(
    drugs,
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    num_diagnose_epochs=5,
    batch_size=64,
    seed=2025,
    n_clusters=3,
    output_path='data/drug_group_ids.json',
):
    """
    Train baseline for a few epochs, collect per-drug gradients from
    fusion_module's last layer, cluster drugs by gradient similarity.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)
    gene_num = gex_features_df.shape[1] - 1

    # Initialize tissue_map by calling get_unlabeled_dataloaders first
    print("[Diagnose] Initializing tissue map...")
    from dataload import get_unlabeled_dataloaders
    _ = get_unlabeled_dataloaders(gex_features_df=gex_features_df, seed=seed, test_ratio=0.25, batch_size=batch_size)
    num_tissues = len(config.tissue_map)

    # Stage 1 pretrain (short, just to get a reasonable encoder)
    print("[Diagnose] Stage 1: quick pretraining...")
    pretrain_model = ProgramTissueClassifier(
        input_dim=gene_num,
        num_tissues=num_tissues,
        num_programs=8,
        program_dim=64,
        hidden_dims=[512, 256, 128, 64],
        drop=0.2,
        backbone_type='convnext1d',
        norm_type='layer',
        num_gene_tokens=256,
        token_dim=64,
        convnext_depth=1,
        convnext_kernel_size=7,
        convnext_expansion=2,
    ).to(device)

    from dataload import get_drug_feat
    drug_feat = torch.from_numpy(get_drug_feat(drugs).astype('float32')).to(device)

    # Build unlabeled dataloaders for stage-1 pretrain
    from dataload import get_unlabeled_dataloaders
    s_dataloaders, t_dataloaders = get_unlabeled_dataloaders(
        gex_features_df=gex_features_df, seed=seed, test_ratio=0.25, batch_size=batch_size
    )

    # Quick stage-1 pretrain
    from mmpa_drp.stage1_program import train_tissue_classifier_program
    trained_model = train_tissue_classifier_program(
        model=pretrain_model,
        s_dataloaders=s_dataloaders,
        t_dataloaders=t_dataloaders,
        device=device,
        num_epochs=3,  # short pretrain
        lr=0.001,
        alpha=1.0,
        weight_decay=1e-4,
        lambda_div=0.01,
        lambda_sparse=0.001,
    )

    # Build labeled dataloaders for gradient diagnosis
    from dataload import get_multi_labeled_dataloader
    labeled_dataloader = get_multi_labeled_dataloader(
        gex_features_df=gex_features_df, seed=seed, batch_size=batch_size,
        drug=drugs, ccle_measurement='Z_SCORE', threshold_gdsc=0.0, threshold_label=0.1,
        n_splits=5
    )
    train_labeled_ccle, test_labeled_ccle, labeled_tcga = next(iter(labeled_dataloader))

    # Build baseline fusion model
    print("[Diagnose] Building baseline fusion model...")
    encoder = deepcopy(trained_model.program_encoder)
    bilinear_dim = 128
    drug_feat_out_dim = 64
    main_fusion_out_dim = bilinear_dim
    # MolecularProgramEncoder uses backbone_output_dim, not hidden_dims
    encoder_out_dim = getattr(encoder, 'backbone_output_dim', 64)
    base_input_dim = encoder_out_dim + drug_feat_out_dim + main_fusion_out_dim

    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=base_input_dim,
        output_dim=1,
        hidden_dims=[256, 128, 64],
        gene_num=gene_num,
        bilinear_dim=bilinear_dim,
        gene_feat_dim=encoder_out_dim,
        drug_feat_dim=drug_feat_out_dim,
        drug_num=len(drugs),
        drop=0.1,
        use_fpi=False,
        use_cdan=True,
        use_ot_alignment=True,
        lambda_ot=0.005,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

    # Collect per-drug gradients
    print(f"[Diagnose] Collecting gradients for {num_diagnose_epochs} epochs...")
    drug_grads = {d: [] for d in range(len(drugs))}

    for epoch in range(num_diagnose_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_labeled_ccle):
            s_x = batch[0].to(device)
            s_y = batch[1].to(device)
            s_mask = (batch[2].to(device) > 0)
            s_tissue = batch[3].to(device)

            optimizer.zero_grad()
            outputs = model(s_x, drug_feat, s_tissue, alpha=1.0, return_dict=True)
            s_yp = outputs["pred"].reshape(s_y.shape)
            s_loss_mat = loss_fn(s_yp, s_y.double())
            s_loss_mat = torch.where(s_mask, s_loss_mat,
                                     torch.zeros_like(s_loss_mat))

            # Per-drug gradient extraction using torch.autograd.grad
            # to avoid gradient accumulation across drugs
            for d in range(len(drugs)):
                drug_mask = s_mask[:, d]
                if drug_mask.sum() == 0:
                    continue
                bce_d = s_loss_mat[:, d][drug_mask].mean()
                if bce_d.item() == 0:
                    continue
                # Compute gradient w.r.t. fusion_module parameters without accumulating
                grads = torch.autograd.grad(
                    bce_d,
                    list(model.fusion_module.parameters()),
                    retain_graph=True,
                    create_graph=False,
                )
                # Concatenate all parameter gradients into a single vector
                if grads is not None and len(grads) > 0:
                    grad_vec = torch.cat([g.detach().clone().flatten() for g in grads if g is not None])
                    drug_grads[d].append(grad_vec.cpu())

            optimizer.zero_grad()

    # Average gradients per drug
    mean_grads = []
    ref_dim = None
    for d in range(len(drugs)):
        if len(drug_grads[d]) > 0:
            g = torch.stack(drug_grads[d]).mean(dim=0)
            if ref_dim is None:
                ref_dim = g.shape[0]
            mean_grads.append(g)
        else:
            # Use zero vector with same dimension as first observed grad
            dim = ref_dim if ref_dim is not None else 1
            mean_grads.append(torch.zeros(dim))
    mean_grads = torch.stack(mean_grads).to(device)

    # Cosine similarity matrix (handle zero vectors by adding small epsilon)
    norms = torch.norm(mean_grads, p=2, dim=1, keepdim=True)
    mean_grads_norm = mean_grads / (norms + 1e-8)
    sim_matrix = torch.mm(mean_grads_norm, mean_grads_norm.t()).cpu().numpy()

    print("\n[Diagnose] Drug Cosine Similarity Matrix:")
    print("         " + " ".join([f"{d[:3]:>4}" for d in drugs]))
    for i, d_name in enumerate(drugs):
        row = " ".join([f"{sim_matrix[i, j]:4.2f}" for j in range(len(drugs))])
        print(f"{d_name[:10]:10} {row}")

    # K-Means clustering
    print(f"\n[Diagnose] K-Means clustering (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    group_ids = kmeans.fit_predict(sim_matrix).tolist()

    # If silhouette is poor with n_clusters, try 2..5 and pick best
    from sklearn.metrics import silhouette_score
    best_score = silhouette_score(sim_matrix, group_ids)
    best_k = n_clusters
    best_group_ids = group_ids
    for k in range(2, 6):
        if k == n_clusters:
            continue
        ids = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(sim_matrix)
        score = silhouette_score(sim_matrix, ids)
        if score > best_score:
            best_score = score
            best_k = k
            best_group_ids = ids.tolist()

    print(f"[Diagnose] Best K={best_k}, silhouette={best_score:.3f}")
    print(f"[Diagnose] group_ids: {best_group_ids}")
    for g in set(best_group_ids):
        members = [drugs[i] for i, gid in enumerate(best_group_ids) if gid == g]
        print(f"  Group {g}: {members}")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "group_ids": best_group_ids,
            "similarity_matrix": sim_matrix.tolist(),
            "drugs": drugs,
            "best_k": int(best_k),
            "silhouette": float(best_score),
        }, f, indent=2)
    print(f"[Diagnose] Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', type=str, default='data/drug_group_ids.json')
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()

    drugs = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel",
             "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"]

    diagnose_drug_groups(
        drugs=drugs,
        n_clusters=args.n_clusters,
        num_diagnose_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_path=args.output,
        seed=args.seed,
    )
