"""
Phase 5: Program Diagnostics
=============================
Compute diagnostic metrics for MolecularProgramEncoder outputs.
"""

import torch
import torch.nn.functional as F


def compute_program_diagnostics(encoder, dataloader, device, max_batches=10):
    """
    Compute program diagnostics from encoder outputs on a subset of batches.

    Args:
        encoder: MolecularProgramEncoder (or model with .program_encoder)
        dataloader: data loader
        device: torch device
        max_batches: number of batches to sample

    Returns:
        dict with diagnostic metrics (all as Python scalars)
    """
    encoder.eval()

    all_q = []
    all_z = []
    all_h = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            inputs = batch[0].to(device)
            out = encoder(inputs)
            all_q.append(out["q"].cpu())
            all_z.append(out["z"].cpu())
            all_h.append(out["h"].cpu())

    if not all_q:
        return {}

    q = torch.cat(all_q, dim=0)  # [N, K]
    z = torch.cat(all_z, dim=0)  # [N, K, d]
    h = torch.cat(all_h, dim=0)  # [N, d]

    # 1. q_entropy: average entropy of program activations
    entropy = -(q * torch.log(q + 1e-8)).sum(dim=-1).mean().item()

    # 2. program_pairwise_cosine: average cosine similarity between program vectors
    # Average over samples first, then compute pairwise cosine
    z_mean = z.mean(dim=0)  # [K, d]
    z_norm = F.normalize(z_mean, p=2, dim=-1)
    cos_sim_matrix = torch.mm(z_norm, z_norm.t())  # [K, K]
    # Exclude diagonal
    mask = ~torch.eye(cos_sim_matrix.size(0), dtype=torch.bool)
    pairwise_cosine = cos_sim_matrix[mask].mean().item()

    # 3. effective_active_programs: average number of programs with q > 0.05 per sample
    effective_active = (q > 0.05).float().sum(dim=-1).mean().item()

    # 4. z_norm_mean
    z_norm_mean = z.norm(p=2, dim=-1).mean().item()

    # 5. z_norm_std
    z_norm_std = z.norm(p=2, dim=-1).std().item()

    # 6. hidden_feature_cosine_distance: average pairwise cosine distance of h
    h_norm = F.normalize(h, p=2, dim=-1)
    h_cos = torch.mm(h_norm, h_norm.t())  # [N, N]
    mask = ~torch.eye(h_cos.size(0), dtype=torch.bool)
    h_cos_mean = h_cos[mask].mean().item()
    hidden_feature_cosine_distance = 1.0 - h_cos_mean  # cosine distance = 1 - similarity

    return {
        "q_entropy": entropy,
        "program_pairwise_cosine": pairwise_cosine,
        "effective_active_programs": effective_active,
        "z_norm_mean": z_norm_mean,
        "z_norm_std": z_norm_std,
        "hidden_feature_cosine_distance": hidden_feature_cosine_distance,
    }
