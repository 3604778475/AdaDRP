"""
MMPA-DRP Program Auxiliary Losses
=================================
Cleaned version: only keeps the program regularization losses used by
Stage 1 (diversity + sparsity + optional reconstruction).
"""

import torch
import torch.nn.functional as F


def program_diversity_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Encourage orthogonality among the K program tokens within each sample.

    Input:
        z: [B, K, d]
    Output:
        scalar loss (mean over batch)
    """
    B, K, d = z.shape
    z_norm = F.normalize(z, p=2, dim=-1)  # [B, K, d]
    sim = torch.bmm(z_norm, z_norm.transpose(1, 2))  # [B, K, K]

    # Mask diagonal
    mask = torch.eye(K, device=z.device).unsqueeze(0)  # [1, K, K]
    sim_off = sim * (1 - mask)

    # Mean squared off-diagonal similarity
    loss = (sim_off ** 2).sum() / (B * K * (K - 1) + 1e-8)
    return loss


def program_sparsity_loss(q: torch.Tensor) -> torch.Tensor:
    """
    Encourage sparse program activation (low entropy over q).

    Input:
        q: [B, K]  (softmax weights)
    Output:
        scalar loss (mean entropy)
    """
    entropy = -(q * torch.log(q + 1e-8)).sum(dim=1).mean()
    return entropy


def program_reconstruction_loss(
    z: torch.Tensor,
    q: torch.Tensor,
    x_recon: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Placeholder for reconstruction loss.
    When no decoder is used (current configuration), returns 0.
    """
    if x_recon is None:
        return torch.tensor(0.0, device=z.device)
    return F.mse_loss(x_recon, x)


def compute_program_aux_losses(
    program_out: dict,
    x_recon: torch.Tensor = None,
    x: torch.Tensor = None,
    lambda_rec: float = 0.0,
    lambda_div: float = 0.01,
    lambda_sparse: float = 0.001,
) -> tuple:
    """
    Compute optional auxiliary losses for program regularization.

    Returns:
        total_aux_loss, rec_loss, div_loss, sparse_loss
    """
    z = program_out["z"]
    q = program_out["q"]

    rec_loss = program_reconstruction_loss(z, q, x_recon, x)
    div_loss = program_diversity_loss(z)
    sparse_loss = program_sparsity_loss(q)

    total = lambda_rec * rec_loss + lambda_div * div_loss + lambda_sparse * sparse_loss
    return total, rec_loss, div_loss, sparse_loss
