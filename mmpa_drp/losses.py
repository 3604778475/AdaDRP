"""
MMPA-DRP Extension Losses
=========================
Phase 1 minimal implementations for program-level alignment and contrastive losses.
These are intended to replace / augment the sample-level contrastive loss in
MACB-DRP Stage 3 with tissue-program prototype alignment and drug-gated program
contrastive separation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def program_prototype_alignment_loss(
    source_z: torch.Tensor,
    target_z: torch.Tensor,
    source_tissue: torch.Tensor,
    target_tissue: torch.Tensor,
    drug_gate: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    ProgramPrototypeAlignmentLoss
    =============================
    Align tissue-level program prototypes across source and target domains,
    weighted by drug mechanism gates.

    Inputs:
        source_z:       [Bs, K, d]   source program tokens
        target_z:       [Bt, K, d]   target program tokens
        source_tissue:  [Bs]         tissue labels for source
        target_tissue:  [Bt]         tissue labels for target
        drug_gate:      [D, K]       drug-specific program gates
    Output:
        loss_align:     scalar

    TODO (Phase 2):
        1. Compute per-tissue program prototypes by averaging z over samples
           within each tissue type, separately for source and target.
        2. Align source prototypes to target prototypes (or vice versa)
           using a cross-domain prototype matching loss.
        3. Optionally weight the alignment by drug_gate to focus on
           drug-relevant programs.
    """
    # Phase 1: return a dummy zero loss to preserve interface
    return torch.tensor(0.0, device=source_z.device, requires_grad=True)


def program_prototype_contrastive_loss(
    source_prototypes: torch.Tensor,
    target_prototypes: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    ProgramPrototypeContrastiveLoss
    ===============================
    Contrastive separation at the tissue-program prototype level.

    Inputs:
        source_prototypes: [C, K, d]  source domain prototypes (C tissue types)
        target_prototypes: [C, K, d]  target domain prototypes (C tissue types)
    Output:
        loss_contrast: scalar

    TODO (Phase 2):
        1. For each tissue c, treat source_prototypes[c] and target_prototypes[c]
           as positive pairs.
        2. Treat prototypes from different tissues as negatives.
        3. Apply a contrastive objective (e.g., InfoNCE) at the program level.
    """
    # Phase 1: return a dummy zero loss to preserve interface
    return torch.tensor(0.0, device=source_prototypes.device, requires_grad=True)


class ProgramAlignmentLoss(nn.Module):
    """
    nn.Module wrapper for program_prototype_alignment_loss.
    Useful if we need to store learnable parameters (e.g. prototype bank) later.
    """

    def __init__(self, temperature: float = 0.1):
        super(ProgramAlignmentLoss, self).__init__()
        self.temperature = temperature

    def forward(
        self,
        source_z: torch.Tensor,
        target_z: torch.Tensor,
        source_tissue: torch.Tensor,
        target_tissue: torch.Tensor,
        drug_gate: torch.Tensor,
    ) -> torch.Tensor:
        return program_prototype_alignment_loss(
            source_z, target_z, source_tissue, target_tissue, drug_gate, self.temperature
        )


class ProgramContrastiveLoss(nn.Module):
    """
    nn.Module wrapper for program_prototype_contrastive_loss.
    """

    def __init__(self, temperature: float = 0.1):
        super(ProgramContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(
        self,
        source_prototypes: torch.Tensor,
        target_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        return program_prototype_contrastive_loss(
            source_prototypes, target_prototypes, self.temperature
        )


# =============================================================================
# Program Auxiliary Losses (Phase 2)
# =============================================================================

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


def program_reconstruction_loss(z: torch.Tensor, q: torch.Tensor, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Placeholder for reconstruction loss.
    In Phase 2 we do not instantiate a decoder yet, so this returns 0.
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


# =============================================================================
# Phase 3A: Disentanglement Losses
# =============================================================================

def orthogonality_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Orthogonality loss inspired by DCDNet DiffLoss.
    Encourages a and b to be orthogonal (low cosine similarity).

    Compatible with:
        a, b: [B, d]   or   [B, K, d]

    Logic:
        1. Flatten to [B, -1]
        2. L2 normalize per sample
        3. Compute batch-wise cosine similarity matrix: sim = a_norm @ b_norm.T
        4. Return mean of sim^2
    """
    # Flatten
    a_flat = a.view(a.size(0), -1)
    b_flat = b.view(b.size(0), -1)

    # L2 normalize (detach norm to avoid degenerate gradients)
    a_norm = torch.norm(a_flat, p=2, dim=1, keepdim=True).detach()
    a_flat = a_flat.div(a_norm.expand_as(a_flat) + 1e-6)

    b_norm = torch.norm(b_flat, p=2, dim=1, keepdim=True).detach()
    b_flat = b_flat.div(b_norm.expand_as(b_flat) + 1e-6)

    # Cosine similarity matrix [B, B]
    sim = torch.mm(a_flat, b_flat.t())
    return torch.mean(sim.pow(2))


def response_domain_adversarial_loss(
    h_response: torch.Tensor,
    domain_label: torch.Tensor,
    discriminator,
    grl_lambda: float = 1.0,
) -> torch.Tensor:
    """
    Adversarial loss on response branch.
    GRL makes h_response domain-invariant.
    """
    from mmpa_drp.models import GradientReversal

    h_rev = GradientReversal.apply(h_response, grl_lambda)
    logits = discriminator(h_rev).squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits, domain_label.float())


def domain_nuisance_loss(
    h_domain: torch.Tensor,
    domain_label: torch.Tensor,
    discriminator,
) -> torch.Tensor:
    """
    Domain classification loss on domain-nuisance branch.
    Makes h_domain retain domain information.
    """
    logits = discriminator(h_domain).squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits, domain_label.float())


# =============================================================================
# Phase 3B: Residual Soft Drug Gate Losses
# =============================================================================

def entropy_floor_loss(g, min_entropy=1.2, eps=1e-8):
    """
    Prevent gate from collapsing to one-hot.
    Only penalizes when entropy < min_entropy.

    g: [D, K] softmax probabilities
    """
    entropy = -(g * torch.log(g + eps)).sum(dim=-1)  # [D]
    loss = F.relu(min_entropy - entropy).pow(2).mean()
    return loss


def gate_load_balance_loss(g, eps=1e-8):
    """
    Encourage uniform usage of programs across all drugs.

    g: [D, K]
    """
    mean_usage = g.mean(dim=0)  # [K]
    K = g.size(1)
    uniform = torch.ones_like(mean_usage) / K
    # KL(mean_usage || uniform)
    loss = (mean_usage * torch.log((mean_usage + eps) / (uniform + eps))).sum()
    return loss


def gate_diversity_loss(g):
    """
    Encourage different drugs to have different gates.
    Penalizes high off-diagonal cosine similarity.

    g: [D, K]
    """
    g_norm = F.normalize(g, p=2, dim=-1)  # [D, K]
    sim = torch.mm(g_norm, g_norm.t())  # [D, D]
    # Mask diagonal
    mask = 1.0 - torch.eye(g.size(0), device=g.device)
    sim_off = sim * mask
    loss = (sim_off ** 2).sum() / (mask.sum() + 1e-8)
    return loss


# =============================================================================
# Phase 8: DSPI Losses
# =============================================================================

def drug_attention_diversity_loss(attn: torch.Tensor) -> torch.Tensor:
    """
    Encourage different drugs to attend to different programs.

    Input:
        attn: [B, D, K]  attention weights over K programs for D drugs
    Output:
        scalar loss

    Steps:
        1. Average over batch: mean_attn = attn.mean(dim=0)  # [D, K]
        2. Normalize each drug's attention vector
        3. Compute cosine similarity matrix [D, D]
        4. Penalize off-diagonal similarity
    """
    mean_attn = attn.mean(dim=0)  # [D, K]
    mean_attn = F.normalize(mean_attn, p=2, dim=-1)  # [D, K]
    sim = torch.mm(mean_attn, mean_attn.t())  # [D, D]

    D = attn.size(1)
    mask = 1.0 - torch.eye(D, device=attn.device)
    sim_off = sim * mask
    loss = (sim_off ** 2).sum() / (mask.sum() + 1e-8)
    return loss


def drug_attention_entropy_loss(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Encourage more focused (lower entropy) attention per drug.

    Input:
        attn: [B, D, K]
    Output:
        scalar loss (mean negative entropy, so minimizing = encouraging focus)
    """
    entropy = -(attn * torch.log(attn + eps)).sum(dim=-1)  # [B, D]
    return entropy.mean()


def program_usage_balance_loss(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Encourage uniform usage of all K programs across drugs and samples.

    Input:
        attn: [B, D, K]
    Output:
        scalar loss (MSE between mean usage and uniform distribution)
    """
    K = attn.size(-1)
    usage = attn.mean(dim=(0, 1))  # [K]
    target = torch.ones_like(usage) / K
    loss = F.mse_loss(usage, target)
    return loss


def drug_gram_consistency_loss(
    h_drug_source: torch.Tensor,
    h_drug_target: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage consistency of drug-drug relation structure across domains.

    Input:
        h_drug_source: [Bs, D, H]
        h_drug_target: [Bt, D, H]
    Output:
        scalar loss

    Steps:
        1. Normalize h_drug per sample
        2. Compute Gram matrix G = V @ V.T  [B, D, D]
        3. Average over batch: G_mean = G.mean(dim=0)  [D, D]
        4. MSE between source and target G_mean
    """
    # Normalize
    V_s = F.normalize(h_drug_source, p=2, dim=-1)  # [Bs, D, H]
    V_t = F.normalize(h_drug_target, p=2, dim=-1)  # [Bt, D, H]

    # Gram matrices
    G_s = torch.einsum("bdh,bkh->bdk", V_s, V_s)  # [Bs, D, D]
    G_t = torch.einsum("bdh,bkh->bdk", V_t, V_t)  # [Bt, D, D]

    # Average over batch
    G_s_mean = G_s.mean(dim=0)  # [D, D]
    G_t_mean = G_t.mean(dim=0)  # [D, D]

    loss = F.mse_loss(G_s_mean, G_t_mean)
    return loss


# =============================================================================
# Phase 8-F: Fusion-Repair DSPI losses
# =============================================================================

def attention_entropy_loss(attn, eps=1e-8):
    """
    Compute mean entropy of attention distributions.
    Args:
        attn: [B, D, K] attention weights (sum to 1 over K)
    Returns:
        scalar: mean entropy
    """
    entropy = -torch.sum(attn * torch.log(attn + eps), dim=-1)  # [B, D]
    return entropy.mean()


def attention_diversity_loss(attn, eps=1e-8):
    """
    Compute diversity loss: penalize similar average attention across drugs.
    Args:
        attn: [B, D, K] attention weights
    Returns:
        scalar: mean off-diagonal similarity
    """
    mean_attn = attn.mean(dim=0)  # [D, K]
    mean_attn = F.normalize(mean_attn, p=2, dim=-1)
    sim = torch.mm(mean_attn, mean_attn.t())  # [D, D]

    # Off-diagonal mean
    D = sim.size(0)
    mask = 1.0 - torch.eye(D, device=sim.device)
    off_diag_sum = (sim * mask).sum()
    off_diag_count = D * (D - 1)
    return off_diag_sum / (off_diag_count + eps)


def gamma_l1_loss(gamma):
    """
    L1 regularization on gamma values.
    Args:
        gamma: [D] per-drug gamma
    Returns:
        scalar: mean absolute gamma
    """
    return gamma.abs().mean()


def gamma_l2_loss(gamma):
    """
    L2 regularization on gamma values (alternative to L1).
    Args:
        gamma: [D] per-drug gamma
    Returns:
        scalar: mean squared gamma
    """
    return (gamma ** 2).mean()


# =============================================================================
# Phase 13: Reliable Stage-3 Contrastive Anchoring (S3R)
# =============================================================================

def build_reliable_contrastive_samples(
    anchor_feat,
    all_target_feat,
    topk_idx,
    drug_ids,
    topk_sim=None,
    same_drug_negatives=True,
    suppress_false_negatives=True,
    max_negatives_per_anchor=0,
    weighted_positives=False,
    positive_temperature=0.1,
):
    """
    Build reliable positive/negative pools for Stage-3 contrastive anchoring.

    Args:
        anchor_feat: [N, F]
        all_target_feat: [T, D, F]
        topk_idx: [N, K]
        drug_ids: [N]
        topk_sim: [N, K] or None
        same_drug_negatives: if True, negatives only from same-drug target pool
        suppress_false_negatives: if True, remove topk positives from negative pool
        max_negatives_per_anchor: if >0, randomly subsample negatives
        weighted_positives: if True, weight positives by similarity or rank
        positive_temperature: temperature for positive weighting softmax

    Returns:
        dict with keys:
            anchors: [N_valid, F]
            positives: [N_valid, K, F]
            negatives: list[Tensor], each [M_i, F]
            positive_weights: [N_valid, K] or None
            valid_mask: [N] bool
            num_negatives: [N_valid]
    """
    import math
    N, feat_dim = anchor_feat.shape
    T, D, _ = all_target_feat.shape
    K = topk_idx.size(1)

    positives_list = []
    negatives_list = []
    positive_weights_list = []
    valid_mask = torch.ones(N, dtype=torch.bool, device=anchor_feat.device)
    num_neg_list = []

    for n in range(N):
        drug_id = int(drug_ids[n].item())
        pos_target_ids = topk_idx[n]  # [K]

        # Positives
        pos = all_target_feat[pos_target_ids, drug_id, :]  # [K, F]
        positives_list.append(pos)

        # Negatives
        if same_drug_negatives:
            candidate = all_target_feat[:, drug_id, :]  # [T, feat_dim]
            candidate_ids = torch.arange(T, device=candidate.device)
        else:
            candidate = all_target_feat.reshape(T * D, feat_dim)  # [T*D, feat_dim]
            candidate_ids = torch.arange(T * D, device=candidate.device)

        if suppress_false_negatives:
            if same_drug_negatives:
                # pos_target_ids are target indices under same drug
                mask = torch.ones(T, dtype=torch.bool, device=candidate.device)
                mask[pos_target_ids] = False
            else:
                # pair ids = pos_target_ids * D + drug_id
                pair_ids = pos_target_ids * D + drug_id
                mask = torch.ones(T * D, dtype=torch.bool, device=candidate.device)
                mask[pair_ids] = False
            candidate = candidate[mask]
            candidate_ids = candidate_ids[mask]

        # Check if any negatives remain
        if candidate.size(0) == 0:
            valid_mask[n] = False
            negatives_list.append(torch.empty(0, feat_dim, device=anchor_feat.device))
            num_neg_list.append(0)
            continue

        # Subsample negatives if needed
        if max_negatives_per_anchor > 0 and candidate.size(0) > max_negatives_per_anchor:
            perm = torch.randperm(candidate.size(0), device=candidate.device)
            selected = perm[:max_negatives_per_anchor]
            candidate = candidate[selected]

        negatives_list.append(candidate)
        num_neg_list.append(candidate.size(0))

    # Positive weights
    if weighted_positives:
        if topk_sim is not None:
            for n in range(N):
                sim = topk_sim[n]  # [K]
                w = F.softmax(sim / positive_temperature, dim=0)
                positive_weights_list.append(w)
        else:
            rank_score = -torch.arange(K, dtype=torch.float32, device=anchor_feat.device)
            w = F.softmax(rank_score / positive_temperature, dim=0)
            for n in range(N):
                positive_weights_list.append(w.clone())
    else:
        positive_weights_list = None

    # Filter valid anchors
    if valid_mask.sum() == 0:
        return {
            "anchors": torch.empty(0, feat_dim, device=anchor_feat.device),
            "positives": torch.empty(0, K, feat_dim, device=anchor_feat.device),
            "negatives": [],
            "positive_weights": None,
            "valid_mask": valid_mask,
            "num_negatives": torch.tensor([], dtype=torch.long, device=anchor_feat.device),
        }

    anchors = anchor_feat[valid_mask]
    positives = torch.stack([positives_list[i] for i in range(N) if valid_mask[i]], dim=0)
    negatives = [negatives_list[i] for i in range(N) if valid_mask[i]]
    num_negatives = torch.tensor([num_neg_list[i] for i in range(N) if valid_mask[i]],
                                   dtype=torch.long, device=anchor_feat.device)

    if positive_weights_list is not None:
        positive_weights = torch.stack([positive_weights_list[i] for i in range(N) if valid_mask[i]], dim=0)
    else:
        positive_weights = None

    return {
        "anchors": anchors,
        "positives": positives,
        "negatives": negatives,
        "positive_weights": positive_weights,
        "valid_mask": valid_mask,
        "num_negatives": num_negatives,
    }


def reliable_contrastive_loss(
    anchors,
    positives,
    negatives,
    temperature=0.1,
    positive_weights=None,
):
    """
    Reliable contrastive loss with per-anchor negatives and optional positive weighting.

    Args:
        anchors: [N, F]
        positives: [N, K, F]
        negatives: list[Tensor], each [M_i, F]
        temperature: float
        positive_weights: [N, K] or None

    Returns:
        dict with keys:
            loss: scalar
            pos_sim_mean: float
            neg_sim_mean: float
            num_valid: int
            num_neg_mean: float
    """
    import math
    N = anchors.size(0)
    K = positives.size(1)

    # Normalize
    anchors_norm = F.normalize(anchors, dim=-1)
    positives_norm = F.normalize(positives, dim=-1)

    losses = []
    pos_sims = []
    neg_sims = []
    neg_counts = []

    for i in range(N):
        anc = anchors_norm[i]  # [F]
        pos = positives_norm[i]  # [K, F]
        neg = negatives[i]  # [M_i, F]

        # Positive similarities
        sim_pos = torch.matmul(pos, anc) / temperature  # [K]

        # Negative similarities
        if neg.size(0) > 0:
            neg_norm = F.normalize(neg, dim=-1)
            sim_neg = torch.matmul(neg_norm, anc) / temperature  # [M_i]
        else:
            sim_neg = torch.empty(0, device=anc.device)

        # Numerator
        if positive_weights is not None:
            w = positive_weights[i]  # [K]
            log_numerator = torch.logsumexp(sim_pos + torch.log(w + 1e-12), dim=0)
        else:
            log_numerator = torch.logsumexp(sim_pos, dim=0) - math.log(max(K, 1))

        # Denominator
        all_logits = torch.cat([sim_pos, sim_neg], dim=0)
        log_denominator = torch.logsumexp(all_logits, dim=0)

        loss_i = -(log_numerator - log_denominator)
        losses.append(loss_i)

        pos_sims.append(sim_pos.mean().item())
        if sim_neg.numel() > 0:
            neg_sims.append(sim_neg.mean().item())
        neg_counts.append(neg.size(0))

    loss = torch.stack(losses).mean()
    pos_sim_mean = sum(pos_sims) / len(pos_sims) if pos_sims else 0.0
    neg_sim_mean = sum(neg_sims) / len(neg_sims) if neg_sims else 0.0
    num_neg_mean = sum(neg_counts) / len(neg_counts) if neg_counts else 0.0

    return {
        "loss": loss,
        "pos_sim_mean": pos_sim_mean,
        "neg_sim_mean": neg_sim_mean,
        "num_valid": N,
        "num_neg_mean": num_neg_mean,
    }


def drug_wise_pairwise_ranking_loss(
    logits,
    labels,
    mask=None,
    drug_weights=None,
    max_pairs_per_drug=0,
    margin=0.0,
    reduction="mean",
    return_details=False,
):
    """
    Drug-wise pairwise ranking loss on source labeled data.

    For each drug d:
        positive samples: labels[:, d] > 0.5
        negative samples: labels[:, d] <= 0.5
        loss_d = mean( softplus( -(pos_logit - neg_logit - margin) ) )

    Args:
        logits:  [B, D]  raw final logits (before sigmoid)
        labels:  [B, D]  binary labels (0 or 1)
        mask:    [B, D] or None  valid mask (True = valid)
        drug_weights: [D] or None  per-drug loss weight
        max_pairs_per_drug: int  max pairs per drug (0 = unlimited)
        margin: float  ranking margin
        reduction: "mean" or "sum"
        return_details: bool

    Returns:
        loss tensor, or dict with diagnostics if return_details=True.
    """
    B, D = logits.shape
    device = logits.device

    losses = []
    loss_per_drug = torch.zeros(D, device=device)
    num_pos_per_drug = torch.zeros(D, dtype=torch.long, device=device)
    num_neg_per_drug = torch.zeros(D, dtype=torch.long, device=device)
    num_pairs_per_drug = torch.zeros(D, dtype=torch.long, device=device)
    valid_drug_mask = torch.zeros(D, dtype=torch.bool, device=device)
    mean_pos_logit_per_drug = torch.full((D,), float('nan'), device=device)
    mean_neg_logit_per_drug = torch.full((D,), float('nan'), device=device)
    mean_pair_diff_per_drug = torch.full((D,), float('nan'), device=device)

    mean_pos_logit = 0.0
    mean_neg_logit = 0.0
    mean_pair_diff = 0.0
    total_pairs = 0

    for d in range(D):
        if mask is not None:
            valid = mask[:, d]
        else:
            valid = torch.ones(B, dtype=torch.bool, device=device)

        y_d = labels[:, d]
        s_d = logits[:, d]

        pos_mask = valid & (y_d > 0.5)
        neg_mask = valid & (y_d <= 0.5)

        pos = s_d[pos_mask]
        neg = s_d[neg_mask]
        num_pos = pos.numel()
        num_neg = neg.numel()
        num_pos_per_drug[d] = num_pos
        num_neg_per_drug[d] = num_neg

        if num_pos == 0 or num_neg == 0:
            continue

        # Pairwise differences: [P, N]
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)  # [P, N]
        diff = diff.reshape(-1)

        if max_pairs_per_drug > 0 and diff.numel() > max_pairs_per_drug:
            idx = torch.randperm(diff.numel(), device=device)[:max_pairs_per_drug]
            diff = diff[idx]

        loss_d = F.softplus(-(diff - margin)).mean()

        if drug_weights is not None:
            loss_d = loss_d * drug_weights[d]

        losses.append(loss_d)

        loss_per_drug[d] = loss_d.detach()
        num_pairs_per_drug[d] = diff.numel()
        valid_drug_mask[d] = True
        mean_pos_logit_per_drug[d] = pos.mean().item()
        mean_neg_logit_per_drug[d] = neg.mean().item()
        mean_pair_diff_per_drug[d] = diff.mean().item()

        mean_pos_logit += pos.mean().item() * num_pos
        mean_neg_logit += neg.mean().item() * num_neg
        mean_pair_diff += diff.mean().item() * diff.numel()
        total_pairs += diff.numel()

    if len(losses) == 0:
        loss = logits.new_tensor(0.0)
        if return_details:
            return {
                "loss": loss,
                "num_valid_drugs": 0,
                "num_pairs_total": 0,
                "loss_per_drug": loss_per_drug,
                "num_pos_per_drug": num_pos_per_drug,
                "num_neg_per_drug": num_neg_per_drug,
                "num_pairs_per_drug": num_pairs_per_drug,
                "valid_drug_mask": valid_drug_mask,
                "mean_pos_logit_per_drug": mean_pos_logit_per_drug,
                "mean_neg_logit_per_drug": mean_neg_logit_per_drug,
                "mean_pair_diff_per_drug": mean_pair_diff_per_drug,
                "mean_pos_logit": 0.0,
                "mean_neg_logit": 0.0,
                "mean_pair_diff": 0.0,
            }
        return loss

    if reduction == "mean":
        loss = torch.stack(losses).mean()
    else:
        loss = torch.stack(losses).sum()

    if total_pairs > 0:
        mean_pos_logit /= total_pairs
        mean_neg_logit /= total_pairs
        mean_pair_diff /= total_pairs

    if return_details:
        return {
            "loss": loss,
            "num_valid_drugs": int(valid_drug_mask.sum().item()),
            "num_pairs_total": int(total_pairs),
            "loss_per_drug": loss_per_drug,
            "num_pos_per_drug": num_pos_per_drug,
            "num_neg_per_drug": num_neg_per_drug,
            "num_pairs_per_drug": num_pairs_per_drug,
            "valid_drug_mask": valid_drug_mask,
            "mean_pos_logit_per_drug": mean_pos_logit_per_drug,
            "mean_neg_logit_per_drug": mean_neg_logit_per_drug,
            "mean_pair_diff_per_drug": mean_pair_diff_per_drug,
            "mean_pos_logit": mean_pos_logit,
            "mean_neg_logit": mean_neg_logit,
            "mean_pair_diff": mean_pair_diff,
        }
    return loss


def drugwise_pairwise_ranking_loss(
    logits,
    labels,
    mask=None,
    missing_value=-1,
    min_pos=1,
    min_neg=1,
    drug_names=None,
    margin=0.0,
    return_stats=False,
):
    """
    Phase 15-B: Drug-wise pairwise ranking loss on source labeled data.

    For each drug d:
        positive samples: labels[:, d] == 1 (sensitive)
        negative samples: labels[:, d] == 0 (resistant)
        loss_d = mean( softplus( -(pos_logit - neg_logit - margin) ) )

    Final loss = mean(loss_d for valid drugs)

    Args:
        logits:  [B, D]  raw final logits (before sigmoid)
        labels:  [B, D]  binary labels (0 or 1 or missing_value or NaN)
        mask:    [B, D] or None  valid mask (True = valid). If provided, overrides missing_value logic.
        missing_value: scalar indicating missing label. Default -1.
        min_pos: int  minimum positive samples required per drug
        min_neg: int  minimum negative samples required per drug
        drug_names: list of str length D, for diagnostics
        margin: float  ranking margin
        return_stats: bool  if True, return dict with detailed per-drug stats

    Returns:
        loss tensor, or (loss, stats_dict) if return_stats=True.
    """
    import math
    B, D = logits.shape
    device = logits.device

    losses = []
    per_drug_stats = []

    for d in range(D):
        drug_name = drug_names[d] if drug_names is not None and d < len(drug_names) else f"drug_{d}"
        labels_d = labels[:, d]
        logits_d = logits[:, d]

        # Build valid mask
        if mask is not None:
            valid = mask[:, d] if mask.dim() >= 2 else mask
            if valid.dtype != torch.bool:
                valid = valid > 0
        else:
            # Handle missing_value and NaN
            valid = torch.ones(B, dtype=torch.bool, device=device)
            if labels.dtype.is_floating_point:
                valid = valid & (~torch.isnan(labels_d))
            valid = valid & (labels_d != missing_value)

        y_valid = labels_d[valid]
        s_valid = logits_d[valid]

        pos_mask = y_valid > 0.5
        neg_mask = y_valid <= 0.5

        pos_logits = s_valid[pos_mask]
        neg_logits = s_valid[neg_mask]
        num_pos = pos_logits.numel()
        num_neg = neg_logits.numel()

        if num_pos < min_pos or num_neg < min_neg:
            per_drug_stats.append({
                "drug_index": d,
                "drug_name": drug_name,
                "num_valid": int(y_valid.numel()),
                "num_pos": int(num_pos),
                "num_neg": int(num_neg),
                "pair_count": 0,
                "rank_loss": float('nan'),
                "pair_acc": float('nan'),
                "pos_logit_mean": float('nan'),
                "neg_logit_mean": float('nan'),
                "mean_diff": float('nan'),
                "skipped": True,
            })
            continue

        # Pairwise differences: [P, N]
        diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)  # [P, N]
        pair_loss = F.softplus(-(diff - margin))
        rank_loss_d = pair_loss.mean()

        # Pairwise accuracy: fraction of pairs where pos_logit > neg_logit
        with torch.no_grad():
            pair_acc_d = (diff > 0).float().mean().item()
            pos_mean = pos_logits.mean().item()
            neg_mean = neg_logits.mean().item()
            mean_diff = pos_mean - neg_mean

        losses.append(rank_loss_d)

        per_drug_stats.append({
            "drug_index": d,
            "drug_name": drug_name,
            "num_valid": int(y_valid.numel()),
            "num_pos": int(num_pos),
            "num_neg": int(num_neg),
            "pair_count": int(diff.numel()),
            "rank_loss": rank_loss_d.item(),
            "pair_acc": pair_acc_d,
            "pos_logit_mean": pos_mean,
            "neg_logit_mean": neg_mean,
            "mean_diff": mean_diff,
            "skipped": False,
        })

    if len(losses) == 0:
        loss = logits.new_tensor(0.0)
    else:
        loss = torch.stack(losses).mean()

    if return_stats:
        # Aggregate stats
        valid_stats = [s for s in per_drug_stats if not s["skipped"]]
        mean_pair_acc = sum(s["pair_acc"] for s in valid_stats) / len(valid_stats) if valid_stats else float('nan')
        mean_pos_logit = sum(s["pos_logit_mean"] for s in valid_stats) / len(valid_stats) if valid_stats else 0.0
        mean_neg_logit = sum(s["neg_logit_mean"] for s in valid_stats) / len(valid_stats) if valid_stats else 0.0
        mean_diff = sum(s["mean_diff"] for s in valid_stats) / len(valid_stats) if valid_stats else 0.0
        stats = {
            "loss": loss,
            "num_valid_drugs": len(valid_stats),
            "num_skipped_drugs": len(per_drug_stats) - len(valid_stats),
            "mean_pair_acc": mean_pair_acc,
            "mean_pos_logit": mean_pos_logit,
            "mean_neg_logit": mean_neg_logit,
            "mean_diff": mean_diff,
            "per_drug": per_drug_stats,
        }
        return loss, stats
    return loss
