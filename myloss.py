# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from utility import uniform
import config
device = config.CUDA_ID
import torch.nn.functional as F

def contrastive_loss(anchor, pos_samples, neg_samples, temperature=0.1, anchor_weights=None):
    """
    Legacy multi-label BCE contrastive loss.
    anchor: [B, D]
    pos_samples: [B, K, D]
    neg_samples: [M, D]
    anchor_weights: [B] optional per-anchor weights
    """
    K = pos_samples.shape[1]

    pos_sim = torch.bmm(pos_samples, anchor.unsqueeze(2)).squeeze(2) / temperature  # [B, K]
    neg_sim = torch.matmul(anchor, neg_samples.T) / temperature  # [B, M]

    logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, K+M]
    labels = torch.zeros_like(logits)
    labels[:, :K] = 1

    if anchor_weights is not None:
        # Per-anchor weighted BCE
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')  # [B, K+M]
        loss = loss.mean(dim=1)  # [B]
        loss = (loss * anchor_weights).mean() / anchor_weights.mean()
    else:
        loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss


def contrastive_loss_infonce(anchor, pos_samples, neg_samples, temperature=0.1,
                              pos_weights=None, hard_negatives_only=False, num_hard_negs=50,
                              anchor_weights=None):
    """
    Standard InfoNCE contrastive loss (replaces legacy BCE).

    Args:
        anchor: [B, D]
        pos_samples: [B, K, D]
        neg_samples: [M, D]
        temperature: scalar
        pos_weights: [B, K] optional similarity weights for positive samples
        hard_negatives_only: if True, only use top-N hardest negatives
        num_hard_negs: number of hard negatives to keep
        anchor_weights: [B] optional per-anchor weights for differentiated contrastive learning
    Returns:
        loss: scalar
    """
    B, K, D = pos_samples.shape
    M = neg_samples.shape[0]

    # Positive similarities: [B, K]
    pos_sim = torch.bmm(pos_samples, anchor.unsqueeze(2)).squeeze(2) / temperature

    # Negative similarities: [B, M]
    neg_sim = torch.matmul(anchor, neg_samples.T) / temperature

    # Hard negative mining: keep only top-N hardest negatives per anchor
    if hard_negatives_only and M > num_hard_negs:
        # Select top-N highest similarity negatives (most confusing)
        hard_neg_sim, _ = torch.topk(neg_sim, num_hard_negs, dim=1, largest=True, sorted=False)
    else:
        hard_neg_sim = neg_sim

    # Pool positive similarities: logsumexp over K positives
    if pos_weights is not None:
        pos_sim_weighted = pos_sim + torch.log(pos_weights.clamp(min=1e-6))
    else:
        pos_sim_weighted = pos_sim

    pos_logsumexp = torch.logsumexp(pos_sim_weighted, dim=1)  # [B]

    # Denominator
    neg_logsumexp = torch.logsumexp(hard_neg_sim, dim=1)  # [B]

    denominator = torch.logaddexp(pos_logsumexp, neg_logsumexp)
    loss = -pos_logsumexp + denominator  # [B]

    if anchor_weights is not None:
        loss = (loss * anchor_weights).mean() / anchor_weights.mean()
    else:
        loss = loss.mean()

    return loss



