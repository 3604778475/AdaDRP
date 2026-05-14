# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from utility import uniform
import config
device = config.CUDA_ID
import torch.nn.functional as F

def contrastive_loss(anchor, pos_samples, neg_samples, temperature=0.1):
    """
    anchor: [B, D]
    pos_samples: [B, K, D]
    neg_samples: [M, D]
    """
    K = pos_samples.shape[1]

    pos_sim = torch.bmm(pos_samples, anchor.unsqueeze(2)).squeeze(2) / temperature  # [B, K]
    neg_sim = torch.matmul(anchor, neg_samples.T) / temperature  # [B, M]

    logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, K+M]
    labels = torch.zeros_like(logits)
    labels[:, :K] = 1

    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return loss



