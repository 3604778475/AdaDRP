"""
DrugFragmentEncoder: encode fragment fingerprints into fragment embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugFragmentEncoder(nn.Module):
    """
    Encode fragment fingerprints into hidden embeddings.

    Input:
        frag_fp:   [D, M, F]
        frag_mask: [D, M]
    Output:
        frag_emb:  [D, M, H]
    """

    def __init__(self, fp_dim=1024, hidden_dim=64, dropout=0.1):
        super(DrugFragmentEncoder, self).__init__()
        self.fp_dim = fp_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, frag_fp, frag_mask):
        """
        Args:
            frag_fp:   [D, M, F]
            frag_mask: [D, M]
        Returns:
            frag_emb:  [D, M, H]
        """
        D, M, F = frag_fp.shape
        x = frag_fp.view(D * M, F)
        x = self.mlp(x)
        frag_emb = x.view(D, M, self.hidden_dim)
        # Zero out padding fragments
        frag_emb = frag_emb * frag_mask.unsqueeze(-1)
        return frag_emb
