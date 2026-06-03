"""
Phase 5: ConvNeXt V2-inspired Backbones for MolecularProgramEncoder
====================================================================
Backbones:
- GRNMLPBackbone: MLP with GRN (Global Response Normalization)
- ConvNeXt1DBackbone: 1D ConvNeXt V2 blocks with depthwise conv + GRN

GRN (Global Response Normalization) from ConvNeXt V2:
https://arxiv.org/abs/2301.00808
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN1DVector(nn.Module):
    """
    GRN for vector input [B, C].
    When gamma=beta=0, behaves as identity (residual path only).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x: torch.Tensor):
        # x: [B, C]
        gx = torch.norm(x, p=2, dim=1, keepdim=True)  # [B, 1]
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)  # [B, 1]
        out = self.gamma * (x * nx) + self.beta + x  # [B, C]
        return out


class GRN1DToken(nn.Module):
    """
    GRN for token sequence input [B, L, C].
    When gamma=beta=0, behaves as identity.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor):
        # x: [B, L, C]
        gx = torch.norm(x, p=2, dim=1, keepdim=True)  # [B, 1, C]
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)  # [B, 1, C]
        out = self.gamma * (x * nx) + self.beta + x  # [B, L, C]
        return out


class GRNMLPBackbone(nn.Module):
    """
    2-layer MLP with GRN1DVector normalization.
    Default: LayerNorm + GELU (stable for small batches).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.norm_type = norm_type

        # Block 1
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.norm1 = self._build_norm(hidden_dim)
        self.grn1 = GRN1DVector(hidden_dim)
        self.drop1 = nn.Dropout(dropout)

        # Block 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.norm2 = self._build_norm(hidden_dim)
        self.grn2 = GRN1DVector(hidden_dim)
        self.drop2 = nn.Dropout(dropout)

        self._reset_parameters()

    def _build_norm(self, dim):
        if self.norm_type == "layernorm":
            return nn.LayerNorm(dim)
        elif self.norm_type == "batchnorm":
            return nn.BatchNorm1d(dim)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # Block 1
        h = self.fc1(x)
        if isinstance(self.norm1, nn.BatchNorm1d):
            h = self.norm1(h)
        else:
            h = self.norm1(h)
        h = F.gelu(h)
        h = self.grn1(h)
        h = self.drop1(h)

        # Block 2
        h = self.fc2(h)
        if isinstance(self.norm2, nn.BatchNorm1d):
            h = self.norm2(h)
        else:
            h = self.norm2(h)
        h = F.gelu(h)
        h = self.grn2(h)
        h = self.drop2(h)

        return h


class ConvNeXt1DBlock(nn.Module):
    """
    ConvNeXt V2 block adapted for 1D sequences.
    Structure:
      residual = x
      x = DepthwiseConv1d(x)
      x = LayerNorm
      x = Linear(expansion)
      x = GELU
      x = GRN1DToken
      x = Linear(contract)
      x = Dropout
      return residual + x
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.expansion = expansion

        # Depthwise separable conv1d
        self.dwconv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )

        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.grn = GRN1DToken(expansion * dim)
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # x: [B, L, C]
        residual = x

        # Depthwise conv: [B, L, C] -> [B, C, L] -> [B, C, L] -> [B, L, C]
        h = x.transpose(1, 2)  # [B, C, L]
        h = self.dwconv(h)     # [B, C, L]
        h = h.transpose(1, 2)  # [B, L, C]

        h = self.norm(h)
        h = self.pwconv1(h)    # [B, L, expansion*C]
        h = self.act(h)
        h = self.grn(h)
        h = self.pwconv2(h)    # [B, L, C]
        h = self.drop(h)

        return residual + h


class ConvNeXt1DBackbone(nn.Module):
    """
    ConvNeXt V2-inspired 1D backbone for gene expression.
    Uses learned tokenizer to project [B, G] -> [B, L, C] tokens,
    then applies ConvNeXt1D blocks, then mean-pools to [B, C].
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_gene_tokens: int = 64,
        token_dim: int = 64,
        convnext_depth: int = 2,
        convnext_kernel_size: int = 7,
        convnext_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gene_tokens = num_gene_tokens
        self.token_dim = token_dim
        self.convnext_depth = convnext_depth

        # Learned tokenizer: project gene vector to token sequence
        self.tokenizer = nn.Linear(input_dim, num_gene_tokens * token_dim, bias=True)

        # ConvNeXt blocks
        self.blocks = nn.ModuleList([
            ConvNeXt1DBlock(
                dim=token_dim,
                kernel_size=convnext_kernel_size,
                expansion=convnext_expansion,
                dropout=dropout,
            )
            for _ in range(convnext_depth)
        ])

        # Final norm + optional projection
        self.final_norm = nn.LayerNorm(token_dim)
        if token_dim != hidden_dim:
            self.proj = nn.Linear(token_dim, hidden_dim, bias=True)
        else:
            self.proj = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # x: [B, G]
        B = x.size(0)

        # Tokenize: [B, G] -> [B, num_gene_tokens * token_dim] -> [B, L, C]
        tokens = self.tokenizer(x)
        tokens = tokens.view(B, self.num_gene_tokens, self.token_dim)

        # ConvNeXt blocks
        h = tokens
        for block in self.blocks:
            h = block(h)

        # Pool over token dimension
        h = h.mean(dim=1)  # [B, token_dim]
        h = self.final_norm(h)
        h = self.proj(h)   # [B, hidden_dim]

        return h


class ResidualBlock(nn.Module):
    """
    Original MACB ResidualBlock.
    Extracted from pretraining.py for use as a backbone in MolecularProgramEncoder.
    """

    def __init__(self, in_features, out_features, drop=0.2):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SELU(),
            nn.Dropout(drop)
        )

        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        out += residual
        return F.selu(out)


class ResNetBackbone(nn.Module):
    """
    Original MACB ResNet backbone for ablation study.
    Mirrors the feature extraction path of ResNetTissueClassifier
    (input_layer + res_blocks) without the attention/classifier heads.

    Default hidden_dims=[512, 256, 128, 64] matches original MACB.
    Output: [B, hidden_dims[-1]]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=None,
        drop: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        self.hidden_dims = hidden_dims

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SELU(),
            nn.Dropout(drop)
        )

        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1], drop)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # x: [B, G]
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        # h: [B, hidden_dims[-1]]
        return h
