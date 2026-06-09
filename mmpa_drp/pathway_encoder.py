"""
mmpa_drp/pathway_encoder.py
===========================
Hallmark gene-set mask defined program encoders.

Replaces or augments latent MolecularProgramEncoder with biologically-defined
tokens based on MSigDB Hallmark gene sets.

Classes:
- GeneSetMaskProgramEncoder: hard/soft gene-set mask encoder
- HybridGeneSetLatentProgramEncoder: concatenates gene-set + latent tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneSetMaskProgramEncoder(nn.Module):
    """
    Hallmark gene-set mask defined program encoder.

    Input:  x [B, G]
    Output: dict with z [B, P, H], q [B, P], h [B, H]

    Each program token corresponds to a pre-defined gene set (e.g., DNA_REPAIR).
    Gene contributions are weighted by expression and aggregated via a
    row-normalized gene-set mask.
    """

    def __init__(
        self,
        input_dim: int,
        gene_set_mask: torch.Tensor,
        program_dim: int = 64,
        dropout: float = 0.1,
        use_pathway_embedding: bool = True,
        use_soft_mask: bool = False,
        soft_mask_scale: float = 0.05,
        program_names: list = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.program_dim = program_dim
        self.num_programs = gene_set_mask.size(0)
        self.dropout = dropout
        self.use_pathway_embedding = use_pathway_embedding
        self.use_soft_mask = use_soft_mask
        self.soft_mask_scale = soft_mask_scale
        self.program_names = program_names or [f"GENE_SET_{i}" for i in range(self.num_programs)]

        assert gene_set_mask.ndim == 2, (
            f"gene_set_mask must be 2D [P, G], got shape {gene_set_mask.shape}"
        )
        assert gene_set_mask.size(1) == input_dim, (
            f"Gene-set mask gene dim mismatch: "
            f"mask {gene_set_mask.shape[1]} vs input {input_dim}"
        )

        # Hard mask: fixed, non-trainable
        self.register_buffer("gene_set_mask", gene_set_mask.float())

        # Optional soft mask for fine-tuning gene set boundaries
        if use_soft_mask:
            self.soft_mask = nn.Parameter(torch.zeros_like(self.gene_set_mask))
        else:
            self.soft_mask = None

        # Learnable gene embeddings [G, H]
        self.gene_emb = nn.Parameter(torch.randn(input_dim, program_dim) * 0.02)

        # Optional pathway bias [P, H]
        if use_pathway_embedding:
            self.pathway_emb = nn.Parameter(torch.randn(self.num_programs, program_dim) * 0.02)
        else:
            self.pathway_emb = None

        # Shared token MLP
        self.token_mlp = nn.Sequential(
            nn.Linear(program_dim, program_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(program_dim, program_dim),
            nn.LayerNorm(program_dim),
        )

        # Activation projection: token -> scalar weight
        self.activation_proj = nn.Linear(program_dim, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, domain_idx=None):
        """
        Args:
            x: [B, G]
            domain_idx: ignored (kept for interface compatibility)
        Returns:
            dict with keys "z", "q", "h", "hidden", "q_logits"
        """
        B, G = x.shape
        P = self.num_programs
        H = self.program_dim

        # Build effective mask
        if self.use_soft_mask and self.soft_mask is not None:
            effective_mask = (
                self.gene_set_mask
                + self.soft_mask_scale * torch.sigmoid(self.soft_mask)
            )
        else:
            effective_mask = self.gene_set_mask

        # Row-normalize: each gene set sums to 1
        mask_norm = effective_mask / (effective_mask.sum(dim=1, keepdim=True) + 1e-8)  # [P, G]

        # Expression-weighted gene embeddings
        # gene_emb: [G, H] -> [1, G, H]
        # x: [B, G] -> [B, G, 1]
        gene_value_emb = x.unsqueeze(-1) * self.gene_emb.unsqueeze(0)  # [B, G, H]

        # Aggregate by gene sets: [P, G] @ [B, G, H] -> [B, P, H]
        z = torch.einsum("pg,bgh->bph", mask_norm, gene_value_emb)

        # Add pathway bias
        if self.pathway_emb is not None:
            z = z + self.pathway_emb.unsqueeze(0)  # [B, P, H]

        # Shared token MLP
        z = self.token_mlp(z)  # [B, P, H]

        # Activation weights
        q_logits = self.activation_proj(z).squeeze(-1)  # [B, P]
        q = F.softmax(q_logits, dim=1)  # [B, P]

        # Pooled representation
        h = (q.unsqueeze(-1) * z).sum(dim=1)  # [B, H]

        return {
            "z": z,
            "q": q,
            "h": h,
            "hidden": h,
            "q_logits": q_logits,
        }


class GeneSetSemanticSelectorEncoder(nn.Module):
    """
    Gene-set guided semantic selector encoder.

    Flow:
        x [B, G] -> GeneSetMaskProgramEncoder -> Zg [B, P, H] (P=16)
        -> cross-attn selector (K=3 queries attend to P=16 gene-set tokens)
        -> z [B, K, H] -> q [B, K] -> h [B, H]

    The K final program tokens are learned combinations of the P gene-set
    semantic tokens, preserving interpretability via selector_attn.
    """

    def __init__(
        self,
        input_dim: int,
        gene_set_mask: torch.Tensor,
        program_dim: int = 64,
        num_final_programs: int = 3,
        dropout: float = 0.1,
        use_pathway_embedding: bool = True,
        use_soft_mask: bool = False,
        soft_mask_scale: float = 0.05,
        selector_num_heads: int = 4,
        selector_dropout: float = 0.1,
        program_names: list = None,
        use_query_gate: bool = False,
        lambda_query_gate_l1: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.program_dim = program_dim
        self.num_programs = num_final_programs
        self.num_gene_sets = gene_set_mask.size(0)
        self.dropout = dropout
        self.use_query_gate = use_query_gate
        self.lambda_query_gate_l1 = lambda_query_gate_l1

        # Gene-set encoder: produces P=16 semantic tokens
        self.gene_set_encoder = GeneSetMaskProgramEncoder(
            input_dim=input_dim,
            gene_set_mask=gene_set_mask,
            program_dim=program_dim,
            dropout=dropout,
            use_pathway_embedding=use_pathway_embedding,
            use_soft_mask=use_soft_mask,
            soft_mask_scale=soft_mask_scale,
            program_names=program_names,
        )

        # Learnable queries: K final program tokens
        self.program_queries = nn.Parameter(
            torch.randn(num_final_programs, program_dim) * 0.02
        )

        # Query gate: learnable per-query activation (for dynamic K)
        if use_query_gate:
            self.query_gate_logits = nn.Parameter(torch.zeros(num_final_programs))
        else:
            self.query_gate_logits = None

        # Cross-attention selector: queries -> gene-set tokens
        self.selector_attn = nn.MultiheadAttention(
            embed_dim=program_dim,
            num_heads=selector_num_heads,
            dropout=selector_dropout,
            batch_first=True,
        )

        self.selector_norm = nn.LayerNorm(program_dim)

        self.selector_mlp = nn.Sequential(
            nn.Linear(program_dim, program_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(program_dim, program_dim),
            nn.LayerNorm(program_dim),
        )

        self.activation_proj = nn.Linear(program_dim, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.activation_proj.weight)
        nn.init.zeros_(self.activation_proj.bias)

    def forward(self, x, domain_idx=None):
        """
        Args:
            x: [B, G]
            domain_idx: ignored (kept for interface compatibility)
        Returns:
            dict with keys:
                z: [B, K, H]
                q: [B, K]
                h: [B, H]
                hidden: [B, H]
                q_logits: [B, K]
                gene_set_tokens: [B, P, H]
                gene_set_q: [B, P] or None
                selector_attn: [B, K, P]
        """
        # 1. Gene-set encoder
        gene_out = self.gene_set_encoder(x, domain_idx=domain_idx)
        Zg = gene_out["z"]  # [B, P, H]

        # 2. Expand learnable queries
        B = x.size(0)
        queries = self.program_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, H]

        # 2.5 Query gate: learnable per-query activation
        query_gate = None
        query_gate_loss = torch.tensor(0.0, device=x.device)
        if self.use_query_gate and self.query_gate_logits is not None:
            query_gate = torch.sigmoid(self.query_gate_logits)  # [K]
            queries = query_gate.unsqueeze(0).unsqueeze(-1) * queries  # [B, K, H]
            query_gate_loss = self.lambda_query_gate_l1 * query_gate.sum()

        # 3. Cross-attention: queries attend to gene-set tokens
        selected, attn_weights = self.selector_attn(
            query=queries,
            key=Zg,
            value=Zg,
            need_weights=True,
            average_attn_weights=False,
        )
        # selected: [B, K, H]
        # attn_weights: [B, num_heads, K, P]
        selector_attn = attn_weights.mean(dim=1)  # [B, K, P]

        # 4. Residual + norm
        z = self.selector_norm(selected + queries)  # [B, K, H]

        # 5. MLP refinement + residual
        z = z + self.selector_mlp(z)
        z = self.selector_norm(z)

        # 6. Activation weights
        q_logits = self.activation_proj(z).squeeze(-1)  # [B, K]
        q = F.softmax(q_logits, dim=1)  # [B, K]

        # 7. Pooled representation
        h = (q.unsqueeze(-1) * z).sum(dim=1)  # [B, H]

        return {
            "z": z,
            "q": q,
            "h": h,
            "hidden": h,
            "q_logits": q_logits,
            "gene_set_tokens": Zg,
            "gene_set_q": gene_out.get("q", None),
            "selector_attn": selector_attn,
            "query_gate": query_gate,
            "query_gate_loss": query_gate_loss,
        }


class HybridGeneSetLatentProgramEncoder(nn.Module):
    """
    Hybrid encoder: concatenates gene-set tokens and latent tokens,
    then re-computes unified activation weights over the combined token pool.
    """

    def __init__(
        self,
        gene_set_encoder: GeneSetMaskProgramEncoder,
        latent_encoder: nn.Module,
        program_dim: int = 64,
    ):
        super().__init__()
        self.gene_set_encoder = gene_set_encoder
        self.latent_encoder = latent_encoder
        self.program_dim = program_dim
        self.num_programs = gene_set_encoder.num_programs + latent_encoder.num_programs

        # Build program names: gene set names + latent token names
        gene_names = getattr(gene_set_encoder, 'program_names', [f"GENE_SET_{i}" for i in range(gene_set_encoder.num_programs)])
        latent_names = [f"LATENT_{i}" for i in range(latent_encoder.num_programs)]
        self.program_names = gene_names + latent_names

        self.activation_proj = nn.Linear(program_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.activation_proj.weight)
        nn.init.zeros_(self.activation_proj.bias)

    def forward(self, x, domain_idx=None):
        """
        Args:
            x: [B, G]
            domain_idx: passed through to sub-encoders
        Returns:
            dict with keys "z", "q", "h", "hidden", "z_gene_set", "z_latent", "q_logits"
        """
        # Gene-set branch
        gene_out = self.gene_set_encoder(x, domain_idx=domain_idx)
        z_gene = gene_out["z"]  # [B, P_gene, H]

        # Latent branch
        latent_out = self.latent_encoder(x, domain_idx=domain_idx)
        z_latent = latent_out["z"]  # [B, P_latent, H]

        # Concatenate in token dimension
        z = torch.cat([z_gene, z_latent], dim=1)  # [B, P_total, H]

        # Re-compute unified activation over ALL tokens
        q_logits = self.activation_proj(z).squeeze(-1)  # [B, P_total]
        q = F.softmax(q_logits, dim=1)  # [B, P_total]

        # Pooled representation
        h = (q.unsqueeze(-1) * z).sum(dim=1)  # [B, H]

        return {
            "z": z,
            "q": q,
            "h": h,
            "hidden": h,
            "z_gene_set": z_gene,
            "z_latent": z_latent,
            "q_logits": q_logits,
        }
