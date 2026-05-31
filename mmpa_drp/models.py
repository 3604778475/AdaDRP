"""
MMPA-DRP Extension Models
=========================
Phase 1 minimal implementations for:
- MolecularProgramEncoder
- DrugMechanismGate
- DrugGatedProgramAggregator

These modules are designed to replace the single gene representation h:[B,d] in
MACB-DRP with a set of K molecular program tokens z:[B,K,d] and a drug-specific
gating mechanism.

Phase 5: Added backbone_type support (mlp / grn_mlp / convnext1d)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import config

from mmpa_drp.backbones import GRNMLPBackbone, ConvNeXt1DBackbone
from mmpa_drp.fragment_encoder import DrugFragmentEncoder
from mmpa_drp.fpi import FragmentProgramInteraction


class MolecularProgramEncoder(nn.Module):
    """
    Encode a gene expression vector into K molecular program tokens.

    Input:
        x: [B, G]  (batch of gene expression vectors)
    Output:
        z: [B, K, d]  (K program tokens per sample)
        q: [B, K]     (activation weights over programs, softmax-normalized)
        h: [B, d]     (pooled representation = sum_k q_k * z_k)

    Phase 5: Supports multiple backbone types:
        - "mlp": original MLP backbone (Phase 2 compatible)
        - "grn_mlp": GRN-enhanced 2-layer MLP
        - "convnext1d": ConvNeXt V2-inspired 1D backbone
    """

    def __init__(
        self,
        input_dim: int,
        num_programs: int = 8,
        program_dim: int = 64,
        hidden_dims=None,
        drop: float = 0.1,
        act_fn=nn.SELU,
        # Phase 5 backbone selection
        backbone_type: str = "mlp",
        # GRN-MLP specific
        norm_type: str = "layernorm",
        # ConvNeXt1D specific
        num_gene_tokens: int = 64,
        token_dim: int = 64,
        convnext_depth: int = 2,
        convnext_kernel_size: int = 7,
        convnext_expansion: int = 4,
        # Phase 18: domain separation
        use_domain_sep: bool = False,
        domain_sep_hidden_dim: int = 256,
        domain_sep_output_dim: int = None,
    ):
        super(MolecularProgramEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_programs = num_programs
        self.program_dim = program_dim
        self.backbone_type = backbone_type

        # ------------------------------------------------------------------
        # Backbone selection (Phase 5)
        # ------------------------------------------------------------------
        if backbone_type == "mlp":
            if hidden_dims is None:
                hidden_dims = [512, 256, 128]
            hidden_dims = deepcopy(hidden_dims)
            hidden_dims.insert(0, input_dim)

            modules = []
            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                        act_fn(),
                        nn.BatchNorm1d(hidden_dims[i + 1]),
                        nn.Dropout(drop),
                    )
                )
            self.backbone = nn.Sequential(*modules)
            self.backbone_output_dim = hidden_dims[-1]

        elif backbone_type == "grn_mlp":
            if hidden_dims is None:
                hidden_dims = [512, 256, 128]
            # For GRN-MLP we use the last hidden_dim as the backbone output dim.
            # If hidden_dims has multiple layers, we keep the original structure
            # but replace with GRN-enhanced blocks.
            # Simplified: use a 2-layer GRNMLP with hidden_dim = hidden_dims[-1]
            grn_hidden = hidden_dims[-1] if hidden_dims else 64
            self.backbone = GRNMLPBackbone(
                input_dim=input_dim,
                hidden_dim=grn_hidden,
                dropout=drop,
                norm_type=norm_type,
            )
            self.backbone_output_dim = grn_hidden

        elif backbone_type == "convnext1d":
            # ConvNeXt uses token_dim internally; final proj maps to hidden_dim
            grn_hidden = hidden_dims[-1] if hidden_dims else 64
            self.backbone = ConvNeXt1DBackbone(
                input_dim=input_dim,
                hidden_dim=grn_hidden,
                num_gene_tokens=num_gene_tokens,
                token_dim=token_dim,
                convnext_depth=convnext_depth,
                convnext_kernel_size=convnext_kernel_size,
                convnext_expansion=convnext_expansion,
                dropout=drop,
            )
            self.backbone_output_dim = grn_hidden

        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # Project hidden state to K * d program tokens
        self.program_proj = nn.Linear(self.backbone_output_dim, num_programs * program_dim, bias=True)
        # Project hidden state to K activation logits
        self.activation_proj = nn.Linear(self.backbone_output_dim, num_programs, bias=True)

        # Phase 18: domain-separated encoder
        self.use_domain_sep = use_domain_sep
        if use_domain_sep:
            self.domain_sep = DomainSeparatedEncoder(
                input_dim=self.backbone_output_dim,
                hidden_dim=domain_sep_hidden_dim,
                output_dim=domain_sep_output_dim or program_dim,
                num_domains=2,
                drop=drop,
            )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, domain_idx: torch.Tensor = None):
        """
        Args:
            x: [B, G]
            domain_idx: [B] int tensor in {0, 1}, optional. Used to select
                        private branch when use_domain_sep=True.
        Returns:
            dict with keys:
                z: [B, K, d]  program tokens
                q: [B, K]     activation weights (softmax)
                h: [B, d]     pooled representation = sum_k q_k * z_k
                hidden: [B, H] backbone output (Phase 18)
                h_shared: [B, d_sep] shared repr (Phase 18, if enabled)
                h_private: [B, d_sep] private repr (Phase 18, if enabled)
                domain_logits: [B, 2] domain classifier logits (Phase 18, if enabled)
        """
        hidden = self.backbone(x)  # [B, H]

        # Program tokens
        z_flat = self.program_proj(hidden)  # [B, K * d]
        z = z_flat.view(-1, self.num_programs, self.program_dim)  # [B, K, d]

        # Activation weights
        q_logits = self.activation_proj(hidden)  # [B, K]
        q = F.softmax(q_logits, dim=1)  # [B, K]

        # Pooled representation (drop-in replacement for original h)
        h_pool = (q.unsqueeze(-1) * z).sum(dim=1)  # [B, d]

        result = {"z": z, "q": q, "h": h_pool, "hidden": hidden}

        if self.use_domain_sep:
            h_shared, h_private, domain_logits = self.domain_sep(hidden, domain_idx=domain_idx)
            result["h_shared"] = h_shared
            result["h_private"] = h_private
            result["domain_logits"] = domain_logits

        return result


class DomainSeparatedEncoder(nn.Module):
    """
    Phase 18: DCDNet-style domain-level shared/private decomposition.

    Given backbone hidden state [B, H], produce:
      - h_shared: [B, output_dim]   cross-domain shared representation
      - h_private: [B, output_dim]  domain-specific representation
      - domain_logits: [B, num_domains]  domain classifier logits

    Two private MLPs (one per domain) are learned. During forward,
    domain_idx determines which private branch to activate per sample.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_domains: int = 2,
        drop: float = 0.1,
    ):
        super(DomainSeparatedEncoder, self).__init__()
        self.num_domains = num_domains

        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.private_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SELU(),
                nn.Dropout(drop),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            ) for _ in range(num_domains)
        ])

        self.domain_gate = nn.Linear(input_dim, num_domains)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hidden: torch.Tensor, domain_idx: torch.Tensor = None):
        """
        Args:
            hidden: [B, input_dim]
            domain_idx: [B] int tensor in {0, 1, ...}, or None
        Returns:
            h_shared: [B, output_dim]
            h_private: [B, output_dim]
            domain_logits: [B, num_domains]
        """
        h_shared = self.shared_mlp(hidden)
        domain_logits = self.domain_gate(hidden)

        B = hidden.size(0)
        output_dim = h_shared.size(1)

        if domain_idx is not None:
            h_private = torch.zeros(B, output_dim, device=hidden.device, dtype=hidden.dtype)
            for d in range(self.num_domains):
                mask = (domain_idx == d)
                if mask.any():
                    h_private[mask] = self.private_mlps[d](hidden[mask])
        else:
            # When domain unknown, average all private branches
            h_private = sum(m(hidden) for m in self.private_mlps) / self.num_domains

        return h_shared, h_private, domain_logits


class PseudoDomainGuide(nn.Module):
    """
    Phase 10: Pseudo-Domain Guided Fusion.

    Input:
        h_pool: [B, H]  (gene representation)
        centroids: [K, H]  (precomputed pseudo-domain centroids)
    Output dict:
        pdg_emb: [B, pdg_dim]  (projected centroid embedding)
        assign: [B]  (hard nearest centroid assignment)
        sim: [B, K]  (cosine similarities)
    """

    def __init__(
        self,
        centroid_dim: int,
        pdg_dim: int = 32,
        dropout: float = 0.1,
        assign_mode: str = "nearest",
        normalize: bool = True,
    ):
        super(PseudoDomainGuide, self).__init__()
        self.assign_mode = assign_mode
        self.normalize = normalize
        self.proj = nn.Sequential(
            nn.Linear(centroid_dim, pdg_dim),
            nn.LayerNorm(pdg_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_pool, centroids):
        if centroids is None:
            raise ValueError("PDG centroids are required when use_pdg=True")

        h = h_pool
        c = centroids

        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)
            c = F.normalize(c, p=2, dim=-1)

        sim = torch.matmul(h, c.t())  # [B, K]
        assign = sim.argmax(dim=1)    # [B]

        with torch.no_grad():
            c_sel = centroids[assign]  # [B, H]

        pdg_emb = self.proj(c_sel)  # [B, pdg_dim]

        return {
            "pdg_emb": pdg_emb,
            "assign": assign,
            "sim": sim,
        }


class DrugMechanismGate(nn.Module):
    """
    Produce a drug-specific mechanism gate over K molecular programs.

    Input:
        drug_embedding: [D, d]  (drug representations, e.g. from Drug_MLP)
    Output:
        g: [D, K]  (g[d,k] = dependency of drug d on program k)
    """

    def __init__(
        self,
        drug_dim: int,
        num_programs: int = 8,
        hidden_dim: int = 128,
        drop: float = 0.1,
        act_fn=nn.SELU,
    ):
        super(DrugMechanismGate, self).__init__()
        self.drug_dim = drug_dim
        self.num_programs = num_programs

        self.mlp = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim, bias=True),
            act_fn(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_programs, bias=True),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_embedding: torch.Tensor, return_logits=False):
        """
        Args:
            drug_embedding: [D, d]
            return_logits: if True, return raw logits before softmax
        Returns:
            g: [D, K] softmax probabilities, or logits if return_logits=True
        """
        logits = self.mlp(drug_embedding)  # [D, K]
        if return_logits:
            return logits
        g = F.softmax(logits, dim=1)  # [D, K]
        return g


class ProgramTissueClassifier(nn.Module):
    """
    Stage 1 drop-in replacement for ResNetTissueClassifier.
    Uses MolecularProgramEncoder to produce h_pool, then applies
    tissue classification and domain discrimination on h_pool.

    Interface matches ResNetTissueClassifier:
        forward(x, alpha=1.0) -> (tissue_pred, domain_pred, gene_feat)
    where gene_feat == h_pool.
    """

    def __init__(
        self,
        input_dim: int,
        num_tissues: int,
        num_programs: int = 8,
        program_dim: int = 64,
        hidden_dims=None,
        drop: float = 0.2,
        act_fn=nn.SELU,
        # Phase 5 backbone args
        backbone_type: str = "mlp",
        norm_type: str = "layernorm",
        num_gene_tokens: int = 64,
        token_dim: int = 64,
        convnext_depth: int = 2,
        convnext_kernel_size: int = 7,
        convnext_expansion: int = 4,
        # Phase 18: domain separation
        use_domain_sep: bool = False,
        domain_sep_hidden_dim: int = 256,
        domain_sep_output_dim: int = None,
    ):
        super(ProgramTissueClassifier, self).__init__()
        self.program_encoder = MolecularProgramEncoder(
            input_dim=input_dim,
            num_programs=num_programs,
            program_dim=program_dim,
            hidden_dims=hidden_dims,
            drop=drop,
            act_fn=act_fn,
            backbone_type=backbone_type,
            norm_type=norm_type,
            num_gene_tokens=num_gene_tokens,
            token_dim=token_dim,
            convnext_depth=convnext_depth,
            convnext_kernel_size=convnext_kernel_size,
            convnext_expansion=convnext_expansion,
            use_domain_sep=use_domain_sep,
            domain_sep_hidden_dim=domain_sep_hidden_dim,
            domain_sep_output_dim=domain_sep_output_dim,
        )
        self.use_domain_sep = use_domain_sep
        feat_dim = (domain_sep_output_dim or program_dim) if use_domain_sep else program_dim

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.BatchNorm1d(feat_dim // 2),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(feat_dim // 2, num_tissues),
        )

        self.domain_discriminator = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.BatchNorm1d(feat_dim // 2),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(feat_dim // 2, 1),
        )

        self.private_domain_discriminator = None
        if use_domain_sep:
            self.private_domain_discriminator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.BatchNorm1d(feat_dim // 2),
                nn.SELU(),
                nn.Dropout(drop),
                nn.Linear(feat_dim // 2, 1),
            )

        self.last_domain_sep_out = None

    def forward(self, x, alpha=1.0, domain_idx=None):
        out = self.program_encoder(x, domain_idx=domain_idx)

        if self.use_domain_sep:
            h = out["h_shared"]  # [B, d_sep]
            self.last_domain_sep_out = {
                "h_shared": out["h_shared"],
                "h_private": out["h_private"],
                "domain_logits": out["domain_logits"],
                "hidden": out["hidden"],
            }
        else:
            h = out["h"]  # [B, d]
            self.last_domain_sep_out = None

        tissue_pred = self.classifier(h)
        reversed_h = GradientReversal.apply(h, alpha)
        domain_pred = self.domain_discriminator(reversed_h)
        return tissue_pred, domain_pred, h


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class SharedDrugGate(nn.Module):
    """
    Shared drug gate: all drugs use the same program gate.
    Output shape [1, K] broadcasts to [D, K] in DrugGatedProgramAggregator.
    """
    def __init__(self, num_programs=8):
        super(SharedDrugGate, self).__init__()
        self.gate = nn.Parameter(torch.ones(num_programs) / num_programs)

    def forward(self, drug_embedding=None):
        return self.gate.unsqueeze(0)  # [1, K]


# =============================================================================
# Phase 8: DrugSelectiveProgramInteraction (DSPI)
# =============================================================================

class DrugSelectiveProgramInteraction(nn.Module):
    """
    Signal-inspired Drug-Selective Program Interaction.

    Given MolecularProgramEncoder outputs and drug embeddings, learns
    drug-specific program attention via intra-drug and inter-drug attention.

    Input:
        z: [B, K, H]   program tokens
        q: [B, K]      base program weights
        h: [B, H]      pooled representation
        drug_emb: [D, H]  drug embeddings

    Output dict:
        h_drug: [B, D, H]  residual-preserving drug-specific representation
        h_select: [B, D, H] selected representation (before residual)
        attn: [B, D, K]    final attention weights
        attn_intra: [B, D, K] intra-drug attention
        attn_inter: [B, D, K] inter-drug attention
        drug_sim: [D, D]    drug-drug similarity
        rho: scalar or Tensor
    """

    def __init__(
        self,
        program_dim: int = 64,
        drug_dim: int = 64,
        num_programs: int = 8,
        num_drugs: int = 9,
        temperature: float = 1.0,
        inter_weight: float = 0.2,
        q_prior_weight: float = 0.1,
        drug_sim_temperature: float = 1.0,
        rho_mode: str = "fixed",
        fixed_rho: float = 0.1,
        rho_max: float = 0.3,
        eps: float = 1e-8,
    ):
        super(DrugSelectiveProgramInteraction, self).__init__()
        self.program_dim = program_dim
        self.drug_dim = drug_dim
        self.num_programs = num_programs
        self.num_drugs = num_drugs
        self.temperature = temperature
        self.inter_weight = inter_weight
        self.q_prior_weight = q_prior_weight
        self.drug_sim_temperature = drug_sim_temperature
        self.rho_mode = rho_mode
        self.fixed_rho = fixed_rho
        self.rho_max = rho_max
        self.eps = eps

        # Intra-drug attention projections
        self.W_d = nn.Linear(drug_dim, program_dim, bias=False)
        self.W_z = nn.Linear(program_dim, program_dim, bias=False)

        # Rho parameter
        if rho_mode == "fixed":
            self.register_buffer("rho", torch.tensor(float(fixed_rho)))
        elif rho_mode == "learnable_scalar":
            # Initialize so sigmoid(raw_rho) ≈ init_rho
            import numpy as np
            init_val = -np.log(1.0 / max(fixed_rho, 1e-6) - 1.0) if 0 < fixed_rho < 1 else 0.0
            self.raw_rho = nn.Parameter(torch.tensor(float(init_val)))
        elif rho_mode == "schedule":
            self.register_buffer("rho", torch.tensor(float(fixed_rho)))
        else:
            raise ValueError(f"Unknown rho_mode: {rho_mode}")

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_rho(self):
        """Get effective rho value."""
        if self.rho_mode == "fixed":
            return self.rho
        elif self.rho_mode == "learnable_scalar":
            return torch.sigmoid(self.raw_rho) * self.rho_max
        elif self.rho_mode == "schedule":
            return self.rho
        return self.rho

    def forward(self, z, q, h, drug_emb):
        """
        Args:
            z: [B, K, H]
            q: [B, K]
            h: [B, H]
            drug_emb: [D, H]
        Returns:
            dict with keys: h_drug, h_select, attn, attn_intra, attn_inter, drug_sim, rho
        """
        B, K, H = z.shape
        D = drug_emb.size(0)

        # 2.1 Intra-drug program selection
        drug_q = self.W_d(drug_emb)  # [D, H]
        prog_k = self.W_z(z)         # [B, K, H]

        # score_intra = einsum("dh,bkh->bdk", drug_q, prog_k) / sqrt(H)
        score_intra = torch.einsum("dh,bkh->bdk", drug_q, prog_k) / (H ** 0.5)  # [B, D, K]

        # Add q prior: score_intra = score_intra + q_prior_weight * log(q + eps)
        q_log = torch.log(q + self.eps)  # [B, K]
        score_intra = score_intra + self.q_prior_weight * q_log.unsqueeze(1)  # [B, D, K]

        attn_intra = F.softmax(score_intra / self.temperature, dim=-1)  # [B, D, K]

        # 2.2 Inter-drug program selection
        drug_norm = F.normalize(drug_emb, p=2, dim=-1)  # [D, H]
        drug_sim = torch.mm(drug_norm, drug_norm.t())   # [D, D]

        # Mask diagonal with very negative value
        very_negative = -1e9
        drug_sim_masked = drug_sim - torch.eye(D, device=drug_sim.device) * very_negative
        drug_sim_soft = F.softmax(drug_sim_masked / self.drug_sim_temperature, dim=-1)  # [D, D]

        # attn_inter[b,d,k] = sum_{d' != d} drug_sim[d,d'] * attn_intra[b,d',k]
        attn_inter = torch.einsum("dd,bdk->bdk", drug_sim_soft, attn_intra)  # [B, D, K]

        # 2.3 Soft union
        attn_raw = (1 - self.inter_weight) * attn_intra + self.inter_weight * attn_inter
        attn = attn_raw / (attn_raw.sum(dim=-1, keepdim=True) + self.eps)  # [B, D, K]

        # 2.4 Drug-specific selected representation
        h_select = torch.einsum("bdk,bkh->bdh", attn, z)  # [B, D, H]

        # 2.5 Residual-preserving representation
        h_base = h.unsqueeze(1).expand(-1, D, -1)  # [B, D, H]
        rho_val = self.get_rho()
        h_drug = h_base + rho_val * (h_select - h_base)  # [B, D, H]

        return {
            "h_drug": h_drug,
            "h_select": h_select,
            "attn": attn,
            "attn_intra": attn_intra,
            "attn_inter": attn_inter,
            "drug_sim": drug_sim,
            "rho": rho_val.detach().clone() if isinstance(rho_val, torch.Tensor) else torch.tensor(rho_val),
        }


# =============================================================================
# Phase 8-F: Fusion-Repair DSPI — DrugSelectiveProgramSelector (no inter-drug)
# =============================================================================

class DrugSelectiveProgramSelector(nn.Module):
    """
    Clean drug-selective program selector WITHOUT inter-drug attention.
    Produces h_select[b,d,:] = sum_k attn[b,d,k] * z[b,k,:]
    where attn is driven primarily by drug_emb, with a weak q prior.
    """

    def __init__(
        self,
        program_dim: int = 64,
        drug_dim: int = 64,
        num_programs: int = 8,
        num_drugs: int = 9,
        attention_temperature: float = 1.0,
        q_prior_weight: float = 0.01,  # weaker than DSPI's 0.1
        eps: float = 1e-8,
    ):
        super(DrugSelectiveProgramSelector, self).__init__()
        self.program_dim = program_dim
        self.drug_dim = drug_dim
        self.num_programs = num_programs
        self.num_drugs = num_drugs
        self.attention_temperature = attention_temperature
        self.q_prior_weight = q_prior_weight
        self.eps = eps

        self.drug_proj = nn.Linear(drug_dim, program_dim, bias=False)
        self.prog_proj = nn.Linear(program_dim, program_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, q, h_pool, drug_emb):
        """
        Args:
            z: [B, K, H]
            q: [B, K]
            h_pool: [B, H]
            drug_emb: [D, H]
        Returns:
            dict: h_select [B,D,H], attn [B,D,K], score [B,D,K]
        """
        B, K, H = z.shape
        D = drug_emb.size(0)

        drug_q = self.drug_proj(drug_emb)  # [D, H]
        prog_k = self.prog_proj(z)         # [B, K, H]

        score = torch.einsum("dh,bkh->bdk", drug_q, prog_k) / (H ** 0.5)  # [B, D, K]

        # Weak q prior (configurable, default 0.01 to avoid h_select ≈ h_pool)
        if self.q_prior_weight > 0:
            q_log = torch.log(q + self.eps)  # [B, K]
            score = score + self.q_prior_weight * q_log.unsqueeze(1)  # [B, D, K]

        attn = F.softmax(score / self.attention_temperature, dim=-1)  # [B, D, K]

        h_select = torch.einsum("bdk,bkh->bdh", attn, z)  # [B, D, H]

        return {
            "h_select": h_select,
            "attn": attn,
            "score": score,
        }


# =============================================================================
# Phase 8-F: BasePreservingAdapterFusion (BPAF)
# =============================================================================

class BasePreservingAdapterFusion(nn.Module):
    """
    Base-Preserving Adapter Fusion.

    h_eff_d = h_pool + gamma_d * Adapter([h_select_d, drug_emb_d, h_pool_d])

    gamma_d is per-drug learnable, initialized near 0.
    Adapter receives h_pool so it can compute meaningful deltas.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_drugs: int = 9,
        gamma_max: float = 0.05,
        gamma_init: float = 0.02,  # changed from 0.0 to avoid gamma trap
        dropout: float = 0.1,
        adapter_include_h_pool: bool = True,
    ):
        super(BasePreservingAdapterFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_drugs = num_drugs
        self.gamma_max = gamma_max
        self.gamma_init = gamma_init
        self.dropout = dropout
        self.adapter_include_h_pool = adapter_include_h_pool

        # Adapter MLP
        adapter_input_dim = hidden_dim * 3 if adapter_include_h_pool else hidden_dim * 2
        self.adapter = nn.Sequential(
            nn.Linear(adapter_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Small init on last linear to reduce initial perturbation
        nn.init.normal_(self.adapter[-2].weight, std=1e-3)
        nn.init.zeros_(self.adapter[-2].bias)

        # Per-drug gamma (raw, before sigmoid)
        self.raw_gamma = nn.Parameter(torch.zeros(num_drugs))
        self._init_gamma()

    def _init_gamma(self):
        if self.gamma_init <= 0:
            # Near-zero init: sigmoid(-6) ≈ 0.0025
            init_val = -6.0
        else:
            ratio = self.gamma_init / self.gamma_max
            ratio = max(min(ratio, 0.999), 0.001)
            init_val = math.log(ratio / (1.0 - ratio))
        with torch.no_grad():
            self.raw_gamma.fill_(init_val)

    def get_gamma(self):
        return torch.sigmoid(self.raw_gamma) * self.gamma_max

    def forward(self, h_pool, h_select, drug_emb):
        """
        Args:
            h_pool:   [B, H]
            h_select: [B, D, H]
            drug_emb: [D, H]
        Returns:
            dict: h_eff [B,D,H], h_adapter [B,D,H], gamma [D]
        """
        B, D, H = h_select.shape

        h_pool_expand = h_pool.unsqueeze(1).expand(-1, D, -1)   # [B, D, H]
        drug_expand = drug_emb.unsqueeze(0).expand(B, -1, -1)   # [B, D, H]

        if self.adapter_include_h_pool:
            adapter_input = torch.cat([h_select, drug_expand, h_pool_expand], dim=-1)  # [B, D, 3H]
        else:
            adapter_input = torch.cat([h_select, drug_expand], dim=-1)  # [B, D, 2H]

        h_adapter = self.adapter(adapter_input)  # [B, D, H]

        gamma = self.get_gamma()  # [D]
        h_eff = h_pool_expand + gamma.view(1, D, 1) * h_adapter  # [B, D, H]

        return {
            "h_eff": h_eff,
            "h_adapter": h_adapter,
            "gamma": gamma,
        }


# =============================================================================
# Phase 8-F: LateLogitResidualFusion (LLRF)
# =============================================================================

class LateLogitResidualFusion(nn.Module):
    """
    Late Logit Residual Fusion.

    residual_logits_d = Head([h_select_d, drug_emb_d])
    final_logits_d = base_logits_d + gamma_d * residual_logits_d

    Head outputs unbounded logits (no tanh) for expressiveness.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_drugs: int = 9,
        gamma_max: float = 0.1,
        gamma_init: float = 0.02,
        dropout: float = 0.1,
    ):
        super(LateLogitResidualFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_drugs = num_drugs
        self.gamma_max = gamma_max
        self.gamma_init = gamma_init

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Small init
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

        self.raw_gamma = nn.Parameter(torch.zeros(num_drugs))
        self._init_gamma()

    def _init_gamma(self):
        if self.gamma_init <= 0:
            init_val = -6.0
        else:
            ratio = self.gamma_init / self.gamma_max
            ratio = max(min(ratio, 0.999), 0.001)
            init_val = math.log(ratio / (1.0 - ratio))
        with torch.no_grad():
            self.raw_gamma.fill_(init_val)

    def get_gamma(self):
        return torch.sigmoid(self.raw_gamma) * self.gamma_max

    def forward(self, h_pool, h_select, drug_emb, base_logits):
        """
        Args:
            h_pool:      [B, H]      (not used in head, but kept for interface)
            h_select:    [B, D, H]
            drug_emb:    [D, H]
            base_logits: [B, D]      (before sigmoid)
        Returns:
            dict: final_logits [B,D], residual_logits [B,D], gamma [D]
        """
        B, D, H = h_select.shape
        drug_expand = drug_emb.unsqueeze(0).expand(B, -1, -1)  # [B, D, H]

        head_input = torch.cat([h_select, drug_expand], dim=-1)  # [B, D, 2H]
        residual_logits = self.head(head_input).squeeze(-1)  # [B, D]

        gamma = self.get_gamma()  # [D]
        final_logits = base_logits + gamma.view(1, D) * residual_logits  # [B, D]

        return {
            "final_logits": final_logits,
            "residual_logits": residual_logits,
            "gamma": gamma,
        }


class ResidualDecoupledFusion(nn.Module):
    """
    Phase 10-D: Residual Decoupled Fusion (DCD-R).

    Decomposes h_pool into shared and private components, then computes
    a per-drug residual logit correction applied to base_logits.

    Design (base-preserving residual):
    - shared_mlp  : h_pool -> h_shared   (domain-invariant background)
    - private_mlp : h_pool -> h_private  (response-specific information)
    - residual_head: [h_shared, h_private, drug_emb, (h_private * drug_emb)]
                     -> residual_logits per drug
    - gamma: per-drug sigmoid-scaled scalar (gamma_max=0.03)
    - residual_logits tanh-constrained for stability
    - final_base_logits = base_logits + gamma * residual_logits

    Args:
        hidden_dim: dimension of h_pool (H)
        drug_dim: dimension of drug_emb (H, typically same)
        num_drugs: number of drugs (D)
        dcd_hidden_dim: hidden dim for residual head (default same as hidden_dim)
        gamma_max: max residual scale (default 0.03)
        gamma_init: initial residual scale (default 0.005)
        dropout: dropout rate
        use_product: include h_private * drug_emb element-wise product
        use_absdiff: include abs(h_private - drug_emb) feature
    """

    def __init__(
        self,
        hidden_dim=64,
        drug_dim=64,
        num_drugs=9,
        dcd_hidden_dim=64,
        gamma_max=0.03,
        gamma_init=0.005,
        dropout=0.1,
        use_product=True,
        use_absdiff=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drug_dim = drug_dim
        self.num_drugs = num_drugs
        self.dcd_hidden_dim = dcd_hidden_dim
        self.gamma_max = gamma_max
        self.gamma_init = gamma_init
        self.use_product = use_product
        self.use_absdiff = use_absdiff

        # Shared MLP: two-layer with LayerNorm
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Private MLP: same structure
        self.private_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Residual head input size
        head_input_dim = hidden_dim * 3  # shared + private + drug_emb
        if use_product:
            head_input_dim += hidden_dim  # h_private * drug_emb
        if use_absdiff:
            head_input_dim += hidden_dim  # abs(h_private - drug_emb)

        self.residual_head = nn.Sequential(
            nn.Linear(head_input_dim, dcd_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dcd_hidden_dim, 1),
        )

        # Small init on last layer for near-zero initial residual
        nn.init.normal_(self.residual_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual_head[-1].bias)

        # Per-drug learnable gamma (sigmoid-scaled)
        self.raw_gamma = nn.Parameter(torch.zeros(num_drugs))
        self._init_gamma()

    def _init_gamma(self):
        if self.gamma_init <= 0:
            init_val = -8.0
        else:
            ratio = self.gamma_init / self.gamma_max
            ratio = max(min(ratio, 0.999), 0.001)
            init_val = math.log(ratio / (1.0 - ratio))
        with torch.no_grad():
            self.raw_gamma.fill_(init_val)

    def get_gamma(self):
        return torch.sigmoid(self.raw_gamma) * self.gamma_max

    def forward(self, h_pool, drug_emb, base_logits):
        """
        Args:
            h_pool:      [B, H]
            drug_emb:    [D, H]
            base_logits: [B, D] (before any sigmoid)
        Returns:
            dict with keys:
                final_base_logits: [B, D]
                residual_logits:   [B, D]
                gamma:             [D]
                h_shared:          [B, H]
                h_private:         [B, H]
                dcd_effect_abs_mean: scalar
        """
        B, H = h_pool.shape
        D = self.num_drugs

        # Shared and private representations
        h_shared = self.shared_mlp(h_pool)   # [B, H]
        h_private = self.private_mlp(h_pool)  # [B, H]

        # Expand for per-drug residual computation
        shared_expand = h_shared.unsqueeze(1).expand(-1, D, -1)    # [B, D, H]
        private_expand = h_private.unsqueeze(1).expand(-1, D, -1)  # [B, D, H]
        drug_expand = drug_emb.unsqueeze(0).expand(B, -1, -1)      # [B, D, H]

        # Build residual head input
        head_parts = [shared_expand, private_expand, drug_expand]
        if self.use_product:
            head_parts.append(private_expand * drug_expand)
        if self.use_absdiff:
            head_parts.append(torch.abs(private_expand - drug_expand))

        head_input = torch.cat(head_parts, dim=-1)  # [B, D, head_input_dim]
        residual_logits = self.residual_head(head_input).squeeze(-1)  # [B, D]
        residual_logits = torch.tanh(residual_logits)  # stability constraint

        gamma = self.get_gamma()  # [D]

        # Apply residual correction
        final_base_logits = base_logits + gamma.unsqueeze(0) * residual_logits  # [B, D]

        # Diagnostic: mean absolute effect
        dcd_effect_abs_mean = (gamma.unsqueeze(0) * residual_logits).abs().mean()

        return {
            "final_base_logits": final_base_logits,
            "residual_logits": residual_logits,
            "gamma": gamma,
            "h_shared": h_shared,
            "h_private": h_private,
            "dcd_effect_abs_mean": dcd_effect_abs_mean,
        }


class DrugConditionedLowRankFusion(nn.Module):
    """
    Phase 11: Drug-Conditioned Low-Rank Bilinear Fusion (DCF-LRBF).

    Replaces the static bilinear(h, d) interaction with a drug-conditioned,
    low-rank multiplicative fusion.

    Two modes:
        - "lrbf":     low-rank bilinear only
                      u_lr = W_h(h) * W_d(d)
        - "dcf_lrbf": drug-conditioned FiLM + low-rank bilinear
                      gamma, beta = MLP(drug)
                      h_mod = h * (1 + gamma) + beta
                      u_lr = W_h(h_mod) * W_d(d)

    Args:
        gene_dim: dimension of h_pool
        drug_dim: dimension of drug_emb
        rank_dim: low-rank dimension R
        out_dim:  output fusion feature dimension
        mode: "lrbf" or "dcf_lrbf"
        dropout: dropout rate
        use_layernorm: use LayerNorm in MLPs
        film_init_scale: std for FiLM output layer init (near-identity start)
    """

    def __init__(
        self,
        gene_dim: int,
        drug_dim: int,
        rank_dim: int = 64,
        out_dim: int = 128,
        mode: str = "lrbf",
        dropout: float = 0.1,
        use_layernorm: bool = True,
        film_init_scale: float = 1e-3,
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.drug_dim = drug_dim
        self.rank_dim = rank_dim
        self.out_dim = out_dim
        self.mode = mode

        # Projection layers for low-rank interaction
        self.gene_proj = nn.Linear(gene_dim, rank_dim)
        self.drug_proj = nn.Linear(drug_dim, rank_dim)

        # Optional FiLM modulation for dcf_lrbf mode
        if mode == "dcf_lrbf":
            self.film_mlp = nn.Sequential(
                nn.Linear(drug_dim, drug_dim),
                nn.LayerNorm(drug_dim) if use_layernorm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(drug_dim, drug_dim * 2),
            )
            # Small init for near-identity modulation
            nn.init.normal_(self.film_mlp[-1].weight, std=film_init_scale)
            nn.init.zeros_(self.film_mlp[-1].bias)

        # Final MLP to produce fusion_feature
        # Input: [u_lr, h_expand, d_expand] = [R + H + H]
        mlp_input_dim = rank_dim + gene_dim + drug_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, out_dim),
            nn.LayerNorm(out_dim) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_pool, drug_emb):
        """
        Args:
            h_pool:   [B, H]
            drug_emb: [D, H]
        Returns:
            dict with keys:
                fusion_feature: [B*D, out_dim]
                u_lr:           [B*D, rank_dim]
                h_mod:          [B, D, H] or None
                film_gamma:     [D, H] or None
                film_beta:      [D, H] or None
        """
        B, H = h_pool.shape
        D = drug_emb.size(0)

        h_mod = None
        film_gamma = None
        film_beta = None

        if self.mode == "dcf_lrbf":
            # Generate FiLM parameters from drug_emb
            film = self.film_mlp(drug_emb)  # [D, 2H]
            film_gamma, film_beta = film.chunk(2, dim=-1)  # each [D, H]

            # Expand h_pool and apply FiLM
            h_pool_expand = h_pool.unsqueeze(1).expand(-1, D, -1)  # [B, D, H]
            h_mod = h_pool_expand * (1.0 + film_gamma.unsqueeze(0)) + film_beta.unsqueeze(0)  # [B, D, H]

            # Project modulated h
            h_proj = self.gene_proj(h_mod.reshape(B * D, H))  # [B*D, R]
        else:
            # LRBF mode: project h_pool directly
            h_proj_raw = self.gene_proj(h_pool)  # [B, R]
            h_proj = h_proj_raw.unsqueeze(1).expand(-1, D, -1).reshape(B * D, self.rank_dim)  # [B*D, R]

        # Project drug_emb
        d_proj_raw = self.drug_proj(drug_emb)  # [D, R]
        d_proj = d_proj_raw.unsqueeze(0).expand(B, -1, -1).reshape(B * D, self.rank_dim)  # [B*D, R]

        # Low-rank multiplicative interaction
        u_lr = h_proj * d_proj  # [B*D, R]

        # Build fusion feature
        h_expand = h_pool.unsqueeze(1).expand(-1, D, -1).reshape(B * D, H)
        d_expand = drug_emb.unsqueeze(0).expand(B, -1, -1).reshape(B * D, H)

        fusion_input = torch.cat([u_lr, h_expand, d_expand], dim=-1)
        fusion_feature = self.fusion_mlp(fusion_input)  # [B*D, out_dim]

        return {
            "fusion_feature": fusion_feature,
            "u_lr": u_lr,
            "h_mod": h_mod,
            "film_gamma": film_gamma,
            "film_beta": film_beta,
        }


class VariationalContrastiveBottleneck(nn.Module):
    """
    Phase 12: Variational Contrastive Bottleneck (VCB).

    Applies a Variational Information Bottleneck to fused features
    used in Stage 3 contrastive anchoring. The compressed latent z
    is normalized for contrastive loss, while KL is computed on the
    raw posterior parameters.

    Args:
        in_dim: input feature dimension
        z_dim: latent dimension (default: in_dim)
        hidden_dim: backbone hidden dimension (default: in_dim)
        dropout: dropout rate
        logvar_min: minimum clamp value for logvar
        logvar_max: maximum clamp value for logvar
        normalize_z: whether to L2-normalize z for contrastive
    """

    def __init__(
        self,
        in_dim: int,
        z_dim: int = None,
        hidden_dim: int = None,
        dropout: float = 0.1,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
        normalize_z: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim or in_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.normalize_z = normalize_z

        h_dim = hidden_dim or in_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(h_dim, self.z_dim)
        self.logvar = nn.Linear(h_dim, self.z_dim)

    def forward(self, x: torch.Tensor, sample: bool = True) -> dict:
        """
        Args:
            x: [N, F] fused feature
            sample: if True and self.training, use reparameterization
        Returns:
            dict with keys:
                z: [N, Z] (normalized if normalize_z=True)
                mu: [N, Z]
                logvar: [N, Z]
                kl: [N] KL(q(z|x) || N(0,I))
        """
        h = self.backbone(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(self.logvar_min, self.logvar_max)

        if self.training and sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_raw = mu + eps * std
        else:
            z_raw = mu

        # KL against standard normal prior (per sample, sum over z_dim)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)

        z = F.normalize(z_raw, dim=-1) if self.normalize_z else z_raw

        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "kl": kl,
        }


class DrugConditionedDomainAligner(nn.Module):
    """
    Phase DCDA: Drug-Conditioned Domain Diffusion Alignment.
    Inspired by HorizonRec's DPD (Dual-oriented Preference Diffusion).
    Aligns gene expression features across source (GDSC) and target (TCGA) domains
    under drug-specific conditions, before fusion.
    """
    def __init__(self, feat_dim=512, timesteps=10):
        super().__init__()
        self.feat_dim = feat_dim
        self.timesteps = timesteps

        # Denoiser MLP: [noisy_feat, time_emb, drug_cond] -> clean_feat
        self.denoiser = nn.Sequential(
            nn.Linear(feat_dim * 3, feat_dim * 2),
            nn.LayerNorm(feat_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim * 2, feat_dim),
        )

        # Sinusoidal time embedding (HorizonRec-style)
        half_dim = feat_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer("time_emb_base", torch.exp(torch.arange(half_dim) * -emb))

        # Beta schedule (lightweight)
        betas = torch.linspace(0.0001, 0.02, timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod)
        )

    def get_time_embedding(self, t, device):
        half_dim = self.feat_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb_base = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb_base[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

    def q_sample(self, x_start, t):
        noise = torch.randn_like(x_start)
        s = self.sqrt_alphas_cumprod[t].to(x_start.device)
        s1 = self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device)
        return s[:, None] * x_start + s1[:, None] * noise

    def forward(self, gene_feat, drug_cond, training=True):
        """
        Args:
            gene_feat: [B, D, feat_dim]
            drug_cond: [B, D, feat_dim]
        Returns:
            aligned_gene: [B, D, feat_dim]
            diff_loss: scalar MSE loss (0.0 if not training)
        """
        device = gene_feat.device
        B, D, H = gene_feat.shape
        gene_flat = gene_feat.reshape(B * D, H)
        drug_flat = drug_cond.reshape(B * D, H)

        if training:
            t = torch.randint(0, self.timesteps, (B * D,), device=device)
            noisy = self.q_sample(gene_flat, t)
            time_emb = self.get_time_embedding(t, device)

            denoiser_input = torch.cat([noisy, time_emb, drug_flat], dim=-1)
            predicted = self.denoiser(denoiser_input)
            diff_loss = F.mse_loss(predicted, gene_flat)
            return predicted.view(B, D, H), diff_loss
        else:
            t = torch.zeros(B * D, dtype=torch.long, device=device)
            time_emb = self.get_time_embedding(t, device)
            denoiser_input = torch.cat([gene_flat, time_emb, drug_flat], dim=-1)
            predicted = self.denoiser(denoiser_input)
            return predicted.view(B, D, H), torch.tensor(0.0, device=device)


class BMLReliabilityRouter(nn.Module):
    """
    Phase 18: BML-style reliability router for shared/private prediction fusion.

    Input: shared_logits [N], private_logits [N] (before sigmoid)
    Output: rel_weights [N, 2] (softmax-normalized, rel_weights[:,0] = shared weight)

    Reliability features per sample-drug pair:
      - entropy_shared: prediction entropy of shared branch
      - entropy_private: prediction entropy of private branch
      - agreement: negative symmetric KL between shared and private predictions
    """
    def __init__(self, hidden_dim=16):
        super(BMLReliabilityRouter, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, shared_logits, private_logits):
        """
        Args:
            shared_logits: [N] raw logits
            private_logits: [N] raw logits
        Returns:
            rel_weights: [N, 2] softmax over {shared, private}
        """
        p_s = torch.sigmoid(shared_logits)
        p_p = torch.sigmoid(private_logits)
        eps = 1e-8

        entropy_s = -(p_s * torch.log(p_s + eps) + (1 - p_s) * torch.log(1 - p_s + eps))
        entropy_p = -(p_p * torch.log(p_p + eps) + (1 - p_p) * torch.log(1 - p_p + eps))

        kl_sp = p_s * torch.log((p_s + eps) / (p_p + eps)) + (1 - p_s) * torch.log((1 - p_s + eps) / (1 - p_p + eps))
        kl_ps = p_p * torch.log((p_p + eps) / (p_s + eps)) + (1 - p_p) * torch.log((1 - p_p + eps) / (1 - p_s + eps))
        agreement = -(kl_sp + kl_ps)

        features = torch.stack([entropy_s, entropy_p, agreement], dim=1)  # [N, 3]
        logits = self.mlp(features)
        weights = F.softmax(logits, dim=1)
        return weights


class BMLTriBranchRouter(nn.Module):
    """
    Phase 18 v4: Three-branch reliability router (base + shared + private).
    Inspired by DCDNet's MGDF Fusion module with temperature-softmax
    and min_weight clamping to prevent branch collapse.

    Input: base_logits [N], shared_logits [N], private_logits [N] (before sigmoid)
    Output: rel_weights [N, 3] (softmax-normalized, rel_weights[:,0] = base weight)

    Reliability features per sample-drug pair:
      - entropy_base, entropy_shared, entropy_private
      - agreement_base_shared, agreement_base_private, agreement_shared_private
        (negative symmetric KL between each pair of predictions)
    """
    def __init__(self, hidden_dim=32, temperature=0.7, min_weight=0.05):
        super(BMLTriBranchRouter, self).__init__()
        self.temperature = temperature
        self.min_weight = min_weight
        # 3 entropies + 3 pairwise agreements = 6 features
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def _compute_entropy(self, logits):
        p = torch.sigmoid(logits)
        eps = 1e-8
        return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

    def _compute_agreement(self, logits_a, logits_b):
        """Negative symmetric KL between two predictions (higher = more agreement)."""
        p_a = torch.sigmoid(logits_a)
        p_b = torch.sigmoid(logits_b)
        eps = 1e-8
        kl_ab = p_a * torch.log((p_a + eps) / (p_b + eps)) + (1 - p_a) * torch.log((1 - p_a + eps) / (1 - p_b + eps))
        kl_ba = p_b * torch.log((p_b + eps) / (p_a + eps)) + (1 - p_b) * torch.log((1 - p_b + eps) / (1 - p_a + eps))
        return -(kl_ab + kl_ba)

    def forward(self, base_logits, shared_logits, private_logits):
        """
        Args:
            base_logits: [N] raw logits
            shared_logits: [N] raw logits
            private_logits: [N] raw logits
        Returns:
            rel_weights: [N, 3] softmax over {base, shared, private}
        """
        entropy_b = self._compute_entropy(base_logits)
        entropy_s = self._compute_entropy(shared_logits)
        entropy_p = self._compute_entropy(private_logits)

        agreement_bs = self._compute_agreement(base_logits, shared_logits)
        agreement_bp = self._compute_agreement(base_logits, private_logits)
        agreement_sp = self._compute_agreement(shared_logits, private_logits)

        features = torch.stack([
            entropy_b, entropy_s, entropy_p,
            agreement_bs, agreement_bp, agreement_sp,
        ], dim=1)  # [N, 6]

        logits = self.mlp(features)
        weights = F.softmax(logits / self.temperature, dim=1)

        # DCDNet-style min_weight clamping: no branch can collapse to zero
        if self.min_weight > 0:
            weights = torch.clamp(weights, min=self.min_weight)
            weights = weights / weights.sum(dim=1, keepdim=True)

        return weights


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion to replace Bilinear.

    Cell feature attends to all drugs (query=cell, KV=drugs) to extract
    a drug-context-aware cell representation.  Then a *per-drug* interaction
    is formed via Hadamard product (element-wise multiplication) between the
    cell context and each drug embedding, preserving drug specificity that
    the original concat+MLP design lost.
    """
    def __init__(self, gene_feat_dim=64, drug_feat_dim=64, bilinear_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.gene_proj = nn.Linear(gene_feat_dim, drug_feat_dim)
        self.cross_attn = nn.MultiheadAttention(drug_feat_dim, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(drug_feat_dim, 2 * drug_feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * drug_feat_dim, drug_feat_dim),
        )
        self.norm1 = nn.LayerNorm(drug_feat_dim)
        self.norm2 = nn.LayerNorm(drug_feat_dim)
        # Concatenation of cell context + drug embedding -> bilinear_dim
        self.output_proj = nn.Sequential(
            nn.Linear(drug_feat_dim + drug_feat_dim, bilinear_dim),
            nn.LayerNorm(bilinear_dim),
            nn.GELU(),
        )

    def forward(self, gene_feat, drug_feat_mlp):
        """
        Args:
            gene_feat: [B, gene_feat_dim] — cell-level gene representation
            drug_feat_mlp: [D, drug_feat_dim] — drug embeddings
        Returns:
            z: [B*D, bilinear_dim] — replaces bilinear output
            attn_weights: [B, 1, D] — interpretable attention per cell over drugs
        """
        B = gene_feat.size(0)
        D = drug_feat_mlp.size(0)

        # Cell as query, drugs as keys/values
        gene_q = self.gene_proj(gene_feat).unsqueeze(1)  # [B, 1, drug_feat_dim]
        drug_kv = drug_feat_mlp.unsqueeze(0).expand(B, -1, -1)  # [B, D, drug_feat_dim]

        # Cross-Attention: each cell attends to all drugs
        attn_out, attn_weights = self.cross_attn(gene_q, drug_kv, drug_kv)
        # attn_out: [B, 1, drug_feat_dim]
        # attn_weights: [B, 1, D]

        # Residual + FFN
        attn_out = self.norm1(gene_q + attn_out)
        ffn_out = self.ffn(attn_out)
        cell_context = self.norm2(attn_out + ffn_out).squeeze(1)  # [B, drug_feat_dim]

        # Expand to per-drug and concatenate with drug embedding
        cell_context_expand = cell_context.unsqueeze(1).expand(-1, D, -1).reshape(B * D, -1)  # [B*D, drug_dim]
        drug_expand = drug_feat_mlp.unsqueeze(0).expand(B, -1, -1).reshape(B * D, -1)  # [B*D, drug_dim]

        z = self.output_proj(torch.cat([cell_context_expand, drug_expand], dim=-1))  # [B*D, bilinear_dim]
        return z, attn_weights


class ProgramFusionMLP(nn.Module):
    """
    Stage 2/3 drop-in replacement for Fusion_MLP.
    The encoder is expected to be a MolecularProgramEncoder (or any encoder
    that returns a dict with key 'h' for the pooled gene representation).

    Phase 3B: supports optional DrugMechanismGate for drug-gated program aggregation.
    When use_drug_gate=False, behaves identically to Phase 2.

    Phase 8: supports optional DrugSelectiveProgramInteraction (DSPI).
    When use_dspi=False, behaves identically to Phase 2.
    """

    def __init__(
        self,
        encoder,
        input_dim,
        output_dim,
        hidden_dims=None,
        gene_num=1306,
        bilinear_dim=128,
        gene_feat_dim=64,
        drug_feat_dim=64,
        drug_num=9,
        drop=0.1,
        act_fn=nn.SELU,
        use_drug_gate=False,
        shared_gate=False,
        num_programs=8,
        gate_mode="additive_bias",
        gate_temperature=5.0,
        rho_init=0.0,
        beta_init=0.0,
        # Phase 8: DSPI parameters
        use_dspi=False,
        dspi_config=None,
        return_dspi=False,
        # Phase 8-F: BPAF / LLRF parameters
        use_bpaf=False,
        bpaf_config=None,
        return_bpaf=False,
        use_llrf=False,
        llrf_config=None,
        # Phase 9: FPI parameters
        use_fpi=False,
        fpi_config=None,
        return_fpi=False,
        return_dict=False,
        frag_fp=None,
        frag_mask=None,
        # Phase 9-R: FPI Router
        use_fpi_router=False,
        fpi_router_config=None,
        # Phase 10: Pseudo-Domain Guide
        use_pdg=False,
        pdg_config=None,
        pdg_centroids=None,
        # Phase 10-D: DCD-R
        use_dcd_residual=False,
        dcd_config=None,
        # Phase 11: Main Fusion Replacement
        main_fusion_type="bilinear",
        main_fusion_config=None,
        # Phase 11-C: Cross-Attention Fusion
        use_cross_attn=False,
        cross_attn_num_heads=4,
        # Phase 12: VCB
        use_vcb=False,
        vcb_config=None,
        # Phase DCDA: Drug-Conditioned Domain Diffusion Alignment
        use_dcda=False,
        dcda_config=None,
        **kwargs,
    ):
        super(ProgramFusionMLP, self).__init__()
        self.encoder = encoder
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drug_num = drug_num
        self.drop = drop
        self.use_drug_gate = use_drug_gate
        self.shared_gate = shared_gate
        self.gate_mode = gate_mode
        self.last_gate = None
        self.last_gate_diagnostics = None
        # Phase 8: DSPI
        self.use_dspi = use_dspi
        self.return_dspi = return_dspi
        self.dspi_config = dspi_config or {}
        # Phase 8-F: BPAF / LLRF
        self.use_bpaf = use_bpaf
        self.bpaf_config = bpaf_config or {}
        self.return_bpaf = return_bpaf
        self.use_llrf = use_llrf
        self.llrf_config = llrf_config or {}
        # Phase 9: FPI
        self.use_fpi = use_fpi
        self.fpi_config = fpi_config or {}
        self.return_fpi = return_fpi
        self.return_dict = return_dict
        # Phase 9-R: FPI Router
        self.use_fpi_router = use_fpi_router
        self.fpi_router_config = fpi_router_config or {}
        # Phase 10: Pseudo-Domain Guide
        self.use_pdg = use_pdg
        self.pdg_config = pdg_config or {}
        self.last_pdg_assign = None
        self.last_pdg_sim = None
        self.last_pdg_emb = None
        # Phase 10-D: DCD-R
        self.use_dcd_residual = use_dcd_residual
        self.dcd_config = dcd_config or {}
        self.last_dcd_out = None
        # Phase 11: Main Fusion Replacement
        self.main_fusion_type = main_fusion_type
        self.main_fusion_config = main_fusion_config or {}
        self.last_main_fusion_out = None
        # Phase 12: VCB
        self.use_vcb = use_vcb
        self.vcb_config = vcb_config or {}
        self.vcb = None

        # Phase DCDA
        self.use_dcda = use_dcda
        self.dcda_config = dcda_config or {}
        self.domain_aligner = None
        self.dcda_condition_proj = None
        self.dcda_interaction_proj = None
        self.last_dcda_info = None
        if self.use_dcda:
            dcda_cfg = self.dcda_config
            dcda_feat_dim = dcda_cfg.get("feat_dim", gene_feat_dim)
            dcda_timesteps = dcda_cfg.get("timesteps", 10)
            dcda_interaction_dim = dcda_cfg.get("interaction_dim", bilinear_dim)
            self.domain_aligner = DrugConditionedDomainAligner(
                feat_dim=dcda_feat_dim, timesteps=dcda_timesteps
            )
            self.dcda_condition_proj = nn.Sequential(
                nn.Linear(drug_feat_dim, dcda_feat_dim),
                nn.LayerNorm(dcda_feat_dim),
                nn.GELU(),
            )
            self.dcda_interaction_proj = nn.Sequential(
                nn.Linear(dcda_feat_dim + drug_feat_dim, dcda_interaction_dim),
                nn.LayerNorm(dcda_interaction_dim),
                nn.GELU(),
            )

        if use_bpaf and use_llrf:
            raise ValueError("use_bpaf and use_llrf cannot both be True")
        if use_bpaf and use_dspi:
            raise ValueError("use_bpaf and use_dspi cannot both be True")
        if use_llrf and use_dspi:
            raise ValueError("use_llrf and use_dspi cannot both be True")
        if use_fpi and (use_bpaf or use_llrf or use_dspi or use_drug_gate):
            raise ValueError("use_fpi is incompatible with use_bpaf, use_llrf, use_dspi, use_drug_gate")
        if use_fpi_router and (use_fpi or use_bpaf or use_llrf or use_dspi or use_drug_gate):
            raise ValueError("use_fpi_router is incompatible with use_fpi, use_bpaf, use_llrf, use_dspi, use_drug_gate")

        self.encoder_input_dim = gene_num
        self.attention_masks = nn.Parameter(
            torch.ones(len(config.tissue_map), self.encoder_input_dim)
        )
        self.attention_activation = nn.Sigmoid()

        # Lazy import to avoid pulling in the full original model graph
        # (which triggers dataload -> rdkit chain at import time)
        from models import FeatMLP, Drug_MLP

        self.drug_mlp = Drug_MLP(
            input_dim=128,
            output_dim=64,
            hidden_dims=[128, 64],
            drop=0.1,
        )

        # Phase 3B: drug gate
        if self.use_drug_gate:
            if self.shared_gate:
                self.drug_gate = SharedDrugGate(num_programs=num_programs)
            else:
                self.drug_gate = DrugMechanismGate(
                    drug_dim=drug_feat_dim,
                    num_programs=num_programs,
                    hidden_dim=128,
                    drop=drop,
                    act_fn=act_fn,
                )
            self.aggregator = DrugGatedProgramAggregator(
                gate_mode=gate_mode,
                gate_temperature=gate_temperature,
                rho_init=rho_init,
                beta_init=beta_init,
                num_programs=num_programs,
            )
        else:
            self.drug_gate = None
            self.aggregator = None

        # Phase 8: DSPI module
        if self.use_dspi:
            dspi_cfg = self.dspi_config
            self.dspi = DrugSelectiveProgramInteraction(
                program_dim=dspi_cfg.get("program_dim", gene_feat_dim),
                drug_dim=dspi_cfg.get("drug_dim", drug_feat_dim),
                num_programs=dspi_cfg.get("num_programs", num_programs),
                num_drugs=dspi_cfg.get("num_drugs", drug_num),
                temperature=dspi_cfg.get("temperature", 1.0),
                inter_weight=dspi_cfg.get("inter_weight", 0.2),
                q_prior_weight=dspi_cfg.get("q_prior_weight", 0.1),
                drug_sim_temperature=dspi_cfg.get("drug_sim_temperature", 1.0),
                rho_mode=dspi_cfg.get("rho_mode", "fixed"),
                fixed_rho=dspi_cfg.get("fixed_rho", 0.1),
                rho_max=dspi_cfg.get("rho_max", 0.3),
            )
        else:
            self.dspi = None

        # Phase 8-F: BPAF / LLRF modules
        self.program_selector = None
        self.bpaf = None
        self.llrf = None

        if self.use_bpaf or self.use_llrf:
            # Shared clean selector (no inter-drug attention)
            sel_cfg = self.bpaf_config if self.use_bpaf else self.llrf_config
            self.program_selector = DrugSelectiveProgramSelector(
                program_dim=sel_cfg.get("program_dim", gene_feat_dim),
                drug_dim=sel_cfg.get("drug_dim", drug_feat_dim),
                num_programs=sel_cfg.get("num_programs", num_programs),
                num_drugs=sel_cfg.get("num_drugs", drug_num),
                attention_temperature=sel_cfg.get("attention_temperature", 1.0),
                q_prior_weight=sel_cfg.get("q_prior_weight", 0.01),
            )

        if self.use_bpaf:
            bpaf_cfg = self.bpaf_config
            self.bpaf = BasePreservingAdapterFusion(
                hidden_dim=bpaf_cfg.get("hidden_dim", gene_feat_dim),
                num_drugs=bpaf_cfg.get("num_drugs", drug_num),
                gamma_max=bpaf_cfg.get("gamma_max", 0.05),
                gamma_init=bpaf_cfg.get("gamma_init", 0.02),
                dropout=bpaf_cfg.get("dropout", drop),
                adapter_include_h_pool=bpaf_cfg.get("adapter_include_h_pool", True),
            )

        if self.use_llrf:
            llrf_cfg = self.llrf_config
            self.llrf = LateLogitResidualFusion(
                hidden_dim=llrf_cfg.get("hidden_dim", gene_feat_dim),
                num_drugs=llrf_cfg.get("num_drugs", drug_num),
                gamma_max=llrf_cfg.get("gamma_max", 0.1),
                gamma_init=llrf_cfg.get("gamma_init", 0.02),
                dropout=llrf_cfg.get("dropout", drop),
            )

        # Phase 9: FPI modules
        self.fragment_encoder = None
        self.fpi = None
        self.fpi_router = None
        self.register_buffer("_frag_fp", None)
        self.register_buffer("_frag_mask", None)
        if self.use_fpi:
            fpi_cfg = self.fpi_config
            self.fragment_encoder = DrugFragmentEncoder(
                fp_dim=fpi_cfg.get("fp_dim", 1024),
                hidden_dim=fpi_cfg.get("hidden_dim", gene_feat_dim),
                dropout=fpi_cfg.get("dropout", drop),
            )
            self.fpi = FragmentProgramInteraction(
                program_dim=fpi_cfg.get("program_dim", gene_feat_dim),
                drug_dim=fpi_cfg.get("drug_dim", drug_feat_dim),
                fragment_dim=fpi_cfg.get("fragment_dim", gene_feat_dim),
                num_programs=fpi_cfg.get("num_programs", num_programs),
                num_drugs=fpi_cfg.get("num_drugs", drug_num),
                max_fragments=fpi_cfg.get("max_fragments", 16),
                q_prior_weight=fpi_cfg.get("q_prior_weight", 0.01),
                gamma_max=fpi_cfg.get("gamma_max", 0.05),
                gamma_init=fpi_cfg.get("gamma_init", 0.0),
                dropout=fpi_cfg.get("dropout", drop),
                use_fpi_sim=fpi_cfg.get("use_fpi_sim", False),
                fpi_sim_config=fpi_cfg.get("fpi_sim_config", None),
            )
            self.fpi_detach_mode = fpi_cfg.get("detach_mode", "none")
            if frag_fp is not None:
                self.register_buffer("_frag_fp", frag_fp)
            if frag_mask is not None:
                self.register_buffer("_frag_mask", frag_mask)

        # Phase 9-R: FPI Router modules
        if self.use_fpi_router:
            from mmpa_drp.fpi import DualFPIResidualRouter
            router_cfg = self.fpi_router_config
            self.fragment_encoder = DrugFragmentEncoder(
                fp_dim=router_cfg.get("fp_dim", 1024),
                hidden_dim=router_cfg.get("hidden_dim", gene_feat_dim),
                dropout=router_cfg.get("dropout", drop),
            )
            self.fpi_router = DualFPIResidualRouter(
                program_dim=router_cfg.get("program_dim", gene_feat_dim),
                drug_dim=router_cfg.get("drug_dim", drug_feat_dim),
                fragment_dim=router_cfg.get("fragment_dim", gene_feat_dim),
                num_programs=router_cfg.get("num_programs", num_programs),
                num_drugs=router_cfg.get("num_drugs", drug_num),
                max_fragments=router_cfg.get("max_fragments", 16),
                q_prior_weight=router_cfg.get("q_prior_weight", 0.01),
                gamma_global_max=router_cfg.get("gamma_global_max", 0.05),
                gamma_global_init=router_cfg.get("gamma_global_init", 0.0),
                gamma_cross_max=router_cfg.get("gamma_cross_max", 0.05),
                gamma_cross_init=router_cfg.get("gamma_cross_init", 0.0),
                dropout=router_cfg.get("dropout", drop),
                router_cross_start_epoch=router_cfg.get("router_cross_start_epoch", 0),
                router_cross_ramp_epochs=router_cfg.get("router_cross_ramp_epochs", 0),
                router_cross_mode=router_cfg.get("router_cross_mode", "learnable"),
                router_oracle_cross_drugs=router_cfg.get("router_oracle_cross_drugs", ""),
                router_residual_global_scale=router_cfg.get("router_residual_global_scale", 1.0),
                router_residual_cross_scale=router_cfg.get("router_residual_cross_scale", 1.0),
                router_branch_effect_threshold=router_cfg.get("router_branch_effect_threshold", 1e-4),
                router_cross_detach_mode=router_cfg.get("router_cross_detach_mode", "none"),
                debug_fixed_gamma_global=router_cfg.get("debug_fixed_gamma_global", None),
                debug_fixed_gamma_cross=router_cfg.get("debug_fixed_gamma_cross", None),
            )
            if frag_fp is not None:
                self.register_buffer("_frag_fp", frag_fp)
            if frag_mask is not None:
                self.register_buffer("_frag_mask", frag_mask)

        # Phase 10: Pseudo-Domain Guide modules
        self.pdg = None
        if self.use_pdg:
            if pdg_centroids is None:
                raise ValueError("use_pdg=True requires pdg_centroids")
            pdg_cfg = self.pdg_config
            self.pdg = PseudoDomainGuide(
                centroid_dim=gene_feat_dim,
                pdg_dim=pdg_cfg.get("pdg_dim", 32),
                dropout=pdg_cfg.get("dropout", drop),
                assign_mode=pdg_cfg.get("assign_mode", "nearest"),
                normalize=pdg_cfg.get("normalize", True),
            )
            self.register_buffer("_pdg_centroids", pdg_centroids.float())

        # Phase 10-D: DCD-R module
        self.dcd_residual = None
        if self.use_dcd_residual:
            dcd_cfg = self.dcd_config
            self.dcd_residual = ResidualDecoupledFusion(
                hidden_dim=dcd_cfg.get("hidden_dim", gene_feat_dim),
                drug_dim=dcd_cfg.get("drug_dim", drug_feat_dim),
                num_drugs=dcd_cfg.get("num_drugs", drug_num),
                dcd_hidden_dim=dcd_cfg.get("dcd_hidden_dim", gene_feat_dim),
                gamma_max=dcd_cfg.get("gamma_max", 0.03),
                gamma_init=dcd_cfg.get("gamma_init", 0.005),
                dropout=dcd_cfg.get("dropout", drop),
                use_product=dcd_cfg.get("use_product", True),
                use_absdiff=dcd_cfg.get("use_absdiff", False),
            )

        # Phase 11: Main Fusion Replacement
        self.main_fusion_replacement = None
        if self.main_fusion_type in ("lrbf", "dcf_lrbf"):
            mf_cfg = self.main_fusion_config
            self.main_fusion_replacement = DrugConditionedLowRankFusion(
                gene_dim=gene_feat_dim,
                drug_dim=drug_feat_dim,
                rank_dim=mf_cfg.get("rank_dim", 64),
                out_dim=mf_cfg.get("out_dim", bilinear_dim),
                mode=self.main_fusion_type,
                dropout=mf_cfg.get("dropout", drop),
                use_layernorm=mf_cfg.get("use_layernorm", True),
                film_init_scale=mf_cfg.get("film_init_scale", 1e-3),
            )

        # Phase 11-C: Cross-Attention Fusion (additive residual to Bilinear)
        self.use_cross_attn = use_cross_attn
        self.cross_attn_fusion = None
        if self.use_cross_attn:
            self.cross_attn_fusion = CrossAttentionFusion(
                gene_feat_dim=gene_feat_dim,
                drug_feat_dim=drug_feat_dim,
                bilinear_dim=bilinear_dim,
                num_heads=cross_attn_num_heads,
                dropout=drop,
            )
        self.bilinear = nn.Bilinear(gene_feat_dim, drug_feat_dim, bilinear_dim)

        # Phase 11-C: Cross-Attention Fusion (additive residual to Bilinear)
        self.cross_attn_fusion = None
        if self.use_cross_attn:
            self.cross_attn_fusion = CrossAttentionFusion(
                gene_feat_dim=gene_feat_dim,
                drug_feat_dim=drug_feat_dim,
                bilinear_dim=bilinear_dim,
                num_heads=cross_attn_num_heads,
                dropout=drop,
            )

        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.fusion_module = FeatMLP(
            input_dim, output_dim, hidden_dims, drop=drop, act_fn=act_fn
        )
        self.fused_feature_dim = hidden_dims[-1]

        # Phase 12: VCB module
        if self.use_vcb:
            vcfg = self.vcb_config
            self.vcb = VariationalContrastiveBottleneck(
                in_dim=self.fused_feature_dim,
                z_dim=vcfg.get("z_dim", None) or self.fused_feature_dim,
                hidden_dim=vcfg.get("hidden_dim", None) or self.fused_feature_dim,
                dropout=vcfg.get("dropout", drop),
                logvar_min=vcfg.get("logvar_min", -6.0),
                logvar_max=vcfg.get("logvar_max", 2.0),
                normalize_z=vcfg.get("normalize_z", True),
            )

        cdan_input_dim = hidden_dims[-1] + drug_num
        self.cdan_discriminator = nn.Sequential(
            nn.Linear(cdan_input_dim, cdan_input_dim // 2),
            nn.BatchNorm1d(cdan_input_dim // 2),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(cdan_input_dim // 2, cdan_input_dim // 4),
            nn.BatchNorm1d(cdan_input_dim // 4),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(cdan_input_dim // 4, 1),
        )

        self.drug_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dims[-1] // 2, drug_num),
        )

        # Phase 18: Domain Separation support
        self.use_domain_sep = getattr(encoder, 'use_domain_sep', False)
        self.domain_sep_proj = None
        self.bml_router = None
        if self.use_domain_sep:
            # Check incompatible modules
            incompatible = []
            if self.use_bpaf: incompatible.append("use_bpaf")
            if self.use_llrf: incompatible.append("use_llrf")
            if self.use_dspi: incompatible.append("use_dspi")
            if self.use_drug_gate: incompatible.append("use_drug_gate")
            if self.use_fpi_router: incompatible.append("use_fpi_router")
            if self.use_dcd_residual: incompatible.append("use_dcd_residual")
            if self.use_dcda: incompatible.append("use_dcda")
            if incompatible:
                raise ValueError(
                    f"Phase 18 domain_sep is incompatible with: {incompatible}. "
                    "Please disable these modules when use_domain_sep=True."
                )
            ds_out_dim = encoder.domain_sep.shared_mlp[4].out_features
            if ds_out_dim != gene_feat_dim:
                self.domain_sep_proj = nn.Sequential(
                    nn.Linear(ds_out_dim, gene_feat_dim),
                    nn.BatchNorm1d(gene_feat_dim),
                    nn.SELU(),
                )
            self.bml_router = BMLReliabilityRouter(hidden_dim=16)
            self.last_domain_sep_pred = None

    def forward(self, inputs, drug_feat, tissue_idx, alpha=1.0, return_encoder_out=False, return_fused_feature=False, return_dict=False, frag_fp=None, frag_mask=None, frag_smiles=None, current_epoch=None, drug_names=None, domain_idx=None):
        if tissue_idx is None:
            attention_weights = self.attention_activation(
                self.attention_masks.mean(dim=0)
            )
        else:
            attention_weights = self.attention_activation(
                self.attention_masks[tissue_idx]
            )

        inputs = inputs * attention_weights
        encoder_out = self.encoder(inputs, domain_idx=domain_idx)

        drug_feat_mlp, _ = self.drug_mlp(drug_feat)  # [drug_num, D]
        drug_feat_mlp = F.normalize(drug_feat_mlp, p=2, dim=1)

        # Phase 18: Domain-separated fast path
        if self.use_domain_sep:
            return self._forward_domain_sep(
                encoder_out, drug_feat_mlp, tissue_idx, alpha,
                return_encoder_out, return_fused_feature, return_dict,
                domain_idx=domain_idx,
                frag_fp=frag_fp, frag_mask=frag_mask,
                current_epoch=current_epoch, drug_names=drug_names,
            )

        dspi_out = None

        # Phase 8-F: BPAF path
        if self.use_bpaf:
            z = encoder_out["z"]  # [B, K, d]
            q = encoder_out["q"]  # [B, K]
            h_pool = encoder_out["h"]  # [B, d]

            selector_out = self.program_selector(
                z=z, q=q, h_pool=h_pool, drug_emb=drug_feat_mlp
            )
            h_select = selector_out["h_select"]  # [B, D, d]

            bpaf_out = self.bpaf(
                h_pool=h_pool,
                h_select=h_select,
                drug_emb=drug_feat_mlp,
            )
            h_eff = bpaf_out["h_eff"]  # [B, D, d]

            # Store for loss/diagnostics
            self.last_selector_attn = selector_out["attn"].detach().clone()
            self.last_bpaf_gamma = bpaf_out["gamma"].detach().clone()
            self.last_bpaf_out = bpaf_out

            # gene_feat for OT/contrastive stays h_pool (base-preserving)
            gene_feat = F.normalize(h_pool, p=2, dim=1)
            gene_feat_for_bilinear = h_eff.reshape(-1, h_eff.size(-1))  # [B*D, d]

        # Phase 8-F: LLRF path
        elif self.use_llrf:
            z = encoder_out["z"]  # [B, K, d]
            q = encoder_out["q"]  # [B, K]
            h_pool = encoder_out["h"]  # [B, d]

            selector_out = self.program_selector(
                z=z, q=q, h_pool=h_pool, drug_emb=drug_feat_mlp
            )
            h_select = selector_out["h_select"]  # [B, D, d]

            # Store for LLRF application after base logits
            self.last_selector_out = selector_out
            self.last_h_pool = h_pool
            self.last_h_select = h_select

            # gene_feat for OT/contrastive stays h_pool
            gene_feat = F.normalize(h_pool, p=2, dim=1)
            gene_feat_for_bilinear = h_pool.unsqueeze(1).expand(
                -1, self.drug_num, -1
            ).reshape(-1, h_pool.size(-1))  # [B*D, d]

        elif self.use_dspi:
            # Phase 8: DSPI path
            z = encoder_out["z"]  # [B, K, d]
            q = encoder_out["q"]  # [B, K]
            h_pool = encoder_out["h"]  # [B, d]

            dspi_out = self.dspi(z=z, q=q, h=h_pool, drug_emb=drug_feat_mlp)
            h_drug = dspi_out["h_drug"]  # [B, D, d]

            # Store DSPI outputs for loss computation in training step
            self.last_dspi_attn = dspi_out["attn"].detach().clone()
            self.last_dspi_h_drug = dspi_out["h_drug"].detach().clone()
            # Track source vs target for gram consistency
            if not hasattr(self, '_dspi_source_count'):
                self._dspi_source_count = 0
                self._dspi_target_count = 0
            # Heuristic: if we've seen more sources than targets, this is likely target
            if self._dspi_source_count > self._dspi_target_count:
                self.last_dspi_h_drug_target = dspi_out["h_drug"].detach().clone()
                self._dspi_target_count += 1
            else:
                self.last_dspi_h_drug_source = dspi_out["h_drug"].detach().clone()
                self._dspi_source_count += 1

            # gene_feat for contrastive matching (use h_pool for compatibility)
            gene_feat = F.normalize(h_pool, p=2, dim=1)
            gene_feat_for_bilinear = h_drug.reshape(-1, h_drug.size(-1))  # [B*D, d]
        elif self.use_drug_gate:
            z = encoder_out["z"]  # [B, K, d]
            q = encoder_out["q"]  # [B, K]
            h_pool = encoder_out["h"]  # [B, d]

            # Drug gate: [D, K] or [1, K]
            gate_input = drug_feat_mlp if not self.shared_gate else None
            if self.shared_gate:
                drug_gate_probs = self.drug_gate(gate_input)
                if drug_gate_probs.size(0) == 1:
                    drug_gate_probs = drug_gate_probs.expand(self.drug_num, -1)
                # For shared gate, derive pseudo-logits from probabilities for additive_bias mode
                if self.gate_mode in ("additive_bias", "additive_bias_no_residual"):
                    drug_gate_logits = torch.log(drug_gate_probs + 1e-8)
                else:
                    drug_gate_logits = None
            else:
                if self.gate_mode in ("additive_bias", "additive_bias_no_residual"):
                    drug_gate_logits = self.drug_gate(gate_input, return_logits=True)
                    drug_gate_probs = F.softmax(drug_gate_logits, dim=1)
                else:
                    drug_gate_logits = None
                    drug_gate_probs = self.drug_gate(gate_input)

            self.last_gate = drug_gate_probs.detach().clone()
            self.last_gate_probs = drug_gate_probs  # keep gradient for gate loss

            # Drug-gated aggregation: h_drug [B, D, d]
            h_drug, diagnostics = self.aggregator(z, q, h_pool, drug_gate_logits, drug_gate_probs)
            self.last_gate_diagnostics = diagnostics

            # gene_feat for contrastive matching
            if self.gate_mode in ("residual_multiplicative", "additive_bias"):
                gene_feat = h_pool
            else:
                gene_feat = h_drug.mean(dim=1)
            gene_feat = F.normalize(gene_feat, p=2, dim=1)
            gene_feat_for_bilinear = h_drug.reshape(-1, h_drug.size(-1))  # [B*D, d]
        else:
            gene_feat = encoder_out["h"]  # [B, d]
            gene_feat = F.normalize(gene_feat, p=2, dim=1)
            gene_feat_for_bilinear = gene_feat.unsqueeze(1).expand(
                -1, self.drug_num, -1
            ).reshape(-1, gene_feat.size(-1))  # [B*D, d]

        drug_feat_expand = drug_feat_mlp.unsqueeze(0).expand(
            gene_feat_for_bilinear.size(0) // self.drug_num, -1, -1
        ).reshape(-1, drug_feat_mlp.size(-1))  # [B*D, d_drug]

        # Phase DCDA: Drug-Conditioned Domain Diffusion Alignment
        if self.use_dcda:
            B = gene_feat.size(0)
            D = self.drug_num
            # Expand gene feat per drug: [B, D, gene_feat_dim]
            gene_expanded = gene_feat.unsqueeze(1).expand(-1, D, -1)
            # Drug condition from drug_feat_mlp: [D, gene_feat_dim]
            drug_cond = self.dcda_condition_proj(drug_feat_mlp)
            drug_cond_expanded = drug_cond.unsqueeze(0).expand(B, -1, -1)
            # Diffusion alignment
            aligned_gene, diff_loss = self.domain_aligner(
                gene_expanded, drug_cond_expanded, training=self.training
            )
            self.last_dcda_info = {
                "diff_loss": diff_loss,
                "aligned_gene": aligned_gene.detach() if not self.training else aligned_gene,
            }
            # Flatten for fusion: [B*D, gene_feat_dim]
            aligned_gene_flat = aligned_gene.reshape(-1, aligned_gene.size(-1))
            # Interaction feature replaces bilinear: [B*D, interaction_dim]
            interaction = self.dcda_interaction_proj(
                torch.cat([aligned_gene_flat, drug_feat_expand], dim=-1)
            )
            z = interaction
            gene_feat_for_bilinear = aligned_gene_flat
            # Update gene_feat for downstream (OT/CDAN) to aligned mean
            gene_feat = aligned_gene.mean(dim=1)
            gene_feat = F.normalize(gene_feat, p=2, dim=1)
        else:
            # Phase 11: Main Fusion Replacement (Bilinear always computed; cross-attn additive residual)
            z_bilinear = self.bilinear(
                gene_feat_for_bilinear,
                drug_feat_expand,
            )  # [B*drug_num, bilinear_dim]
            if self.use_cross_attn:
                z_cross, _ = self.cross_attn_fusion(
                    encoder_out["h"] if not self.use_dcda else gene_feat,
                    drug_feat_mlp,
                )  # [B*drug_num, bilinear_dim]
                z = z_bilinear + z_cross
            elif self.main_fusion_type == "bilinear":
                z = z_bilinear
            else:
                main_fusion_out = self.main_fusion_replacement(
                    h_pool=encoder_out["h"],
                    drug_emb=drug_feat_mlp,
                )
                z = main_fusion_out["fusion_feature"]  # [B*drug_num, out_dim]
                self.last_main_fusion_out = main_fusion_out

        fusion_parts = [
            z,
            gene_feat_for_bilinear,
            drug_feat_expand,
        ]

        if self.use_pdg:
            pdg_source = gene_feat  # [B, H]
            pdg_out = self.pdg(pdg_source, self._pdg_centroids)
            pdg_emb = pdg_out["pdg_emb"]  # [B, pdg_dim]
            pdg_assign = pdg_out["assign"]  # [B]

            pdg_expand = pdg_emb.unsqueeze(1).expand(
                -1, self.drug_num, -1
            ).reshape(-1, pdg_emb.size(-1))  # [B*D, pdg_dim]

            fusion_parts.append(pdg_expand)

            self.last_pdg_assign = pdg_assign.detach()
            self.last_pdg_sim = pdg_out["sim"].detach()
            self.last_pdg_emb = pdg_emb.detach()

        fusion_input = torch.cat(fusion_parts, dim=1)  # [B*drug_num, ...]

        probs, feat = self.fusion_module(fusion_input)

        # Phase 10-D: DCD-R applies residual on base logits (before LLRF/FPI)
        dcd_out = None
        if self.use_dcd_residual:
            batch_size = gene_feat_for_bilinear.size(0) // self.drug_num
            base_logits = probs.view(batch_size, self.drug_num)  # [B, D]

            # Always use encoder h_pool as DCD input (available in all paths)
            dcd_out = self.dcd_residual(
                h_pool=encoder_out["h"],
                drug_emb=drug_feat_mlp,
                base_logits=base_logits,
            )
            probs = dcd_out["final_base_logits"].view(-1, 1)  # [B*D, 1]

            # Store for diagnostics
            self.last_dcd_out = dcd_out

        # Phase 8-F: LLRF applies residual on logits
        if self.use_llrf:
            batch_size = gene_feat_for_bilinear.size(0) // self.drug_num
            base_logits = probs.view(batch_size, self.drug_num)  # [B, D]

            llrf_out = self.llrf(
                h_pool=self.last_h_pool,
                h_select=self.last_h_select,
                drug_emb=drug_feat_mlp,
                base_logits=base_logits,
            )
            final_logits = llrf_out["final_logits"]  # [B, D]
            probs = final_logits.view(-1, 1)  # [B*D, 1]

            # Store for diagnostics
            self.last_llrf_out = llrf_out

        # Phase 9: FPI applies residual on logits
        fpi_out = None
        effective_frag_fp = frag_fp if frag_fp is not None else self._frag_fp
        effective_frag_mask = frag_mask if frag_mask is not None else self._frag_mask
        if self.use_fpi and effective_frag_fp is not None and effective_frag_mask is not None:
            batch_size = gene_feat_for_bilinear.size(0) // self.drug_num
            base_logits = probs.view(batch_size, self.drug_num)  # [B, D]

            # Stop-gradient control for FPI branch
            z_fpi = encoder_out["z"]
            q_fpi = encoder_out["q"]
            h_pool_fpi = encoder_out["h"]
            drug_emb_fpi = drug_feat_mlp
            frag_emb_fpi = None

            detach_mode = getattr(self, "fpi_detach_mode", "none")
            if detach_mode == "z":
                z_fpi = z_fpi.detach()
            elif detach_mode == "zq":
                z_fpi = z_fpi.detach()
                q_fpi = q_fpi.detach()
            elif detach_mode == "zq_drug":
                z_fpi = z_fpi.detach()
                q_fpi = q_fpi.detach()
                drug_emb_fpi = drug_emb_fpi.detach()
            elif detach_mode == "all":
                z_fpi = z_fpi.detach()
                q_fpi = q_fpi.detach()
                h_pool_fpi = h_pool_fpi.detach()
                drug_emb_fpi = drug_emb_fpi.detach()
            elif detach_mode != "none":
                raise ValueError(f"Unknown fpi_detach_mode: {detach_mode}")

            frag_emb = self.fragment_encoder(effective_frag_fp, effective_frag_mask)
            frag_emb_fpi = frag_emb
            if detach_mode == "all":
                frag_emb_fpi = frag_emb_fpi.detach()

            fpi_out = self.fpi(
                z=z_fpi,
                q=q_fpi,
                h_pool=h_pool_fpi,
                drug_emb=drug_emb_fpi,
                frag_emb=frag_emb_fpi,
                frag_mask=effective_frag_mask,
            )
            residual_logits = fpi_out["residual_logits"]  # [B, D]
            gamma = fpi_out["gamma"]  # [D]

            final_logits = base_logits + gamma.unsqueeze(0) * residual_logits
            probs = final_logits.view(-1, 1)  # [B*D, 1]

            # Store for diagnostics
            self.last_fpi_out = fpi_out

        # Phase 9-R: FPI Router
        router_out = None
        effective_frag_fp = frag_fp if frag_fp is not None else self._frag_fp
        effective_frag_mask = frag_mask if frag_mask is not None else self._frag_mask
        if self.use_fpi_router and effective_frag_fp is not None and effective_frag_mask is not None:
            batch_size = gene_feat_for_bilinear.size(0) // self.drug_num
            base_logits = probs.view(batch_size, self.drug_num)  # [B, D]

            frag_emb = self.fragment_encoder(effective_frag_fp, effective_frag_mask)
            router_out = self.fpi_router(
                z=encoder_out["z"],
                q=encoder_out["q"],
                h_pool=encoder_out["h"],
                drug_emb=drug_feat_mlp,
                frag_emb=frag_emb,
                frag_mask=effective_frag_mask,
                base_logits=base_logits,
                current_epoch=current_epoch,
                drug_names=drug_names,
            )
            final_logits = router_out["final_logits"]  # [B, D]
            probs = final_logits.view(-1, 1)  # [B*D, 1]

            # Store for diagnostics
            self.last_router_out = router_out

        outputs = torch.sigmoid(probs)

        # Compute fused_feature [B, D, H] from feat [B*D, H]
        batch_size = gene_feat_for_bilinear.size(0) // self.drug_num
        fused_feature = feat.view(batch_size, self.drug_num, -1)

        # CDAN
        if self.training:
            drug_pred = self.drug_classifier(feat)  # [B*drug_num, drug_num]
            drug_prob = F.softmax(drug_pred, dim=1)  # [B*drug_num, drug_num]

            conditional_feat = torch.cat(
                [feat, drug_prob], dim=1
            )  # [B*drug_num, feat_dim + drug_num]

            reversed_conditional_feat = ReverseLayerF.apply(conditional_feat, alpha)
            cdan_domain_output = self.cdan_discriminator(reversed_conditional_feat)
        else:
            cdan_domain_output = None
            drug_pred = None

        if tissue_idx is not None:
            expanded_tissue_idx = tissue_idx.repeat_interleave(self.drug_num)
        else:
            expanded_tissue_idx = None

        if return_fused_feature or return_dict:
            # final_logits = probs before sigmoid, shape [B*drug_num, 1]
            result = {
                "pred": outputs,
                "final_logits": probs,
                "gene_feat": gene_feat,
                "fused_feature": fused_feature,
                "tissue_idx": expanded_tissue_idx,
                "cdan_domain_output": cdan_domain_output,
                "drug_pred": drug_pred,
            }
            if return_encoder_out:
                result["encoder_out"] = encoder_out
            if self.use_bpaf and getattr(self, "last_bpaf_out", None) is not None:
                result["bpaf_out"] = self.last_bpaf_out
            if self.use_llrf and getattr(self, "last_llrf_out", None) is not None:
                result["llrf_out"] = self.last_llrf_out
            if (self.use_bpaf or self.use_llrf) and getattr(self, "last_selector_attn", None) is not None:
                result["selector_attn"] = self.last_selector_attn
            if self.use_fpi and fpi_out is not None:
                result["fpi_out"] = fpi_out
            if self.use_fpi_router and router_out is not None:
                result["router_out"] = router_out
            if self.use_pdg and self.last_pdg_assign is not None:
                result["pdg_assign"] = self.last_pdg_assign
                result["pdg_sim"] = self.last_pdg_sim
            if self.use_dcd_residual and dcd_out is not None:
                result["dcd_out"] = dcd_out
            if self.main_fusion_type in ("lrbf", "dcf_lrbf") and getattr(self, "last_main_fusion_out", None) is not None:
                result["main_fusion_out"] = self.last_main_fusion_out
            return result

        if return_encoder_out:
            return outputs, gene_feat, feat, expanded_tissue_idx, cdan_domain_output, drug_pred, encoder_out
        return outputs, gene_feat, feat, expanded_tissue_idx, cdan_domain_output, drug_pred

    def _forward_domain_sep(self, encoder_out, drug_feat_mlp, tissue_idx, alpha,
                           return_encoder_out, return_fused_feature, return_dict, domain_idx=None,
                           frag_fp=None, frag_mask=None, current_epoch=None, drug_names=None):
        """
        Phase 18: Domain-separated forward path.
        Shared branch + Private branch + BML Router fusion.
        CRITICAL: Target (TCGA, domain=1) samples do NOT use private branch
        because private_TCGA was trained on tissue classification only,
        not drug response. Private branch is only helpful for Source (GDSC).
        """
        h_shared = encoder_out["h_shared"]  # [B, d_sep]
        h_private = encoder_out["h_private"]  # [B, d_sep]

        # Optional projection to match gene_feat_dim
        if self.domain_sep_proj is not None:
            h_shared = self.domain_sep_proj(h_shared)
            h_private = self.domain_sep_proj(h_private)

        gene_feat = F.normalize(h_shared, p=2, dim=1)  # for OT alignment

        # Expand per drug
        drug_feat_expand = drug_feat_mlp.unsqueeze(0).expand(
            h_shared.size(0), -1, -1
        ).reshape(-1, drug_feat_mlp.size(-1))  # [B*D, d_drug]

        # Shared path (Bilinear + optional Cross-Attention residual)
        gene_feat_expand_s = h_shared.unsqueeze(1).expand(-1, self.drug_num, -1).reshape(-1, h_shared.size(-1))
        z_s_bilinear = self.bilinear(gene_feat_expand_s, drug_feat_expand)  # [B*D, bilinear_dim]
        if self.use_cross_attn:
            z_s_cross, _ = self.cross_attn_fusion(h_shared, drug_feat_mlp)
            z_s = z_s_bilinear + z_s_cross
        else:
            z_s = z_s_bilinear
        fusion_input_s = torch.cat([z_s, gene_feat_expand_s, drug_feat_expand], dim=1)
        probs_s, feat_s = self.fusion_module(fusion_input_s)

        # Private path
        gene_feat_expand_p = h_private.unsqueeze(1).expand(-1, self.drug_num, -1).reshape(-1, h_private.size(-1))
        z_p_bilinear = self.bilinear(gene_feat_expand_p, drug_feat_expand)  # [B*D, bilinear_dim]
        if self.use_cross_attn:
            z_p_cross, _ = self.cross_attn_fusion(h_private, drug_feat_mlp)
            z_p = z_p_bilinear + z_p_cross
        else:
            z_p = z_p_bilinear
        fusion_input_p = torch.cat([z_p, gene_feat_expand_p, drug_feat_expand], dim=1)
        probs_p, feat_p = self.fusion_module(fusion_input_p)

        # BML Router fusion
        batch_size = h_shared.size(0)
        pred_s_logits = probs_s.view(batch_size, self.drug_num)  # [B, D]
        pred_p_logits = probs_p.view(batch_size, self.drug_num)  # [B, D]

        rel_weights = self.bml_router(
            pred_s_logits.view(-1),
            pred_p_logits.view(-1),
        )  # [B*D, 2]
        rel_weights = rel_weights.view(batch_size, self.drug_num, 2)

        # CRITICAL FIX: Target (TCGA, domain=1) should NOT use private branch
        # because private_TCGA has never seen drug response labels.
        if domain_idx is not None and (domain_idx == 1).all():
            # Target: force shared-only
            rel_weights = torch.zeros_like(rel_weights)
            rel_weights[:, :, 0] = 1.0  # shared weight = 1
            rel_weights[:, :, 1] = 0.0  # private weight = 0

        # Fused logits
        final_logits = rel_weights[:, :, 0] * pred_s_logits + rel_weights[:, :, 1] * pred_p_logits
        probs = final_logits.view(-1, 1)  # [B*D, 1]

        # Phase 9: FPI applies residual on logits (domain-sep compatible)
        effective_frag_fp = frag_fp if frag_fp is not None else self._frag_fp
        effective_frag_mask = frag_mask if frag_mask is not None else self._frag_mask
        if self.use_fpi and effective_frag_fp is not None and effective_frag_mask is not None:
            base_logits = final_logits  # [B, D]

            # Stop-gradient control for FPI branch
            z_fpi = encoder_out["z"]
            q_fpi = encoder_out["q"]
            h_pool_fpi = encoder_out["h"]
            drug_emb_fpi = drug_feat_mlp
            frag_emb_fpi = None

            detach_mode = getattr(self, "fpi_detach_mode", "none")
            if detach_mode == "z":
                z_fpi = z_fpi.detach()
            elif detach_mode == "zq":
                z_fpi = z_fpi.detach()
                q_fpi = q_fpi.detach()
            elif detach_mode == "zq_drug":
                z_fpi = z_fpi.detach()
                q_fpi = q_fpi.detach()
                drug_emb_fpi = drug_emb_fpi.detach()
            elif detach_mode == "all":
                z_fpi = z_fpi.detach()
                q_fpi = q_fpi.detach()
                h_pool_fpi = h_pool_fpi.detach()
                drug_emb_fpi = drug_emb_fpi.detach()
            elif detach_mode != "none":
                raise ValueError(f"Unknown fpi_detach_mode: {detach_mode}")

            frag_emb = self.fragment_encoder(effective_frag_fp, effective_frag_mask)
            frag_emb_fpi = frag_emb
            if detach_mode == "all":
                frag_emb_fpi = frag_emb_fpi.detach()

            fpi_out = self.fpi(
                z=z_fpi,
                q=q_fpi,
                h_pool=h_pool_fpi,
                drug_emb=drug_emb_fpi,
                frag_emb=frag_emb_fpi,
                frag_mask=effective_frag_mask,
            )
            residual_logits = fpi_out["residual_logits"]  # [B, D]
            gamma = fpi_out["gamma"]  # [D]

            final_logits = base_logits + gamma.unsqueeze(0) * residual_logits
            probs = final_logits.view(-1, 1)  # [B*D, 1]

            # Store for diagnostics
            self.last_fpi_out = fpi_out

        # Store for loss computation
        self.last_domain_sep_pred = {
            "pred_shared": pred_s_logits,
            "pred_private": pred_p_logits,
            "rel_weights": rel_weights,
            "feat_shared": feat_s,
            "feat_private": feat_p,
        }

        outputs = torch.sigmoid(probs)
        feat = feat_s  # Use shared feat for downstream (CDAN, etc.)
        fused_feature = feat_s.view(batch_size, self.drug_num, -1)

        # CDAN on shared features
        if self.training:
            drug_pred = self.drug_classifier(feat_s)  # [B*D, drug_num]
            drug_prob = F.softmax(drug_pred, dim=1)
            conditional_feat = torch.cat([feat_s, drug_prob], dim=1)
            reversed_conditional_feat = ReverseLayerF.apply(conditional_feat, alpha)
            cdan_domain_output = self.cdan_discriminator(reversed_conditional_feat)
        else:
            cdan_domain_output = None
            drug_pred = None

        if tissue_idx is not None:
            expanded_tissue_idx = tissue_idx.repeat_interleave(self.drug_num)
        else:
            expanded_tissue_idx = None

        if return_fused_feature or return_dict:
            result = {
                "pred": outputs,
                "final_logits": probs,
                "gene_feat": gene_feat,
                "fused_feature": fused_feature,
                "tissue_idx": expanded_tissue_idx,
                "cdan_domain_output": cdan_domain_output,
                "drug_pred": drug_pred,
                "domain_sep_pred": self.last_domain_sep_pred,
            }
            if return_encoder_out:
                result["encoder_out"] = encoder_out
            return result

        if return_encoder_out:
            return outputs, gene_feat, feat_s, expanded_tissue_idx, cdan_domain_output, drug_pred, encoder_out
        return outputs, gene_feat, feat_s, expanded_tissue_idx, cdan_domain_output, drug_pred

    def encode_contrastive_feature(self, feat, sample=True):
        """
        Phase 12: Encode fused feature for contrastive anchoring via VCB.

        Args:
            feat: [N, F] fused feature (flattened across batch and drugs)
            sample: whether to sample from posterior during training
        Returns:
            dict with keys:
                z: [N, Z] encoded feature for contrastive
                mu: [N, Z] or None
                logvar: [N, Z] or None
                kl: [N] KL divergence per sample
        """
        if not self.use_vcb or self.vcb is None:
            return {
                "z": feat,
                "mu": None,
                "logvar": None,
                "kl": torch.zeros(feat.shape[0], device=feat.device, dtype=feat.dtype),
            }
        return self.vcb(feat, sample=sample)


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DrugGatedProgramAggregator(nn.Module):
    def __init__(self, gate_mode="additive_bias", gate_temperature=5.0,
                 rho_init=0.0, beta_init=0.0, num_programs=8):
        super().__init__()
        self.gate_mode = gate_mode
        self.gate_temperature = gate_temperature
        self.eps = 1e-8

        # Residual scaling (learnable)
        self.rho = nn.Parameter(torch.tensor(float(rho_init)))

        # Beta for additive_bias mode
        if gate_mode in ("additive_bias", "additive_bias_no_residual"):
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        else:
            self.register_buffer('beta', torch.tensor(0.0))

    def forward(self, z, q, h_pool, drug_gate_logits, drug_gate_probs):
        """
        z: [B, K, d]
        q: [B, K]
        h_pool: [B, d]
        drug_gate_logits: [D, K] or None
        drug_gate_probs: [D, K] or None

        Returns:
            h_drug: [B, D, d]
            diagnostics: dict
        """
        B, K, d = z.shape
        D = drug_gate_probs.size(0)

        if self.gate_mode == "hard_multiplicative":
            weight = q.unsqueeze(1) * drug_gate_probs.unsqueeze(0)  # [B, D, K]
            a = weight / (weight.sum(dim=-1, keepdim=True) + self.eps)
            v = (a.unsqueeze(-1) * z.unsqueeze(1)).sum(dim=2)  # [B, D, d]
            h_drug = v

        elif self.gate_mode == "residual_multiplicative":
            weight = q.unsqueeze(1) * drug_gate_probs.unsqueeze(0)
            a = weight / (weight.sum(dim=-1, keepdim=True) + self.eps)
            v = (a.unsqueeze(-1) * z.unsqueeze(1)).sum(dim=2)
            h_drug = h_pool.unsqueeze(1) + self.rho * (v - h_pool.unsqueeze(1))

        elif self.gate_mode in ("additive_bias", "additive_bias_no_residual"):
            log_q = torch.log(q + self.eps)  # [B, K]
            # logits = log_q[:, None, :] + beta * gate_logits[None, :, :] / T
            logits = log_q.unsqueeze(1) + self.beta * drug_gate_logits.unsqueeze(0) / self.gate_temperature
            a = F.softmax(logits, dim=-1)  # [B, D, K]
            v = (a.unsqueeze(-1) * z.unsqueeze(1)).sum(dim=2)  # [B, D, d]

            if self.gate_mode == "additive_bias_no_residual":
                h_drug = v
            else:
                h_drug = h_pool.unsqueeze(1) + self.rho * (v - h_pool.unsqueeze(1))
        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

        # Diagnostics
        with torch.no_grad():
            entropy = -(drug_gate_probs * torch.log(drug_gate_probs + self.eps)).sum(dim=-1)
            effective_programs = (drug_gate_probs > 0.05).float().sum(dim=-1)
            program_usage = drug_gate_probs.mean(dim=0)
            g_norm = F.normalize(drug_gate_probs, p=2, dim=-1)
            cos_sim = torch.mm(g_norm, g_norm.t())
            max_prob = drug_gate_probs.max(dim=-1)[0]

        diagnostics = {
            "rho": self.rho.detach().clone(),
            "beta": self.beta.detach().clone() if hasattr(self.beta, 'detach') else torch.tensor(0.0),
            "gate_entropy": entropy,
            "effective_programs": effective_programs,
            "program_usage": program_usage,
            "drug_gate_cosine": cos_sim,
            "max_prob_per_drug": max_prob,
            "attention_weights": a.detach().clone(),
        }

        return h_drug, diagnostics


# =============================================================================
# Phase 3A: Disentangled Program Encoder
# =============================================================================

class DisentangledProgramEncoder(nn.Module):
    """
    Wrap MolecularProgramEncoder and add response/domain projection heads.
    Phase 3A-R: residual-preserving disentanglement.

    Input:
        x: [B, G]
    Output dict:
        z_base:     [B, K, d]
        q:          [B, K]
        h_base:     [B, d]
        z_response: [B, K, d]
        z_domain:   [B, K, d]
        h_response: [B, d]
        h_domain:   [B, d]
        h_pred:     [B, d]  (h_base + alpha * h_response)
        residual_alpha: scalar or [d]
    """

    def __init__(
        self,
        base_encoder,
        response_head_hidden_dim=None,
        residual_mode="learnable_scalar",
        residual_alpha_init=0.0,
        residual_alpha_fixed=0.1,
    ):
        super(DisentangledProgramEncoder, self).__init__()
        self.base_encoder = base_encoder
        d = base_encoder.program_dim
        self.residual_mode = residual_mode

        if response_head_hidden_dim is None:
            response_head_hidden_dim = d

        self.response_head = nn.Sequential(
            nn.Linear(d, response_head_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(response_head_hidden_dim, d, bias=True),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(d, response_head_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(response_head_hidden_dim, d, bias=True),
        )

        # Residual scaling alpha
        if residual_mode == "fixed":
            self.register_buffer("residual_alpha", torch.tensor(residual_alpha_fixed))
        elif residual_mode == "learnable_scalar":
            self.residual_alpha = nn.Parameter(torch.tensor(residual_alpha_init))
        elif residual_mode == "learnable_vector":
            self.residual_alpha = nn.Parameter(torch.zeros(d))
        else:
            raise ValueError(f"Unknown residual_mode: {residual_mode}")

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.base_encoder(x)
        z_base = out["z"]
        q = out["q"]
        h_base = out["h"]

        z_response = self.response_head(z_base)
        z_domain = self.domain_head(z_base)

        h_response = (q.unsqueeze(-1) * z_response).sum(dim=1)
        h_domain = (q.unsqueeze(-1) * z_domain).sum(dim=1)

        if self.residual_mode == "learnable_vector":
            alpha = self.residual_alpha.unsqueeze(0)  # [1, d]
            h_pred = h_base + alpha * h_response
        else:
            h_pred = h_base + self.residual_alpha * h_response

        return {
            "z_base": z_base,
            "q": q,
            "h_base": h_base,
            "z_response": z_response,
            "z_domain": z_domain,
            "h_response": h_response,
            "h_domain": h_domain,
            "h_pred": h_pred,
            "residual_alpha": self.residual_alpha.detach().clone(),
        }


class BinaryDomainDiscriminator(nn.Module):
    """
    Simple MLP domain discriminator.
    Input:  [B, d]
    Output: [B, 1] (logits)
    """

    def __init__(self, input_dim, hidden_dim=64, drop=0.2):
        super(BinaryDomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, 1, bias=True),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class DisentangledProgramTissueClassifier(nn.Module):
    """
    Stage 1 drop-in replacement that uses DisentangledProgramEncoder.

    Interface is intentionally a superset of ProgramTissueClassifier:
        forward(x, alpha=1.0) -> dict with tissue_pred, domain_pred, gene_feat, etc.

    Key differences:
    - tissue classifier operates on h_pred (not h_pool)
    - original domain discriminator operates on h_pred (preserved for compatibility)
    - NEW: response_domain_discriminator on GRL(h_response)
    - NEW: domain_nuisance_discriminator on h_domain
    """

    def __init__(
        self,
        input_dim,
        num_tissues,
        num_programs=8,
        program_dim=64,
        hidden_dims=None,
        drop=0.2,
        act_fn=nn.SELU,
        disentangle_hidden_dim=None,
        residual_mode="learnable_scalar",
        residual_alpha_init=0.0,
        residual_alpha_fixed=0.1,
        # Phase 5 backbone args
        backbone_type: str = "mlp",
        norm_type: str = "layernorm",
        num_gene_tokens: int = 64,
        token_dim: int = 64,
        convnext_depth: int = 2,
        convnext_kernel_size: int = 7,
        convnext_expansion: int = 4,
    ):
        super(DisentangledProgramTissueClassifier, self).__init__()

        base_encoder = MolecularProgramEncoder(
            input_dim=input_dim,
            num_programs=num_programs,
            program_dim=program_dim,
            hidden_dims=hidden_dims,
            drop=drop,
            act_fn=act_fn,
            backbone_type=backbone_type,
            norm_type=norm_type,
            num_gene_tokens=num_gene_tokens,
            token_dim=token_dim,
            convnext_depth=convnext_depth,
            convnext_kernel_size=convnext_kernel_size,
            convnext_expansion=convnext_expansion,
        )
        self.program_encoder = DisentangledProgramEncoder(
            base_encoder=base_encoder,
            response_head_hidden_dim=disentangle_hidden_dim,
            residual_mode=residual_mode,
            residual_alpha_init=residual_alpha_init,
            residual_alpha_fixed=residual_alpha_fixed,
        )
        feat_dim = program_dim

        # Tissue classifier (on h_pred)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.BatchNorm1d(feat_dim // 2),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(feat_dim // 2, num_tissues),
        )

        # Original domain discriminator on h_pred (preserved flow)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.BatchNorm1d(feat_dim // 2),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(feat_dim // 2, 1),
        )

        # Phase 3A: response branch adversarial discriminator
        self.response_domain_discriminator = BinaryDomainDiscriminator(feat_dim)

        # Phase 3A: domain-nuisance discriminator
        self.domain_nuisance_discriminator = BinaryDomainDiscriminator(feat_dim)

    def forward(self, x, alpha=1.0):
        out = self.program_encoder(x)
        h_pred = out["h_pred"]
        h_response = out["h_response"]
        h_domain = out["h_domain"]

        tissue_pred = self.classifier(h_pred)

        # Original domain discriminator (h_pred + GRL)
        reversed_h_pred = GradientReversal.apply(h_pred, alpha)
        domain_pred = self.domain_discriminator(reversed_h_pred)

        # Response branch: GRL + domain discriminator
        reversed_h_response = GradientReversal.apply(h_response, alpha)
        response_domain_pred = self.response_domain_discriminator(reversed_h_response)

        # Domain-nuisance branch: normal domain discriminator
        domain_nuisance_pred = self.domain_nuisance_discriminator(h_domain)

        return {
            "tissue_pred": tissue_pred,
            "domain_pred": domain_pred,
            "response_domain_pred": response_domain_pred,
            "domain_nuisance_pred": domain_nuisance_pred,
            "gene_feat": h_pred,          # backward-compatible
            "h_response": h_response,
            "h_domain": h_domain,
            "z_response": out["z_response"],
            "z_domain": out["z_domain"],
            "q": out["q"],
            "z_base": out["z_base"],
            "h_base": out["h_base"],
        }


class DisentangledProgramFusionMLP(nn.Module):
    """
    Stage 2/3 drop-in replacement that uses DisentangledProgramEncoder.

    Identical to ProgramFusionMLP except:
    - encoder is expected to be a DisentangledProgramEncoder
    - gene_feat = encoder_out["h_pred"] instead of encoder_out["h"]
    """

    def __init__(
        self,
        encoder,
        input_dim,
        output_dim,
        hidden_dims=None,
        gene_num=1306,
        bilinear_dim=128,
        gene_feat_dim=64,
        drug_feat_dim=64,
        drug_num=9,
        drop=0.1,
        act_fn=nn.SELU,
        use_cross_attn=False,
        cross_attn_num_heads=4,
        **kwargs,
    ):
        super(DisentangledProgramFusionMLP, self).__init__()
        self.encoder = encoder
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drug_num = drug_num
        self.drop = drop
        self.use_cross_attn = use_cross_attn

        self.encoder_input_dim = gene_num
        self.attention_masks = nn.Parameter(
            torch.ones(len(config.tissue_map), self.encoder_input_dim)
        )
        self.attention_activation = nn.Sigmoid()

        # Lazy import to avoid pulling in the full original model graph
        from models import FeatMLP, Drug_MLP

        self.drug_mlp = Drug_MLP(
            input_dim=128,
            output_dim=64,
            hidden_dims=[128, 64],
            drop=0.1,
        )

        self.bilinear = nn.Bilinear(gene_feat_dim, drug_feat_dim, bilinear_dim)

        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.fusion_module = FeatMLP(
            input_dim, output_dim, hidden_dims, drop=drop, act_fn=act_fn
        )

        cdan_input_dim = hidden_dims[-1] + drug_num
        self.cdan_discriminator = nn.Sequential(
            nn.Linear(cdan_input_dim, cdan_input_dim // 2),
            nn.BatchNorm1d(cdan_input_dim // 2),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(cdan_input_dim // 2, cdan_input_dim // 4),
            nn.BatchNorm1d(cdan_input_dim // 4),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(cdan_input_dim // 4, 1),
        )

        self.drug_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.SELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dims[-1] // 2, drug_num),
        )

    def forward(self, inputs, drug_feat, tissue_idx, alpha=1.0):
        if tissue_idx is None:
            attention_weights = self.attention_activation(
                self.attention_masks.mean(dim=0)
            )
        else:
            attention_weights = self.attention_activation(
                self.attention_masks[tissue_idx]
            )

        inputs = inputs * attention_weights
        encoder_out = self.encoder(inputs)
        gene_feat = encoder_out["h_pred"]  # <-- Phase 3A: use h_pred
        gene_feat = F.normalize(gene_feat, p=2, dim=1)

        drug_feat, _ = self.drug_mlp(drug_feat)
        drug_feat = F.normalize(drug_feat, p=2, dim=1)

        gene_feat_expand = gene_feat.unsqueeze(1).expand(
            -1, self.drug_num, -1
        )
        drug_feat_expand = drug_feat.unsqueeze(0).expand(
            gene_feat.size(0), -1, -1
        )

        if self.use_cross_attn:
            z, _ = self.cross_attn_fusion(gene_feat, drug_feat)
        else:
            z = self.bilinear(
                gene_feat_expand.reshape(-1, gene_feat_expand.size(-1)),
                drug_feat_expand.reshape(-1, drug_feat_expand.size(-1)),
            )

        fusion_input = torch.cat(
            [
                z,
                gene_feat_expand.reshape(-1, gene_feat_expand.size(-1)),
                drug_feat_expand.reshape(-1, drug_feat_expand.size(-1)),
            ],
            dim=1,
        )

        probs, feat = self.fusion_module(fusion_input)
        outputs = torch.sigmoid(probs)

        # CDAN
        if self.training:
            drug_pred = self.drug_classifier(feat)
            drug_prob = F.softmax(drug_pred, dim=1)

            conditional_feat = torch.cat([feat, drug_prob], dim=1)

            reversed_conditional_feat = ReverseLayerF.apply(conditional_feat, alpha)
            cdan_domain_output = self.cdan_discriminator(reversed_conditional_feat)
        else:
            cdan_domain_output = None
            drug_pred = None

        if tissue_idx is not None:
            expanded_tissue_idx = tissue_idx.repeat_interleave(self.drug_num)
        else:
            expanded_tissue_idx = None
        return outputs, gene_feat, feat, expanded_tissue_idx, cdan_domain_output, drug_pred
