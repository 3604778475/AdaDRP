"""
MMPA-DRP Extension Models (cleaned)
====================================
Kept modules for the current configuration:
- MolecularProgramEncoder
- DomainSeparatedEncoder
- ProgramTissueClassifier
- BMLReliabilityRouter
- ProgramFusionMLP

Phase 5: Added backbone_type support (mlp / grn_mlp / convnext1d)
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import config

from mmpa_drp.backbones import GRNMLPBackbone, ConvNeXt1DBackbone, ResNetBackbone
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

        elif backbone_type == "resnet":
            # Original MACB ResNet backbone for ablation
            resnet_hidden_dims = hidden_dims if hidden_dims else [512, 256, 128, 64]
            self.backbone = ResNetBackbone(
                input_dim=input_dim,
                hidden_dims=resnet_hidden_dims,
                drop=drop,
            )
            self.backbone_output_dim = resnet_hidden_dims[-1]

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
        # Hallmark gene-set encoder config
        program_encoder_type: str = "latent",
        gene_set_mask_path: str = None,
        gene_set_names_path: str = None,
        gene_set_token_dropout: float = 0.1,
        use_pathway_embedding: bool = True,
        use_soft_gene_set_mask: bool = False,
        soft_mask_scale: float = 0.05,
        latent_num_programs: int = 3,
        gene_set_num_final_programs: int = 3,
        selector_num_heads: int = 4,
        selector_dropout: float = 0.1,
        use_query_gate: bool = False,
        lambda_query_gate_l1: float = 0.01,
    ):
        super(ProgramTissueClassifier, self).__init__()

        if program_encoder_type != "latent":
            raise ValueError(
                f"Only program_encoder_type='latent' is supported in this cleaned version, "
                f"got: {program_encoder_type}"
            )

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

        print(f"[ProgramTissueClassifier] program_encoder_type={program_encoder_type}, "
              f"num_programs={self.program_encoder.num_programs}")
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

        # Save selector attention for interpretability
        self.last_selector_attn = out.get("selector_attn", None)

        if self.use_domain_sep and "h_shared" in out:
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


class ProgramFusionMLP(nn.Module):
    """
    Stage 2/3 drop-in replacement for Fusion_MLP.
    Cleaned version for: program encoder + domain separation + FPI + CDAN.
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
        use_fpi=False,
        fpi_config=None,
        frag_fp=None,
        frag_mask=None,
        **kwargs,
    ):
        super(ProgramFusionMLP, self).__init__()
        self.encoder = encoder
        self.num_programs = getattr(encoder, 'num_programs', 8)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drug_num = drug_num
        self.drop = drop
        self.use_fpi = use_fpi
        self.fpi_config = fpi_config or {}
        self.return_dict = False
        self.fragment_encoder = None
        self.fpi = None
        self.register_buffer("_frag_fp", None)
        self.register_buffer("_frag_mask", None)

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

        # Phase 9: FPI modules
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
                num_programs=fpi_cfg.get("num_programs", self.num_programs),
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

        self.bilinear = nn.Bilinear(gene_feat_dim, drug_feat_dim, bilinear_dim)

        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.fusion_module = FeatMLP(
            input_dim, output_dim, hidden_dims, drop=drop, act_fn=act_fn
        )
        self.fused_feature_dim = hidden_dims[-1]

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
            ds_out_dim = encoder.domain_sep.shared_mlp[4].out_features
            if ds_out_dim != gene_feat_dim:
                self.domain_sep_proj = nn.Sequential(
                    nn.Linear(ds_out_dim, gene_feat_dim),
                    nn.BatchNorm1d(gene_feat_dim),
                    nn.SELU(),
                )
            self.bml_router = BMLReliabilityRouter(hidden_dim=16)
            self.last_domain_sep_pred = None

    def encode_drug(self, drug_feat):
        """
        Drug encoding interface (original Drug_MLP path only).
        Output shape: [drug_num, drug_emb_dim]
        """
        r_fp, _ = self.drug_mlp(drug_feat)
        drug_emb = F.normalize(r_fp, p=2, dim=1)

        if not hasattr(self, "_drug_enc_logged"):
            self._drug_enc_logged = True
            print(f"[Drug Encoder] drug_feat shape={drug_feat.shape}")
            print(f"[Drug Encoder] final drug_emb shape={drug_emb.shape}")

        return drug_emb

    def forward(self, inputs, drug_feat, tissue_idx, alpha=1.0,
                return_encoder_out=False, return_fused_feature=False, return_dict=False,
                frag_fp=None, frag_mask=None, frag_smiles=None, current_epoch=None,
                drug_names=None, domain_idx=None):
        if tissue_idx is None:
            attention_weights = self.attention_activation(self.attention_masks.mean(dim=0))
        else:
            attention_weights = self.attention_activation(self.attention_masks[tissue_idx])

        inputs = inputs * attention_weights
        encoder_out = self.encoder(inputs, domain_idx=domain_idx)
        drug_feat_mlp = self.encode_drug(drug_feat)  # [drug_num, D]

        # Phase 18: Domain-separated fast path
        if self.use_domain_sep:
            return self._forward_domain_sep(
                encoder_out, drug_feat_mlp, tissue_idx, alpha,
                return_encoder_out, return_fused_feature, return_dict,
                domain_idx=domain_idx,
                frag_fp=frag_fp, frag_mask=frag_mask,
            )

        # Base path: h_pool + bilinear fusion
        gene_feat = F.normalize(encoder_out["h"], p=2, dim=1)
        gene_feat_for_bilinear = gene_feat.unsqueeze(1).expand(
            -1, self.drug_num, -1
        ).reshape(-1, gene_feat.size(-1))  # [B*D, d]

        drug_feat_expand = drug_feat_mlp.unsqueeze(0).expand(
            gene_feat.size(0), -1, -1
        ).reshape(-1, drug_feat_mlp.size(-1))  # [B*D, d_drug]

        z = self.bilinear(gene_feat_for_bilinear, drug_feat_expand)  # [B*D, bilinear_dim]

        fusion_input = torch.cat([z, gene_feat_for_bilinear, drug_feat_expand], dim=1)
        probs, feat = self.fusion_module(fusion_input)

        # Phase 9: FPI applies residual on logits
        fpi_out = None
        effective_frag_fp = frag_fp if frag_fp is not None else self._frag_fp
        effective_frag_mask = frag_mask if frag_mask is not None else self._frag_mask
        if self.use_fpi and effective_frag_fp is not None and effective_frag_mask is not None:
            batch_size = gene_feat_for_bilinear.size(0) // self.drug_num
            base_logits = probs.view(batch_size, self.drug_num)  # [B, D]

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

            self.last_fpi_out = fpi_out

        outputs = torch.sigmoid(probs)

        batch_size = gene_feat_for_bilinear.size(0) // self.drug_num
        fused_feature = feat.view(batch_size, self.drug_num, -1)

        # CDAN
        if self.training:
            drug_pred = self.drug_classifier(feat)  # [B*drug_num, drug_num]
            drug_prob = F.softmax(drug_pred, dim=1)  # [B*drug_num, drug_num]
            conditional_feat = torch.cat([feat, drug_prob], dim=1)
            reversed_conditional_feat = ReverseLayerF.apply(conditional_feat, alpha)
            cdan_domain_output = self.cdan_discriminator(reversed_conditional_feat)
        else:
            cdan_domain_output = None
            drug_pred = None

        expanded_tissue_idx = tissue_idx.repeat_interleave(self.drug_num) if tissue_idx is not None else None

        if return_fused_feature or return_dict:
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
            if self.use_fpi and fpi_out is not None:
                result["fpi_out"] = fpi_out
            return result

        if return_encoder_out:
            return outputs, gene_feat, feat, expanded_tissue_idx, cdan_domain_output, drug_pred, encoder_out
        return outputs, gene_feat, feat, expanded_tissue_idx, cdan_domain_output, drug_pred

    def _forward_domain_sep(self, encoder_out, drug_feat_mlp, tissue_idx, alpha,
                           return_encoder_out, return_fused_feature, return_dict,
                           domain_idx=None, frag_fp=None, frag_mask=None):
        """
        Phase 18: Domain-separated forward path.
        Shared branch + Private branch + BML Router fusion.
        Target (TCGA, domain=1) samples are forced to shared-only.
        """
        h_shared = encoder_out["h_shared"]  # [B, d_sep]
        h_private = encoder_out["h_private"]  # [B, d_sep]

        if self.domain_sep_proj is not None:
            h_shared = self.domain_sep_proj(h_shared)
            h_private = self.domain_sep_proj(h_private)

        gene_feat = F.normalize(h_shared, p=2, dim=1)  # for OT alignment

        drug_feat_expand = drug_feat_mlp.unsqueeze(0).expand(
            h_shared.size(0), -1, -1
        ).reshape(-1, drug_feat_mlp.size(-1))  # [B*D, d_drug]

        # Shared path
        gene_feat_expand_s = h_shared.unsqueeze(1).expand(-1, self.drug_num, -1).reshape(-1, h_shared.size(-1))
        z_s = self.bilinear(gene_feat_expand_s, drug_feat_expand)  # [B*D, bilinear_dim]
        fusion_input_s = torch.cat([z_s, gene_feat_expand_s, drug_feat_expand], dim=1)
        probs_s, feat_s = self.fusion_module(fusion_input_s)

        # Private path
        gene_feat_expand_p = h_private.unsqueeze(1).expand(-1, self.drug_num, -1).reshape(-1, h_private.size(-1))
        z_p = self.bilinear(gene_feat_expand_p, drug_feat_expand)  # [B*D, bilinear_dim]
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

        # Target: force shared-only
        if domain_idx is not None and (domain_idx == 1).all():
            rel_weights = torch.zeros_like(rel_weights)
            rel_weights[:, :, 0] = 1.0
            rel_weights[:, :, 1] = 0.0

        final_logits = rel_weights[:, :, 0] * pred_s_logits + rel_weights[:, :, 1] * pred_p_logits
        probs = final_logits.view(-1, 1)  # [B*D, 1]

        # Phase 9: FPI applies residual on logits (domain-sep compatible)
        effective_frag_fp = frag_fp if frag_fp is not None else self._frag_fp
        effective_frag_mask = frag_mask if frag_mask is not None else self._frag_mask
        if self.use_fpi and effective_frag_fp is not None and effective_frag_mask is not None:
            base_logits = final_logits  # [B, D]

            z_fpi = encoder_out["z"]
            q_fpi = encoder_out["q"]
            h_pool_fpi = encoder_out["h"]
            drug_emb_fpi = drug_feat_mlp
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
            self.last_fpi_out = fpi_out

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

        expanded_tissue_idx = tissue_idx.repeat_interleave(self.drug_num) if tissue_idx is not None else None

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
        VCB is disabled in this cleaned version; returns identity.
        """
        return {
            "z": feat,
            "mu": None,
            "logvar": None,
            "kl": torch.zeros(feat.shape[0], device=feat.device, dtype=feat.dtype),
        }


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


