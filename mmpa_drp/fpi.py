"""
FragmentProgramInteraction (FPI)
================================
Signal-inspired fragment-program interaction.

Given cell program tokens z and drug fragment embeddings,
learns fine-grained program-fragment interactions and produces
a residual logit added to the base bilinear fusion logit.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPISIMInteraction(nn.Module):
    """
    FPI-SIM: Signal-style Selective Interaction.

    Replaces global softmax over all program-fragment pairs with:
        - program intra selection (h_pool selects important programs)
        - fragment intra selection (drug_emb selects important fragments)
        - program-fragment inter selection (cross-modal pair attention)
        - soft-union (combines intra + inter weights)
        - selected interaction pooling

    Input:
        z:          [B, K, H]
        q:          [B, K]
        h_pool:     [B, H]
        drug_emb:   [D, H]
        frag_emb:   [D, M, H]
        frag_mask:  [D, M]

    Output dict:
        u:                  [B, D, H]
        attn:               [B, D, K, M]
        cross_attn:         [B, D, K, M]
        program_weight:     [B, K]
        fragment_weight:    [D, M]
        union_weight:       [B, D, K, M]
        score_cross:        [B, D, K, M]
        score_program:      [B, K]
        score_fragment:     [D, M]
        residual_logits:    [B, D]
        gamma:              [D]
    """

    def __init__(
        self,
        program_dim=64,
        drug_dim=64,
        fragment_dim=64,
        num_programs=8,
        num_drugs=9,
        max_fragments=16,
        q_prior_weight=0.01,
        gamma_max=0.05,
        gamma_init=0.0,
        dropout=0.1,
        program_temperature=1.0,
        fragment_temperature=1.0,
        cross_temperature=1.0,
        union_alpha=1.0,
        union_beta=1.0,
        use_modal_interaction=False,
        eps=1e-8,
    ):
        super(FPISIMInteraction, self).__init__()
        self.program_dim = program_dim
        self.drug_dim = drug_dim
        self.fragment_dim = fragment_dim
        self.num_programs = num_programs
        self.num_drugs = num_drugs
        self.max_fragments = max_fragments
        self.q_prior_weight = q_prior_weight
        self.gamma_max = gamma_max
        self.gamma_init = gamma_init
        self.dropout = dropout
        self.program_temperature = program_temperature
        self.fragment_temperature = fragment_temperature
        self.cross_temperature = cross_temperature
        self.union_alpha = union_alpha
        self.union_beta = union_beta
        self.use_modal_interaction = use_modal_interaction
        self.eps = eps

        # 4.1 Program intra selection
        self.hpool_proj = nn.Linear(program_dim, program_dim, bias=False)
        self.program_intra_proj = nn.Linear(program_dim, program_dim, bias=False)

        # 4.2 Fragment intra selection
        self.drug_proj = nn.Linear(drug_dim, program_dim, bias=False)
        self.fragment_intra_proj = nn.Linear(fragment_dim, program_dim, bias=False)

        # 4.3 Program-fragment inter selection
        self.program_cross_proj = nn.Linear(program_dim, program_dim, bias=False)
        self.fragment_cross_proj = nn.Linear(fragment_dim, program_dim, bias=False)

        # Pair MLP: operates on concatenated [z, r, z*r]
        pair_input_dim = program_dim * 3
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, program_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(program_dim, program_dim),
            nn.LayerNorm(program_dim),
        )

        # Residual head: u -> residual logit
        self.residual_head = nn.Sequential(
            nn.Linear(program_dim, program_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(program_dim, 1),
        )

        # Small init on final residual head output to reduce initial perturbation
        nn.init.normal_(self.residual_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual_head[-1].bias)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, q, h_pool, drug_emb, frag_emb, frag_mask):
        B, K, H = z.shape
        D, M, _ = frag_emb.shape

        # 4.1 Program intra selection
        hq = self.hpool_proj(h_pool)                  # [B, H]
        zk = self.program_intra_proj(z)               # [B, K, H]
        score_program = torch.einsum("bh,bkh->bk", hq, zk) / math.sqrt(H)

        if self.q_prior_weight > 0:
            q_log = torch.log(q + self.eps)           # [B, K]
            score_program = score_program + self.q_prior_weight * q_log

        program_weight = F.softmax(score_program / self.program_temperature, dim=-1)  # [B, K]

        # 4.2 Fragment intra selection
        dq = self.drug_proj(drug_emb)                 # [D, H]
        fk = self.fragment_intra_proj(frag_emb)       # [D, M, H]
        score_fragment = torch.einsum("dh,dmh->dm", dq, fk) / math.sqrt(H)

        if frag_mask is not None:
            score_fragment = score_fragment.masked_fill(frag_mask == 0, -1e9)

        fragment_weight = F.softmax(score_fragment / self.fragment_temperature, dim=-1)  # [D, M]

        # 4.3 Program-fragment inter selection
        zq = self.program_cross_proj(z)               # [B, K, H]
        fk_cross = self.fragment_cross_proj(frag_emb)  # [D, M, H]
        score_cross = torch.einsum("bkh,dmh->bdkm", zq, fk_cross) / math.sqrt(H)

        if frag_mask is not None:
            score_cross = score_cross.masked_fill(
                frag_mask[None, :, None, :] == 0,
                -1e9,
            )

        score_cross_flat = score_cross.reshape(B, D, K * M)
        cross_attn_flat = F.softmax(score_cross_flat / self.cross_temperature, dim=-1)
        cross_attn = cross_attn_flat.reshape(B, D, K, M)

        # 4.4 Soft-union
        p = program_weight[:, None, :, None]          # [B, 1, K, 1]
        f = fragment_weight[None, :, None, :]         # [1, D, 1, M]

        union_weight = cross_attn * (1.0 + self.union_alpha * p) * (1.0 + self.union_beta * f)

        if frag_mask is not None:
            union_weight = union_weight * frag_mask[None, :, None, :]

        # Normalize over K*M
        union_sum = union_weight.sum(dim=(2, 3), keepdim=True)
        attn = union_weight / (union_sum + self.eps)

        # 4.5 Pair representation
        z_expand = z[:, None, :, None, :]             # [B, 1, K, 1, H]
        r_expand = frag_emb[None, :, None, :, :]      # [1, D, 1, M, H]

        pair = torch.cat([
            z_expand.expand(-1, D, -1, M, -1),
            r_expand.expand(B, -1, K, -1, -1),
            z_expand.expand(-1, D, -1, M, -1) * r_expand.expand(B, -1, K, -1, -1),
        ], dim=-1)  # [B, D, K, M, 3H]

        pair_hidden = self.pair_mlp(pair)  # [B, D, K, M, H]

        # 4.6 Selected interaction pooling
        u = (attn.unsqueeze(-1) * pair_hidden).sum(dim=(2, 3))  # [B, D, H]

        # 4.7 Residual logits (gamma applied by FragmentProgramInteraction)
        residual_logits = self.residual_head(u).squeeze(-1)  # [B, D]
        residual_logits = torch.tanh(residual_logits)

        return {
            "u": u,
            "attn": attn,
            "cross_attn": cross_attn,
            "program_weight": program_weight,
            "fragment_weight": fragment_weight,
            "union_weight": union_weight,
            "score_cross": score_cross,
            "score_program": score_program,
            "score_fragment": score_fragment,
            "residual_logits": residual_logits,
        }


class FragmentProgramInteraction(nn.Module):
    """
    Fragment-Program Interaction module.
    Routes to FPISIMInteraction when use_fpi_sim=True,
    otherwise uses original global softmax FPI.

    Input:
        z:          [B, K, H]
        q:          [B, K]
        h_pool:     [B, H]
        drug_emb:   [D, H]
        frag_emb:   [D, M, H]
        frag_mask:  [D, M]

    Output dict:
        u:                  [B, D, H]
        attn:               [B, D, K, M]
        score:              [B, D, K, M]
        residual_logits:    [B, D]
        gamma:              [D]
    """

    def __init__(
        self,
        program_dim=64,
        drug_dim=64,
        fragment_dim=64,
        num_programs=8,
        num_drugs=9,
        max_fragments=16,
        q_prior_weight=0.01,
        gamma_max=0.05,
        gamma_init=0.0,
        dropout=0.1,
        eps=1e-8,
        use_fpi_sim=False,
        fpi_sim_config=None,
    ):
        super(FragmentProgramInteraction, self).__init__()
        self.program_dim = program_dim
        self.drug_dim = drug_dim
        self.fragment_dim = fragment_dim
        self.num_programs = num_programs
        self.num_drugs = num_drugs
        self.max_fragments = max_fragments
        self.q_prior_weight = q_prior_weight
        self.gamma_max = gamma_max
        self.gamma_init = gamma_init
        self.dropout = dropout
        self.eps = eps
        self.use_fpi_sim = use_fpi_sim

        if self.use_fpi_sim:
            sim_cfg = fpi_sim_config or {}
            self.sim_interaction = FPISIMInteraction(
                program_dim=program_dim,
                drug_dim=drug_dim,
                fragment_dim=fragment_dim,
                num_programs=num_programs,
                num_drugs=num_drugs,
                max_fragments=max_fragments,
                q_prior_weight=sim_cfg.get("q_prior_weight", q_prior_weight),
                dropout=dropout,
                program_temperature=sim_cfg.get("program_temperature", 1.0),
                fragment_temperature=sim_cfg.get("fragment_temperature", 1.0),
                cross_temperature=sim_cfg.get("cross_temperature", 1.0),
                union_alpha=sim_cfg.get("union_alpha", 1.0),
                union_beta=sim_cfg.get("union_beta", 1.0),
                use_modal_interaction=sim_cfg.get("use_modal_interaction", False),
                eps=eps,
            )
        else:
            self.sim_interaction = None

            # Original global FPI modules
            self.program_proj = nn.Linear(program_dim, program_dim, bias=False)
            self.fragment_proj = nn.Linear(fragment_dim, program_dim, bias=False)

            pair_input_dim = program_dim * 3
            self.pair_mlp = nn.Sequential(
                nn.Linear(pair_input_dim, program_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(program_dim, program_dim),
                nn.LayerNorm(program_dim),
            )

            self.residual_head = nn.Sequential(
                nn.Linear(program_dim, program_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(program_dim, 1),
            )

            nn.init.normal_(self.residual_head[-1].weight, std=1e-3)
            nn.init.zeros_(self.residual_head[-1].bias)

            self.reset_parameters()

        # Gamma is always owned by FragmentProgramInteraction regardless of mode
        self.raw_gamma = nn.Parameter(torch.zeros(num_drugs))
        self._init_gamma()

    def _init_gamma(self):
        if self.gamma_init <= 0:
            init_val = -10.0
        else:
            ratio = self.gamma_init / self.gamma_max
            ratio = max(min(ratio, 0.999), 0.001)
            init_val = math.log(ratio / (1.0 - ratio))
        with torch.no_grad():
            self.raw_gamma.fill_(init_val)

    def get_gamma(self):
        return torch.sigmoid(self.raw_gamma) * self.gamma_max

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_global(self, z, q, h_pool, drug_emb, frag_emb, frag_mask):
        """Original global softmax FPI (Phase 9 baseline)."""
        B, K, H = z.shape
        D, M, _ = frag_emb.shape

        prog_q = self.program_proj(z)
        frag_k = self.fragment_proj(frag_emb)

        score = torch.einsum("bkh,dmh->bdkm", prog_q, frag_k) / math.sqrt(H)

        if self.q_prior_weight > 0:
            q_log = torch.log(q + self.eps)
            score = score + self.q_prior_weight * q_log[:, None, :, None]

        if frag_mask is not None:
            score = score.masked_fill(frag_mask[None, :, None, :] == 0, -1e9)

        score_flat = score.reshape(B, D, K * M)
        attn_flat = F.softmax(score_flat, dim=-1)
        attn = attn_flat.reshape(B, D, K, M)

        z_expand = z[:, None, :, None, :]
        r_expand = frag_emb[None, :, None, :, :]

        pair = torch.cat([
            z_expand.expand(-1, D, -1, M, -1),
            r_expand.expand(B, -1, K, -1, -1),
            z_expand.expand(-1, D, -1, M, -1) * r_expand.expand(B, -1, K, -1, -1),
        ], dim=-1)

        pair_hidden = self.pair_mlp(pair)
        u = (attn.unsqueeze(-1) * pair_hidden).sum(dim=(2, 3))

        residual_logits = self.residual_head(u).squeeze(-1)
        residual_logits = torch.tanh(residual_logits)
        gamma = self.get_gamma()

        return {
            "u": u,
            "attn": attn,
            "score": score,
            "residual_logits": residual_logits,
            "gamma": gamma,
        }

    def forward(self, z, q, h_pool, drug_emb, frag_emb, frag_mask):
        if self.use_fpi_sim:
            sim_out = self.sim_interaction(z, q, h_pool, drug_emb, frag_emb, frag_mask)
            sim_out["gamma"] = self.get_gamma()
            return sim_out
        else:
            out = self.forward_global(z, q, h_pool, drug_emb, frag_emb, frag_mask)
            out["gamma"] = self.get_gamma()
            return out


class DualFPIResidualRouter(nn.Module):
    """
    Dual FPI Residual Router.

    Combines two FPI residual branches:
        - Global branch: global softmax over all program-fragment pairs
        - Cross branch: FPI-SIM cross-only (union_alpha=0, union_beta=0)

    Drug-wise gamma gates control each branch's contribution:
        final_logits = base_logits
                       + gamma_global * residual_global
                       + effective_gamma_cross * residual_cross

    Input:
        z:          [B, K, H]
        q:          [B, K]
        h_pool:     [B, H]
        drug_emb:   [D, H]
        frag_emb:   [D, M, H]
        frag_mask:  [D, M]
        base_logits: [B, D]
        current_epoch: Optional[int]
        drug_names: Optional[List[str]]

    Output dict:
        final_logits:               [B, D]
        residual_global:            [B, D]
        residual_cross:             [B, D]
        gamma_global:               [D]
        gamma_cross:                [D]
        effective_gamma_cross:      [D]
        global_out:                 dict
        cross_out:                  dict
        cross_active_mask:          [D]
        branch_global_effect:       [D]
        branch_cross_effect:        [D]
        branch_global_ratio:        [D]
        branch_cross_ratio:         [D]
        branch_preference:          List[str]
        residual_global_to_base_ratio: [D]
        residual_cross_to_base_ratio:  [D]
    """

    def __init__(
        self,
        program_dim=64,
        drug_dim=64,
        fragment_dim=64,
        num_programs=8,
        num_drugs=9,
        max_fragments=16,
        q_prior_weight=0.01,
        gamma_global_max=0.05,
        gamma_global_init=0.0,
        gamma_cross_max=0.05,
        gamma_cross_init=0.0,
        dropout=0.1,
        router_cross_start_epoch=0,
        router_cross_ramp_epochs=0,
        router_cross_mode="learnable",
        router_oracle_cross_drugs="",
        router_residual_global_scale=1.0,
        router_residual_cross_scale=1.0,
        router_branch_effect_threshold=1e-4,
        router_cross_detach_mode="none",
        debug_fixed_gamma_global=None,
        debug_fixed_gamma_cross=None,
        eps=1e-8,
    ):
        super(DualFPIResidualRouter, self).__init__()
        self.program_dim = program_dim
        self.drug_dim = drug_dim
        self.fragment_dim = fragment_dim
        self.num_programs = num_programs
        self.num_drugs = num_drugs
        self.max_fragments = max_fragments
        self.q_prior_weight = q_prior_weight
        self.gamma_global_max = gamma_global_max
        self.gamma_global_init = gamma_global_init
        self.gamma_cross_max = gamma_cross_max
        self.gamma_cross_init = gamma_cross_init
        self.dropout = dropout
        self.router_cross_start_epoch = router_cross_start_epoch
        self.router_cross_ramp_epochs = router_cross_ramp_epochs
        self.router_cross_mode = router_cross_mode
        self.router_oracle_cross_drugs = router_oracle_cross_drugs
        self.router_residual_global_scale = router_residual_global_scale
        self.router_residual_cross_scale = router_residual_cross_scale
        self.router_branch_effect_threshold = router_branch_effect_threshold
        self.router_cross_detach_mode = router_cross_detach_mode
        self.debug_fixed_gamma_global = debug_fixed_gamma_global
        self.debug_fixed_gamma_cross = debug_fixed_gamma_cross
        self.eps = eps
        self._current_epoch = 0

        # Global branch: original global softmax FPI
        # gamma_max=1.0, gamma_init=1.0 to disable internal gating;
        # Router's gamma_global takes full control.
        self.global_fpi = FragmentProgramInteraction(
            program_dim=program_dim,
            drug_dim=drug_dim,
            fragment_dim=fragment_dim,
            num_programs=num_programs,
            num_drugs=num_drugs,
            max_fragments=max_fragments,
            q_prior_weight=q_prior_weight,
            gamma_max=1.0,
            gamma_init=1.0,
            dropout=dropout,
            use_fpi_sim=False,
        )

        # Cross branch: FPI-SIM cross-only (union_alpha=0, union_beta=0)
        # Only create if cross_max > 0; otherwise use zero residual to avoid parameter bloat.
        if gamma_cross_max > 0:
            cross_sim_cfg = {
                "q_prior_weight": q_prior_weight,
                "program_temperature": 1.0,
                "fragment_temperature": 1.0,
                "cross_temperature": 1.0,
                "union_alpha": 0.0,
                "union_beta": 0.0,
                "use_modal_interaction": False,
            }
            self.cross_fpi = FragmentProgramInteraction(
                program_dim=program_dim,
                drug_dim=drug_dim,
                fragment_dim=fragment_dim,
                num_programs=num_programs,
                num_drugs=num_drugs,
                max_fragments=max_fragments,
                q_prior_weight=q_prior_weight,
                gamma_max=1.0,
                gamma_init=1.0,
                dropout=dropout,
                use_fpi_sim=True,
                fpi_sim_config=cross_sim_cfg,
            )
        else:
            self.cross_fpi = None

        # Drug-wise gamma gates (owned by router, not by branch FPIs)
        self.raw_gamma_global = nn.Parameter(torch.zeros(num_drugs))
        self.raw_gamma_cross = nn.Parameter(torch.zeros(num_drugs))
        self._init_gamma(self.raw_gamma_global, gamma_global_init, gamma_global_max)
        self._init_gamma(self.raw_gamma_cross, gamma_cross_init, gamma_cross_max)

    def _init_gamma(self, param, gamma_init, gamma_max):
        if gamma_init <= 0:
            init_val = -10.0
        else:
            ratio = gamma_init / gamma_max
            ratio = max(min(ratio, 0.999), 0.001)
            init_val = math.log(ratio / (1.0 - ratio))
        with torch.no_grad():
            param.fill_(init_val)

    def get_gamma_global(self):
        if self.debug_fixed_gamma_global is not None:
            return torch.full_like(self.raw_gamma_global, self.debug_fixed_gamma_global)
        return torch.sigmoid(self.raw_gamma_global) * self.gamma_global_max

    def get_gamma_cross(self):
        if self.debug_fixed_gamma_cross is not None:
            return torch.full_like(self.raw_gamma_cross, self.debug_fixed_gamma_cross)
        if self.debug_fixed_gamma_global is not None and self.gamma_cross_max == 0:
            return torch.zeros_like(self.raw_gamma_cross)
        return torch.sigmoid(self.raw_gamma_cross) * self.gamma_cross_max

    def set_current_epoch(self, epoch):
        self._current_epoch = epoch

    def _compute_schedule_factor(self):
        epoch = self._current_epoch
        start = self.router_cross_start_epoch
        ramp = self.router_cross_ramp_epochs

        if epoch < start:
            return 0.0
        if ramp > 0:
            return min(1.0, (epoch - start + 1) / ramp)
        return 1.0

    def _compute_cross_active_mask(self, drug_names):
        if self.router_cross_mode == "oracle_mask":
            mask = torch.zeros(self.num_drugs, device=self.raw_gamma_cross.device)
            if drug_names is not None and self.router_oracle_cross_drugs:
                oracle_set = set(d.strip() for d in self.router_oracle_cross_drugs.split(","))
                for d_idx, d_name in enumerate(drug_names):
                    if d_name in oracle_set and d_idx < self.num_drugs:
                        mask[d_idx] = 1.0
            return mask
        else:
            # learnable mode: all drugs can use cross branch
            return torch.ones(self.num_drugs, device=self.raw_gamma_cross.device)

    def forward(self, z, q, h_pool, drug_emb, frag_emb, frag_mask,
                base_logits, current_epoch=None, drug_names=None):
        B, D = base_logits.shape

        if current_epoch is not None:
            self.set_current_epoch(current_epoch)

        # Compute both branches
        global_out = self.global_fpi(z, q, h_pool, drug_emb, frag_emb, frag_mask)
        residual_global = global_out["residual_logits"]  # [B, D]

        if self.cross_fpi is not None:
            # Cross detach mode: only detach inputs to cross branch, not global branch
            z_cross = z
            q_cross = q
            h_pool_cross = h_pool
            drug_emb_cross = drug_emb
            frag_emb_cross = frag_emb

            mode = self.router_cross_detach_mode
            if mode == "z":
                z_cross = z.detach()
            elif mode == "zq":
                z_cross = z.detach()
                q_cross = q.detach()
            elif mode == "zq_drug":
                z_cross = z.detach()
                q_cross = q.detach()
                drug_emb_cross = drug_emb.detach()
            elif mode == "all":
                z_cross = z.detach()
                q_cross = q.detach()
                h_pool_cross = h_pool.detach()
                drug_emb_cross = drug_emb.detach()
                frag_emb_cross = frag_emb.detach()
            elif mode != "none":
                raise ValueError(f"Unknown router_cross_detach_mode: {mode}")

            cross_out = self.cross_fpi(z_cross, q_cross, h_pool_cross, drug_emb_cross, frag_emb_cross, frag_mask)
            residual_cross = cross_out["residual_logits"]    # [B, D]
        else:
            cross_out = None
            residual_cross = torch.zeros_like(residual_global)

        # Gamma gates
        gamma_global = self.get_gamma_global()  # [D]
        gamma_cross = self.get_gamma_cross()    # [D]

        # Schedule factor for cross branch delayed activation
        schedule_factor = self._compute_schedule_factor()
        cross_active_mask = self._compute_cross_active_mask(drug_names)
        effective_gamma_cross = schedule_factor * gamma_cross * cross_active_mask  # [D]

        # Combine residuals
        final_logits = (
            base_logits
            + gamma_global.unsqueeze(0) * self.router_residual_global_scale * residual_global
            + effective_gamma_cross.unsqueeze(0) * self.router_residual_cross_scale * residual_cross
        )

        # Branch effect diagnostics
        with torch.no_grad():
            branch_global_effect = (gamma_global.abs().unsqueeze(0) * residual_global.abs()).mean(dim=0)  # [D]
            branch_cross_effect = (effective_gamma_cross.abs().unsqueeze(0) * residual_cross.abs()).mean(dim=0)  # [D]
            base_abs_mean = base_logits.abs().mean(dim=0)  # [D]

            branch_global_ratio = branch_global_effect / (base_abs_mean + self.eps)
            branch_cross_ratio = branch_cross_effect / (base_abs_mean + self.eps)
            residual_global_to_base_ratio = residual_global.abs().mean(dim=0) / (base_abs_mean + self.eps)
            residual_cross_to_base_ratio = residual_cross.abs().mean(dim=0) / (base_abs_mean + self.eps)

            # Determine branch preference per drug
            threshold = self.router_branch_effect_threshold
            branch_preference = []
            for d in range(D):
                ge = branch_global_effect[d].item()
                ce = branch_cross_effect[d].item()
                if ge < threshold and ce < threshold:
                    branch_preference.append("base_only")
                elif ge >= ce:
                    branch_preference.append("global_fpi")
                else:
                    branch_preference.append("cross_fpi")

        return {
            "final_logits": final_logits,
            "residual_global": residual_global,
            "residual_cross": residual_cross,
            "gamma_global": gamma_global,
            "gamma_cross": gamma_cross,
            "effective_gamma_cross": effective_gamma_cross,
            "global_out": global_out,
            "cross_out": cross_out,
            "cross_active_mask": cross_active_mask,
            "branch_global_effect": branch_global_effect,
            "branch_cross_effect": branch_cross_effect,
            "branch_global_ratio": branch_global_ratio,
            "branch_cross_ratio": branch_cross_ratio,
            "branch_preference": branch_preference,
            "residual_global_to_base_ratio": residual_global_to_base_ratio,
            "residual_cross_to_base_ratio": residual_cross_to_base_ratio,
            "schedule_factor": torch.tensor(schedule_factor, device=base_logits.device),
        }
