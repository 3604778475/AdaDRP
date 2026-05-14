"""
Optimal Transport utilities for MMPA-DRP.
Inspired by RAM model/ot_solver.py (KCOT/LPD/TKT) and GCL-OT filter prompt.

Key design: OT is used as a TRAINING-ONLY alignment loss.
It does NOT change inference (forward pass returns same outputs).

Phase 4: Drug-Adaptive OT Soft Alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_log_domain(a, b, M, reg=0.1, max_iter=100, thresh=1e-3):
    """
    Balanced entropic Sinkhorn in log-domain for numerical stability.

    Args:
        a: [B, N] or [N] source marginal (must sum to 1)
        b: [B, M] or [M] target marginal (must sum to 1)
        M: [B, N, M] or [N, M] cost matrix
        reg: entropy regularization
        max_iter: maximum iterations
        thresh: convergence threshold

    Returns:
        T: transport plan, same shape prefix as M
    """
    if M.dim() == 2:
        M = M.unsqueeze(0)
        a = a.unsqueeze(0) if a.dim() == 1 else a
        b = b.unsqueeze(0) if b.dim() == 1 else b
        squeeze_out = True
    else:
        squeeze_out = False

    B, N, Mdim = M.shape

    log_a = torch.log(a + 1e-8)
    log_b = torch.log(b + 1e-8)
    log_K = -M / reg

    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    for _ in range(max_iter):
        u_prev = u
        u = log_a - torch.logsumexp(log_K + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(log_K + u.unsqueeze(2), dim=1)
        if (u - u_prev).abs().max() < thresh:
            break

    log_T = log_K + u.unsqueeze(2) + v.unsqueeze(1)
    T = torch.exp(log_T)

    if squeeze_out:
        T = T.squeeze(0)
    return T


def compute_ot_cost(source_feat, target_feat, cost_type="cosine"):
    """
    Compute cost matrix between source and target features.

    Args:
        source_feat: [B_s, d] or [B, N, d]
        target_feat: [B_t, d] or [B, M, d]
        cost_type: "cosine" | "euclidean" | "l2"

    Returns:
        cost: [B_s, B_t] or [B, N, M]
        sim: [B_s, B_t] or [B, N, M]
    """
    if source_feat.dim() == 2 and target_feat.dim() == 2:
        source_feat = F.normalize(source_feat, p=2, dim=-1)
        target_feat = F.normalize(target_feat, p=2, dim=-1)

        if cost_type == "cosine":
            sim = torch.matmul(source_feat, target_feat.t())
            cost = 1.0 - sim
        elif cost_type in ("euclidean", "l2"):
            diff = source_feat.unsqueeze(1) - target_feat.unsqueeze(0)
            cost = torch.norm(diff, p=2, dim=-1)
            sim = 1.0 / (1.0 + cost)
        else:
            raise ValueError(f"Unknown cost_type: {cost_type}")

    elif source_feat.dim() == 3 and target_feat.dim() == 3:
        source_feat = F.normalize(source_feat, p=2, dim=-1)
        target_feat = F.normalize(target_feat, p=2, dim=-1)

        if cost_type == "cosine":
            sim = torch.matmul(source_feat, target_feat.transpose(-2, -1))
            cost = 1.0 - sim
        elif cost_type in ("euclidean", "l2"):
            diff = source_feat.unsqueeze(2) - target_feat.unsqueeze(1)
            cost = torch.norm(diff, p=2, dim=-1)
            sim = 1.0 / (1.0 + cost)
        else:
            raise ValueError(f"Unknown cost_type: {cost_type}")
    else:
        raise ValueError(
            f"Unsupported feature dims: source {source_feat.shape}, target {target_feat.shape}"
        )

    return cost, sim


def build_unbalanced_marginal(similarity, temperature=0.1, dim=-1):
    with torch.no_grad():
        sim_max = similarity.max(dim=dim)[0]
        marginal = (sim_max / temperature).softmax(dim=-1)
    return marginal


def apply_teacher_cost(cost, teacher_plan, lambda_teacher=0.05):
    cost_tilde = cost - lambda_teacher * torch.log(teacher_plan + 1e-8)
    return cost_tilde


def ot_alignment_loss(
    source_feat,
    target_feat,
    reg=0.1,
    max_iter=50,
    cost_type="cosine",
    marginal_type="uniform",
    marginal_temperature=0.1,
    return_plan=False,
    source_marginal=None,
    target_marginal=None,
):
    cost, sim = compute_ot_cost(source_feat, target_feat, cost_type=cost_type)

    if cost.dim() == 2:
        N, M = cost.shape
        device = cost.device

        if source_marginal is not None and target_marginal is not None:
            u = source_marginal
            v = target_marginal
        elif marginal_type == "uniform":
            u = torch.ones(N, device=device) / N
            v = torch.ones(M, device=device) / M
        elif marginal_type in ("unbalanced", "max_sim"):
            u = build_unbalanced_marginal(sim.unsqueeze(0), temperature=marginal_temperature, dim=2).squeeze(0)
            v = build_unbalanced_marginal(sim.unsqueeze(0), temperature=marginal_temperature, dim=1).squeeze(0)
        else:
            raise ValueError(f"Unknown marginal_type: {marginal_type}")

        with torch.no_grad():
            T = sinkhorn_log_domain(u, v, cost, reg=reg, max_iter=max_iter)
        ot_dist = (T.detach() * cost).sum()

    elif cost.dim() == 3:
        B, N, M = cost.shape
        device = cost.device

        if source_marginal is not None and target_marginal is not None:
            u = source_marginal
            v = target_marginal
        elif marginal_type == "uniform":
            u = torch.ones(B, N, device=device) / N
            v = torch.ones(B, M, device=device) / M
        elif marginal_type in ("unbalanced", "max_sim"):
            u = build_unbalanced_marginal(sim, temperature=marginal_temperature, dim=2)
            v = build_unbalanced_marginal(sim, temperature=marginal_temperature, dim=1)
        else:
            raise ValueError(f"Unknown marginal_type: {marginal_type}")

        with torch.no_grad():
            T = sinkhorn_log_domain(u, v, cost, reg=reg, max_iter=max_iter)
        ot_dist = (T.detach() * cost).sum() / B

    else:
        raise ValueError(f"Unsupported cost dim: {cost.dim()}")

    if return_plan:
        return ot_dist, T
    return ot_dist


class DrugAdaptiveOTLoss(nn.Module):
    """
    Drug-Adaptive Optimal Transport alignment loss.

    Supports 4 modes:
    - "global": all drugs share lambda_global
    - "per_drug_fixed": read per-drug lambda from config dict
    - "per_drug_validation": select lambda per drug via inner validation (TODO)
    - "per_drug_discrepancy": unsupervised gap-based lambda (TODO)

    When use_drug_conditioned_feature=False:
        All drugs share the same source_feat/target_feat.
        L_OT_total = mean_d(lambda_d) * L_OT   (if all drugs share same feat)
        But we still keep per-drug weighting for consistency.

    When use_drug_conditioned_feature=True:
        source_feat: [B_s, D, d] or [B_s*D, d] (will be reshaped)
        target_feat: [B_t, D, d] or [B_t*D, d]
        L_OT_d = OT(source_feat[:, d, :], target_feat[:, d, :])
        L_OT_total = mean_d(lambda_d * L_OT_d)
    """

    def __init__(
        self,
        mode="global",
        lambda_global=0.01,
        reg=0.1,
        max_iter=50,
        cost_type="cosine",
        marginal_type="uniform",
        marginal_temperature=0.1,
        use_drug_conditioned_feature=False,
        per_drug_lambda=None,
        discrepancy_config=None,
    ):
        super().__init__()
        self.mode = mode
        self.lambda_global = lambda_global
        self.reg = reg
        self.max_iter = max_iter
        self.cost_type = cost_type
        self.marginal_type = marginal_type
        self.marginal_temperature = marginal_temperature
        self.use_drug_conditioned_feature = use_drug_conditioned_feature
        self.per_drug_lambda = per_drug_lambda or {}
        self.discrepancy_config = discrepancy_config or {}

    def _compute_single_ot(self, source_feat, target_feat):
        """Compute OT distance for a single feature pair."""
        cost, sim = compute_ot_cost(source_feat, target_feat, cost_type=self.cost_type)

        if cost.dim() != 2:
            raise ValueError(f"Expected 2D cost in _compute_single_ot, got {cost.dim()}D")

        N, M = cost.shape
        device = cost.device

        if self.marginal_type == "uniform":
            u = torch.ones(N, device=device) / N
            v = torch.ones(M, device=device) / M
        else:
            u = build_unbalanced_marginal(sim.unsqueeze(0), temperature=self.marginal_temperature, dim=2).squeeze(0)
            v = build_unbalanced_marginal(sim.unsqueeze(0), temperature=self.marginal_temperature, dim=1).squeeze(0)

        with torch.no_grad():
            T = sinkhorn_log_domain(u, v, cost, reg=self.reg, max_iter=self.max_iter)

        # Check for NaN/Inf
        if torch.isnan(T).any() or torch.isinf(T).any():
            import warnings
            warnings.warn("NaN/Inf in transport plan, returning zero loss")
            return torch.tensor(0.0, device=device), {
                "transport_cost": 0.0,
                "transport_entropy": 0.0,
                "cost_mean": cost.mean().item(),
                "cost_std": cost.std().item(),
                "P_max": 0.0,
                "P_min": 0.0,
            }

        ot_dist = (T.detach() * cost).sum()

        # Diagnostics
        entropy = -(T * torch.log(T + 1e-8)).sum()
        diagnostics = {
            "transport_cost": ot_dist.item(),
            "transport_entropy": entropy.item(),
            "cost_mean": cost.mean().item(),
            "cost_std": cost.std().item(),
            "P_max": T.max().item(),
            "P_min": T.min().item(),
        }

        return ot_dist, diagnostics

    def _get_lambda_per_drug(self, drug_names, device):
        """Get lambda tensor [D] for each drug."""
        D = len(drug_names)

        if self.mode == "global":
            lambdas = torch.full((D,), self.lambda_global, device=device)

        elif self.mode == "per_drug_fixed":
            lambdas = torch.tensor(
                [self.per_drug_lambda.get(d, self.lambda_global) for d in drug_names],
                dtype=torch.float32,
                device=device,
            )

        elif self.mode == "per_drug_discrepancy":
            # Placeholder: will be filled in forward when gap is available
            lambdas = torch.full((D,), self.lambda_global, device=device)

        else:
            lambdas = torch.full((D,), self.lambda_global, device=device)

        return lambdas

    def forward(
        self,
        source_feat,
        target_feat,
        drug_names=None,
        source_feat_per_drug=None,
        target_feat_per_drug=None,
    ):
        """
        Args:
            source_feat: [B_s, d] — shared source feature (when use_drug_conditioned_feature=False)
            target_feat: [B_t, d] — shared target feature
            drug_names: list of drug names, length D
            source_feat_per_drug: [B_s, D, d] or [B_s*D, d] (optional, for conditioned mode)
            target_feat_per_drug: [B_t, D, d] or [B_t*D, d] (optional, for conditioned mode)

        Returns:
            loss: scalar — weighted OT loss
            loss_per_drug: dict {drug_name: weighted_loss_value}
            lambda_per_drug: dict {drug_name: lambda_value}
            diagnostics: dict — aggregate diagnostics
        """
        device = source_feat.device
        D = len(drug_names) if drug_names is not None else 1
        lambdas = self._get_lambda_per_drug(drug_names or ["global"], device)

        loss_per_drug = {}
        diag_per_drug = {}

        if self.use_drug_conditioned_feature and source_feat_per_drug is not None:
            # Drug-conditioned features: compute OT per drug
            # Reshape if needed
            if source_feat_per_drug.dim() == 2:
                # [B_s*D, d] -> [B_s, D, d]
                B_s = source_feat_per_drug.size(0) // D
                source_feat_pd = source_feat_per_drug.view(B_s, D, -1)
            else:
                source_feat_pd = source_feat_per_drug  # [B_s, D, d]

            if target_feat_per_drug.dim() == 2:
                B_t = target_feat_per_drug.size(0) // D
                target_feat_pd = target_feat_per_drug.view(B_t, D, -1)
            else:
                target_feat_pd = target_feat_per_drug  # [B_t, D, d]

            for d_idx, drug_name in enumerate(drug_names or ["global"]):
                ot_dist, diag = self._compute_single_ot(
                    source_feat_pd[:, d_idx, :],
                    target_feat_pd[:, d_idx, :],
                )
                loss_per_drug[drug_name] = lambdas[d_idx].item() * ot_dist.item()
                diag_per_drug[drug_name] = diag

        else:
            # Shared features: compute OT once, weight per drug
            ot_dist, diag = self._compute_single_ot(source_feat, target_feat)
            for d_idx, drug_name in enumerate(drug_names or ["global"]):
                loss_per_drug[drug_name] = lambdas[d_idx].item() * ot_dist.item()
                diag_per_drug[drug_name] = diag

        # Total loss: mean over drugs
        if D > 0:
            loss = sum(loss_per_drug.values()) / D
        else:
            loss = torch.tensor(0.0, device=device)

        # Convert to tensor for backward if needed
        # Note: when use_drug_conditioned_feature=False, all drugs share same ot_dist,
        # so loss = ot_dist * mean(lambdas)
        if not self.use_drug_conditioned_feature:
            loss = ot_dist * lambdas.mean()

        lambda_per_drug = {
            name: lambdas[i].item() for i, name in enumerate(drug_names or ["global"])
        }

        # Aggregate diagnostics
        diagnostics = {
            "mode": self.mode,
            "use_drug_conditioned_feature": self.use_drug_conditioned_feature,
            "lambda_mean": lambdas.mean().item(),
            "lambda_std": lambdas.std().item(),
            "lambda_min": lambdas.min().item(),
            "lambda_max": lambdas.max().item(),
        }
        if diag_per_drug:
            for key in ["transport_cost", "transport_entropy", "cost_mean", "cost_std", "P_max", "P_min"]:
                vals = [d[key] for d in diag_per_drug.values() if key in d]
                if vals:
                    diagnostics[f"{key}_mean"] = sum(vals) / len(vals)

        return loss, loss_per_drug, lambda_per_drug, diagnostics



class DrugConditionedOTLoss(nn.Module):
    """
    Drug-Conditioned Optimal Transport alignment loss.

    For each drug d, compute OT on drug-conditioned fused features:
        L_OT_d = OT(F_s^d, F_t^d)

    Supports three modes:
    - "global": all drugs share lambda_global
    - "fixed": per-drug lambda from config dict
    - "discrepancy": unsupervised gap-based adaptive lambda

    Inputs:
        source_fused: [B_s, D, H]
        target_fused: [B_t, D, H]
        drug_names: list[str] length D

    Outputs:
        loss_ot_total: scalar tensor
        loss_per_drug: Tensor[D]
        lambda_per_drug: Tensor[D]
        gap_per_drug: Tensor[D]
        diagnostics: dict
    """

    def __init__(
        self,
        mode="global",
        global_lambda=0.005,
        reg=0.1,
        max_iter=50,
        cost="cosine",
        marginal="uniform",
        fixed_lambda_dict=None,
        discrepancy_config=None,
    ):
        super().__init__()
        self.mode = mode
        self.global_lambda = global_lambda
        self.reg = reg
        self.max_iter = max_iter
        self.cost = cost
        self.marginal = marginal
        self.fixed_lambda_dict = fixed_lambda_dict or {}
        self.discrepancy_config = discrepancy_config or {}

        # EMA state for discrepancy mode
        self._ema_gap = None
        self._ema_initialized = False

    def _compute_gap_cosine(self, source_feat, target_feat):
        """
        Compute cosine-based gap for each drug.
        source_feat: [B_s, D, H]
        target_feat: [B_t, D, H]
        Returns: gap [D] where gap_d = 1 - cosine(mu_s_d, mu_t_d)
        """
        mu_s = source_feat.mean(dim=0)  # [D, H]
        mu_t = target_feat.mean(dim=0)  # [D, H]

        mu_s_norm = F.normalize(mu_s, p=2, dim=-1)
        mu_t_norm = F.normalize(mu_t, p=2, dim=-1)

        cosine_sim = (mu_s_norm * mu_t_norm).sum(dim=-1)  # [D]
        gap = 1.0 - cosine_sim
        return gap

    def _compute_gap_l2(self, source_feat, target_feat):
        """
        Compute L2-based gap for each drug.
        """
        mu_s = source_feat.mean(dim=0)  # [D, H]
        mu_t = target_feat.mean(dim=0)  # [D, H]
        gap = torch.norm(mu_s - mu_t, p=2, dim=-1)  # [D]
        return gap

    def _compute_single_ot(self, x, y):
        """
        Compute OT distance for a single drug's feature pair.
        x: [B_s, H]
        y: [B_t, H]
        Returns: ot_dist, diagnostics dict
        """
        cost, sim = compute_ot_cost(x, y, cost_type=self.cost)

        if cost.dim() != 2:
            raise ValueError(f"Expected 2D cost in _compute_single_ot, got {cost.dim()}D")

        N, M = cost.shape
        device = cost.device

        if self.marginal == "uniform":
            u = torch.ones(N, device=device) / N
            v = torch.ones(M, device=device) / M
        else:
            u = build_unbalanced_marginal(sim.unsqueeze(0), temperature=0.1, dim=2).squeeze(0)
            v = build_unbalanced_marginal(sim.unsqueeze(0), temperature=0.1, dim=1).squeeze(0)

        with torch.no_grad():
            T = sinkhorn_log_domain(u, v, cost, reg=self.reg, max_iter=self.max_iter)

        if torch.isnan(T).any() or torch.isinf(T).any():
            import warnings
            warnings.warn("NaN/Inf in transport plan, returning zero loss")
            return torch.tensor(0.0, device=device), {
                "transport_cost": 0.0,
                "transport_entropy": 0.0,
                "cost_mean": cost.mean().item(),
                "cost_std": cost.std().item(),
                "P_max": 0.0,
                "P_min": 0.0,
            }

        ot_dist = (T.detach() * cost).sum()
        entropy = -(T * torch.log(T + 1e-8)).sum()

        diagnostics = {
            "transport_cost": ot_dist.item(),
            "transport_entropy": entropy.item(),
            "cost_mean": cost.mean().item(),
            "cost_std": cost.std().item(),
            "P_max": T.max().item(),
            "P_min": T.min().item(),
        }
        return ot_dist, diagnostics

    def _get_lambda_per_drug(self, gap, drug_names, device):
        """
        Compute lambda_per_drug [D] based on mode.
        """
        D = len(drug_names)

        if self.mode == "global":
            lambdas = torch.full((D,), self.global_lambda, device=device)
            return lambdas, gap

        elif self.mode == "fixed":
            lambdas = torch.tensor(
                [self.fixed_lambda_dict.get(d, self.global_lambda) for d in drug_names],
                dtype=torch.float32,
                device=device,
            )
            return lambdas, gap

        elif self.mode == "discrepancy":
            cfg = self.discrepancy_config
            lambda_min = cfg.get("lambda_min", 0.0)
            lambda_max = cfg.get("lambda_max", 0.01)
            temperature = cfg.get("temperature", 5.0)
            threshold = cfg.get("threshold", "mean")
            detach_gap = cfg.get("detach_gap", True)
            normalize_gap = cfg.get("normalize_gap", True)
            ema_gap = cfg.get("ema_gap", True)
            ema_momentum = cfg.get("ema_momentum", 0.9)
            gap_type = cfg.get("gap_type", "cosine")

            if gap is None:
                raise ValueError("gap must be provided for discrepancy mode")

            if detach_gap:
                gap = gap.detach()

            # EMA update
            if ema_gap:
                if not self._ema_initialized or self._ema_gap is None:
                    self._ema_gap = gap.clone()
                    self._ema_initialized = True
                else:
                    self._ema_gap = ema_momentum * self._ema_gap + (1 - ema_momentum) * gap
                gap_used = self._ema_gap
            else:
                gap_used = gap

            eps = 1e-8
            if normalize_gap:
                gap_norm = (gap_used - gap_used.mean()) / (gap_used.std() + eps)
            else:
                gap_norm = gap_used

            if threshold == "mean":
                score = temperature * gap_norm
            else:
                threshold_val = float(threshold)
                score = temperature * (gap_used - threshold_val)

            lambdas = lambda_min + (lambda_max - lambda_min) * torch.sigmoid(score)

            if detach_gap:
                lambdas = lambdas.detach()

            return lambdas, gap_used

        else:
            raise ValueError(f"Unknown DrugConditionedOTLoss mode: {self.mode}")

    def forward(self, source_fused, target_fused, drug_names=None):
        """
        Args:
            source_fused: [B_s, D, H]
            target_fused: [B_t, D, H]
            drug_names: list[str] length D

        Returns:
            loss_ot_total: scalar tensor
            info: dict containing:
                - loss_per_drug: Tensor[D]
                - lambda_per_drug: Tensor[D]
                - gap_per_drug: Tensor[D]
                - diagnostics: dict
        """
        device = source_fused.device
        B_s, D, H = source_fused.shape
        B_t, D2, H2 = target_fused.shape
        assert D == D2, f"Drug count mismatch: {D} vs {D2}"
        assert H == H2, f"Feature dim mismatch: {H} vs {H2}"

        if drug_names is None:
            drug_names = [f"drug_{i}" for i in range(D)]

        # Compute gap for discrepancy mode (or return None)
        if self.mode == "discrepancy":
            gap_type = self.discrepancy_config.get("gap_type", "cosine")
            if gap_type == "l2":
                gap = self._compute_gap_l2(source_fused, target_fused)
            else:
                gap = self._compute_gap_cosine(source_fused, target_fused)
        else:
            gap = None

        lambdas, gap_out = self._get_lambda_per_drug(gap, drug_names, device)

        loss_per_drug = torch.zeros(D, device=device)
        diag_list = []

        for d_idx in range(D):
            x = source_fused[:, d_idx, :]  # [B_s, H]
            y = target_fused[:, d_idx, :]  # [B_t, H]
            ot_dist, diag = self._compute_single_ot(x, y)
            loss_per_drug[d_idx] = lambdas[d_idx] * ot_dist
            diag_list.append(diag)

        loss_ot_total = loss_per_drug.mean()

        # Aggregate diagnostics
        diagnostics = {
            "mode": self.mode,
            "global_lambda": self.global_lambda,
            "lambda_mean": lambdas.mean().item(),
            "lambda_std": lambdas.std().item(),
            "lambda_min": lambdas.min().item(),
            "lambda_max": lambdas.max().item(),
        }
        if gap_out is not None:
            diagnostics["gap_mean"] = gap_out.mean().item()
            diagnostics["gap_std"] = gap_out.std().item()
            diagnostics["gap_min"] = gap_out.min().item()
            diagnostics["gap_max"] = gap_out.max().item()

        for key in ["transport_cost", "transport_entropy", "cost_mean", "cost_std", "P_max", "P_min"]:
            vals = [d[key] for d in diag_list if key in d]
            if vals:
                diagnostics[f"{key}_mean"] = sum(vals) / len(vals)

        # Compute per-drug fused feature norms
        source_fused_norm = torch.norm(source_fused, p=2, dim=-1).mean(dim=0)  # [D]
        target_fused_norm = torch.norm(target_fused, p=2, dim=-1).mean(dim=0)  # [D]

        info = {
            "loss_per_drug": loss_per_drug.detach(),
            "lambda_per_drug": lambdas.detach(),
            "gap_per_drug": gap_out.detach() if gap_out is not None else torch.zeros(D, device=device),
            "diagnostics": diagnostics,
            "diag_per_drug": diag_list,
            "source_fused_norm": source_fused_norm.detach(),
            "target_fused_norm": target_fused_norm.detach(),
        }
        return loss_ot_total, info


def q_align_loss(source_q, target_q, eps=1e-6):
    """
    Symmetric KL divergence between mean program activation distributions.

    Args:
        source_q: [B_s, K]
        target_q: [B_t, K]

    Returns:
        loss: scalar
        diagnostics: dict
    """
    mean_q_s = source_q.mean(dim=0)  # [K]
    mean_q_t = target_q.mean(dim=0)  # [K]

    # Normalize to probability distributions
    p_s = mean_q_s / (mean_q_s.sum() + eps)
    p_t = mean_q_t / (mean_q_t.sum() + eps)

    kl_st = (p_s * (torch.log(p_s + eps) - torch.log(p_t + eps))).sum()
    kl_ts = (p_t * (torch.log(p_t + eps) - torch.log(p_s + eps))).sum()
    loss = kl_st + kl_ts

    diagnostics = {
        "mean_q_source": mean_q_s.detach().cpu().numpy(),
        "mean_q_target": mean_q_t.detach().cpu().numpy(),
        "q_gap_l1": (mean_q_s - mean_q_t).abs().sum().item(),
        "q_gap_l2": (mean_q_s - mean_q_t).norm(2).item(),
        "q_entropy_source": -(p_s * torch.log(p_s + eps)).sum().item(),
        "q_entropy_target": -(p_t * torch.log(p_t + eps)).sum().item(),
    }
    return loss, diagnostics


class ProgramOTLoss(nn.Module):
    """
    Program-level Optimal Transport alignment loss.

    Modes:
    - "same_program_uniform": L = 1/K * sum_k OT(z_s^k, z_t^k), uniform marginals
    - "same_program_q_weighted": same but marginals weighted by q
    - "program_only": only program OT, no sample OT
    - "flattened_program": z reshape to [B*K, d], single OT
    - "q_mass_flattened": z reshape to [B*K, d], OT with q as transport mass

    Training-only: does NOT change inference.
    """

    def __init__(
        self,
        mode="same_program_uniform",
        lambda_program_ot=0.01,
        reg=0.1,
        max_iter=50,
        cost_type="cosine",
        marginal_type="uniform",
        marginal_temperature=0.1,
        detach_q_marginal=True,
        q_temperature=1.0,
        q_power=1.0,
        min_mass=1e-6,
    ):
        super().__init__()
        valid_modes = ("same_program_uniform", "same_program_q_weighted",
                       "program_only", "flattened_program", "q_mass_flattened")
        if mode not in valid_modes:
            raise ValueError(f"Unknown ProgramOT mode: {mode}")
        self.mode = mode
        self.lambda_program_ot = lambda_program_ot
        self.reg = reg
        self.max_iter = max_iter
        self.cost_type = cost_type
        self.marginal_type = marginal_type
        self.marginal_temperature = marginal_temperature
        self.detach_q_marginal = detach_q_marginal
        self.q_temperature = q_temperature
        self.q_power = q_power
        self.min_mass = min_mass

    def _build_q_mass(self, source_q, target_q):
        """Build q-mass marginals for q_mass_flattened mode."""
        source_mass = source_q.reshape(-1)
        target_mass = target_q.reshape(-1)

        if self.q_temperature != 1.0:
            source_mass = F.softmax(torch.log(source_mass + 1e-8) / self.q_temperature, dim=0)
            target_mass = F.softmax(torch.log(target_mass + 1e-8) / self.q_temperature, dim=0)

        if self.q_power != 1.0:
            source_mass = source_mass ** self.q_power
            target_mass = target_mass ** self.q_power

        source_mass = torch.clamp(source_mass, min=self.min_mass)
        target_mass = torch.clamp(target_mass, min=self.min_mass)

        source_mass = source_mass / source_mass.sum()
        target_mass = target_mass / target_mass.sum()

        if self.detach_q_marginal:
            source_mass = source_mass.detach()
            target_mass = target_mass.detach()

        return source_mass, target_mass

    def forward(self, source_z, source_q, target_z, target_q):
        device = source_z.device

        if self.mode == "q_mass_flattened":
            B_s, K, d = source_z.shape
            B_t = target_z.size(0)
            source_feat = source_z.reshape(B_s * K, d)
            target_feat = target_z.reshape(B_t * K, d)

            source_mass, target_mass = self._build_q_mass(source_q, target_q)

            cost, sim = compute_ot_cost(source_feat, target_feat, cost_type=self.cost_type)
            N, M = cost.shape

            with torch.no_grad():
                T = sinkhorn_log_domain(source_mass, target_mass, cost, reg=self.reg, max_iter=self.max_iter)
            ot_dist = (T.detach() * cost).sum()

            # Marginal diagnostics
            marginal_a_entropy = -(source_mass * torch.log(source_mass + 1e-8)).sum().item()
            marginal_b_entropy = -(target_mass * torch.log(target_mass + 1e-8)).sum().item()

            diagnostics = {
                "mode": self.mode,
                "ot_dist": ot_dist.item(),
                "lambda": self.lambda_program_ot,
                "transport_cost": ot_dist.item(),
                "transport_entropy": -(T * torch.log(T + 1e-8)).sum().item(),
                "cost_mean": cost.mean().item(),
                "cost_std": cost.std().item(),
                "P_max": T.max().item(),
                "P_min": T.min().item(),
                "marginal_a_entropy": marginal_a_entropy,
                "marginal_b_entropy": marginal_b_entropy,
                "marginal_a_min": source_mass.min().item(),
                "marginal_a_max": source_mass.max().item(),
                "marginal_b_min": target_mass.min().item(),
                "marginal_b_max": target_mass.max().item(),
            }
            loss = self.lambda_program_ot * ot_dist
            return loss, diagnostics

        if self.mode == "flattened_program":
            B_s, K, d = source_z.shape
            B_t = target_z.size(0)
            source_feat = source_z.reshape(B_s * K, d)
            target_feat = target_z.reshape(B_t * K, d)

            ot_dist = ot_alignment_loss(
                source_feat, target_feat,
                reg=self.reg, max_iter=self.max_iter,
                cost_type=self.cost_type, marginal_type=self.marginal_type,
                marginal_temperature=self.marginal_temperature,
            )
            loss = self.lambda_program_ot * ot_dist
            diagnostics = {"mode": self.mode, "ot_dist": ot_dist.item(), "lambda": self.lambda_program_ot}
            return loss, diagnostics

        # For same_program_* modes: transpose to [K, B, d]
        source_z_t = source_z.transpose(0, 1)
        target_z_t = target_z.transpose(0, 1)
        K = source_z_t.size(0)

        if self.mode == "same_program_q_weighted":
            source_q_t = source_q.transpose(0, 1)
            target_q_t = target_q.transpose(0, 1)
            source_marginal = source_q_t / (source_q_t.sum(dim=1, keepdim=True) + 1e-8)
            target_marginal = target_q_t / (target_q_t.sum(dim=1, keepdim=True) + 1e-8)
        else:
            source_marginal = None
            target_marginal = None

        ot_dist = ot_alignment_loss(
            source_z_t, target_z_t,
            reg=self.reg, max_iter=self.max_iter,
            cost_type=self.cost_type, marginal_type=self.marginal_type,
            marginal_temperature=self.marginal_temperature,
            source_marginal=source_marginal,
            target_marginal=target_marginal,
        )

        loss = self.lambda_program_ot * ot_dist
        diagnostics = {"mode": self.mode, "ot_dist": ot_dist.item(), "lambda": self.lambda_program_ot, "num_programs": K}
        return loss, diagnostics
