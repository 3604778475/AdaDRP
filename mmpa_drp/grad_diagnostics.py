"""
Gradient diagnostics for Program OT (Phase 6-R).
"""
import torch


def compute_grad_norm(loss, params, retain_graph=True):
    """
    Compute gradient norm of loss w.r.t. params using torch.autograd.grad.
    Does NOT modify optimizer's current gradients.

    Args:
        loss: scalar tensor
        params: list[nn.Parameter]
        retain_graph: bool

    Returns:
        grad_norm_total: float
        num_params_with_grad: int
        num_params_without_grad: int
    """
    grads = torch.autograd.grad(
        loss, params,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )
    total_norm = 0.0
    num_with = 0
    num_without = 0
    for g in grads:
        if g is not None:
            total_norm += g.detach().norm(2).item() ** 2
            num_with += 1
        else:
            num_without += 1
    return total_norm ** 0.5, num_with, num_without


def diagnose_program_ot_gradients(
    s_loss,
    cdan_domain_loss,
    ot_loss,
    program_ot_loss,
    encoder_params,
    device,
):
    """
    Diagnose gradient norms for each loss component.
    Call this BEFORE loss.backward().

    Returns:
        dict with grad norms and ratios.
    """
    diag = {}

    # Helper to safely compute grad norm
    def _norm(loss_tensor):
        if loss_tensor is None or loss_tensor.item() == 0.0:
            return 0.0, 0, len(encoder_params)
        try:
            return compute_grad_norm(loss_tensor, encoder_params, retain_graph=True)
        except Exception:
            return 0.0, 0, len(encoder_params)

    diag["grad_norm_response"], _, _ = _norm(s_loss)
    diag["grad_norm_cdan"], _, _ = _norm(cdan_domain_loss)
    diag["grad_norm_sample_ot"], _, _ = _norm(ot_loss)
    diag["grad_norm_program_ot_raw"], _, _ = _norm(program_ot_loss)

    # Weighted program OT
    if program_ot_loss is not None and program_ot_loss.item() != 0.0:
        diag["grad_norm_program_ot_weighted"] = diag["grad_norm_program_ot_raw"]
    else:
        diag["grad_norm_program_ot_weighted"] = 0.0

    # Total loss gradient
    total_loss = s_loss + cdan_domain_loss + ot_loss + program_ot_loss
    diag["grad_norm_total"], _, _ = _norm(total_loss)

    # Ratios
    total = diag["grad_norm_total"] + 1e-12
    diag["ratio_program_raw_to_total"] = diag["grad_norm_program_ot_raw"] / total
    diag["ratio_program_weighted_to_total"] = diag["grad_norm_program_ot_weighted"] / total
    diag["ratio_sample_ot_to_total"] = diag["grad_norm_sample_ot"] / total

    return diag
