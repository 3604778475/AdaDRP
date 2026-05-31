#!/usr/bin/env python3
"""
Router global-only equivalence test.

Goal: Verify that Router global-only (cross_max=0, fixed_gamma_global=0.05)
produces identical logits to FPI2_warmup (single global FPI, gamma=0.05).

Usage:
    cd MACB-DRP-main
    python tests/test_fpi_router_equivalence.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
if not hasattr(config, 'tissue_map') or getattr(config.tissue_map, 'shape', None) == ():
    config.tissue_map = {"Tissue_0": 0, "Tissue_1": 1, "Tissue_2": 2}
    config.drug_feat = torch.randn(9, 256)

from mmpa_drp.fpi import FragmentProgramInteraction, DualFPIResidualRouter


def test_forward_global_shape():
    """Verify forward_global score reshape semantics."""
    B, K, H, D, M = 2, 8, 64, 9, 16
    z = torch.randn(B, K, H)
    q = torch.softmax(torch.randn(B, K), dim=-1)
    h_pool = torch.randn(B, H)
    drug_emb = torch.randn(D, H)
    frag_emb = torch.randn(D, M, H)
    frag_mask = torch.ones(D, M)

    fpi = FragmentProgramInteraction(
        program_dim=H, drug_dim=H, fragment_dim=H,
        num_programs=K, num_drugs=D, max_fragments=M,
        gamma_max=0.1, gamma_init=0.05, use_fpi_sim=False,
    )

    with torch.no_grad():
        out = fpi(z, q, h_pool, drug_emb, frag_emb, frag_mask)

    residual = out["residual_logits"]  # [B, D]
    attn = out["attn"]  # [B, D, K, M]

    print("=== forward_global shape check ===")
    print(f"  residual shape: {residual.shape} (expected [{B}, {D}])")
    print(f"  attn shape:     {attn.shape} (expected [{B}, {D}, {K}, {M}])")
    assert residual.shape == (B, D), f"residual shape mismatch: {residual.shape}"
    assert attn.shape == (B, D, K, M), f"attn shape mismatch: {attn.shape}"

    # Verify that score.reshape(B, D, K*M) preserves drug-wise semantics
    # i.e., attn[b, d, :, :] should be a valid distribution over K*M pairs
    for b in range(B):
        for d in range(D):
            assert torch.isclose(attn[b, d].sum(), torch.tensor(1.0), atol=1e-5), \
                f"attn[{b}, {d}] does not sum to 1"
    print("  All attention distributions sum to 1.0 [PASS]")


def test_router_vs_standalone_fpi():
    """
    Compare DualFPIResidualRouter.global_fpi output vs standalone FragmentProgramInteraction.
    Since parameters are randomly initialized, we manually copy weights from standalone to router.
    """
    B, K, H, D, M = 2, 8, 64, 9, 16
    z = torch.randn(B, K, H)
    q = torch.softmax(torch.randn(B, K), dim=-1)
    h_pool = torch.randn(B, H)
    drug_emb = torch.randn(D, H)
    frag_emb = torch.randn(D, M, H)
    frag_mask = torch.ones(D, M)

    # Standalone FPI (FPI2_warmup config)
    standalone = FragmentProgramInteraction(
        program_dim=H, drug_dim=H, fragment_dim=H,
        num_programs=K, num_drugs=D, max_fragments=M,
        gamma_max=0.1, gamma_init=0.05, use_fpi_sim=False,
    )

    # Router with cross disabled
    router = DualFPIResidualRouter(
        program_dim=H, drug_dim=H, fragment_dim=H,
        num_programs=K, num_drugs=D, max_fragments=M,
        gamma_global_max=0.05, gamma_global_init=0.05,
        gamma_cross_max=0.0, gamma_cross_init=0.0,
        debug_fixed_gamma_global=0.05,
    )

    # Copy standalone weights into router.global_fpi
    # (they have the same architecture)
    result = router.global_fpi.load_state_dict(standalone.state_dict(), strict=False)
    if result.missing_keys:
        print(f"  WARNING: missing keys in router.global_fpi: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  WARNING: unexpected keys in router.global_fpi: {result.unexpected_keys}")

    # Verify weight-by-weight match
    mismatch_count = 0
    for name, p_stand in standalone.named_parameters():
        p_router = dict(router.global_fpi.named_parameters()).get(name)
        if p_router is None:
            print(f"  MISSING in router: {name}")
            mismatch_count += 1
            continue
        if not torch.allclose(p_stand, p_router, atol=1e-6):
            diff = (p_stand - p_router).abs().max().item()
            print(f"  MISMATCH {name}: max_diff={diff:.8f}")
            mismatch_count += 1
    if mismatch_count == 0:
        print("  All parameters exactly matched after load_state_dict")

    base_logits = torch.randn(B, D)

    standalone.eval()
    router.eval()
    with torch.no_grad():
        standalone_out = standalone(z, q, h_pool, drug_emb, frag_emb, frag_mask)
        router_out = router(
            z=z, q=q, h_pool=h_pool, drug_emb=drug_emb,
            frag_emb=frag_emb, frag_mask=frag_mask,
            base_logits=base_logits, current_epoch=0, drug_names=None,
        )

    standalone_residual = standalone_out["residual_logits"]
    router_residual = router_out["residual_global"]
    standalone_gamma = standalone_out["gamma"]
    router_gamma = router_out["gamma_global"]

    # Standalone: final = base + gamma * residual (done in ProgramFusionMLP)
    standalone_final = base_logits + standalone_gamma.unsqueeze(0) * standalone_residual

    # Router: final already computed inside router
    router_final = router_out["final_logits"]

    print("\n=== Equivalence Test (weight-copied) ===")
    print(f"  standalone_gamma:      mean={standalone_gamma.mean():.6f}, max={standalone_gamma.max():.6f}")
    print(f"  router_gamma_global:   mean={router_gamma.mean():.6f}, max={router_gamma.max():.6f}")
    print(f"  standalone_residual:   mean_abs={standalone_residual.abs().mean():.6f}")
    print(f"  router_residual_global: mean_abs={router_residual.abs().mean():.6f}")

    res_diff = (standalone_residual - router_residual).abs().max()
    final_diff = (standalone_final - router_final).abs().max()
    print(f"\n  Max residual diff:     {res_diff:.8f}")
    print(f"  Max final_logits diff: {final_diff:.8f}")

    if res_diff < 1e-5 and final_diff < 1e-5:
        print("  [PASS] Router global branch is mathematically equivalent to standalone FPI")
    else:
        print(f"  [FAIL] diff too large (residual={res_diff:.8f}, final={final_diff:.8f})")
        # Show detailed breakdown
        print(f"\n  base_logits mean_abs: {base_logits.abs().mean():.6f}")
        print(f"  standalone_final mean_abs: {standalone_final.abs().mean():.6f}")
        print(f"  router_final mean_abs:     {router_final.abs().mean():.6f}")


def test_router_cross_disabled_no_params():
    """Verify that when cross_max=0, cross_fpi is not created."""
    router = DualFPIResidualRouter(
        program_dim=64, drug_dim=64, fragment_dim=64,
        num_programs=8, num_drugs=9, max_fragments=16,
        gamma_global_max=0.05, gamma_global_init=0.05,
        gamma_cross_max=0.0, gamma_cross_init=0.0,
    )
    assert router.cross_fpi is None, "cross_fpi should be None when gamma_cross_max=0"
    # Count parameters
    total_params = sum(p.numel() for p in router.parameters())
    print(f"\n=== Cross-disabled parameter count ===")
    print(f"  Total parameters: {total_params}")
    print("  [PASS] cross_fpi not created, no wasted parameters")


if __name__ == "__main__":
    test_forward_global_shape()
    test_router_cross_disabled_no_params()
    test_router_vs_standalone_fpi()
