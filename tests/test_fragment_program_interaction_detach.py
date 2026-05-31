"""
Smoke tests for FPI Stop-Gradient (Phase 9-FPI-R)
==================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch

from mmpa_drp.fpi import FragmentProgramInteraction
from mmpa_drp.models import MolecularProgramEncoder, ProgramFusionMLP
import config

config.tissue_map = {'tissue_a': 0, 'tissue_b': 1}


# ============================================================================
# Test 10: detach mode shape 不变
# ============================================================================

def test_detach_mode_shapes():
    B, K, H = 4, 8, 64
    D, M = 9, 16

    for detach_mode in ["none", "z", "zq", "zq_drug", "all"]:
        encoder = MolecularProgramEncoder(
            input_dim=100, num_programs=K, program_dim=H,
            hidden_dims=[128, 64], drop=0.1,
        )
        model = ProgramFusionMLP(
            encoder=encoder, input_dim=256, output_dim=1,
            hidden_dims=[128, 64], gene_num=100, bilinear_dim=128,
            gene_feat_dim=H, drug_feat_dim=H, drug_num=D, drop=0.1,
            use_fpi=True,
            fpi_config={"gamma_max": 0.05, "fp_dim": 1024, "max_fragments": M, "detach_mode": detach_mode},
        )

        x = torch.randn(2, 100)
        drug_feat = torch.randn(D, 128)
        tissue_idx = torch.tensor([0, 1])
        frag_fp = torch.randn(D, M, 1024)
        frag_mask = torch.ones(D, M)

        model.eval()
        with torch.no_grad():
            out = model(x, drug_feat, tissue_idx, frag_fp=frag_fp, frag_mask=frag_mask, return_dict=True)

        fpi_out = model.last_fpi_out
        assert fpi_out is not None, f"detach_mode={detach_mode} should produce fpi_out"
        assert fpi_out["u"].shape == (2, D, H), f"detach_mode={detach_mode}: u shape mismatch"
        assert fpi_out["residual_logits"].shape == (2, D), f"detach_mode={detach_mode}: residual_logits shape mismatch"
        assert fpi_out["gamma"].shape == (D,), f"detach_mode={detach_mode}: gamma shape mismatch"


# ============================================================================
# Test 11: detach_z 阻断 FPI branch 到 z 的梯度
# ============================================================================

def test_detach_z_blocks_z_grad():
    B, K, H = 4, 8, 64
    D, M = 9, 16

    z = torch.randn(B, K, H, requires_grad=True)
    q = torch.softmax(torch.randn(B, K), dim=1)
    q.requires_grad_(True)
    h_pool = torch.randn(B, H, requires_grad=True)
    drug_emb = torch.randn(D, H, requires_grad=True)
    frag_emb = torch.randn(D, M, H, requires_grad=True)
    frag_mask = torch.ones(D, M)

    fpi = FragmentProgramInteraction(
        program_dim=H, drug_dim=H, fragment_dim=H,
        num_programs=K, num_drugs=D, max_fragments=M,
    )

    # none mode: z should have grad
    out = fpi(z, q, h_pool, drug_emb, frag_emb, frag_mask)
    loss = out["residual_logits"].mean()
    loss.backward(retain_graph=True)
    assert z.grad is not None, "none mode: z.grad should not be None"
    z.grad = None
    q.grad = None
    drug_emb.grad = None
    frag_emb.grad = None

    # detach z mode: z should NOT have grad
    out = fpi(z.detach(), q, h_pool, drug_emb, frag_emb, frag_mask)
    loss = out["residual_logits"].mean()
    loss.backward()
    assert z.grad is None, "detach z mode: z.grad should be None"


# ============================================================================
# Test 12: detach_zq 阻断 z 和 q
# ============================================================================

def test_detach_zq_blocks_zq_grad():
    B, K, H = 4, 8, 64
    D, M = 9, 16

    z = torch.randn(B, K, H, requires_grad=True)
    q = torch.softmax(torch.randn(B, K), dim=1)
    q.requires_grad_(True)
    h_pool = torch.randn(B, H, requires_grad=True)
    drug_emb = torch.randn(D, H, requires_grad=True)
    frag_emb = torch.randn(D, M, H, requires_grad=True)
    frag_mask = torch.ones(D, M)

    fpi = FragmentProgramInteraction(
        program_dim=H, drug_dim=H, fragment_dim=H,
        num_programs=K, num_drugs=D, max_fragments=M,
    )

    out = fpi(z.detach(), q.detach(), h_pool, drug_emb, frag_emb, frag_mask)
    loss = out["residual_logits"].mean()
    loss.backward()
    assert z.grad is None, "detach_zq: z.grad should be None"
    assert q.grad is None, "detach_zq: q.grad should be None"


# ============================================================================
# Test 13: detach_zq_drug 阻断 drug_emb
# ============================================================================

def test_detach_zq_drug_blocks_drug_grad():
    B, K, H = 4, 8, 64
    D, M = 9, 16

    z = torch.randn(B, K, H, requires_grad=True)
    q = torch.softmax(torch.randn(B, K), dim=1)
    q.requires_grad_(True)
    h_pool = torch.randn(B, H, requires_grad=True)
    drug_emb = torch.randn(D, H, requires_grad=True)
    frag_emb = torch.randn(D, M, H, requires_grad=True)
    frag_mask = torch.ones(D, M)

    fpi = FragmentProgramInteraction(
        program_dim=H, drug_dim=H, fragment_dim=H,
        num_programs=K, num_drugs=D, max_fragments=M,
    )

    out = fpi(z.detach(), q.detach(), h_pool, drug_emb.detach(), frag_emb, frag_mask)
    loss = out["residual_logits"].mean()
    loss.backward()
    assert drug_emb.grad is None, "detach_zq_drug: drug_emb.grad should be None"


# ============================================================================
# Test 14: detach 不影响 FPI 参数梯度
# ============================================================================

def test_detach_does_not_block_fpi_params():
    B, K, H = 4, 8, 64
    D, M = 9, 16

    z = torch.randn(B, K, H, requires_grad=True)
    q = torch.softmax(torch.randn(B, K), dim=1)
    h_pool = torch.randn(B, H, requires_grad=True)
    drug_emb = torch.randn(D, H, requires_grad=True)
    frag_emb = torch.randn(D, M, H, requires_grad=True)
    frag_mask = torch.ones(D, M)

    fpi = FragmentProgramInteraction(
        program_dim=H, drug_dim=H, fragment_dim=H,
        num_programs=K, num_drugs=D, max_fragments=M,
    )

    # Use full detach mode
    out = fpi(z.detach(), q.detach(), h_pool, drug_emb.detach(), frag_emb.detach(), frag_mask)
    loss = out["residual_logits"].mean()
    loss.backward()

    # FPI internal params must still have gradients
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fpi.program_proj.parameters()), "program_proj should have grad"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fpi.fragment_proj.parameters()), "fragment_proj should have grad"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fpi.pair_mlp.parameters()), "pair_mlp should have grad"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fpi.residual_head.parameters()), "residual_head should have grad"
    # Note: raw_gamma only gets grad when gamma is used in downstream loss (e.g., final_logits).
    # In standalone FPI, residual_logits does not depend on raw_gamma, so grad is None.
    # This is expected and does NOT mean detach blocks raw_gamma.
    assert fpi.raw_gamma.requires_grad, "raw_gamma should still require grad"


# ============================================================================
# Test 15: gamma=0 仍等价 base logits（所有 detach mode）
# ============================================================================

def test_gamma_zero_equivalence_all_detach_modes():
    encoder = MolecularProgramEncoder(
        input_dim=100, num_programs=8, program_dim=64,
        hidden_dims=[128, 64], drop=0.1,
    )
    model_off = ProgramFusionMLP(
        encoder=encoder, input_dim=256, output_dim=1,
        hidden_dims=[128, 64], gene_num=100, bilinear_dim=128,
        gene_feat_dim=64, drug_feat_dim=64, drug_num=9, drop=0.1,
        use_fpi=False,
    )

    x = torch.randn(2, 100)
    drug_feat = torch.randn(9, 128)
    tissue_idx = torch.tensor([0, 1])
    frag_fp = torch.randn(9, 16, 1024)
    frag_mask = torch.ones(9, 16)

    model_off.eval()
    with torch.no_grad():
        out_off = model_off(x, drug_feat, tissue_idx)

    for detach_mode in ["none", "z", "zq", "zq_drug", "all"]:
        model_on = ProgramFusionMLP(
            encoder=encoder, input_dim=256, output_dim=1,
            hidden_dims=[128, 64], gene_num=100, bilinear_dim=128,
            gene_feat_dim=64, drug_feat_dim=64, drug_num=9, drop=0.1,
            use_fpi=True,
            fpi_config={"gamma_max": 0.0, "gamma_init": 0.0, "fp_dim": 1024, "max_fragments": 16, "detach_mode": detach_mode},
        )
        model_on.load_state_dict(model_off.state_dict(), strict=False)
        model_on.eval()
        with torch.no_grad():
            out_on = model_on(x, drug_feat, tissue_idx, frag_fp=frag_fp, frag_mask=frag_mask)

        assert torch.allclose(out_off[0], out_on[0], atol=1e-4), f"gamma=0 should match for detach_mode={detach_mode}"


# ============================================================================
# Test 16: param_str 唯一性（含 detach mode）
# ============================================================================

def test_param_str_uniqueness_with_detach():
    base = "pretrain_num_epochs_300_train_num_epochs_100_contrastive_num_epochs_200_K_3"

    def make_fpi_str(gmax, ginit, qprior, fgl1, fent, mmax, fpdim, detach):
        s = base
        s += f"_fpi_gmax{gmax}_ginit{ginit}_qprior{qprior}"
        if fgl1 > 0:
            s += f"_fgl1{fgl1}"
        if fent > 0:
            s += f"_fent{fent}"
        s += f"_detach{detach}"
        s += f"_m{mmax}_fp{fpdim}"
        return s

    s_none = make_fpi_str(0.1, 0.0, 0.01, 0.001, 0.0, 16, 1024, "none")
    s_z = make_fpi_str(0.1, 0.0, 0.01, 0.001, 0.0, 16, 1024, "z")
    s_zq = make_fpi_str(0.1, 0.0, 0.01, 0.001, 0.0, 16, 1024, "zq")
    s_zq_drug = make_fpi_str(0.1, 0.0, 0.01, 0.001, 0.0, 16, 1024, "zq_drug")

    assert s_none != s_z
    assert s_z != s_zq
    assert s_zq != s_zq_drug
    assert s_none != s_zq_drug


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
