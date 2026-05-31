"""
Smoke tests for FPI-SIM (Phase 9-FPI-SIM)
==========================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch

from mmpa_drp.fpi import FPISIMInteraction, FragmentProgramInteraction
from mmpa_drp.models import MolecularProgramEncoder, ProgramFusionMLP
import config

config.tissue_map = {'tissue_a': 0, 'tissue_b': 1}


# ============================================================================
# Test 1: FPISIMInteraction shape
# ============================================================================

def test_fpi_sim_shape():
    B, K, H = 4, 8, 64
    D, M = 9, 16

    z = torch.randn(B, K, H)
    q = torch.softmax(torch.randn(B, K), dim=1)
    h_pool = torch.randn(B, H)
    drug_emb = torch.randn(D, H)
    frag_emb = torch.randn(D, M, H)
    frag_mask = torch.ones(D, M)
    frag_mask[:, 12:] = 0.0

    sim = FPISIMInteraction(
        program_dim=H, drug_dim=H, fragment_dim=H,
        num_programs=K, num_drugs=D, max_fragments=M,
    )
    out = sim(z, q, h_pool, drug_emb, frag_emb, frag_mask)

    assert out["u"].shape == (B, D, H)
    assert out["residual_logits"].shape == (B, D)
    assert out["attn"].shape == (B, D, K, M)
    assert out["cross_attn"].shape == (B, D, K, M)
    assert out["program_weight"].shape == (B, K)
    assert out["fragment_weight"].shape == (D, M)
    assert out["union_weight"].shape == (B, D, K, M)
    assert out["score_cross"].shape == (B, D, K, M)
    assert out["score_program"].shape == (B, K)
    assert out["score_fragment"].shape == (D, M)


# ============================================================================
# Test 2: attention normalization
# ============================================================================

def test_fpi_sim_normalization():
    B, K, H = 4, 8, 64
    D, M = 9, 16

    z = torch.randn(B, K, H)
    q = torch.softmax(torch.randn(B, K), dim=1)
    h_pool = torch.randn(B, H)
    drug_emb = torch.randn(D, H)
    frag_emb = torch.randn(D, M, H)
    frag_mask = torch.ones(D, M)

    sim = FPISIMInteraction(
        program_dim=H, drug_dim=H, fragment_dim=H,
        num_programs=K, num_drugs=D, max_fragments=M,
    )
    out = sim(z, q, h_pool, drug_emb, frag_emb, frag_mask)

    # cross_attn sums to 1 over K*M
    cross_sum = out["cross_attn"].reshape(B, D, K * M).sum(dim=-1)
    assert torch.allclose(cross_sum, torch.ones(B, D), atol=1e-4)

    # program_weight sums to 1 over K
    prog_sum = out["program_weight"].sum(dim=-1)
    assert torch.allclose(prog_sum, torch.ones(B), atol=1e-4)

    # fragment_weight sums to 1 over M
    frag_sum = out["fragment_weight"].sum(dim=-1)
    assert torch.allclose(frag_sum, torch.ones(D), atol=1e-4)

    # final attn sums to 1 over K*M
    attn_sum = out["attn"].reshape(B, D, K * M).sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones(B, D), atol=1e-4)


# ============================================================================
# Test 3: padding fragment mask
# ============================================================================

def test_fpi_sim_padding_mask():
    B, K, H = 4, 8, 64
    D, M = 9, 16

    z = torch.randn(B, K, H)
    q = torch.softmax(torch.randn(B, K), dim=1)
    h_pool = torch.randn(B, H)
    drug_emb = torch.randn(D, H)
    frag_emb = torch.randn(D, M, H)
    frag_mask = torch.ones(D, M)
    frag_mask[:, 12:] = 0.0  # last 4 fragments are padding

    sim = FPISIMInteraction(
        program_dim=H, drug_dim=H, fragment_dim=H,
        num_programs=K, num_drugs=D, max_fragments=M,
    )
    out = sim(z, q, h_pool, drug_emb, frag_emb, frag_mask)

    # attn on padding should be near zero
    assert torch.allclose(out["attn"][:, :, :, 12:].sum(), torch.tensor(0.0), atol=1e-4)

    # fragment_weight on padding should be near zero
    assert torch.allclose(out["fragment_weight"][:, 12:].sum(), torch.tensor(0.0), atol=1e-4)


# ============================================================================
# Test 4: gamma=0 等价 base logits
# ============================================================================

def test_gamma_zero_equivalence_sim():
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

    model_on = ProgramFusionMLP(
        encoder=encoder, input_dim=256, output_dim=1,
        hidden_dims=[128, 64], gene_num=100, bilinear_dim=128,
        gene_feat_dim=64, drug_feat_dim=64, drug_num=9, drop=0.1,
        use_fpi=True,
        fpi_config={"gamma_max": 0.0, "gamma_init": 0.0, "fp_dim": 1024, "max_fragments": 16,
                    "use_fpi_sim": True, "fpi_sim_config": {}},
    )
    model_on.load_state_dict(model_off.state_dict(), strict=False)

    x = torch.randn(2, 100)
    drug_feat = torch.randn(9, 128)
    tissue_idx = torch.tensor([0, 1])
    frag_fp = torch.randn(9, 16, 1024)
    frag_mask = torch.ones(9, 16)

    model_off.eval()
    model_on.eval()
    with torch.no_grad():
        out_off = model_off(x, drug_feat, tissue_idx)
        out_on = model_on(x, drug_feat, tissue_idx, frag_fp=frag_fp, frag_mask=frag_mask)

    assert torch.allclose(out_off[0], out_on[0], atol=1e-4)


# ============================================================================
# Test 5: use_fpi_sim=False 等价当前 FPI2 path
# ============================================================================

def test_use_fpi_sim_false_routing():
    encoder = MolecularProgramEncoder(
        input_dim=100, num_programs=8, program_dim=64,
        hidden_dims=[128, 64], drop=0.1,
    )
    model_global = ProgramFusionMLP(
        encoder=encoder, input_dim=256, output_dim=1,
        hidden_dims=[128, 64], gene_num=100, bilinear_dim=128,
        gene_feat_dim=64, drug_feat_dim=64, drug_num=9, drop=0.1,
        use_fpi=True,
        fpi_config={"gamma_max": 0.05, "fp_dim": 1024, "max_fragments": 16, "use_fpi_sim": False},
    )

    x = torch.randn(2, 100)
    drug_feat = torch.randn(9, 128)
    tissue_idx = torch.tensor([0, 1])
    frag_fp = torch.randn(9, 16, 1024)
    frag_mask = torch.ones(9, 16)

    model_global.eval()
    with torch.no_grad():
        out = model_global(x, drug_feat, tissue_idx, frag_fp=frag_fp, frag_mask=frag_mask, return_dict=True)

    # Should use forward_global path
    assert model_global.fpi.use_fpi_sim is False
    assert model_global.fpi.sim_interaction is None
    assert out["fpi_out"] is not None


# ============================================================================
# Test 6: gradient flow
# ============================================================================

def test_fpi_sim_gradient_flow():
    encoder = MolecularProgramEncoder(
        input_dim=100, num_programs=8, program_dim=64,
        hidden_dims=[128, 64], drop=0.1,
    )
    model = ProgramFusionMLP(
        encoder=encoder, input_dim=256, output_dim=1,
        hidden_dims=[128, 64], gene_num=100, bilinear_dim=128,
        gene_feat_dim=64, drug_feat_dim=64, drug_num=9, drop=0.1,
        use_fpi=True,
        fpi_config={"gamma_max": 0.05, "fp_dim": 1024, "max_fragments": 16,
                    "use_fpi_sim": True, "fpi_sim_config": {}},
    )

    x = torch.randn(2, 100, requires_grad=True)
    drug_feat = torch.randn(9, 128)
    tissue_idx = torch.tensor([0, 1])
    frag_fp = torch.randn(9, 16, 1024)
    frag_mask = torch.ones(9, 16)

    out = model(x, drug_feat, tissue_idx, frag_fp=frag_fp, frag_mask=frag_mask)
    loss = out[0].mean()
    loss.backward()

    assert model.fpi.use_fpi_sim is True
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.fpi.sim_interaction.pair_mlp.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.fragment_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.encoder.parameters())


# ============================================================================
# Test 7: no target label leakage
# ============================================================================

def test_no_target_label_leakage_sim():
    sim = FPISIMInteraction()
    import inspect
    sig = inspect.signature(sim.forward)
    params = list(sig.parameters.keys())
    assert 'y_target' not in params
    assert 'target_response' not in params
    assert 'test_auc' not in params


# ============================================================================
# Test 8: param_str uniqueness
# ============================================================================

def test_param_str_uniqueness_sim():
    base = "pretrain_num_epochs_300_train_num_epochs_100_contrastive_num_epochs_200_K_3"

    def make_str(use_sim, ua, ub):
        s = base + "_fpi_gmax0.1_ginit0.0_qprior0.01_fgl10.001"
        if use_sim:
            s += "_sim"
            s += f"_ua{ua}_ub{ub}"
            s += "_pt1.0_ft1.0_ct1.0"
        else:
            s += "_global"
        s += "_m16_fp1024"
        return s

    s_global = make_str(False, 0, 0)
    s_sim_soft = make_str(True, 1.0, 1.0)
    s_sim_cross = make_str(True, 0.0, 0.0)
    s_sim_frag = make_str(True, 0.0, 1.0)

    assert s_global != s_sim_soft
    assert s_sim_soft != s_sim_cross
    assert s_sim_cross != s_sim_frag
    assert s_global != s_sim_frag


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
