#!/usr/bin/env python3
"""
Phase 9-R: DualFPIResidualRouter Smoke Tests

Validates:
1. Router instantiation and forward pass shapes
2. Gamma gate ranges and initialization
3. Cross branch schedule (ramp)
4. Oracle mask mode (diagnostic)
5. Integration with ProgramFusionMLP
6. Mutual exclusion: use_fpi_router vs use_fpi
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytest

# Ensure config.tissue_map is set for isolated tests
import config
if not hasattr(config, 'tissue_map') or getattr(config.tissue_map, 'shape', None) == ():
    config.tissue_map = {"Tissue_0": 0, "Tissue_1": 1, "Tissue_2": 2}
    config.drug_feat = torch.randn(9, 256)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture
def dummy_inputs():
    B, K, H, D, M = 4, 8, 64, 9, 16
    z = torch.randn(B, K, H)
    q = torch.softmax(torch.randn(B, K), dim=-1)
    h_pool = torch.randn(B, H)
    drug_emb = torch.randn(D, H)
    frag_emb = torch.randn(D, M, H)
    frag_mask = torch.ones(D, M)
    base_logits = torch.randn(B, D)
    return {
        "z": z, "q": q, "h_pool": h_pool,
        "drug_emb": drug_emb, "frag_emb": frag_emb,
        "frag_mask": frag_mask, "base_logits": base_logits,
    }


@pytest.fixture
def drug_names():
    return [
        "5-Fluorouracil", "Cisplatin", "Cyclophosphamide",
        "Docetaxel", "Doxorubicin", "Etoposide",
        "Gemcitabine", "Paclitaxel", "Temozolomide"
    ]


# ------------------------------------------------------------------
# Test 1: Basic instantiation and forward
# ------------------------------------------------------------------
def test_router_instantiation_and_forward(dummy_inputs, drug_names):
    from mmpa_drp.fpi import DualFPIResidualRouter

    router = DualFPIResidualRouter(
        program_dim=64, drug_dim=64, fragment_dim=64,
        num_programs=8, num_drugs=9, max_fragments=16,
        gamma_global_max=0.05, gamma_global_init=0.05,
        gamma_cross_max=0.05, gamma_cross_init=0.0,
    )

    out = router(
        **dummy_inputs,
        current_epoch=0,
        drug_names=drug_names,
    )

    B, D = dummy_inputs["base_logits"].shape
    assert out["final_logits"].shape == (B, D)
    assert out["residual_global"].shape == (B, D)
    assert out["residual_cross"].shape == (B, D)
    assert out["gamma_global"].shape == (D,)
    assert out["gamma_cross"].shape == (D,)
    assert out["effective_gamma_cross"].shape == (D,)
    assert len(out["branch_preference"]) == D
    assert "schedule_factor" in out


# ------------------------------------------------------------------
# Test 2: Gamma ranges
# ------------------------------------------------------------------
def test_gamma_ranges(dummy_inputs, drug_names):
    from mmpa_drp.fpi import DualFPIResidualRouter

    router = DualFPIResidualRouter(
        gamma_global_max=0.05, gamma_global_init=0.05,
        gamma_cross_max=0.03, gamma_cross_init=0.0,
    )
    out = router(**dummy_inputs, current_epoch=0, drug_names=drug_names)

    gamma_g = out["gamma_global"]
    gamma_c = out["gamma_cross"]

    assert gamma_g.min() >= 0.0 and gamma_g.max() <= 0.05
    assert gamma_c.min() >= 0.0 and gamma_c.max() <= 0.03


# ------------------------------------------------------------------
# Test 3: Cross branch schedule (ramp)
# ------------------------------------------------------------------
def test_cross_branch_schedule(dummy_inputs, drug_names):
    from mmpa_drp.fpi import DualFPIResidualRouter

    router = DualFPIResidualRouter(
        gamma_global_max=0.05, gamma_global_init=0.05,
        gamma_cross_max=0.05, gamma_cross_init=0.05,
        router_cross_start_epoch=5,
        router_cross_ramp_epochs=5,
    )

    # Before start: cross should be 0
    out0 = router(**dummy_inputs, current_epoch=0, drug_names=drug_names)
    assert out0["schedule_factor"].item() == 0.0
    assert out0["effective_gamma_cross"].abs().max().item() < 1e-6

    # At start: partial ramp
    out5 = router(**dummy_inputs, current_epoch=5, drug_names=drug_names)
    assert out5["schedule_factor"].item() == pytest.approx(0.2, abs=0.01)

    # After ramp: full
    out10 = router(**dummy_inputs, current_epoch=10, drug_names=drug_names)
    assert out10["schedule_factor"].item() == 1.0
    assert out10["effective_gamma_cross"].abs().max().item() > 1e-6


# ------------------------------------------------------------------
# Test 4: Oracle mask mode
# ------------------------------------------------------------------
def test_oracle_mask_mode(dummy_inputs, drug_names):
    from mmpa_drp.fpi import DualFPIResidualRouter

    router = DualFPIResidualRouter(
        gamma_global_max=0.05, gamma_global_init=0.05,
        gamma_cross_max=0.05, gamma_cross_init=0.05,
        router_cross_mode="oracle_mask",
        router_oracle_cross_drugs="Temozolomide",
    )
    out = router(**dummy_inputs, current_epoch=0, drug_names=drug_names)

    cross_active = out["cross_active_mask"]
    # Temozolomide is last in drug_names (index 8)
    assert cross_active[8].item() == 1.0
    # All others should be 0
    assert cross_active[:8].sum().item() == 0.0


# ------------------------------------------------------------------
# Test 5: Integration with ProgramFusionMLP
# ------------------------------------------------------------------
def _make_dummy_encoder(input_dim, program_dim, num_programs):
    """Create a minimal dummy encoder for testing."""
    class DummyEncoder(nn.Module):
        def __init__(self, input_dim, program_dim, num_programs):
            super().__init__()
            self.fc = nn.Linear(input_dim, program_dim)
            self.num_programs = num_programs
        def forward(self, x):
            B = x.size(0)
            h = self.fc(x)
            z = h.unsqueeze(1).expand(B, self.num_programs, -1)
            q = torch.softmax(torch.randn(B, self.num_programs, device=x.device), dim=-1)
            return {"h": h, "z": z, "q": q}
    return DummyEncoder(input_dim, program_dim, num_programs)


def test_integration_with_program_fusion_mlp(dummy_inputs, drug_names):
    from mmpa_drp.models import ProgramFusionMLP

    B, K, H, D, M = 4, 8, 64, 9, 16
    gene_num = 100
    # fusion_input = bilinear(128) + gene_feat(64) + drug_feat(64) = 256
    input_dim = 128 + 64 + 64
    output_dim = 1
    drug_feat = torch.randn(D, 128)
    frag_fp = torch.randn(D, M, 1024)
    frag_mask = torch.ones(D, M)
    encoder = _make_dummy_encoder(gene_num, H, K)

    # Ensure config is valid for model init
    import config as _cfg
    if not hasattr(_cfg, 'tissue_map') or getattr(_cfg.tissue_map, 'shape', None) == ():
        _cfg.tissue_map = {"Tissue_0": 0, "Tissue_1": 1, "Tissue_2": 2}
        _cfg.drug_feat = torch.randn(9, 256)

    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=input_dim,
        output_dim=output_dim,
        gene_num=gene_num,
        drug_num=D,
        num_programs=K,
        program_dim=H,
        bilinear_dim=128,
        drug_feat=drug_feat,
        return_dict=True,
        use_fpi_router=True,
        fpi_router_config={
            "program_dim": H,
            "drug_dim": 64,
            "fragment_dim": H,
            "num_programs": K,
            "num_drugs": D,
            "max_fragments": M,
            "fp_dim": 1024,
            "gamma_global_max": 0.05,
            "gamma_global_init": 0.05,
            "gamma_cross_max": 0.05,
            "gamma_cross_init": 0.0,
        },
        frag_fp=frag_fp,
        frag_mask=frag_mask,
    )

    inputs = torch.randn(B, gene_num)
    tissue_idx = torch.zeros(B, dtype=torch.long)

    out = model(
        inputs, drug_feat, tissue_idx,
        return_dict=True,
        current_epoch=5,
        drug_names=drug_names,
    )

    assert "pred" in out
    assert out["pred"].shape == (B * D, 1)
    assert "router_out" in out
    router_out = out["router_out"]
    assert router_out["final_logits"].shape == (B, D)


# ------------------------------------------------------------------
# Test 6: Mutual exclusion in ProgramFusionMLP
# ------------------------------------------------------------------
def test_mutual_exclusion_fpi_router():
    from mmpa_drp.models import ProgramFusionMLP
    import config as _cfg
    if not hasattr(_cfg, 'tissue_map') or getattr(_cfg.tissue_map, 'shape', None) == ():
        _cfg.tissue_map = {"Tissue_0": 0, "Tissue_1": 1, "Tissue_2": 2}
        _cfg.drug_feat = torch.randn(9, 256)

    encoder = _make_dummy_encoder(100, 64, 8)
    with pytest.raises(ValueError):
        ProgramFusionMLP(
            encoder=encoder,
            input_dim=256,
            output_dim=1,
            gene_num=100, drug_num=9,
            use_fpi=True,
            use_fpi_router=True,
        )


# ------------------------------------------------------------------
# Test 7: No fragment data raises error or gracefully skips
# ------------------------------------------------------------------
def test_router_without_fragments():
    from mmpa_drp.models import ProgramFusionMLP
    import torch
    import config as _cfg
    if not hasattr(_cfg, 'tissue_map') or getattr(_cfg.tissue_map, 'shape', None) == ():
        _cfg.tissue_map = {"Tissue_0": 0, "Tissue_1": 1, "Tissue_2": 2}
        _cfg.drug_feat = torch.randn(9, 256)

    encoder = _make_dummy_encoder(100, 64, 8)
    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=256,
        output_dim=1,
        gene_num=100, drug_num=9,
        use_fpi_router=True,
        fpi_router_config={
            "program_dim": 64,
            "drug_dim": 64,
            "fragment_dim": 64,
            "num_programs": 8,
            "num_drugs": 9,
            "max_fragments": 16,
        },
        # NO frag_fp, frag_mask provided
    )

    B, gene_num = 2, 100
    inputs = torch.randn(B, gene_num)
    drug_feat = torch.randn(9, 128)
    tissue_idx = torch.zeros(B, dtype=torch.long)

    # Should run without error (router is bypassed when no frag data)
    out = model(inputs, drug_feat, tissue_idx, return_dict=True)
    assert "pred" in out
    assert "router_out" not in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
