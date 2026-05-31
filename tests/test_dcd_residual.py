"""
Tests for Phase 10-D: Residual Decoupled Fusion (DCD-R)
"""
import torch
import torch.nn as nn
import pytest
import sys
import os

# Ensure project root is on path and config loads correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
# tissue_map may be a 0-d numpy object array which breaks len(); coerce to dict
if hasattr(config, "tissue_map"):
    try:
        _ = len(config.tissue_map)
    except TypeError:
        config.tissue_map = {"breast": 0, "blood": 1, "lung": 2, "skin": 3, "colon": 4}
else:
    config.tissue_map = {"breast": 0, "blood": 1, "lung": 2, "skin": 3, "colon": 4}

from mmpa_drp.models import ResidualDecoupledFusion, ProgramFusionMLP, MolecularProgramEncoder


class TestResidualDecoupledFusion:
    def test_shape_and_output(self):
        """Test 1: ResidualDecoupledFusion output shapes"""
        B, D, H = 4, 9, 64
        h_pool = torch.randn(B, H)
        drug_emb = torch.randn(D, H)
        base_logits = torch.randn(B, D)

        dcd = ResidualDecoupledFusion(
            hidden_dim=H,
            drug_dim=H,
            num_drugs=D,
            gamma_max=0.03,
            gamma_init=0.005,
        )
        out = dcd(h_pool, drug_emb, base_logits)

        assert out["final_base_logits"].shape == (B, D)
        assert out["residual_logits"].shape == (B, D)
        assert out["gamma"].shape == (D,)
        assert out["h_shared"].shape == (B, H)
        assert out["h_private"].shape == (B, H)
        assert isinstance(out["dcd_effect_abs_mean"], torch.Tensor)

    def test_gamma_near_init(self):
        """Test 2: gamma initialization close to gamma_init"""
        B, D, H = 4, 9, 64
        h_pool = torch.randn(B, H)
        drug_emb = torch.randn(D, H)
        base_logits = torch.randn(B, D)

        dcd = ResidualDecoupledFusion(
            hidden_dim=H,
            drug_dim=H,
            num_drugs=D,
            gamma_max=0.03,
            gamma_init=0.005,
        )
        out = dcd(h_pool, drug_emb, base_logits)
        gamma = out["gamma"]

        assert gamma.abs().mean().item() < 0.03
        # Should be near 0.005 initially
        assert (gamma - 0.005).abs().mean().item() < 1e-3

    def test_residual_tanh_bound(self):
        """Residual logits should be bounded by tanh"""
        B, D, H = 4, 9, 64
        h_pool = torch.randn(B, H)
        drug_emb = torch.randn(D, H)
        base_logits = torch.randn(B, D)

        dcd = ResidualDecoupledFusion(hidden_dim=H, drug_dim=H, num_drugs=D)
        out = dcd(h_pool, drug_emb, base_logits)

        assert out["residual_logits"].abs().max().item() <= 1.0 + 1e-5

    def test_use_product_and_absdiff(self):
        """Test with use_product=True and use_absdiff=True"""
        B, D, H = 4, 9, 64
        h_pool = torch.randn(B, H)
        drug_emb = torch.randn(D, H)
        base_logits = torch.randn(B, D)

        dcd = ResidualDecoupledFusion(
            hidden_dim=H,
            drug_dim=H,
            num_drugs=D,
            use_product=True,
            use_absdiff=True,
        )
        out = dcd(h_pool, drug_emb, base_logits)
        assert out["final_base_logits"].shape == (B, D)


class TestProgramFusionMLPWithDCD:
    def test_dcd_disabled_equivalent(self):
        """Test 3: use_dcd_residual=False should not create dcd_residual"""
        encoder = MolecularProgramEncoder(input_dim=1306, num_programs=8, program_dim=64)
        model = ProgramFusionMLP(
            encoder=encoder,
            input_dim=256,
            output_dim=1,
            gene_num=1306,
            gene_feat_dim=64,
            drug_feat_dim=64,
            drug_num=9,
            use_dcd_residual=False,
        )
        assert model.dcd_residual is None
        assert model.use_dcd_residual is False

    def test_dcd_enabled_forward(self):
        """Test 4: use_dcd_residual=True forward works and produces dcd_out"""
        encoder = MolecularProgramEncoder(input_dim=1306, num_programs=8, program_dim=64)
        model = ProgramFusionMLP(
            encoder=encoder,
            input_dim=256,
            output_dim=1,
            gene_num=1306,
            gene_feat_dim=64,
            drug_feat_dim=64,
            drug_num=9,
            use_dcd_residual=True,
            dcd_config={"gamma_max": 0.03, "gamma_init": 0.005},
        )
        model.eval()

        B = 4
        inputs = torch.randn(B, 1306)
        drug_feat = torch.randn(9, 128)
        tissue_idx = torch.randint(0, 5, (B,))

        out = model(inputs, drug_feat, tissue_idx, return_dict=True)
        assert "dcd_out" in out
        assert out["dcd_out"] is not None
        assert out["pred"].shape == (B * 9, 1)

    def test_dcd_plus_fpi_compatible(self):
        """Test 5: DCD + FPI coexistence"""
        encoder = MolecularProgramEncoder(input_dim=1306, num_programs=8, program_dim=64)
        model = ProgramFusionMLP(
            encoder=encoder,
            input_dim=256,
            output_dim=1,
            gene_num=1306,
            gene_feat_dim=64,
            drug_feat_dim=64,
            drug_num=9,
            use_dcd_residual=True,
            dcd_config={"gamma_max": 0.03, "gamma_init": 0.005},
            use_fpi=True,
            fpi_config={"gamma_max": 0.05, "gamma_init": 0.05},
            frag_fp=torch.randint(0, 2, (9, 16, 1024)).float(),
            frag_mask=torch.ones(9, 16),
        )
        model.eval()

        B = 4
        inputs = torch.randn(B, 1306)
        drug_feat = torch.randn(9, 128)
        tissue_idx = torch.randint(0, 5, (B,))

        out = model(inputs, drug_feat, tissue_idx, return_dict=True)
        assert "dcd_out" in out
        assert "fpi_out" in out
        assert out["dcd_out"] is not None
        assert out["fpi_out"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
