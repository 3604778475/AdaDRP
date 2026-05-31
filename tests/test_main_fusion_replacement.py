"""
Tests for Phase 11: Main Fusion Replacement (DCF-LRBF)
"""
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# tissue_map fix for ProgramFusionMLP import
import config
if hasattr(config, "tissue_map"):
    try:
        _ = len(config.tissue_map)
    except TypeError:
        config.tissue_map = {"breast": 0, "blood": 1, "lung": 2, "skin": 3, "colon": 4}
else:
    config.tissue_map = {"breast": 0, "blood": 1, "lung": 2, "skin": 3, "colon": 4}

from mmpa_drp.models import DrugConditionedLowRankFusion, ProgramFusionMLP, MolecularProgramEncoder


class TestDrugConditionedLowRankFusion:
    def test_lrbf_shape(self):
        """Test 1: LRBF mode output shapes"""
        B, D, H = 4, 9, 64
        h_pool = torch.randn(B, H)
        drug_emb = torch.randn(D, H)

        fusion = DrugConditionedLowRankFusion(
            gene_dim=H, drug_dim=H, rank_dim=64, out_dim=128, mode="lrbf"
        )
        out = fusion(h_pool, drug_emb)

        assert out["fusion_feature"].shape == (B * D, 128)
        assert out["u_lr"].shape == (B * D, 64)
        assert out["h_mod"] is None
        assert out["film_gamma"] is None
        assert out["film_beta"] is None

    def test_dcf_lrbf_shape(self):
        """Test 2: DCF-LRBF mode output shapes"""
        B, D, H = 4, 9, 64
        h_pool = torch.randn(B, H)
        drug_emb = torch.randn(D, H)

        fusion = DrugConditionedLowRankFusion(
            gene_dim=H, drug_dim=H, rank_dim=64, out_dim=128, mode="dcf_lrbf"
        )
        out = fusion(h_pool, drug_emb)

        assert out["fusion_feature"].shape == (B * D, 128)
        assert out["h_mod"] is not None
        assert out["h_mod"].shape == (B, D, H)
        assert out["film_gamma"] is not None
        assert out["film_gamma"].shape == (D, H)
        assert out["film_beta"] is not None
        assert out["film_beta"].shape == (D, H)

    def test_dcf_near_identity_init(self):
        """Test 3: DCF initial modulation ≈ identity"""
        B, D, H = 4, 9, 64
        h_pool = torch.randn(B, H)
        drug_emb = torch.randn(D, H)

        fusion = DrugConditionedLowRankFusion(
            gene_dim=H, drug_dim=H, rank_dim=64, out_dim=128,
            mode="dcf_lrbf", film_init_scale=1e-3
        )
        out = fusion(h_pool, drug_emb)
        h_mod = out["h_mod"]
        h_pool_expand = h_pool.unsqueeze(1).expand(-1, D, -1)

        diff = (h_mod - h_pool_expand).abs().mean()
        assert diff.item() < 0.05, f"DCF init not near identity: diff={diff.item():.4f}"

    def test_different_out_dim(self):
        """Test that out_dim changes fusion_feature shape"""
        B, D, H = 2, 3, 16
        h_pool = torch.randn(B, H)
        drug_emb = torch.randn(D, H)

        fusion = DrugConditionedLowRankFusion(
            gene_dim=H, drug_dim=H, rank_dim=8, out_dim=256, mode="lrbf"
        )
        out = fusion(h_pool, drug_emb)
        assert out["fusion_feature"].shape == (B * D, 256)


class TestProgramFusionMLPMainFusion:
    def test_bilinear_path_unchanged(self):
        """Test 4: bilinear mode is default and works"""
        encoder = MolecularProgramEncoder(input_dim=1306, num_programs=8, program_dim=64)
        model = ProgramFusionMLP(
            encoder=encoder, input_dim=256, output_dim=1,
            gene_num=1306, gene_feat_dim=64, drug_feat_dim=64, drug_num=9,
            main_fusion_type="bilinear",
        )
        assert model.main_fusion_replacement is None
        assert model.main_fusion_type == "bilinear"

        model.eval()
        inputs = torch.randn(4, 1306)
        drug_feat = torch.randn(9, 128)
        out = model(inputs, drug_feat, torch.randint(0, 5, (4,)), return_dict=True)
        assert out["pred"].shape == (4 * 9, 1)
        assert getattr(model, "last_main_fusion_out", None) is None

    def test_lrbf_path(self):
        """Test 5: lrbf main fusion works"""
        encoder = MolecularProgramEncoder(input_dim=1306, num_programs=8, program_dim=64)
        model = ProgramFusionMLP(
            encoder=encoder, input_dim=256, output_dim=1,
            gene_num=1306, gene_feat_dim=64, drug_feat_dim=64, drug_num=9,
            main_fusion_type="lrbf",
            main_fusion_config={"rank_dim": 64, "out_dim": 128},
        )
        assert model.main_fusion_replacement is not None

        model.eval()
        inputs = torch.randn(4, 1306)
        drug_feat = torch.randn(9, 128)
        out = model(inputs, drug_feat, torch.randint(0, 5, (4,)), return_dict=True)
        assert out["pred"].shape == (4 * 9, 1)
        assert "main_fusion_out" in out
        assert out["main_fusion_out"] is not None

    def test_dcf_lrbf_with_fpi(self):
        """Test 6: dcf_lrbf + FPI coexistence"""
        encoder = MolecularProgramEncoder(input_dim=1306, num_programs=8, program_dim=64)
        model = ProgramFusionMLP(
            encoder=encoder, input_dim=256, output_dim=1,
            gene_num=1306, gene_feat_dim=64, drug_feat_dim=64, drug_num=9,
            main_fusion_type="dcf_lrbf",
            main_fusion_config={"rank_dim": 64, "out_dim": 128, "film_init_scale": 1e-3},
            use_fpi=True,
            fpi_config={"gamma_max": 0.05, "gamma_init": 0.05},
            frag_fp=torch.randint(0, 2, (9, 16, 1024)).float(),
            frag_mask=torch.ones(9, 16),
        )
        model.eval()
        inputs = torch.randn(4, 1306)
        drug_feat = torch.randn(9, 128)
        out = model(inputs, drug_feat, torch.randint(0, 5, (4,)), return_dict=True)
        assert out["pred"].shape == (4 * 9, 1)
        assert "main_fusion_out" in out
        assert "fpi_out" in out


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
