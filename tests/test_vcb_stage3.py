"""
Phase 12: Variational Contrastive Bottleneck (VCB) Smoke Tests
===============================================================
Test 1-4: VariationalContrastiveBottleneck standalone
Test 5-6: ProgramFusionMLP encode_contrastive_feature integration
"""

import pytest
import torch
import torch.nn as nn
import config

# tissue_map fix for ProgramFusionMLP import
if hasattr(config, "tissue_map"):
    try:
        _ = len(config.tissue_map)
    except TypeError:
        config.tissue_map = {"breast": 0, "blood": 1, "lung": 2, "skin": 3, "colon": 4}
else:
    config.tissue_map = {"breast": 0, "blood": 1, "lung": 2, "skin": 3, "colon": 4}

from mmpa_drp.models import VariationalContrastiveBottleneck, ProgramFusionMLP


class DummyEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.out_dim = out_dim
        self.program_dim = out_dim
        self.num_programs = 8

    def forward(self, x):
        B = x.size(0)
        z = torch.randn(B, self.num_programs, self.program_dim)
        q = torch.softmax(torch.randn(B, self.num_programs), dim=1)
        h = (q.unsqueeze(-1) * z).sum(dim=1)
        return {"z": z, "q": q, "h": h}


class TestVariationalContrastiveBottleneck:
    def test_shape(self):
        N, F = 16, 128
        vcb = VariationalContrastiveBottleneck(in_dim=F, z_dim=128, hidden_dim=128)
        x = torch.randn(N, F)
        out = vcb(x, sample=True)

        assert out["z"].shape == (N, 128)
        assert out["mu"].shape == (N, 128)
        assert out["logvar"].shape == (N, 128)
        assert out["kl"].shape == (N,)
        assert (out["kl"] >= 0).all()

    def test_z_dim_change(self):
        N, F = 16, 128
        vcb = VariationalContrastiveBottleneck(in_dim=F, z_dim=64, hidden_dim=128)
        x = torch.randn(N, F)
        out = vcb(x, sample=True)

        assert out["z"].shape == (N, 64)
        assert out["mu"].shape == (N, 64)
        assert out["logvar"].shape == (N, 64)

    def test_eval_mode_deterministic(self):
        N, F = 16, 128
        vcb = VariationalContrastiveBottleneck(in_dim=F, z_dim=128, hidden_dim=128)
        x = torch.randn(N, F)
        vcb.eval()
        out1 = vcb(x, sample=True)
        out2 = vcb(x, sample=True)

        assert torch.allclose(out1["z"], out2["z"], atol=1e-6)

    def test_train_mode_stochastic(self):
        N, F = 16, 128
        vcb = VariationalContrastiveBottleneck(in_dim=F, z_dim=128, hidden_dim=128)
        x = torch.randn(N, F)
        vcb.train()
        out1 = vcb(x, sample=True)
        out2 = vcb(x, sample=True)

        # Extremely unlikely to be exactly equal with stochastic sampling
        assert not torch.allclose(out1["z"], out2["z"], atol=1e-6)

    def test_kl_nonnegative(self):
        N, F = 16, 128
        vcb = VariationalContrastiveBottleneck(in_dim=F, z_dim=128, hidden_dim=128)
        x = torch.randn(N, F)
        out = vcb(x, sample=True)
        assert (out["kl"] >= 0).all()

    def test_normalize_z(self):
        N, F = 16, 128
        vcb = VariationalContrastiveBottleneck(in_dim=F, z_dim=128, hidden_dim=128, normalize_z=True)
        x = torch.randn(N, F)
        out = vcb(x, sample=True)
        norms = out["z"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_no_normalize_z(self):
        N, F = 16, 128
        vcb = VariationalContrastiveBottleneck(in_dim=F, z_dim=128, hidden_dim=128, normalize_z=False)
        x = torch.randn(N, F)
        out = vcb(x, sample=True)
        norms = out["z"].norm(dim=-1)
        # Should not be all ones
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2)


class TestProgramFusionMLPVCB:
    def _make_model(self, use_vcb=False, vcb_config=None):
        encoder = DummyEncoder(out_dim=64)
        model = ProgramFusionMLP(
            encoder=encoder,
            input_dim=64 + 64 + 128,
            output_dim=1,
            hidden_dims=[64, 32],
            gene_num=1306,
            bilinear_dim=128,
            gene_feat_dim=64,
            drug_feat_dim=64,
            drug_num=9,
            drop=0.1,
            use_vcb=use_vcb,
            vcb_config=vcb_config,
        )
        return model

    def test_encode_contrastive_feature_vcb_false(self):
        model = self._make_model(use_vcb=False)
        feat = torch.randn(16, 32)
        out = model.encode_contrastive_feature(feat, sample=True)

        assert torch.equal(out["z"], feat)
        assert out["mu"] is None
        assert out["logvar"] is None
        assert out["kl"].shape == (16,)
        assert (out["kl"] == 0).all()

    def test_encode_contrastive_feature_vcb_true(self):
        model = self._make_model(use_vcb=True, vcb_config={"z_dim": 32, "hidden_dim": 32})
        feat = torch.randn(16, 32)
        out = model.encode_contrastive_feature(feat, sample=True)

        assert out["z"].shape == (16, 32)
        assert out["mu"].shape == (16, 32)
        assert out["logvar"].shape == (16, 32)
        assert out["kl"].shape == (16,)
        assert (out["kl"] >= 0).all()

    def test_encode_contrastive_feature_vcb_true_eval_deterministic(self):
        model = self._make_model(use_vcb=True, vcb_config={"z_dim": 32, "hidden_dim": 32})
        feat = torch.randn(16, 32)
        model.eval()
        out1 = model.encode_contrastive_feature(feat, sample=True)
        out2 = model.encode_contrastive_feature(feat, sample=True)

        assert torch.allclose(out1["z"], out2["z"], atol=1e-6)

    def test_vcb_forward_integration(self):
        model = self._make_model(use_vcb=True, vcb_config={"z_dim": 32, "hidden_dim": 32})
        model.eval()
        B = 4
        inputs = torch.randn(B, 1306)
        drug_feat = torch.randn(9, 128)
        tissue_idx = torch.randint(0, 5, (B,))

        with torch.no_grad():
            outputs = model(inputs, drug_feat, tissue_idx, return_fused_feature=True)

        assert "pred" in outputs
        assert "fused_feature" in outputs
        assert model.vcb is not None
