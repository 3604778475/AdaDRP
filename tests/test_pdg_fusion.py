"""
Phase 10 PDG Smoke Tests
========================
Run: pytest tests/test_pdg_fusion.py -q
"""

import pytest
import torch
import torch.nn as nn

import numpy as np
import config
# config.tissue_map may be an unsized ndarray when imported in isolation;
# patch it to a minimal sized array for smoke tests.
if not hasattr(config, "tissue_map") or getattr(config.tissue_map, "shape", None) is None or len(getattr(config.tissue_map, "shape", ())) == 0:
    config.tissue_map = np.array([0] * 10)

from mmpa_drp.models import PseudoDomainGuide, ProgramFusionMLP, MolecularProgramEncoder


class DummyEncoder(nn.Module):
    def __init__(self, h_dim=64):
        super().__init__()
        self.h_dim = h_dim

    def forward(self, x):
        B = x.size(0)
        return {
            "z": torch.randn(B, 8, self.h_dim),
            "q": torch.softmax(torch.randn(B, 8), dim=1),
            "h": torch.randn(B, self.h_dim),
        }


def test_pseudo_domain_guide_shape():
    """Test 1: PseudoDomainGuide output shapes."""
    B, H, K, pdg_dim = 4, 64, 8, 32
    h_pool = torch.randn(B, H)
    centroids = torch.randn(K, H)

    pdg = PseudoDomainGuide(centroid_dim=H, pdg_dim=pdg_dim)
    out = pdg(h_pool, centroids)

    assert out["pdg_emb"].shape == (B, pdg_dim)
    assert out["assign"].shape == (B,)
    assert out["sim"].shape == (B, K)
    assert out["assign"].dtype == torch.int64


def test_program_fusion_mlp_no_pdg_backward_compatible():
    """Test 2: use_pdg=False must produce same input_dim and not error."""
    encoder = DummyEncoder(h_dim=64)
    base_input_dim = 64 + 64 + 128  # gene_feat_dim + drug_feat_dim + bilinear_dim

    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=base_input_dim,
        output_dim=1,
        hidden_dims=[64, 32],
        gene_num=1306,
        bilinear_dim=128,
        gene_feat_dim=64,
        drug_feat_dim=64,
        drug_num=9,
        use_pdg=False,
    )

    B = 3
    inputs = torch.randn(B, 1306)
    drug_feat = torch.randn(9, 128)
    tissue_idx = torch.randint(0, 5, (B,))

    out = model(inputs, drug_feat, tissue_idx)
    assert out[0].shape == (B * 9, 1)  # pred shape
    assert model.last_pdg_assign is None


def test_program_fusion_mlp_pdg_shape():
    """Test 3: use_pdg=True shape correctness."""
    encoder = DummyEncoder(h_dim=64)
    base_input_dim = 64 + 64 + 128
    pdg_dim = 32
    input_dim = base_input_dim + pdg_dim
    K = 8
    centroids = torch.randn(K, 64)

    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=input_dim,
        output_dim=1,
        hidden_dims=[64, 32],
        gene_num=1306,
        bilinear_dim=128,
        gene_feat_dim=64,
        drug_feat_dim=64,
        drug_num=9,
        use_pdg=True,
        pdg_config={"pdg_dim": pdg_dim, "assign_mode": "nearest", "normalize": True},
        pdg_centroids=centroids,
    )

    B = 3
    inputs = torch.randn(B, 1306)
    drug_feat = torch.randn(9, 128)
    tissue_idx = torch.randint(0, 5, (B,))

    out = model(inputs, drug_feat, tissue_idx)
    assert out[0].shape == (B * 9, 1)
    assert model.last_pdg_assign is not None
    assert model.last_pdg_assign.shape == (B,)


def test_pdg_plus_fpi_compatible():
    """Test 4: PDG + FPI must coexist."""
    encoder = DummyEncoder(h_dim=64)
    base_input_dim = 64 + 64 + 128
    pdg_dim = 32
    input_dim = base_input_dim + pdg_dim
    K = 8
    centroids = torch.randn(K, 64)

    # Dummy fragment tensors
    frag_fp = torch.randn(9, 16, 1024)
    frag_mask = torch.ones(9, 16, dtype=torch.long)

    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=input_dim,
        output_dim=1,
        hidden_dims=[64, 32],
        gene_num=1306,
        bilinear_dim=128,
        gene_feat_dim=64,
        drug_feat_dim=64,
        drug_num=9,
        use_pdg=True,
        pdg_config={"pdg_dim": pdg_dim},
        pdg_centroids=centroids,
        use_fpi=True,
        fpi_config={
            "program_dim": 64,
            "drug_dim": 64,
            "fragment_dim": 64,
            "num_programs": 8,
            "num_drugs": 9,
            "max_fragments": 16,
            "fp_dim": 1024,
            "gamma_max": 0.05,
            "gamma_init": 0.0,
        },
        frag_fp=frag_fp,
        frag_mask=frag_mask,
    )

    B = 3
    inputs = torch.randn(B, 1306)
    drug_feat = torch.randn(9, 128)
    tissue_idx = torch.randint(0, 5, (B,))

    out = model(inputs, drug_feat, tissue_idx, return_dict=True)
    assert out["pred"].shape == (B * 9, 1)
    assert model.last_pdg_assign is not None
    assert model.last_fpi_out is not None


def test_pdg_missing_centroids_raises():
    """Test 5: use_pdg=True without centroids must raise ValueError."""
    encoder = DummyEncoder(h_dim=64)
    with pytest.raises(ValueError, match="use_pdg=True requires pdg_centroids"):
        ProgramFusionMLP(
            encoder=encoder,
            input_dim=64 + 64 + 128 + 32,
            output_dim=1,
            hidden_dims=[64, 32],
            gene_num=1306,
            bilinear_dim=128,
            gene_feat_dim=64,
            drug_feat_dim=64,
            drug_num=9,
            use_pdg=True,
            pdg_config={"pdg_dim": 32},
            pdg_centroids=None,
        )
