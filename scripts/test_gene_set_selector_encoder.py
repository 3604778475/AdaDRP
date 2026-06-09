"""
scripts/test_gene_set_selector_encoder.py
=========================================
Unit test for GeneSetSemanticSelectorEncoder.

Usage:
    python scripts/test_gene_set_selector_encoder.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from mmpa_drp.pathway_encoder import GeneSetSemanticSelectorEncoder


def test_shapes():
    B, G, P, H, K = 4, 1306, 16, 64, 3

    # Random mask: 16 gene sets x 1306 genes
    mask = torch.zeros(P, G)
    for i in range(P):
        # Each gene set covers ~50 random genes
        idx = torch.randperm(G)[:50]
        mask[i, idx] = 1.0

    encoder = GeneSetSemanticSelectorEncoder(
        input_dim=G,
        gene_set_mask=mask,
        program_dim=H,
        num_final_programs=K,
        dropout=0.1,
        use_pathway_embedding=True,
        use_soft_mask=False,
        soft_mask_scale=0.05,
        selector_num_heads=4,
        selector_dropout=0.1,
    )

    x = torch.randn(B, G)
    out = encoder(x)

    # Check shapes
    assert out["z"].shape == (B, K, H), f"z shape mismatch: {out['z'].shape}"
    assert out["q"].shape == (B, K), f"q shape mismatch: {out['q'].shape}"
    assert out["h"].shape == (B, H), f"h shape mismatch: {out['h'].shape}"
    assert out["gene_set_tokens"].shape == (B, P, H), f"gene_set_tokens shape mismatch: {out['gene_set_tokens'].shape}"
    assert out["selector_attn"].shape == (B, K, P), f"selector_attn shape mismatch: {out['selector_attn'].shape}"

    # Check softmax normalization
    q_sum = out["q"].sum(dim=1)
    assert torch.allclose(q_sum, torch.ones_like(q_sum), atol=1e-5), "q not softmax-normalized"

    # Check selector attention normalization (over gene-set dim)
    attn_sum = out["selector_attn"].sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), "selector_attn not normalized over gene sets"

    print("[PASS] Shape and normalization tests passed")


def test_fpi_integration():
    """Test that FPI receives correct shapes when using gene_set_selector."""
    from mmpa_drp.models import ProgramFusionMLP, FragmentProgramInteraction

    B, G, P, H, K, D, M = 4, 1306, 16, 64, 3, 9, 16

    mask = torch.zeros(P, G)
    for i in range(P):
        idx = torch.randperm(G)[:50]
        mask[i, idx] = 1.0

    encoder = GeneSetSemanticSelectorEncoder(
        input_dim=G,
        gene_set_mask=mask,
        program_dim=H,
        num_final_programs=K,
        selector_num_heads=4,
        selector_dropout=0.1,
    )

    # Dummy config
    class DummyConfig:
        tissue_map = {f"t{i}": i for i in range(5)}

    # Build ProgramFusionMLP with FPI
    fusion = ProgramFusionMLP(
        encoder=encoder,
        input_dim=G,
        output_dim=1,
        hidden_dims=[128, 64],
        gene_num=G,
        bilinear_dim=128,
        gene_feat_dim=H,
        drug_feat_dim=64,
        drug_num=D,
        drop=0.1,
        use_fpi=True,
        fpi_config={
            "fragment_dim": H,
            "num_programs": K,
            "num_drugs": D,
            "max_fragments": M,
            "dropout": 0.1,
            "gamma_max": 0.1,
            "gamma_init": 0.05,
            "q_prior_weight": 0.01,
        },
        frag_fp=torch.randn(D, M, 1024),
        frag_mask=torch.ones(D, M, dtype=torch.bool),
    )

    # Forward
    x = torch.randn(B, G)
    drug_feat = torch.randn(D, 64)
    tissue_idx = torch.randint(0, 5, (B,))

    out = fusion(
        x, drug_feat, tissue_idx,
        return_dict=True,
        return_encoder_out=True,
    )

    assert "pred" in out
    assert out["pred"].shape == (B * D, 1)
    assert "encoder_out" in out
    encoder_out = out["encoder_out"]
    assert encoder_out["z"].shape == (B, K, H)
    assert encoder_out["q"].shape == (B, K)
    assert "selector_attn" in encoder_out
    assert encoder_out["selector_attn"].shape == (B, K, P)

    # Check FPI output uses K=3
    if "fpi_out" in out:
        fpi_out = out["fpi_out"]
        assert fpi_out["attn"].shape[2] == K, f"FPI attn should have K={K} program dim, got {fpi_out['attn'].shape}"

    print("[PASS] FPI integration test passed")


def test_program_names():
    mask = torch.zeros(16, 1306)
    mask[:, :10] = 1.0

    names = [f"SET_{i}" for i in range(16)]
    encoder = GeneSetSemanticSelectorEncoder(
        input_dim=1306,
        gene_set_mask=mask,
        program_dim=64,
        num_final_programs=3,
        program_names=names,
    )

    assert encoder.gene_set_encoder.program_names == names
    print("[PASS] Program names test passed")


if __name__ == "__main__":
    test_shapes()
    test_fpi_integration()
    test_program_names()
    print("\n=== All tests passed ===")
