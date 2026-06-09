"""
scripts/test_gene_set_encoder.py
================================
Unit test for GeneSetMaskProgramEncoder and its integration with FPI.

Usage:
    python scripts/test_gene_set_encoder.py
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mmpa_drp.pathway_encoder import GeneSetMaskProgramEncoder
from mmpa_drp.fpi import FragmentProgramInteraction


def test_gene_set_encoder():
    print("=" * 60)
    print("Test 1: GeneSetMaskProgramEncoder forward shapes")
    print("=" * 60)

    B, G, P, H = 4, 100, 16, 64
    x = torch.randn(B, G)
    mask = torch.zeros(P, G)
    # Randomly assign genes to sets
    for p in range(P):
        indices = torch.randperm(G)[:10]
        mask[p, indices] = 1.0

    encoder = GeneSetMaskProgramEncoder(
        input_dim=G,
        gene_set_mask=mask,
        program_dim=H,
        dropout=0.1,
        use_pathway_embedding=True,
        use_soft_mask=False,
    )

    out = encoder(x)

    assert out["z"].shape == (B, P, H), f"z shape mismatch: {out['z'].shape} vs ({B},{P},{H})"
    assert out["q"].shape == (B, P), f"q shape mismatch: {out['q'].shape} vs ({B},{P})"
    assert out["h"].shape == (B, H), f"h shape mismatch: {out['h'].shape} vs ({B},{H})"

    q_sum = out["q"].sum(dim=1)
    assert torch.allclose(q_sum, torch.ones_like(q_sum), atol=1e-5), \
        f"q does not sum to 1: {q_sum}"

    print(f"  z shape      : {out['z'].shape}   [OK]")
    print(f"  q shape      : {out['q'].shape}   [OK]")
    print(f"  h shape      : {out['h'].shape}   [OK]")
    print(f"  q sum check  : {q_sum[0].item():.6f}  [OK]")
    print("\n[Test 1] PASSED")


def test_fpi_integration():
    print("\n" + "=" * 60)
    print("Test 2: FPI integration with GeneSetMaskProgramEncoder output")
    print("=" * 60)

    B, G, P, H, D, M = 4, 100, 16, 64, 9, 16
    x = torch.randn(B, G)
    mask = torch.zeros(P, G)
    for p in range(P):
        indices = torch.randperm(G)[:10]
        mask[p, indices] = 1.0

    encoder = GeneSetMaskProgramEncoder(
        input_dim=G,
        gene_set_mask=mask,
        program_dim=H,
        dropout=0.1,
        use_pathway_embedding=True,
    )
    out = encoder(x)
    z, q, h_pool = out["z"], out["q"], out["h"]

    # Dummy drug and fragment embeddings
    drug_emb = torch.randn(D, H)
    frag_emb = torch.randn(D, M, H)
    frag_mask = torch.ones(D, M)

    fpi = FragmentProgramInteraction(
        program_dim=H,
        drug_dim=H,
        fragment_dim=H,
        num_programs=P,
        num_drugs=D,
        max_fragments=M,
    )

    fpi_out = fpi(z=z, q=q, h=h_pool, drug_emb=drug_emb, frag_emb=frag_emb, frag_mask=frag_mask)

    assert fpi_out["residual_logits"].shape == (B, D), \
        f"residual_logits shape mismatch: {fpi_out['residual_logits'].shape} vs ({B},{D})"
    assert fpi_out["attn"].shape == (B, D, P, M), \
        f"attn shape mismatch: {fpi_out['attn'].shape} vs ({B},{D},{P},{M})"

    print(f"  residual_logits shape : {fpi_out['residual_logits'].shape}  [OK]")
    print(f"  attn shape            : {fpi_out['attn'].shape}  [OK]")
    print("\n[Test 2] PASSED")


def test_hybrid_encoder():
    print("\n" + "=" * 60)
    print("Test 3: HybridGeneSetLatentProgramEncoder")
    print("=" * 60)

    from mmpa_drp.pathway_encoder import HybridGeneSetLatentProgramEncoder
    from mmpa_drp.models import MolecularProgramEncoder

    B, G, P_gene, P_latent, H = 4, 100, 16, 3, 64
    x = torch.randn(B, G)

    mask = torch.zeros(P_gene, G)
    for p in range(P_gene):
        indices = torch.randperm(G)[:10]
        mask[p, indices] = 1.0

    gene_set_encoder = GeneSetMaskProgramEncoder(
        input_dim=G,
        gene_set_mask=mask,
        program_dim=H,
        dropout=0.1,
    )
    latent_encoder = MolecularProgramEncoder(
        input_dim=G,
        num_programs=P_latent,
        program_dim=H,
        hidden_dims=[512, 256, 128, 64],
        drop=0.1,
    )
    hybrid = HybridGeneSetLatentProgramEncoder(
        gene_set_encoder=gene_set_encoder,
        latent_encoder=latent_encoder,
        program_dim=H,
    )

    out = hybrid(x)
    P_total = P_gene + P_latent

    assert out["z"].shape == (B, P_total, H), f"z shape mismatch: {out['z'].shape}"
    assert out["q"].shape == (B, P_total), f"q shape mismatch: {out['q'].shape}"
    assert out["h"].shape == (B, H), f"h shape mismatch: {out['h'].shape}"
    assert out["z_gene_set"].shape == (B, P_gene, H), f"z_gene_set shape mismatch"
    assert out["z_latent"].shape == (B, P_latent, H), f"z_latent shape mismatch"

    q_sum = out["q"].sum(dim=1)
    assert torch.allclose(q_sum, torch.ones_like(q_sum), atol=1e-5)

    print(f"  z shape        : {out['z'].shape}  [OK]")
    print(f"  q shape        : {out['q'].shape}  [OK]")
    print(f"  h shape        : {out['h'].shape}  [OK]")
    print(f"  z_gene_set     : {out['z_gene_set'].shape}  [OK]")
    print(f"  z_latent       : {out['z_latent'].shape}  [OK]")
    print(f"  q sum check    : {q_sum[0].item():.6f}  [OK]")
    print(f"  num_programs   : {hybrid.num_programs}  [OK]")
    print("\n[Test 3] PASSED")


if __name__ == "__main__":
    test_gene_set_encoder()
    test_fpi_integration()
    test_hybrid_encoder()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
