"""
Phase 5 Smoke Tests: ConvNeXt V2-inspired Backbones
====================================================
Test GRN1DVector, GRN1DToken, GRNMLPBackbone, ConvNeXt1DBlock,
ConvNeXt1DBackbone, and MolecularProgramEncoder with different backbone_type.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

def test_grn1d_vector_identity():
    from mmpa_drp.backbones import GRN1DVector
    x = torch.randn(4, 64)
    grn = GRN1DVector(64)
    out = grn(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    # gamma/beta init to 0 => identity
    assert torch.allclose(out, x, atol=1e-6), "GRN1DVector should be identity at init"
    print("✅ test_grn1d_vector_identity passed")


def test_grn1d_token_identity():
    from mmpa_drp.backbones import GRN1DToken
    x = torch.randn(4, 32, 64)
    grn = GRN1DToken(64)
    out = grn(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert torch.allclose(out, x, atol=1e-6), "GRN1DToken should be identity at init"
    print("✅ test_grn1d_token_identity passed")


def test_grn_mlp_backbone():
    from mmpa_drp.backbones import GRNMLPBackbone
    x = torch.randn(4, 1306)
    backbone = GRNMLPBackbone(input_dim=1306, hidden_dim=64, dropout=0.1, norm_type="layernorm")
    out = backbone(x)
    assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"
    print("✅ test_grn_mlp_backbone passed")


def test_convnext1d_block():
    from mmpa_drp.backbones import ConvNeXt1DBlock
    x = torch.randn(4, 64, 64)
    block = ConvNeXt1DBlock(dim=64, kernel_size=7, expansion=4, dropout=0.1)
    out = block(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print("✅ test_convnext1d_block passed")


def test_convnext1d_backbone():
    from mmpa_drp.backbones import ConvNeXt1DBackbone
    x = torch.randn(4, 1306)
    backbone = ConvNeXt1DBackbone(
        input_dim=1306,
        hidden_dim=64,
        num_gene_tokens=64,
        token_dim=64,
        convnext_depth=2,
        convnext_kernel_size=7,
        convnext_expansion=4,
        dropout=0.1,
    )
    out = backbone(x)
    assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"
    print("✅ test_convnext1d_backbone passed")


def test_molecular_program_encoder_mlp():
    from mmpa_drp.models import MolecularProgramEncoder
    x = torch.randn(4, 1306)
    encoder = MolecularProgramEncoder(
        input_dim=1306, num_programs=8, program_dim=64,
        hidden_dims=[512, 256, 128, 64], drop=0.1,
        backbone_type="mlp",
    )
    out = encoder(x)
    assert out["z"].shape == (4, 8, 64)
    assert out["q"].shape == (4, 8)
    assert out["h"].shape == (4, 64)
    print("✅ test_molecular_program_encoder_mlp passed")


def test_molecular_program_encoder_grn_mlp():
    from mmpa_drp.models import MolecularProgramEncoder
    x = torch.randn(4, 1306)
    encoder = MolecularProgramEncoder(
        input_dim=1306, num_programs=8, program_dim=64,
        hidden_dims=[512, 256, 128, 64], drop=0.1,
        backbone_type="grn_mlp", norm_type="layernorm",
    )
    out = encoder(x)
    assert out["z"].shape == (4, 8, 64)
    assert out["q"].shape == (4, 8)
    assert out["h"].shape == (4, 64)
    print("✅ test_molecular_program_encoder_grn_mlp passed")


def test_molecular_program_encoder_convnext1d():
    from mmpa_drp.models import MolecularProgramEncoder
    x = torch.randn(4, 1306)
    encoder = MolecularProgramEncoder(
        input_dim=1306, num_programs=8, program_dim=64,
        hidden_dims=[512, 256, 128, 64], drop=0.1,
        backbone_type="convnext1d",
        num_gene_tokens=64, token_dim=64,
        convnext_depth=2, convnext_kernel_size=7, convnext_expansion=4,
    )
    out = encoder(x)
    assert out["z"].shape == (4, 8, 64)
    assert out["q"].shape == (4, 8)
    assert out["h"].shape == (4, 64)
    print("✅ test_molecular_program_encoder_convnext1d passed")


def test_backward():
    from mmpa_drp.models import MolecularProgramEncoder
    for btype in ["mlp", "grn_mlp", "convnext1d"]:
        x = torch.randn(4, 1306, requires_grad=True)
        encoder = MolecularProgramEncoder(
            input_dim=1306, num_programs=8, program_dim=64,
            hidden_dims=[512, 256, 128, 64], drop=0.1,
            backbone_type=btype,
            num_gene_tokens=64, token_dim=64,
            convnext_depth=1, convnext_kernel_size=7, convnext_expansion=4,
        )
        out = encoder(x)
        loss = out["h"].mean() + out["z"].mean()
        loss.backward()
        assert x.grad is not None, f"{btype}: input gradient missing"
        # Check backbone has gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.backbone.parameters())
        assert has_grad, f"{btype}: backbone parameters have no gradient"
        print(f"✅ test_backward passed for {btype}")


if __name__ == "__main__":
    test_grn1d_vector_identity()
    test_grn1d_token_identity()
    test_grn_mlp_backbone()
    test_convnext1d_block()
    test_convnext1d_backbone()
    test_molecular_program_encoder_mlp()
    test_molecular_program_encoder_grn_mlp()
    test_molecular_program_encoder_convnext1d()
    test_backward()
    print("\n🎉 All smoke tests passed!")
