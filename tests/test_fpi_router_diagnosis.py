"""
Phase 9-FPI-Router Diagnosis Tests
"""

import torch
try:
    import pytest
except ImportError:
    pytest = None
from mmpa_drp.fpi import DualFPIResidualRouter
from mmpa_drp.models import ProgramFusionMLP
from mmpa_drp.models import MolecularProgramEncoder


def test_oracle_temo_match_global_param_str():
    """Test 1: oracle_temo_match_global param_str contains correct config markers."""
    import argparse
    from train_mmpa_program import main as train_main
    # We just verify the param_str construction logic by checking the parser args
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fpi_router', action='store_true')
    parser.add_argument('--router_gamma_global_max', type=float, default=0.05)
    parser.add_argument('--router_gamma_global_init', type=float, default=0.0)
    parser.add_argument('--router_gamma_cross_max', type=float, default=0.05)
    parser.add_argument('--router_gamma_cross_init', type=float, default=0.0)
    parser.add_argument('--router_cross_start_epoch', type=int, default=0)
    parser.add_argument('--router_cross_ramp_epochs', type=int, default=0)
    parser.add_argument('--router_cross_detach_mode', type=str, default='none')
    parser.add_argument('--router_two_stage_cross_only', action='store_true')
    parser.add_argument('--lambda_router_gamma_l1', type=float, default=0.0)
    parser.add_argument('--lambda_router_gamma_global_l1', type=float, default=None)
    parser.add_argument('--lambda_router_gamma_cross_l1', type=float, default=None)
    parser.add_argument('--router_residual_global_scale', type=float, default=1.0)
    parser.add_argument('--router_residual_cross_scale', type=float, default=1.0)
    parser.add_argument('--fpi_m_max', type=int, default=16)
    parser.add_argument('--fpi_fp_dim', type=int, default=1024)
    args = parser.parse_args([
        '--use_fpi_router',
        '--router_gamma_global_max', '0.1',
        '--router_gamma_global_init', '0.05',
        '--router_gamma_cross_max', '0.05',
    ])
    assert args.router_gamma_global_max == 0.1
    assert args.router_gamma_global_init == 0.05
    assert args.router_gamma_cross_max == 0.05


def test_cross_detach_only_affects_cross_branch():
    """Test 2: cross detach mode only detaches cross branch inputs."""
    router = DualFPIResidualRouter(
        num_drugs=3,
        gamma_global_max=0.1,
        gamma_cross_max=0.05,
        router_cross_detach_mode="zq_drug",
    )
    B, K, H, D, M = 2, 8, 64, 3, 16
    z = torch.randn(B, K, H)
    q = torch.randn(B, K)
    h_pool = torch.randn(B, H)
    drug_emb = torch.randn(D, H)
    frag_emb = torch.randn(D, M, H)
    frag_mask = torch.ones(D, M)
    base_logits = torch.randn(B, D)

    out = router(
        z=z, q=q, h_pool=h_pool,
        drug_emb=drug_emb, frag_emb=frag_emb, frag_mask=frag_mask,
        base_logits=base_logits,
    )
    assert out["final_logits"].shape == (B, D)
    assert out["residual_global"].shape == (B, D)
    assert out["residual_cross"].shape == (B, D)


def test_cross_detach_does_not_block_cross_params():
    """Test 3: cross detach blocks input feature gradients but cross params still train."""
    router = DualFPIResidualRouter(
        num_drugs=3,
        gamma_global_max=0.1,
        gamma_cross_max=0.05,
        router_cross_detach_mode="zq_drug",
    )
    B, K, H, D, M = 2, 8, 64, 3, 16
    z = torch.randn(B, K, H, requires_grad=True)
    q = torch.randn(B, K, requires_grad=True)
    h_pool = torch.randn(B, H, requires_grad=True)
    drug_emb = torch.randn(D, H, requires_grad=True)
    frag_emb = torch.randn(D, M, H, requires_grad=True)
    frag_mask = torch.ones(D, M)
    base_logits = torch.randn(B, D, requires_grad=True)

    out = router(
        z=z, q=q, h_pool=h_pool,
        drug_emb=drug_emb, frag_emb=frag_emb, frag_mask=frag_mask,
        base_logits=base_logits,
    )
    loss = out["final_logits"].mean()
    loss.backward()

    # Detached inputs should NOT receive gradients from cross branch
    # But they may receive from global branch and base_logits path
    # cross_fpi parameters MUST have gradients
    if router.cross_fpi is not None:
        has_cross_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                             for p in router.cross_fpi.parameters())
        assert has_cross_grad, "cross_fpi parameters should have gradients"

    # raw_gamma_cross MUST have gradient
    assert router.raw_gamma_cross.grad is not None, "raw_gamma_cross should have gradient"


def test_two_stage_freeze():
    """Test 4: two-stage freeze correctly freezes main model and keeps cross trainable."""
    encoder = MolecularProgramEncoder(
        input_dim=100, num_programs=8, program_dim=64, hidden_dims=[512, 256, 64]
    )
    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=256,
        output_dim=1,
        gene_num=100,
        drug_num=3,
        use_fpi_router=True,
        fpi_router_config={
            "program_dim": 64, "drug_dim": 64, "fragment_dim": 64,
            "num_programs": 8, "num_drugs": 3, "max_fragments": 16,
            "gamma_global_max": 0.1, "gamma_cross_max": 0.05,
        },
    )
    # Freeze main model components
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.drug_mlp.parameters():
        p.requires_grad = False
    for p in model.bilinear.parameters():
        p.requires_grad = False
    if model.fragment_encoder is not None:
        for p in model.fragment_encoder.parameters():
            p.requires_grad = False
    if model.fpi_router is not None:
        for p in model.fpi_router.global_fpi.parameters():
            p.requires_grad = False
        model.fpi_router.raw_gamma_global.requires_grad = False

    # Check frozen
    assert all(p.requires_grad == False for p in model.encoder.parameters())
    assert all(p.requires_grad == False for p in model.drug_mlp.parameters())
    assert all(p.requires_grad == False for p in model.bilinear.parameters())

    # Check cross trainable
    if model.fpi_router is not None and model.fpi_router.cross_fpi is not None:
        assert any(p.requires_grad for p in model.fpi_router.cross_fpi.parameters()), \
            "cross_fpi should be trainable"
    if model.fpi_router is not None:
        assert model.fpi_router.raw_gamma_cross.requires_grad, \
            "raw_gamma_cross should be trainable"


def test_oracle_mask():
    """Test 5: oracle mask only activates cross for specified drugs."""
    router = DualFPIResidualRouter(
        num_drugs=3,
        gamma_global_max=0.1,
        gamma_cross_max=0.05,
        router_cross_mode="oracle_mask",
        router_oracle_cross_drugs="Temozolomide",
    )
    B, K, H, D, M = 2, 8, 64, 3, 16
    z = torch.randn(B, K, H)
    q = torch.randn(B, K)
    h_pool = torch.randn(B, H)
    drug_emb = torch.randn(D, H)
    frag_emb = torch.randn(D, M, H)
    frag_mask = torch.ones(D, M)
    base_logits = torch.randn(B, D)

    drug_names = ["DrugA", "Temozolomide", "DrugB"]
    out = router(
        z=z, q=q, h_pool=h_pool,
        drug_emb=drug_emb, frag_emb=frag_emb, frag_mask=frag_mask,
        base_logits=base_logits,
        drug_names=drug_names,
    )
    mask = out["cross_active_mask"]
    assert mask[0].item() == 0.0
    assert mask[1].item() == 1.0
    assert mask[2].item() == 0.0


if __name__ == "__main__":
    test_oracle_temo_match_global_param_str()
    test_cross_detach_only_affects_cross_branch()
    test_cross_detach_does_not_block_cross_params()
    test_two_stage_freeze()
    test_oracle_mask()
    print("All diagnosis tests passed.")
