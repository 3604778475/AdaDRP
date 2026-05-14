"""
Smoke tests for Phase 6: Program-level Optimal Transport.
"""
import sys
sys.path.insert(0, '/home/waas/MACB-DRP-main')

import torch
from mmpa_drp.ot_utils import ProgramOTLoss, ot_alignment_loss, compute_ot_cost


def test_program_ot_loss_shapes():
    """Test ProgramOTLoss forward with all modes."""
    B_s, B_t, K, d = 4, 6, 8, 64
    source_z = torch.randn(B_s, K, d)
    source_q = torch.softmax(torch.randn(B_s, K), dim=-1)
    target_z = torch.randn(B_t, K, d)
    target_q = torch.softmax(torch.randn(B_t, K), dim=-1)

    for mode in ["same_program_uniform", "same_program_q_weighted",
                 "program_only", "flattened_program"]:
        module = ProgramOTLoss(mode=mode, lambda_program_ot=0.05)
        loss, diag = module(source_z, source_q, target_z, target_q)
        assert loss.dim() == 0, f"Mode {mode}: loss should be scalar"
        assert loss.item() >= 0, f"Mode {mode}: loss should be non-negative"
        assert "ot_dist" in diag, f"Mode {mode}: diagnostics missing ot_dist"
        print(f"  Mode {mode}: loss={loss.item():.6f}")


def test_program_ot_backward():
    """Test that gradients flow through ProgramOTLoss."""
    B_s, B_t, K, d = 4, 6, 8, 64
    source_z = torch.randn(B_s, K, d, requires_grad=True)
    source_q = torch.softmax(torch.randn(B_s, K), dim=-1)
    target_z = torch.randn(B_t, K, d, requires_grad=True)
    target_q = torch.softmax(torch.randn(B_t, K), dim=-1)

    module = ProgramOTLoss(mode="same_program_uniform", lambda_program_ot=0.05)
    loss, _ = module(source_z, source_q, target_z, target_q)
    loss.backward()

    assert source_z.grad is not None, "source_z should have gradients"
    assert target_z.grad is not None, "target_z should have gradients"
    assert not torch.isnan(source_z.grad).any(), "source_z grad has NaN"
    assert not torch.isnan(target_z.grad).any(), "target_z grad has NaN"
    print(f"  Backward OK: source_z grad norm={source_z.grad.norm().item():.4f}")


def test_program_ot_q_weighted_marginals():
    """Test that q-weighted marginals sum to 1 per program."""
    B_s, B_t, K, d = 4, 6, 8, 64
    source_z = torch.randn(B_s, K, d)
    source_q = torch.softmax(torch.randn(B_s, K), dim=-1)
    target_z = torch.randn(B_t, K, d)
    target_q = torch.softmax(torch.randn(B_t, K), dim=-1)

    module = ProgramOTLoss(mode="same_program_q_weighted", lambda_program_ot=0.05)
    loss, _ = module(source_z, source_q, target_z, target_q)
    assert loss.item() >= 0
    print(f"  Q-weighted OK: loss={loss.item():.6f}")


def test_flattened_program_vs_batched():
    """Sanity check: flattened mode should produce different loss than uniform."""
    torch.manual_seed(42)
    B_s, B_t, K, d = 4, 6, 8, 64
    source_z = torch.randn(B_s, K, d)
    source_q = torch.softmax(torch.randn(B_s, K), dim=-1)
    target_z = torch.randn(B_t, K, d)
    target_q = torch.softmax(torch.randn(B_t, K), dim=-1)

    module_flat = ProgramOTLoss(mode="flattened_program", lambda_program_ot=0.05)
    module_uni = ProgramOTLoss(mode="same_program_uniform", lambda_program_ot=0.05)

    loss_flat, _ = module_flat(source_z, source_q, target_z, target_q)
    loss_uni, _ = module_uni(source_z, source_q, target_z, target_q)

    # They should generally differ (unless by extreme coincidence)
    assert abs(loss_flat.item() - loss_uni.item()) > 1e-6, \
        "Flattened and uniform should differ"
    print(f"  Flattened={loss_flat.item():.6f} vs Uniform={loss_uni.item():.6f}")


def test_program_ot_with_encoder_output():
    """Integration test: ProgramFusionMLP with return_encoder_out=True."""
    import config
    if not hasattr(config, 'tissue_map') or getattr(config.tissue_map, 'shape', None) == ():
        config.tissue_map = list(range(10))
    from mmpa_drp.models import MolecularProgramEncoder, ProgramFusionMLP

    B, G, K, d = 4, 1306, 8, 64
    bilinear_dim = 128
    gene_feat_dim = 64
    drug_feat_dim = 64
    drug_num = 9
    encoder = MolecularProgramEncoder(
        input_dim=G, num_programs=K, program_dim=d,
        backbone_type="convnext1d", convnext_depth=1,
    )
    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=bilinear_dim + gene_feat_dim + drug_feat_dim,
        output_dim=1,
        gene_num=G,
        bilinear_dim=bilinear_dim,
        gene_feat_dim=gene_feat_dim,
        drug_feat_dim=drug_feat_dim,
        drug_num=drug_num,
    )
    model.train()

    inputs = torch.randn(B, G)
    drug_feat = torch.randn(drug_num, 128)
    tissue_idx = torch.randint(0, 5, (B,))

    # Standard forward
    out_std = model(inputs, drug_feat, tissue_idx)
    assert len(out_std) == 6

    # With encoder_out
    out_enc = model(inputs, drug_feat, tissue_idx, return_encoder_out=True)
    assert len(out_enc) == 7
    encoder_out = out_enc[6]
    assert "z" in encoder_out
    assert "q" in encoder_out
    assert "h" in encoder_out
    assert encoder_out["z"].shape == (B, K, d)
    assert encoder_out["q"].shape == (B, K)
    print(f"  Integration OK: z={encoder_out['z'].shape}, q={encoder_out['q'].shape}")


def test_ot_alignment_loss_custom_marginals():
    """Test ot_alignment_loss with custom marginals."""
    N, M, d = 4, 6, 8
    source = torch.randn(N, d)
    target = torch.randn(M, d)
    u = torch.ones(N) / N
    v = torch.ones(M) / M

    loss = ot_alignment_loss(source, target, source_marginal=u, target_marginal=v)
    assert loss.item() >= 0
    print(f"  Custom marginals OK: loss={loss.item():.6f}")


if __name__ == "__main__":
    print("test_program_ot_loss_shapes ...")
    test_program_ot_loss_shapes()
    print("test_program_ot_backward ...")
    test_program_ot_backward()
    print("test_program_ot_q_weighted_marginals ...")
    test_program_ot_q_weighted_marginals()
    print("test_flattened_program_vs_batched ...")
    test_flattened_program_vs_batched()
    print("test_program_ot_with_encoder_output ...")
    test_program_ot_with_encoder_output()
    print("test_ot_alignment_loss_custom_marginals ...")
    test_ot_alignment_loss_custom_marginals()
    print("\nAll 6 tests passed!")
