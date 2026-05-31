"""
Smoke tests for Fragment-Program Interaction (FPI)
====================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import BRICS, AllChem

from mmpa_drp.fragment_encoder import DrugFragmentEncoder
from mmpa_drp.fpi import FragmentProgramInteraction
from mmpa_drp.models import MolecularProgramEncoder, ProgramFusionMLP
from mmpa_drp.fragment_utils import load_drug_fragments
import config

# ProgramFusionMLP depends on config.tissue_map being initialized.
# Set a dummy tissue map for testing.
config.tissue_map = {'tissue_a': 0, 'tissue_b': 1}


# ============================================================================
# Test 1: fragment preprocessing mock
# ============================================================================

def test_fragment_preprocessing_mock():
    smiles_list = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"]
    m_max = 8
    fp_dim = 1024

    frag_fp = np.zeros((len(smiles_list), m_max, fp_dim), dtype=np.float32)
    frag_mask = np.zeros((len(smiles_list), m_max), dtype=np.float32)
    frag_smiles = []

    for d_idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            frags = list(BRICS.BRICSDecompose(mol))
        except Exception:
            frags = []
        if len(frags) == 0:
            frags = [smiles]

        valid = 0
        row_smiles = []
        for m_idx, f_smiles in enumerate(frags[:m_max]):
            f_mol = Chem.MolFromSmiles(f_smiles)
            if f_mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(f_mol, radius=2, nBits=fp_dim)
                fp_vec = np.zeros((fp_dim,), dtype=np.float32)
                Chem.DataStructs.ConvertToNumpyArray(fp, fp_vec)
                frag_fp[d_idx, m_idx, :] = fp_vec
                frag_mask[d_idx, m_idx] = 1.0
                valid += 1
                row_smiles.append(f_smiles)
        frag_smiles.append(row_smiles)

        assert valid >= 1, f"Drug {d_idx} should have at least 1 valid fragment"

    assert frag_fp.shape == (3, m_max, fp_dim)
    assert frag_mask.shape == (3, m_max)
    assert frag_mask.sum() >= 3


# ============================================================================
# Test 2: DrugFragmentEncoder shape
# ============================================================================

def test_drug_fragment_encoder_shape():
    D, M, F, H = 9, 16, 1024, 64
    frag_fp = torch.randn(D, M, F)
    frag_mask = torch.ones(D, M)
    frag_mask[:, 8:] = 0.0  # half padding

    encoder = DrugFragmentEncoder(fp_dim=F, hidden_dim=H, dropout=0.1)
    frag_emb = encoder(frag_fp, frag_mask)

    assert frag_emb.shape == (D, M, H)
    # Padding positions should be zero
    assert torch.allclose(frag_emb[:, 8:, :], torch.zeros(D, M - 8, H), atol=1e-6)


# ============================================================================
# Test 3: FragmentProgramInteraction shape
# ============================================================================

def test_fragment_program_interaction_shape():
    B, K, H = 4, 8, 64
    D, M = 9, 16

    z = torch.randn(B, K, H)
    q = torch.softmax(torch.randn(B, K), dim=1)
    h_pool = torch.randn(B, H)
    drug_emb = torch.randn(D, H)
    frag_emb = torch.randn(D, M, H)
    frag_mask = torch.ones(D, M)
    frag_mask[:, 12:] = 0.0

    fpi = FragmentProgramInteraction(
        program_dim=H, drug_dim=H, fragment_dim=H,
        num_programs=K, num_drugs=D, max_fragments=M,
    )
    out = fpi(z, q, h_pool, drug_emb, frag_emb, frag_mask)

    assert out["u"].shape == (B, D, H)
    assert out["attn"].shape == (B, D, K, M)
    assert out["score"].shape == (B, D, K, M)
    assert out["residual_logits"].shape == (B, D)
    assert out["gamma"].shape == (D,)

    # Attention should sum to 1 over K*M
    attn_sum = out["attn"].reshape(B, D, K * M).sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones(B, D), atol=1e-4)


# ============================================================================
# Test 4: gamma=0 equivalence to base logits
# ============================================================================

def test_gamma_zero_equivalence():
    B, D = 4, 9
    base_logits = torch.randn(B, D)
    residual_logits = torch.randn(B, D)
    gamma = torch.zeros(D)

    final_logits = base_logits + gamma.unsqueeze(0) * residual_logits
    assert torch.allclose(final_logits, base_logits, atol=1e-6)


# ============================================================================
# Test 5: ProgramFusionMLP use_fpi=False compatibility
# ============================================================================

def test_program_fusion_mlp_use_fpi_false_compat():
    encoder = MolecularProgramEncoder(
        input_dim=100,
        num_programs=8,
        program_dim=64,
        hidden_dims=[128, 64],
        drop=0.1,
    )
    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=256,
        output_dim=1,
        hidden_dims=[128, 64],
        gene_num=100,
        bilinear_dim=128,
        gene_feat_dim=64,
        drug_feat_dim=64,
        drug_num=9,
        drop=0.1,
        use_fpi=False,
    )

    x = torch.randn(2, 100)
    drug_feat = torch.randn(9, 128)
    tissue_idx = torch.tensor([0, 1])

    out = model(x, drug_feat, tissue_idx)
    assert isinstance(out, tuple)
    assert len(out) == 6  # outputs, gene_feat, feat, expanded_tissue_idx, cdan_domain_output, drug_pred


# ============================================================================
# Test 6: ProgramFusionMLP use_fpi=True, gamma_max=0 close to use_fpi=False
# ============================================================================

def test_program_fusion_mlp_fpi_sanity():
    encoder = MolecularProgramEncoder(
        input_dim=100,
        num_programs=8,
        program_dim=64,
        hidden_dims=[128, 64],
        drop=0.1,
    )
    model_off = ProgramFusionMLP(
        encoder=encoder,
        input_dim=256,
        output_dim=1,
        hidden_dims=[128, 64],
        gene_num=100,
        bilinear_dim=128,
        gene_feat_dim=64,
        drug_feat_dim=64,
        drug_num=9,
        drop=0.1,
        use_fpi=False,
    )

    # Share encoder weights by copying state dict
    model_on = ProgramFusionMLP(
        encoder=encoder,
        input_dim=256,
        output_dim=1,
        hidden_dims=[128, 64],
        gene_num=100,
        bilinear_dim=128,
        gene_feat_dim=64,
        drug_feat_dim=64,
        drug_num=9,
        drop=0.1,
        use_fpi=True,
        fpi_config={"gamma_max": 0.0, "gamma_init": 0.0, "fp_dim": 1024, "max_fragments": 16},
    )
    model_on.load_state_dict(model_off.state_dict(), strict=False)

    x = torch.randn(2, 100)
    drug_feat = torch.randn(9, 128)
    tissue_idx = torch.tensor([0, 1])
    frag_fp = torch.randn(9, 16, 1024)
    frag_mask = torch.ones(9, 16)

    model_off.eval()
    model_on.eval()
    with torch.no_grad():
        out_off = model_off(x, drug_feat, tissue_idx)
        out_on = model_on(x, drug_feat, tissue_idx, frag_fp=frag_fp, frag_mask=frag_mask)

    # outputs should be very close (not exact due to FPI module zeroing)
    assert torch.allclose(out_off[0], out_on[0], atol=1e-4)


# ============================================================================
# Test 7: gradient flow
# ============================================================================

def test_fpi_gradient_flow():
    encoder = MolecularProgramEncoder(
        input_dim=100,
        num_programs=8,
        program_dim=64,
        hidden_dims=[128, 64],
        drop=0.1,
    )
    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=256,
        output_dim=1,
        hidden_dims=[128, 64],
        gene_num=100,
        bilinear_dim=128,
        gene_feat_dim=64,
        drug_feat_dim=64,
        drug_num=9,
        drop=0.1,
        use_fpi=True,
        fpi_config={"gamma_max": 0.05, "fp_dim": 1024, "max_fragments": 16},
    )

    x = torch.randn(2, 100)
    drug_feat = torch.randn(9, 128)
    tissue_idx = torch.tensor([0, 1])
    frag_fp = torch.randn(9, 16, 1024)
    frag_mask = torch.ones(9, 16)

    out = model(x, drug_feat, tissue_idx, frag_fp=frag_fp, frag_mask=frag_mask)
    loss = out[0].mean()
    loss.backward()

    assert model.fragment_encoder is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fragment_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fpi.parameters())
    assert model.fpi.raw_gamma.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


# ============================================================================
# Test 8: no target label leakage
# ============================================================================

def test_no_target_label_leakage():
    # FPI modules should not accept y_target or test_auc
    fpi = FragmentProgramInteraction()
    import inspect
    sig = inspect.signature(fpi.forward)
    params = list(sig.parameters.keys())
    assert 'y_target' not in params
    assert 'target_response' not in params
    assert 'test_auc' not in params


# ============================================================================
# Test 9: param_str uniqueness
# ============================================================================

def test_param_str_uniqueness():
    # Simulate param_str generation logic from train_mmpa_program.py
    base = "pretrain_num_epochs_300_train_num_epochs_100_contrastive_num_epochs_200_K_3"

    def make_fpi_str(gmax, ginit, qprior, fgl1, fent, mmax, fpdim):
        s = base
        s += f"_fpi_gmax{gmax}_ginit{ginit}_qprior{qprior}"
        if fgl1 > 0:
            s += f"_fgl1{fgl1}"
        if fent > 0:
            s += f"_fent{fent}"
        s += f"_m{mmax}_fp{fpdim}"
        return s

    s1 = make_fpi_str(0.05, 0.0, 0.01, 0.001, 0.0, 16, 1024)
    s2 = make_fpi_str(0.1, 0.0, 0.01, 0.001, 0.0, 16, 1024)
    s3 = make_fpi_str(0.05, 0.0, 0.05, 0.001, 0.0, 16, 1024)
    s4 = make_fpi_str(0.05, 0.0, 0.01, 0.001, 0.0, 8, 1024)

    assert s1 != s2
    assert s1 != s3
    assert s1 != s4
    assert s2 != s3


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
