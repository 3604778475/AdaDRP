#!/usr/bin/env python3
"""
Diagnostic script for DualFPIResidualRouter.
Checks actual gamma values and residual scales during forward pass.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

import config
if not hasattr(config, 'tissue_map') or getattr(config.tissue_map, 'shape', None) == ():
    config.tissue_map = {"Tissue_0": 0, "Tissue_1": 1, "Tissue_2": 2}
    config.drug_feat = torch.randn(9, 256)

from mmpa_drp.models import ProgramFusionMLP
from mmpa_drp.fpi import DualFPIResidualRouter


def _make_dummy_encoder(input_dim, program_dim, num_programs):
    class DummyEncoder(nn.Module):
        def __init__(self, input_dim, program_dim, num_programs):
            super().__init__()
            self.fc = nn.Linear(input_dim, program_dim)
            self.num_programs = num_programs
        def forward(self, x):
            B = x.size(0)
            h = self.fc(x)
            z = h.unsqueeze(1).expand(B, self.num_programs, -1)
            q = torch.softmax(torch.randn(B, self.num_programs, device=x.device), dim=-1)
            return {"h": h, "z": z, "q": q}
    return DummyEncoder(input_dim, program_dim, num_programs)


def test_router_scales():
    B, K, H, D, M = 4, 8, 64, 9, 16
    gene_num = 100
    input_dim = 256
    output_dim = 1
    drug_feat = torch.randn(D, 128)
    frag_fp = torch.randn(D, M, 1024)
    frag_mask = torch.ones(D, M)
    encoder = _make_dummy_encoder(gene_num, H, K)

    model = ProgramFusionMLP(
        encoder=encoder,
        input_dim=input_dim,
        output_dim=output_dim,
        gene_num=gene_num,
        drug_num=D,
        num_programs=K,
        program_dim=H,
        bilinear_dim=128,
        drug_feat=drug_feat,
        return_dict=True,
        use_fpi_router=True,
        fpi_router_config={
            "program_dim": H,
            "drug_dim": 64,
            "fragment_dim": H,
            "num_programs": K,
            "num_drugs": D,
            "max_fragments": M,
            "fp_dim": 1024,
            "gamma_global_max": 0.05,
            "gamma_global_init": 0.05,
            "gamma_cross_max": 0.05,
            "gamma_cross_init": 0.0,
        },
        frag_fp=frag_fp,
        frag_mask=frag_mask,
    )

    inputs = torch.randn(B, gene_num)
    tissue_idx = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        out = model(
            inputs, drug_feat, tissue_idx,
            return_dict=True,
            current_epoch=0,
            drug_names=["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel", "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"],
        )

    router_out = out["router_out"]
    print("=== Router Diagnostics ===")
    print(f"gamma_global:      mean={router_out['gamma_global'].mean():.6f}, max={router_out['gamma_global'].max():.6f}")
    print(f"gamma_cross:       mean={router_out['gamma_cross'].mean():.6f}, max={router_out['gamma_cross'].max():.6f}")
    print(f"effective_gamma_cross: mean={router_out['effective_gamma_cross'].mean():.6f}, max={router_out['effective_gamma_cross'].max():.6f}")
    print(f"residual_global:   mean_abs={router_out['residual_global'].abs().mean():.6f}, max_abs={router_out['residual_global'].abs().max():.6f}")
    print(f"residual_cross:    mean_abs={router_out['residual_cross'].abs().mean():.6f}, max_abs={router_out['residual_cross'].abs().max():.6f}")
    print(f"base_logits:       mean_abs={router_out.get('base_logits', torch.tensor(0.0)).abs().mean():.6f}")
    print(f"final_logits:      mean_abs={router_out['final_logits'].abs().mean():.6f}")
    print(f"schedule_factor:   {router_out['schedule_factor'].item():.4f}")
    print(f"cross_active_mask: {router_out['cross_active_mask'].tolist()}")
    print(f"branch_preference: {router_out['branch_preference']}")

    # Check global_fpi internal gamma
    internal_gamma_g = model.fpi_router.global_fpi.get_gamma()
    internal_gamma_c = model.fpi_router.cross_fpi.get_gamma()
    print(f"\n=== Internal FPI Gammas ===")
    print(f"global_fpi internal gamma: mean={internal_gamma_g.mean():.6f}, max={internal_gamma_g.max():.6f}")
    print(f"cross_fpi internal gamma:  mean={internal_gamma_c.mean():.6f}, max={internal_gamma_c.max():.6f}")

    # Compare with standalone FPI
    from mmpa_drp.fpi import FragmentProgramInteraction
    standalone_fpi = FragmentProgramInteraction(
        program_dim=H, drug_dim=64, fragment_dim=H,
        num_programs=K, num_drugs=D, max_fragments=M,
        gamma_max=0.1, gamma_init=0.05, use_fpi_sim=False,
    )
    z = encoder(inputs)["z"]
    q = encoder(inputs)["q"]
    h_pool = encoder(inputs)["h"]
    frag_emb = model.fragment_encoder(frag_fp, frag_mask)
    with torch.no_grad():
        standalone_out = standalone_fpi(z, q, h_pool, drug_feat_mlp=drug_feat, frag_emb=frag_emb, frag_mask=frag_mask)
    print(f"\n=== Standalone FPI (FPI2_warmup config) ===")
    print(f"gamma:             mean={standalone_out['gamma'].mean():.6f}, max={standalone_out['gamma'].max():.6f}")
    print(f"residual_logits:   mean_abs={standalone_out['residual_logits'].abs().mean():.6f}, max_abs={standalone_out['residual_logits'].abs().max():.6f}")

    # Compare residuals directly
    print(f"\n=== Residual Comparison ===")
    print(f"Router global residual mean_abs: {router_out['residual_global'].abs().mean():.6f}")
    print(f"Standalone residual mean_abs:    {standalone_out['residual_logits'].abs().mean():.6f}")
    diff = (router_out['residual_global'] - standalone_out['residual_logits']).abs().mean()
    print(f"Mean abs diff: {diff:.6f}")


if __name__ == "__main__":
    test_router_scales()
