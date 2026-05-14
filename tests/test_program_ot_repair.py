"""Smoke tests for Phase 6-R."""
import sys
sys.path.insert(0, "/home/waas/MACB-DRP-main")
import torch
import torch.nn.functional as F
from mmpa_drp.ot_utils import ProgramOTLoss, q_align_loss
from mmpa_drp.grad_diagnostics import compute_grad_norm

def test_q_mass():
    source_z = torch.randn(4, 8, 64)
    source_q = F.softmax(torch.randn(4, 8), dim=-1)
    target_z = torch.randn(5, 8, 64)
    target_q = F.softmax(torch.randn(5, 8), dim=-1)
    module = ProgramOTLoss(mode="q_mass_flattened", lambda_program_ot=0.05)
    loss, diag = module(source_z, source_q, target_z, target_q)
    assert loss.dim() == 0 and loss.item() >= 0
    print("q_mass OK")

def test_q_align():
    source_q = F.softmax(torch.randn(4, 8), dim=-1)
    target_q = F.softmax(torch.randn(5, 8), dim=-1)
    loss, diag = q_align_loss(source_q, target_q)
    assert loss.item() >= 0
    print("q_align OK")

def test_grad():
    import torch.nn as nn
    model = nn.Linear(10, 1)
    loss = ((model(torch.randn(4, 10)) - torch.randn(4, 1)) ** 2).mean()
    norm, n_with, n_without = compute_grad_norm(loss, list(model.parameters()), retain_graph=True)
    assert norm > 0
    print("grad OK")

if __name__ == "__main__":
    test_q_mass()
    test_q_align()
    test_grad()
    print("All passed!")
