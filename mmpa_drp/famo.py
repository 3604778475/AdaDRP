"""
Fast Adaptive Multitask Optimization (FAMO)
Minimal implementation adapted for MACB-DRP drug-wise multitask learning.

Reference: Liu et al., "FAMO: Fast Adaptive Multitask Optimization", NeurIPS 2023.
"""

import torch
import torch.nn.functional as F
from typing import Union, List


class FAMO:
    """
    Fast Adaptive Multitask Optimization.

    Replaces simple mean/sum of task losses with a learned dynamic weighting
    that balances task loss decrease rates.
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 0.01,
        w_lr: float = 0.025,
        max_norm: float = 1.0,
    ):
        self.n_tasks = n_tasks
        self.device = device
        self.max_norm = max_norm

        # Minimum observed loss per task (reference point)
        self.min_losses = torch.zeros(n_tasks, device=device)

        # Learnable task logits (weights before softmax)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)

        # Track previous loss for update step
        self.prev_loss = None

    def set_min_losses(self, losses: Union[torch.Tensor, List[float]]):
        """Set the reference minimum losses for each task."""
        if not isinstance(losses, torch.Tensor):
            losses = torch.tensor(losses, device=self.device, dtype=torch.float32)
        self.min_losses = losses.to(self.device)

    def get_weighted_loss(self, losses: torch.Tensor):
        """
        Compute FAMO weighted loss from per-task losses.

        Args:
            losses: tensor of shape [n_tasks]

        Returns:
            weighted_loss: scalar tensor
            info: dict with weights
        """
        self.prev_loss = losses.detach().clone()
        z = F.softmax(self.w, dim=-1)  # task weights [n_tasks]
        D = losses - self.min_losses
        D = torch.clamp(D, min=1e-8)
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss, {"weights": z.detach().cpu()}

    def update(self, curr_loss: torch.Tensor):
        """
        Update task weight logits after optimizer step.

        Args:
            curr_loss: new per-task losses after model update [n_tasks]
        """
        if self.prev_loss is None:
            return

        prev_D = torch.clamp(self.prev_loss - self.min_losses, min=1e-8)
        curr_D = torch.clamp(curr_loss - self.min_losses, min=1e-8)
        delta = prev_D.log() - curr_D.log()

        with torch.enable_grad():
            d = torch.autograd.grad(
                F.softmax(self.w, dim=-1),
                self.w,
                grad_outputs=delta.detach(),
            )[0]

        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.Parameter], torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted loss and backpropagate.

        Args:
            losses: per-task losses [n_tasks]
            shared_parameters: parameters to clip gradients on

        Returns:
            weighted_loss: scalar tensor
        """
        loss, _ = self.get_weighted_loss(losses)
        loss.backward()

        if self.max_norm > 0 and shared_parameters is not None:
            if isinstance(shared_parameters, torch.Tensor):
                torch.nn.utils.clip_grad_norm_([shared_parameters], self.max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        return loss
