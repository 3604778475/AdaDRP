"""
Phase 14-A Debug: Rank Loss Sanity Tests
=========================================
Test 1-6: drug_wise_pairwise_ranking_loss sanity checks.
"""

import pytest
import torch

from mmpa_drp.losses import drug_wise_pairwise_ranking_loss


class TestRankLossSanity:
    def test_mean_scale_reasonable(self):
        """Loss should be in reasonable range, not explode with more pairs."""
        B, D = 16, 9
        logits = torch.randn(B, D)
        labels = torch.randint(0, 2, (B, D)).float()

        loss = drug_wise_pairwise_ranking_loss(logits, labels)
        assert torch.isfinite(loss)
        assert 0 <= loss.item() <= 10  # softplus(-diff) with diff ~ N(0,2) -> ~0.7

    def test_sum_vs_mean_protection(self):
        """With many pairs, loss should not scale linearly."""
        B, D = 32, 9
        logits = torch.randn(B, D)
        labels = torch.ones(B, D)
        labels[:B//2, :] = 0.0  # Half positive, half negative

        loss_unlimited = drug_wise_pairwise_ranking_loss(
            logits, labels, max_pairs_per_drug=0
        )
        loss_limited = drug_wise_pairwise_ranking_loss(
            logits, labels, max_pairs_per_drug=50
        )
        # Both should be finite and in similar range
        assert torch.isfinite(loss_unlimited)
        assert torch.isfinite(loss_limited)
        assert abs(loss_unlimited.item() - loss_limited.item()) < 2.0

    def test_perfect_ranking_lower_than_reversed(self):
        """pos > neg should give lower loss than pos < neg."""
        B, D = 8, 3
        labels = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0],
                               [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]]).float()
        # Good: positive logits high, negative logits low
        logits_good = labels * 2.0 + (1 - labels) * (-2.0)
        # Bad: positive logits low, negative logits high
        logits_bad = labels * (-2.0) + (1 - labels) * 2.0

        loss_good = drug_wise_pairwise_ranking_loss(logits_good, labels)
        loss_bad = drug_wise_pairwise_ranking_loss(logits_bad, labels)

        assert loss_good < loss_bad

    def test_label_direction_flip_changes_loss(self):
        """Flipping labels should significantly change loss."""
        B, D = 8, 3
        logits = torch.randn(B, D)
        labels = torch.randint(0, 2, (B, D)).float()

        loss1 = drug_wise_pairwise_ranking_loss(logits, labels)
        loss2 = drug_wise_pairwise_ranking_loss(logits, 1 - labels)

        assert abs(loss1.item() - loss2.item()) > 1e-3

    def test_skip_invalid_drug(self):
        """Drug with all-positive or all-negative should be skipped."""
        B, D = 8, 3
        logits = torch.randn(B, D)
        labels = torch.randint(0, 2, (B, D)).float()
        labels[:, 0] = 1.0  # All positive
        labels[:, 1] = 0.0  # All negative

        out = drug_wise_pairwise_ranking_loss(logits, labels, return_details=True)

        assert out["num_valid_drugs"] <= 1
        assert out["valid_drug_mask"][0] == False
        assert out["valid_drug_mask"][1] == False
        assert torch.isfinite(out["loss"])

    def test_return_details_keys(self):
        """All expected diagnostics keys should be present."""
        B, D = 8, 3
        logits = torch.randn(B, D)
        labels = torch.randint(0, 2, (B, D)).float()

        out = drug_wise_pairwise_ranking_loss(logits, labels, return_details=True)

        expected_keys = [
            "loss", "num_valid_drugs", "num_pairs_total",
            "loss_per_drug", "num_pos_per_drug", "num_neg_per_drug",
            "num_pairs_per_drug", "valid_drug_mask",
            "mean_pos_logit_per_drug", "mean_neg_logit_per_drug",
            "mean_pair_diff_per_drug",
            "mean_pos_logit", "mean_neg_logit", "mean_pair_diff",
        ]
        for k in expected_keys:
            assert k in out, f"Missing key: {k}"
