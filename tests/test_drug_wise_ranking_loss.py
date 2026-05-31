"""
Phase 14-A: Drug-wise Pairwise Ranking Loss Tests
==================================================
Test 1-6: drug_wise_pairwise_ranking_loss
"""

import pytest
import torch

from mmpa_drp.losses import drug_wise_pairwise_ranking_loss


class TestDrugWisePairwiseRankingLoss:
    def test_basic_shape_and_finite(self):
        B, D = 8, 9
        logits = torch.randn(B, D)
        labels = torch.randint(0, 2, (B, D)).float()

        loss = drug_wise_pairwise_ranking_loss(logits, labels)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_perfect_ranking_lower_loss(self):
        B, D = 8, 9
        # Good: positive logits > negative logits
        logits_good = torch.randn(B, D)
        labels = torch.randint(0, 2, (B, D)).float()
        # Make positives high and negatives low
        logits_good = torch.where(labels > 0.5, logits_good + 2.0, logits_good - 2.0)

        # Bad: positive logits < negative logits
        logits_bad = torch.randn(B, D)
        logits_bad = torch.where(labels > 0.5, logits_bad - 2.0, logits_bad + 2.0)

        loss_good = drug_wise_pairwise_ranking_loss(logits_good, labels)
        loss_bad = drug_wise_pairwise_ranking_loss(logits_bad, labels)

        assert loss_good < loss_bad

    def test_skip_drug_with_no_positive(self):
        B, D = 8, 9
        logits = torch.randn(B, D)
        labels = torch.randint(0, 2, (B, D)).float()
        # Drug 0: all negative
        labels[:, 0] = 0.0

        out = drug_wise_pairwise_ranking_loss(logits, labels, return_details=True)

        assert out["num_valid_drugs"] <= D - 1
        assert out["valid_drug_mask"][0] == False
        assert torch.isfinite(out["loss"])

    def test_skip_drug_with_no_negative(self):
        B, D = 8, 9
        logits = torch.randn(B, D)
        labels = torch.randint(0, 2, (B, D)).float()
        # Drug 0: all positive
        labels[:, 0] = 1.0

        out = drug_wise_pairwise_ranking_loss(logits, labels, return_details=True)

        assert out["num_valid_drugs"] <= D - 1
        assert out["valid_drug_mask"][0] == False
        assert torch.isfinite(out["loss"])

    def test_mask_works(self):
        B, D = 8, 9
        logits = torch.randn(B, D)
        labels = torch.ones(B, D)
        # Mask out half the labels for drug 0
        mask = torch.ones(B, D, dtype=torch.bool)
        mask[:4, 0] = False
        labels[:4, 0] = 0.0  # These are masked out, should not count

        out = drug_wise_pairwise_ranking_loss(logits, labels, mask=mask, return_details=True)

        # Drug 0 should have 4 positives and 0 negatives (masked negatives don't count)
        # So drug 0 should be skipped
        assert out["valid_drug_mask"][0] == False

    def test_max_pairs_per_drug_works(self):
        B, D = 8, 9
        logits = torch.randn(B, D)
        labels = torch.randint(0, 2, (B, D)).float()

        out = drug_wise_pairwise_ranking_loss(
            logits, labels, max_pairs_per_drug=3, return_details=True
        )

        # Each valid drug should have at most 3 pairs
        for d in range(D):
            if out["valid_drug_mask"][d]:
                assert out["num_pairs_per_drug"][d] <= 3

    def test_margin_increases_loss(self):
        B, D = 8, 9
        logits = torch.randn(B, D)
        labels = torch.randint(0, 2, (B, D)).float()

        loss_no_margin = drug_wise_pairwise_ranking_loss(logits, labels, margin=0.0)
        loss_with_margin = drug_wise_pairwise_ranking_loss(logits, labels, margin=1.0)

        assert loss_with_margin >= loss_no_margin
