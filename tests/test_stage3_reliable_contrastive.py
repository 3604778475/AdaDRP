"""
Phase 13: Reliable Stage-3 Contrastive Anchoring (S3R) Smoke Tests
===================================================================
Test 1-6: build_reliable_contrastive_samples + reliable_contrastive_loss
"""

import pytest
import torch
import torch.nn.functional as F

from mmpa_drp.losses import build_reliable_contrastive_samples, reliable_contrastive_loss


class TestBuildReliableContrastiveSamples:
    def test_false_negative_suppression_same_drug(self):
        T, D, feat_dim, N, K = 10, 3, 8, 4, 2
        anchor_feat = torch.randn(N, feat_dim)
        all_target_feat = torch.randn(T, D, feat_dim)
        topk_idx = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        drug_ids = torch.tensor([0, 1, 2, 0])

        out = build_reliable_contrastive_samples(
            anchor_feat=anchor_feat,
            all_target_feat=all_target_feat,
            topk_idx=topk_idx,
            drug_ids=drug_ids,
            same_drug_negatives=True,
            suppress_false_negatives=True,
        )

        assert out["anchors"].shape[0] == N
        assert out["positives"].shape == (N, K, feat_dim)
        assert len(out["negatives"]) == N
        # Same drug negatives: T - K = 10 - 2 = 8 negatives per anchor
        for i in range(N):
            assert out["negatives"][i].shape[0] == T - K
        assert out["positive_weights"] is None
        assert out["valid_mask"].all()

    def test_false_negative_suppression_all_drug(self):
        T, D, feat_dim, N, K = 10, 3, 8, 4, 2
        anchor_feat = torch.randn(N, feat_dim)
        all_target_feat = torch.randn(T, D, feat_dim)
        topk_idx = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        drug_ids = torch.tensor([0, 1, 2, 0])

        out = build_reliable_contrastive_samples(
            anchor_feat=anchor_feat,
            all_target_feat=all_target_feat,
            topk_idx=topk_idx,
            drug_ids=drug_ids,
            same_drug_negatives=False,
            suppress_false_negatives=True,
        )

        assert out["anchors"].shape[0] == N
        # All drug negatives: T*D - K = 30 - 2 = 28 negatives per anchor
        for i in range(N):
            assert out["negatives"][i].shape[0] == T * D - K

    def test_no_suppression(self):
        T, D, feat_dim, N, K = 10, 3, 8, 4, 2
        anchor_feat = torch.randn(N, feat_dim)
        all_target_feat = torch.randn(T, D, feat_dim)
        topk_idx = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        drug_ids = torch.tensor([0, 1, 2, 0])

        out = build_reliable_contrastive_samples(
            anchor_feat=anchor_feat,
            all_target_feat=all_target_feat,
            topk_idx=topk_idx,
            drug_ids=drug_ids,
            same_drug_negatives=True,
            suppress_false_negatives=False,
        )

        # No suppression: all T same-drug features remain
        for i in range(N):
            assert out["negatives"][i].shape[0] == T

    def test_weighted_positives_with_topk_sim(self):
        T, D, feat_dim, N, K = 10, 3, 8, 4, 2
        anchor_feat = torch.randn(N, feat_dim)
        all_target_feat = torch.randn(T, D, feat_dim)
        topk_idx = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        drug_ids = torch.tensor([0, 1, 2, 0])
        topk_sim = torch.tensor([[0.9, 0.5], [0.8, 0.4], [0.7, 0.3], [0.6, 0.2]])

        out = build_reliable_contrastive_samples(
            anchor_feat=anchor_feat,
            all_target_feat=all_target_feat,
            topk_idx=topk_idx,
            drug_ids=drug_ids,
            topk_sim=topk_sim,
            weighted_positives=True,
            positive_temperature=0.1,
        )

        assert out["positive_weights"] is not None
        assert out["positive_weights"].shape == (N, K)
        # Each row sums to ~1
        for i in range(N):
            assert torch.allclose(out["positive_weights"][i].sum(), torch.tensor(1.0), atol=1e-4)
        # Higher similarity -> higher weight
        for i in range(N):
            assert out["positive_weights"][i][0] > out["positive_weights"][i][1]

    def test_rank_based_weighted_positives(self):
        T, D, feat_dim, N, K = 10, 3, 8, 4, 2
        anchor_feat = torch.randn(N, feat_dim)
        all_target_feat = torch.randn(T, D, feat_dim)
        topk_idx = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        drug_ids = torch.tensor([0, 1, 2, 0])

        out = build_reliable_contrastive_samples(
            anchor_feat=anchor_feat,
            all_target_feat=all_target_feat,
            topk_idx=topk_idx,
            drug_ids=drug_ids,
            topk_sim=None,
            weighted_positives=True,
            positive_temperature=0.1,
        )

        assert out["positive_weights"] is not None
        assert out["positive_weights"].shape == (N, K)
        # Top-1 should have higher weight than Top-K
        for i in range(N):
            assert out["positive_weights"][i][0] > out["positive_weights"][i][-1]

    def test_max_negatives_per_anchor(self):
        T, D, feat_dim, N, K = 100, 3, 8, 4, 2
        anchor_feat = torch.randn(N, feat_dim)
        all_target_feat = torch.randn(T, D, feat_dim)
        topk_idx = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        drug_ids = torch.tensor([0, 1, 2, 0])

        out = build_reliable_contrastive_samples(
            anchor_feat=anchor_feat,
            all_target_feat=all_target_feat,
            topk_idx=topk_idx,
            drug_ids=drug_ids,
            same_drug_negatives=True,
            suppress_false_negatives=True,
            max_negatives_per_anchor=5,
        )

        for i in range(N):
            assert out["negatives"][i].shape[0] == 5

    def test_all_positives_no_negatives_invalid(self):
        T, D, feat_dim, N, K = 2, 3, 8, 1, 2
        anchor_feat = torch.randn(N, feat_dim)
        all_target_feat = torch.randn(T, D, feat_dim)
        topk_idx = torch.tensor([[0, 1]])
        drug_ids = torch.tensor([0])

        out = build_reliable_contrastive_samples(
            anchor_feat=anchor_feat,
            all_target_feat=all_target_feat,
            topk_idx=topk_idx,
            drug_ids=drug_ids,
            same_drug_negatives=True,
            suppress_false_negatives=True,
        )

        # T=2, K=2, suppress=True -> no negatives left -> invalid
        assert not out["valid_mask"][0]
        assert out["anchors"].shape[0] == 0


class TestReliableContrastiveLoss:
    def test_basic_shape_and_finite(self):
        N, feat_dim, K, M = 4, 8, 2, 10
        anchors = torch.randn(N, feat_dim)
        positives = torch.randn(N, K, feat_dim)
        negatives = [torch.randn(M, feat_dim) for _ in range(N)]

        out = reliable_contrastive_loss(anchors, positives, negatives, temperature=0.1)

        assert torch.isfinite(out["loss"])
        assert out["num_valid"] == N
        assert out["num_neg_mean"] == M
        assert out["pos_sim_mean"] != 0.0
        assert out["neg_sim_mean"] != 0.0

    def test_with_positive_weights(self):
        N, feat_dim, K, M = 4, 8, 2, 10
        anchors = torch.randn(N, feat_dim)
        positives = torch.randn(N, K, feat_dim)
        negatives = [torch.randn(M, feat_dim) for _ in range(N)]
        positive_weights = torch.softmax(torch.randn(N, K), dim=-1)

        out = reliable_contrastive_loss(
            anchors, positives, negatives,
            temperature=0.1, positive_weights=positive_weights
        )

        assert torch.isfinite(out["loss"])

    def test_varying_negatives_per_anchor(self):
        N, feat_dim, K = 3, 8, 2
        anchors = torch.randn(N, feat_dim)
        positives = torch.randn(N, K, feat_dim)
        negatives = [
            torch.randn(5, feat_dim),
            torch.randn(8, feat_dim),
            torch.randn(3, feat_dim),
        ]

        out = reliable_contrastive_loss(anchors, positives, negatives, temperature=0.1)

        assert torch.isfinite(out["loss"])
        assert out["num_neg_mean"] == (5 + 8 + 3) / 3

    def test_pos_sim_greater_than_neg_sim(self):
        N, feat_dim, K, M = 4, 8, 2, 10
        # Make positives very similar to anchors
        anchors = F.normalize(torch.randn(N, feat_dim), dim=-1)
        positives = anchors.unsqueeze(1).expand(-1, K, -1) + 0.01 * torch.randn(N, K, feat_dim)
        negatives = [torch.randn(M, feat_dim) for _ in range(N)]

        out = reliable_contrastive_loss(anchors, positives, negatives, temperature=0.1)

        # Positive similarity should be higher than negative similarity
        assert out["pos_sim_mean"] > out["neg_sim_mean"]

    def test_loss_decreases_with_better_alignment(self):
        N, feat_dim, K, M = 4, 8, 2, 10
        anchors = F.normalize(torch.randn(N, feat_dim), dim=-1)
        # Bad alignment: positives are random
        positives_bad = torch.randn(N, K, feat_dim)
        negatives = [torch.randn(M, feat_dim) for _ in range(N)]

        out_bad = reliable_contrastive_loss(anchors, positives_bad, negatives, temperature=0.1)

        # Good alignment: positives = anchors
        positives_good = anchors.unsqueeze(1).expand(-1, K, -1)
        out_good = reliable_contrastive_loss(anchors, positives_good, negatives, temperature=0.1)

        # Better alignment should yield lower loss
        assert out_good["loss"] < out_bad["loss"]
