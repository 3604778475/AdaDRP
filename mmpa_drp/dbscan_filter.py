"""
DBSCAN-based contrastive positive filtering.

For each drug, cluster source + target fused features with DBSCAN,
then keep only Top-K candidates that fall into the same cluster as the anchor.
"""

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


def dbscan_filter_topk(
    all_source_fusion_feat,
    all_target_fusion_feat,
    topk_idx,
    eps=0.3,
    min_samples=5,
    distance="cosine",
):
    """
    Filter Top-K positive candidates using per-drug DBSCAN clustering.

    Args:
        all_source_fusion_feat: [N_s, drug_num, D] torch tensor (CPU or GPU)
        all_target_fusion_feat: [N_t, drug_num, D] torch tensor
        topk_idx: [N_s, drug_num, K] torch tensor, indices into target domain
        eps: DBSCAN neighborhood radius
        min_samples: DBSCAN min samples
        distance: 'cosine' or 'euclidean'

    Returns:
        topk_mask: [N_s, drug_num, K] torch float tensor (1 = keep, 0 = filter out)
    """
    source_np = all_source_fusion_feat.detach().cpu().numpy()
    target_np = all_target_fusion_feat.detach().cpu().numpy()
    topk_idx_np = topk_idx.detach().cpu().numpy()

    N_s, drug_num, D = source_np.shape
    _, _, K = topk_idx_np.shape
    topk_mask = np.zeros((N_s, drug_num, K), dtype=np.float32)

    for d in range(drug_num):
        s_feat = source_np[:, d, :].reshape(N_s, D)
        t_feat = target_np[:, d, :]
        combined = np.concatenate([s_feat, t_feat], axis=0)

        if distance == "cosine":
            dist_mat = cosine_distances(combined)
        elif distance == "euclidean":
            dist_mat = np.linalg.norm(combined[:, None, :] - combined[None, :, :], axis=2)
        else:
            raise ValueError(f"Unknown distance: {distance}")

        cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
        labels = cluster.fit_predict(dist_mat)

        src_labels = labels[:N_s]
        tgt_labels = labels[N_s:]

        for i in range(N_s):
            c = src_labels[i]
            if c == -1:
                continue  # outlier anchor: no positives
            candidates = topk_idx_np[i, d, :]
            cand_labels = tgt_labels[candidates]
            topk_mask[i, d, :] = (cand_labels == c).astype(np.float32)

    return torch.from_numpy(topk_mask).to(all_source_fusion_feat.device)


def dbscan_adaptive_neighbors(
    all_source_fusion_feat,
    all_target_fusion_feat,
    K,
    eps=0.3,
    min_samples=5,
    distance="cosine",
    fallback_on_outlier=False,
):
    """
    DBSCAN-based adaptive neighbor selection (SPRED-style, no pre Top-K).

    For each drug and each source anchor, positive neighbors are drawn from the
    same DBSCAN cluster as the anchor. The number of positives is adaptive per
    anchor but capped at K. If a cluster contains fewer than K samples, the
    closest positive is repeated to pad to K. If the anchor is a noise point and
    ``fallback_on_outlier`` is True, ordinary Top-K by cosine similarity is used
    as a fallback.

    Args:
        all_source_fusion_feat: [N_s, drug_num, D] torch tensor
        all_target_fusion_feat: [N_t, drug_num, D] torch tensor
        K: maximum number of positives per anchor
        eps: DBSCAN neighborhood radius
        min_samples: DBSCAN min samples
        distance: 'cosine' or 'euclidean' (clustering distance only)
        fallback_on_outlier: if True, outlier anchors fall back to similarity Top-K

    Returns:
        topk_sim: [N_s, drug_num, K] torch float tensor
        topk_idx: [N_s, drug_num, K] torch long tensor
        keep_ratio: float, average ratio of anchors with at least one cluster mate
    """
    source_np = all_source_fusion_feat.detach().cpu().numpy()
    target_np = all_target_fusion_feat.detach().cpu().numpy()

    N_s, drug_num, D = source_np.shape
    N_t = target_np.shape[0]

    topk_idx = np.zeros((N_s, drug_num, K), dtype=np.int64)
    topk_sim = np.zeros((N_s, drug_num, K), dtype=np.float32)
    anchor_valid_mask = np.zeros((N_s, drug_num), dtype=np.float32)

    for d in range(drug_num):
        s_feat = source_np[:, d, :].reshape(N_s, D)
        t_feat = target_np[:, d, :]
        combined = np.concatenate([s_feat, t_feat], axis=0)

        if distance == "cosine":
            dist_mat = cosine_distances(combined)
        elif distance == "euclidean":
            dist_mat = np.linalg.norm(combined[:, None, :] - combined[None, :, :], axis=2)
        else:
            raise ValueError(f"Unknown distance: {distance}")

        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1).fit_predict(dist_mat)
        src_labels = labels[:N_s]
        tgt_labels = labels[N_s:]

        # Cosine similarity is used for ranking positives regardless of clustering distance.
        sim = cosine_similarity(s_feat, t_feat)

        for i in range(N_s):
            c = src_labels[i]
            if c == -1:
                # Outlier anchor: no reliable target neighbors.
                if fallback_on_outlier:
                    order = np.argsort(-sim[i])[:K]
                    sel = order
                    sel_sim = sim[i, sel]
                    anchor_valid_mask[i, d] = 1.0
                else:
                    sel = np.zeros(K, dtype=np.int64)
                    sel_sim = np.zeros(K, dtype=np.float32)
                    anchor_valid_mask[i, d] = 0.0
            else:
                candidates = np.where(tgt_labels == c)[0]
                cand_sim = sim[i, candidates]
                order = np.argsort(-cand_sim)
                ordered = candidates[order]
                ordered_sim = cand_sim[order]
                n = len(ordered)

                if n >= K:
                    sel = ordered[:K]
                    sel_sim = ordered_sim[:K]
                elif n > 0:
                    # pad by repeating the closest positive
                    pad = K - n
                    sel = np.concatenate([ordered, np.full(pad, ordered[-1], dtype=np.int64)])
                    sel_sim = np.concatenate([ordered_sim, np.full(pad, ordered_sim[-1], dtype=np.float32)])
                else:
                    # empty cluster fallback: keep the anchor valid but use similarity Top-K
                    order = np.argsort(-sim[i])[:K]
                    sel = order
                    sel_sim = sim[i, sel]

                anchor_valid_mask[i, d] = 1.0

            topk_idx[i, d, :] = sel
            topk_sim[i, d, :] = sel_sim

    keep_ratio = float(anchor_valid_mask.mean())
    return (
        torch.from_numpy(topk_sim).to(all_source_fusion_feat.device),
        torch.from_numpy(topk_idx).to(all_source_fusion_feat.device),
        torch.from_numpy(anchor_valid_mask).to(all_source_fusion_feat.device),
        keep_ratio,
    )
