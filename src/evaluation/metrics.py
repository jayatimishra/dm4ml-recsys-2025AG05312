# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Evaluation metrics.

Ranking (top-K, per user, then averaged over users with >=1 relevant item):
  Precision@K  hits / K
  Recall@K     hits / |relevant|
  NDCG@K       DCG / ideal DCG, binary relevance, log2 discount
  MAP@K        mean of precision at each hit, / min(|relevant|, K)
  HitRate@K    share of users with at least one hit
Beyond-accuracy:
  Coverage     distinct items recommended across all users / catalogue size
  Diversity    intra-list diversity: share of item pairs inside a user's top-K that
               belong to DIFFERENT categories (0 = all same category, 1 = all different)
  Novelty      mean self-information -log2(p(item)) of recommended items, p = interaction share
Rating prediction (explicit 1-5 ratings): RMSE, MAE.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def ranking_metrics(model, truth: dict, k: int = 10) -> dict:
    """truth: user_id -> set(relevant product_ids) in the holdout window (unseen items only)."""
    p, r, ndcg, ap, hit, ild, nov, rec_items = [], [], [], [], [], [], [], set()
    disc = 1 / np.log2(np.arange(2, k + 2))
    cat = dict(zip(model.item_ids, model.item_category))
    pop = model.X.getnnz(axis=0).astype(float) + 1.0
    info = dict(zip(model.item_ids, -np.log2(pop / pop.sum())))
    for u, rel in truth.items():
        recs = model.recommend(u, k)
        rec_items.update(recs)
        hits = np.array([i in rel for i in recs], dtype=float)
        p.append(hits.sum() / k)
        r.append(hits.sum() / len(rel))
        ndcg.append((hits * disc).sum() / disc[: min(len(rel), k)].sum())
        ap.append(((np.cumsum(hits) / np.arange(1, k + 1)) * hits).sum() / min(len(rel), k))
        hit.append(float(hits.any()))
        pairs = list(combinations(recs, 2))
        ild.append(np.mean([cat[a] != cat[b] for a, b in pairs]))
        nov.append(np.mean([info[i] for i in recs]))
    return {f"precision_at_{k}": float(np.mean(p)), f"recall_at_{k}": float(np.mean(r)),
            f"ndcg_at_{k}": float(np.mean(ndcg)), f"map_at_{k}": float(np.mean(ap)),
            f"hit_rate_at_{k}": float(np.mean(hit)),
            "coverage": len(rec_items) / len(model.item_ids),
            "diversity": float(np.mean(ild)), "novelty": float(np.mean(nov)),
            "n_eval_users": len(truth)}


def rating_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "n_ratings": int(len(y_true))}
