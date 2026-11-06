# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Recommender algorithms + MLflow pyfunc wrapper (metrics live in src/evaluation/metrics.py).

All models consume feature-store outputs only:
  user_item_interaction.ui_strength      -> implicit feedback matrix
  item_popularity_quality                -> popularity / trending prior, item category
  user_category_affinity.uc_affinity     -> content signal for the hybrid
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def _minmax_rows(m: np.ndarray) -> np.ndarray:
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    return (m - lo) / np.where(hi - lo == 0, 1, hi - lo)


class Recommender:
    """Common fit/score/recommend interface. Scores are dense (n_users x n_items)."""

    name = "base"

    def __init__(self, **params):
        self.params = params

    def fit(self, ui: pd.DataFrame, items: pd.DataFrame, affinity: pd.DataFrame) -> "Recommender":
        self.watermark = "Roll No: 2025AG05312"
        self.user_ids = np.sort(ui.user_id.unique())
        self.item_ids = np.sort(items.product_id.unique())
        self.u_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.i_idx = {p: i for i, p in enumerate(self.item_ids)}
        rows = ui.user_id.map(self.u_idx).values
        cols = ui.product_id.map(self.i_idx).values
        self.X = sparse.csr_matrix((np.log1p(ui.ui_strength.clip(lower=0).values), (rows, cols)),
                                   shape=(len(self.user_ids), len(self.item_ids)))
        it = items.set_index("product_id").reindex(self.item_ids)
        pop = (it.i_n_purchases * 5 + it.i_n_carts * 3 + it.i_n_views).fillna(0).values
        trend = it.i_events_last_7d.fillna(0).values
        self.pop = 0.5 * pop / max(pop.max(), 1) + 0.5 * trend / max(trend.max(), 1)
        self.item_category = it.category.values
        cats = sorted(set(self.item_category))
        aff = (affinity.pivot_table(index="user_id", columns="category", values="uc_affinity", fill_value=0)
               .reindex(index=self.user_ids, columns=cats, fill_value=0))
        cat_pos = {c: j for j, c in enumerate(cats)}
        self.content = aff.values[:, [cat_pos[c] for c in self.item_category]]  # user x item
        self._fit()
        self.scores = self._score()
        return self

    def _fit(self):
        pass

    def _score(self) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_id, k: int = 10, exclude_seen: bool = True) -> list:
        if user_id not in self.u_idx:                 # cold-start user -> popularity
            return [int(self.item_ids[j]) for j in np.argsort(-self.pop)[:k]]
        u = self.u_idx[user_id]
        s = self.scores[u].copy()
        if exclude_seen:
            s[self.X[u].indices] = -np.inf
        top = np.argpartition(-s, k)[:k]
        return [int(self.item_ids[j]) for j in top[np.argsort(-s[top])]]


class Popularity(Recommender):
    name = "popularity"

    def _score(self):
        return np.tile(self.pop, (len(self.user_ids), 1))


class ItemKNN(Recommender):
    """Item-item collaborative filtering with cosine similarity, top-n neighbours kept."""
    name = "item_knn"

    def _fit(self):
        Xn = normalize(self.X.T.tocsr(), axis=1)
        S = (Xn @ Xn.T).toarray()
        np.fill_diagonal(S, 0)
        n = self.params.get("neighbours", 50)
        if n < S.shape[0]:
            thresh = -np.sort(-S, axis=1)[:, n - 1:n]
            S[S < thresh] = 0
        self.S = S

    def _score(self):
        return np.asarray(self.X @ self.S)


class SVDMF(Recommender):
    """Matrix factorisation of the implicit matrix with truncated SVD."""
    name = "svd_mf"

    def _fit(self):
        svd = TruncatedSVD(n_components=self.params.get("factors", 32), random_state=529)
        self.U = svd.fit_transform(self.X)
        self.V = svd.components_
        self.explained_variance = float(svd.explained_variance_ratio_.sum())

    def _score(self):
        return self.U @ self.V


class Hybrid(Recommender):
    """(1-a-b) * CF(SVD) + a * content(category affinity) + b * popularity prior."""
    name = "hybrid"

    def _fit(self):
        self.mf = SVDMF(factors=self.params.get("factors", 32))
        self.mf.__dict__.update({k: v for k, v in self.__dict__.items() if k not in ("params", "mf")})
        self.mf._fit()

    def _score(self):
        a, b = self.params.get("alpha", 0.2), self.params.get("beta", 0.1)
        cf = _minmax_rows(self.mf._score())
        return (1 - a - b) * cf + a * self.content + b * self.pop[None, :]


MODELS = {m.name: m for m in [Popularity, ItemKNN, SVDMF, Hybrid]}


# ---------------------------------------------------------------- MLflow pyfunc wrapper
try:
    import mlflow.pyfunc

    class RecommenderPyfunc(mlflow.pyfunc.PythonModel):
        """Input: DataFrame[user_id, (k)]. Output: DataFrame[user_id, rank, product_id, score]."""

        def load_context(self, context):
            import pickle
            with open(context.artifacts["model"], "rb") as f:
                self.model = pickle.load(f)

        def predict(self, context, model_input, params=None):
            rows = []
            for rec in pd.DataFrame(model_input).to_dict("records"):
                u, k = int(rec["user_id"]), int(rec.get("k", 10) or 10)
                items = self.model.recommend(u, k)
                for rank, it in enumerate(items, 1):
                    j = self.model.i_idx[it]
                    s = float(self.model.scores[self.model.u_idx[u], j]) if u in self.model.u_idx \
                        else float(self.model.pop[j])
                    rows.append({"user_id": u, "rank": rank, "product_id": it, "score": round(s, 5)})
            return pd.DataFrame(rows)
except ImportError:  # mlflow optional for pure evaluation
    pass
