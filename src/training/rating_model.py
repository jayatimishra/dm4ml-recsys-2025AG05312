# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Explicit-rating predictors (supplementary to the top-K recommenders).

Only ~7% of events carry a 1-5 rating, so ranking quality is the primary
objective. These two models exist so RMSE/MAE can be reported as the brief asks:
  GlobalMean  baseline: predict the training mean for every pair
  BiasModel   personalised: r_ui = mu + b_u + b_i, biases fit by alternating
              regularised least squares (shrinks users/items with few ratings to mu)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class GlobalMean:
    name = "global_mean"

    def fit(self, df: pd.DataFrame):
        self.mu = float(df.rating.mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mu)


class BiasModel:
    name = "bias_model"

    def __init__(self, reg: float = 5.0, iters: int = 15):
        self.reg, self.iters = reg, iters

    def fit(self, df: pd.DataFrame):
        self.mu = float(df.rating.mean())
        bu = pd.Series(0.0, index=df.user_id.unique())
        bi = pd.Series(0.0, index=df.product_id.unique())
        for _ in range(self.iters):
            res = df.rating - self.mu - df.user_id.map(bu).values
            g = res.groupby(df.product_id)
            bi = g.sum() / (g.count() + self.reg)
            res = df.rating - self.mu - df.product_id.map(bi).values
            g = res.groupby(df.user_id)
            bu = g.sum() / (g.count() + self.reg)
        self.bu, self.bi = bu, bi
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = self.mu + df.user_id.map(self.bu).fillna(0).values + df.product_id.map(self.bi).fillna(0).values
        return np.clip(p, 1, 5)
