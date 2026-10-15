"""Learned models and the frozen hyper-parameter grids (PROTOCOL.md section 6)."""
from __future__ import annotations

import itertools
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LGBM_SEEDS = (0, 1, 2)

GRID: dict[str, list[dict[str, Any]]] = {
    "ridge": [{"alpha": a} for a in (1.0, 10.0, 100.0)],
    "rf": [{"min_samples_leaf": m} for m in (5, 20)],
    "lgbm": [
        {"n_estimators": n, "num_leaves": l}
        for n, l in itertools.product((150, 400), (7, 15))
    ],
}


def _lgbm_params(params: dict[str, Any], seed: int, objective: str, alpha: float | None) -> dict:
    p = dict(
        objective=objective,
        learning_rate=0.05,
        n_estimators=params["n_estimators"],
        num_leaves=params["num_leaves"],
        min_child_samples=40,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        verbose=-1,
    )
    if alpha is not None:
        p["alpha"] = alpha
    return p


class LGBMEnsemble:
    """Mean of LightGBM fits over seeds. objective 'l1' (point) or 'quantile'."""

    def __init__(self, params, objective="l1", alpha=None, seeds=LGBM_SEEDS):
        self.params, self.objective, self.alpha, self.seeds = params, objective, alpha, seeds
        self.models: list = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMEnsemble":
        self.models = []
        for s in self.seeds:
            m = lgb.LGBMRegressor(**_lgbm_params(self.params, s, self.objective, self.alpha))
            m.fit(X, y)
            self.models.append(m)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(np.mean([m.predict(X) for m in self.models], axis=0), 0.0, None)

    def importance_gain(self, columns) -> pd.Series:
        g = np.mean([m.booster_.feature_importance("gain") for m in self.models], axis=0)
        return pd.Series(g / g.sum(), index=list(columns))


def fit_predict(
    family: str,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    objective: str = "l1",
    alpha: float | None = None,
    seeds: tuple[int, ...] = LGBM_SEEDS,
):
    """Predictions for X_test, clipped at 0 Wh."""
    if family == "ridge":
        m = make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))
        m.fit(X_train, np.log1p(y_train))
        pred = np.expm1(m.predict(X_test))
    elif family == "rf":
        m = RandomForestRegressor(
            n_estimators=300,
            max_features=0.5,
            min_samples_leaf=params["min_samples_leaf"],
            n_jobs=2,
            random_state=0,
        )
        m.fit(X_train, np.log1p(y_train))
        pred = np.expm1(m.predict(X_test))
    elif family == "lgbm":
        return LGBMEnsemble(params, objective, alpha, seeds).fit(X_train, y_train).predict(X_test)
    else:
        raise ValueError(family)
    return np.clip(pred, 0.0, None)


def all_configs() -> list[tuple[str, dict[str, Any]]]:
    return [(fam, p) for fam, plist in GRID.items() for p in plist]
