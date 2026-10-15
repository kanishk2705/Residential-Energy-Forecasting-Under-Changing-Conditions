"""Domain-classifier AUC (PROTOCOL.md section 9): how separable are two windows of inputs?"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

CALENDAR_PREFIXES = ("tod_", "dow")


def input_columns(X: pd.DataFrame) -> list[str]:
    """Non-calendar inputs: calendar columns describe the clock, not the conditions."""
    return [c for c in X.columns if not c.startswith(CALENDAR_PREFIXES)]


def domain_auc(X_ref: pd.DataFrame, X_recent: pd.DataFrame, seed: int = 0) -> float:
    """Out-of-fold AUC of a small LightGBM separating `recent` (1) from `ref` (0).

    Folds are whole calendar days, so autocorrelated neighbours never straddle the
    train/validation boundary. 0.5 = indistinguishable, 1.0 = fully separable.
    """
    cols = input_columns(X_ref)
    X = pd.concat([X_ref[cols], X_recent[cols]])
    lab = np.r_[np.zeros(len(X_ref)), np.ones(len(X_recent))]
    groups = pd.Index(X.index.normalize()).factorize()[0]
    n_groups = len(np.unique(groups))
    n_splits = min(3, n_groups)
    if n_splits < 2 or lab.min() == lab.max():
        return float("nan")
    oof = np.full(len(X), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, lab, groups):
        if lab[tr].min() == lab[tr].max():
            continue
        m = lgb.LGBMClassifier(
            n_estimators=50, num_leaves=7, learning_rate=0.1, min_child_samples=20,
            random_state=seed, n_jobs=2, verbose=-1,
        )
        m.fit(X.iloc[tr], lab[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    ok = ~np.isnan(oof)
    if lab[ok].min() == lab[ok].max():
        return float("nan")
    return float(roc_auc_score(lab[ok], oof[ok]))
