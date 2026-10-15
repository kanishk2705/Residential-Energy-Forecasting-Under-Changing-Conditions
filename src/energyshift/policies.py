"""Retraining policies P0..P4 (PROTOCOL.md section 8, E3).

Implementation details not spelled out in the protocol, recorded here and in the
README:
  * a retraining at time T uses origins whose target row is strictly before T;
  * the drift statistic's "recent" window excludes any origin that is also in the
    "reference" window, and is skipped (no trigger) if fewer than 100 rows remain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import STEP
from .features import build_features
from .models import LGBMEnsemble
from .shift import domain_auc

MIN_RECENT_ROWS = 100


@dataclass
class PolicyData:
    """Everything a policy simulation needs for one (horizon, feature set)."""

    h: int
    fset: str
    X: pd.DataFrame  # features at every origin with complete history (no NaN)
    y: pd.Series  # Appliances at origin + h (NaN where unknown / beyond the data)
    cache: dict = field(default_factory=dict)

    @classmethod
    def build(cls, df: pd.DataFrame, h: int, fset: str) -> "PolicyData":
        X = build_features(df, h, fset)
        X = X.loc[X.notna().all(axis=1)]
        y = df["Appliances"].astype(float).shift(-h).reindex(X.index)
        return cls(h, fset, X, y)


def _train_index(pd_: PolicyData, label_end: pd.Timestamp, window_days: int | None):
    """Origins usable for training with labels observed up to `label_end` (inclusive)."""
    idx = pd_.X.index
    ok = (idx + pd_.h * STEP <= label_end) & pd_.y.notna().to_numpy()
    if window_days is not None:
        ok &= idx > label_end - pd.Timedelta(days=window_days)
    return idx[ok]


def fit_at(pd_: PolicyData, params: dict, label_end: pd.Timestamp,
           window_days: int | None = None, objective: str = "l1",
           alpha: float | None = None) -> LGBMEnsemble:
    key = (pd_.h, pd_.fset, tuple(sorted(params.items())), label_end, window_days, objective, alpha)
    if key not in pd_.cache:
        tr = _train_index(pd_, label_end, window_days)
        pd_.cache[key] = LGBMEnsemble(params, objective, alpha).fit(pd_.X.loc[tr], pd_.y.loc[tr])
    return pd_.cache[key]


def predict_segment(pd_: PolicyData, model: LGBMEnsemble, start: pd.Timestamp,
                    stop: pd.Timestamp) -> pd.Series:
    """Predictions for origins in [start, stop) that have a known target."""
    idx = pd_.X.index
    m = (idx >= start) & (idx < stop) & pd_.y.notna().to_numpy()
    sel = idx[m]
    return pd.Series(model.predict(pd_.X.loc[sel]), index=sel)


def run_schedule(pd_: PolicyData, params: dict, serve_start: pd.Timestamp, serve_end: pd.Timestamp,
                 schedule: list[tuple[pd.Timestamp, pd.Timestamp]], window_days: int | None = None
                 ) -> pd.Series:
    """schedule = [(serve_from, label_end_inclusive), ...] in time order.

    The model fitted for entry k serves origins in [serve_from_k, serve_from_{k+1}).
    """
    parts = []
    for k, (t0, label_end) in enumerate(schedule):
        t1 = schedule[k + 1][0] if k + 1 < len(schedule) else serve_end
        model = fit_at(pd_, params, label_end, window_days)
        parts.append(predict_segment(pd_, model, max(t0, serve_start), min(t1, serve_end)))
    return pd.concat(parts)


def mondays(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Every Monday 00:00 in [start, end)."""
    days = pd.date_range(start.normalize(), end, freq="D")
    return [d for d in days if d.dayofweek == 0 and start <= d < end]


def daily(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return [d for d in pd.date_range(start.normalize(), end, freq="D") if start <= d < end]


def schedule_at(times: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [(t, t - STEP) for t in times]  # labels strictly before t


def shift_statistic(pd_: PolicyData, T: pd.Timestamp, label_end: pd.Timestamp,
                    window_days: int | None = None) -> float:
    """AUC between the model's last 28 days of training origins and the last 7 days of
    observed inputs before T (overlap removed). NaN if the recent window is too small."""
    train_idx = _train_index(pd_, label_end, window_days)
    ref = train_idx[train_idx > train_idx.max() - pd.Timedelta(days=28)] if len(train_idx) else train_idx
    idx = pd_.X.index
    recent = idx[(idx >= T - pd.Timedelta(days=7)) & (idx < T)]
    recent = recent.difference(ref)
    if len(ref) < MIN_RECENT_ROWS or len(recent) < MIN_RECENT_ROWS:
        return float("nan")
    return domain_auc(pd_.X.loc[ref], pd_.X.loc[recent])


def run_drift_triggered(pd_: PolicyData, params: dict, serve_start: pd.Timestamp,
                        serve_end: pd.Timestamp, p0_label_end: pd.Timestamp,
                        threshold: float, min_gap_days: int = 3,
                        window_days: int | None = None):
    """P3. Checked every day at 00:00; retrain (expanding) when AUC >= threshold and
    at least `min_gap_days` have passed since the last retrain. The starting model is
    the P0 model (trained with labels up to `p0_label_end`).

    Returns (predictions, retrain_times, daily_stat_series).
    """
    cur_label_end = p0_label_end
    last_retrain = serve_start - pd.Timedelta(days=min_gap_days)  # first check may fire
    schedule = [(serve_start, p0_label_end)]
    stats = {}
    retrains = []
    for d in daily(serve_start, serve_end):
        stat = shift_statistic(pd_, d, cur_label_end, window_days)
        stats[d] = stat
        if (not np.isnan(stat)) and stat >= threshold and (d - last_retrain) >= pd.Timedelta(days=min_gap_days):
            cur_label_end = d - STEP
            last_retrain = d
            retrains.append(d)
            schedule.append((d, cur_label_end))
    preds = run_schedule(pd_, params, serve_start, serve_end, schedule, window_days)
    return preds, retrains, pd.Series(stats, name="shift_auc")
