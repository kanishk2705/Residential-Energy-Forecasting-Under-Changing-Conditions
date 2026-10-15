"""E3 (retraining policies) and E4 (prediction intervals) as functions of explicit dates,
so the same code runs on the development rehearsal and on the final test."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import intervals as iv
from .config import STEP
from .policies import (PolicyData, daily, fit_at, mondays, predict_segment, run_drift_triggered,
                       run_schedule, schedule_at, shift_statistic)

P3_THRESHOLD = 0.65
P3_SENSITIVITY = (0.60, 0.70, 0.75)
SLIDING_DAYS = 28


def run_policies(pdx: PolicyData, params: dict, serve_start: pd.Timestamp, serve_end: pd.Timestamp,
                 p0_label_end: pd.Timestamp, with_sensitivity: bool = True) -> dict:
    """serve window is [serve_start, serve_end). Returns {name: {"pred": Series, "retrains": int}}.

    P0 frozen (labels up to p0_label_end); P1 expanding weekly; P2 sliding 28 d weekly;
    P3 drift-triggered (AUC >= 0.65, >= 3 days apart); P4 expanding daily.
    """
    weekly = mondays(serve_start, serve_end)
    if not weekly or weekly[0] != serve_start:
        weekly = [serve_start] + [m for m in weekly if m > serve_start]
    out = {}
    out["P0_frozen"] = dict(pred=run_schedule(pdx, params, serve_start, serve_end,
                                              [(serve_start, p0_label_end)]), retrains=0)
    out["P1_expanding_weekly"] = dict(
        pred=run_schedule(pdx, params, serve_start, serve_end, schedule_at(weekly)),
        retrains=len(weekly))
    out["P2_sliding28_weekly"] = dict(
        pred=run_schedule(pdx, params, serve_start, serve_end, schedule_at(weekly), SLIDING_DAYS),
        retrains=len(weekly))
    p3, rt, stat = run_drift_triggered(pdx, params, serve_start, serve_end, p0_label_end, P3_THRESHOLD)
    out["P3_drift_triggered"] = dict(pred=p3, retrains=len(rt), retrain_times=[str(t) for t in rt],
                                     daily_stat=stat)
    dly = daily(serve_start, serve_end)
    out["P4_expanding_daily"] = dict(
        pred=run_schedule(pdx, params, serve_start, serve_end, schedule_at(dly)), retrains=len(dly))
    if with_sensitivity:
        for thr in P3_SENSITIVITY:
            p, rt, _ = run_drift_triggered(pdx, params, serve_start, serve_end, p0_label_end, thr)
            out[f"P3_threshold_{thr:.2f}"] = dict(pred=p, retrains=len(rt),
                                                  retrain_times=[str(t) for t in rt])
    idx = out["P0_frozen"]["pred"].index
    for k, v in out.items():
        assert v["pred"].index.equals(idx), f"{k} serves different origins"
    return out


def run_intervals(pdx: PolicyData, params: dict, serve_start: pd.Timestamp, serve_end_incl: pd.Timestamp,
                  cal_start: pd.Timestamp, cal_end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Quantile models (0.1/0.9) trained on origins whose target row is before `cal_start`;
    calibration targets in [cal_start, cal_end]; variants I0..I3 on [serve_start, serve_end_incl]."""
    label_end = cal_start - STEP
    lo_m = fit_at(pdx, params, label_end, None, objective="quantile", alpha=0.1)
    hi_m = fit_at(pdx, params, label_end, None, objective="quantile", alpha=0.9)
    idx = pdx.X.index
    keep = (idx + pdx.h * STEP >= cal_start) & pdx.y.notna().to_numpy() & (idx <= serve_end_incl)
    origins = idx[keep]
    lo = lo_m.predict(pdx.X.loc[origins])
    hi = hi_m.predict(pdx.X.loc[origins])
    frame = iv.interval_frame(origins, lo, hi, pdx.y.loc[origins].to_numpy(), pdx.h)
    return iv.build_variants(frame, serve_start, cal_start, cal_end, serve_end_incl)
