"""Prediction-interval variants I0..I3 (PROTOCOL.md section 8, E4).

All variants start from the same pair of quantile forecasts (lo = 0.1, hi = 0.9,
nominal coverage 0.80) and differ only in how the interval is widened:

  I0 raw quantiles;
  I1 static conformalised quantile regression (CQR), calibrated once;
  I2 rolling CQR, recalibrated each day at 00:00 from the last 14 days of observed scores;
  I3 adaptive conformal inference (ACI): like I2 but the target miscoverage alpha_t is
     nudged after every outcome (alpha += gamma * (alpha_nominal - miss)).

Implementation details not spelled out in the protocol (documented in the README):
  * lower bounds are clipped at 0 Wh (Appliances is non-negative) in every variant;
  * ACI clips alpha_t to [0.005, 0.5] so the conformal level stays finite;
  * an outcome is usable at time t only if its target row <= t (delayed feedback);
  * the conformal quantile uses the finite-sample level ceil((n+1)(1-alpha))/n.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import STEP

ALPHA = 0.2
WINDOW_DAYS = 14
ACI_GAMMA = 0.01
ACI_CLIP = (0.005, 0.5)


def cqr_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(lo - y, y - hi)


def conformal_q(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return float("inf")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def _finish(lo, hi, q):
    return np.clip(lo - q, 0.0, None), hi + q


def interval_frame(origins: pd.DatetimeIndex, lo, hi, y, h: int) -> pd.DataFrame:
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)  # fix any quantile crossing
    df = pd.DataFrame({"lo": lo, "hi": hi, "y": y}, index=origins)
    df["target_time"] = origins + h * STEP
    df["score"] = cqr_scores(df["lo"].to_numpy(), df["hi"].to_numpy(), df["y"].to_numpy())
    return df


def build_variants(frame: pd.DataFrame, test_start: pd.Timestamp, cal_start: pd.Timestamp,
                   cal_end: pd.Timestamp, test_end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """frame: raw quantile forecasts for origins whose target is observed, sorted by
    time, covering the calibration period through the end of the test period.

    Returns {variant: DataFrame(lo, hi, y)} for origins in [test_start, test_end].
    """
    frame = frame.sort_index()
    tt = frame["target_time"].to_numpy()
    scores = frame["score"].to_numpy()
    test_mask = (frame.index >= test_start) & (frame.index <= test_end)
    test = frame.loc[test_mask]
    out: dict[str, pd.DataFrame] = {}

    lo, hi = test["lo"].to_numpy(), test["hi"].to_numpy()
    l0, h0 = np.clip(lo, 0.0, None), hi
    out["I0_raw"] = pd.DataFrame({"lo": l0, "hi": h0, "y": test["y"]}, index=test.index)

    # I1: calibrated once on calibration-period targets
    cal = (frame["target_time"] >= cal_start) & (frame["target_time"] <= cal_end)
    q1 = conformal_q(scores[cal.to_numpy()], ALPHA)
    l1, h1 = _finish(lo, hi, q1)
    out["I1_static_cqr"] = pd.DataFrame({"lo": l1, "hi": h1, "y": test["y"]}, index=test.index)

    win = np.timedelta64(WINDOW_DAYS, "D")

    def window_slice(t_now: pd.Timestamp, inclusive: bool):
        t64 = np.datetime64(t_now)
        hi_i = np.searchsorted(tt, t64, side="right" if inclusive else "left")
        lo_i = np.searchsorted(tt, t64 - win, side="right")
        return lo_i, hi_i

    # I2: daily recalibration, scores with target strictly before the day start
    day_q = {}
    l2 = np.empty(len(test)); h2 = np.empty(len(test))
    for k, t in enumerate(test.index):
        d = t.normalize()
        if d not in day_q:
            a, b = window_slice(d, inclusive=False)
            day_q[d] = conformal_q(scores[a:b], ALPHA)
        l2[k], h2[k] = _finish(lo[k], hi[k], day_q[d])
    out["I2_rolling_cqr"] = pd.DataFrame({"lo": l2, "hi": h2, "y": test["y"]}, index=test.index)

    # I3: ACI with delayed feedback
    alpha_t = ALPHA
    l3 = np.empty(len(test)); h3 = np.empty(len(test))
    ttime_test = test["target_time"].to_numpy()
    order = np.argsort(ttime_test, kind="stable")  # target times are monotone in origin
    ptr = 0
    y_test = test["y"].to_numpy()
    alphas = np.empty(len(test))
    for k, t in enumerate(test.index):
        # ingest outcomes whose target row is observed by now (target_time <= t)
        while ptr < len(order) and ttime_test[order[ptr]] <= np.datetime64(t):
            j = order[ptr]
            assert j < k  # h >= 1, so the origin was issued strictly earlier
            miss = float(not (l3[j] <= y_test[j] <= h3[j]))
            alpha_t = float(np.clip(alpha_t + ACI_GAMMA * (ALPHA - miss), *ACI_CLIP))
            ptr += 1
        a, b = window_slice(t, inclusive=True)
        q = conformal_q(scores[a:b], alpha_t)
        l3[k], h3[k] = _finish(lo[k], hi[k], q)
        alphas[k] = alpha_t
    d3 = pd.DataFrame({"lo": l3, "hi": h3, "y": test["y"]}, index=test.index)
    d3["alpha_t"] = alphas
    out["I3_aci"] = d3
    return out


def interval_metrics(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-origin arrays whose means are coverage, width and the interval score."""
    lo, hi, y = df["lo"].to_numpy(), df["hi"].to_numpy(), df["y"].to_numpy()
    covered = ((y >= lo) & (y <= hi)).astype(float)
    width = hi - lo
    score = width + (2 / ALPHA) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))
    return {"coverage": covered, "width": width, "interval_score": score}
