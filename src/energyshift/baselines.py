"""Baselines B1..B6 (PROTOCOL.md section 6). Each returns a prediction per origin."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SPD, SPW, STEP, TARGET

BASELINES = ("B1_persistence", "B2_mean_last_hour", "B3_same_time_yesterday",
             "B4_same_time_last_week", "B5_slot_of_day_median", "B6_slot_of_week_median")


def _slot_day(index: pd.DatetimeIndex) -> np.ndarray:
    return (np.asarray(index.hour) * 60 + np.asarray(index.minute)) // 10


def _slot_week(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.dayofweek) * SPD + _slot_day(index)


def slot_medians(y_train: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Median of y by slot of day (144) and pooled-neighbour slot of week (1008)."""
    sd = _slot_day(y_train.index)
    sw = _slot_week(y_train.index)
    vals = y_train.to_numpy(dtype=float)
    med_day = np.array([np.median(vals[sd == s]) if (sd == s).any() else np.nan for s in range(SPD)])
    med_week = np.full(SPW, np.nan)
    for s in range(SPW):
        m = (sw == s) | (sw == (s - 1) % SPW) | (sw == (s + 1) % SPW)
        if m.any():
            med_week[s] = np.median(vals[m])
    # fill any empty slot with the global median (only matters for very short training sets)
    g = np.nanmedian(vals)
    return np.where(np.isnan(med_day), g, med_day), np.where(np.isnan(med_week), g, med_week)


def baseline_predictions(
    df: pd.DataFrame, origins: pd.DatetimeIndex, h: int, y_train: pd.Series
) -> pd.DataFrame:
    """Predictions of all baselines for `origins`.

    `df` supplies history (rows <= origin only are ever read); `y_train` is the
    series of Appliances values with labels known at forecast time, used only by
    B5/B6.
    """
    y = df[TARGET].astype(float)
    out = pd.DataFrame(index=origins)
    out["B1_persistence"] = y.shift(0).reindex(origins).to_numpy()
    out["B2_mean_last_hour"] = y.rolling(6).mean().reindex(origins).to_numpy()
    out["B3_same_time_yesterday"] = y.shift(SPD - h).reindex(origins).to_numpy()
    out["B4_same_time_last_week"] = y.shift(SPW - h).reindex(origins).to_numpy()
    med_day, med_week = slot_medians(y_train)
    tau = origins + h * STEP
    out["B5_slot_of_day_median"] = med_day[_slot_day(tau)]
    out["B6_slot_of_week_median"] = med_week[_slot_week(tau)]
    return out
