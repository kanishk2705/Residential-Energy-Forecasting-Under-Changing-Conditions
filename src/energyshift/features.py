"""Forecast inputs, built so that a feature at origin row i uses only rows <= i.

Origin i, horizon h: the target is Appliances at row i+h. Calendar features describe
the target time (known in advance). Airport-weather columns are interpolated
between hourly nodes at HH:00 in the source file, so a value at HH:10 already
encodes the HH+1:00 reading; only the value at the most recent HH:00 row <= i is
used (`hourly_node`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    INDOOR_RH,
    INDOOR_T,
    RANDOM_COLS,
    SPD,
    STEP,
    TARGET,
    WEATHER_NODE,
)

FEATURE_SETS = ("A", "B", "C", "N")


def hourly_node(series: pd.Series) -> pd.Series:
    """Value at the most recent row with minute == 0 (a real hourly observation)."""
    return series.where(series.index.minute == 0).ffill()


def calendar_features(index: pd.DatetimeIndex, h: int) -> pd.DataFrame:
    tau = index + h * STEP
    slot = (np.asarray(tau.hour) * 60 + np.asarray(tau.minute)) // 10
    out: dict[str, np.ndarray] = {}
    for k in (1, 2, 3):
        ang = 2 * np.pi * k * slot / SPD
        out[f"tod_sin{k}"] = np.sin(ang)
        out[f"tod_cos{k}"] = np.cos(ang)
    dow = np.asarray(tau.dayofweek)
    for d in range(7):
        out[f"dow{d}"] = (dow == d).astype(float)
    return pd.DataFrame(out, index=index)


def _set_a(df: pd.DataFrame, h: int) -> pd.DataFrame:
    y = df[TARGET].astype(float)
    f: dict[str, pd.Series] = {}
    for k in (0, 1, 2, 3, 5, 11):
        f[f"y_lag{k}"] = y.shift(k)
    f["y_mean6"] = y.rolling(6).mean()
    f["y_mean36"] = y.rolling(36).mean()
    f["y_mean144"] = y.rolling(144).mean()
    f["y_std36"] = y.rolling(36).std()
    yday = y.shift(SPD - h)  # y[i+h-144]; shift >= 0 because h <= 144
    f["y_yday"] = yday
    f["y_yday_mean7"] = yday.rolling(7).mean()  # rows i+h-150 .. i+h-144
    out = pd.DataFrame(f)
    return pd.concat([calendar_features(df.index, h), out], axis=1)


def _set_b_extra(df: pd.DataFrame) -> pd.DataFrame:
    f: dict[str, pd.Series] = {}
    f["lights"] = df["lights"].astype(float)
    f["lights_mean6"] = df["lights"].astype(float).rolling(6).mean()
    for c in INDOOR_T + INDOOR_RH:
        f[c] = df[c].astype(float)
    mt = df[INDOOR_T].astype(float).mean(axis=1)
    f["indoor_T_mean"] = mt
    f["indoor_T_delta6"] = mt - mt.shift(6)
    return pd.DataFrame(f)


def _set_c_extra(df: pd.DataFrame) -> pd.DataFrame:
    f: dict[str, pd.Series] = {}
    nodes = {c: hourly_node(df[c].astype(float)) for c in WEATHER_NODE}
    for c, s in nodes.items():
        f[f"{c}_node"] = s
    t_node = nodes["T_out"]
    f["T_out_node_d1h"] = t_node - t_node.shift(6)
    f["T_out_node_d6h"] = t_node - t_node.shift(36)
    hourly = df["T_out"].astype(float)[df.index.minute == 0]
    f["T_out_mean24nodes"] = hourly.rolling(24).mean().reindex(df.index, method="ffill")
    f["T6"] = df["T6"].astype(float)
    f["RH_6"] = df["RH_6"].astype(float)
    mt = df[INDOOR_T].astype(float).mean(axis=1)
    f["indoor_minus_T6"] = mt - df["T6"].astype(float)
    return pd.DataFrame(f)


def build_features(df: pd.DataFrame, h: int, fset: str) -> pd.DataFrame:
    """Feature matrix for every row as an origin (NaN where history is too short)."""
    if not 1 <= h <= SPD:
        raise ValueError("horizon must be between 1 and 144 steps")
    if fset not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {fset}")
    parts = [_set_a(df, h)]
    if fset in ("B", "C", "N"):
        parts.append(_set_b_extra(df))
    if fset in ("C", "N"):
        parts.append(_set_c_extra(df))
    if fset == "N":
        parts.append(df[RANDOM_COLS].astype(float))
    return pd.concat(parts, axis=1)


def make_xy(df: pd.DataFrame, h: int, fset: str) -> tuple[pd.DataFrame, pd.Series]:
    """Aligned (X, y) at origin rows where features and the target both exist."""
    X = build_features(df, h, fset)
    y = df[TARGET].astype(float).shift(-h)
    ok = X.notna().all(axis=1) & y.notna()
    return X.loc[ok], y.loc[ok].rename("y_target")


def label_known(origins: pd.DatetimeIndex, h: int, last_label_time: pd.Timestamp) -> np.ndarray:
    """True where the target row (origin + h steps) is observed by `last_label_time`."""
    return np.asarray(origins + h * STEP <= last_label_time)
