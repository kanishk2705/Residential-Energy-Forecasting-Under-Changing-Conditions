"""Day-block bootstrap (PROTOCOL.md section 10).

A statistic that is a ratio of sums over origins (MAE, MSE, coverage, ...) is
bootstrapped by resampling whole calendar days with replacement. Differences
between two methods use the same resampled days (paired).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BOOTSTRAP_B, BOOTSTRAP_SEED


class DayBlocks:
    """Per-day sums so that B resamples cost one matrix product."""

    def __init__(self, index: pd.DatetimeIndex, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED):
        days = pd.Index(index.normalize())
        self.day_ids, self.inv = np.unique(days.to_numpy(), return_inverse=True)
        self.D = len(self.day_ids)
        self.n_d = np.bincount(self.inv, minlength=self.D).astype(float)
        rng = np.random.default_rng(seed)
        # multinomial counts == resampling D days with replacement
        self.counts = rng.multinomial(self.D, np.full(self.D, 1 / self.D), size=B).astype(float)

    def day_sum(self, values: np.ndarray) -> np.ndarray:
        return np.bincount(self.inv, weights=np.asarray(values, dtype=float), minlength=self.D)

    def ratio(self, values: np.ndarray) -> tuple[float, np.ndarray]:
        """Point estimate and bootstrap replicates of mean(values)."""
        s = self.day_sum(values)
        point = s.sum() / self.n_d.sum()
        reps = (self.counts @ s) / (self.counts @ self.n_d)
        return float(point), reps

    def diff(self, a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
        """mean(a) - mean(b), paired over the same resampled days."""
        pa, ra = self.ratio(a)
        pb, rb = self.ratio(b)
        return pa - pb, ra - rb


def ci(reps: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.quantile(reps, [a, 1 - a])
    return float(lo), float(hi)


def summarize(point: float, reps: np.ndarray, level: float = 0.95) -> dict:
    lo, hi = ci(reps, level)
    return {"estimate": point, "ci_lo": lo, "ci_hi": hi, "level": level}


def moving_block_reps(values_a, values_b=None, block: int = 288, B: int = 2000,
                      seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Circular moving-block bootstrap replicates of mean(a) [- mean(b)] (sensitivity check)."""
    a = np.asarray(values_a, dtype=float)
    b = None if values_b is None else np.asarray(values_b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    idx = idx.reshape(B, -1)[:, :n]
    reps = a[idx].mean(axis=1)
    if b is not None:
        reps = reps - b[idx].mean(axis=1)
    return reps


def spearman_day_ci(x: np.ndarray, y: np.ndarray, B: int = BOOTSTRAP_B,
                    seed: int = BOOTSTRAP_SEED) -> dict:
    """Spearman correlation with a bootstrap-over-days CI (exploratory)."""
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    reps = []
    n = len(x)
    for _ in range(B):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        reps.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"rho": float(rho), "ci_lo": float(lo), "ci_hi": float(hi), "n_days": int(n)}
