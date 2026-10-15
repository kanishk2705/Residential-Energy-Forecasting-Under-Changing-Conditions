import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energyshift.config import INDOOR_RH, INDOOR_T, WEATHER_NODE  # noqa: E402


def make_synthetic(start="2016-03-01 00:00", days=25, seed=0) -> pd.DataFrame:
    """Frame with the real column names; weather is piecewise-linear between HH:00
    nodes, as in the source file."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 144, freq="10min")
    n = len(idx)
    tod = (idx.hour * 60 + idx.minute) / 1440
    cols = {}
    cols["Appliances"] = np.round(
        60 + 40 * np.sin(2 * np.pi * tod) + rng.gamma(2, 15, n)
    )
    cols["lights"] = rng.integers(0, 4, n) * 10.0
    for c in INDOOR_T + ["T6"]:
        cols[c] = 19 + rng.normal(0, 0.3, n).cumsum() * 0.02 + 0.5 * np.sin(2 * np.pi * tod)
    for c in INDOOR_RH + ["RH_6"]:
        cols[c] = 45 + rng.normal(0, 1, n)
    hours = pd.date_range(idx[0], idx[-1], freq="60min")
    for c in WEATHER_NODE:
        node = rng.normal(10 if c == "T_out" else 50, 3, len(hours))
        s = pd.Series(node, index=hours).reindex(idx).interpolate(method="time")
        cols[c] = s.values
    cols["rv1"] = rng.uniform(0, 50, n)
    cols["rv2"] = rng.uniform(0, 50, n)
    return pd.DataFrame(cols, index=idx)


@pytest.fixture(scope="session")
def synth():
    return make_synthetic()
