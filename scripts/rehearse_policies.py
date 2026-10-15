"""Dry run of the E3/E4 machinery on the DEVELOPMENT region only (window = fold V4).

Purpose: catch bugs before the final test is read. Nothing here selects or tunes a
protocol parameter. Reads only data on or before DEV_END.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energyshift import intervals as iv  # noqa: E402
from energyshift.config import DEV_END, FOLDS, STEP  # noqa: E402
from energyshift.data import load_dev  # noqa: E402
from energyshift.experiments import run_intervals, run_policies  # noqa: E402
from energyshift.policies import PolicyData  # noqa: E402

PARAMS = {"n_estimators": 150, "num_leaves": 7}


def main(h: int = 6, fset: str = "A"):
    dev = load_dev()
    assert dev.index.max() <= DEV_END
    start, end = FOLDS["V4"]
    serve_end = end + STEP
    pdx = PolicyData.build(dev, h, fset)
    pol = run_policies(pdx, PARAMS, start, serve_end, start - STEP, with_sensitivity=False)
    y = pdx.y
    for k, v in pol.items():
        p = v["pred"]
        mae = float(np.mean(np.abs(p.to_numpy() - y.loc[p.index].to_numpy())))
        print(f"{k:24s} n={len(p):5d} retrains={v['retrains']:2d} MAE={mae:.2f}")
    print("P3 daily shift AUC:", pol["P3_drift_triggered"]["daily_stat"].round(2).to_dict())
    cal_start = start - pd.Timedelta(days=14)
    out = run_intervals(pdx, PARAMS, start, end, cal_start, start - STEP)
    for k, d in out.items():
        m = iv.interval_metrics(d)
        print(f"{k:16s} n={len(d)} cov={m['coverage'].mean():.3f} width={m['width'].mean():.1f}")


if __name__ == "__main__":
    main()
