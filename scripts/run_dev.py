"""Development-region experiments: rolling-origin grid, selection, negative controls.

Reads ONLY data on or before DEV_END (via load_dev). Writes results/dev/*.
Resumable: a horizon whose fold CSV already exists is skipped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energyshift.baselines import BASELINES, baseline_predictions  # noqa: E402
from energyshift.config import FOLDS, HORIZONS, RESULTS, STEP, TARGET  # noqa: E402
from energyshift.data import load_dev  # noqa: E402
from energyshift.features import label_known, make_xy  # noqa: E402
from energyshift.models import LGBMEnsemble, all_configs, fit_predict  # noqa: E402

OUT = RESULTS / "dev"
SETS = ("A", "B", "C", "N")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fold_slices(X: pd.DataFrame, h: int, fold: str):
    vs, ve = FOLDS[fold]
    train = label_known(X.index, h, vs - STEP)  # labels strictly before validation start
    val = (X.index >= vs) & (X.index <= ve)
    return train, val


def run_horizon(dev: pd.DataFrame, h: int) -> pd.DataFrame:
    rows = []
    data = {s: make_xy(dev, h, s) for s in SETS}
    # every feature set must define the same validation origins (they differ only in warm-up)
    for fold in FOLDS:
        vals = [data[s][0].index[fold_slices(data[s][0], h, fold)[1]] for s in SETS]
        assert all(v.equals(vals[0]) for v in vals), "validation origins differ between sets"
    for fold in FOLDS:
        vs, _ = FOLDS[fold]
        X0, y0 = data["A"]
        _, val0 = fold_slices(X0, h, fold)
        origins, yv = X0.index[val0], y0[val0].to_numpy()
        base = baseline_predictions(dev, origins, h, dev[TARGET].loc[: vs - STEP].astype(float))
        for b in BASELINES:
            e = base[b].to_numpy() - yv
            rows.append(dict(h=h, fold=fold, fset="-", family="baseline", params=b,
                             mae=float(np.nanmean(np.abs(e))), rmse=float(np.sqrt(np.nanmean(e**2))),
                             n=int(np.isfinite(e).sum())))
        for s in SETS:
            X, y = data[s]
            tr, va = fold_slices(X, h, fold)
            for fam, p in all_configs():
                pred = fit_predict(fam, p, X[tr], y[tr], X[va])
                e = pred - y[va].to_numpy()
                rows.append(dict(h=h, fold=fold, fset=s, family=fam, params=json.dumps(p, sort_keys=True),
                                 mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e**2))),
                                 n=len(e)))
        log(f"h={h} fold={fold} done")
    return pd.DataFrame(rows)


def select(all_rows: pd.DataFrame) -> dict:
    sel = {}
    for h, g in all_rows.groupby("h"):
        mean = (g.groupby(["fset", "family", "params"])
                  .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), folds=("fold", "nunique"))
                  .reset_index())
        mean = mean[mean.folds == len(FOLDS)]
        base = mean[mean.family == "baseline"].sort_values("mae")
        learned = mean[(mean.family != "baseline") & (mean.fset.isin(["A", "B", "C"]))].sort_values("mae")
        lg = learned[learned.family == "lgbm"]
        per_fold = lambda r: g[(g.fset == r.fset) & (g.family == r.family) & (g.params == r.params)
                               ].set_index("fold")["mae"].round(3).to_dict()
        pick = lambda r: dict(fset=r.fset, family=r.family, params=json.loads(r.params),
                              dev_mae=round(float(r.mae), 4), dev_rmse=round(float(r.rmse), 4),
                              fold_mae=per_fold(r))
        b = base.iloc[0]
        sel[int(h)] = {
            "M_star": pick(learned.iloc[0]),
            "L_star": pick(lg.iloc[0]),
            "B_star": dict(name=b.params, dev_mae=round(float(b.mae), 4), dev_rmse=round(float(b.rmse), 4),
                           fold_mae=per_fold(b)),
            "baseline_dev_mae": {r.params: round(float(r.mae), 4) for r in base.itertuples()},
        }
    return sel


def controls(dev: pd.DataFrame, all_rows: pd.DataFrame, sel: dict) -> dict:
    out: dict = {}
    for h in HORIZONS:
        g = all_rows[(all_rows.h == h) & (all_rows.family != "baseline")]
        best_by_set = (g.groupby(["fset", "family", "params"]).mae.mean().reset_index()
                        .groupby("fset").mae.min().round(4).to_dict())
        fold_spread = (g[g.family == "lgbm"].groupby(["fset", "params"]).mae.mean().std())
        o = {"best_dev_mae_by_set": best_by_set}
        # (a) importance of the two random columns in a set-N LightGBM (last fold's training data)
        XN, yN = make_xy(dev, h, "N")
        tr, _ = fold_slices(XN, h, "V4")
        lp = sel[h]["L_star"]["params"]
        ens = LGBMEnsemble(lp).fit(XN[tr], yN[tr])
        imp = ens.importance_gain(XN.columns)
        o["rv_gain_share"] = round(float(imp[["rv1", "rv2"]].sum()), 4)
        o["rv_gain_share_if_uniform"] = round(2 / XN.shape[1], 4)
        # (b) block-shuffled training targets must not beat the slot-of-day median
        XA, yA = make_xy(dev, h, "A")
        rng = np.random.default_rng(0)
        shuf, ref = [], []
        for fold in FOLDS:
            tr, va = fold_slices(XA, h, fold)
            ytr = yA[tr].to_numpy().copy()
            nb = int(np.ceil(len(ytr) / 144))
            order = rng.permutation(nb)
            ytr_sh = np.concatenate([ytr[b * 144:(b + 1) * 144] for b in order])[: len(ytr)]
            pred = LGBMEnsemble({"n_estimators": 150, "num_leaves": 7}).fit(XA[tr], pd.Series(ytr_sh, index=XA.index[tr])).predict(XA[va])
            shuf.append(float(np.mean(np.abs(pred - yA[va].to_numpy()))))
            vs, _ = FOLDS[fold]
            base = baseline_predictions(dev, XA.index[va], h, dev[TARGET].loc[: vs - STEP].astype(float))
            ref.append(float(np.mean(np.abs(base["B5_slot_of_day_median"].to_numpy() - yA[va].to_numpy()))))
        o["shuffled_targets_mae_over_B5_mae"] = round(float(np.mean(shuf) / np.mean(ref)), 4)
        # (c) seed-to-seed spread for the L* configuration
        ls = sel[h]["L_star"]
        X, y = make_xy(dev, h, ls["fset"])
        per_seed = []
        for sd in (0, 1, 2):
            maes = []
            for fold in FOLDS:
                tr, va = fold_slices(X, h, fold)
                pred = LGBMEnsemble(ls["params"], seeds=(sd,)).fit(X[tr], y[tr]).predict(X[va])
                maes.append(float(np.mean(np.abs(pred - y[va].to_numpy()))))
            per_seed.append(float(np.mean(maes)))
        o["lstar_single_seed_dev_mae"] = [round(v, 4) for v in per_seed]
        o["lstar_ensemble_dev_mae"] = ls["dev_mae"]
        out[int(h)] = o
        log(f"controls h={h} done")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = load_dev()
    assert dev.index.max() <= pd.Timestamp("2016-04-22 23:50")
    parts = []
    for h in HORIZONS:
        f = OUT / f"fold_metrics_h{h}.csv"
        if f.exists():
            log(f"h={h}: reuse {f.name}")
        else:
            run_horizon(dev, h).to_csv(f, index=False)
        parts.append(pd.read_csv(f))
    all_rows = pd.concat(parts, ignore_index=True)
    all_rows.to_csv(OUT / "fold_metrics.csv", index=False)
    sel = select(all_rows)
    (OUT / "selection.json").write_text(json.dumps(sel, indent=2))
    log("selection written")
    (OUT / "controls.json").write_text(json.dumps(controls(dev, all_rows, sel), indent=2))
    log("all dev work done")


if __name__ == "__main__":
    main()
