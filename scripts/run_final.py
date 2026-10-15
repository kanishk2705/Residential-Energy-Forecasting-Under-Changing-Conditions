"""Final-test run (PROTOCOL.md section 8). Reads the test period exactly once.

Refuses to run unless results/dev/selection.json is identical to the copy committed on
branch `frozen/selection`. Writes results/final/results.json and per-horizon predictions.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm  # noqa: E402
import sklearn  # noqa: E402

from energyshift import intervals as iv  # noqa: E402
from energyshift.baselines import BASELINES, baseline_predictions  # noqa: E402
from energyshift.config import (CAL_START, DEV_END, HORIZONS, RESULTS, ROOT, STEP, TARGET,  # noqa: E402
                                TEST_END, TEST_START)
from energyshift.data import load_full_for_final  # noqa: E402
from energyshift.experiments import run_intervals, run_policies  # noqa: E402
from energyshift.features import label_known, make_xy  # noqa: E402
from energyshift.models import all_configs, fit_predict  # noqa: E402
from energyshift.policies import PolicyData, shift_statistic  # noqa: E402
from energyshift.shift import domain_auc  # noqa: E402
from energyshift.stats import DayBlocks, ci, moving_block_reps, spearman_day_ci, summarize  # noqa: E402

OUT = RESULTS / "final"
SEL_PATH = RESULTS / "dev" / "selection.json"
LEDGER = OUT / "test_access.json"
PROTOCOL = ROOT / "PROTOCOL.md"
SERVE_END = TEST_END + STEP  # exclusive


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def require_frozen_selection() -> dict:
    if not SEL_PATH.exists():
        raise SystemExit("results/dev/selection.json missing")
    r = subprocess.run(["git", "show", "frozen/selection:results/dev/selection.json"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("branch frozen/selection with results/dev/selection.json not found")
    if r.stdout != SEL_PATH.read_text():
        raise SystemExit("selection.json differs from the frozen copy")
    return json.loads(SEL_PATH.read_text())


def mae_stat(db: DayBlocks, err: np.ndarray, level=0.95) -> dict:
    p, reps = db.ratio(np.abs(err))
    return summarize(p, reps, level)


def rmse_stat(db: DayBlocks, err: np.ndarray) -> dict:
    p, reps = db.ratio(err**2)
    return summarize(float(np.sqrt(p)), np.sqrt(reps))


def paired(db: DayBlocks, a: np.ndarray, b: np.ndarray, level=0.95) -> dict:
    """MAE(a) - MAE(b) with a paired day-block CI (a, b are signed errors)."""
    p, reps = db.diff(np.abs(a), np.abs(b))
    return summarize(p, reps, level)


def week_index(idx: pd.DatetimeIndex) -> np.ndarray:
    return ((idx.normalize() - TEST_START.normalize()).days // 7).to_numpy()


def main():
    global OUT, LEDGER
    csv = None
    if len(sys.argv) > 2 and sys.argv[1] == "--smoke":
        # Synthetic-data dry run: never touches the real test period or the real ledger.
        smoke = Path(sys.argv[2])
        OUT, LEDGER, csv = smoke / "out", smoke / "out" / "test_access.json", smoke / "energydata_complete.csv"
        sel = json.loads((smoke / "selection.json").read_text())
        from energyshift import models
        models.GRID = {"ridge": [{"alpha": 10.0}], "rf": [], "lgbm": [{"n_estimators": 20, "num_leaves": 7}]}
    else:
        sel = require_frozen_selection()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_full_for_final(LEDGER, PROTOCOL, *( [csv] if csv else [] ))
    assert df.index.max() == TEST_END
    res: dict = {"meta": {
        "git_commit": json.loads(LEDGER.read_text())["git_commit"],
        "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
        "test_start": str(TEST_START), "test_end": str(TEST_END),
    }, "selection": sel, "horizons": {}}

    for h in HORIZONS:
        s = sel[str(h)]
        log(f"=== h={h} ===")
        H: dict = {}
        data = {f: make_xy(df, h, f) for f in ("A", "B", "C")}
        XA, yA = data["A"]
        tmask = (XA.index >= TEST_START) & (XA.index <= TEST_END)
        origins = XA.index[tmask]
        y = yA[tmask].to_numpy()
        db = DayBlocks(origins)
        H["n_origins"], H["n_days"] = int(len(origins)), int(db.D)

        # ---------------- E1: models and baselines trained on development labels only
        preds: dict[str, np.ndarray] = {}
        base = baseline_predictions(df, origins, h, df[TARGET].loc[:DEV_END].astype(float))
        for b in BASELINES:
            preds[b] = base[b].to_numpy()
        for f in ("A", "B", "C"):
            X, yy = data[f]
            tr = label_known(X.index, h, DEV_END)
            te = (X.index >= TEST_START) & (X.index <= TEST_END)
            assert X.index[te].equals(origins)
            for fam, p in all_configs():
                preds[f"{fam}|{f}|{json.dumps(p, sort_keys=True)}"] = fit_predict(fam, p, X[tr], yy[tr], X[te])
            log(f"h={h} grid fitted for set {f}")
        name = lambda m: f"{m['family']}|{m['fset']}|{json.dumps(m['params'], sort_keys=True)}"
        M, L, Bn = name(s["M_star"]), name(s["L_star"]), s["B_star"]["name"]
        err = {k: v - y for k, v in preds.items()}
        H["names"] = {"M_star": M, "L_star": L, "B_star": Bn}
        H["methods"] = {k: {"mae": mae_stat(db, e), "rmse": rmse_stat(db, e)} for k, e in err.items()}
        pt, reps = db.diff(np.abs(err[M]), np.abs(err[Bn]))
        sk_p = 1 - db.ratio(np.abs(err[M]))[0] / db.ratio(np.abs(err[Bn]))[0]
        sk_reps = 1 - db.ratio(np.abs(err[M]))[1] / db.ratio(np.abs(err[Bn]))[1]
        H["E1_headline"] = {
            "delta_mae_M_minus_B": {**summarize(pt, reps), "ci98_33": list(ci(reps, 1 - 0.05 / 3))},
            "skill_vs_B": summarize(sk_p, sk_reps),
            "delta_mae_L_minus_B": summarize(*db.diff(np.abs(err[L]), np.abs(err[Bn]))),
            "delta_rmse_M_minus_B": None,
        }
        pm, rm = db.ratio(err[M] ** 2); pb, rb = db.ratio(err[Bn] ** 2)
        H["E1_headline"]["delta_rmse_M_minus_B"] = summarize(np.sqrt(pm) - np.sqrt(pb), np.sqrt(rm) - np.sqrt(rb))
        mb = moving_block_reps(np.abs(err[M]), np.abs(err[Bn]), block=288)
        H["E1_headline"]["delta_mae_M_minus_B_moving_block288"] = {
            "estimate": pt, "ci_lo": float(np.quantile(mb, 0.025)), "ci_hi": float(np.quantile(mb, 0.975))}
        log(f"h={h} E1 done: dMAE={pt:.2f}")

        # ---------------- E2: input groups with L*'s hyper-parameters
        lp = json.dumps(s["L_star"]["params"], sort_keys=True)
        e2 = {f: err[f"lgbm|{f}|{lp}"] for f in ("A", "B", "C")}
        H["E2_input_groups"] = {
            "mae": {f: mae_stat(db, e) for f, e in e2.items()},
            "B_minus_A": summarize(*db.diff(np.abs(e2["B"]), np.abs(e2["A"]))),
            "C_minus_B": summarize(*db.diff(np.abs(e2["C"]), np.abs(e2["B"]))),
            "C_minus_A": summarize(*db.diff(np.abs(e2["C"]), np.abs(e2["A"]))),
        }

        # ---------------- E3: retraining policies with L*
        pdx = PolicyData.build(df, h, s["L_star"]["fset"])
        pol = run_policies(pdx, s["L_star"]["params"], TEST_START, SERVE_END, DEV_END)
        assert pol["P0_frozen"]["pred"].index.equals(origins), "policy origins differ from E1 origins"
        perr = {k: v["pred"].to_numpy() - y for k, v in pol.items()}
        H["E3_policies"] = {}
        for k, v in pol.items():
            H["E3_policies"][k] = {
                "retrains": v["retrains"], "retrain_times": v.get("retrain_times"),
                "mae": mae_stat(db, perr[k]),
                "delta_mae_vs_P0": summarize(*db.diff(np.abs(perr[k]), np.abs(perr["P0_frozen"]))),
            }
        np.testing.assert_allclose(perr["P0_frozen"], err[L], atol=1e-9)  # P0 == frozen L* (consistency)
        H["P3_daily_shift_auc"] = {str(k.date()): (None if np.isnan(v) else float(v))
                                   for k, v in pol["P3_drift_triggered"]["daily_stat"].items()}
        log(f"h={h} E3 done")

        # ---------------- E4: intervals
        out = run_intervals(pdx, s["L_star"]["params"], TEST_START, TEST_END, CAL_START, DEV_END)
        wk = week_index(origins)
        H["E4_intervals"] = {}
        for k, d in out.items():
            assert d.index.equals(origins)
            m = iv.interval_metrics(d)
            H["E4_intervals"][k] = {
                "coverage": summarize(*db.ratio(m["coverage"])),
                "width": summarize(*db.ratio(m["width"])),
                "interval_score": summarize(*db.ratio(m["interval_score"])),
                "coverage_by_week": {int(w): float(m["coverage"][wk == w].mean()) for w in np.unique(wk)},
                "alpha_final": float(d["alpha_t"].iloc[-1]) if "alpha_t" in d else None,
            }
        log(f"h={h} E4 done")

        # ---------------- E5: shift description (exploratory)
        ref28 = pdx.X.index[(pdx.X.index > DEV_END - pd.Timedelta(days=28)) & (pdx.X.index <= DEV_END)]
        t_node = df["T_out"][df.index.minute == 0]
        dev_T28 = float(t_node.loc[DEV_END - pd.Timedelta(days=28): DEV_END].mean())
        dev_T = float(t_node.loc[:DEV_END].mean())
        dev_y28 = float(df[TARGET].loc[DEV_END - pd.Timedelta(days=28): DEV_END].mean())
        weeks = []
        for w in np.unique(wk):
            o = origins[wk == w]
            ys = y[wk == w]
            ew = lambda e: float(np.mean(np.abs(e[wk == w])))
            weeks.append({
                "week": int(w), "start": str(o.min().date()), "end": str(o.max().date()),
                "n_origins": int(len(o)),
                "mean_T_out_node": float(t_node.loc[o.min().normalize(): o.max()].mean()),
                "mean_T_out_last28_dev": dev_T28, "mean_T_out_dev": dev_T,
                "mean_appliances": float(ys.mean()), "mean_appliances_last28_dev": dev_y28,
                "shift_auc_vs_last28_dev": float(domain_auc(pdx.X.loc[ref28], pdx.X.loc[o])),
                "mae_B_star": ew(err[Bn]), "mae_L_star_P0": ew(perr["P0_frozen"]),
                "coverage_I0": H["E4_intervals"]["I0_raw"]["coverage_by_week"][int(w)],
                "coverage_I2": H["E4_intervals"]["I2_rolling_cqr"]["coverage_by_week"][int(w)],
                "coverage_I3": H["E4_intervals"]["I3_aci"]["coverage_by_week"][int(w)],
            })
        H["E5_weekly"] = weeks
        days = pd.DatetimeIndex(sorted(set(origins.normalize())))
        dayrows = []
        for d in days:
            msk = np.asarray(origins.normalize() == d)
            auc = shift_statistic(pdx, d)
            dayrows.append({"day": str(d.date()), "shift_auc_monitor": None if np.isnan(auc) else float(auc),
                            "mae_L_P0": float(np.mean(np.abs(perr["P0_frozen"][msk]))),
                            "mae_B_star": float(np.mean(np.abs(err[Bn][msk])))})
        dd = pd.DataFrame(dayrows).dropna()
        dd["mae_ratio_L_over_B"] = dd["mae_L_P0"] / dd["mae_B_star"]
        H["E5_daily"] = dd.to_dict(orient="records")
        H["E5_spearman_exploratory"] = {
            "shift_vs_mae_ratio_L_over_B": spearman_day_ci(dd["shift_auc_monitor"], dd["mae_ratio_L_over_B"]),
            "shift_vs_mae_L": spearman_day_ci(dd["shift_auc_monitor"], dd["mae_L_P0"]),
        }

        # ---------------- predictions for figures and audit
        frame = pd.DataFrame({"y": y, "B_star": preds[Bn], "M_star": preds[M], "L_star": preds[L]}, index=origins)
        for k, v in pol.items():
            frame[k] = v["pred"].to_numpy()
        for k, d in out.items():
            frame[f"{k}_lo"], frame[f"{k}_hi"] = d["lo"].to_numpy(), d["hi"].to_numpy()
        frame.to_csv(OUT / f"predictions_h{h}.csv", index_label="origin")
        res["horizons"][str(h)] = H
        log(f"h={h} complete")

    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    log("final results written")


if __name__ == "__main__":
    main()
