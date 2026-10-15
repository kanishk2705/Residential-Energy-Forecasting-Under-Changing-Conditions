"""POST-HOC, EXPLORATORY diagnostics, run AFTER the final test was read.

These explain the headline E2 result (why extra inputs hurt at day-ahead). They use test
outcomes, so nothing here is used to select anything and everything is labelled
post-hoc in the write-up. Reads the full CSV directly (the one-time ledger already exists).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energyshift.config import DEV_END, FOLDS, RESULTS, STEP, TEST_END, TEST_START  # noqa: E402
from energyshift.data import _read  # noqa: E402
from energyshift.features import build_features, label_known  # noqa: E402
from energyshift.models import LGBMEnsemble  # noqa: E402
from energyshift.shift import domain_auc  # noqa: E402
from energyshift.stats import DayBlocks, summarize  # noqa: E402

GROUPS = {
    "lights": ["lights", "lights_mean6"],
    "indoor_temperature": ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9", "indoor_T_mean", "indoor_T_delta6"],
    "indoor_humidity": ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"],
    "airport_weather": ["T_out_node", "Press_mm_hg_node", "RH_out_node", "Windspeed_node", "Visibility_node",
                        "Tdewpoint_node", "T_out_node_d1h", "T_out_node_d6h", "T_out_mean24nodes"],
    "on_site_outdoor_sensor": ["T6", "RH_6", "indoor_minus_T6"],
}


def main():
    sel = json.loads((RESULTS / "dev" / "selection.json").read_text())
    df = _read()
    out: dict = {"note": "post-hoc exploratory; uses test outcomes; not used for any selection"}
    for h in (6, 144):
        params = sel[str(h)]["L_star"]["params"]
        XC = build_features(df, h, "C")
        y = df["Appliances"].astype(float).shift(-h)
        ok = XC.notna().all(axis=1) & y.notna()
        XC, y = XC.loc[ok], y.loc[ok]
        cols_A = list(build_features(df.iloc[:2000], h, "A").columns)
        for g, cs in GROUPS.items():
            missing = [c for c in cs if c not in XC.columns]
            assert not missing, missing
        variants = {"A (own history)": cols_A}
        for g, cs in GROUPS.items():
            variants[f"A + {g}"] = cols_A + cs
        variants["C (everything)"] = list(XC.columns.drop(["rv1", "rv2"], errors="ignore"))
        tr_final = label_known(XC.index, h, DEV_END)
        te = (XC.index >= TEST_START) & (XC.index <= TEST_END)
        db = DayBlocks(XC.index[te])
        res_h: dict = {}
        errA = None
        for name, cols in variants.items():
            pred = LGBMEnsemble(params).fit(XC.loc[tr_final, cols], y[tr_final]).predict(XC.loc[te, cols])
            err = np.abs(pred - y[te].to_numpy())
            if name.startswith("A (own"):
                errA = err
            dev_maes = []
            for f, (vs, ve) in FOLDS.items():
                trf = label_known(XC.index, h, vs - STEP)
                va = (XC.index >= vs) & (XC.index <= ve)
                p = LGBMEnsemble(params).fit(XC.loc[trf, cols], y[trf]).predict(XC.loc[va, cols])
                dev_maes.append(float(np.mean(np.abs(p - y[va].to_numpy()))))
            res_h[name] = {"test_mae": summarize(*db.ratio(err)), "dev_mean_fold_mae": float(np.mean(dev_maes)),
                           "n_features": len(cols), "_err": err}
        for name, r in res_h.items():
            r["test_minus_A"] = summarize(*db.diff(r.pop("_err"), errA))
        out[f"h{h}_group_ablation"] = res_h
        print(f"h={h} ablation done", flush=True)

    # input-only view of the shift: which inputs moved most, and how far outside the dev range?
    XC = build_features(df, 144, "C").dropna()
    dev_rows = XC.loc[XC.index <= DEV_END]
    test_rows = XC.loc[(XC.index >= TEST_START) & (XC.index <= TEST_END)]
    ref28 = dev_rows.loc[dev_rows.index > DEV_END - pd.Timedelta(days=28)]
    feats = [c for c in XC.columns if not c.startswith(("tod_", "dow"))]
    per_feat = []
    for c in feats:
        a = domain_auc(ref28[[c]], test_rows[[c]])
        lo, hi = dev_rows[c].min(), dev_rows[c].max()
        outside = float(((test_rows[c] < lo) | (test_rows[c] > hi)).mean())
        per_feat.append({"feature": c, "auc_last28dev_vs_test": a, "share_test_outside_dev_range": outside,
                         "dev_mean": float(dev_rows[c].mean()), "test_mean": float(test_rows[c].mean())})
    out["input_shift"] = sorted(per_feat, key=lambda r: -(r["auc_last28dev_vs_test"]
                                                         if not np.isnan(r["auc_last28dev_vs_test"]) else 0))
    (RESULTS / "final" / "diagnostics.json").write_text(json.dumps(out, indent=2, default=float))
    print("diagnostics written")


if __name__ == "__main__":
    main()
