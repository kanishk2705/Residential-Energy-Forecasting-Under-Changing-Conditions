import numpy as np
import pandas as pd
import pytest

from energyshift import intervals as iv
from energyshift.baselines import baseline_predictions, slot_medians
from energyshift.config import SPD, STEP
from energyshift.features import build_features
from energyshift.policies import (PolicyData, _train_index, fit_at, run_schedule,
                                  schedule_at, shift_statistic)
from energyshift.shift import domain_auc
from energyshift.stats import DayBlocks, ci, moving_block_reps


# ---------- bootstrap ----------
def test_dayblocks_point_equals_mean_and_ci_brackets_it():
    idx = pd.date_range("2016-04-25", periods=20 * 144, freq="10min")
    rng = np.random.default_rng(0)
    v = rng.normal(5, 2, len(idx))
    db = DayBlocks(idx, B=2000, seed=1)
    p, reps = db.ratio(v)
    assert p == pytest.approx(v.mean())
    lo, hi = ci(reps)
    assert lo < p < hi


def test_paired_difference_of_identical_series_is_zero():
    idx = pd.date_range("2016-04-25", periods=10 * 144, freq="10min")
    v = np.random.default_rng(0).normal(size=len(idx))
    d, reps = DayBlocks(idx, B=500).diff(v, v)
    assert d == 0 and np.allclose(reps, 0)


def test_dayblocks_ci_has_roughly_nominal_coverage():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2016-04-25", periods=30 * 144, freq="10min")
    hits, trials = 0, 150
    for t in range(trials):
        day_effect = np.repeat(rng.normal(0, 1, 30), 144)
        v = day_effect + rng.normal(0, 0.5, len(idx))  # day-level dependence
        _, reps = DayBlocks(idx, B=400, seed=t).ratio(v)
        lo, hi = ci(reps)
        hits += lo <= 0 <= hi
    assert 0.85 <= hits / trials <= 1.0


def test_moving_block_shape():
    reps = moving_block_reps(np.arange(1000.0), None, block=100, B=50)
    assert reps.shape == (50,)


# ---------- baselines ----------
def test_baselines_use_only_the_past(synth):
    h = 6
    origins = synth.index[1200:1210]
    y_train = synth["Appliances"].iloc[:1000]
    ref = baseline_predictions(synth, origins, h, y_train)
    corrupt = synth.copy()
    corrupt.iloc[1210:] = 1e6  # everything after the last origin
    got = baseline_predictions(corrupt, origins, h, y_train)
    pd.testing.assert_frame_equal(ref, got)


def test_slot_medians_shape(synth):
    d, w = slot_medians(synth["Appliances"])
    assert d.shape == (SPD,) and w.shape == (1008,)
    assert not np.isnan(d).any() and not np.isnan(w).any()


# ---------- policies ----------
def test_training_set_never_contains_labels_after_label_end(synth):
    pdx = PolicyData.build(synth, 6, "A")
    label_end = synth.index[2000]
    tr = _train_index(pdx, label_end, None)
    assert (tr + 6 * STEP).max() <= label_end
    tr28 = _train_index(pdx, label_end, 5)
    assert tr28.min() > label_end - pd.Timedelta(days=5)


def test_schedule_trains_with_strictly_earlier_labels(synth):
    t = synth.index[1500]
    assert schedule_at([t]) == [(t, t - STEP)]


def test_policy_predictions_ignore_labels_after_label_end(synth):
    """Changing y after the label end must not change a model trained at that end."""
    params = {"n_estimators": 20, "num_leaves": 7}
    h = 6
    base = PolicyData.build(synth, h, "A")
    label_end = synth.index[2500]
    p1 = fit_at(base, params, label_end).predict(base.X.iloc[3000:3010])
    corrupt = synth.copy()
    corrupt.iloc[2501 + 0 :, corrupt.columns.get_loc("Appliances")] = 1e5
    # features of early rows change only via rows <= origin; rows used for training have
    # origin + h <= label_end, so Appliances after label_end never enters training labels,
    # but features at origins near label_end read rows up to the origin only.
    c = PolicyData.build(corrupt, h, "A")
    tr = _train_index(c, label_end, None)
    assert (tr + h * STEP).max() <= label_end
    X_tr_base = base.X.loc[tr]
    X_tr_c = c.X.loc[tr]
    pd.testing.assert_frame_equal(X_tr_base, X_tr_c)
    pd.testing.assert_series_equal(base.y.loc[tr], c.y.loc[tr])
    assert np.isfinite(p1).all()


def test_run_schedule_segments_are_served_by_their_own_model(synth):
    params = {"n_estimators": 20, "num_leaves": 7}
    pdx = PolicyData.build(synth, 6, "A")
    t0, t1 = synth.index[2000], synth.index[2300]
    sched = schedule_at([t0, t1])
    preds = run_schedule(pdx, params, t0, synth.index[2600], sched)
    assert preds.index.min() == t0 and preds.index.max() < synth.index[2600]
    m0 = fit_at(pdx, params, t0 - STEP)
    seg0 = preds.loc[t0 : t1 - STEP]
    np.testing.assert_allclose(seg0.to_numpy(), m0.predict(pdx.X.loc[seg0.index]))


# ---------- shift statistic ----------
def test_domain_auc_near_half_without_shift_and_high_with_shift():
    rng = np.random.default_rng(0)
    idx1 = pd.date_range("2016-03-01", periods=28 * 144, freq="10min")
    idx2 = pd.date_range("2016-03-29", periods=7 * 144, freq="10min")
    ref = pd.DataFrame({"a": rng.normal(size=len(idx1)), "b": rng.normal(size=len(idx1))}, index=idx1)
    same = pd.DataFrame({"a": rng.normal(size=len(idx2)), "b": rng.normal(size=len(idx2))}, index=idx2)
    moved = same.assign(a=same["a"] + 3)
    assert 0.35 < domain_auc(ref, same) < 0.65
    assert domain_auc(ref, moved) > 0.9


# ---------- intervals ----------
def test_conformal_quantile_finite_sample_level():
    s = np.arange(1.0, 11.0)
    assert iv.conformal_q(s, 0.2) == 9.0 or iv.conformal_q(s, 0.2) == 10.0
    assert iv.conformal_q(np.array([]), 0.2) == float("inf")


def _frame(n=6 * SPD, h=6, seed=0, bias=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-04-19", periods=n, freq="10min")
    y = rng.normal(100, 20, n)
    lo = np.full(n, 90.0) + bias
    hi = np.full(n, 110.0) + bias
    return iv.interval_frame(idx, lo, hi, y, h)


def test_intervals_do_not_read_outcomes_that_have_not_happened():
    h = 6
    f = _frame(h=h)
    test_start = pd.Timestamp("2016-04-22 00:00")
    args = dict(test_start=test_start, cal_start=pd.Timestamp("2016-04-19 12:00"),
                cal_end=pd.Timestamp("2016-04-21 23:50"), test_end=f.index.max())
    ref = iv.build_variants(f, **args)
    k = 400
    t = ref["I2_rolling_cqr"].index[k]
    g = f.copy()
    future = g["target_time"] > t
    g.loc[future, "y"] = 1e6
    g = iv.interval_frame(g.index, g["lo"].to_numpy(), g["hi"].to_numpy(), g["y"].to_numpy(), h)
    got = iv.build_variants(g, **args)
    for name in ("I1_static_cqr", "I2_rolling_cqr", "I3_aci"):
        a = ref[name].loc[t, ["lo", "hi"]]
        b = got[name].loc[t, ["lo", "hi"]]
        pd.testing.assert_series_equal(a, b)


def test_aci_pulls_coverage_toward_nominal_when_raw_intervals_are_too_narrow():
    h = 6
    f = _frame(n=20 * SPD, h=h, seed=2)
    f = iv.interval_frame(f.index, np.full(len(f), 98.0), np.full(len(f), 102.0), f["y"].to_numpy(), h)
    out = iv.build_variants(
        f, test_start=pd.Timestamp("2016-04-25"), cal_start=pd.Timestamp("2016-04-19 12:00"),
        cal_end=pd.Timestamp("2016-04-24 23:50"), test_end=f.index.max())
    cov = {k: iv.interval_metrics(v)["coverage"].mean() for k, v in out.items()}
    assert cov["I0_raw"] < 0.3  # deliberately far too narrow
    assert abs(cov["I3_aci"] - 0.8) < 0.1 and abs(cov["I2_rolling_cqr"] - 0.8) < 0.1


# ---------- the leakage test must be able to fail ----------
def test_leakage_detector_would_catch_a_leaky_feature(synth):
    """Mutation check: a feature that reads y[i+1] must make the future-corruption test fail."""
    def leaky(df):
        f = build_features(df, 1, "A")
        f["leak"] = df["Appliances"].shift(-1)
        return f
    i = 600
    base = leaky(synth).iloc[i]
    corrupt = synth.copy()
    corrupt.iloc[i + 1 :] = 1e6
    assert not leaky(corrupt).iloc[i].equals(base)


def test_shift_statistic_uses_only_inputs_before_T(synth):
    """The monitor at time T reads origins < T only and ignores every row after T."""
    pdx = PolicyData.build(synth, 6, "C")
    T = synth.index[3000]
    base = shift_statistic(pdx, T)
    corrupt = synth.copy()
    corrupt.iloc[3000:] = np.random.default_rng(0).normal(1e5, 1e4, size=corrupt.iloc[3000:].shape)
    pdx2 = PolicyData.build(corrupt, 6, "C")
    assert shift_statistic(pdx2, T) == base
    assert np.isnan(shift_statistic(pdx, synth.index[100]))  # not enough history -> no signal


def test_schedule_with_zero_length_first_segment(synth):
    params = {"n_estimators": 10, "num_leaves": 7}
    pdx = PolicyData.build(synth, 6, "A")
    t0 = synth.index[2000]
    sched = [(t0, t0 - 10 * STEP), (t0, t0 - STEP)]  # second model starts where the first would
    preds = run_schedule(pdx, params, t0, synth.index[2200], sched)
    assert preds.index.min() == t0 and len(preds) > 0


def test_segment_without_observed_targets_is_empty_not_an_error(synth):
    from energyshift.policies import predict_segment
    params = {"n_estimators": 10, "num_leaves": 7}
    pdx = PolicyData.build(synth, 144, "A")
    model = fit_at(pdx, params, synth.index[2000])
    end = synth.index[-1] + STEP
    out = predict_segment(pdx, model, end - pd.Timedelta(hours=1), end)
    assert len(out) == 0
