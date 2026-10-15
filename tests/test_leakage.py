"""Leakage tests required by PROTOCOL.md section 3."""
import numpy as np
import pandas as pd
import pytest

from energyshift.config import SPD, STEP, WEATHER_NODE
from energyshift.features import (
    FEATURE_SETS,
    build_features,
    calendar_features,
    hourly_node,
    label_known,
    make_xy,
)


@pytest.mark.parametrize("fset", FEATURE_SETS)
@pytest.mark.parametrize("h", [1, 6, 144])
def test_features_ignore_all_future_rows(synth, fset, h):
    """Corrupt every row after origin i; features at i must not move."""
    rng = np.random.default_rng(1)
    full = build_features(synth, h, fset)
    origins = rng.choice(np.arange(400, len(synth) - 5), size=6, replace=False)
    for i in origins:
        corrupt = synth.copy()
        corrupt.iloc[i + 1 :] = rng.normal(1e6, 1e5, size=corrupt.iloc[i + 1 :].shape)
        got = build_features(corrupt, h, fset).iloc[i]
        want = full.iloc[i]
        pd.testing.assert_series_equal(got, want, check_names=False)


@pytest.mark.parametrize("h", [1, 6, 144])
def test_weather_features_use_only_real_hourly_nodes(synth, h):
    """Interpolated airport-weather rows (minute != 0) must not influence any feature."""
    base = build_features(synth, h, "C")
    corrupt = synth.copy()
    off_node = corrupt.index.minute != 0
    for c in WEATHER_NODE:
        corrupt.loc[off_node, c] = 9999.0
    pd.testing.assert_frame_equal(build_features(corrupt, h, "C"), base)


def test_hourly_node_is_last_real_observation(synth):
    s = synth["T_out"]
    node = hourly_node(s)
    i = np.where(s.index == pd.Timestamp("2016-03-05 14:40"))[0][0]
    assert node.iloc[i] == s.loc["2016-03-05 14:00"]
    j = np.where(s.index == pd.Timestamp("2016-03-05 15:00"))[0][0]
    assert node.iloc[j] == s.iloc[j]


@pytest.mark.parametrize("h", [1, 6, 144])
def test_target_alignment(synth, h):
    X, y = make_xy(synth, h, "A")
    t = y.index[10]
    assert y.loc[t] == synth["Appliances"].loc[t + h * STEP]
    # nothing in X for the last h rows (their target does not exist)
    assert X.index.max() <= synth.index[-1] - h * STEP


def test_calendar_features_describe_target_time(synth):
    h = 6
    cal = calendar_features(synth.index, h)
    t = pd.Timestamp("2016-03-05 14:40")
    tau = t + h * STEP
    slot = (tau.hour * 60 + tau.minute) // 10
    assert cal.loc[t, "tod_sin1"] == pytest.approx(np.sin(2 * np.pi * slot / SPD))
    assert cal.loc[t, f"dow{tau.dayofweek}"] == 1.0
    assert cal.loc[t, [f"dow{d}" for d in range(7)]].sum() == 1.0


def test_seasonal_lag_is_never_in_the_future(synth):
    for h in (1, 6, 144):
        f = build_features(synth, h, "A")
        y = synth["Appliances"]
        i = 500
        assert f["y_yday"].iloc[i] == y.iloc[i + h - SPD]
        assert i + h - SPD <= i


def test_label_known_boundary():
    idx = pd.date_range("2016-04-20 00:00", periods=5, freq="10min")
    last = pd.Timestamp("2016-04-20 00:30")
    # origins 00:00..00:40; targets (h=1) 00:10..00:50; labels known up to 00:30
    assert label_known(idx, 1, last).tolist() == [True, True, True, False, False]
    # h=3: targets 00:30..01:10
    assert label_known(idx, 3, last).tolist() == [True, False, False, False, False]


def test_horizon_bounds(synth):
    with pytest.raises(ValueError):
        build_features(synth, 145, "A")
    with pytest.raises(ValueError):
        build_features(synth, 0, "A")
