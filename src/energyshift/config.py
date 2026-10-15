"""Constants frozen in PROTOCOL.md. Change nothing here without a deviation-log entry."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "energydata_complete.csv"
RESULTS = ROOT / "results"

STEP = pd.Timedelta(minutes=10)
SPD = 144  # rows per day
SPW = 1008  # rows per week

DEV_END = pd.Timestamp("2016-04-22 23:50")
TEST_START = pd.Timestamp("2016-04-25 00:00")
TEST_END = pd.Timestamp("2016-05-27 18:00")

HORIZONS = (1, 6, 144)

# Rolling-origin development folds: validation origins (inclusive).
FOLDS = {
    "V1": (pd.Timestamp("2016-02-29 00:00"), pd.Timestamp("2016-03-13 23:50")),
    "V2": (pd.Timestamp("2016-03-14 00:00"), pd.Timestamp("2016-03-27 23:50")),
    "V3": (pd.Timestamp("2016-03-28 00:00"), pd.Timestamp("2016-04-10 23:50")),
    "V4": (pd.Timestamp("2016-04-11 00:00"), pd.Timestamp("2016-04-22 23:50")),
}

# Interval study: the last 14 development days are the calibration set.
CAL_START = DEV_END.normalize() - pd.Timedelta(days=13)  # 2016-04-09 00:00
CAL_DAYS = 14

TARGET = "Appliances"
INDOOR_T = ["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]
INDOOR_RH = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]
WEATHER_NODE = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
RANDOM_COLS = ["rv1", "rv2"]

BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20261015
