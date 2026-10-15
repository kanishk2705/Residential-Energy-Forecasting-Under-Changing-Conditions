"""Download the UCI Appliances Energy Prediction CSV and verify it.

Usage:
    python scripts/get_data.py                 # try UCI, then the authors' GitHub copy
    python scripts/get_data.py --from-file P   # use a local copy of energydata_complete.csv

The raw file is deliberately not committed (see DATA.md).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "energydata_complete.csv"
EXPECTED_SHA256 = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
GITHUB_RAW = (
    "https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/"
    "e3e4c27a4ae2b41b21f84b4c9ac2d822d3d0ace1/energydata_complete.csv"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def structural_check(path: Path) -> list[str]:
    """Checks that do not need the exact bytes: row count, span, 10-minute grid."""
    problems = []
    d = pd.read_csv(path, usecols=["date"], parse_dates=["date"])["date"]
    if len(d) != 19735:
        problems.append(f"expected 19735 rows, found {len(d)}")
    if str(d.iloc[0]) != "2016-01-11 17:00:00" or str(d.iloc[-1]) != "2016-05-27 18:00:00":
        problems.append(f"unexpected span {d.iloc[0]} .. {d.iloc[-1]}")
    if not (d.diff().dropna() == pd.Timedelta(minutes=10)).all():
        problems.append("timestamps are not a gap-free 10-minute grid")
    return problems


def fetch_uci() -> bytes:
    with urllib.request.urlopen(UCI_ZIP, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    return z.read("energydata_complete.csv")


def fetch_github() -> bytes:
    with urllib.request.urlopen(GITHUB_RAW, timeout=60) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", type=Path)
    args = ap.parse_args()
    DEST.parent.mkdir(parents=True, exist_ok=True)

    if args.from_file:
        shutil.copyfile(args.from_file, DEST)
    else:
        for name, fn in (("UCI", fetch_uci), ("authors' GitHub copy", fetch_github)):
            try:
                DEST.write_bytes(fn())
                print(f"downloaded from {name}")
                break
            except Exception as e:  # network blocked, 403, etc.
                print(f"{name} failed: {e}", file=sys.stderr)
        else:
            print("could not download; pass --from-file", file=sys.stderr)
            return 1

    digest = sha256(DEST)
    if digest == EXPECTED_SHA256:
        print("sha256 OK")
    else:
        print(f"sha256 differs ({digest}); running structural check instead")
    problems = structural_check(DEST)
    if problems:
        print("STRUCTURAL PROBLEMS:", *problems, sep="\n  ")
        return 2
    print("structure OK: 19735 rows, gap-free 10-minute grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
