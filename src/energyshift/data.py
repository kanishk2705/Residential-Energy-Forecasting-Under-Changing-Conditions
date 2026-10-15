"""Loading with an enforced development / final-test boundary.

`load_dev` can never return a row after DEV_END. `load_full_for_final` returns the
whole file but only once: it writes a ledger (commit + protocol hash + time) and
refuses to run again if the ledger already exists. This is discipline, not
tamper-proofing; the git history is the audit trail.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DEV_END, RAW_CSV, ROOT


def _read(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df.apply(pd.to_numeric, errors="coerce")  # values like "  60" arrive as text
    if df.isna().any().any():
        raise ValueError("unexpected missing values after numeric parsing")
    if not df.index.is_monotonic_increasing:
        raise ValueError("timestamps not sorted")
    return df


def load_dev(path: Path = RAW_CSV) -> pd.DataFrame:
    """Development region only (rows <= DEV_END)."""
    df = _read(path).loc[:DEV_END].copy()
    assert df.index.max() <= DEV_END
    return df


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # not a git checkout
        return "unknown"


def load_full_for_final(ledger_path: Path, protocol_path: Path, path: Path = RAW_CSV) -> pd.DataFrame:
    """Whole file, exactly once. Writes the access ledger before returning."""
    ledger_path = Path(ledger_path)
    if ledger_path.exists():
        raise RuntimeError(
            f"final-test ledger already exists at {ledger_path}; the test period has "
            "already been read once. Delete it only as a logged protocol deviation."
        )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = {
        "read_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "protocol_sha256": hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    ledger_path.write_text(json.dumps(ledger, indent=2))
    return _read(path)
