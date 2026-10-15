import json

import pandas as pd
import pytest

from energyshift import data
from energyshift.config import DEV_END


@pytest.fixture()
def csv_across_boundary(tmp_path, synth):
    idx = pd.date_range("2016-04-20 00:00", periods=8 * 144, freq="10min")
    df = synth.iloc[: len(idx)].copy()
    df.index = idx
    df.index.name = "date"
    p = tmp_path / "energy.csv"
    df.to_csv(p)
    return p


def test_load_dev_never_returns_test_rows(csv_across_boundary):
    dev = data.load_dev(csv_across_boundary)
    assert dev.index.max() <= DEV_END
    full = data._read(csv_across_boundary)
    assert full.index.max() > DEV_END


def test_final_test_can_be_read_only_once(csv_across_boundary, tmp_path):
    protocol = tmp_path / "PROTOCOL.md"
    protocol.write_text("frozen")
    ledger = tmp_path / "final" / "test_access.json"
    df = data.load_full_for_final(ledger, protocol, csv_across_boundary)
    assert df.index.max() > DEV_END
    rec = json.loads(ledger.read_text())
    assert {"read_at_utc", "git_commit", "protocol_sha256", "csv_sha256"} <= set(rec)
    with pytest.raises(RuntimeError):
        data.load_full_for_final(ledger, protocol, csv_across_boundary)
