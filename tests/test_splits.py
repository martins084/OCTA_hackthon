"""The blind sets must stay unreachable from dev code."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import splits  # noqa: E402


def test_dev_trips_exclude_blind_driver():
    dev = splits.uah_trip_dirs(ROOT / "data/raw/uah", "dev")
    drivers = {d.parent.name for d in dev}
    assert drivers.isdisjoint(splits.SPLITS["uah"]["blind_drivers"])


@pytest.mark.parametrize("call", [
    lambda: splits.uah_trip_dirs(ROOT / "data/raw/uah", "blind"),
    lambda: splits.pvs_trip_dirs("blind"),
    lambda: splits.assert_path_allowed(ROOT / "data/raw/blind_pvs/PVS_7"),
    lambda: splits.assert_path_allowed(ROOT / "data/raw/blind_uah/D3"),
    lambda: splits.assert_path_allowed(ROOT / "data/raw/blind_own/x"),
])
def test_blind_access_blocked(call):
    with pytest.raises(splits.BlindDataError):
        call()


def test_gate_closes_after_use():
    with splits.blind_access():
        splits.pvs_trip_dirs("blind")
    with pytest.raises(splits.BlindDataError):
        splits.pvs_trip_dirs("blind")


def test_dev_pvs_excludes_blind():
    names = {p.name for p in splits.pvs_trip_dirs("dev")}
    assert names.isdisjoint(splits.SPLITS["pvs"]["blind_trips"])


def test_grab_blind_is_sealed_and_stable():
    import pandas as pd
    df = pd.DataFrame({"bookingID": range(10_000)})
    dev = splits.grab_split(df)
    assert 0.17 < 1 - len(dev) / len(df) < 0.23                 # ~20% sealed
    assert splits.grab_is_blind([123456])[0] == splits.grab_is_blind([123456])[0]
    with pytest.raises(splits.BlindDataError):
        splits.grab_split(df, split="blind")
