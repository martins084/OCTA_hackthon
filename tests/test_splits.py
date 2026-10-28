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
    lambda: splits.event_dataset("ferreira"),
    lambda: splits.assert_path_allowed(ROOT / "data/raw/blind_ferreira/data"),
    lambda: splits.assert_path_allowed(ROOT / "data/raw/blind_own/x"),
])
def test_blind_access_blocked(call):
    with pytest.raises(splits.BlindDataError):
        call()


def test_gate_closes_after_use():
    with splits.blind_access():
        splits.event_dataset("ferreira")
    with pytest.raises(splits.BlindDataError):
        splits.event_dataset("ferreira")
