"""Alignment must recover the car frame and be independent of phone orientation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocess import align, random_rotation, resample, rotate, to_si  # noqa: E402

BYD = next(Path(__file__).resolve().parents[1].glob("data/raw/events_byd/*/data/processed/imu_samples_clean.csv"), None)
pytestmark = pytest.mark.skipif(BYD is None, reason="run scripts/fetch_data.sh first")


def _session():
    b = pd.read_csv(BYD)
    x = b[b.session_id == "harsh_braking_20260207_090906"]
    d = x.rename(columns={"t_seconds": "t", "acc_x": "ax", "acc_y": "ay", "acc_z": "az",
                          "gyro_x": "gx", "gyro_y": "gy", "gyro_z": "gz"})
    return resample(to_si(d[["t", "ax", "ay", "az", "gx", "gy", "gz", "event_class"]]))


def test_recovers_documented_mounting():
    # BYD docs: +X = vehicle front, +Y = left, +Z = up
    a, info = align(_session())
    assert info["forward"][0] > 0.95 and info["left"][1] > 0.95
    assert a.lon[a.event_class == 1].mean() < -2      # harsh braking decelerates


@pytest.mark.parametrize("seed", range(10))
def test_rotation_invariant(seed):
    d = _session()
    a, _ = align(d)
    r, _ = align(rotate(d, random_rotation(seed)))
    for c in ("lon", "lat", "yaw_rate"):
        assert np.corrcoef(a[c], r[c])[0, 1] > 0.999
