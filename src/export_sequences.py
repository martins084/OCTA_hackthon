"""Export UAH dev-driver trips as fixed-length 10 Hz sequences for the GPU sequence model.

Only dev drivers (src/splits.py), so the blind driver never reaches Kaggle in trainable form.
Output: data/processed/uah_dev_seq.npz  (X: [n, T, C], label, driver, trip)
"""
from pathlib import Path

import numpy as np

from driving_features import resample_trip
from splits import uah_trip_dirs
from train_driving import CLASSES, read_uah_trip

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["speed", "over_limit", "lon_g", "lat_g", "acc_mag", "jerk_mag"]
WIN, STEP = 600, 150          # 60 s windows, 15 s step at 10 Hz (same as the GBM)


def main():
    X, y, drv, trip = [], [], [], []
    for folder in uah_trip_dirs(ROOT / "data/raw/uah", "dev"):
        parts = folder.name.split("-")
        d = resample_trip(*read_uah_trip(folder))
        d["over_limit"] = (d.speed - d.speed_limit).fillna(0)
        arr = d[CHANNELS].to_numpy(np.float32)
        for s in range(0, len(arr) - WIN + 1, STEP):
            if (d.speed.iloc[s:s + WIN] > 5).mean() < 0.5:
                continue
            X.append(arr[s:s + WIN])
            y.append(CLASSES.index(parts[3].rstrip("0123456789")))
            drv.append(parts[2])
            trip.append(folder.name)
    out = ROOT / "data/processed/uah_dev_seq.npz"
    np.savez_compressed(out, X=np.stack(X), y=np.array(y), driver=np.array(drv), trip=np.array(trip),
                        channels=np.array(CHANNELS), classes=np.array(CLASSES))
    print(out, np.stack(X).shape)


if __name__ == "__main__":
    main()
