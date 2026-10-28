"""Grab safety data -> trip features (dev and blind in separate files) + dev sequences for the GPU model.

Outputs
  data/processed/grab_dev_features.parquet         dev trips, features + label
  data/raw/blind_grab/grab_blind_features.parquet  blind trips, features + label (read only by blind_eval)
  data/processed/grab_dev_seq.npz                  dev trips as 1 Hz sequences [n, T, C] for the GPU model
Usage: python src/build_grab.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from splits import grab_is_blind
from trip_features import CHUNK, clean_trip, grab_trip_frame, trip_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/grab"
SEQ_T = 1800                    # first 30 min at 1 Hz (median Grab trip is ~14 min)
SEQ_CH = ["speed", "a_lon", "hmag", "vert", "yaw", "offaxis"]

DTYPES = {"bookingID": "int64", "Accuracy": "float32", "Bearing": "float32", "acceleration_x": "float32",
          "acceleration_y": "float32", "acceleration_z": "float32", "gyro_x": "float32", "gyro_y": "float32",
          "gyro_z": "float32", "second": "float32", "Speed": "float32"}


def sequence(d: pd.DataFrame) -> np.ndarray:
    """1 Hz orientation-invariant channels, zero-padded to SEQ_T."""
    d = clean_trip(d)
    t = d.t.to_numpy()
    s = np.floor(t).astype(int)
    keep = s < SEQ_T
    d, s, t = d[keep], s[keep], t[keep]
    acc = d[["ax", "ay", "az"]].to_numpy(float)
    gyr = d[["gx", "gy", "gz"]].to_numpy(float)
    chunk = (t // CHUNK).astype(int)
    grav = pd.DataFrame(acc).groupby(chunk).transform("median").to_numpy()
    up = grav / (np.linalg.norm(grav, axis=1, keepdims=True) + 1e-9)
    lin = acc - grav
    vert = np.sum(lin * up, axis=1)
    hmag = np.linalg.norm(lin - vert[:, None] * up, axis=1)
    yaw = np.sum(gyr * up, axis=1)
    off = np.sqrt(np.maximum(np.sum(gyr ** 2, axis=1) - yaw ** 2, 0))
    v = pd.Series(d.speed.to_numpy(float)).rolling(3, center=True, min_periods=1).median().to_numpy()
    a_lon = np.nan_to_num(np.gradient(np.nan_to_num(v), np.maximum(t, 0) + np.arange(len(t)) * 1e-6)) / 9.81
    out = np.zeros((SEQ_T, len(SEQ_CH)), np.float32)
    frame = pd.DataFrame({"s": s, "speed": np.nan_to_num(v) * 3.6, "a_lon": np.clip(a_lon, -1.5, 1.5), "hmag": hmag,
                          "vert": vert, "yaw": yaw, "offaxis": off}).groupby("s").mean()
    out[frame.index.to_numpy()] = frame[SEQ_CH].to_numpy(np.float32)
    return out


def one(bid, raw):
    """Never raises: a malformed trip is reported and skipped, not silently dropped."""
    try:
        d = grab_trip_frame(raw)
        f = trip_features(d)
        return bid, f, sequence(d) if f else None
    except Exception as e:                       # noqa: BLE001
        return bid, {"_error": f"{type(e).__name__}: {e}"}, None


def main():
    labels = pd.read_csv(next(RAW.glob("labels/*.csv")))
    conflicted = labels.groupby("bookingID").label.nunique().loc[lambda s: s > 1].index
    labels = labels[~labels.bookingID.isin(conflicted)].drop_duplicates("bookingID")
    print(f"labels: {len(labels)} bookings after dropping {len(conflicted)} conflicting, {labels.label.mean():.1%} dangerous")

    raw = pd.concat([pd.read_csv(f, dtype=DTYPES) for f in sorted(glob.glob(str(RAW / "features/*.csv")))],
                    ignore_index=True)
    print(f"raw rows: {len(raw):,}, bookings {raw.bookingID.nunique():,}")
    groups = raw.groupby("bookingID", sort=False)
    res = Parallel(n_jobs=12, batch_size=64)(delayed(one)(bid, g) for bid, g in groups)
    del raw, groups

    errors = [(bid, f["_error"]) for bid, f, _ in res if "_error" in f]
    print(f"skipped {len(errors)} malformed trips", errors[:3])
    res = [r for r in res if "_error" not in r[1]]
    rows = [dict(bookingID=bid, **f) for bid, f, _ in res if f]
    df = pd.DataFrame(rows).merge(labels, on="bookingID", how="inner")
    blind = grab_is_blind(df.bookingID.to_numpy())
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    df[~blind].to_parquet(ROOT / "data/processed/grab_dev_features.parquet", index=False)
    bdir = ROOT / "data/raw/blind_grab"
    bdir.mkdir(parents=True, exist_ok=True)
    df[blind].to_parquet(bdir / "grab_blind_features.parquet", index=False)
    print(f"features: dev {int((~blind).sum())}, blind {int(blind.sum())}, {df.shape[1] - 2} features")

    lab = dict(zip(labels.bookingID, labels.label))
    seq = [(bid, s) for bid, f, s in res if s is not None and bid in lab and not grab_is_blind([bid])[0]]
    np.savez_compressed(ROOT / "data/processed/grab_dev_seq.npz", X=np.stack([s for _, s in seq]),
                        y=np.array([lab[b] for b, _ in seq]), bookingID=np.array([b for b, _ in seq]),
                        channels=np.array(SEQ_CH))
    print(f"dev sequences: {len(seq)} x {SEQ_T} x {len(SEQ_CH)}")


if __name__ == "__main__":
    main()
