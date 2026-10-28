"""Driving-style model on UAH-DriveSet dev drivers, evaluated leave-one-driver-out (LODO).

The model is always tested on a driver it has never seen, which is what happens in production.
Blind driver(s) from data_splits.json are excluded by src/splits.py.

Usage: python src/train_driving.py [--uah data/raw/uah] [--freeze]
  --freeze  refit on all dev drivers and save models/style_gbm/ for src/blind_eval.py
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

from driving_features import FEATURES, resample_trip, window_features
from splits import uah_trip_dirs

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["NORMAL", "AGGRESSIVE", "DROWSY"]
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
              class_weight="balanced")
MODEL_DIR = ROOT / "models/style_gbm"


def read_uah_trip(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(folder / "RAW_ACCELEROMETERS.txt", sep=r"\s+", header=None).iloc[:, :8]
    a.columns = ["t", "active", "rx", "ry", "rz", "x", "y", "z"]          # x,y,z: Kalman-filtered, in g
    g = pd.read_csv(folder / "RAW_GPS.txt", sep=r"\s+", header=None).iloc[:, :8]
    g.columns = ["t", "speed", "lat", "lon", "alt", "vacc", "hacc", "course"]
    osm = folder / "PROC_OPENSTREETMAP_DATA.txt"
    if osm.exists() and osm.stat().st_size:
        o = pd.read_csv(osm, sep=r"\s+", header=None).iloc[:, :2]
        o.columns = ["t", "speed_limit"]
        o.loc[o.speed_limit <= 0, "speed_limit"] = np.nan
        g["speed_limit"] = np.interp(g.t, o.t, o.speed_limit.ffill().bfill())
    return g, a


def build_dataset(trip_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for folder in trip_dirs:
        parts = folder.name.split("-")          # 20151111125233-24km-D1-AGGRESSIVE-MOTORWAY
        label = parts[3].rstrip("0123456789")
        gps, acc = read_uah_trip(folder)
        w = window_features(resample_trip(gps, acc))
        frames.append(w.assign(driver=parts[2], label=label, road=parts[4], trip=folder.name))
    return pd.concat(frames, ignore_index=True)


def lodo(df: pd.DataFrame, target: pd.Series, n_class: int) -> np.ndarray:
    """Out-of-fold predicted probabilities, one fold per held-out driver."""
    oof = np.zeros((len(df), n_class))
    for drv in sorted(df.driver.unique()):
        tr, te = df.driver != drv, df.driver == drv
        m = lgb.LGBMClassifier(**PARAMS).fit(df.loc[tr, FEATURES], target[tr])
        oof[te.to_numpy()] = m.predict_proba(df.loc[te, FEATURES])
    return oof


def trip_table(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Average window probabilities per trip -> trip prediction."""
    t = df.assign(**{f"p_{c}": proba[:, i] for i, c in enumerate(CLASSES)}).groupby("trip").agg(
        label=("label", "first"), driver=("driver", "first"), road=("road", "first"),
        **{f"p_{c}": (f"p_{c}", "mean") for c in CLASSES})
    t["pred"] = [CLASSES[i] for i in t[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1)]
    t["p_risky"] = 1 - t.p_NORMAL
    return t


def metrics(df: pd.DataFrame, proba: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Window + trip metrics, with bootstrap CIs over trips (windows of one trip are not independent)."""
    y = df.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    trip = trip_table(df, proba)
    out = dict(window_macro_f1=f1_score(y, proba.argmax(1), average="macro"),
               trip_accuracy=(trip.pred == trip.label).mean(),
               trip_auc_risky=roc_auc_score(trip.label != "NORMAL", trip.p_risky),
               n_trips=len(trip), n_windows=len(df))
    rng = np.random.default_rng(seed)
    correct = (trip.pred == trip.label).to_numpy()
    boots = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n_boot)]
    out["trip_accuracy_ci95"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def freeze(df: pd.DataFrame, dev_metrics: dict):
    y = df.label.map({c: i for i, c in enumerate(CLASSES)})
    m = lgb.LGBMClassifier(**PARAMS).fit(df[FEATURES], y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    m.booster_.save_model(MODEL_DIR / "model.txt")
    sha = hashlib.sha256((MODEL_DIR / "model.txt").read_bytes()).hexdigest()[:12]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    meta = dict(features=FEATURES, classes=CLASSES, params=PARAMS, model_sha=sha, git_commit=commit,
                dirty=bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                          text=True, cwd=ROOT).stdout.strip()),
                trained_on_drivers=sorted(df.driver.unique()), dev_lodo=dev_metrics)
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nfrozen -> {MODEL_DIR} (sha {sha})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uah", type=Path, default=ROOT / "data/raw/uah")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    df = build_dataset(uah_trip_dirs(args.uah, "dev"))
    print(f"{len(df)} windows from {df.trip.nunique()} trips, drivers {sorted(df.driver.unique())}")

    y = df.label.map({c: i for i, c in enumerate(CLASSES)})
    oof = lodo(df, y, 3)
    print("\n== 3-class, per 60 s window (leave-one-driver-out)")
    print(classification_report(y, oof.argmax(1), target_names=CLASSES, digits=3))
    print(pd.DataFrame(confusion_matrix(y, oof.argmax(1)), index=CLASSES, columns=CLASSES))
    trip = trip_table(df, oof)
    print("\n== trips")
    print(pd.crosstab(trip.label, trip.pred))
    m = metrics(df, oof)
    print("\n== dev LODO metrics")
    print(json.dumps(m, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "uah_dev_windows.parquet")
    trip.to_csv(args.out / "uah_dev_trip_predictions.csv")
    if args.freeze:
        freeze(df, m)


if __name__ == "__main__":
    main()
