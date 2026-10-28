"""The ONLY entry point that reads blind test data. Evaluates frozen models; never trains.

Every run appends a row to docs/BLIND_LOG.md, so every look at the blind data is on record.
Run once per milestone, not as a tuning loop (docs/PLAN.md section 3).

Usage: python src/blind_eval.py style --confirm --note "milestone description"
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

import events as ev
from splits import ROOT, SPLITS, blind_access, uah_trip_dirs
from train_driving import CLASSES, MODEL_DIR, build_dataset, metrics, trip_table

LOG = ROOT / "docs/BLIND_LOG.md"


def eval_style() -> tuple[dict, str]:
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    sha = hashlib.sha256((MODEL_DIR / "model.txt").read_bytes()).hexdigest()[:12]
    if sha != meta["model_sha"]:
        raise RuntimeError("model.txt does not match meta.json; refreeze")
    booster = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
    with blind_access():
        df = build_dataset(uah_trip_dirs(ROOT / "data/raw/uah", "blind"))
    proba = booster.predict(df[meta["features"]])
    m = metrics(df, np.asarray(proba))
    trip = trip_table(df, np.asarray(proba))
    print(trip[["label", "pred", "p_risky"] + [f"p_{c}" for c in CLASSES]].round(3).to_string())
    m["dev_lodo_trip_accuracy"] = meta["dev_lodo"]["trip_accuracy"]
    return m, f"style_gbm {sha} (commit {meta['git_commit']}{', dirty' if meta['dirty'] else ''})"


def eval_pvs() -> tuple[dict, str]:
    """Whole engine on PVS 7-9 (vehicle/driver 3, normal driving incl. cobblestone, dirt, speed bumps).
    Pre-registered (docs/PLAN.md): primary = GPS-confirmed harsh events per hour; pass if <= 2/h and
    every trip scores >= 80."""
    from engine import analyze_trip, load_pvs
    from splits import pvs_trip_dirs
    meta = json.loads((ev.MODEL_DIR / "meta.json").read_text())
    rows = []
    with blind_access():
        trips = pvs_trip_dirs("blind")
    for trip in trips:
        rep = analyze_trip(*load_pvs(trip))
        hours = rep["trip"]["duration_min"] / 60
        confirmed = sum(n for k, n in rep["imu_event_counts"].items() if k in ("harsh_brake", "harsh_accel", "turn"))
        rows.append(dict(trip=trip.name, score=rep["score"], hours=round(hours, 2),
                         confirmed_harsh_per_hour=round(confirmed / hours, 2),
                         raw_imu_events_per_hour=round(len(rep["imu_events"]) / hours, 1),
                         rejected_by_gps=rep["imu_events_rejected_by_gps"], counts=rep["imu_event_counts"],
                         speed_limit_coverage=rep["trip"]["speed_limit_coverage"]))
        print(rows[-1])
    total_h = sum(r["hours"] for r in rows)
    per_h = sum(r["confirmed_harsh_per_hour"] * r["hours"] for r in rows) / total_h
    out = dict(confirmed_harsh_per_hour=round(per_h, 2), min_score=min(r["score"] for r in rows),
               scores=[r["score"] for r in rows],
               raw_imu_per_hour=round(sum(r["raw_imu_events_per_hour"] * r["hours"] for r in rows) / total_h, 1),
               passed=bool(per_h <= 2 and min(r["score"] for r in rows) >= 80))
    return out, f"engine v2: event_gbm {meta['model_sha']} + style_gbm + GPS confirmation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["style", "pvs"])
    ap.add_argument("--confirm", action="store_true", help="required: this reads blind data")
    ap.add_argument("--note", required=True)
    args = ap.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to touch blind data without --confirm (see docs/PLAN.md section 3).")

    m, model = {"style": eval_style, "pvs": eval_pvs}[args.target]()
    print(json.dumps(m, indent=2))
    if not LOG.exists():
        LOG.write_text("# Blind evaluation log\n\nEvery read of blind data, appended by src/blind_eval.py.\n\n"
                       "| time | target | model | note | metrics |\n|---|---|---|---|---|\n", encoding="utf-8")
    row = (f"| {dt.datetime.now():%Y-%m-%d %H:%M} | {args.target} | {model} | {args.note} | "
           f"{json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})} |\n")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(row)


if __name__ == "__main__":
    main()
