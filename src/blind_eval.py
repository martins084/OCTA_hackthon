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


def eval_grab() -> tuple[dict, str]:
    """Both frozen trip risk models on the sealed 20% of Grab trips (behaviour-only = primary)."""
    out, names = {}, []
    for d in ("trip_risk", "trip_risk_full"):
        m, name = _eval_grab_model(ROOT / "models" / d)
        out[d] = m
        names.append(name)
    return out, " + ".join(names)


def _eval_grab_model(mdir) -> tuple[dict, str]:
    """Pre-registered (docs/PLAN.md): primary = ROC-AUC; pass if blind AUC >= lower bound of dev-CV 95% CI."""
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression  # noqa: F401  (calibration is monotone: AUC unaffected)
    from sklearn.metrics import average_precision_score, roc_auc_score
    meta = json.loads((mdir / "meta.json").read_text())
    sha = hashlib.sha256((mdir / "model.txt").read_bytes()).hexdigest()[:12]
    if sha != meta["model_sha"]:
        raise RuntimeError("trip_risk model.txt does not match meta.json; refreeze")
    booster = lgb.Booster(model_file=str(mdir / "model.txt"))
    with blind_access():
        df = pd.read_parquet(ROOT / SPLITS["grab"]["blind_path"] / "grab_blind_features.parquet")
    p = booster.predict(df[meta["features"]])
    y = df.label.to_numpy()
    rng = np.random.default_rng(0)
    boots = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    dev = meta["dev_cv"][meta["model"]]
    auc = float(roc_auc_score(y, p))
    out = dict(auc=auc, auc_ci95=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
               pr_auc=float(average_precision_score(y, p)), n=len(y), positive_rate=float(y.mean()),
               dev_cv_auc=dev["auc"], dev_cv_ci95=dev["auc_ci95"], passed=bool(auc >= dev["auc_ci95"][0]))
    return out, f"trip_risk {meta['model']} {sha}"


def eval_grab_ssl() -> tuple[dict, str]:
    """Pre-registered (docs/PLAN.md): primary = paired AUC gain of the GBM+Transformer ensemble over the full GBM
    on the sealed Grab trips; success only if the 95% CI of the gain is entirely above zero."""
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score
    from ssl_model import Classifier, Encoder, predict
    sdir = ROOT / "models/trip_ssl"
    meta = json.loads((sdir / "meta.json").read_text())
    sha = hashlib.sha256(b"".join((sdir / f).read_bytes() for f in meta["files"])).hexdigest()[:12]
    if sha != meta["model_sha"]:
        raise RuntimeError("trip_ssl weights do not match meta.json")
    cfg = meta["config"]
    models = []
    for f in meta["files"]:
        m = Classifier(Encoder(meta["channels"], meta["n_patch"], cfg["patch"], cfg["d"], 4, cfg["layers"]),
                       meta["channels"], cfg["d"])
        m.load_state_dict(torch.load(sdir / f, map_location="cpu"))
        models.append(m)
    gdir = ROOT / "models/trip_risk_full"
    gmeta = json.loads((gdir / "meta.json").read_text())
    booster = lgb.Booster(model_file=str(gdir / "model.txt"))
    bp = ROOT / SPLITS["grab"]["blind_path"]
    with blind_access():
        feats = pd.read_parquet(bp / "grab_blind_features.parquet")
        z = np.load(bp / "grab_blind_seq.npz")
    seq = pd.DataFrame({"bookingID": z["bookingID"], "i": np.arange(len(z["bookingID"]))})
    df = feats.merge(seq, on="bookingID")
    p_ssl = predict(models, z["X"][df.i.to_numpy()].astype(np.float32), np.array(meta["mu"]), np.array(meta["sd"]),
                    cfg["patch"])
    p_gbm = booster.predict(df[gmeta["features"]])
    w = meta["ensemble_weight"]
    rank = lambda a: pd.Series(a).rank(pct=True).to_numpy()
    ens = (1 - w) * rank(p_gbm) + w * rank(p_ssl)
    y = df.label.to_numpy()
    rng = np.random.default_rng(0)
    gains = [roc_auc_score(y[i], ens[i]) - roc_auc_score(y[i], p_gbm[i])
             for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    lo, hi = float(np.quantile(gains, .025)), float(np.quantile(gains, .975))
    out = dict(auc_gbm_full=float(roc_auc_score(y, p_gbm)), auc_ssl=float(roc_auc_score(y, p_ssl)),
               auc_ensemble=float(roc_auc_score(y, ens)), gain=float(np.mean(gains)), gain_ci95=[lo, hi],
               ensemble_weight=w, n=len(y), passed=bool(lo > 0))
    return out, f"trip_ssl {sha} + trip_risk_full {gmeta['model_sha']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["style", "pvs", "grab", "grab_ssl"])
    ap.add_argument("--confirm", action="store_true", help="required: this reads blind data")
    ap.add_argument("--note", required=True)
    args = ap.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to touch blind data without --confirm (see docs/PLAN.md section 3).")

    m, model = {"style": eval_style, "pvs": eval_pvs, "grab": eval_grab,
                    "grab_ssl": eval_grab_ssl}[args.target]()
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
