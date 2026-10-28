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

import pandas as pd

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


FERREIRA_KIND = [("nao_agressivo", "normal"), ("freada", "brake"), ("acelera", "accel"),
                 ("curva", "turn"), ("troca", "lane_change")]


def load_ferreira() -> tuple[list, pd.DataFrame]:
    """Ferreira trips -> (sessions without labels, ground-truth events incl. non-aggressive ones)."""
    root = ROOT / SPLITS["events"]["blind_path"] / "data"
    sessions, truth = [], []
    for trip in sorted(p for p in root.iterdir() if p.is_dir()):
        t0 = json.loads((trip / "viagem.json").read_text())["firstCollectionUptimeNanos"]
        acc = pd.read_csv(trip / "acelerometro_terra.csv")
        gyr = pd.read_csv(trip / "giroscopio_terra.csv")
        t = (acc.uptimeNanos - t0) / 1e9
        tg = (gyr.uptimeNanos - t0) / 1e9
        d = pd.DataFrame({"t": t, "ax": acc.x, "ay": acc.y, "az": acc.z,
                          **{f"g{c}": np.interp(t, tg, gyr[c]) for c in "xyz"}})
        sid = f"ferreira/{trip.name}"
        sessions.append((sid, d))
        gt = pd.read_csv(trip / "groundTruth.csv", skipinitialspace=True)
        for r in gt.itertuples():
            kind = next(k for key, k in FERREIRA_KIND if key in r.evento)
            truth.append(dict(session=sid, label=kind, t0=r.inicio, t1=r.fim))
    return sessions, pd.DataFrame(truth)


def eval_events() -> tuple[dict, str]:
    meta = json.loads((ev.MODEL_DIR / "meta.json").read_text())
    sha = hashlib.sha256((ev.MODEL_DIR / "model.txt").read_bytes()).hexdigest()[:12]
    if sha != meta["model_sha"]:
        raise RuntimeError("event model.txt does not match meta.json; refreeze")
    booster = lgb.Booster(model_file=str(ev.MODEL_DIR / "model.txt"))
    with blind_access():
        sessions, truth = load_ferreira()
    w, _ = ev.build({"ferreira": lambda: sessions}, meta["smooth_s"])
    wr, _ = ev.build({"ferreira": lambda: sessions}, ev.RULES_SMOOTH_S)
    p = booster.predict(w[meta["features"]])
    gbm = np.array(meta["classes"], dtype=object)[np.asarray(p).argmax(1)]
    hours = w.groupby("session").t1.max().sum() / 3600
    out, normal = {}, truth[truth.label == "normal"]
    # Pre-registered (docs/PLAN.md): rules = primary, gbm = secondary.
    for name, ww, pred in [("rules (primary)", wr, ev.physics_rules(wr)), ("gbm (secondary)", w, gbm)]:
        P = ev.merge_predictions(ww, pred)
        sc = ev.event_scores(truth[truth.label != "normal"], P, hours)
        print(f"\n== blind Ferreira, {name}")
        print(sc.round(3).to_string(index=False))
        flagged = sum(any((P.session == r.session) & (P.t0 <= r.t1) & (P.t1 >= r.t0)) for r in normal.itertuples())
        out[name] = dict(macro_f1=float(sc.f1.mean()), **{f"f1_{r.event}": float(r.f1) for r in sc.itertuples()},
                         non_aggressive_flagged=f"{flagged}/{len(normal)}")
    return out, f"event_gbm {sha} + physics rules"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["style", "events"])
    ap.add_argument("--confirm", action="store_true", help="required: this reads blind data")
    ap.add_argument("--note", required=True)
    args = ap.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to touch blind data without --confirm (see docs/PLAN.md section 3).")

    m, model = {"style": eval_style, "events": eval_events}[args.target]()
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
