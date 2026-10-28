"""Experiment: which event features transfer to an unseen sensor? Leave-one-dataset-out over the
3 dev event datasets. Brake+accel merged into 'speed_change' (GPS gives the sign in the app)."""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import events as E  # noqa: E402

MERGE = {"brake": "speed_change", "accel": "speed_change"}
E.EVENTS = ["normal", "speed_change", "turn", "lane_change"]


def run(w, truth, feats, name):
    E.FEATURES = feats
    lab = w.dropna(subset=["label"])
    res = []
    with contextlib.redirect_stdout(io.StringIO()):
        for t in w.dataset.unique():
            te = w.dataset == t
            res.append(E.evaluate(t, w[te], E.predict(E.fit(lab[lab.dataset != t]), w[te]), truth))
    r = pd.concat(res)
    tab = r.pivot(index="setting", columns="event", values="f1").round(2)
    print(f"\n### {name}: mean F1 over held-out datasets = {r.dropna(subset=['recall']).f1.mean():.3f}")
    print(tab.to_string())
    return r


def main_compare():
    smooth = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1][0].isdigit() else 0.5
    sessions = {ds: [(sid, d.assign(label=d.label.replace(MERGE))) for sid, d in load()]
                for ds, load in E.DEV_LOADERS.items()}
    truth = E.truth_events([s for ss in sessions.values() for s in ss])
    w, _ = E.build({ds: (lambda ss=ss: ss) for ds, ss in sessions.items()}, smooth)
    w["label"] = w.label.replace(MERGE)
    free_rel = [f + "_rel" for f in E.FREE_FEATURES]
    print(f"smooth={smooth}s windows:", w.label.value_counts().to_dict())
    run(w, truth, E.ABS_FEATURES + E.REL_FEATURES, "aligned abs+rel")
    run(w, truth, E.FREE_FEATURES, "alignment-free abs")
    run(w, truth, E.FREE_FEATURES + free_rel, "alignment-free abs+rel")
    run(w, truth, free_rel, "alignment-free rel only")


def augment(sessions, n_copies=2, seed=0):
    """Noisy/gain-jittered copies of raw IMU sessions: simulate cheaper/noisier phones."""
    rng = np.random.default_rng(seed)
    out = []
    for sid, d in sessions:
        for k in range(n_copies):
            a = d.copy()
            noise = rng.uniform(0.0, 1.5)
            gain = rng.uniform(0.7, 1.3)
            a[["ax", "ay", "az"]] = a[["ax", "ay", "az"]] * gain + rng.normal(0, noise, (len(a), 3))
            a[["gx", "gy", "gz"]] = a[["gx", "gy", "gz"]] * rng.uniform(0.9, 1.1) + rng.normal(0, 0.02, (len(a), 3))
            a.attrs = dict(d.attrs)
            out.append((f"{sid}#aug{k}", a))
    return out


def run_aug(sessions, truth, smooth, feats_fn):
    """LODO where only training datasets get augmented copies."""
    aug = {ds: augment(ss) for ds, ss in sessions.items()}
    w, _ = E.build({ds: (lambda ss=ss + aug[ds]: ss) for ds, ss in sessions.items()}, smooth)
    w["label"] = w.label.replace(MERGE)
    is_aug = w.session.str.contains("#aug")
    E.FEATURES = feats_fn()
    lab = w.dropna(subset=["label"])
    res = []
    with contextlib.redirect_stdout(io.StringIO()):
        for t in w.dataset.unique():
            te = (w.dataset == t) & ~is_aug
            tr = lab[lab.dataset != t]
            res.append(E.evaluate(t, w[te], E.predict(E.fit(tr), w[te]), truth))
    r = pd.concat(res)
    print(f"\n### augmented, alignment-free abs+rel, smooth {smooth}: mean F1 = {r.dropna(subset=['recall']).f1.mean():.3f}")
    print(r.pivot(index="setting", columns="event", values="f1").round(2).to_string())
    return r


def main_aug():
    for smooth in (0.5, 1.0):
        sessions = {ds: [(sid, d.assign(label=d.label.replace(MERGE))) for sid, d in load()]
                    for ds, load in E.DEV_LOADERS.items()}
        truth = E.truth_events([s for ss in sessions.values() for s in ss])
        run_aug(sessions, truth, smooth, lambda: E.FREE_FEATURES + [f + "_rel" for f in E.FREE_FEATURES])


if __name__ == "__main__":
    main_aug() if "--aug" in sys.argv else main_compare()
