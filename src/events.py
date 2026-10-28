"""Driving-event detector: harsh brake / harsh accel / sharp turn / lane change vs normal.

Works on IMU only (no GPS), in any phone orientation (src/preprocess.py aligns to the car frame).
Dev datasets: BYD (Sri Lanka), saboor (phone). Blind: Ferreira (Brazil); see data_splits.json.

Evaluation is per *event*: consecutive windows with the same predicted class are merged into one
predicted event, which counts as a hit if it overlaps a ground-truth event of that class.

Usage: python src/events.py [--freeze]
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from preprocess import HZ, align, resample, to_si
from splits import ROOT, event_dataset

EVENTS = ["normal", "brake", "accel", "turn", "lane_change"]
WIN_S, STEP_S = 3.0, 0.5
# Low-pass for lon/lat/yaw: manoeuvres are < 1 Hz, phone/road vibration is not. Chosen on dev
# (docs/RESULTS.md): GBM best at 0.5 s (pooled CV), physics rules best at 1.5 s.
SMOOTH_S = 0.5
RULES_SMOOTH_S = 1.5
MODEL_DIR = ROOT / "models/event_gbm"
RAW = ROOT / "data/raw"
IMU = ["t", "ax", "ay", "az", "gx", "gy", "gz"]


# ---------------------------------------------------------------- loaders -> list of sessions
# Each session: DataFrame [t, ax..gz, label] at HZ, label in EVENTS or None (= unlabelled, ignore).

def load_byd() -> list[tuple[str, pd.DataFrame]]:
    event_dataset("byd")
    b = pd.read_csv(next(RAW.glob("events_byd/*/data/processed/imu_samples_clean.csv")))
    out = []
    for sid, x in b.groupby("session_id"):
        if (x.event_class == 0).all():          # no marks at all: may hold unlabelled events, skip
            continue
        d = x.rename(columns={"t_seconds": "t", "acc_x": "ax", "acc_y": "ay", "acc_z": "az",
                              "gyro_x": "gx", "gyro_y": "gy", "gyro_z": "gz"})
        kind = "brake" if sid.startswith("harsh_braking") else "lane_change"
        d["label"] = np.where(d.event_class != 0, kind, "normal")
        out.append((f"byd/{sid}", d[IMU + ["label"]]))
    return out


SABOOR_KIND = {"sudden-acc": "accel", "sudden-acceleration": "accel", "sudden-brake": "brake",
               "sudden-left-turn": "turn", "sudden-right-turn": "turn",
               "sudden-left-line-chg": "lane_change", "sudden-right-line-chg": "lane_change",
               "sudden-line-left": "lane_change", "sudden-line-right": "lane_change",
               "cons-20-linechange": "lane_change"}


def load_saboor() -> list[tuple[str, pd.DataFrame]]:
    event_dataset("saboor")
    out = []
    for f in sorted(glob.glob(str(RAW / "events_saboor/*.csv"))):
        x = pd.read_csv(f)
        ts = pd.to_datetime(x.timestamp.str.replace("_", " "), format="%d-%m-%Y %H:%M:%S:%f")
        d = x.rename(columns={"acc_x": "ax", "acc_y": "ay", "acc_z": "az", "gyro_x": "gx",
                              "gyro_y": "gy", "gyro_z": "gz"}).assign(t=(ts - ts.iloc[0]).dt.total_seconds())
        kind = SABOOR_KIND.get(x.harsh_event.iloc[0], "normal")
        d["label"] = np.where(d.event_class == 1, kind, "normal")
        out.append((f"saboor/{Path(f).stem}", d[IMU + ["label"]]))
    return out


DEV_LOADERS = {"byd": load_byd, "saboor": load_saboor}


# ---------------------------------------------------------------- features

def prepare(d: pd.DataFrame, smooth_s: float | None = None) -> tuple[pd.DataFrame, dict]:
    lab = d[["t", "label"]] if "label" in d else None
    r = resample(to_si(d[IMU]))
    if lab is not None:   # nearest-neighbour label onto the resampled grid
        idx = np.searchsorted(lab.t.to_numpy(), r.t.to_numpy()).clip(0, len(lab) - 1)
        r["label"] = lab.label.to_numpy()[idx]
    a, info = align(r)
    n = max(1, int((SMOOTH_S if smooth_s is None else smooth_s) * HZ))
    for c in ("lon", "lat", "yaw_rate"):
        a[c] = a[c].rolling(n, center=True, min_periods=1).mean()
    a["vert"] = a["vert"].rolling(int(0.25 * HZ), center=True, min_periods=1).mean()   # keep vibration
    return a, info


def window_features(a: pd.DataFrame) -> pd.DataFrame:
    n, step = int(WIN_S * HZ), int(STEP_S * HZ)
    lon, lat, vert, yaw = (a[c].to_numpy() for c in ("lon", "lat", "vert", "yaw_rate"))
    has_label = "label" in a
    rows = []
    for s in range(0, len(a) - n + 1, step):
        sl = slice(s, s + n)
        lo, la, ve, yw = lon[sl], lat[sl], vert[sl], yaw[sl]
        heading = np.sum(yw) / HZ                                   # net heading change [rad]
        r = dict(t0=a.t.iloc[s], t1=a.t.iloc[s + n - 1],
                 lon_min=lo.min(), lon_max=lo.max(), lon_mean=lo.mean(), lon_std=lo.std(),
                 lon_range=lo.max() - lo.min(), lon_jerk=np.abs(np.diff(lo)).max() * HZ,
                 lat_absmax=np.abs(la).max(), lat_absmean=np.abs(la).mean(), lat_std=la.std(),
                 lat_range=la.max() - la.min(), lat_jerk=np.abs(np.diff(la)).max() * HZ,
                 # lane change: lateral accel swings both ways, heading returns to ~0
                 lat_swing=min(la.max(), -la.min()) if la.max() > 0 > la.min() else 0.0,
                 yaw_absmax=np.abs(yw).max(), yaw_absmean=np.abs(yw).mean(), heading_abs=abs(heading),
                 yaw_swing=min(yw.max(), -yw.min()) if yw.max() > 0 > yw.min() else 0.0,
                 vert_std=ve.std(), vert_absmax=np.abs(ve).max(),
                 h_absmax=np.sqrt(lo ** 2 + la ** 2).max())
        if has_label:
            lab = a.label.iloc[sl]
            ev = lab[lab != "normal"]
            if len(ev) >= 0.3 * n:
                r["label"] = ev.mode().iloc[0]
            elif len(ev) == 0:
                r["label"] = "normal"
            else:
                r["label"] = None                                    # partial overlap: ambiguous
        rows.append(r)
    return pd.DataFrame(rows)


ABS_FEATURES = ["lon_min", "lon_max", "lon_mean", "lon_std", "lon_range", "lon_jerk", "lat_absmax",
                "lat_absmean", "lat_std", "lat_range", "lat_jerk", "lat_swing", "yaw_absmax", "yaw_absmean",
                "heading_abs", "yaw_swing", "vert_std", "vert_absmax", "h_absmax"]
# Same features divided by the session's typical (median) level: sensor/phone/car independent.
REL_FEATURES = [f + "_rel" for f in ABS_FEATURES if f != "lon_mean"] + ["lon_mean_z"]
FEATURES = ABS_FEATURES + REL_FEATURES


def add_relative(w: pd.DataFrame) -> pd.DataFrame:
    g = w.groupby("session")
    for f in ABS_FEATURES:
        if f == "lon_mean":
            w["lon_mean_z"] = (w.lon_mean - g.lon_mean.transform("median")) / (g.lon_std.transform("median") + 1e-6)
        else:
            w[f + "_rel"] = w[f].abs() / (g[f].transform(lambda x: x.abs().median()) + 1e-6)
    return w


def build(loaders: dict, smooth_s: float | None = None) -> tuple[pd.DataFrame, dict]:
    frames, info = [], {}
    for ds, load in loaders.items():
        for sid, d in load():
            a, inf = prepare(d, smooth_s)
            w = window_features(a)
            frames.append(w.assign(dataset=ds, session=sid))
            info[sid] = inf
    return add_relative(pd.concat(frames, ignore_index=True)), info


def truth_events(sessions: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for sid, d in sessions:
        lab = d.label.to_numpy()
        t = d.t.to_numpy()
        start = 0
        for i in range(1, len(lab) + 1):
            if i == len(lab) or lab[i] != lab[start]:
                if lab[start] != "normal":
                    rows.append(dict(session=sid, label=lab[start], t0=t[start], t1=t[i - 1]))
                start = i
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- models

def physics_rules(w: pd.DataFrame) -> np.ndarray:
    """Threshold baseline in g-units common in telematics (0.3 g brake/accel, 0.3 g lateral)."""
    g = 9.81
    pred = np.full(len(w), "normal", dtype=object)
    pred[(w.lat_absmax > 0.3 * g) & (w.heading_abs > 0.5)] = "turn"
    pred[(w.lat_swing > 0.15 * g) & (w.heading_abs < 0.3)] = "lane_change"
    pred[w.lon_max > 0.3 * g] = "accel"
    pred[w.lon_min < -0.3 * g] = "brake"
    return pred


PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              class_weight="balanced", verbose=-1)


def fit(w: pd.DataFrame) -> lgb.LGBMClassifier:
    y = w.label.map({c: i for i, c in enumerate(EVENTS)})
    return lgb.LGBMClassifier(**PARAMS).fit(w[FEATURES], y)


def predict(model, w: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(w[FEATURES])
    full = np.zeros((len(w), len(EVENTS)))
    full[:, model.classes_] = p
    return np.array(EVENTS, dtype=object)[full.argmax(1)]


def grouped_cv(w: pd.DataFrame, lab: pd.DataFrame, k: int = 5) -> np.ndarray:
    """Out-of-fold predictions; a session's windows are never split between train and test."""
    from sklearn.model_selection import GroupKFold
    pred = np.full(len(w), "normal", dtype=object)
    sessions = w.session.unique()
    for _, te_idx in GroupKFold(k).split(sessions, groups=sessions):
        te_s = set(sessions[te_idx])
        te = w.session.isin(te_s)
        pred[te.to_numpy()] = predict(fit(lab[~lab.session.isin(te_s)]), w[te])
    return pred


def merge_predictions(w: pd.DataFrame, pred: np.ndarray, min_windows: int = 2) -> pd.DataFrame:
    """Consecutive windows with the same non-normal class -> one predicted event."""
    rows = []
    for sid, g in w.assign(pred=pred).groupby("session", sort=False):
        g = g.sort_values("t0")
        run = (g.pred != g.pred.shift()).cumsum()
        for _, r in g.groupby(run):
            if r.pred.iloc[0] != "normal" and len(r) >= min_windows:
                rows.append(dict(session=sid, label=r.pred.iloc[0], t0=r.t0.min(), t1=r.t1.max()))
    return pd.DataFrame(rows, columns=["session", "label", "t0", "t1"])


def event_scores(truth: pd.DataFrame, pred: pd.DataFrame, hours: float) -> pd.DataFrame:
    rows = []
    for c in EVENTS[1:]:
        T, P = truth[truth.label == c], pred[pred.label == c]
        hit_t = sum(any((P.session == r.session) & (P.t0 <= r.t1) & (P.t1 >= r.t0)) for r in T.itertuples())
        hit_p = sum(any((T.session == r.session) & (T.t0 <= r.t1) & (T.t1 >= r.t0)) for r in P.itertuples())
        rec = hit_t / len(T) if len(T) else np.nan
        prec = hit_p / len(P) if len(P) else np.nan
        rows.append(dict(event=c, n_true=len(T), n_pred=len(P), recall=rec, precision=prec,
                         f1=2 * rec * prec / (rec + prec) if rec and prec else 0.0,
                         false_per_hour=(len(P) - hit_p) / hours))
    return pd.DataFrame(rows)


def evaluate(name: str, w: pd.DataFrame, pred: np.ndarray, truth: pd.DataFrame) -> pd.DataFrame:
    hours = w.groupby("session").t1.max().sum() / 3600
    sc = event_scores(truth[truth.session.isin(w.session.unique())], merge_predictions(w, pred), hours)
    print(f"\n== {name}  ({w.session.nunique()} sessions, {hours * 60:.0f} min)")
    print(sc.round(3).to_string(index=False))
    return sc.assign(setting=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    sessions = {ds: load() for ds, load in DEV_LOADERS.items()}
    truth = truth_events([s for ss in sessions.values() for s in ss])
    w, info = build({ds: (lambda ss=ss: ss) for ds, ss in sessions.items()})
    print("alignment quality (turn_corr) median by dataset:",
          {ds: float(np.median([info[s]["turn_corr"] for s, _ in ss])) for ds, ss in sessions.items()})
    lab = w.dropna(subset=["label"])
    print("windows per label:", lab.label.value_counts().to_dict())
    print("true events:", truth.label.value_counts().to_dict())

    results = [evaluate("physics rules (no training)", w, physics_rules(w), truth)]

    results.append(evaluate("GBM 5-fold grouped by session", w, grouped_cv(w, lab), truth))

    # Leave-one-dataset-out: train on one country/sensor, test on the other.
    for test_ds in DEV_LOADERS:
        tr = lab[lab.dataset != test_ds]
        te = w.dataset == test_ds
        results.append(evaluate(f"GBM trained on others -> {test_ds}", w[te], predict(fit(tr), w[te]), truth))

    out = pd.concat(results)
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "data/processed/event_dev_results.csv", index=False)

    if args.freeze:
        m = fit(lab)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        m.booster_.save_model(MODEL_DIR / "model.txt")
        sha = hashlib.sha256((MODEL_DIR / "model.txt").read_bytes()).hexdigest()[:12]
        (MODEL_DIR / "meta.json").write_text(json.dumps(dict(
            features=FEATURES, classes=[EVENTS[i] for i in m.classes_], params=PARAMS, model_sha=sha,
            win_s=WIN_S, step_s=STEP_S, smooth_s=SMOOTH_S, trained_on=list(DEV_LOADERS)), indent=2))
        print(f"\nfrozen -> {MODEL_DIR} (sha {sha})")


if __name__ == "__main__":
    main()
