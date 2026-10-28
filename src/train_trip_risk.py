"""Learned trip risk model: P(trip flagged dangerous) from phone telematics, trained on Grab safety labels.

- 5-fold stratified CV on Grab dev trips (blind 20% sealed, src/splits.py). Grab has no driver IDs, so
  folds are by booking; the same driver may appear in several folds (noted in RESULTS.md).
- Compares logistic regression, LightGBM, and monotone-constrained LightGBM (risk can only go up with
  more harsh events / speeding: fair and explainable).
- Calibrated with isotonic regression on out-of-fold predictions.
- External test: the frozen model scores UAH trips (Spain, other phone, never used for training).

Usage: python src/train_trip_risk.py [--freeze]
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/trip_risk"
# Direction of risk for the monotone model (+1: more is riskier, 0: unconstrained).
MONO = {"speed_p99": 1, "speed_max": 1, "frac_over_80": 1, "frac_over_100": 1, "brake_per_h": 1,
        "brake_hard_per_h": 1, "accel_per_h": 1, "turn_per_h": 1, "harsh_total_per_h": 1, "lon_std": 1,
        "lon_p99": 1, "lon_p01": -1, "lat_p99": 1, "hmag_p99": 1, "hmag_over_3_per_h": 1,
        "reorient_30_per_h": 1, "handling_frac": 1}
DROP = {"bookingID", "label", "n_samples", "sample_rate", "acc_g_norm"}   # ids / sensor artefacts, not behaviour
# Exposure: longer trips get flagged more just by lasting longer. The driving score must reflect *how* you drive;
# the insurer prices exposure (km, hours) separately. --behaviour-only removes these.
EXPOSURE = {"duration_s", "distance_km"}
PARAMS = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1)


def features(df: pd.DataFrame, behaviour_only: bool = False) -> list[str]:
    return [c for c in df.columns if c not in DROP and not (behaviour_only and c in EXPOSURE)]


def models(feats):
    mono = [MONO.get(f, 0) for f in feats]
    return {
        "logistic": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          LogisticRegression(C=0.3, max_iter=2000)),
        "gbm": lambda: lgb.LGBMClassifier(**PARAMS),
        "gbm_monotone": lambda: lgb.LGBMClassifier(**PARAMS, monotone_constraints=mono,
                                                   monotone_constraints_method="advanced"),
    }


def cv(df, feats, k=5, seed=0):
    X, y = df[feats], df.label.to_numpy()
    oof = {name: np.zeros(len(df)) for name in models(feats)}
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        for name, make in models(feats).items():
            m = make().fit(X.iloc[tr], y[tr])
            oof[name][te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def bootstrap_auc(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    s = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def uah_external(model, feats, calib) -> dict:
    """Score every UAH trip (never used for this model) at 1 Hz; does risk rank aggressive/drowsy above normal?"""
    from engine import load_uah
    from splits import blind_access, uah_trip_dirs
    from trip_features import to_1hz, trip_features
    with blind_access():      # D3's style-model blind run is already logged; this model never saw UAH at all
        dirs = uah_trip_dirs(ROOT / "data/raw/uah", "dev") + uah_trip_dirs(ROOT / "data/raw/uah", "blind")
    rows = []
    for f in dirs:
        gps, imu, _ = load_uah(f)
        d = pd.DataFrame({"t": imu.t, "ax": imu.ax, "ay": imu.ay, "az": imu.az})
        d["speed"] = np.interp(d.t, gps.t, gps.speed / 3.6)
        d["bearing"] = np.interp(d.t, gps.t, np.degrees(np.unwrap(np.radians(gps.course)))) % 360
        tf = trip_features(to_1hz(d))
        label = f.name.split("-")[3].rstrip("0123456789")
        rows.append(dict(trip=f.name, driver=f.name.split("-")[2], label=label, **tf))
    u = pd.DataFrame(rows)
    p = calib.predict(model.predict_proba(u.reindex(columns=feats))[:, 1])
    u["p_danger"] = p
    risky = u.label != "NORMAL"
    pairs = ok = 0
    for _, g in u.groupby("driver"):
        for a in g[g.label == "NORMAL"].p_danger:
            for b in g[g.label != "NORMAL"].p_danger:
                pairs += 1
                ok += b > a
    out = dict(auc_risky=roc_auc_score(risky, p),
               auc_aggressive=roc_auc_score(u.label[u.label != "DROWSY"] == "AGGRESSIVE", u.p_danger[u.label != "DROWSY"]),
               auc_drowsy=roc_auc_score(u.label[u.label != "AGGRESSIVE"] == "DROWSY", u.p_danger[u.label != "AGGRESSIVE"]),
               within_driver_pairs=ok / pairs, n_trips=len(u),
               mean_p=u.groupby("label").p_danger.mean().round(3).to_dict())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--behaviour-only", action="store_true")
    args = ap.parse_args()
    df = pd.read_parquet(ROOT / "data/processed/grab_dev_features.parquet")
    feats = features(df, args.behaviour_only)
    y = df.label.to_numpy()
    print(f"{len(df)} dev trips, {y.mean():.1%} dangerous, {len(feats)} features")

    oof = cv(df, feats)
    res = {}
    for name, p in oof.items():
        lo, hi = bootstrap_auc(y, p)
        res[name] = dict(auc=roc_auc_score(y, p), auc_ci95=[lo, hi], pr_auc=average_precision_score(y, p))
        print(f"{name:13s} AUC {res[name]['auc']:.4f} [{lo:.3f}, {hi:.3f}]  PR-AUC {res[name]['pr_auc']:.3f}")

    best = "gbm_monotone" if res["gbm_monotone"]["auc"] >= res["gbm"]["auc"] - 0.005 else "gbm"
    print(f"chosen: {best} (monotone preferred unless it costs > 0.005 AUC)")
    calib = IsotonicRegression(out_of_bounds="clip").fit(oof[best], y)
    pc = calib.predict(oof[best])
    res["calibration"] = dict(brier_raw=brier_score_loss(y, oof[best]), brier_calibrated=brier_score_loss(y, pc))
    # risk by decile: the insurer view
    dec = pd.qcut(pc, 10, labels=False, duplicates="drop")
    lift = pd.Series(y).groupby(dec).mean()
    res["dangerous_rate_by_decile"] = lift.round(3).tolist()
    print("dangerous rate by predicted-risk decile:", lift.round(3).tolist())

    model = models(feats)[best]().fit(df[feats], y)
    gain = pd.Series(model.booster_.feature_importance("gain"), feats)
    print("top features:", (gain / gain.sum()).sort_values(ascending=False).head(12).round(3).to_dict())

    ext = uah_external(model, feats, calib)
    res["uah_external"] = ext
    print("\nexternal UAH (Spain, never trained on):", json.dumps(ext))
    tag = "_behaviour" if args.behaviour_only else ""
    (ROOT / f"data/processed/trip_risk_dev{tag}.json").write_text(json.dumps(res, indent=2, default=float))
    pd.DataFrame({"bookingID": df.bookingID, "y": y, "oof": oof[best]}).to_parquet(
        ROOT / f"data/processed/trip_risk_oof{tag}.parquet", index=False)

    if args.freeze:
        import hashlib
        global MODEL_DIR
        MODEL_DIR = MODEL_DIR if args.behaviour_only else MODEL_DIR.with_name("trip_risk_full")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(MODEL_DIR / "model.txt")
        sha = hashlib.sha256((MODEL_DIR / "model.txt").read_bytes()).hexdigest()[:12]
        # population CDF of calibrated risk -> "safer than X% of trips"
        cdf = np.quantile(pc, np.linspace(0, 1, 101)).tolist()
        (MODEL_DIR / "meta.json").write_text(json.dumps(dict(
            model=best, features=feats, model_sha=sha, monotone=MONO, params=PARAMS,
            isotonic_x=calib.X_thresholds_.tolist(), isotonic_y=calib.y_thresholds_.tolist(),
            risk_cdf=cdf, base_rate=float(y.mean()), behaviour_only=args.behaviour_only, dev_cv=res),
            indent=2, default=float))
        print(f"frozen -> {MODEL_DIR} (sha {sha})")


if __name__ == "__main__":
    main()
