"""Personal baseline: how unusual is this trip *for this driver*?

An insurer sees each driver's history. Each trip is compared with the driver's own other normal trips
(location from the driver, scale from the population), on road-type-robust features. The anomaly score is
the RMS of the top-3 z-scores. Evaluated leave-one-trip-out on UAH dev drivers: can "unusual for you"
separate aggressive / drowsy trips from normal ones without any population labels?

Usage: python src/personal_baseline.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
# Road-type robust: relative to the limit or to the driver's own manoeuvres, not absolute speed.
FEATS = ["speed_ratio_mean", "speed_ratio_p95", "frac_over_limit_10", "lon_p05", "lon_p95", "lon_jerk_std",
         "lat_abs_p95", "lat_x_speed", "mag_std", "jerk_p95", "band_high_frac", "band_low_frac"]


# Within-driver spread of normal trips (UAH dev drivers), used as the z-score scale for new drivers too.
POP_SCALE_FILE = ROOT / "models/personal_scale.json"
POP_SCALE = json.loads(POP_SCALE_FILE.read_text()) if POP_SCALE_FILE.exists() else {}
NAMES = {"speed_ratio_mean": "speed relative to the limit", "speed_ratio_p95": "peak speed vs limit",
         "frac_over_limit_10": "time over the limit", "lon_p05": "braking strength",
         "lon_p95": "acceleration strength", "lon_jerk_std": "jerky speed changes",
         "lat_abs_p95": "cornering force", "lat_x_speed": "cornering at speed", "mag_std": "overall motion",
         "jerk_p95": "sharp movements", "band_high_frac": "vibration", "band_low_frac": "slow swaying"}


# Features where a *lower* value means more intense driving (braking is negative acceleration).
LOWER_IS_HARDER = {"lon_p05"}
# Road surface / phone mount properties: used in the anomaly score, never shown to the driver as a reason.
NOT_BEHAVIOUR = {"band_high_frac", "band_low_frac", "mag_std"}


def explain_z(z: pd.Series, k: int = 3) -> list[str]:
    z = z.drop(index=[f for f in NOT_BEHAVIOUR if f in z.index])
    top = z.abs().sort_values(ascending=False).head(k)
    out = []
    for f in top.index:
        harder = (z[f] < 0) if f in LOWER_IS_HARDER else (z[f] > 0)
        word = {"lon_p05": ("harder", "gentler")}.get(f, ("higher", "lower"))[0 if harder else 1]
        label = "braking" if f == "lon_p05" else NAMES[f]
        out.append(f"{label} {word} than your usual ({abs(z[f]):.1f} sd)")
    return out


def trip_table(windows: pd.DataFrame) -> pd.DataFrame:
    agg = windows.groupby("trip").agg(**{f: (f, "mean") for f in FEATS},
                                      driver=("driver", "first"), label=("label", "first"), road=("road", "first"))
    return agg


def anomaly_scores(trips: pd.DataFrame, top_k: int = 3) -> pd.Series:
    # population scale: within-driver spread of normal trips, pooled
    normal = trips[trips.label == "NORMAL"]
    scale = normal.groupby("driver")[FEATS].transform(lambda x: x - x.mean()).std() + 1e-6
    out = {}
    for trip, r in trips.iterrows():
        base = normal[(normal.driver == r.driver) & (normal.index != trip)]
        if len(base) < 1:
            continue
        z = np.abs((r[FEATS].astype(float) - base[FEATS].mean()) / scale)
        out[trip] = float(np.sqrt(np.mean(np.sort(z.to_numpy())[-top_k:] ** 2)))
    return pd.Series(out, name="anomaly")


def explain(trip: pd.Series, base: pd.DataFrame, scale: pd.Series, k: int = 3) -> list[str]:
    """Plain-language top deviations for the driver view."""
    return explain_z((trip[FEATS].astype(float) - base[FEATS].mean()) / scale, k)


def main():
    w = pd.read_parquet(ROOT / "data/processed/uah_dev_windows.parquet")
    trips = trip_table(w)
    a = anomaly_scores(trips)
    t = trips.join(a, how="inner")
    risky = t.label != "NORMAL"
    auc = roc_auc_score(risky, t.anomaly)
    auc_aggr = roc_auc_score(t.label[t.label != "DROWSY"] == "AGGRESSIVE", t.anomaly[t.label != "DROWSY"])
    auc_drow = roc_auc_score(t.label[t.label != "AGGRESSIVE"] == "DROWSY", t.anomaly[t.label != "AGGRESSIVE"])
    print(t.sort_values(["driver", "anomaly"])[["driver", "label", "road", "anomaly"]].round(2).to_string())
    print(f"\nAUC risky vs normal: {auc:.3f} | aggressive vs normal: {auc_aggr:.3f} | drowsy vs normal: {auc_drow:.3f}")
    print(t.groupby("label").anomaly.median().round(2).to_dict())
    # fusion with the population score: does the personal view add information?
    sc = pd.read_csv(ROOT / "data/processed/uah_dev_trip_predictions.csv", index_col=0)
    j = t.join(sc[["p_risky"]], how="inner")
    rank = lambda s: s.rank(pct=True)
    fused = 0.5 * rank(j.p_risky) + 0.5 * rank(j.anomaly)
    res = dict(auc_personal=auc, auc_aggressive=auc_aggr, auc_drowsy=auc_drow,
               auc_population=roc_auc_score(j.label != "NORMAL", j.p_risky),
               auc_fused=roc_auc_score(j.label != "NORMAL", fused), n_trips=len(t))
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}, indent=1))
    (ROOT / "data/processed/personal_baseline_dev.json").write_text(json.dumps(res, indent=2))
    normal = trips[trips.label == "NORMAL"]
    scale = normal.groupby("driver")[FEATS].transform(lambda x: x - x.mean()).std() + 1e-6
    POP_SCALE_FILE.write_text(json.dumps(scale.round(6).to_dict(), indent=1))
    # example explanation
    normal = trips[trips.label == "NORMAL"]
    scale = normal.groupby("driver")[FEATS].transform(lambda x: x - x.mean()).std() + 1e-6
    ex = t[t.label == "AGGRESSIVE"].sort_values("anomaly").iloc[-1]
    print("\nexample:", ex.name, explain(ex, normal[normal.driver == ex.driver], scale))


if __name__ == "__main__":
    main()
