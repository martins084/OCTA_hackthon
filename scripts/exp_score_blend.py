"""Dev check: which trip score ranks risky trips best on UAH dev drivers?
expert = hand-weighted score (out-of-fold style probabilities); learned = Grab-trained behaviour model
(never saw UAH); blend = mean of the two percentile ranks."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import engine as E  # noqa: E402
import train_driving as T  # noqa: E402
from score import score_trip  # noqa: E402

df = T.build_dataset(T.uah_trip_dirs(T.ROOT / "data/raw/uah", "dev"))
y = df.label.map({c: i for i, c in enumerate(T.CLASSES)})
trips = T.trip_table(df, T.lodo(df, y, 3))
rows = []
for f in T.uah_trip_dirs(T.ROOT / "data/raw/uah", "dev"):
    gps, imu, _ = E.load_uah(f)
    lr = E.learned_risk(gps, imu)
    w = df[df.trip == f.name]
    rows.append(dict(trip=f.name, driver=w.driver.iloc[0], label=w.label.iloc[0],
                     expert=score_trip(w, trips.loc[f.name, "p_risky"]).score, learned=lr["safer_than_pct"]))
r = pd.DataFrame(rows)
r["blend"] = 0.5 * r.expert.rank(pct=True) + 0.5 * r.learned.rank(pct=True)
for c in ("expert", "learned", "blend"):
    auc = roc_auc_score(r.label == "NORMAL", r[c])
    aggr = roc_auc_score(r.label[r.label != "DROWSY"] == "NORMAL", r[c][r.label != "DROWSY"])
    ok = pairs = 0
    for _, g in r.groupby("driver"):
        for a in g[g.label == "NORMAL"][c]:
            for b in g[g.label != "NORMAL"][c]:
                pairs += 1; ok += a > b
    print(f"{c:8s} AUC normal-vs-risky {auc:.3f} | normal-vs-aggressive {aggr:.3f} | within-driver {ok}/{pairs}")
print(r.groupby("label")[["expert", "learned"]].mean().round(1))
