"""Dev comparison: self-supervised Transformer vs GBM, and their ensemble (out-of-fold predictions only).

The Transformer sees zero-padded 30-min sequences, so it can infer trip length from the padding: it is
compared with the *full* GBM (which also uses duration), not the behaviour-only one.
Paired bootstrap gives a CI for the AUC difference.
Usage: python scripts/exp_ssl_ensemble.py <ssl_oof.npz> [more_ssl_oof.npz ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def paired_diff(y, a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return float(np.mean(d)), float(np.quantile(d, .025)), float(np.quantile(d, .975))


def main(paths):
    import subprocess
    # full-feature GBM out-of-fold predictions (re-created if missing)
    oof_full = ROOT / "data/processed/trip_risk_oof.parquet"
    if not oof_full.exists():
        subprocess.run([sys.executable, str(ROOT / "src/train_trip_risk.py")], check=True, cwd=ROOT / "src")
    g = pd.read_parquet(oof_full).rename(columns={"oof": "gbm_full"})
    gb = pd.read_parquet(ROOT / "data/processed/trip_risk_oof_behaviour.parquet").rename(columns={"oof": "gbm_behaviour"})
    df = g.merge(gb[["bookingID", "gbm_behaviour"]], on="bookingID")
    for p in paths:
        z = np.load(p)
        tag = Path(p).stem.replace("ssl_oof", "") or "_kaggle"
        s = pd.DataFrame({"bookingID": z["bookingID"], f"ssl_scratch{tag}": z["scratch"], f"ssl_pre{tag}": z["pretrained"]})
        df = df.merge(s, on="bookingID")
    y = df.y.to_numpy()
    rank = lambda c: df[c].rank(pct=True).to_numpy()
    print(f"{len(df)} dev trips with all predictions")
    for c in [c for c in df.columns if c not in ("bookingID", "y")]:
        print(f"  {c:24s} AUC {roc_auc_score(y, df[c]):.4f}")
    for c in [c for c in df.columns if c.startswith("ssl_pre")]:
        for w in (0.3, 0.5):
            ens = (1 - w) * rank("gbm_full") + w * rank(c)
            m, lo, hi = paired_diff(y, ens, df.gbm_full.to_numpy())
            print(f"  ensemble gbm_full + {w:.1f}*{c}: AUC {roc_auc_score(y, ens):.4f}  "
                  f"gain vs gbm_full {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        m, lo, hi = paired_diff(y, df[c].to_numpy(), df[c.replace("pre", "scratch")].to_numpy())
        print(f"  pre-training gain ({c} vs scratch): {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main(sys.argv[1:])
