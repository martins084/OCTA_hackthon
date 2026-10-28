"""Export the sealed Grab trips as 1 Hz sequences into the blind folder (inputs only; nothing is scored here).
Only src/blind_eval.py-style code may read the result."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from build_grab import DTYPES, RAW, sequence
from splits import ROOT, SPLITS, blind_access, grab_is_blind
from trip_features import grab_trip_frame


def one(bid, g):
    try:
        return bid, sequence(grab_trip_frame(g))
    except Exception:                                   # noqa: BLE001
        return bid, None


def main():
    with blind_access():
        raw = pd.concat([pd.read_csv(f, dtype=DTYPES) for f in sorted(glob.glob(str(RAW / "features/*.csv")))],
                        ignore_index=True)
        raw = raw[grab_is_blind_fast(raw.bookingID)]
        res = Parallel(n_jobs=10, batch_size=64)(delayed(one)(b, g) for b, g in raw.groupby("bookingID", sort=False))
        res = [(b, s) for b, s in res if s is not None]
        out = ROOT / SPLITS["grab"]["blind_path"] / "grab_blind_seq.npz"
        np.savez_compressed(out, X=np.stack([s for _, s in res]), bookingID=np.array([b for b, _ in res]))
    print(f"{len(res)} blind sequences -> {out}")


def grab_is_blind_fast(ids: pd.Series) -> np.ndarray:
    u = ids.unique()
    m = dict(zip(u, grab_is_blind(u)))
    return ids.map(m).to_numpy()


if __name__ == "__main__":
    main()
