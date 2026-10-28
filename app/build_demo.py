"""Pre-compute demo trips through the full engine -> app/static/data/*.json (the demo runs offline).

Replay trips: UAH driver D3 (never seen by the style model; its blind evaluation is already logged)
and one PVS trip per vehicle. Portfolio: every UAH trip, flagged when the driver was in training.
Reading blind folders here is display only, after their blind runs (docs/BLIND_LOG.md); nothing is tuned.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import engine as E  # noqa: E402
from splits import SPLITS, blind_access, pvs_trip_dirs, uah_trip_dirs  # noqa: E402

OUT = ROOT / "app/static/data"
PROFILE = dict(Area="C", VehBrand="B12", VehGas="Regular", Region="R11", VehPower=6, VehAge=5,
               DrivAge=35, BonusMalus=50, LogDensity=float(np.log(1000)))   # same profile for all: isolates telematics
STYLE_TRAIN = set(json.loads((ROOT / "models/style_gbm/meta.json").read_text())["trained_on_drivers"])


def run(trip_id, title, subtitle, loader, path, tags):
    gps, imu, start = loader(path)
    rep = E.analyze_trip(gps, imu, start, driver=PROFILE, detail=True)
    rep.update(id=trip_id, title=title, subtitle=subtitle, tags=tags)
    (OUT / f"trip_{trip_id}.json").write_text(json.dumps(rep, default=float))
    print(f"{trip_id:40s} score {rep['score']:5.1f}  events {rep['imu_event_counts']}")
    return {k: rep[k] for k in ("id", "title", "subtitle", "tags", "score", "insurer_multiplier")} | dict(
        km=rep["trip"]["distance_km"], minutes=rep["trip"]["duration_min"], premium=rep["premium"],
        style=rep["style_probabilities"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trips = []
    with blind_access():
        uah = uah_trip_dirs(ROOT / "data/raw/uah", "blind") + uah_trip_dirs(ROOT / "data/raw/uah", "dev")
        pvs = pvs_trip_dirs("dev") + pvs_trip_dirs("blind")
    for f in uah:
        _, km, drv, style, road = f.name.split("-")
        label = style.rstrip("0123456789").title()
        seen = drv in STYLE_TRAIN
        trips.append(run(f"uah_{f.name}", f"Driver {drv}: {label.lower()} drive",
                         f"{road.title()} road, {km}, Madrid (UAH-DriveSet)", E.load_uah, f,
                         dict(source="UAH-DriveSet", driver=drv, label=label, road=road.lower(),
                              replay=drv in SPLITS["uah"]["blind_drivers"], seen_in_training=seen)))
    for f in [p for p in pvs if p.name in ("PVS_1", "PVS_4", "PVS_7")]:
        n = int(f.name.split("_")[1])
        trips.append(run(f"pvs_{f.name}", f"Vehicle {(n - 1) // 3 + 1}: everyday drive",
                         "Asphalt, cobblestone and dirt roads with speed bumps, Brazil (PVS)", E.load_pvs, f,
                         dict(source="PVS", driver=f"PVS-V{(n - 1) // 3 + 1}", label="Normal", road="mixed",
                              replay=True, seen_in_training=False)))   # style model never saw PVS
    (OUT / "trips.json").write_text(json.dumps(trips, indent=1, default=float))
    print(f"\n{len(trips)} trips -> {OUT}")


if __name__ == "__main__":
    main()
