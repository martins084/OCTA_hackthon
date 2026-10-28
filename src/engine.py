"""OCTA driving engine: raw phone recording -> events, score, reasons, context, insurer factor.

    from engine import analyze_trip, load_sensor_logger
    report = analyze_trip(*load_sensor_logger("path/to/export_folder"))

Inputs (any source; loaders below for Sensor Logger, PVS and UAH):
  gps: t [s], lat, lon, speed [km/h], optional course [deg]
  imu: t [s], ax, ay, az [m/s^2, incl. gravity], optional gx, gy, gz [rad/s]
  start: trip start as a timezone-aware pandas Timestamp (for weather / daylight)

CLI: python src/engine.py <sensor_logger_folder> | --pvs PVS_1 | --uah <trip_folder>
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

import events as ev
from context_risk import sun_elevation
from driving_features import FEATURES as STYLE_FEATURES
from driving_features import harsh_events, resample_trip, window_features
from score import Context, load_context_tables, score_trip

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"
G = 9.81
# GPS confirmation of IMU events: a real 0.25 g brake over 2 s changes speed by ~18 km/h.
MIN_DV_KMH = 8.0
MIN_TURN_DEG = 30.0

OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
UA = {"User-Agent": "octa-driving-prototype/0.1 (hackathon research)"}

# Latvian default limits by OSM road class when a way has no maxspeed tag (CSDD rules).
LV_DEFAULT_LIMIT = {"motorway": 110, "trunk": 90, "primary": 90, "secondary": 90, "tertiary": 90,
                    "unclassified": 90, "residential": 50, "living_street": 20, "service": 20}


# ---------------------------------------------------------------- loaders

def load_sensor_logger(folder: str | Path):
    """Sensor Logger (Android/iOS) export: Accelerometer.csv (user accel, no gravity), Gravity.csv,
    Gyroscope.csv, Location.csv; columns time [ns], seconds_elapsed, x/y/z ..."""
    f = Path(folder)
    acc = pd.read_csv(f / "Accelerometer.csv")
    imu = pd.DataFrame({"t": acc.seconds_elapsed, "ax": acc.x, "ay": acc.y, "az": acc.z})
    if (f / "Gravity.csv").exists():                       # re-add gravity so 'up' is known
        g = pd.read_csv(f / "Gravity.csv")
        for c in "xyz":
            imu["a" + c] += np.interp(imu.t, g.seconds_elapsed, g[c])
    if (f / "Gyroscope.csv").exists():
        gy = pd.read_csv(f / "Gyroscope.csv")
        for c in "xyz":
            imu["g" + c] = np.interp(imu.t, gy.seconds_elapsed, gy[c])
    loc = pd.read_csv(f / "Location.csv")
    gps = pd.DataFrame({"t": loc.seconds_elapsed, "lat": loc.latitude, "lon": loc.longitude,
                        "speed": loc.speed.clip(lower=0) * 3.6})
    if "bearing" in loc:
        gps["course"] = loc.bearing
    start = pd.to_datetime(acc.time.iloc[0], unit="ns", utc=True)
    return gps, imu, start


def load_pvs(trip: str | Path):
    x = pd.read_csv(Path(trip) / "dataset_gps_mpu_left.csv")
    t = x.timestamp - x.timestamp.iloc[0]
    imu = pd.DataFrame({"t": t, "ax": x.acc_x_dashboard, "ay": x.acc_y_dashboard, "az": x.acc_z_dashboard,
                        **{f"g{c}": np.radians(x[f"gyro_{c}_dashboard"]) for c in "xyz"}})
    g = x.drop_duplicates("timestamp_gps")[["timestamp_gps", "latitude", "longitude", "speed"]]
    gps = pd.DataFrame({"t": g.timestamp_gps - x.timestamp.iloc[0], "lat": g.latitude,
                        "lon": g.longitude, "speed": g.speed * 3.6})
    return gps, imu, pd.to_datetime(x.timestamp.iloc[0], unit="s", utc=True)


def load_uah(folder: str | Path):
    """UAH has no gyroscope and gravity-removed accel in g: IMU events are skipped."""
    from train_driving import read_uah_trip
    g, a = read_uah_trip(Path(folder))
    imu = pd.DataFrame({"t": a.t, "ax": a.x * G, "ay": a.y * G, "az": a.z * G + G})
    gps = g.rename(columns={})[["t", "lat", "lon", "speed", "course"] + (["speed_limit"] if "speed_limit" in g else [])]
    stamp = Path(folder).name.split("-")[0]
    return gps, imu, pd.Timestamp(stamp, tz="Europe/Madrid")


# ---------------------------------------------------------------- enrichment

def course_from_track(gps: pd.DataFrame) -> pd.Series:
    lat1, lon1 = np.radians(gps.lat.to_numpy()), np.radians(gps.lon.to_numpy())
    dlon = np.diff(lon1, append=lon1[-1])
    lat2 = np.append(lat1[1:], lat1[-1])
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    c = pd.Series(np.degrees(np.arctan2(y, x)) % 360, index=gps.index)
    c[gps.speed.to_numpy() < 3] = np.nan            # heading meaningless when stopped
    return c.ffill().bfill()


def speed_limits(gps: pd.DataFrame, timeout: int = 25) -> pd.Series | None:
    """Nearest OSM road's maxspeed (or Latvian default for its class) for each GPS point.
    Uses the public Overpass API, cached per trip bounding box. None if unavailable."""
    pad = 0.002
    bbox = (gps.lat.min() - pad, gps.lon.min() - pad, gps.lat.max() + pad, gps.lon.max() + pad)
    key = CACHE / ("osm_" + "_".join(f"{b:.4f}" for b in bbox) + ".json")
    try:
        if key.exists():
            data = json.loads(key.read_text())
        else:
            q = (f"[out:json][timeout:{timeout}];way[highway~\"^(motorway|trunk|primary|secondary|tertiary|"
                 f"unclassified|residential|living_street)(_link)?$\"]"
                 f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out tags geom;")
            data = None
            for attempt, url in enumerate(OVERPASS * 2):         # mirrors, two rounds
                try:
                    r = requests.post(url, data={"data": q}, timeout=timeout + 5, headers=UA)
                    if r.status_code == 200:
                        data = r.json()
                        break
                except requests.RequestException:
                    pass
                time.sleep(2 * attempt)
            if data is None:
                return None
            CACHE.mkdir(parents=True, exist_ok=True)
            key.write_text(json.dumps(data))
    except Exception:
        return None
    pts, lim = [], []
    for way in data.get("elements", []):
        tags = way.get("tags", {})
        cls = tags.get("highway", "").replace("_link", "")
        ms = str(tags.get("maxspeed", "")).replace("km/h", "").strip()
        try:
            v = float(ms.split(";")[0])                   # "50", "50;30" -> 50
        except ValueError:                                # missing, "LV:urban", "signals", ...
            v = {"LV:urban": 50, "LV:rural": 90, "LV:motorway": 110}.get(ms, LV_DEFAULT_LIMIT.get(cls))
        if v is None or "geometry" not in way:
            continue
        for p in way["geometry"]:
            pts.append((p["lat"], p["lon"]))
            lim.append(v)
    if not pts:
        return None
    from scipy.spatial import cKDTree
    pts = np.array(pts)
    scale = np.cos(np.radians(gps.lat.mean()))
    tree = cKDTree(np.c_[pts[:, 0], pts[:, 1] * scale])
    dist, idx = tree.query(np.c_[gps.lat, gps.lon * scale])
    out = pd.Series(np.array(lim)[idx], index=gps.index)
    out[dist > 0.0005] = np.nan                         # > ~50 m from any road: unknown
    return out


def trip_context(gps: pd.DataFrame, start: pd.Timestamp) -> tuple[Context, dict]:
    lat, lon = float(gps.lat.median()), float(gps.lon.median())
    times = start + pd.to_timedelta(gps.t.to_numpy(), unit="s")
    elev = sun_elevation(pd.DatetimeIndex(times).tz_convert("UTC").tz_localize(None), lat, lon)
    dark = float(np.mean(elev < -6))
    info = dict(lat=lat, lon=lon, start=start.isoformat(), dark_share=round(dark, 3))
    snow, frost, surface = "none", False, "dry"
    try:
        day = start.tz_convert("UTC").strftime("%Y-%m-%d")
        prev = (start - pd.Timedelta(days=1)).tz_convert("UTC").strftime("%Y-%m-%d")
        recent = (pd.Timestamp.now(tz="UTC") - start) < pd.Timedelta(days=5)
        url = ("https://api.open-meteo.com/v1/forecast" if recent
               else "https://archive-api.open-meteo.com/v1/archive")
        r = requests.get(url, timeout=20, params=dict(
            latitude=lat, longitude=lon, start_date=prev, end_date=day, timezone="UTC",
            daily="precipitation_sum,snowfall_sum,temperature_2m_min,temperature_2m_max",
            hourly="precipitation,snowfall,temperature_2m,snow_depth"))
        r.raise_for_status()
        dly = pd.DataFrame(r.json()["daily"])
        hr = pd.DataFrame(r.json()["hourly"])
        today, yday = dly.iloc[-1], dly.iloc[0]
        snow = "heavy" if today.snowfall_sum >= 5 else "light" if today.snowfall_sum >= 0.5 else "none"
        frost = bool(today.temperature_2m_min <= 0 and today.temperature_2m_max > -5 and yday.precipitation_sum >= 0.5)
        hr["time"] = pd.to_datetime(hr.time, utc=True)
        h = hr[hr.time <= start].tail(3)
        if len(h):
            if (h.snow_depth.iloc[-1] or 0) >= 0.01 and h.temperature_2m.iloc[-1] <= 1:
                surface = "snow_ice"
            elif h.precipitation.sum() >= 0.1:
                surface = "wet"
        info.update(weather_source=url.split("//")[1].split("/")[0], precipitation_mm=float(today.precipitation_sum),
                    snowfall_cm=float(today.snowfall_sum), tmin=float(today.temperature_2m_min), surface=surface)
    except Exception as e:                                    # offline: daylight-only context
        info["weather_error"] = type(e).__name__
    return Context(dark_share=dark, snow_day=snow, frost_after_wet=frost, surface=surface), info


# ---------------------------------------------------------------- models

def _load(model_dir: Path):
    meta = json.loads((model_dir / "meta.json").read_text())
    return lgb.Booster(model_file=str(model_dir / "model.txt")), meta


def detect_imu_events(imu: pd.DataFrame, gps: pd.DataFrame) -> pd.DataFrame:
    """Frozen IMU event model; brake vs accel decided by GPS speed change at the event."""
    booster, meta = _load(ev.MODEL_DIR)
    a, _ = ev.prepare(imu[ev.IMU], meta.get("smooth_s"))
    w = ev.add_relative(ev.window_features(a).assign(session="trip"))
    p = booster.predict(w[meta["features"]])
    pred = np.array(meta["classes"], dtype=object)[np.asarray(p).argmax(1)]
    events = ev.merge_predictions(w, pred)
    rows = []
    for r in events.itertuples():
        g = gps[(gps.t >= r.t0 - 1) & (gps.t <= r.t1 + 1)]
        kind = r.label
        dv = float(g.speed.iloc[-1] - g.speed.iloc[0]) if len(g) >= 2 else np.nan
        dh = float(np.abs(((g.course.diff() + 180) % 360 - 180).sum())) if len(g) >= 2 else np.nan
        confirmed = None                                     # None = GPS can't tell (lane change)
        if kind in ("speed_change", "brake", "accel"):
            kind = "harsh_brake" if (dv < 0 if not np.isnan(dv) else kind == "brake") else "harsh_accel"
            confirmed = bool(abs(dv) >= MIN_DV_KMH) if not np.isnan(dv) else False
        elif kind == "turn":
            confirmed = bool(dh >= MIN_TURN_DEG and g.speed.mean() >= 10) if not np.isnan(dh) else False
        mid = gps.iloc[(gps.t - (r.t0 + r.t1) / 2).abs().argsort()[:1]]
        rows.append(dict(type=kind, t0=round(r.t0, 1), t1=round(r.t1, 1), speed_change_kmh=round(dv, 1),
                         heading_change_deg=round(dh, 0), gps_confirmed=confirmed,
                         lat=float(mid.lat.iloc[0]), lon=float(mid.lon.iloc[0])))
    return pd.DataFrame(rows, columns=["type", "t0", "t1", "speed_change_kmh", "heading_change_deg",
                                       "gps_confirmed", "lat", "lon"])


def analyze_trip(gps: pd.DataFrame, imu: pd.DataFrame, start: pd.Timestamp,
                 driver: dict | None = None, use_osm: bool = True, detail: bool = False) -> dict:
    """detail=True adds what a replay UI needs: GPS track, score timeline, what-if context variants."""
    gps = gps.sort_values("t").reset_index(drop=True)
    if "course" not in gps or gps.course.isna().all():
        gps["course"] = course_from_track(gps)
    if "speed_limit" not in gps or gps.speed_limit.isna().all():
        lim = speed_limits(gps) if use_osm else None
        gps["speed_limit"] = lim.interpolate(limit_direction="both") if lim is not None else np.nan
    acc_g = pd.DataFrame({"t": imu.t, **{c: (imu["a" + c] - imu["a" + c].median()) / G for c in "xyz"}})
    resampled = resample_trip(gps, acc_g)
    windows = window_features(resampled)
    if windows.empty:
        return dict(error="trip too short or vehicle not moving (need >= 60 s of driving)")

    style, smeta = _load(ROOT / "models/style_gbm")
    proba = np.asarray(style.predict(windows[smeta["features"]]))
    p_style = dict(zip(smeta["classes"], proba.mean(0).round(3).tolist()))
    p_risky = 1 - p_style["NORMAL"]

    ctx, ctx_info = trip_context(gps, start)
    s = score_trip(windows, p_risky, ctx, load_context_tables())

    imu_events = (detect_imu_events(imu, gps) if {"gx", "gy", "gz"} <= set(imu.columns)
                  else pd.DataFrame(columns=["type"]))
    km = float(windows.km.sum() / 4)                      # 60 s windows at 15 s step -> 4x overlap
    report = dict(
        score=s.score, insurer_multiplier=round(s.insurer_multiplier, 3), reasons=s.reasons,
        penalties=s.penalties, style_probabilities=p_style,
        trip=dict(duration_min=round(float(gps.t.iloc[-1] - gps.t.iloc[0]) / 60, 1), distance_km=round(km, 1),
                  mean_speed_kmh=round(float(gps.speed[gps.speed > 5].mean()), 1),
                  speed_limit_coverage=round(float(gps.speed_limit.notna().mean()), 2)),
        context=dict(**ctx_info, multiplier=s.context_multiplier),
        imu_events=imu_events.to_dict("records"),
        imu_event_counts=(imu_events[imu_events.gps_confirmed.fillna(True).astype(bool)].type.value_counts().to_dict()
                          if len(imu_events) else {}),
        imu_events_rejected_by_gps=int((imu_events.gps_confirmed == False).sum()) if len(imu_events) else 0,  # noqa: E712
    )
    if driver:
        report["premium"] = premium(driver, s.insurer_multiplier)
    if detail:
        report.update(_detail(gps, windows, proba, smeta, ctx))
        ge = harsh_events(resampled)
        idx = np.searchsorted(gps.t.to_numpy(), ge.t0.to_numpy()).clip(0, len(gps) - 1)
        ge["lat"], ge["lon"] = gps.lat.to_numpy()[idx], gps.lon.to_numpy()[idx]
        report["gps_events"] = ge.round(6).to_dict("records")
    return report


def _detail(gps, windows, proba, smeta, ctx) -> dict:
    tables = load_context_tables()
    i_norm = smeta["classes"].index("NORMAL")
    step = max(1, int(round(2 / max(gps.t.diff().median(), 0.1))))      # ~2 s track resolution
    tr = gps.iloc[::step]
    track = [dict(t=round(float(r.t), 1), lat=round(float(r.lat), 6), lon=round(float(r.lon), 6),
                  speed=round(float(r.speed), 1),
                  limit=None if pd.isna(r.speed_limit) else float(r.speed_limit)) for r in tr.itertuples()]
    timeline = []
    w_end = windows.t0.to_numpy() + 60
    for t in np.arange(gps.t.iloc[0] + 90, gps.t.iloc[-1] + 30, 30):
        m = w_end <= t
        if m.sum() < 1:
            continue
        p_r = float(1 - proba[m, i_norm].mean())
        sc = score_trip(windows[m], p_r, ctx, tables)
        timeline.append(dict(t=round(float(t), 1), score=sc.score, p_risky=round(p_r, 3)))
    p_r = float(1 - proba[:, i_norm].mean())
    variants = {}
    for name, c in [("as_driven", ctx), ("at_night", Context(dark_share=1.0)),
                    ("snowy_night", Context(dark_share=1.0, snow_day="light", frost_after_wet=True, surface="snow_ice"))]:
        sc = score_trip(windows, p_r, c, tables)
        variants[name] = dict(score=sc.score, context_multiplier=sc.context_multiplier,
                              insurer_multiplier=round(sc.insurer_multiplier, 3), reasons=sc.reasons,
                              # behaviour x conditions: what the insurer carries for this trip
                              trip_risk=round(sc.insurer_multiplier * sc.context_multiplier, 3))
    return dict(track=track, timeline=timeline, what_if=variants)


def premium(driver: dict, telematics_multiplier: float, base_premium_eur: float = 120.0) -> dict:
    """Expected claim frequency from the MTPL GBM (driver/vehicle profile) x telematics factor.
    base_premium_eur is the price at the portfolio-average frequency (illustrative)."""
    booster, meta = _load(ROOT / "models/insurer_gbm")
    row = pd.DataFrame([driver])
    for c in meta["categorical"]:
        row[c] = row[c].astype("category")
    freq = float(booster.predict(row[meta["features"]])[0])
    avg = 0.1005                                          # freMTPL2 training-portfolio frequency
    return dict(expected_claims_per_year=round(freq, 4), profile_factor=round(freq / avg, 3),
                telematics_factor=round(telematics_multiplier, 3),
                indicative_premium_eur=round(base_premium_eur * freq / avg * telematics_multiplier, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--pvs")
    ap.add_argument("--uah")
    ap.add_argument("--no-osm", action="store_true")
    args = ap.parse_args()
    if args.pvs:
        from splits import assert_path_allowed
        p = ROOT / "data/raw/pvs" / args.pvs
        assert_path_allowed(p)
        trip = load_pvs(p)
    elif args.uah:
        from splits import assert_path_allowed
        assert_path_allowed(Path(args.uah))
        trip = load_uah(args.uah)
    else:
        trip = load_sensor_logger(args.path)
    print(json.dumps(analyze_trip(*trip, use_osm=not args.no_osm), indent=2, default=str))


if __name__ == "__main__":
    main()
