"""Trip-level features at ~1 Hz, for the learned trip risk model (trained on Grab safety labels).

Works for any source resampled to 1 Hz, so a Grab-trained model can be applied to UAH / PVS / own trips.
Input frame columns: t [s], speed [m/s, NaN if unknown], bearing [deg, NaN if unknown],
ax, ay, az [m/s^2 incl. gravity], gx, gy, gz [rad/s], optional accuracy [m].
Everything is phone-orientation invariant: gravity gives 'up', GPS gives longitudinal/lateral.
"""
import numpy as np
import pandas as pd

G = 9.81
CHUNK = 10          # seconds per gravity estimate (phone may be moved during a trip)


def _q(x, q):
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else np.nan


def _runs(mask: np.ndarray, min_len: int = 1) -> int:
    m = mask.astype(int)
    e = np.diff(np.r_[0, m, 0])
    return int(np.sum((np.flatnonzero(e == -1) - np.flatnonzero(e == 1)) >= min_len))


def clean_trip(d: pd.DataFrame, max_s: float = 6 * 3600) -> pd.DataFrame:
    d = d.sort_values("t").drop_duplicates("t")
    d = d[(d.t >= 0) & (d.t <= max_s)]
    d = d.assign(t=d.t - d.t.iloc[0]) if len(d) else d
    if "speed" in d:
        d.loc[d.speed < 0, "speed"] = np.nan
    if "bearing" in d:
        d.loc[(d.bearing < 0) | (d.bearing > 360), "bearing"] = np.nan
    return d


def to_1hz(d: pd.DataFrame, mode: str = "sample") -> pd.DataFrame:
    """Resample a higher-rate trip to 1 Hz the way Grab phones did: one instantaneous sample per second.
    Averaging ("mean") smooths away the peaks the risk model learned from Grab, so every trip looks calm."""
    s = np.floor(d.t).astype(int)
    if mode == "sample":
        out = d.groupby(s).head(1).copy()
        return out.reset_index(drop=True)
    agg = d.drop(columns=[c for c in ("bearing",) if c in d]).groupby(s).mean(numeric_only=True)
    if "bearing" in d:
        b = np.radians(d.bearing)
        agg["bearing"] = np.degrees(np.arctan2(np.sin(b).groupby(s).mean(), np.cos(b).groupby(s).mean())) % 360
    agg["t"] = agg.index.astype(float)
    return agg.reset_index(drop=True)


def trip_features(d: pd.DataFrame) -> dict:
    d = clean_trip(d)
    n = len(d)
    if n < 30:
        return {}
    t = d.t.to_numpy()
    dt = np.diff(t, prepend=t[0])
    dur = float(t[-1] - t[0])
    f = dict(duration_s=dur, n_samples=n, sample_rate=n / max(dur, 1))

    # ---- GPS speed, longitudinal and lateral acceleration
    v = d.speed.to_numpy(dtype=float) if "speed" in d else np.full(n, np.nan)
    v_med = pd.Series(v).rolling(3, center=True, min_periods=1).median().to_numpy()   # glitch filter
    gap = np.maximum(np.diff(t, prepend=t[0] - 1), 1e-3)
    a_lon = np.gradient(v_med, t) / G
    a_lon[(gap > 3) | ~np.isfinite(a_lon)] = np.nan
    km = float(np.nansum(v_med * np.minimum(dt, 3)) / 1000)
    f.update(distance_km=km, speed_mean=np.nanmean(v_med) * 3.6 if np.isfinite(v_med).any() else np.nan,
             speed_p50=_q(v_med, .5) * 3.6, speed_p90=_q(v_med, .9) * 3.6, speed_p99=_q(v_med, .99) * 3.6,
             speed_max=np.nanmax(v_med) * 3.6 if np.isfinite(v_med).any() else np.nan,
             speed_std=np.nanstd(v_med) * 3.6, speed_missing=float(np.mean(~np.isfinite(v))),
             frac_moving=float(np.nanmean(v_med > 2)),
             frac_over_80=float(np.nanmean(v_med > 80 / 3.6)), frac_over_100=float(np.nanmean(v_med > 100 / 3.6)),
             lon_p01=_q(a_lon, .01), lon_p05=_q(a_lon, .05), lon_p95=_q(a_lon, .95), lon_p99=_q(a_lon, .99),
             lon_std=float(np.nanstd(a_lon)))
    hours = max(dur / 3600, 1 / 60)
    for name, thr in (("brake", -0.3), ("brake_hard", -0.45)):
        f[f"{name}_per_h"] = _runs(np.nan_to_num(a_lon, nan=0) < thr) / hours
    f["accel_per_h"] = _runs(np.nan_to_num(a_lon, nan=0) > 0.3) / hours
    if "bearing" in d and d.bearing.notna().mean() > 0.5:
        b = np.unwrap(np.radians(d.bearing.ffill().bfill().to_numpy()))
        yaw_gps = np.gradient(b, t)
        yaw_gps[(gap > 3) | (v_med < 3)] = np.nan
        a_lat = np.abs(yaw_gps * v_med) / G
        f.update(lat_p95=_q(a_lat, .95), lat_p99=_q(a_lat, .99), turn_per_h=_runs(np.nan_to_num(a_lat) > 0.35) / hours)
    else:
        f.update(lat_p95=np.nan, lat_p99=np.nan, turn_per_h=np.nan)

    # ---- IMU: gravity per chunk -> up; linear accel magnitude; yaw rate; phone handling
    acc = d[["ax", "ay", "az"]].to_numpy(dtype=float)
    gyr = d[["gx", "gy", "gz"]].to_numpy(dtype=float) if {"gx", "gy", "gz"} <= set(d) else None
    chunk = (t // CHUNK).astype(int)
    grav = pd.DataFrame(acc).groupby(chunk).transform("median").to_numpy()
    gnorm = np.linalg.norm(grav, axis=1, keepdims=True) + 1e-9
    up = grav / gnorm
    lin = acc - grav
    vert = np.sum(lin * up, axis=1)
    hmag = np.linalg.norm(lin - vert[:, None] * up, axis=1)
    f.update(acc_g_norm=float(np.median(gnorm)), hmag_mean=float(np.mean(hmag)), hmag_p95=_q(hmag, .95),
             hmag_p99=_q(hmag, .99), hmag_max=float(np.max(hmag)), vert_std=float(np.std(vert)),
             vert_p99=_q(np.abs(vert), .99), hjerk_p99=_q(np.abs(np.diff(hmag)) / np.maximum(np.diff(t), 1e-3), .99),
             hmag_over_3_per_h=_runs(hmag > 3.0) / hours)
    # orientation changes between consecutive chunks: phone picked up / moved
    cu = pd.DataFrame(up).groupby(chunk).first().to_numpy()
    if len(cu) > 1:
        ang = np.degrees(np.arccos(np.clip(np.sum(cu[1:] * cu[:-1], axis=1), -1, 1)))
        f.update(reorient_30_per_h=float(np.sum(ang > 30)) / hours, reorient_mean_deg=float(np.mean(ang)))
    else:
        f.update(reorient_30_per_h=0.0, reorient_mean_deg=0.0)
    if gyr is not None:
        gmag = np.linalg.norm(gyr, axis=1)
        yaw = np.sum(gyr * up, axis=1)
        off_axis = np.sqrt(np.maximum(gmag ** 2 - yaw ** 2, 0))        # rotation not about vertical: handling
        f.update(gyro_p95=_q(gmag, .95), gyro_p99=_q(gmag, .99), yaw_p95=_q(np.abs(yaw), .95),
                 yaw_p99=_q(np.abs(yaw), .99), offaxis_p95=_q(off_axis, .95), offaxis_p99=_q(off_axis, .99),
                 handling_frac=float(np.mean(off_axis > 0.5)))
    if "accuracy" in d:
        f.update(gps_acc_mean=float(np.nanmean(d.accuracy)), gps_acc_p90=_q(d.accuracy.to_numpy(float), .9))
    # combined exposure-normalised intensity
    f["harsh_total_per_h"] = f["brake_per_h"] + f["accel_per_h"] + (f["turn_per_h"] if np.isfinite(f["turn_per_h"]) else 0)
    return f


FEATURES = None      # set on first use from the keys of trip_features


def grab_trip_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Grab raw rows (one booking) -> standard 1 Hz trip frame."""
    return pd.DataFrame({"t": raw.second, "speed": raw.Speed, "bearing": raw.Bearing,
                         "ax": raw.acceleration_x, "ay": raw.acceleration_y, "az": raw.acceleration_z,
                         "gx": raw.gyro_x, "gy": raw.gyro_y, "gz": raw.gyro_z, "accuracy": raw.Accuracy})
