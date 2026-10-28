"""Window features for driving-style scoring, from phone GPS + accelerometer.

Every feature is phone-orientation invariant: longitudinal/lateral acceleration come from GPS
(speed change, heading change x speed); the accelerometer is used only via its magnitude. This
is deliberate: in the app the phone can sit anywhere in the car.

Inputs per trip (any source, resampled to 10 Hz here):
  gps: t [s], speed [km/h], course [deg], (optional) speed_limit [km/h]
  acc: t [s], x, y, z [g], gravity removed
"""
import numpy as np
import pandas as pd

HZ = 10
G = 9.81
HARSH_LON_G = 0.25   # harsh braking / acceleration threshold on GPS-derived accel
HARSH_LAT_G = 0.30   # harsh cornering


def resample_trip(gps: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    t = np.arange(max(gps.t.min(), acc.t.min()), min(gps.t.max(), acc.t.max()), 1 / HZ)
    d = pd.DataFrame({"t": t})
    # 3-fix median: single-fix GPS speed glitches otherwise look like harsh brakes (PVS dev: 10-18/100 km)
    speed = gps.speed.rolling(3, center=True, min_periods=1).median()
    d["speed"] = np.interp(t, gps.t, speed)
    course = np.degrees(np.unwrap(np.radians(gps.course.to_numpy())))
    d["course"] = np.interp(t, gps.t, course)
    if "speed_limit" in gps and gps.speed_limit.notna().any():
        lim = gps.dropna(subset=["speed_limit"])
        d["speed_limit"] = np.interp(t, lim.t, lim.speed_limit)
    else:
        d["speed_limit"] = np.nan
    for c in ("x", "y", "z"):
        d[c] = np.interp(t, acc.t, acc[c])

    v = d.speed.rolling(2 * HZ, center=True, min_periods=1).mean() / 3.6           # m/s, 2 s smoothing
    d["lon_g"] = np.gradient(v, 1 / HZ) / G
    # 3 s heading smoothing: 1 Hz GPS course jitter x motorway speed otherwise looks like sharp turns
    yaw = np.gradient(d.course.rolling(3 * HZ, center=True, min_periods=1).mean(), 1 / HZ)
    d["lat_g"] = (np.radians(yaw) * v / G).clip(-1.5, 1.5)
    d["acc_mag"] = np.sqrt(d.x ** 2 + d.y ** 2 + d.z ** 2)
    d["jerk_mag"] = np.abs(np.gradient(d.acc_mag, 1 / HZ))
    return d


def _events(mask: pd.Series, min_s: float = 1.0) -> int:
    """Number of separate runs of True lasting >= min_s (real manoeuvres persist, GPS glitches don't)."""
    m = mask.to_numpy().astype(int)
    if not len(m):
        return 0
    edges = np.diff(np.r_[0, m, 0])
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int(np.sum(lengths >= min_s * HZ))


def harsh_events(d: pd.DataFrame, min_s: float = 1.0) -> pd.DataFrame:
    """Located GPS-derived harsh events (same thresholds/persistence as the window counts)."""
    rows = []
    for kind, mask, sig in (("harsh_brake", d.lon_g < -HARSH_LON_G, d.lon_g),
                            ("harsh_accel", d.lon_g > HARSH_LON_G, d.lon_g),
                            ("harsh_turn", d.lat_g.abs() > HARSH_LAT_G, d.lat_g.abs())):
        m = mask.to_numpy().astype(int)
        edges = np.diff(np.r_[0, m, 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a >= min_s * HZ:
                peak = float(sig.iloc[a:b].abs().max())
                rows.append(dict(type=kind, t0=float(d.t.iloc[a]), t1=float(d.t.iloc[b - 1]),
                                 peak_g=round(peak, 2), speed_kmh=round(float(d.speed.iloc[a]), 1)))
    return pd.DataFrame(rows, columns=["type", "t0", "t1", "peak_g", "speed_kmh"])


def window_features(d: pd.DataFrame, win_s: int = 60, step_s: int = 15) -> pd.DataFrame:
    rows = []
    n, step = win_s * HZ, step_s * HZ
    for start in range(0, len(d) - n + 1, step):
        w = d.iloc[start:start + n]
        moving = w.speed > 5
        if moving.mean() < 0.5:     # parked / traffic jam: not informative about style
            continue
        km = w.speed.mean() / 3600 * win_s
        over = (w.speed - w.speed_limit).clip(lower=0)
        mag = w.acc_mag - w.acc_mag.rolling(5 * HZ, min_periods=1).mean()   # remove slow drift
        spec = np.abs(np.fft.rfft(mag.to_numpy() - mag.mean())) ** 2
        f = np.fft.rfftfreq(len(mag), 1 / HZ)
        rows.append(dict(
            t0=w.t.iloc[0], km=km,
            speed_mean=w.speed.mean(), speed_std=w.speed.std(), speed_max=w.speed.max(),
            over_limit_mean=over.mean(), frac_over_limit_10=(w.speed > w.speed_limit * 1.1).mean(),
            speed_ratio_mean=(w.speed / w.speed_limit).mean(), speed_ratio_p95=(w.speed / w.speed_limit).quantile(0.95),
            frac_over_limit_20=(w.speed > w.speed_limit + 20).mean(),
            lon_p05=w.lon_g.quantile(0.05), lon_p95=w.lon_g.quantile(0.95), lon_std=w.lon_g.std(),
            lon_jerk_std=np.gradient(w.lon_g, 1 / HZ).std(),
            lat_abs_p95=w.lat_g.abs().quantile(0.95), lat_std=w.lat_g.std(),
            lat_x_speed=(w.lat_g.abs() * w.speed).mean(),
            harsh_brake=_events(w.lon_g < -HARSH_LON_G), harsh_accel=_events(w.lon_g > HARSH_LON_G),
            harsh_turn=_events(w.lat_g.abs() > HARSH_LAT_G),
            mag_std=mag.std(), mag_p95=mag.abs().quantile(0.95), mag_max=mag.abs().max(),
            jerk_p95=w.jerk_mag.quantile(0.95),
            band_low=spec[(f > 0.1) & (f < 0.5)].sum(), band_mid=spec[(f >= 0.5) & (f < 2)].sum(),
            band_high=spec[f >= 2].sum(),
        ))
    out = pd.DataFrame(rows)
    if len(out):
        tot = out[["band_low", "band_mid", "band_high"]].sum(axis=1) + 1e-12
        for b in ("band_low", "band_mid", "band_high"):
            out[b + "_frac"] = out[b] / tot
    return out


FEATURES = ["speed_mean", "speed_std", "speed_max", "over_limit_mean", "frac_over_limit_10",
            "speed_ratio_mean", "speed_ratio_p95",
            "frac_over_limit_20", "lon_p05", "lon_p95", "lon_std", "lon_jerk_std", "lat_abs_p95",
            "lat_std", "lat_x_speed", "harsh_brake", "harsh_accel", "harsh_turn", "mag_std", "mag_p95",
            "mag_max", "jerk_p95", "band_low_frac", "band_mid_frac", "band_high_frac"]
