"""Common trip format + phone-orientation alignment to the car frame (no GPS required).

Input: t [s], ax, ay, az (m/s^2 or g, with or without gravity), gx, gy, gz (rad/s or deg/s), any axes.
Output adds: lon (forward +), lat (left +), vert, yaw_rate (left turn +), all in m/s^2 and rad/s.

Alignment:
  up     = gravity direction (median accel) if gravity is present, else the main rotation axis of
           the gyro (cars rotate almost only about the vertical axis).
  left   = horizontal direction whose acceleration best tracks the yaw rate (centripetal accel
           points into the turn: a_lat = v * omega with v > 0).
  forward = left x up. Flipping up also flips left, so forward is unaffected by the up-sign ambiguity.
"""
import numpy as np
import pandas as pd

HZ = 20
G = 9.81
CHUNK_S = 120       # seconds per alignment chunk


def resample(df: pd.DataFrame, hz: int = HZ) -> pd.DataFrame:
    df = df.sort_values("t").drop_duplicates("t")
    t = np.arange(df.t.iloc[0], df.t.iloc[-1], 1 / hz)
    cols = [c for c in df.columns if c != "t" and pd.api.types.is_numeric_dtype(df[c])]
    return pd.DataFrame({"t": t, **{c: np.interp(t, df.t, df[c]) for c in cols}})


def to_si(df: pd.DataFrame) -> pd.DataFrame:
    """Detect units: accel in g -> m/s^2, gyro in deg/s -> rad/s."""
    df = df.copy()
    acc = df[["ax", "ay", "az"]].to_numpy()
    mag = np.median(np.linalg.norm(acc, axis=1))
    if 0.5 < mag < 1.5 or (mag < 0.5 and np.abs(acc).max() < 4):   # with gravity ~1 g, or small linear accel in g
        df[["ax", "ay", "az"]] *= G
    gyr = df[["gx", "gy", "gz"]].to_numpy()
    if np.quantile(np.abs(gyr), 0.99) > 8:                          # >8 rad/s sustained is implausible in a car
        df[["gx", "gy", "gz"]] = np.radians(gyr)
    return df


def _smooth(x: np.ndarray, seconds: float, hz: int = HZ) -> np.ndarray:
    n = max(1, int(seconds * hz))
    return pd.DataFrame(x).rolling(n, center=True, min_periods=1).mean().to_numpy(copy=True)


def align(df: pd.DataFrame, hz: int = HZ, earth_frame: bool = False) -> tuple[pd.DataFrame, dict]:
    """earth_frame=True: data already rotated to East/North/Up (e.g. Android 'world' frame), so the
    car's forward axis turns with its heading; we de-rotate by the integrated yaw before aligning."""
    acc = df[["ax", "ay", "az"]].to_numpy()
    gyr = df[["gx", "gy", "gz"]].to_numpy()
    g_vec = np.median(acc, axis=0)
    has_gravity = np.linalg.norm(g_vec) > 5
    if has_gravity:
        up = g_vec / np.linalg.norm(g_vec)
        lin = acc - g_vec
    else:
        gs = _smooth(gyr, 0.5, hz)
        up = np.linalg.eigh(np.cov(gs.T))[1][:, -1]
        lin = acc - acc.mean(axis=0)
    omega = gyr @ up                                               # yaw rate (sign follows up)
    h = lin - np.outer(lin @ up, up)                               # horizontal accel
    e1 = np.cross(up, [1, 0, 0] if abs(up[0]) < 0.9 else [0, 1, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    h2 = np.c_[h @ e1, h @ e2]
    if earth_frame:
        # heading psi(t) = integral of yaw rate; rotate horizontal accel by -psi into a car-fixed frame.
        # Subtract the median yaw rate first: a trip's true mean yaw rate is ~0, a gyro bias is not,
        # and an uncorrected 0.01 rad/s bias would drift the heading by radians over a trip.
        psi = np.cumsum(omega - np.median(omega)) / hz
        c, s_ = np.cos(-psi), np.sin(-psi)
        h2 = np.c_[c * h2[:, 0] - s_ * h2[:, 1], s_ * h2[:, 0] + c * h2[:, 1]]

    # left = direction d in the plane maximising corr(h2.d, omega), on 1 s smoothed signals.
    hs, ws = _smooth(h2, 1.0, hz), _smooth(omega[:, None], 1.0, hz)[:, 0]

    def left_dir(idx):
        turning = idx & (np.abs(ws) > np.quantile(np.abs(ws[idx]), 0.5))
        beta, *_ = np.linalg.lstsq(hs[turning], ws[turning], rcond=None)
        return beta / (np.linalg.norm(beta) + 1e-12)

    everywhere = np.ones(len(ws), bool)
    left_global = left_dir(everywhere)
    # Chunked re-estimation: handles heading drift (earth frame) and a phone moved mid-trip.
    # A chunk needs real turning (yaw > 0.1 rad/s for >= 5 s) to trust its own estimate.
    n = CHUNK_S * hz
    left2_t = np.tile(left_global, (len(ws), 1))
    for s in range(0, len(ws), n):
        idx = np.zeros(len(ws), bool)
        idx[s:s + n] = True
        if np.sum(np.abs(ws[idx]) > 0.1) >= 5 * hz:
            left2_t[idx] = left_dir(idx)
    left2_t = _smooth(left2_t, 10.0, hz)                       # no jumps at chunk borders
    left2_t /= np.linalg.norm(left2_t, axis=1, keepdims=True) + 1e-12
    fwd2_t = np.c_[left2_t[:, 1], -left2_t[:, 0]]              # left x up, in (e1, e2) coordinates
    left = left_global[0] * e1 + left_global[1] * e2
    fwd = np.cross(left, up)
    out = df.copy()
    out["lon"] = np.sum(h2 * fwd2_t, axis=1)
    out["lat"] = np.sum(h2 * left2_t, axis=1)
    out["vert"], out["yaw_rate"] = lin @ up, omega
    t = np.abs(ws) > np.quantile(np.abs(ws), 0.5)
    quality = float(np.corrcoef(np.sum(hs * left2_t, axis=1)[t], ws[t])[0, 1])
    return out, dict(up=up.round(3).tolist(), forward=fwd.round(3).tolist(), left=left.round(3).tolist(),
                     has_gravity=bool(has_gravity), turn_corr=round(quality, 3))


def random_rotation(seed: int) -> np.ndarray:
    q = np.random.default_rng(seed).normal(size=4)
    a, b, c, d = q / np.linalg.norm(q)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def rotate(df: pd.DataFrame, R: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out[["ax", "ay", "az"]] = df[["ax", "ay", "az"]].to_numpy() @ R.T
    out[["gx", "gy", "gz"]] = df[["gx", "gy", "gz"]].to_numpy() @ R.T
    return out
