"""Demo server: replay pre-computed trips, analyse uploads, analyse trips recorded in the browser.

Run:  .venv/Scripts/python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
Open: http://localhost:8000   (phone sensors in the browser need HTTPS; see README)
"""
import io
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import engine as E  # noqa: E402

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="OCTA driving engine demo")
PROFILE = dict(Area="C", VehBrand="B12", VehGas="Regular", Region="R11", VehPower=6, VehAge=5,
               DrivAge=35, BonusMalus=50, LogDensity=float(np.log(1000)))


def _clean(obj):
    """JSON-safe: numpy scalars -> python, NaN -> None."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _report(gps, imu, start, title):
    rep = E.analyze_trip(gps, imu, start, driver=PROFILE, detail=True)
    if "error" in rep:
        raise HTTPException(422, rep["error"])
    rep.update(id="upload", title=title, subtitle="Your recording", tags=dict(source="upload", replay=True))
    return _clean(rep)


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    """Sensor Logger export (.zip) -> full report."""
    data = await file.read()
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(413, "file too large")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for m in z.infolist():                       # only flat CSVs; no path traversal
                    name = Path(m.filename).name
                    if name.endswith(".csv") and not m.is_dir():
                        (Path(tmp) / name).write_bytes(z.read(m))
        except zipfile.BadZipFile:
            raise HTTPException(400, "expected a Sensor Logger .zip export")
        need = {"Accelerometer.csv", "Location.csv"}
        if not need <= {p.name for p in Path(tmp).iterdir()}:
            raise HTTPException(400, f"zip must contain {sorted(need)} (Sensor Logger export)")
        return _report(*E.load_sensor_logger(tmp), title=file.filename or "Uploaded trip")


class LiveTrip(BaseModel):
    """Recorded by static/record.js: accelerationIncludingGravity [m/s^2], rotationRate [deg/s], GPS."""
    start_ms: float
    acc: list[list[float]]      # [t_s, x, y, z]
    gyro: list[list[float]]     # [t_s, alpha, beta, gamma] (deg/s, about z, x, y)
    gps: list[list[float]]      # [t_s, lat, lon, speed_mps (or -1), heading_deg (or -1)]


@app.post("/api/live")
def analyze_live(trip: LiveTrip):
    if len(trip.gps) < 30 or len(trip.acc) < 600:
        raise HTTPException(422, "recording too short (need at least ~1 minute of driving)")
    a = np.array(trip.acc)
    imu = pd.DataFrame({"t": a[:, 0], "ax": a[:, 1], "ay": a[:, 2], "az": a[:, 3]})
    if trip.gyro:
        g = np.array(trip.gyro)
        # DeviceMotion rotationRate: alpha about z, beta about x, gamma about y (deg/s)
        for col, idx in (("gx", 2), ("gy", 3), ("gz", 1)):
            imu[col] = np.radians(np.interp(imu.t, g[:, 0], g[:, idx]))
    p = np.array(trip.gps)
    gps = pd.DataFrame({"t": p[:, 0], "lat": p[:, 1], "lon": p[:, 2], "speed": p[:, 3] * 3.6})
    if (gps.speed < 0).all():                              # browser gave no speed: derive from track
        d = np.hypot(np.diff(gps.lat) * 111_320, np.diff(gps.lon) * 111_320 * np.cos(np.radians(gps.lat[:-1])))
        gps["speed"] = np.r_[0, d / np.maximum(np.diff(gps.t), 1e-3)] * 3.6
    gps["speed"] = gps.speed.clip(lower=0)
    start = pd.to_datetime(trip.start_ms, unit="ms", utc=True)
    return _report(gps, imu, start, title="Live recording")


SHELL = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
         '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>{}</body></html>')


@app.get("/", response_class=HTMLResponse)
def index():
    """index.html is a page fragment (so it can also be published as a hosted page); wrap it here."""
    return SHELL.format((STATIC / "index.html").read_text(encoding="utf-8"))


app.mount("/", StaticFiles(directory=STATIC), name="static")
