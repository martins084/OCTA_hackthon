# OCTA Drive Score

Hackathon prototype for Latvian OCTA (motor third-party liability) insurance. A phone's GPS,
accelerometer and gyroscope recording of a car trip goes in. Out come detected driving events, a
0–100 driving score with reasons, conditions-aware risk from Latvian accident data, and a telematics
factor on an MTPL claim-frequency model. It serves both the driver (feedback) and the insurer (pricing).

## Run the demo

```sh
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows path; use .venv/bin on macOS/Linux
scripts/fetch_data.sh                                                     # data from Kaggle, data.gov.lv, GitHub
.venv/Scripts/python app/build_demo.py                                    # score demo trips -> app/static/data
.venv/Scripts/python -m uvicorn app.server:app --port 8000                # open http://localhost:8000
```

- **Trip replay:** real recorded trips play back with a live score, events on the route, and a
  "same driving, different conditions" switch (as driven / at night / snowy night).
- **OCTA portfolio:** every trip and driver with tier, telematics factor and indicative premium.
- **Evidence:** dev vs blind results, including what failed.
- **Analyse a trip** (local server only): upload a Sensor Logger export, or record in the browser
  (phone browsers need HTTPS for motion sensors, e.g. through a tunnel).

`app/build_artifact.py` bundles a self-contained single page (data embedded) for sharing.

## Engine

`src/engine.py: analyze_trip(gps, imu, start)` runs the whole chain:

1. Phone-orientation alignment (`src/preprocess.py`).
2. GPS window features with OSM speed limits (`src/driving_features.py`).
3. Driving-style model (`src/train_driving.py`).
4. IMU event model with GPS confirmation (`src/events.py`).
5. Weather and daylight context (Open-Meteo, sun position) with Latvian multipliers (`src/context_risk.py`).
6. Score and reasons (`src/score.py`) and MTPL premium (`src/insurer_model.py`).

CLI: `python src/engine.py <sensor_logger_folder>`.

## How we evaluate

Blind test sets are locked in `data_splits.json` and guarded by `src/splits.py`. Only
`src/blind_eval.py` reads them, and every run is appended to `docs/BLIND_LOG.md`.

| Doc | Contents |
|---|---|
| `docs/PLAN.md` | plan and pre-registered criteria |
| `docs/RESULTS.md` | all results |
| `docs/MODEL_CARD.md` | model card |
| `docs/DATA.md` | data sources |
| `docs/RECORDING.md` | how to record our own Latvian drives |

Tests: `python -m pytest tests`.
