# Model plan: driving-risk engine

Goal: from a phone's sensors (GPS + accelerometer + gyroscope) plus context (weather, light,
road), produce (1) detected driving events, (2) a driving-style assessment, (3) a 0–100 trip
score with explanations, (4) a risk multiplier an insurer can price with. Every claim we make
about accuracy must come from **data the model and we never tuned on**.

## 1. Architecture

```
phone raw IMU + GPS ──► [A] preprocessing ──► [B] event detector ──► events (type, severity, t, lat/lon)
   (any orientation)      resample, gravity,      harsh brake / accel /      │
                          align to car frame      turn / lane change          ▼
                                    │                              [C] style model (windows → trip)
                                    └──► GPS features: speed,          normal / aggressive / drowsy
                                         speeding vs OSM limit               │
context: Open-Meteo weather, sun position, road type, time ──► [D] context multipliers (Latvia)
                                                                             ▼
                                                     [E] trip score 0–100 + reasons ("3 harsh brakes in the dark")
                                                                             ▼
                                                     [F] insurer risk: MTPL frequency model × telematics factor
```

Why two stages (events, then style) rather than one model: event labels exist in four
independent datasets (several countries, cars and sensors), so the event detector can be
trained on some and **blind-tested on others**. Style labels exist only in UAH (6 drivers).
Event rates are also what both the driver and the insurer understand.

## 2. Data inventory and roles

| Dataset | Country / sensor | Labels | Role |
|---|---|---|---|
| UAH-DriveSet | Spain, phone, IMU+GPS+OSM, 6 drivers, 40 trips | trip style N/A/D; algorithmic events | Style model: **dev = 5 drivers, blind = 1 driver (D3)** |
| Ferreira `jair-jr/driverBehaviorDataset` | Brazil, phone 50–200 Hz, Earth-frame | 69 events, video-labelled ground truth | **Blind event test (sealed)** |
| BYD Atto BMI160 | Sri Lanka, IMU board 20 Hz | harsh braking, lane change (push-button) | Event training |
| saboorahmad47 harsh-driving | phone, paved roads | sudden acc/brake/turn/lane change vs safe, at 20/40/60 km/h | Event training |
| shashwatwork driving-behavior | phone acc+gyro | 4 event classes | Event training |
| PVS | Brazil, IMU+GPS, 9 drives | road surface | Robustness: potholes must not count as harsh events |
| outofskills | phone acc+gyro | 1 recording per class | Sanity only (class = recording, so it would leak) |
| CAIS Latvia + Open-Meteo | Latvia | 209k accidents | Context multipliers (done, `src/context_risk.py`) |
| freMTPL2 | France MTPL | claims | Insurer frequency model |
| **Own recordings** (team phone, Riga) | Latvia, our phone, our app | we label | **Final blind test, in our real deployment setting** |

## 3. Blind-test protocol (the most important part)

1. **Lock splits before any more modelling.** `data_splits.json` (repo root, tracked in git) is the single source of truth.
   All dev loaders go through `src/splits.py`, which refuses to return blind items.
2. Blind sets: UAH driver **D3** (the only driver with all 6 style × road combinations),
   the **whole Ferreira dataset**, and our own recordings once collected.
   Note: D3 appeared once as a LODO fold in the first baseline (no tuning was done on it). That is
   recorded here for honesty.
3. Dev evaluation = leave-one-driver-out on the 4–5 dev drivers, plus leave-one-dataset-out for events.
   All model, feature and threshold choices are made on dev only.
4. `src/blind_eval.py` is the only code that reads blind data. It needs `--confirm`, evaluates a
   **frozen** model artefact, and appends a row (time, git commit, model hash, metrics) to
   `docs/BLIND_LOG.md`. We run it **once per milestone**, not as a tuning loop. Every peek is visible in the log.
5. Report both dev-CV and blind numbers. If blind is much worse, that is a finding, not something to hide.

### Pre-registered before the first blind run (2026-10-28)
- Style: frozen `models/style_gbm` (GBM, 60 s windows, trained on D1, D2, D4, D5, D6). Primary metric: trip accuracy on D3.
- Events: **primary = physics rules on 1.5 s low-passed car-frame signals** (dev cross-dataset evidence: the GBM did
  not transfer between sensors). Secondary = `models/event_gbm` (abs+rel features, 0.5 s, trained on BYD + saboor).
  Primary metric: macro event-F1 on Ferreira; also non-aggressive events wrongly flagged.

### Pre-registered before the v2 blind run (2026-10-28)
- Event blind set is now **PVS 7–9** (whole engine, normal driving). Primary metric: GPS-confirmed harsh events
  (brake + accel + turn) per hour. **Pass = ≤ 2 per hour and every trip scores ≥ 80.** Secondary: raw IMU events/hour.
- Event model v2 = alignment-free features + noise/gain augmentation + PVS 1–6 negatives, chosen on
  leave-one-dataset-out dev results only (scripts/exp_events_lodo.py, src/events.py).
- Own recordings: same engine; evaluated when collected (docs/RECORDING.md).

### Pre-registered before the Grab blind run (2026-10-28)
- Trip risk models, chosen on 5-fold dev CV (monotone GBM unless it costs > 0.005 AUC vs plain GBM):
  **primary `models/trip_risk` = behaviour-only** (no trip duration/distance: exposure is priced separately, the
  driving score must not punish long trips); reference `models/trip_risk_full` (with exposure).
  Blind set: sealed 20% of Grab trips (4,027). **Primary: ROC-AUC. Pass = blind AUC ≥ lower bound of dev-CV 95% CI.**
- Self-supervised Transformer: compared on dev CV only; enters the blind run only if it improves the GBM ensemble on dev.

### Pre-registered before any Transformer blind run (2026-10-28)
- Self-supervised patch Transformer (`models/trip_ssl`) enters the blind run only if, on dev out-of-fold predictions,
  pre-training beats scratch and the GBM-full + Transformer ensemble beats GBM-full with a paired-bootstrap CI above 0.
  The ensemble weight is fixed on dev. It is compared with the *full* GBM, because zero padding reveals trip length.
- **Blind primary: paired AUC gain of the ensemble over GBM-full on the sealed 4,027 Grab trips. Success only if the
  95% CI of the gain is entirely above zero.**

## 4. Metrics

- Events: per-event precision/recall/F1 per type (an event counts as detected if the prediction overlaps the
  ground-truth interval); false events per hour on normal driving and on rough roads (PVS).
- Style: window macro-F1, trip accuracy, trip-level ROC-AUC for risky vs normal, calibration (Brier, ECE).
- Score: monotonic sanity (aggressive trips < normal trips for every held-out driver), stability
  (same trip resampled / rotated gives the same score within ±3 points).
- Insurer: Poisson deviance / Gini on freMTPL2 holdout.

## 5. Workstreams and order

**Phase 0: lockdown (now).** data_splits.json, splits.py, blind_eval gate + log; fetch event datasets.

**Phase 1: preprocessing [A].**
- Common trip format (`t, ax, ay, az, gx, gy, gz, [speed, course, lat, lon, speed_limit]`), resampled to 20 Hz.
- Gravity estimate (low-pass) → vertical axis; horizontal axes aligned to the car's forward direction by
  (a) correlating with GPS acceleration when GPS is present, or (b) PCA of horizontal acceleration while driving straight.
- Unit tests: random 3-D rotations of a trip must give the same features (orientation invariance).

**Phase 2: event detector [B].**
- Baseline: physics thresholds on aligned longitudinal/lateral accel + yaw rate (interpretable; needs no training).
- ML: windowed GBM on aligned-frame features, trained on BYD + saboor + shashwat (+ UAH algorithmic events as weak labels).
- Dev: leave-one-dataset-out. Blind: Ferreira.
- Robustness: false positives on PVS rough-road segments.

**Phase 3: style model [C].**
- GBM on window features + event rates (current baseline, refit on dev drivers only).
- Sequence model on Kaggle GPU: 1D-CNN / GRU on 10 Hz channels with orientation augmentation; compare in LODO.
- Trip aggregation: mean / attention pooling of window probabilities; calibrate (isotonic, nested in LODO).
- Augmentations: random rotation, sensor noise, GPS dropout, sample-rate jitter, so the model survives real phones.

**Phase 4: score + explanations [E].**
- Score = 100 − weighted penalties (event rates per 100 km × context multipliers from [D], speeding share,
  style probability), weights set so the score is monotone with risk. SHAP / event list → "why" text.

**Phase 5: insurer model [F].**
- Poisson GBM on freMTPL2 (Gini, deviance). Telematics factor from published usage-based-insurance studies,
  mapped from our score. Clearly labelled as an assumption, since no public data links telematics to claims.

**Phase 6: own data collection.** Record with Sensor Logger / phyphox: normal city + highway drives in Riga.
Harsh events only in a safe, empty area (parking lot), never in traffic. Label by voice notes or a button.
This becomes the third blind set and the live demo.

**Phase 7: blind evaluation and model card.** Freeze → `blind_eval.py --confirm` → `docs/MODEL_CARD.md`
(data, metrics dev vs blind, failure modes, caveats).

## 6. Risks

- Few drivers (6) → high variance; report CIs (bootstrap over drivers/trips).
- Event labels differ by dataset ("sudden" vs "aggressive"); harmonise to {brake, accel, turn, lane_change, normal}.
- Different sample rates and sensors → resample + augment; leave-one-dataset-out tells us if it transfers.
- Drowsiness from IMU alone is weak (UAH measures it with a camera); present it as experimental.
