# Model card: OCTA driving-risk engine (prototype)

**What it does.** From a phone's GPS + accelerometer + gyroscope recording of one car trip, plus automatically
fetched context (weather, daylight, OSM speed limits), it returns: detected driving events, a driving-style
assessment, a 0–100 trip score with plain-language reasons, and a telematics multiplier on an MTPL
(OCTA-equivalent) claim-frequency model. Entry point: `src/engine.py: analyze_trip()`.

**Intended use.** Feedback for drivers and a pricing signal for an insurer in a pilot. Not for automated
decisions about individuals without human review, and not validated for claims handling.

## Components

| Component | Model | Training data | How it was evaluated |
|---|---|---|---|
| Car-frame alignment | physics (gravity + yaw/centripetal correlation, chunked) | none | BYD documented mounting; rotation-invariance tests |
| Driving style | LightGBM, 60 s windows of GPS/IMU features | UAH-DriveSet D1, D2, D4, D5, D6 (Spain) | leave-one-driver-out; **blind driver D3** |
| IMU events | LightGBM, alignment-free features, noise/gain augmentation | BYD (Sri Lanka), saboor, Ferreira (Brazil), PVS 1–6 (Brazil) | leave-one-*dataset*-out; **blind PVS 7–9** (whole engine) |
| GPS confirmation | rule: harsh speed change ≥ 8 km/h; turn ≥ 30° heading | none | PVS dev / blind false events per hour |
| Context multipliers | exposure-based relative risk + daily quasi-Poisson | CAIS Latvia 2013–2023, Open-Meteo | CIs; holiday sanity check |
| Trip score | expert-weighted penalties | none (weights are assumptions) | within-driver ordering on dev drivers |
| Insurer frequency | Poisson LightGBM | freMTPL2 (France) | 20% policy holdout: deviance, Gini |

## Results

See `docs/RESULTS.md` (all numbers) and `docs/BLIND_LOG.md` (every blind run).

## Known limitations

- **Small style data:** 6 drivers. Blind check is on 7 trips of one driver; treat it as "consistent", not as a precise accuracy.
- **Drowsiness** detection is weak without a camera (lane position); experimental.
- **Lane changes** are not reliably detected on unseen sensors; shown to the user, never penalised.
- **Telematics → claims link is assumed.** No public dataset links driving behaviour to claims; the score → multiplier
  mapping must be recalibrated with an insurer's own data in a pilot.
- **Context data:** hourly traffic profile is assumed; weather is per-trip from reanalysis/forecast, not on-road.
- **Not yet validated on Latvian phone recordings** (own recordings pending, `docs/RECORDING.md`).

## Fairness and privacy notes

- Inputs are driving signals and context; no protected attributes are used by the engine.
  The MTPL baseline uses driver age (as current pricing does); a pilot should review that separately.
- GPS traces are sensitive. A production version should process them on the phone or pseudonymise them,
  and keep only aggregates (scores, event counts) on the server.
