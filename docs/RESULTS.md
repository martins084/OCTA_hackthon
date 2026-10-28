# Results

Reproduce: `pip install -r requirements.txt`, `scripts/fetch_data.sh`, then the `src/` scripts named below.
Plan and blind protocol: `docs/PLAN.md`. Every blind run: `docs/BLIND_LOG.md`.

## Headline: dev vs blind (milestone v1, 2026-10-28)

| Component | Dev estimate | **Blind result** | Verdict |
|---|---|---|---|
| Style model, trip accuracy (normal/aggressive/drowsy) | 79% (26/33), CI 64–91% | **86% (6/7), CI 57–100%** on unseen driver D3 | holds up |
| Style model, risky-vs-normal trip AUC | 0.92 | **1.00** | holds up |
| Event detector, primary (physics rules) | F1 0.57 (dev) | **macro F1 0.17** on Ferreira (Brazil); brakes/accels 0% recall | **fails** |
| Event detector, secondary (GBM) | F1 0.77 (pooled CV), 0.08–0.38 cross-dataset | **macro F1 0.37**; brake 0.51, turn 0.71, accel 0.0; 6/14 gentle events wrongly flagged | weak, as cross-dataset dev predicted |

The pooled-CV number (0.77) overstated the event detector; the cross-dataset dev number (≤0.38)
predicted the blind result well. Lesson: for anything that must work on a new phone, trust cross-dataset numbers.

## Style model (`src/train_driving.py`, frozen `models/style_gbm`)

UAH-DriveSet, dev drivers D1, D2, D4, D5, D6 (33 trips), 60 s windows, leave-one-driver-out.
Phone-orientation-invariant features: GPS-derived longitudinal/lateral acceleration, accelerometer
magnitude and vibration bands, speed vs OSM limit. Errors are drowsy ↔ normal (UAH measures drowsiness
mainly via a lane camera we don't use). Tried and not adopted (no gain beyond noise): 30/120 s windows,
1D-CNN on Kaggle GPU (trip acc 79%, AUC 0.82), GBM+CNN ensemble.

## Trip score (`src/score.py`)

Penalties for harsh events per 100 km (weighted by Latvian context multipliers), speeding and style
probability. Dev drivers: aggressive mean 53, drowsy 77, normal 86; score AUC 0.90; normal > risky for
the same driver in 50/54 pairs (93%). Fixed on the way: GPS heading jitter at motorway speed created fake
"sharp turns" (3 s heading smoothing, events must last ≥ 1 s).

## Event detector (`src/events.py`), dev

Car-frame alignment (`src/preprocess.py`) verified: recovers the documented BYD mounting in all 10 sessions;
invariant to random 3-D phone rotations (corr 1.000; `tests/test_preprocess.py`).

| Setting | macro F1 |
|---|---|
| GBM, 5-fold grouped CV (BYD + saboor pooled) | 0.77 |
| GBM, trained on one dataset → tested on the other | 0.08–0.38 |
| Physics rules (0.3 g), 1.5 s low-pass | 0.57 |

## Insurer model (`src/insurer_model.py`), freMTPL2 holdout (135k policies)

| Model | Poisson deviance ↓ | Gini ↑ |
|---|---|---|
| constant | 0.627 | 0 |
| Poisson GLM (actuarial standard) | 0.593 | 0.305 |
| **Poisson GBM** | **0.572** | **0.361** |

## Latvian context risk (`src/context_risk.py`, CAIS 2013–2023, 209k accidents)

- Dark: 1.33× accidents per hour driven; 1.32× as likely to be serious. Dark + unlit: fatality 1.7% vs 0.6% by day.
- Daily model (year-month + weekday + holiday fixed effects): light snowfall 1.14× (1.12–1.17), heavy snowfall
  1.64× (1.51–1.77, 14 days), refreeze after wet day 1.03×, snow cover 1.04×, rain ~1.0×; holidays 0.67× (sanity check).
- Severity given an accident: fog 1.8×, freezing rain 1.6×; snow/ice 0.85× (slower speeds).
- Caveats: assumed hourly traffic profile; 9-town weather average; police-registered accidents only.
