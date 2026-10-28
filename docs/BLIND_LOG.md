# Blind evaluation log

Every read of blind data, appended by src/blind_eval.py.

| time | target | model | note | metrics |
|---|---|---|---|---|
| 2026-10-28 12:57 | style | style_gbm 59d0dda52e19 (commit b17ed09, dirty) | v1 milestone: first blind run, frozen dev-only GBM | {"window_macro_f1": 0.693, "trip_accuracy": 0.857, "trip_auc_risky": 1.0, "n_trips": 7, "n_windows": 337, "trip_accuracy_ci95": [0.5714285714285714, 1.0], "dev_lodo_trip_accuracy": 0.788} |
| 2026-10-28 12:57 | events | event_gbm 502a638bcadf + physics rules | v1 milestone: first blind run, pre-registered rules primary / GBM secondary | {"rules (primary)": {"macro_f1": 0.16897081413210446, "f1_brake": 0.0, "f1_accel": 0.0, "f1_turn": 0.5806451612903226, "f1_lane_change": 0.09523809523809525, "non_aggressive_flagged": "0/14"}, "gbm (secondary)": {"macro_f1": 0.37034721905370266, "f1_brake": 0.5142857142857143, "f1_accel": 0.0, "f1_turn": 0.7096774193548387, "f1_lane_change": 0.25742574257425743, "non_aggressive_flagged": "6/14"}} |

**2026-10-28, split change (not a run):** Ferreira moved from blind to **dev** after the v1 run above, to diagnose the
event-detector failure with a third sensor. All later Ferreira numbers are dev numbers. New blind sets for events:
**PVS 7–9** (vehicle/driver 3, normal driving incl. speed bumps → false events per hour) and **own recordings**.
| 2026-10-28 13:55 | pvs | engine v2: event_gbm a262715b4a19 + style_gbm + GPS confirmation | v2 milestone: whole engine on unseen vehicle/driver 3, pre-registered pass <=2/h and scores >=80 | {"confirmed_harsh_per_hour": 4.21, "min_score": 69.8, "scores": [95.6, 69.8, 79.3], "raw_imu_per_hour": 19.1, "passed": false} |
