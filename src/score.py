"""Trip score 0-100 (higher = safer) with reasons, plus the insurer's risk multiplier.

score = 100 - penalties, where penalties come from
  - harsh events per 100 km, each weighted by its context (dark, snow, ice...: Latvian multipliers
    from src/context_risk.py),
  - speeding (share of time >10% over the limit, and >20 km/h over),
  - the style model's probability that the trip is risky (aggressive/drowsy).
Weights are expert-set assumptions (no public data links telematics to claims); they are checked for
the property that matters: risky trips must score lower than normal trips of the same driver.

Usage: python src/score.py   (dev-driver evaluation on out-of-fold style probabilities)
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"

EVENT_POINTS = {"harsh_brake": 4.0, "harsh_accel": 2.0, "harsh_turn": 3.0}   # per event per 100 km
SPEEDING_POINTS = {"frac_over_limit_10": 25.0, "frac_over_limit_20": 40.0}  # x share of driving time
STYLE_POINTS = 30.0                                                          # x P(risky)
MAX_EVENT_PENALTY = 40.0


@dataclass
class Context:
    """Trip context. Multipliers come from the Latvian accident data (docs/RESULTS.md)."""
    dark_share: float = 0.0          # share of trip in darkness
    snow_day: str = "none"           # none / light / heavy
    frost_after_wet: bool = False
    surface: str = "dry"             # dry / wet / snow_ice

    def multiplier(self, risk: pd.DataFrame | None = None, daily: pd.DataFrame | None = None) -> float:
        """Combined accident-rate multiplier for this trip's conditions vs an average trip."""
        dark_rr, snow = 1.33, {"none": 1.0, "light": 1.14, "heavy": 1.64}
        frost = 1.03
        if risk is not None:
            dark_rr = float(risk.query("factor == 'light' and level == 'dark'").relative_risk.iloc[0])
        if daily is not None:
            snow = {"none": 1.0, "light": float(daily.loc["C(snow_level)[T.light]", "multiplier"]),
                    "heavy": float(daily.loc["C(snow_level)[T.heavy]", "multiplier"])}
            frost = float(daily.loc["frost_after_wet[T.True]", "multiplier"])
        m = (1 - self.dark_share) + self.dark_share * dark_rr
        m *= snow[self.snow_day]
        m *= frost if self.frost_after_wet else 1.0
        return m


@dataclass
class TripScore:
    score: float
    penalties: dict
    reasons: list = field(default_factory=list)
    context_multiplier: float = 1.0

    @property
    def insurer_multiplier(self) -> float:
        """Assumed telematics factor on the MTPL baseline frequency: 0.7 (score 100) .. ~1.75 (score 0).
        Log-linear, = 1.0 at score 75 (a typical driver). To be recalibrated with real claims."""
        return float(np.exp(-0.0123 * (self.score - 75) - 0.0) if self.score >= 75 else
                     np.exp(-0.0075 * (self.score - 75)))


def load_context_tables():
    risk = pd.read_csv(PROC / "lv_condition_risk.csv") if (PROC / "lv_condition_risk.csv").exists() else None
    daily = (pd.read_csv(PROC / "lv_daily_weather_multipliers.csv", index_col=0)
             if (PROC / "lv_daily_weather_multipliers.csv").exists() else None)
    return risk, daily


def score_trip(windows: pd.DataFrame, p_risky: float, ctx: Context | None = None, tables=None) -> TripScore:
    """windows: output of driving_features.window_features for one trip."""
    ctx = ctx or Context()
    cm = ctx.multiplier(*(tables or (None, None)))
    km = max(windows.km.sum(), 0.1)
    pen, reasons = {}, []
    # windows overlap (60 s / 15 s step): each event is counted in ~4 windows
    overlap = 60 / 15
    ev_total = 0.0
    for col, pts in EVENT_POINTS.items():
        n = windows[col].sum() / overlap
        per100 = n / km * 100
        p = pts * per100 * cm
        ev_total += p
        if n >= 1:
            reasons.append(f"{n:.0f} {col.replace('_', ' ')} event(s) ({per100:.1f} per 100 km)")
    pen["events"] = min(ev_total, MAX_EVENT_PENALTY)
    for col, pts in SPEEDING_POINTS.items():
        share = float(windows[col].mean()) if col in windows else 0.0
        pen[col] = pts * share
        if share > 0.05:
            reasons.append(f"{share:.0%} of the time {'>10% over' if col.endswith('10') else '>20 km/h over'} the limit")
    pen["style"] = STYLE_POINTS * p_risky
    if p_risky > 0.5:
        reasons.append(f"driving pattern looks risky (p={p_risky:.2f})")
    if cm > 1.05:
        reasons.append(f"conditions raised risk x{cm:.2f} (darkness/snow/ice): events weighted more")
    score = float(np.clip(100 - sum(pen.values()), 0, 100))
    return TripScore(score=round(score, 1), penalties={k: round(v, 2) for k, v in pen.items()},
                     reasons=reasons, context_multiplier=round(cm, 3))


def main():
    """Dev check: out-of-fold style probabilities (train_driving.py LODO) + window features."""
    from sklearn.metrics import roc_auc_score
    import train_driving as T

    df = T.build_dataset(T.uah_trip_dirs(T.ROOT / "data/raw/uah", "dev"))
    y = df.label.map({c: i for i, c in enumerate(T.CLASSES)})
    trips = T.trip_table(df, T.lodo(df, y, 3))
    rows = []
    for trip, w in df.groupby("trip"):
        s = score_trip(w, trips.loc[trip, "p_risky"])
        rows.append(dict(trip=trip, driver=w.driver.iloc[0], label=w.label.iloc[0], road=w.road.iloc[0],
                         score=s.score, insurer_x=round(s.insurer_multiplier, 2), **s.penalties))
    r = pd.DataFrame(rows).sort_values(["driver", "score"])
    print(r.drop(columns="trip").to_string(index=False))
    auc = roc_auc_score(r.label == "NORMAL", r.score)
    # Per driver: every normal trip should outscore every risky trip of the same driver.
    pairs = ok = 0
    for _, g in r.groupby("driver"):
        for a in g[g.label == "NORMAL"].score:
            for b in g[g.label != "NORMAL"].score:
                pairs += 1
                ok += a > b
    print(f"\nAUC(score separates normal from risky trips) = {auc:.3f}")
    print(f"within-driver ordering: normal > risky in {ok}/{pairs} pairs ({ok / pairs:.1%})")
    print(r.groupby("label").score.describe()[["mean", "min", "max"]].round(1))
    (PROC / "score_dev.json").write_text(json.dumps(dict(auc=auc, within_driver_pairs=ok / pairs), indent=2))


if __name__ == "__main__":
    main()
