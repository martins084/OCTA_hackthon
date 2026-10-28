"""Latvian condition risk table: how much riskier is driving in rain / snow / darkness / on ice?

relative_risk = (share of accidents in condition) / (share of driving time in condition)
  - accidents: CAIS (data.gov.lv), police-registered accidents 2013-2023
  - driving time: Open-Meteo hourly weather + sun position for Latvian towns, weighted by a
    typical diurnal traffic profile (people drive less at 3am, so 3am weather counts less)
severity = P(serious or fatal injury | accident, condition): needs no exposure data at all.

Usage: python src/context_risk.py   ->  data/processed/lv_condition_risk.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CAIS = ROOT / "data/raw/latvia/cais_accidents_2013_2023.csv"
CACHE = ROOT / "data/raw/latvia/openmeteo_hourly.parquet"
OUT = ROOT / "data/processed/lv_condition_risk.csv"

# (name, lat, lon, weight ~ population share of the sample)
TOWNS = [
    ("Riga", 56.95, 24.11, 0.40), ("Daugavpils", 55.87, 26.54, 0.10), ("Liepaja", 56.51, 21.01, 0.09),
    ("Jelgava", 56.65, 23.72, 0.08), ("Valmiera", 57.54, 25.43, 0.08), ("Rezekne", 56.51, 27.33, 0.08),
    ("Ventspils", 57.39, 21.56, 0.07), ("Madona", 56.85, 26.22, 0.05), ("Kuldiga", 56.97, 21.97, 0.05),
]
# Assumption: share of daily vehicle-km by local hour (generic European urban/interurban profile).
TRAFFIC_BY_HOUR = np.array([0.6, 0.4, 0.3, 0.3, 0.5, 1.2, 3.0, 6.0, 7.0, 5.5, 5.0, 5.2,
                            5.5, 5.6, 5.9, 6.6, 7.4, 7.6, 6.2, 4.6, 3.5, 2.8, 2.0, 1.3])
TRAFFIC_BY_HOUR /= TRAFFIC_BY_HOUR.sum()

WEATHER_MAP = {"SKAIDRS": "clear", "SAULAINS": "clear", "APMĀCIES": "overcast", "LIETUS": "rain",
               "SNIEGS": "snow", "MIGLA": "fog", "APLEDOJUMS": "freezing_rain"}
SURFACE_MAP = {"SAUSA": "dry", "SLAPJA": "wet", "NOBLIETĒTS SNIEGS": "snow_ice", "SLAPJŠ SNIEGS": "snow_ice",
               "APLEDOJUSI": "snow_ice", "NOKAIS.AR PRETSLĪD.MAT.": "snow_ice"}
LIGHT_MAP = {"DIENA": "day", "KRĒSLA": "dusk", "TUMSA": "dark_unlit", "TUMSA, APGAISMOJUMA NAV": "dark_unlit",
             "TUMSA, APGAISM.NAV IESLĒGTS": "dark_unlit", "APGAISMOJUMA NAV": "dark_unlit",
             "APGAISM.NAV IESLĒGTS": "dark_unlit", "TUMSA, APGAISM.IESLĒGTS": "dark_lit",
             "APGAISMOJUMS IESLĒGTS": "dark_lit"}
SEVERE = {"SMAGI IEVAINOTS", "GĀJUŠI BOJĀ"}


def load_accidents() -> pd.DataFrame:
    d = pd.read_csv(CAIS, sep=";", encoding="utf-8-sig", low_memory=False).dropna(how="all")
    d["severe"] = d["Dalībnieka miesas bojājumi"].isin(SEVERE)
    d["fatal"] = d["Dalībnieka miesas bojājumi"].eq("GĀJUŠI BOJĀ")
    d["drunk"] = d["Reibums"].eq("REIBUMS IR KONSTATĒTS")
    # One row per participant -> one row per accident.
    a = d.groupby(["Notikuma kārtas nr", "Datums"]).agg(
        weather=("Laika apstākļi", "first"), surface=("Brauktuves stāvoklis", "first"),
        light=("Apgaismojums", "first"), severe=("severe", "any"), fatal=("fatal", "any"),
        drunk=("drunk", "any")).reset_index()
    a["weather"] = a.weather.map(WEATHER_MAP)
    a["surface"] = a.surface.map(SURFACE_MAP)
    a["light"] = a.light.map(LIGHT_MAP)
    a["date"] = pd.to_datetime(a["Datums"], format="%d.%m.%Y")
    return a


def fetch_weather() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    frames = []
    for name, lat, lon, w in TOWNS:
        part = CACHE.with_name(f"openmeteo_{name}.parquet")
        if not part.exists():
            for attempt in range(6):  # free tier rate-limits bursts; back off and retry
                r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=180, params=dict(
                    latitude=lat, longitude=lon, start_date="2013-01-01", end_date="2023-12-31",
                    hourly="weather_code,precipitation,snowfall,snow_depth,temperature_2m",
                    timezone="Europe/Riga"))
                if r.status_code != 429:
                    break
                time.sleep(30 * (attempt + 1))
            r.raise_for_status()
            pd.DataFrame(r.json()["hourly"]).to_parquet(part)
            print("fetched", name)
        frames.append(pd.read_parquet(part).assign(town=name, lat=lat, lon=lon, town_w=w))
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def sun_elevation(t_utc: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Solar elevation in degrees (NOAA simplified formula, good to ~0.5 deg)."""
    doy = t_utc.day_of_year.values
    hour = t_utc.hour.values + t_utc.minute.values / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    ha = np.radians((hour * 60 + eqt + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    return np.degrees(np.arcsin(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)))


def weather_exposure(w: pd.DataFrame) -> pd.DataFrame:
    """Label every town-hour with weather / surface / light classes and a driving weight."""
    w = w.sort_values(["town", "time"]).copy()
    c = w.weather_code
    w["weather"] = np.select(
        [c.isin([56, 57, 66, 67]), c.isin([71, 73, 75, 77, 85, 86]), c.isin([45, 48]),
         c.between(51, 65) | c.between(80, 82) | c.ge(95), c.isin([2, 3])],
        ["freezing_rain", "snow", "fog", "rain", "overcast"], "clear")
    # Surface proxy: snow on the ground or sub-zero after precipitation -> snow_ice; recent rain -> wet.
    recent_p = w.groupby("town").precipitation.transform(lambda s: s.rolling(3, min_periods=1).sum())
    w["surface"] = np.select(
        [(w.snow_depth >= 0.01) & (w.temperature_2m <= 1) | (w.temperature_2m <= 0) & (recent_p >= 0.1),
         recent_p >= 0.1], ["snow_ice", "wet"], "dry")
    # Light: sample 4x per hour so dusk (~40-70 min) is resolved.
    elev = np.zeros(len(w))
    for name, lat, lon, _ in TOWNS:
        m = (w.town == name).values
        t = pd.DatetimeIndex(w.time[m]).tz_localize("Europe/Riga", ambiguous="NaT", nonexistent="NaT")
        t = t.tz_convert("UTC").tz_localize(None)
        elev[m] = np.mean([sun_elevation(t + pd.Timedelta(minutes=k), lat, lon) for k in (7, 22, 37, 52)], axis=0)
    w["light"] = np.select([elev > -0.83, elev > -6], ["day", "dusk"], "dark")
    w["weight"] = w.town_w * TRAFFIC_BY_HOUR[w.time.dt.hour.values]
    return w


def risk_table(acc: pd.DataFrame, exp: pd.DataFrame) -> pd.DataFrame:
    """Severity for every factor; exposure-based relative risk only for light.

    Reanalysis weather codes don't match how police label weather (drizzle vs 'clear'), so
    weather/surface frequency effects come from daily_weather_model() instead.
    """
    rows = []
    for factor in ["weather", "surface", "light"]:
        a = acc.dropna(subset=[factor])
        if factor == "light":  # exposure can't tell lit from unlit roads -> compare 'dark' overall
            a = a.assign(light=a.light.replace({"dark_lit": "dark", "dark_unlit": "dark"}))
        acc_share = a[factor].value_counts(normalize=True)
        sev = a.groupby(factor).agg(n=("severe", "size"), severe_rate=("severe", "mean"),
                                    fatal_rate=("fatal", "mean"), drunk_rate=("drunk", "mean"))
        t = sev.join(acc_share.rename("acc_share"))
        if factor == "light":
            t["exposure_share"] = exp.groupby("light").weight.sum() / exp.weight.sum()
            t["relative_risk"] = t.acc_share / t.exposure_share
        t["severity_vs_base"] = t.severe_rate / a.severe.mean()
        rows.append(t.reset_index(names="level").assign(factor=factor))
    # Lit vs unlit darkness: severity only (no exposure data for street lighting).
    d = acc[acc.light.isin(["dark_lit", "dark_unlit"])]
    s = d.groupby("light").agg(n=("severe", "size"), severe_rate=("severe", "mean"),
                               fatal_rate=("fatal", "mean"), drunk_rate=("drunk", "mean"))
    s["severity_vs_base"] = s.severe_rate / acc.severe.mean()
    rows.append(s.reset_index(names="level").assign(factor="darkness_lighting"))
    cols = ["factor", "level", "n", "acc_share", "exposure_share", "relative_risk",
            "severe_rate", "fatal_rate", "severity_vs_base", "drunk_rate"]
    return pd.concat(rows).reindex(columns=cols)


def daily_weather_model(acc: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    """Quasi-Poisson regression of national accidents/day on that day's weather.

    Year-month fixed effects + weekday + holidays absorb traffic volume, so each wet/snowy day is
    compared with ordinary days of the same month: exp(coef) = accident-rate multiplier.
    """
    import holidays
    import statsmodels.formula.api as smf

    w = w.assign(day=w.time.dt.floor("D"))
    town = w.groupby(["town", "day"]).agg(
        precip=("precipitation", "sum"), snowfall=("snowfall", "sum"), tmin=("temperature_2m", "min"),
        tmax=("temperature_2m", "max"), snow_depth=("snow_depth", "mean"), town_w=("town_w", "first"))
    town["rain"] = town.precip - town.snowfall / 0.7  # Open-Meteo: 1 cm snow ~ 0.7 mm water
    town["prev_wet"] = town.groupby("town").precip.shift(1) >= 0.5
    cols = ["rain", "snowfall", "tmin", "tmax", "snow_depth"]
    day = town[cols].mul(town.town_w, axis=0).groupby("day").sum().div(town.groupby("day").town_w.sum(), axis=0)
    day["prev_wet"] = (town.prev_wet * town.town_w).groupby("day").sum() > 0.5

    d = acc.groupby("date").size().rename("n").to_frame().join(day, how="inner")
    lv_holidays = holidays.Latvia(years=range(2013, 2024))
    d["holiday"] = [x in lv_holidays for x in d.index]
    d["weekday"] = d.index.dayofweek.astype(str)
    d["ym"] = d.index.strftime("%Y-%m")
    d["rain_level"] = pd.cut(d.rain.clip(lower=0), [-1, 0.5, 5, 1e9], labels=["dry", "light", "heavy"])
    d["snow_level"] = pd.cut(d.snowfall, [-1, 0.5, 5, 1e9], labels=["none", "light", "heavy"])
    d["frost_after_wet"] = (d.tmin <= 0) & (d.tmax > -5) & d.prev_wet  # refreeze / black-ice days
    d["snow_cover"] = d.snow_depth >= 0.02
    # First snowfall of each winter (season = Sept..Aug) and the 6 days after: drivers not yet adapted.
    season = d.index.year - (d.index.month < 9)
    first = d[d.snowfall >= 1].groupby(season[d.snowfall >= 1]).apply(lambda g: g.index.min())
    d["first_snow_week"] = [(s in first.index) and 0 <= (t - first[s]).days < 7 for t, s in zip(d.index, season)]

    m = smf.glm("n ~ C(rain_level) + C(snow_level) + frost_after_wet + snow_cover + first_snow_week"
                " + holiday + C(weekday) + C(ym)", data=d,
                family=__import__("statsmodels.api").api.families.Poisson()).fit(scale="X2")
    keep = [k for k in m.params.index if not k.startswith(("C(ym)", "C(weekday)", "Intercept"))]
    ci = m.conf_int().loc[keep]
    out = pd.DataFrame({"multiplier": np.exp(m.params[keep]), "ci_low": np.exp(ci[0]),
                        "ci_high": np.exp(ci[1]), "p": m.pvalues[keep],
                        "days": [int(d.eval(_term_mask(k)).sum()) for k in keep]})
    return out


def _term_mask(term: str) -> str:
    """'C(rain_level)[T.light]' -> "rain_level == 'light'", 'holiday[T.True]' -> 'holiday == True'."""
    name, level = term.replace("C(", "").replace(")", "").split("[T.")
    level = level.rstrip("]")
    return f"{name} == {level}" if level in ("True", "False") else f"{name} == '{level}'"


if __name__ == "__main__":
    acc = load_accidents()
    print(f"{len(acc):,} accidents {acc.date.min():%Y-%m-%d}..{acc.date.max():%Y-%m-%d}, "
          f"severe {acc.severe.mean():.1%}")
    weather = fetch_weather()
    table = risk_table(acc, weather_exposure(weather))
    daily = daily_weather_model(acc, weather)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    daily.to_csv(OUT.with_name("lv_daily_weather_multipliers.csv"))
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print("\nAccident-rate multipliers vs an ordinary day of the same month (95% CI):")
    print(daily.round(3).to_string())
