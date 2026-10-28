# Data

All data lives on Kaggle. Kernels attach it via `dataset_sources` in `kernel-metadata.json`
(see `kaggle/kernel-template/`). Mount path: `/kaggle/input/datasets/<owner>/<slug>/`.
Local copy for exploration: `scripts/fetch_data.sh` (add `--big` for the multi-GB sets) -> `data/raw/` (gitignored).

Our own private dataset **`martinsvarna/octa-driving-data`** holds data that isn't on Kaggle:
`latvia/` (Latvian open data), `uah/` (UAH-DriveSet sensor .txt files only, no videos) and
`processed/` (outputs of `src/`). Update it with
`kaggle datasets version -p kaggle/octa-data -m "msg" -r zip`.

## 1. Driving behaviour: the core score model

| Source (Kaggle ref) | Content | Labels | Use |
|---|---|---|---|
| `outofskills/driving-behavior` | Phone acc+gyro, ~6.7k rows, train/test split | AGGRESSIVE / NORMAL / SLOW | First baseline classifier |
| `shashwatwork/driving-behavior-dataset` | Phone acc+gyro raw + pre-made window features | 4 event classes | Event detector (harsh brake/accel/turn) |
| `jaymovaliya14/uah-driveset` | UAH-DriveSet: 6 drivers × normal/drowsy/aggressive × motorway/secondary, phone IMU+GPS+events | Trip-level style | Validation, trip-level scoring |
| `gloseto/traffic-driving-style-road-surface-condition` | OBD signals | EvenPace/Aggressive + road surface + traffic | Shows context changes the meaning of the same signal |

## 2. Insurance: turning the score into risk and price

| Source | Content | Use |
|---|---|---|
| `floser/french-motor-claims-datasets-fremtpl2freq` | 678k French MTPL policies (the French equivalent of OCTA): claims, exposure, driver age, bonus-malus, vehicle, region | Baseline claim-frequency model (Poisson GLM / GBM). The driving score is a multiplier on top. |

## 3. Context: weather, light, road surface, time

| Source | Content | Notes |
|---|---|---|
| **Latvia CAIS** (`octa-driving-data/cais_accidents_2013_2023.csv`, from data.gov.lv, Iekšlietu ministrijas IC) | **388k participant rows / 209k police-registered accidents, 2013–2023, Latvia.** Weather (clear/cloudy/rain/snow/fog), road surface (dry/wet/packed snow/ice/slush), lighting (day/dusk/dark ± street lights), alcohol, cause, accident type, injury severity, driver age, municipality/street | Date only, **no time of day**. Condition fields are essentially complete (the file has ~300k empty trailing lines; drop them). `;`-separated, UTF-8 with BOM. |
| **Latvia LVC** (`lvceli_serious_accidents_SCSNG*.csv`) | Serious accidents on state roads 2016–2019, **with lat/lon + time** | Small, but has time of day and coordinates. |
| `tsiaras/uk-road-safety-accidents-and-vehicles` (UK STATS19) | ~2M accidents with **time**, weather, light, road surface, speed limit, road type, severity | Gives the time-of-day curve that the Latvian data lacks. |
| `ahmedlahlou/accidents-in-france-from-2005-to-2016` | Same idea, France, with time | Backup |
| **Open-Meteo** archive API (no key, free for non-commercial use) | Hourly precipitation, snowfall, temperature, visibility proxies at any lat/lon, 1940–now | (a) Exposure baseline: how often it rains or snows in Latvia. (b) In the app: attach weather to every trip using its GPS and timestamp. |
| `astral` (Python lib) | Sunrise/sunset/twilight for any point and time | Day / dusk / dark per trip, so no lookup data is needed |

## How we use context

**The exposure trap.** "Most accidents happen in clear weather" doesn't mean clear weather is
dangerous. Most driving happens in clear weather. Risk = accidents ÷ exposure. So:

1. **Relative risk by condition** = (share of accidents in condition) ÷ (share of time in that condition).
   - Weather: accident share from CAIS, time share from Open-Meteo hourly data for Latvian cities 2013–2023.
   - Darkness: CAIS lighting field vs. the share of hours that are dark, computed with astral (better still, weighted by traffic per hour).
   - Time of day / weekday: UK STATS19 (and LVC) hour-of-day counts ÷ a traffic-volume-by-hour curve.
2. **Severity given an accident**: P(serious/fatal | accident, conditions) from CAIS. This needs no
   exposure data. It's a clean signal for "how bad it is when it happens", e.g. darkness without street lights or ice.
3. **Context-aware scoring.** The trip score weights each detected event by its context:
   a harsh brake on an icy road in the dark counts more than the same brake on a dry road in daylight. Driving
   *gently* in bad conditions earns credit ("adapted to conditions"), which works well as feedback to the driver.
4. **Insurer view**: claim frequency = freMTPL2 baseline × driving-score factor × exposure-mix factor
   (how much of the driver's km are at night, in rain or snow, or in risky hours).
