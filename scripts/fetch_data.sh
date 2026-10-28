#!/usr/bin/env bash
# Download all raw data into data/raw/. Usage: scripts/fetch_data.sh [--big]
# Small sets are always fetched; --big also pulls the multi-GB ones.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p "$RAW"

kg() { kaggle datasets download "$1" -p "$RAW/$2" --unzip -q; }

# --- Driving behaviour (phone IMU) ---
kg outofskills/driving-behavior                          driving_behavior_outofskills
kg shashwatwork/driving-behavior-dataset                 driving_behavior_shashwat
kg gloseto/traffic-driving-style-road-surface-condition  driving_style_obd_gloseto

# --- Insurance (MTPL claim frequency) ---
kg floser/french-motor-claims-datasets-fremtpl2freq      fremtpl2

# --- Latvia context (not on Kaggle; mirrored to our private dataset) ---
mkdir -p "$RAW/latvia"
curl -sfL -o "$RAW/latvia/cais_accidents_2013_2023.csv" \
  "https://data.gov.lv/dati/dataset/b6bbc7c8-849f-43e9-812d-af50030b5a54/resource/a26832fc-ad60-4aa6-aa68-2f1442732450/download/cais_dati.csv"
for n in 2 3 5; do
  curl -sfL -o "$RAW/latvia/lvceli_serious_accidents_SCSNG$n.csv" "https://data.lvceli.lv/maps/CSV/CSV_SIC_SCSNG$n.csv"
done

# --- Event datasets (harsh brake / accel / turn / lane change) ---
kg dinupadevinda/byd-atto-bmi160-imu-driving-events      events_byd
kg saboorahmad47/harsh-driving-dataset                   events_saboor
# BLIND event test set: never use for training or tuning (see docs/PLAN.md section 3)
if [[ ! -d "$RAW/blind_ferreira" ]]; then
  git clone -q --depth 1 https://github.com/jair-jr/driverBehaviorDataset "$RAW/blind_ferreira"
  rm -rf "$RAW/blind_ferreira/.git"
fi

if [[ "${1:-}" == "--big" ]]; then
  kg tsiaras/uk-road-safety-accidents-and-vehicles       uk_stats19
  kg ahmedlahlou/accidents-in-france-from-2005-to-2016   france_accidents
  kg jaymovaliya14/uah-driveset                          uah_driveset
fi
echo "Done -> $RAW"
