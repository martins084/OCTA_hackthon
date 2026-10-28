# Recording our own drives (blind test set #3 + demo data)

Our own recordings are the most honest test we have: our phones, Latvian roads, Latvian weather.
They are **blind**: they go to `data/raw/blind_own/` and only `src/blind_eval.py` reads them.
Nobody looks at model outputs on them before the milestone evaluation.

## Safety first
- The driver only drives. A passenger handles the phone and the labels.
- **Harsh manoeuvres only in an empty, closed area** (empty parking lot, with permission), at low speed.
  Never in traffic. On public roads, drive normally and legally; "normal driving" is the most valuable data anyway.

## App and settings
- **Sensor Logger** (Android/iOS, free) or **phyphox**.
- Enable: Accelerometer (uncalibrated is fine), Gyroscope, Gravity, Location (GPS). 50 Hz or "fastest" for motion sensors.
- Phone in any fixed position (dash mount, cup holder, door pocket). Note where it is; don't touch it while recording.
  One recording with the phone loose in a pocket is useful too, as a robustness test.
- Export as CSV (zip), one recording per drive.

## Labelling
- The passenger uses the app's **annotation/marker button** (or says it on a voice memo with the time) at the
  start of each event: `brake`, `accel`, `turn`, `lane_change`, or `normal_event` (a gentle version of the same manoeuvre).
- Per drive, write one line in `data/raw/blind_own/drives.csv`:
  `file,date,start_time,driver_id,phone_model,phone_position,road_type,weather,daylight,style,notes`
  with `style` = `normal` | `calm` | `sporty` (sporty = brisker but legal), `daylight` = day/dusk/dark.

## What to collect (priority order)
1. 3–5 normal drives, 10–20 min each, city + highway, ideally at least one in the dark and one in rain.
2. Parking-lot session: 5× each of brake / accel / turn / lane change, plus 5 gentle versions of each.
3. The same route driven by 2 different team members.

## After recording
Copy the export folders to `data/raw/blind_own/<date>_<driver>_<n>/` and add the `drives.csv` line.
Don't open them in any notebook. When we reach the milestone, `src/blind_eval.py` runs on them once.
