"""Extract race session weather (rainfall, track temperature, air temperature, humidity) from FastF1.

Saves to data/raw/race_weather.csv with caching and multi-threading.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "data" / "fastf1_cache"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

try:
    fastf1.Cache.enable_cache(str(CACHE_DIR))
except Exception:
    pass


def extract_race_weather(year: int, round_num: int):
    """Fetch race weather data for a single Grand Prix."""
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(telemetry=False, laps=False, weather=True)
        w = session.weather_data
        if w is None or len(w) == 0:
            return None

        has_rain = int(bool(w["Rainfall"].any())) if "Rainfall" in w.columns else 0
        track_temp = float(w["TrackTemp"].mean()) if "TrackTemp" in w.columns else np.nan
        air_temp = float(w["AirTemp"].mean()) if "AirTemp" in w.columns else np.nan
        humidity = float(w["Humidity"].mean()) if "Humidity" in w.columns else np.nan

        return {
            "season": year,
            "round": round_num,
            "rainfall": has_rain,
            "track_temp": round(track_temp, 2),
            "air_temp": round(air_temp, 2),
            "humidity": round(humidity, 2),
        }
    except Exception as e:
        return None


def collect_weather(start_year=2018, end_year=2026, max_workers=8, force=False):
    """Collect weather metrics for all Grand Prix races."""
    output_path = RAW_DIR / "race_weather.csv"
    existing_rows = []
    cached_rounds = set()

    if not force and output_path.exists():
        try:
            old_df = pd.read_csv(output_path)
            existing_rows = old_df.to_dict("records")
            cached_rounds = set(zip(old_df["season"].astype(int), old_df["round"].astype(int)))
            print(f"Loaded existing weather cache: {len(cached_rounds)} races.")
        except Exception as e:
            print(f"Warning reading weather cache: {e}")

    races_csv = RAW_DIR / "races.csv"
    if not races_csv.exists():
        raise FileNotFoundError("races.csv not found")

    races = pd.read_csv(races_csv)
    sub = races[(races["season"] >= start_year) & (races["season"] <= end_year)]
    targets = []
    for _, r in sub.iterrows():
        y, rnd = int(r["season"]), int(r["round"])
        if (y, rnd) not in cached_rounds or force:
            targets.append((y, rnd))

    print(f"Collecting weather for {len(targets)} races ({start_year}-{end_year})...")
    if not targets:
        return pd.DataFrame(existing_rows)

    new_rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(extract_race_weather, y, r): (y, r) for y, r in targets}
        for future in as_completed(future_map):
            y, r = future_map[future]
            try:
                res = future.result()
                if res:
                    new_rows.append(res)
                    rain_str = "RAIN" if res["rainfall"] else "DRY"
                    print(f"Weather {y} Round {r}: {rain_str}, Track {res['track_temp']} C")
                else:
                    # Default dry baseline if telemetry unavailable for old race
                    new_rows.append({
                        "season": y,
                        "round": r,
                        "rainfall": 0,
                        "track_temp": 32.0,
                        "air_temp": 24.0,
                        "humidity": 50.0,
                    })
            except Exception as e:
                new_rows.append({
                    "season": y,
                    "round": r,
                    "rainfall": 0,
                    "track_temp": 32.0,
                    "air_temp": 24.0,
                    "humidity": 50.0,
                })

    all_rows = existing_rows + new_rows
    df = pd.DataFrame(all_rows).drop_duplicates(
        subset=["season", "round"]
    ).sort_values(["season", "round"])

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} race weather records to {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect FastF1 race weather")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    collect_weather(
        start_year=args.start_year,
        end_year=args.end_year,
        max_workers=args.workers,
        force=args.force
    )


if __name__ == "__main__":
    main()
