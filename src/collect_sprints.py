"""Collect Sprint race results from Jolpica / Ergast API.

Fetches all sprint races across 2021-2026.
Saves to data/raw/sprint_results.csv.
"""
from pathlib import Path
import time
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.jolpi.ca/ergast/f1"


def collect_sprints(start_year=2021, end_year=2026):
    all_sprints = []
    print(f"Collecting Sprint results ({start_year}-{end_year})...")
    
    session = requests.Session()
    session.headers.update({"User-Agent": "F1RacePredictor/2.0"})

    for year in range(start_year, end_year + 1):
        url = f"{BASE_URL}/{year}/sprint.json?limit=1000"
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            races = resp.json()["MRData"]["RaceTable"].get("Races", [])
            for race in races:
                season = int(race["season"])
                rnd = int(race["round"])
                race_name = race.get("raceName")
                for res in race.get("SprintResults", []):
                    drv = res.get("Driver", {})
                    grid = pd.to_numeric(res.get("grid"), errors="coerce")
                    pos = pd.to_numeric(res.get("position"), errors="coerce")
                    pts = pd.to_numeric(res.get("points"), errors="coerce")
                    laps = pd.to_numeric(res.get("laps"), errors="coerce")
                    
                    all_sprints.append({
                        "season": season,
                        "round": rnd,
                        "race_name": race_name,
                        "driver_id": drv.get("driverId"),
                        "sprint_grid": grid,
                        "sprint_finish": pos,
                        "sprint_points": pts,
                        "sprint_laps": laps,
                        "sprint_status": res.get("status"),
                        "sprint_pos_gain": (grid - pos) if pd.notna(grid) and pd.notna(pos) else 0.0,
                    })
            print(f"Season {year}: {len(races)} sprint races collected.")
            time.sleep(0.2)
        except Exception as e:
            print(f"Error fetching {year} sprints: {e}")

    df = pd.DataFrame(all_sprints).drop_duplicates(subset=["season", "round", "driver_id"])
    out_path = RAW_DIR / "sprint_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} sprint driver records to {out_path}")
    return df


if __name__ == "__main__":
    collect_sprints()
