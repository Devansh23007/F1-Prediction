import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_URL = "https://api.jolpi.ca/ergast/f1"
START_YEAR = 2018
END_YEAR = 2026

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "F1RacePredictor/2.0"})

retry_strategy = Retry(
    total=4,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)


def fetch_json(url):
    """Fetch JSON from the Jolpica API."""
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_races(season):
    """Get the schedule for one season."""
    url = f"{BASE_URL}/{season}/races.json?limit=100"
    data = fetch_json(url)
    return data["MRData"]["RaceTable"].get("Races", [])


def get_round_data(season, round_number, endpoint):
    """Get results or qualifying data for one race."""
    url = f"{BASE_URL}/{season}/{round_number}/{endpoint}.json?limit=100"
    data = fetch_json(url)
    return data["MRData"]["RaceTable"].get("Races", [])


def collect_races(start_year=START_YEAR, end_year=END_YEAR):
    """Collect race schedules across all selected seasons."""
    all_races = []
    for season in range(start_year, end_year + 1):
        print(f"Fetching {season} race schedule...")
        try:
            races = get_races(season)
            for race in races:
                circuit = race.get("Circuit", {})
                location = circuit.get("Location", {})
                all_races.append({
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race.get("raceName"),
                    "race_date": race.get("date"),
                    "circuit_id": circuit.get("circuitId"),
                    "circuit_name": circuit.get("circuitName"),
                    "country": location.get("country"),
                    "locality": location.get("locality"),
                })
            time.sleep(0.15)
        except requests.RequestException as error:
            print(f"Could not fetch schedule for {season}: {error}")
            raise

    df = pd.DataFrame(all_races)
    output = RAW_DIR / "races.csv"
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} races to {output}")
    return df


def collect_results_and_qualifying(races, force=False):
    """Collect driver-level race and qualifying records incrementally."""
    results_path = RAW_DIR / "race_results.csv"
    qualifying_path = RAW_DIR / "qualifying_results.csv"

    existing_results = []
    existing_qualifying = []
    already_collected_results = set()
    already_collected_qual = set()

    if not force and results_path.exists() and qualifying_path.exists():
        try:
            res_df_old = pd.read_csv(results_path)
            qual_df_old = pd.read_csv(qualifying_path)
            existing_results = res_df_old.to_dict("records")
            existing_qualifying = qual_df_old.to_dict("records")
            already_collected_results = set(zip(res_df_old["season"], res_df_old["round"]))
            already_collected_qual = set(zip(qual_df_old["season"], qual_df_old["round"]))
            print(f"Found existing data: {len(already_collected_results)} completed races cached.")
        except Exception as e:
            print(f"Warning: could not read existing cache: {e}. Re-fetching.")
            existing_results = []
            existing_qualifying = []

    new_results = []
    new_qualifying = []

    for race in races.to_dict("records"):
        season = int(race["season"])
        round_number = int(race["round"])
        race_name = race["race_name"]
        key = (season, round_number)

        need_results = key not in already_collected_results
        need_qual = key not in already_collected_qual

        if not need_results and not need_qual:
            continue

        print(f"Fetching {season} Round {round_number}: {race_name}...")

        try:
            if need_results:
                result_races = get_round_data(season, round_number, "results")
                for race_data in result_races:
                    for result in race_data.get("Results", []):
                        driver = result.get("Driver", {})
                        constructor = result.get("Constructor", {})
                        fastest_lap = result.get("FastestLap", {})
                        new_results.append({
                            "season": season,
                            "round": round_number,
                            "race_name": race_name,
                            "race_date": race["race_date"],
                            "circuit_id": race["circuit_id"],
                            "driver_id": driver.get("driverId"),
                            "driver_code": driver.get("code"),
                            "driver_name": (
                                f"{driver.get('givenName', '')} "
                                f"{driver.get('familyName', '')}"
                            ).strip(),
                            "constructor_id": constructor.get("constructorId"),
                            "constructor_name": constructor.get("name"),
                            "grid_position": result.get("grid"),
                            "finish_position": result.get("position"),
                            "finish_position_text": result.get("positionText"),
                            "points": result.get("points"),
                            "laps_completed": result.get("laps"),
                            "status": result.get("status"),
                            "fastest_lap_rank": fastest_lap.get("rank"),
                        })
                time.sleep(0.15)

            if need_qual:
                qualifying_races = get_round_data(season, round_number, "qualifying")
                for race_data in qualifying_races:
                    for result in race_data.get("QualifyingResults", []):
                        driver = result.get("Driver", {})
                        constructor = result.get("Constructor", {})
                        new_qualifying.append({
                            "season": season,
                            "round": round_number,
                            "race_name": race_name,
                            "driver_id": driver.get("driverId"),
                            "constructor_id": constructor.get("constructorId"),
                            "qualifying_position": result.get("position"),
                            "q1_time": result.get("Q1"),
                            "q2_time": result.get("Q2"),
                            "q3_time": result.get("Q3"),
                        })
                time.sleep(0.15)

        except requests.RequestException as error:
            print(f"API request failed for {season} Round {round_number}: {error}")
            continue

    all_results = existing_results + new_results
    all_qualifying = existing_qualifying + new_qualifying

    results_df = pd.DataFrame(all_results).drop_duplicates(
        subset=["season", "round", "driver_id"]
    ).sort_values(["season", "round", "finish_position"])

    qualifying_df = pd.DataFrame(all_qualifying).drop_duplicates(
        subset=["season", "round", "driver_id"]
    ).sort_values(["season", "round", "qualifying_position"])

    results_df.to_csv(results_path, index=False)
    qualifying_df.to_csv(qualifying_path, index=False)

    print("\nData collection completed.")
    print(f"Race results: {len(results_df)} total rows ({len(new_results)} new)")
    print(f"Qualifying results: {len(qualifying_df)} total rows ({len(new_qualifying)} new)")
    print(f"Saved to: {RAW_DIR}")

    return results_df, qualifying_df


def main():
    parser = argparse.ArgumentParser(description="Collect F1 race data from Jolpica API")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--force", action="store_true", help="Force re-fetch all historical data")
    args = parser.parse_args()

    print("=" * 55)
    print("F1 RACE PREDICTOR: DATA COLLECTION")
    print(f"Seasons: {args.start_year}-{args.end_year}")
    print("=" * 55)

    races = collect_races(start_year=args.start_year, end_year=args.end_year)
    if races.empty:
        raise RuntimeError("No race schedules were returned.")

    results, qualifying = collect_results_and_qualifying(races, force=args.force)
    if results.empty:
        raise RuntimeError("No race results were collected.")

    print("\nQuick validation:")
    print(f"Seasons found: {sorted(results['season'].unique())}")
    print(f"Unique races: {results[['season', 'round']].drop_duplicates().shape[0]}")
    print(f"Unique drivers: {results['driver_id'].nunique()}")
    print(f"Unique constructors: {results['constructor_id'].nunique()}")


if __name__ == "__main__":
    main()
