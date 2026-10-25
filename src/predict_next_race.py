import fastf1
import re
from datetime import date
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import requests


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
SEASON = 2026

POSITION_MODEL_PATH = (
    MODEL_DIR / "production_position_model.joblib"
)
PODIUM_MODEL_PATH = (
    MODEL_DIR / "production_podium_model.joblib"
)

NUMERIC_FEATURES = [
    "season",
    "round",
    "grid_position",
    "qualifying_position",
    "grid_vs_qualifying",
    "driver_prev_finish",
    "driver_avg_finish_last3",
    "driver_avg_finish_last5",
    "driver_prev_races",
    "constructor_avg_finish_last5",
]

CATEGORICAL_FEATURES = [
    "race_name",
    "driver_id",
    "constructor_id",
    "circuit_id",
    "country",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

session = requests.Session()
session.headers.update({"User-Agent": "F1RacePredictor/1.0"})


# --------------------------------------------------
# API helpers
# --------------------------------------------------

def fetch_json(url):
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_race_schedule(season):
    url = f"{BASE_URL}/{season}/races.json?limit=100"
    data = fetch_json(url)
    return data["MRData"]["RaceTable"].get("Races", [])


def fetch_round_data(season, round_number, endpoint):
    url = (
        f"{BASE_URL}/{season}/{round_number}/"
        f"{endpoint}.json?limit=100"
    )
    data = fetch_json(url)
    return data["MRData"]["RaceTable"].get("Races", [])


def fetch_race_results(season, round_number):
    races = fetch_round_data(season, round_number, "results")
    if not races:
        return []

    return races[0].get("Results", [])



def fetch_qualifying(season, round_number):
    """Fetch qualifying results using FastF1."""
    session = fastf1.get_session(
        season, round_number, "Q"
    )
    session.load()

    results = session.results

    if results is None or results.empty:
        return []

    return results.to_dict("records")



# --------------------------------------------------
# Find the next race
# --------------------------------------------------

def find_next_race():
    races = fetch_race_schedule(SEASON)
    today = date.today()

    candidates = []

    for race in races:
        race_date = pd.to_datetime(
            race.get("date"), errors="coerce"
        )

        if pd.isna(race_date):
            continue

        if race_date.date() < today:
            continue

        round_number = int(race["round"])

        # Skip races whose official results already exist.
        results = fetch_race_results(SEASON, round_number)
        if results:
            continue

        candidates.append((race_date, race))
        time.sleep(0.2)

    if not candidates:
        raise RuntimeError(
            f"No uncompleted race found in the {SEASON} schedule "
            f"on or after {today}. Check the API schedule, or update "
            "SEASON in this script if you want another season."
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


# --------------------------------------------------
# Historical data
# --------------------------------------------------

def load_historical_results(target_round):
    path = RAW_DIR / "race_results.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Historical data missing: {path}\n"
            "Run the Phase 1 collection script first."
        )

    history = pd.read_csv(path)

    history["season"] = pd.to_numeric(
        history["season"], errors="coerce"
    )
    history["round"] = pd.to_numeric(
        history["round"], errors="coerce"
    )
    history["finish_position"] = pd.to_numeric(
        history["finish_position"], errors="coerce"
    )

    history = history.dropna(
        subset=[
            "season",
            "round",
            "finish_position",
            "driver_id",
            "constructor_id",
        ]
    ).copy()

    # Fetch earlier completed races from the target season.
    # These results weren't present in the original 2018-2025 data.
    for round_number in range(1, target_round):
        results = fetch_race_results(SEASON, round_number)

        if not results:
            continue

        # Avoid duplicating a season-round already in local history.
        already_present = (
            (history["season"] == SEASON)
            & (history["round"] == round_number)
        ).any()

        if already_present:
            continue

        for result in results:
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})

            history = pd.concat(
                [
                    history,
                    pd.DataFrame([{
                        "season": SEASON,
                        "round": round_number,
                        "race_date": None,
                        "driver_id": driver.get("driverId"),
                        "driver_name": (
                            f"{driver.get('givenName', '')} "
                            f"{driver.get('familyName', '')}"
                        ).strip(),
                        "constructor_id": constructor.get(
                            "constructorId"
                        ),
                        "constructor_name": constructor.get("name"),
                        "finish_position": result.get("position"),
                    }]),
                ],
                ignore_index=True,
            )

        time.sleep(0.2)

    history["season"] = pd.to_numeric(
        history["season"], errors="coerce"
    )
    history["round"] = pd.to_numeric(
        history["round"], errors="coerce"
    )
    history["finish_position"] = pd.to_numeric(
        history["finish_position"], errors="coerce"
    )

    history = history.dropna(
        subset=[
            "season",
            "round",
            "finish_position",
            "driver_id",
            "constructor_id",
        ]
    ).copy()

    # Only races before the target race can contribute to history.
    history = history[
        (history["season"] < SEASON)
        | (
            (history["season"] == SEASON)
            & (history["round"] < target_round)
        )
    ].copy()

    history["season"] = history["season"].astype(int)
    history["round"] = history["round"].astype(int)

    history = history.drop_duplicates(
        subset=["season", "round", "driver_id"]
    )

    return history.sort_values(
        ["season", "round"]
    ).reset_index(drop=True)


# --------------------------------------------------
# Build features for the selected race
# --------------------------------------------------


def normalize_name(value):
    """Normalize a name for matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def build_race_entries(race, history):
    season = int(race["season"])
    round_number = int(race["round"])

    qualifying = fetch_qualifying(season, round_number)

    if not qualifying:
        raise RuntimeError(
            f"FastF1 returned no qualifying results for "
            f"{race['raceName']}."
        )

    circuit = race.get("Circuit", {})
    location = circuit.get("Location", {})

    # Map historical driver abbreviations to the IDs used in training.
    driver_code_map = {}

    if "driver_code" in history.columns:
        for _, row in history.dropna(
            subset=["driver_code", "driver_id"]
        ).iterrows():
            driver_code_map[str(row["driver_code"]).upper()] = (
                str(row["driver_id"])
            )

    # Map team names from FastF1 to the historical constructor IDs.
    constructor_map = {}

    for _, row in history.dropna(
        subset=["constructor_name", "constructor_id"]
    ).iterrows():
        key = normalize_name(row["constructor_name"])
        constructor_map[key] = str(row["constructor_id"])

    # Explicit aliases for common naming differences.
    team_aliases = {
        "redbullracing": "red_bull",
        "redbull": "red_bull",
        "racingbulls": "racing_bulls",
        "alphatauri": "alphatauri",
        "tororosso": "toro_rosso",
        "astonmartin": "aston_martin",
        "haasf1team": "haas",
        "haas": "haas",
        "kicksauber": "sauber",
        "stakef1teamkicksauber": "sauber",
        "alpine": "alpine",
        "mercedes": "mercedes",
        "ferrari": "ferrari",
        "mclaren": "mclaren",
        "williams": "williams",
        "audi": "audi",
        "cadillac": "cadillac",
    }

    # Use the latest known driver name for each historical driver ID.
    driver_name_map = {}

    if "driver_name" in history.columns:
        for _, row in history.dropna(
            subset=["driver_id", "driver_name"]
        ).iterrows():
            driver_name_map[str(row["driver_id"])] = (
                str(row["driver_name"])
            )

    entries = []

    for result in qualifying:
        abbreviation = str(result.get("Abbreviation", "")).upper()
        full_name = result.get("FullName", abbreviation)
        team_name = result.get("TeamName", "")

        q_position = pd.to_numeric(
            result.get("Position"), errors="coerce"
        )

        if pd.isna(q_position) or not abbreviation:
            continue

        # Reuse known historical IDs wherever possible.
        driver_id = driver_code_map.get(
            abbreviation, abbreviation.lower()
        )

        team_key = normalize_name(team_name)
        constructor_id = team_aliases.get(team_key)

        if constructor_id is None:
            constructor_id = constructor_map.get(
                team_key, team_key
            )

        entries.append({
            "season": season,
            "round": round_number,
            "race_id": f"{season}_{round_number}",
            "race_name": race["raceName"],
            "race_date": race.get("date"),
            "driver_id": driver_id,
            "driver_name": full_name,
            "constructor_id": constructor_id,
            "constructor_name": team_name,
            "circuit_id": circuit.get("circuitId"),
            "country": location.get("country"),
            "qualifying_position": float(q_position),
            "grid_position": float(q_position),
            "grid_vs_qualifying": 0.0,
        })

    df = pd.DataFrame(entries)

    if df.empty:
        raise RuntimeError("No valid driver entries were produced.")

    # Historical driver performance, using earlier races only.
    history = history.sort_values(["season", "round"]).copy()

    driver_history = {}

    for driver_id, group in history.groupby("driver_id"):
        finishes = group["finish_position"].astype(float).tolist()

        if finishes:
            driver_history[str(driver_id)] = {
                "driver_prev_finish": finishes[-1],
                "driver_avg_finish_last3": float(
                    np.mean(finishes[-3:])
                ),
                "driver_avg_finish_last5": float(
                    np.mean(finishes[-5:])
                ),
                "driver_prev_races": len(finishes),
            }

    # Map FastF1 abbreviations to historical IDs for driver-form lookup.
    for column in [
        "driver_prev_finish",
        "driver_avg_finish_last3",
        "driver_avg_finish_last5",
        "driver_prev_races",
    ]:
        df[column] = df["driver_id"].map(
            lambda driver_id: driver_history.get(
                str(driver_id), {}
            ).get(column, np.nan)
        )

    # Constructor form based on previous completed races.
    constructor_races = (
        history.groupby(
            ["constructor_id", "season", "round"],
            as_index=False,
        )
        .agg(
            constructor_race_avg_finish=(
                "finish_position", "mean"
            )
        )
        .sort_values(["season", "round"])
    )

    constructor_history = {}

    for constructor_id, group in constructor_races.groupby(
        "constructor_id"
    ):
        values = group["constructor_race_avg_finish"].tolist()

        if values:
            constructor_history[str(constructor_id)] = float(
                np.mean(values[-5:])
            )

    df["constructor_avg_finish_last5"] = (
        df["constructor_id"].astype(str).map(
            constructor_history
        )
    )

    return df


# --------------------------------------------------
# Predict and save
# --------------------------------------------------

def main():
    if not POSITION_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Production position model not found. Run "
            "python src\\train_production.py first."
        )

    if not PODIUM_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Production podium model not found. Run "
            "python src\\train_production.py first."
        )

    print(f"Looking for the next uncompleted {SEASON} race...")

    race = find_next_race()
    round_number = int(race["round"])

    print(f"\nSelected race: {race['raceName']}")
    print(f"Date: {race.get('date')}")
    print(f"Round: {round_number}")

    history = load_historical_results(round_number)
    entries = build_race_entries(race, history)

    position_model = joblib.load(POSITION_MODEL_PATH)
    podium_model = joblib.load(PODIUM_MODEL_PATH)

    X = entries[FEATURES]

    entries["predicted_finish_position"] = (
        position_model.predict(X)
    )

    entries["podium_probability"] = (
        podium_model.predict_proba(X)[:, 1]
    )

    entries["predicted_order"] = (
        entries["predicted_finish_position"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    entries["predicted_podium_rank"] = (
        entries["podium_probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    entries = entries.sort_values("predicted_order")

    output_path = REPORT_DIR / "latest_race_prediction.csv"
    entries.to_csv(output_path, index=False)

    display_columns = [
        "predicted_order",
        "driver_name",
        "constructor_name",
        "qualifying_position",
        "predicted_finish_position",
        "podium_probability",
    ]

    display = entries[display_columns].copy()
    display["predicted_finish_position"] = (
        display["predicted_finish_position"].round(1)
    )
    display["podium_probability"] = (
        display["podium_probability"].map(
            lambda value: f"{value:.1%}"
        )
    )

    print("\nPREDICTED FINISHING ORDER")
    print(display.to_string(index=False))

    print(f"\nSaved prediction to: {output_path}")
    print(
        "\nReminder: these are model estimates, not guarantees. "
        "Qualifying position is used as a grid proxy, and actual "
        "grid penalties, weather, incidents, and strategy can change "
        "the result."
    )


if __name__ == "__main__":
    main()
