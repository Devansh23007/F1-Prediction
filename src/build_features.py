
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)


def load_data():
    required = [
        RAW / "race_results.csv",
        RAW / "qualifying_results.csv",
        RAW / "races.csv",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    results = pd.read_csv(required[0])
    qualifying = pd.read_csv(required[1])
    races = pd.read_csv(required[2])

    return results, qualifying, races


def clean_data(results, qualifying, races):
    # Keep race identifiers consistent across datasets.
    for df in (results, qualifying, races):
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        df.dropna(subset=["season", "round"], inplace=True)
        df["season"] = df["season"].astype(int)
        df["round"] = df["round"].astype(int)

    results["race_date"] = pd.to_datetime(
        results["race_date"], errors="coerce"
    )

    # Remove duplicate driver records within a race.
    results = results.drop_duplicates(
        subset=["season", "round", "driver_id"]
    ).copy()

    qualifying = qualifying.drop_duplicates(
        subset=["season", "round", "driver_id"]
    ).copy()

    # Convert numeric columns safely.
    numeric_columns = [
        "grid_position",
        "finish_position",
        "points",
        "laps_completed",
        "fastest_lap_rank",
    ]

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column], errors="coerce"
        )

    qualifying["qualifying_position"] = pd.to_numeric(
        qualifying["qualifying_position"], errors="coerce"
    )

    # A numeric finishing position is the target for our first model.
    # Drivers with nonnumeric classifications remain in the cleaned
    # source but are excluded from this initial supervised dataset.
    results = results.dropna(
        subset=["finish_position", "driver_id", "constructor_id"]
    ).copy()

    results["finish_position"] = results["finish_position"].astype(int)

    return results, qualifying, races


def build_features(results, qualifying, races):
    # Merge qualifying data by race and driver.
    q = qualifying[
        [
            "season",
            "round",
            "driver_id",
            "qualifying_position",
        ]
    ].copy()

    df = results.merge(
        q,
        on=["season", "round", "driver_id"],
        how="left",
        validate="one_to_one",
    )

    # Bring in circuit and calendar information.
    race_info = races[
        ["season", "round", "circuit_id", "country"]
    ].drop_duplicates(["season", "round"])

    df = df.merge(
        race_info,
        on=["season", "round"],
        how="left",
        suffixes=("", "_schedule"),
        validate="many_to_one",
    )

    df["qualifying_position"] = df["qualifying_position"].fillna(
        df["grid_position"]
    )

    df["grid_position"] = df["grid_position"].replace(0, np.nan)

    # Sort chronologically before calculating historical features.
    df = df.sort_values(
        ["season", "round", "driver_id"]
    ).reset_index(drop=True)

    # Race history features: shift first, then calculate rolling means.
    # This ensures the current race result is never used as its own input.
    driver_groups = df.groupby("driver_id", sort=False)

    df["driver_prev_finish"] = driver_groups[
        "finish_position"
    ].shift(1)

    df["driver_avg_finish_last3"] = driver_groups[
        "finish_position"
    ].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    df["driver_avg_finish_last5"] = driver_groups[
        "finish_position"
    ].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    df["driver_prev_races"] = driver_groups.cumcount()


    # Constructor form: first aggregate results by constructor and race.
    # Then shift by one race before calculating the rolling average.
    # This prevents current-race results from leaking into the features.
    constructor_races = (
        df.groupby(
            ["constructor_id", "season", "round"],
            as_index=False
        )
        .agg(
            constructor_race_avg_finish=(
                "finish_position", "mean"
            )
        )
        .sort_values(["season", "round"])
    )

    constructor_races["constructor_avg_finish_last5"] = (
        constructor_races.groupby(
            "constructor_id"
        )["constructor_race_avg_finish"]
        .transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
    )

    df = df.merge(
        constructor_races[
            [
                "constructor_id",
                "season",
                "round",
                "constructor_avg_finish_last5",
            ]
        ],
        on=["constructor_id", "season", "round"],
        how="left",
        validate="many_to_one",
    )

    # Relative qualifying performance within the same race.
    df["grid_position"] = pd.to_numeric(
        df["grid_position"], errors="coerce"
    )

    df["qualifying_position"] = pd.to_numeric(
        df["qualifying_position"], errors="coerce"
    )

    df["grid_vs_qualifying"] = (
        df["grid_position"] - df["qualifying_position"]
    )

    # Targets.
    df["podium"] = (
        df["finish_position"].between(1, 3)
    ).astype(int)

    # Use chronological identifiers for the future train/test split.
    df["race_id"] = (
        df["season"].astype(str)
        + "_"
        + df["round"].astype(str)
    )

    # Remove columns that directly reveal the final race outcome
    # or are not needed as model inputs.
    target_columns = [
        "finish_position",
        "podium",
        "race_id",
        "season",
        "round",
        "race_name",
        "race_date",
        "driver_id",
        "driver_name",
        "driver_code",
        "constructor_id",
        "constructor_name",
        "circuit_id",
        "country",
        "grid_position",
        "qualifying_position",
        "grid_vs_qualifying",
        "driver_prev_finish",
        "driver_avg_finish_last3",
        "driver_avg_finish_last5",
        "driver_prev_races",
        "constructor_avg_finish_last5",
    ]

    available = [c for c in target_columns if c in df.columns]
    df = df[available].copy()

    output = PROCESSED / "model_dataset.csv"
    df.to_csv(output, index=False)

    print(f"\nSaved model dataset: {output}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Races: {df['race_id'].nunique():,}")
    print(f"Drivers: {df['driver_id'].nunique():,}")
    print(f"Podium rate: {df['podium'].mean():.2%}")

    print("\nFeature dataset preview:")
    print(df.head(10).to_string(index=False))

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    return df


def main():
    results, qualifying, races = load_data()
    results, qualifying, races = clean_data(
        results, qualifying, races
    )

    if results.empty:
        raise ValueError(
            "No classified numeric finishing positions found."
        )

    df = build_features(results, qualifying, races)

    if df["podium"].nunique() < 2:
        raise ValueError("Podium target has only one class.")

    print("\nPhase 2 completed successfully.")


if __name__ == "__main__":
    main()
