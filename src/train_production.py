
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "model_dataset.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC = [
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

CATEGORICAL = [
    "race_name",
    "driver_id",
    "constructor_id",
    "circuit_id",
    "country",
]

FEATURES = NUMERIC + CATEGORICAL


def make_pipeline(model):
    preprocessor = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            NUMERIC,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAL,
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run python src\\build_features.py first."
        )

    df = pd.read_csv(DATA_PATH)

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["finish_position"] = pd.to_numeric(
        df["finish_position"], errors="coerce"
    )
    df["podium"] = pd.to_numeric(df["podium"], errors="coerce")

    df = df.dropna(
        subset=["season", "finish_position", "podium"]
    ).copy()

    # Use the full historical dataset for production training.
    X = df[FEATURES]
    y_position = df["finish_position"]
    y_podium = df["podium"].astype(int)

    position_model = make_pipeline(
        RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    )

    podium_model = make_pipeline(
        RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    )

    print("Training production finishing-position model...")
    position_model.fit(X, y_position)

    print("Training production podium-probability model...")
    podium_model.fit(X, y_podium)

    joblib.dump(
        position_model,
        MODEL_DIR / "production_position_model.joblib",
    )

    joblib.dump(
        podium_model,
        MODEL_DIR / "production_podium_model.joblib",
    )

    print("\nProduction training completed.")
    print(f"Training records: {len(df):,}")
    print(f"Seasons: {sorted(df['season'].astype(int).unique())}")
    print(f"Features: {len(FEATURES)}")
    print("\nSaved:")
    print("models/production_position_model.joblib")
    print("models/production_podium_model.joblib")


if __name__ == "__main__":
    main()
