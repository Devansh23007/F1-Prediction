
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    roc_auc_score,
    brier_score_loss,
)


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "model_dataset.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SEASON = 2025

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


def make_model(model):
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
    if not DATA.exists():
        raise FileNotFoundError(
            "Run python src\\build_features.py first."
        )

    df = pd.read_csv(DATA)

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["finish_position"] = pd.to_numeric(
        df["finish_position"], errors="coerce"
    )
    df["podium"] = pd.to_numeric(df["podium"], errors="coerce")

    df = df.dropna(
        subset=["season", "finish_position", "podium"]
    ).copy()

    train = df[df["season"] < TEST_SEASON].copy()
    test = df[df["season"] == TEST_SEASON].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"Need training data before {TEST_SEASON} and "
            f"test data from {TEST_SEASON}."
        )

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    position_model = make_model(
        RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
    )

    podium_model = make_model(
        RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    )

    print("Training champion finishing-position model...")
    position_model.fit(X_train, train["finish_position"])

    print("Training champion podium-probability model...")
    podium_model.fit(X_train, train["podium"].astype(int))

    # Save fitted pipelines, including preprocessing.
    joblib.dump(
        position_model,
        MODEL_DIR / "champion_position_model.joblib",
    )
    joblib.dump(
        podium_model,
        MODEL_DIR / "champion_podium_model.joblib",
    )

    # Evaluate on the untouched 2025 test season.
    position_prediction = position_model.predict(X_test)
    probability = podium_model.predict_proba(X_test)[:, 1]

    actual_podium = test["podium"].astype(int)

    metrics = {
        "champion": "Random Forest",
        "training_seasons": sorted(
            train["season"].astype(int).unique().tolist()
        ),
        "test_season": TEST_SEASON,
        "position_mae": float(
            mean_absolute_error(
                test["finish_position"], position_prediction
            )
        ),
        "podium_roc_auc": float(
            roc_auc_score(actual_podium, probability)
        ),
        "podium_brier_score": float(
            brier_score_loss(actual_podium, probability)
        ),
    }

    with open(
        REPORT_DIR / "champion_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    # Generate transformed feature names and importance reports.
    preprocessor = position_model.named_steps["preprocessor"]
    model = position_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False,
    )

    # Create reusable predictions for the historical test season.
    predictions = test[
        [
            "season",
            "round",
            "race_id",
            "race_name",
            "driver_id",
            "driver_name",
            "constructor_name",
            "finish_position",
            "podium",
        ]
    ].copy()

    predictions["predicted_finish_position"] = position_prediction
    predictions["podium_probability"] = probability

    predictions["predicted_order"] = (
        predictions.groupby("race_id")[
            "predicted_finish_position"
        ].rank(method="first").astype(int)
    )

    predictions = predictions.sort_values(
        ["season", "round", "predicted_order"]
    )

    predictions.to_csv(
        REPORT_DIR / "champion_predictions.csv",
        index=False,
    )

    print("\nChampion training completed.")
    print(json.dumps(metrics, indent=4))
    print("\nSaved models:")
    print("models/champion_position_model.joblib")
    print("models/champion_podium_model.joblib")
    print("\nSaved reports:")
    print("reports/champion_metrics.json")
    print("reports/feature_importance.csv")
    print("reports/champion_predictions.csv")


if __name__ == "__main__":
    main()
