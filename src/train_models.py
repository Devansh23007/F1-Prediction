
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    brier_score_loss,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "model_dataset.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SEASON = 2025

CATEGORICAL_FEATURES = [
    "race_name",
    "driver_id",
    "constructor_id",
    "circuit_id",
    "country",
]

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

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def make_regressor():
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def make_classifier():
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run python src\\build_features.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [
        "finish_position",
        "podium",
        "season",
        "race_id",
        "driver_name",
    ]

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

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
            f"Need both training seasons before {TEST_SEASON} "
            f"and test data from {TEST_SEASON}. "
            f"Available seasons: {sorted(df['season'].unique())}"
        )

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    y_train_position = train["finish_position"]
    y_test_position = test["finish_position"]

    y_train_podium = train["podium"].astype(int)
    y_test_podium = test["podium"].astype(int)

    print("=" * 60)
    print("F1 RACE PREDICTOR: MODEL TRAINING")
    print("=" * 60)
    print(f"Training rows: {len(train):,}")
    print(f"Training seasons: {sorted(train['season'].unique())}")
    print(f"Test rows: {len(test):,}")
    print(f"Test season: {TEST_SEASON}")
    print(f"Test races: {test['race_id'].nunique():,}")

    # Model 1: finishing-position regression.
    print("\nTraining finishing-position model...")
    regressor = make_regressor()
    regressor.fit(X_train, y_train_position)

    predicted_position = regressor.predict(X_test)

    mae = mean_absolute_error(
        y_test_position, predicted_position
    )
    rmse = np.sqrt(
        mean_squared_error(y_test_position, predicted_position)
    )

    # Model 2: podium probability classification.
    print("Training podium-probability model...")
    classifier = make_classifier()
    classifier.fit(X_train, y_train_podium)

    podium_probability = classifier.predict_proba(X_test)[:, 1]
    predicted_podium = (podium_probability >= 0.5).astype(int)

    accuracy = accuracy_score(y_test_podium, predicted_podium)
    brier = brier_score_loss(y_test_podium, podium_probability)

    auc = (
        roc_auc_score(y_test_podium, podium_probability)
        if y_test_podium.nunique() == 2
        else None
    )

    print("\n" + "=" * 60)
    print("TEST METRICS")
    print("=" * 60)
    print(f"Finishing-position MAE: {mae:.3f}")
    print(f"Finishing-position RMSE: {rmse:.3f}")
    print(f"Podium classification accuracy: {accuracy:.3f}")
    print(f"Podium probability Brier score: {brier:.3f}")
    print(f"Podium ROC-AUC: {auc if auc is None else round(auc, 3)}")

    print("\nPodium classification report:")
    print(classification_report(
        y_test_podium,
        predicted_podium,
        zero_division=0,
    ))

    # Rank drivers within each race using predicted finishing position.
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

    predictions["predicted_finish_position"] = predicted_position
    predictions["podium_probability"] = podium_probability

    predictions["predicted_order"] = (
        predictions.groupby("race_id")["predicted_finish_position"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    predictions["predicted_podium"] = (
        predictions.groupby("race_id")["podium_probability"]
        .rank(method="first", ascending=False)
        .le(3)
        .astype(int)
    )

    top3_accuracy = (
        predictions.groupby("race_id")
        .apply(
            lambda race: (
                race.loc[race["predicted_podium"] == 1, "podium"]
                .sum() / 3
            ),
            include_groups=False,
        )
        .mean()
    )

    print(f"Mean actual podium drivers in predicted top 3: "
          f"{top3_accuracy * 3:.2f} out of 3")

    predictions = predictions.sort_values(
        ["season", "round", "predicted_order"]
    )

    predictions.to_csv(
        REPORT_DIR / "2025_race_predictions.csv",
        index=False,
    )

    metrics = {
        "test_season": TEST_SEASON,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_races": int(test["race_id"].nunique()),
        "finishing_position_mae": float(mae),
        "finishing_position_rmse": float(rmse),
        "podium_accuracy_at_threshold_0_5": float(accuracy),
        "podium_brier_score": float(brier),
        "podium_roc_auc": None if auc is None else float(auc),
        "mean_actual_podiums_in_predicted_top3": float(
            top3_accuracy * 3
        ),
    }

    with open(
        REPORT_DIR / "model_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nSample predicted finishing orders:")
    for race_id, race in predictions.groupby("race_id", sort=False):
        print(f"\n{race.iloc[0]['race_name']} ({race_id})")
        print(
            race[
                [
                    "predicted_order",
                    "driver_name",
                    "constructor_name",
                    "predicted_finish_position",
                    "podium_probability",
                    "finish_position",
                ]
            ].head(5).to_string(index=False)
        )

        # Show a few races for inspection.
        if race_id == predictions["race_id"].drop_duplicates().iloc[
            min(2, predictions["race_id"].nunique() - 1)
        ]:
            break

    print("\nSaved predictions to reports/2025_race_predictions.csv")
    print("Saved metrics to reports/model_metrics.json")
    print("\nTraining completed.")


if __name__ == "__main__":
    main()
