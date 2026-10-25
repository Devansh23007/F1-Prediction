
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "model_dataset.csv"
REPORT_DIR = ROOT / "reports"
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


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])


def make_pipeline(model):
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", model),
    ])


def evaluate_model(name, position_model, podium_model, train, test):
    X_train = train[FEATURES]
    X_test = test[FEATURES]

    position_model.fit(X_train, train["finish_position"])
    position_prediction = position_model.predict(X_test)

    podium_model.fit(X_train, train["podium"].astype(int))
    podium_probability = podium_model.predict_proba(X_test)[:, 1]

    position_mae = mean_absolute_error(
        test["finish_position"], position_prediction
    )

    position_rmse = np.sqrt(mean_squared_error(
        test["finish_position"], position_prediction
    ))

    actual_podium = test["podium"].astype(int)

    auc = roc_auc_score(actual_podium, podium_probability)
    brier = brier_score_loss(actual_podium, podium_probability)

    # Evaluate the top three within each race.
    evaluation = test[
        ["race_id", "finish_position", "podium"]
    ].copy()

    evaluation["predicted_position"] = position_prediction
    evaluation["podium_probability"] = podium_probability

    position_hits = []
    probability_hits = []
    exact_podium_matches = 0

    for _, race in evaluation.groupby("race_id"):
        actual = set(race.index[race["podium"] == 1])

        predicted_position_top3 = set(
            race.nsmallest(3, "predicted_position").index
        )
        predicted_probability_top3 = set(
            race.nlargest(3, "podium_probability").index
        )

        hit_position = len(actual & predicted_position_top3)
        hit_probability = len(actual & predicted_probability_top3)

        position_hits.append(hit_position)
        probability_hits.append(hit_probability)

        if hit_probability == 3 and len(actual) == 3:
            exact_podium_matches += 1

    result = {
        "model": name,
        "position_mae": float(position_mae),
        "position_rmse": float(position_rmse),
        "podium_roc_auc": float(auc),
        "podium_brier_score": float(brier),
        "mean_podium_hits_from_position_top3": float(
            np.mean(position_hits)
        ),
        "mean_podium_hits_from_probability_top3": float(
            np.mean(probability_hits)
        ),
        "exact_podium_matches": int(exact_podium_matches),
        "test_races": int(test["race_id"].nunique()),
    }

    return result


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

    train = df[df["season"] < TEST_SEASON].copy()
    test = df[df["season"] == TEST_SEASON].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"Need training seasons before {TEST_SEASON} and "
            f"test data from {TEST_SEASON}."
        )

    # Keep only the required columns and make missing categories explicit.
    train["podium"] = train["podium"].astype(int)
    test["podium"] = test["podium"].astype(int)

    print(f"Training rows: {len(train):,}")
    print(f"Testing rows: {len(test):,}")
    print(f"Test races: {test['race_id'].nunique()}")

    models = {
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=350,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            ),
            XGBClassifier(
                n_estimators=350,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }

    results = []

    for name, (regressor, classifier) in models.items():
        print(f"\nTraining and evaluating {name}...")

        result = evaluate_model(
            name,
            make_pipeline(regressor),
            make_pipeline(classifier),
            train,
            test,
        )

        results.append(result)
        print(json.dumps(result, indent=2))

    results_df = pd.DataFrame(results)
    output = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(output, index=False)

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison to {output}")


if __name__ == "__main__":
    main()
