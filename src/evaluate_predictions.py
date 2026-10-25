
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "reports" / "2025_race_predictions.csv"
REPORTS = ROOT / "reports"

def main():
    if not PREDICTIONS.exists():
        raise FileNotFoundError(
            "Run python src\\train_models.py first."
        )

    df = pd.read_csv(PREDICTIONS)

    required = [
        "race_id",
        "finish_position",
        "podium",
        "predicted_finish_position",
        "podium_probability",
        "predicted_order",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required).copy()

    # Overall regression metric.
    mae = mean_absolute_error(
        df["finish_position"],
        df["predicted_finish_position"],
    )

    race_results = []

    for race_id, race in df.groupby("race_id"):
        # Top three by predicted finishing position.
        predicted_order = race.nsmallest(3, "predicted_order")
        actual_podium = race[race["podium"] == 1]

        predicted_drivers = set(predicted_order.index)
        actual_drivers = set(actual_podium.index)

        overlap = len(predicted_drivers & actual_drivers)

        # Top three by estimated podium probability.
        probability_top3 = race.nlargest(3, "podium_probability")
        probability_drivers = set(probability_top3.index)
        probability_overlap = len(probability_drivers & actual_drivers)

        race_results.append({
            "race_id": race_id,
            "drivers_evaluated": len(race),
            "predicted_order_podium_hits": overlap,
            "probability_top3_podium_hits": probability_overlap,
            "actual_podium_drivers": len(actual_podium),
        })

    report = pd.DataFrame(race_results)

    print("=" * 55)
    print("2025 RACE-LEVEL EVALUATION")
    print("=" * 55)
    print(f"Driver records evaluated: {len(df)}")
    print(f"Races evaluated: {len(report)}")
    print(f"Finishing-position MAE: {mae:.3f}")
    print(
        "Average actual podium drivers in predicted finishing top 3: "
        f"{report['predicted_order_podium_hits'].mean():.2f} / 3"
    )
    print(
        "Average actual podium drivers in probability top 3: "
        f"{report['probability_top3_podium_hits'].mean():.2f} / 3"
    )
    print(
        "Exact podium matches using probability ranking: "
        f"{(report['probability_top3_podium_hits'] == 3).sum()} "
        f"out of {len(report)} races"
    )

    report.to_csv(
        REPORTS / "race_level_evaluation.csv",
        index=False,
    )

    print("\nSaved reports/race_level_evaluation.csv")

if __name__ == "__main__":
    main()
