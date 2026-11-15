"""One-command refresh: pull new races, rebuild features, (optionally) re-run the backtest.

    python src/update_data.py              # after any race weekend
    python src/update_data.py --backtest   # also refresh reports/ (slower)
    python src/update_data.py --skip-collect   # just rebuild features from data/raw
    python src/update_data.py --with-practice  # also FastF1 practice pace (asks for F1 login)

Collectors are incremental: only rounds missing from data/raw are fetched.
The dashboard and live predictions retrain on data/processed/model_dataset_v2.csv,
so once this has run, the model includes every completed race.
"""
import argparse
import datetime as dt
import traceback

import pandas as pd

import collect_data
import collect_practice_pace
import collect_sprints
from features_v2 import build, RAW


def _step(name, fn, required=False):
    print(f"\n=== {name} ===")
    try:
        return fn()
    except Exception as e:
        traceback.print_exc()
        if required:
            raise
        print(f"[warn] {name} failed ({e}); continuing with existing data.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--with-practice", action="store_true",
                    help="also fetch FastF1 practice pace (needs an F1 account login; optional, "
                         "removing it did not change accuracy)")
    ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args()
    year = dt.date.today().year

    if not a.skip_collect:
        def results():
            races = collect_data.collect_races(start_year=2014, end_year=year)
            return collect_data.collect_results_and_qualifying(races)
        _step("Race results + qualifying (Jolpica)", results, required=True)
        _step("Sprint results", lambda: collect_sprints.collect_sprints(2021, year))
        if a.with_practice:
            _step("Practice pace (FastF1)",
                  lambda: collect_practice_pace.collect_practice_pace(start_year=2024, end_year=year))
        else:
            print("\n(Skipping FastF1 practice pace; use --with-practice to include it.)")

    _step("Rebuild features", lambda: build(save=True), required=True)

    res = pd.read_csv(RAW / "race_results.csv")
    last = res.sort_values(["season", "round"]).iloc[-1]
    print(f"\nModel data now runs through {int(last.season)} round {int(last['round'])} "
          f"({last.race_name}, {last.race_date}).")

    if a.backtest:
        import backtest_v2
        _step("Walk-forward backtest + reports", backtest_v2.main)


if __name__ == "__main__":
    main()