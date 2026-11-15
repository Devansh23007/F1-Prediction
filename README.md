# F1 Race Predictor

F1 Race Predictor is a Python project for exploring Grand Prix results and
estimating race outcomes. It combines qualifying and grid position with
historical driver and team performance, practice pace, and circuit information.
A Streamlit dashboard presents predictions, championship simulations, and
historical analysis.

The starting grid is already a strong predictor in Formula 1, so the project
compares its model with a simple grid-order baseline. Predictions are estimates,
not guarantees.

## Getting started

The project is intended for Python 3.10 or later. From PowerShell, clone the
repository and create an environment:

```powershell
git clone https://github.com/Devansh23007/F1-Prediction.git
cd F1-Prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the dashboard:

```powershell
streamlit run dashboard/app_v2.py
```

Run the tests:

```powershell
python -m pytest tests/ -v
```

To rebuild the processed dataset and generate the walk-forward backtest
reports, run:

```powershell
python src/backtest_v2.py
```

The backtest trains on earlier races and predicts later races. Its generated
files are written to `reports/`. Re-run it after changing the data or model;
reported results are specific to the dataset and model version used.

## How predictions are made

The race model estimates how far each driver may finish from their grid
position. A LightGBM model and a regularized linear model estimate this
correction, while a separate logistic regression estimates retirement
likelihood. A Monte Carlo simulation uses those estimates to calculate win,
podium, and top-ten probabilities.

The training features include grid and qualifying information, prior driver
and team form, Elo ratings, previous circuit results, practice pace, and sprint
results that are available before the Grand Prix. Rolling form, Elo ratings,
and circuit history are calculated from earlier races.

Race weather in the source data is measured during the race, so the model does
not use those observations as pre-race inputs. Historical pre-race weather
forecasts are not currently collected.

## Backtest results

The latest walk-forward holdout covers 65 races from 2024 to 2026. Each race
is predicted using earlier races only.

| Model | Rank MAE (lower is better) | Spearman correlation (higher is better) | Winner hit rate | Mean top-three drivers matched |
| --- | ---: | ---: | ---: | ---: |
| Grid order | 3.219 | 0.679 | 60.0% | 2.062 of 3 |
| Combined Elo | 4.019 | 0.570 | 29.2% | 1.492 of 3 |
| Grid plus learned correction | 3.111 | 0.706 | 58.5% | 2.123 of 3 |

The model improves rank error and rank correlation over the grid baseline in
this evaluation, but it does not improve the winner hit rate. The bootstrap
interval for rank-error improvement just touches zero (95% CI -0.007 to +0.221),
while the rank-correlation improvement is statistically significant. The model's podium Brier score
(0.0638) is slightly worse than the grid baseline (0.0631); its win Brier
scores are nearly equal (0.0292 for the model and 0.0292 for the grid).

See `reports/backtest_summary_holdout.csv`, `reports/backtest_per_race.csv`,
and `reports/backtest_meta.json` for more detail.

The holdout period is useful for comparison, but model development and repeated
experimentation can influence future choices. Treat the results as an
evaluation snapshot rather than a guarantee of future performance.

## Keeping the model up to date

The model only knows about races stored in `data/raw`. After each Grand Prix, run:

```powershell
python src/update_data.py             # fetch new races, rebuild features
python src/update_data.py --backtest  # same, plus refresh reports/
```

This is incremental (only missing rounds are downloaded). The dashboard and live
predictions retrain from `data/processed/model_dataset_v2.csv` every time, so they
use the new races immediately; restart Streamlit if it is open. A GitHub Action
(`.github/workflows/update_data.yml`) can do this every Monday automatically.

## Dashboard

The Streamlit app includes:

- Upcoming-race predictions and a local prediction log.
- A Monte Carlo driver and constructor championship simulator.
- Driver and constructor head-to-head comparisons.
- Circuit profiles and historical race information.
- A race explorer with feature-attribution charts.
- A podium-prediction game and leaderboard.
- Model reports and interactive what-if scenarios.

The live race calendar and qualifying features use external services when
available. Some dashboard features require an internet connection.

## Data and project layout

The project uses historical race results and qualifying data, FastF1 practice
and session data, and locally generated feature and report files. The
dashboard's prediction log is stored in `data/predictions_log.db`.

```text
dashboard/
  app_v2.py                  Streamlit dashboard
src/
  collect_data.py            Race and qualifying data collection
  collect_practice_pace.py   Practice pace collection
  collect_sprints.py         Sprint results collection
  collect_weather.py         Historical race weather collection
  features_v2.py             Feature engineering
  model_v2.py                Race and retirement models
  live_predictions.py        Upcoming-race predictions and logging
  backtest_v2.py             Walk-forward evaluation
  update_data.py             One-command data refresh
  championship_sim.py        Championship simulation
  h2h_and_circuits.py        Head-to-head and circuit analysis
  tuning_and_ablation.py     Model tuning and feature ablation
tests/
  test_leakage.py
  test_live_and_sim.py
  test_model_invariance.py
reports/                     Generated evaluation results
data/raw/                    Source data
data/processed/              Engineered model dataset
```

## Limitations

Formula 1 results are uncertain, and a model trained on past seasons may not
capture new regulations, car development, incidents, or changing driver
performance. Practice data can also be incomplete. Compare model predictions
with simple baselines and review the evaluation reports before drawing
conclusions.

This project is for educational and research use. It is not intended for
gambling or financial decisions. Formula 1 and F1 are trademarks of Formula One
Licensing B.V.
