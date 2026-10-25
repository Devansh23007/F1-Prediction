# 🏎️ Formula 1 Race Predictor & Championship Analytics Engine

[![CI](https://github.com/YOUR_GITHUB_REPOSITORY_URL/actions/workflows/pipeline.yml/badge.svg)](https://github.com/YOUR_GITHUB_REPOSITORY_URL/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.50+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Formula 1 race prediction, Monte Carlo championship simulation, and interpretability platform. Spanning the entire Turbo-Hybrid era (**2014–2026**, 269 Grands Prix, 5,479 driver records), this system predicts race finishing positions, podium/win probabilities, and retirement likelihood using leak-free historical signals, FastF1 practice pace, track chaos indices, and official Sprint features.

Every prediction is validated through rigorous **walk-forward backtesting**, cluster-robust bootstrap statistical significance testing, and leave-one-group-out feature ablation.

---

## ⚡ Quick Start

```powershell
# 1. Clone repository & set up environment
git clone YOUR_GITHUB_REPOSITORY_URL
cd "F1 Race Predictor"
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (includes LightGBM, FastF1, Plotly, Streamlit, Pytest)
pip install -r requirements.txt

# 3. Run full walk-forward backtest & statistical calibration
python src/backtest_v2.py

# 4. Launch the Interactive 8-Page Dashboard
streamlit run dashboard/app_v2.py
```

To run the automated leakage prevention and model invariant test suite:
```powershell
pytest tests/ -v
```

---

## 📊 Backtest Performance vs Baselines

Modern Formula 1 starting grids are overwhelmingly informative. Unlike naive models that claim unrealistic accuracy, our system explicitly benchmarks against the **Starting Grid baseline** and a **Combined Elo baseline** across a **65-race holdout set (2024–2026)** that was strictly held out during tuning:

### Holdout Evaluation (65 Races: 2024, 2025, 2026)

| Model / Baseline | Rank MAE ↓ | Spearman Rank Corr. (ρ) ↑ | Winner Hit Rate | Top-3 Hits (of 3) ↑ | Exact Podiums ↑ | Top-10 Hits (of 10) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grid Order Baseline** | 3.219 | 0.679 | 60.0% | 2.062 | 17 | 7.908 |
| **Combined Elo Baseline** | 4.019 | 0.570 | 29.2% | 1.492 | 4 | 7.369 |
| **Grid + Learned Correction (Our Model)** | **3.124** | **0.706** | 56.9% | **2.077** | **19** | **7.923** |

### Statistical Significance (2,000 Cluster-Robust Bootstrap Resamples)
- **Spearman Correlation Gain:** $+0.027$ ($95\%$ CI: $[+0.008, +0.045]$, **$p = 0.0025$**) — *statistically significant*.
- **Rank MAE Advantage:** $+0.095$ spots gained per driver ($p = 0.054$).
- **Exact Podiums Edge:** $+2.0$ full podium locks ($19$ vs $17$).

### 2026 New Regulations Holdout (17 Races)
Under the 2026 aerodynamic and power unit regulation reset:
- **Rank MAE:** **3.412** (Model) vs 3.497 (Grid)
- **Spearman Correlation:** **0.681** (Model) vs 0.647 (Grid)
- **Exact Podiums:** **5** (Model) vs 4 (Grid)

---

## 🔬 Core Modeling Innovations

1. **Grid Correction Formulation:** Rather than predicting raw finishes from scratch, the model learns the non-linear position delta $\Delta = \text{Finish} - \text{Grid}$ via an ensembled blend of LightGBM (`num_leaves=8`, `min_child_samples=30`, `learning_rate=0.03`) and $L_2$-regularized Ridge regression (`\alpha=250`), weighted by an optimal shrinkage coefficient ($\gamma = 0.5$).
2. **Dedicated Retirement (DNF) Classifier:** Replaced heuristic weights with a regularized Logistic Regression pipeline (`C=0.1`) trained on track chaos, historical driver/team DNF rates, ambient temperatures, and practice lap counts. ROC-AUC: **0.625** (vs 0.554 baseline).
3. **Plackett-Luce Simulation with Track Chaos Temperature:** Win and podium odds are evaluated via 12,000 Monte Carlo draws using the Gumbel-Max trick. The simulation temperature dynamically scales per track:
   $$T_{\text{circuit}} = T_0 \times (1 + 0.25 \times z_{\text{chaos}})$$
   Lowering chaos on processional circuits (Monaco, Hungaroring) and widening variance on chaotic circuits (Baku, Jeddah, Interlagos).
4. **Leak-Free Rolling Feature Windows:** Every rolling signal (`drv_fin_l3`, `drv_pts_l5`, `drv_gain_l5`, `_wet_gain`, `high_deg_gain`) applies `.shift(1)` across historical rows to guarantee absolute zero lookahead bias.
5. **Regulation Reset Elo Decay:** Team Elo ratings regress by $55\%$ toward baseline across major aerodynamic/power unit resets (2017, 2022, 2026) to mirror car development shakeups.

---

## 🖥️ Interactive Dashboard Features (`dashboard/app_v2.py`)

The redesigned Streamlit application includes 8 modular analytical pages:

1. **⚡ Live Next Race & Prediction Log:**
   - Automatically detects upcoming Grand Prix rounds via the Jolpica Ergast API (or allows manual round selection).
   - 1-click **"🔄 Fetch Official Qualifying"** with FastF1 fallback.
   - Interactive starting grid editor with persistent state (maintains qualifying positions across runs).
   - Generates predicted finishing positions, expected finishes, win probabilities ($P_{\text{win}}$), and podium odds ($P_{\text{podium}}$).
   - Logs predictions with timestamps and audit columns to SQLite (`data/predictions_log.db`).
   - 1-click CSV & JSON export buttons.

2. **🏆 2026 Championship Monte Carlo Simulator:**
   - Real-time driver and constructor standings directly synced from Jolpica.
   - Simulates remaining calendar races (including Sprint weekends) over 5,000 to 20,000 iterations using official FIA points tables (GP: 25-18-15-12-10-8-6-4-2-1, Sprint: 8-7-6-5-4-3-2-1).
   - Calculates title win probabilities ($P_{\text{champion}}$), podium probabilities, expected points, and min/max ranges for both WDC and WCC.

3. **⚔️ Head-to-Head Rivalry & Elo Evolution:**
   - Compare any two drivers or constructors from 2014 to 2026.
   - Metrics: shared races, race finish battle, qualifying duel, wins, and podiums.
   - Detects teammate pairings and displays shared race counts.
   - Dual-line interactive Plotly timeline tracking dynamic Elo skill trajectories over time.

4. **🗺️ Circuit Profiles & Chaos Index:**
   - Database of all 33 modern Formula 1 circuits.
   - Metrics: empirical DNF rate, chaos index $z$-score, pit stop time loss (seconds), tyre degradation category, pole-to-win conversion rate, and overtaking volatility.
   - Interactive bubble chart highlighting selected circuit vs the entire calendar.
   - Granular historical race winner and pole sitter logs.

5. **🏁 Race Explorer & SHAP Factor Attribution:**
   - Explore any historical Grand Prix from 2014 to 2026.
   - Diverging **TreeSHAP** bar chart explaining *why* the model predicted a driver to advance or drop from their grid position.
   - CSV export for individual Grand Prix predictions.

6. **🎯 Check Yourself (Game & Persistent Leaderboard):**
   - Test your racing intuition against the AI model and the Grid Baseline.
   - Enter your Racer Tag and lock in your podium picks.
   - Persistent SQLite Leaderboard table (`leaderboard` and `game_rounds` in `data/predictions_log.db`) tracking all-time scores, average points, and win rates against AI.

7. **📊 Model Report:**
   - Full holdout and walk-forward summary tables.
   - Leave-One-Group-Out Feature Ablation Study table (`reports/ablation_study.csv`).
   - 2,000-resample Cluster-Robust Bootstrap Significance cards (`reports/bootstrap_significance.json`).
   - Reliability curves and Brier score breakdowns.
   - 1-click CSV export buttons for all reports and backtest tables.

8. **🔮 What-if Lab:**
   - Interactive scenario laboratory: reorder the starting grid, apply hypothetical grid penalties, simulate pit-lane starts, or ramp up race chaos to observe real-time probability shifts.

---

## 🧪 Automated Testing & CI/CD

The repository contains an automated unit and regression test suite run via `pytest`:

```powershell
pytest tests/ -v
```

### Test Coverage Highlights (`tests/`):
- `tests/test_leakage.py`: Confirms driver debut races contain zero prior information; confirms shifted rolling form never uses current race results; verifies Elo ratings are strictly pre-race.
- `tests/test_model_invariance.py`: Proves Monte Carlo probabilities sum to targets ($1.0$ for wins, $3.0$ for podiums); verifies DNF probabilities are bounded in $[0.0, 1.0]$; verifies scoring monotonicity and finite TreeSHAP explanations.
- `tests/test_live_and_sim.py`: Verifies SQLite schema creation; tests calendar lookup; ensures Championship simulator probabilities sum to $100\%$; validates leaderboard match recording.
- `.github/workflows/pipeline.yml`: Continuous Integration workflow running automated tests and validating dashboard compilation on every push and pull request.

---

## 📁 Repository Structure

```text
F1 Race Predictor/
├── .github/workflows/
│   └── pipeline.yml              # GitHub Actions CI workflow
├── .streamlit/
│   └── config.toml               # Streamlit theme & performance configuration
├── data/
│   ├── raw/                      # Ergast/Jolpica race results, weather, FastF1 pace, sprints
│   ├── processed/
│   │   └── model_dataset_v2.csv  # 5,479 rows x 86 columns (2014-2026)
│   └── predictions_log.db        # SQLite database (live predictions & leaderboard)
├── src/
│   ├── collect_data.py           # Ergast/Jolpica data collector (2014-2026)
│   ├── collect_practice_pace.py  # FastF1 long-run representative pace harvester
│   ├── collect_weather.py        # Ambient/track temperature & rain collector
│   ├── collect_sprints.py        # Saturday sprint race harvester
│   ├── features_v2.py            # Feature pipeline with Elo, pit loss & chaos
│   ├── model_v2.py               # LightGBM + Ridge + DNF Classifier + TreeSHAP
│   ├── backtest_v2.py            # Walk-forward backtest & bootstrap significance
│   ├── live_predictions.py       # Next-race discovery, qualifying fetcher & SQLite log
│   ├── championship_sim.py       # Monte Carlo WDC & WCC simulator (10,000+ draws)
│   ├── leaderboard.py            # Persistent SQLite leaderboard manager
│   ├── h2h_and_circuits.py       # Driver/team head-to-head & 33-circuit analytics
│   └── tuning_and_ablation.py    # Cross-validation hyperparameter search & ablation
├── dashboard/
│   └── app_v2.py                 # Interactive 8-page Streamlit application
├── reports/                      # Backtest predictions, ablation study, bootstrap metrics
├── tests/                        # 12 automated pytest unit and invariance tests
├── requirements.txt              # Production Python dependencies
└── README.md
```

---

## ⚖️ Disclaimer

This software is for educational, research, and portfolio purposes only. Formula 1 is inherently uncertain; predictions represent probabilistic estimates and must not be used for commercial gambling or financial wagering. Formula 1, F1, and related marks are trademarks of Formula One Licensing B.V.
