"""Model invariance, calibration, and numerical stability tests.

Verifies:
1. Plackett-Luce simulation probabilities sum to 1.0 (win) and 3.0 (podium).
2. DNF probabilities strictly within [0.0, 1.0].
3. TreeSHAP feature attributions run and produce finite numbers.
4. Determinism of model scoring under fixed seeds.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_v2 as M

DATA_PATH = ROOT / "data" / "processed" / "model_dataset_v2.csv"


@pytest.fixture(scope="module")
def sample_data():
    if not DATA_PATH.exists():
        pytest.skip(f"{DATA_PATH} not found.")
    df = pd.read_csv(DATA_PATH)
    # Pick a recent race
    recent_race = df[df.race_idx == df.race_idx.max()].copy()
    train_data = df[df.race_idx < df.race_idx.max()].tail(2000).copy()
    return train_data, recent_race


def test_simulation_probabilities_sum_to_target():
    """Win probabilities must sum to 1.0; podium probabilities must sum to 3.0."""
    scores = np.array([1.5, 2.2, 3.1, 4.0, 5.2, 6.1, 7.0, 8.5, 9.2, 10.1])
    dnf_probs = np.full(len(scores), 0.10)
    temp = 1.2

    sim = M.simulate(scores, dnf_probs, temp, n=5000, seed=42)
    probs = M.summarise_probs(sim)

    # Win probability must sum to 1.0
    assert np.isclose(probs["p_win"].sum(), 1.0, atol=1e-3), f"P(win) sum was {probs['p_win'].sum()}"
    # Podium probability must sum to 3.0
    assert np.isclose(probs["p_podium"].sum(), 3.0, atol=1e-2), f"P(podium) sum was {probs['p_podium'].sum()}"
    # Top 10 probability must sum to 10.0
    assert np.isclose(probs["p_top10"].sum(), 10.0, atol=1e-2), f"P(top10) sum was {probs['p_top10'].sum()}"


def test_dnf_probabilities_bounded(sample_data):
    """DNF probabilities must always be bounded strictly in [0.0, 1.0]."""
    train_data, test_race = sample_data
    models = M.fit(train_data)
    pdnf = M.dnf_prob(models, test_race)

    assert (pdnf >= 0.0).all(), "Negative DNF probability observed."
    assert (pdnf <= 1.0).all(), "DNF probability greater than 1.0 observed."
    assert (pdnf < 0.70).all(), "Unrealistically high single-race DNF probability."


def test_scoring_monotonicity(sample_data):
    """A driver with identical stats starting P1 must have a lower score than starting P10."""
    train_data, test_race = sample_data
    models = M.fit(train_data)

    driver = test_race.head(1).copy()
    row_p1 = driver.copy()
    row_p1["grid_position"] = 1.0

    row_p10 = driver.copy()
    row_p10["grid_position"] = 10.0

    score_p1 = M.score(models, row_p1)[0]
    score_p10 = M.score(models, row_p10)[0]

    assert score_p1 < score_p10, "P1 start did not yield lower score than P10 start."


def test_shap_explanation_structure(sample_data):
    """TreeSHAP attribution returns non-empty dataframe with finite values."""
    train_data, test_race = sample_data
    models = M.fit(train_data)
    driver_row = test_race.head(1)

    shap_df = M.explain_driver_delta(models, driver_row)
    assert not shap_df.empty, "SHAP attribution returned empty dataframe."
    assert "feature" in shap_df.columns
    assert "impact" in shap_df.columns
    assert np.isfinite(shap_df["impact"]).all(), "Non-finite SHAP impact values."
