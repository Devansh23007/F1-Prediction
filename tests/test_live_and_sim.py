"""Tests for Live Predictions, Championship Simulator, and Leaderboard.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import live_predictions as LP
import championship_sim as CS
import leaderboard as LB
import h2h_and_circuits as HC
import pandas as pd


def test_sqlite_db_init_and_seed():
    """Ensure SQLite database initializes and creates required tables."""
    LP.init_db()
    with LP.get_db_connection() as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "predictions" in tables
    assert "prediction_runs" in tables


def test_upcoming_round_resolution():
    """Resolves upcoming round on the 2026 calendar."""
    upcoming = LP.get_upcoming_round(2026)
    assert "season" in upcoming
    assert "round" in upcoming
    assert upcoming["season"] == 2026
    assert upcoming["round"] >= 1


def test_championship_monte_carlo_sums():
    """Championship simulation probabilities should be valid and sum to ~100%."""
    wdc, wcc = CS.simulate_championship(n_sims=500, chaos_multiplier=1.0)
    assert not wdc.empty
    assert not wcc.empty

    # WDC championship probability should sum to ~100%
    assert np.isclose(wdc["p_champion"].sum(), 100.0, atol=2.0)
    # WCC championship probability should sum to ~100%
    assert np.isclose(wcc["p_champion"].sum(), 100.0, atol=2.0)


def test_leaderboard_recording():
    """Records a match and validates leaderboard aggregations."""
    LB.init_leaderboard()
    player = "PytestRacer"
    LB.record_round(
        player_name=player,
        race_id=9999,
        race_name="Test Grand Prix",
        user_picks=["Driver A", "Driver B", "Driver C"],
        actual_podium=["Driver A", "Driver B", "Driver C"],
        user_score=9,
        model_score=5,
        grid_score=3
    )

    df = LB.get_leaderboard_df()
    assert not df.empty
    p_row = df[df.player_name == player]
    assert not p_row.empty
    assert p_row["rounds_played"].iloc[0] >= 1
    assert p_row["total_user_score"].iloc[0] >= 9


def test_h2h_and_circuits_analytics():
    """Validates Head-to-Head analytics and Circuit profiles."""
    data_path = ROOT / "data" / "processed" / "model_dataset_v2.csv"
    if not data_path.exists():
        pytest.skip("Dataset not found")
    df = pd.read_csv(data_path)
    h2h = HC.get_driver_h2h(df, "Max Verstappen", "Lewis Hamilton")
    assert h2h["total_races"] > 100
    assert h2h["finishes_a_ahead"] > 0
    assert h2h["finishes_b_ahead"] > 0

    circuits = HC.get_all_circuits_table(df)
    assert len(circuits) >= 20
    assert "chaos_index" in circuits.columns
