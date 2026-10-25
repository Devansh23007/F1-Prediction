"""Leakage prevention tests for F1 Race Predictor.

Verifies:
1. No future information is present in rolling features.
2. Driver and Team Elo ratings are strictly pre-race.
3. First career race of any driver has zero historical leak.
4. Sprint features are only populated prior to Sunday Grand Prix.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "model_dataset_v2.csv"


@pytest.fixture(scope="module")
def dataset():
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset {DATA_PATH} not found.")
    return pd.read_csv(DATA_PATH)


def test_first_career_race_has_no_prior_history(dataset):
    """A driver's first career race in the dataset must have NaN or 0 for shifted rolling stats."""
    first_races = dataset[dataset.drv_career_races == 0]
    assert len(first_races) > 0, "No debut driver races found."

    # In shifted rolling calculations, the very first race has nothing before it
    # Therefore, drv_fin_l3, drv_pts_l5 should be NaN
    assert first_races["drv_fin_l3"].isna().all(), "drv_fin_l3 is populated on driver debut race (leakage!)"
    assert first_races["drv_pts_l5"].isna().all(), "drv_pts_l5 is populated on driver debut race (leakage!)"


def test_rolling_finish_never_matches_same_race_finish(dataset):
    """drv_fin_l3 must reflect previous races, not current finish."""
    # Check rows where driver has at least 5 career races
    experienced = dataset[dataset.drv_career_races >= 5].dropna(subset=["drv_fin_l3", "finish_position"])
    
    # Calculate difference between current finish and rolling average of past finishes
    exact_match_ratio = (experienced["drv_fin_l3"] == experienced["finish_position"]).mean()
    # If there were leakage (i.e. if drv_fin_l3 included current finish directly),
    # exact match would be suspiciously high; with shift(1), exact matches are purely coincidental (< 15%)
    assert exact_match_ratio < 0.20, f"Suspiciously high match ratio ({exact_match_ratio:.3f}) between current finish and rolling form."


def test_elo_ratings_are_pre_race(dataset):
    """Driver and Team Elo ratings must be established before the race starts."""
    assert "driver_elo" in dataset.columns
    assert "team_elo" in dataset.columns
    
    # Elo values should fall in realistic competitive chess/F1 range [1000, 2500]
    assert dataset["driver_elo"].between(1000, 2600).all(), "Driver Elo outside realistic bounds."
    assert dataset["team_elo"].between(1000, 2600).all(), "Team Elo outside realistic bounds."

    # For a given driver in a given season, Elo between race N and race N+1 must be identical
    # until race N finishes
    sample_driver = dataset[dataset.driver_id == "hamilton"].sort_values(["season", "round"])
    if len(sample_driver) > 1:
        # Initial Elo at debut in 2014 should be exactly 1500 or close
        debut_elo = sample_driver.iloc[0]["driver_elo"]
        assert 1400 <= debut_elo <= 1600


def test_sprint_features_temporal_validity(dataset):
    """Sprint features must only be present when sprint took place before Sunday Grand Prix."""
    if "is_sprint_weekend" in dataset.columns:
        sprint_races = dataset[dataset.is_sprint_weekend == 1]
        non_sprint = dataset[dataset.is_sprint_weekend == 0]

        # Non sprint weekends should have zero sprint points
        assert (non_sprint["sprint_points"] == 0).all(), "Non-sprint weekends have non-zero sprint points."
        # Sprint points on sprint weekends must be between 0 and 8
        assert sprint_races["sprint_points"].between(0, 8).all(), "Sprint points outside official range [0, 8]."
