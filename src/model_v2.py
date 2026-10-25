"""Shared model code (used by backtest, training script and dashboard).

Design: the model does NOT predict the finishing order from scratch.  It
predicts the *correction* to the starting grid (finish - grid) and adds a
shrunk version of it to the grid.  A strong baseline (the grid) is
therefore the starting point, and the ML only has to learn what the grid
misses (car pace, race pace, reliability, overtaking difficulty, ...).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features_v2 import FEATURES

SMALL = [
    "grid_position", "quali_gap_pct", "quali_vs_teammate", "grid_penalty",
    "driver_vs_car_elo", "driver_vs_car_qelo", "team_elo", "team_qelo",
    "driver_elo", "drv_gain_l5", "drv_dnf_l10", "team_dnf_l10",
    "team_pts_l5", "circuit_overtaking", "circuit_chaos", "reached_q3", "pit_lane_start",
    "fp_longrun_gap_pct", "fp_race_vs_quali_pace", "fp_vs_teammate",
    "rainfall", "track_temp", "drv_wet_skill_active",
    "pit_loss_time", "fast_quali_tyre_wear_penalty", "drv_tyre_mgmt_skill",
    "is_sprint_weekend", "sprint_vs_grid",
    "reg_era", "seasons_since_rules_reset",
]
SHRINK = 0.5   # weight of the blended correction (0.5 each model => 1.0 total)


def _lgb():
    return lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.03, num_leaves=8, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=15,
        random_state=1, verbose=-1)


def _ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         Ridge(alpha=250))


DNF_FEATURES = [
    "grid_position", "circuit_dnf_rate", "circuit_chaos",
    "team_dnf_l10", "drv_dnf_l10", "track_temp", "rainfall", "fp_laps_count",
]


def fit(train: pd.DataFrame):
    y = train["finish_position"] - train["grid_position"]
    m1 = _lgb().fit(train[SMALL].astype(float), y)
    m2 = _ridge().fit(train[FEATURES].astype(float), y)
    m3 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(C=0.1, max_iter=200, random_state=1))
    m3.fit(train[DNF_FEATURES].astype(float), train["dnf"].astype(int))
    return {"lgb": m1, "ridge": m2, "dnf_clf": m3, "base_dnf": float(train["dnf"].mean())}


def predict_delta(models, test: pd.DataFrame):
    return SHRINK * (models["lgb"].predict(test[SMALL].astype(float))
                     + models["ridge"].predict(test[FEATURES].astype(float)))


def score(models, test: pd.DataFrame):
    """Lower score = expected to finish higher."""
    return test["grid_position"].values + predict_delta(models, test)


def dnf_prob(models_or_test, test_or_base=None, base: float = None):
    """Dedicated retirement classifier probability with empirical fallback."""
    if isinstance(models_or_test, dict):
        models = models_or_test
        test = test_or_base
        base_rate = models.get("base_dnf", 0.15)
        if "dnf_clf" in models:
            p_clf = models["dnf_clf"].predict_proba(test[DNF_FEATURES].astype(float))[:, 1]
            p = 0.85 * p_clf + 0.15 * base_rate
        else:
            p = (0.4 * test["drv_dnf_l10"].fillna(base_rate)
                 + 0.4 * test["team_dnf_l10"].fillna(base_rate) + 0.2 * base_rate)
    else:
        test = models_or_test
        base_rate = float(test_or_base if test_or_base is not None else (base if base is not None else 0.15))
        p = (0.4 * test["drv_dnf_l10"].fillna(base_rate)
             + 0.4 * test["team_dnf_l10"].fillna(base_rate) + 0.2 * base_rate)
    return np.asarray(p).clip(0.02, 0.60)


def simulate(scores, p_dnf, temp, n=10000, seed=0):
    """Plackett-Luce (Gumbel trick) + independent retirements.
    Returns array (k drivers, k positions) of probabilities."""
    rng = np.random.default_rng(seed)
    k = len(scores)
    u = -np.asarray(scores) / temp + rng.gumbel(size=(n, k))
    u = u - 1e6 * (rng.random((n, k)) < np.asarray(p_dnf)[None, :])
    order = np.argsort(-u, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(k)[None, :]
    return np.stack([(ranks == p).mean(0) for p in range(k)], axis=1)


def summarise_probs(P):
    pos = np.arange(1, P.shape[1] + 1)
    return {"p_win": P[:, 0], "p_podium": P[:, :3].sum(1),
            "p_top10": P[:, :10].sum(1), "exp_position": (P * pos).sum(1)}
