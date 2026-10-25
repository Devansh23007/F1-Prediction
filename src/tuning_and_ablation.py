"""Systematic Time-Based Hyperparameter Search & Feature Group Ablation Study.

1. Tunes hyperparameters using walk-forward CV strictly on 2022-2023.
2. Evaluates the feature group ablation study across the 2024-2026 holdout.
Saves results to reports/ablation_study.csv and reports/cv_hyperparameters.json.
"""
import itertools
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_dataset_v2.csv"
REPORTS.mkdir(exist_ok=True)

from features_v2 import FEATURES
import model_v2 as M


def evaluate_run(train, test, small_feats, full_feats, lgb_params, ridge_alpha, shrink):
    """Train on train, predict test, compute Rank MAE & Spearman."""
    y_train = train["finish_position"] - train["grid_position"]
    
    m1 = lgb.LGBMRegressor(**lgb_params, random_state=1, verbose=-1)
    m1.fit(train[small_feats].astype(float), y_train)

    m2 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ridge_alpha))
    m2.fit(train[full_feats].astype(float), y_train)

    delta = shrink * (m1.predict(test[small_feats].astype(float)) + m2.predict(test[full_feats].astype(float)))
    test_scores = test["grid_position"].values + delta
    
    # Evaluate per race
    test_copy = test.copy()
    test_copy["score"] = test_scores
    
    maes, sps, pod_hits = [], [], []
    for _, g in test_copy.groupby("race_id"):
        r = g["score"].rank(method="first")
        act = g["finish_position"]
        maes.append((r - act).abs().mean())
        sps.append(spearmanr(r, act)[0])
        a3 = set(g.loc[act <= 3, "driver_id"])
        p3 = set(g.loc[r <= 3, "driver_id"])
        pod_hits.append(int(p3 == a3))

    return np.mean(maes), np.mean(sps), np.sum(pod_hits)


def time_based_cv(df):
    """Walk-forward hyperparameter optimization on 2022-2023 ONLY."""
    tune_df = df[df["season"].isin([2022, 2023])].copy()
    tune_rids = tune_df["race_idx"].unique()
    
    # Split into 3 progressive time folds across 2022-2023
    val_rids = tune_rids[len(tune_rids)//3:]
    
    param_grid = [
        {"num_leaves": 6, "min_child_samples": 40, "learning_rate": 0.03, "reg_lambda": 20, "alpha": 300, "shrink": 0.5},
        {"num_leaves": 8, "min_child_samples": 30, "learning_rate": 0.03, "reg_lambda": 15, "alpha": 250, "shrink": 0.5},
        {"num_leaves": 5, "min_child_samples": 50, "learning_rate": 0.04, "reg_lambda": 30, "alpha": 350, "shrink": 0.45},
        {"num_leaves": 10, "min_child_samples": 30, "learning_rate": 0.02, "reg_lambda": 20, "alpha": 250, "shrink": 0.55},
        {"num_leaves": 6, "min_child_samples": 40, "learning_rate": 0.02, "reg_lambda": 25, "alpha": 300, "shrink": 0.5},
    ]

    print(f"Running Time-Based CV on 2022-2023 across {len(param_grid)} parameter sets...")
    best_score = float("inf")
    best_params = None

    for i, p in enumerate(param_grid):
        lgb_p = {
            "n_estimators": 150, "num_leaves": p["num_leaves"],
            "min_child_samples": p["min_child_samples"], "learning_rate": p["learning_rate"],
            "reg_lambda": p["reg_lambda"], "subsample": 0.8, "colsample_bytree": 0.7,
        }
        
        fold_maes = []
        for rid in val_rids[::3]:  # sample representative test rounds in tuning seasons
            train = df[df["race_idx"] < rid]
            test = df[df["race_idx"] == rid]
            mae, sp, _ = evaluate_run(train, test, M.SMALL, FEATURES, lgb_p, p["alpha"], p["shrink"])
            fold_maes.append(mae)

        avg_mae = np.mean(fold_maes)
        print(f"  Config {i+1}: leaves={p['num_leaves']}, lr={p['learning_rate']}, alpha={p['alpha']}, shrink={p['shrink']} -> Val MAE: {avg_mae:.3f}")
        if avg_mae < best_score:
            best_score = avg_mae
            best_params = p

    print(f"\nBest 2022-2023 Hyperparameters: {best_params} (Val MAE: {best_score:.3f})\n")
    return best_params


def run_ablation_study(df, best_params):
    """Run ablation by sequentially removing each feature group on 2024-2026 holdout."""
    hold_rids = df[df["season"] >= 2024]["race_idx"].unique()
    
    lgb_p = {
        "n_estimators": 150, "num_leaves": best_params["num_leaves"],
        "min_child_samples": best_params["min_child_samples"], "learning_rate": best_params["learning_rate"],
        "reg_lambda": best_params["reg_lambda"], "subsample": 0.8, "colsample_bytree": 0.7,
    }
    alpha = best_params["alpha"]
    shrink = best_params["shrink"]

    feature_groups = {
        "Full Production Model (All Features)": (None, None),
        "Without FastF1 Practice Pace": (
            [c for c in M.SMALL if not c.startswith("fp_")],
            [c for c in FEATURES if not c.startswith("fp_")]
        ),
        "Without Weather & Conditions": (
            [c for c in M.SMALL if c not in ("rainfall", "track_temp", "drv_wet_skill_active")],
            [c for c in FEATURES if c not in ("rainfall", "track_temp", "air_temp", "humidity", "drv_wet_vs_dry_gain", "drv_wet_skill_active")]
        ),
        "Without Tyre & Strategy Signals": (
            [c for c in M.SMALL if c not in ("pit_loss_time", "fast_quali_tyre_wear_penalty", "drv_tyre_mgmt_skill")],
            [c for c in FEATURES if c not in ("pit_loss_time", "pit_loss_scaled", "fast_quali_tyre_wear_penalty", "high_deg_circuit", "drv_tyre_mgmt_skill")]
        ),
        "Without Sprint Weekend Features": (
            [c for c in M.SMALL if c not in ("is_sprint_weekend", "sprint_vs_grid")],
            [c for c in FEATURES if c not in ("is_sprint_weekend", "sprint_finish", "sprint_points", "sprint_pos_gain", "sprint_vs_grid")]
        ),
        "Without Elo Ratings": (
            [c for c in M.SMALL if "elo" not in c],
            [c for c in FEATURES if "elo" not in c]
        ),
        "Without Rolling Form Metrics": (
            [c for c in M.SMALL if not any(x in c for x in ("_l3", "_l5", "_l10"))],
            [c for c in FEATURES if not any(x in c for x in ("_l3", "_l5", "_l10"))]
        ),
    }

    print("Running Feature Group Ablation Study across 65 Holdout Races (2024-2026)...")
    ablation_results = []
    baseline_mae = None
    baseline_sp = None

    for name, (sub_small, sub_full) in feature_groups.items():
        s_feats = sub_small if sub_small is not None else M.SMALL
        f_feats = sub_full if sub_full is not None else FEATURES

        maes, sps, exact_pods = [], [], 0
        for rid in hold_rids:
            train = df[df["race_idx"] < rid]
            test = df[df["race_idx"] == rid]
            mae, sp, exact = evaluate_run(train, test, s_feats, f_feats, lgb_p, alpha, shrink)
            maes.append(mae)
            sps.append(sp)
            exact_pods += exact

        mean_mae = np.mean(maes)
        mean_sp = np.mean(sps)
        
        if baseline_mae is None:
            baseline_mae = mean_mae
            baseline_sp = mean_sp
            delta_mae = 0.0
            delta_sp = 0.0
        else:
            delta_mae = mean_mae - baseline_mae  # Positive = error worsened without feature
            delta_sp = baseline_sp - mean_sp    # Positive = correlation dropped without feature

        ablation_results.append({
            "feature_group": name,
            "rank_mae": round(mean_mae, 3),
            "mae_degradation": round(delta_mae, 3),
            "spearman_corr": round(mean_sp, 3),
            "spearman_drop": round(delta_sp, 3),
            "exact_podiums": int(exact_pods),
        })
        print(f"  {name:42s} | MAE: {mean_mae:.3f} (+{delta_mae:+.3f}) | Spearman: {mean_sp:.3f} (-{delta_sp:+.3f}) | Exact Podiums: {exact_pods}")

    # Add Starting Grid baseline
    grid_maes, grid_sps, grid_exact = [], [], 0
    for rid in hold_rids:
        test = df[df["race_idx"] == rid]
        r_g = test["grid_position"].rank(method="first")
        act = test["finish_position"]
        grid_maes.append((r_g - act).abs().mean())
        grid_sps.append(spearmanr(r_g, act)[0])
        a3 = set(test.loc[act <= 3, "driver_id"])
        p3_g = set(test.loc[r_g <= 3, "driver_id"])
        grid_exact += int(p3_g == a3)

    ablation_results.append({
        "feature_group": "Starting Grid Baseline (No Model)",
        "rank_mae": round(np.mean(grid_maes), 3),
        "mae_degradation": round(np.mean(grid_maes) - baseline_mae, 3),
        "spearman_corr": round(np.mean(grid_sps), 3),
        "spearman_drop": round(baseline_sp - np.mean(grid_sps), 3),
        "exact_podiums": int(grid_exact),
    })

    res_df = pd.DataFrame(ablation_results)
    out_csv = REPORTS / "ablation_study.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nAblation study saved to {out_csv}")
    return res_df


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run features_v2.py first.")
    
    df = pd.read_csv(DATA_PATH)
    best_params = time_based_cv(df)
    json.dump(best_params, open(REPORTS / "cv_hyperparameters.json", "w"), indent=2)

    ablation_df = run_ablation_study(df, best_params)
    print("\n" + "=" * 80)
    print("FEATURE ABLATION STUDY RESULTS (HOLD-OUT 2024-2026, 65 RACES)")
    print("=" * 80)
    print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    main()
