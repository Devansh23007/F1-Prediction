"""Walk-forward backtest: every race is predicted by a model trained only
on earlier races, and compared with simple baselines."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import model_v2 as M
from features_v2 import build

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

FIRST_EVAL_SEASON = 2022
TUNE_SEASONS = (2022, 2023)      # used only to choose the simulation temperature
HOLDOUT_SEASONS = (2024, 2025, 2026)   # headline numbers

MODELS = {"Grid order (baseline)": "score_grid",
          "Combined Elo (baseline)": "score_elo",
          "Grid + learned correction": "score_model"}


def walk_forward(df):
    df = df.sort_values(["race_idx", "driver_id"]).reset_index(drop=True)
    out = []
    for rid in df.loc[df.season >= FIRST_EVAL_SEASON, "race_idx"].unique():
        train, test = df[df.race_idx < rid], df[df.race_idx == rid].copy()
        models = M.fit(train)
        test["score_model"] = M.score(models, test)
        test["score_grid"] = test["grid_position"]
        test["score_elo"] = -test["combined_elo"]
        test["p_dnf"] = M.dnf_prob(models, test)
        out.append(test)
    return pd.concat(out, ignore_index=True)


def race_metrics(g, col):
    r = g[col].rank(method="first")
    a3 = set(g.loc[g.finish_position <= 3, "driver_id"])
    p3 = set(g.loc[r <= 3, "driver_id"])
    a10 = set(g.loc[g.finish_position <= 10, "driver_id"])
    p10 = set(g.loc[r <= 10, "driver_id"])
    win = g.loc[g.finish_position == 1, "driver_id"]
    return {"rank_mae": (r - g.finish_position).abs().mean(),
            "spearman": spearmanr(r, g.finish_position)[0],
            "winner_hit": int(len(win) > 0 and g.loc[r == 1, "driver_id"].iloc[0] == win.iloc[0]),
            "top3_hits": len(p3 & a3), "exact_podium": int(p3 == a3),
            "top10_hits": len(p10 & a10)}


def summarise(pred, seasons, models=MODELS):
    sub, rows = pred[pred.season.isin(seasons)], []
    for name, col in models.items():
        ms = pd.DataFrame([race_metrics(g, col) for _, g in sub.groupby("race_id")])
        m = ms.mean(); m["exact_podium"] = ms.exact_podium.sum()
        m["races"] = len(ms); m["model"] = name
        rows.append(m)
    return pd.DataFrame(rows).set_index("model")


def get_circuit_temps(df, base_temp):
    """Compute per-circuit simulation temperature scaled by track chaos and safety car likelihood."""
    chaos_per_race = df.groupby("race_id")["circuit_chaos"].first().fillna(4.1)
    mean_c = chaos_per_race.mean()
    std_c = chaos_per_race.std()
    z = ((chaos_per_race - mean_c) / (std_c if std_c > 0 else 1.0)).clip(-1.5, 1.5)
    return (base_temp * (1.0 + 0.25 * z)).clip(0.6, 2.5).to_dict()


def probs_for(pred, col, temp, n=4000):
    cols = {k: pd.Series(index=pred.index, dtype=float)
            for k in ["p_win", "p_podium", "p_top10", "exp_position"]}
    is_map = isinstance(temp, dict)
    for rid, g in pred.groupby("race_id"):
        T = float(temp[rid]) if is_map else float(temp)
        P = M.simulate(g[col].values, g["p_dnf"].values, T, n=n)
        for k, v in M.summarise_probs(P).items():
            cols[k].loc[g.index] = v
    return pd.DataFrame(cols)


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def main():
    df = build(save=True)
    pred = walk_forward(df)

    tune = pred[pred.season.isin(TUNE_SEASONS)]
    grid = [0.6, 0.9, 1.2, 1.6, 2.0, 2.6, 3.2]
    def loss(col, T):
        T_map = get_circuit_temps(tune, T)
        p = probs_for(tune, col, T_map, n=2500)
        return brier(p.p_podium, tune.podium) + brier(p.p_win, tune.win)
    T_model = min(grid, key=lambda T: loss("score_model", T))
    T_grid = min(grid, key=lambda T: loss("score_grid", T))
    print(f"base temperatures  model={T_model}  grid={T_grid}")

    T_map_m = get_circuit_temps(pred, T_model)
    T_map_g = get_circuit_temps(pred, T_grid)
    pm = probs_for(pred, "score_model", T_map_m)
    pg = probs_for(pred, "score_grid", T_map_g)
    for c in pm: pred[c] = pm[c]
    for c in pg: pred[c + "_grid"] = pg[c]
    pred["pred_order"] = pred.groupby("race_id")["score_model"].rank(method="first").astype(int)

    hold = pred[pred.season.isin(HOLDOUT_SEASONS)]
    prob = {
        "podium_brier_model": brier(hold.p_podium, hold.podium),
        "podium_brier_grid_baseline": brier(hold.p_podium_grid, hold.podium),
        "podium_brier_constant": brier(np.full(len(hold), hold.podium.mean()), hold.podium),
        "win_brier_model": brier(hold.p_win, hold.win),
        "win_brier_grid_baseline": brier(hold.p_win_grid, hold.win),
        "win_brier_constant": brier(np.full(len(hold), hold.win.mean()), hold.win),
    }
    s_hold = summarise(pred, HOLDOUT_SEASONS)
    s_2026 = summarise(pred, [2026]) if 2026 in pred.season.values else None
    s_all = summarise(pred, sorted(pred.season.unique()))
    print(f"\nHOLD-OUT {HOLDOUT_SEASONS}\n", s_hold.round(3).to_string())
    if s_2026 is not None:
        print("\n2026 NEW REGULATIONS ONLY\n", s_2026.round(3).to_string())
    print(f"\nALL walk-forward {sorted(pred.season.unique())}\n", s_all.round(3).to_string())
    print("\n", json.dumps(prob, indent=2))

    d = hold[["p_podium", "podium"]].copy()
    d["bin"] = pd.cut(d.p_podium, np.linspace(0, 1, 11), include_lowest=True)
    calib = d.groupby("bin", observed=True).agg(
        predicted=("p_podium", "mean"), actual=("podium", "mean"), n=("podium", "size"))
    print("\nCalibration\n", calib.round(3).to_string())

    keep = ["race_id", "season", "round", "race_name", "race_date", "driver_id",
            "driver_name", "driver_code", "constructor_name", "grid_position",
            "qualifying_position", "finish_position", "podium", "win", "status",
            "score_model", "pred_order", "p_win", "p_podium", "p_top10",
            "exp_position", "p_win_grid", "p_podium_grid", "driver_elo", "team_elo"]
    pred[keep].to_csv(REPORTS / "backtest_predictions.csv", index=False)
    s_hold.reset_index().to_csv(REPORTS / "backtest_summary_holdout.csv", index=False)
    s_all.reset_index().to_csv(REPORTS / "backtest_summary_all.csv", index=False)
    calib.reset_index(drop=True).to_csv(REPORTS / "calibration_podium.csv", index=False)
    rows = []
    for name, col in MODELS.items():
        for rid, g in pred.groupby("race_id"):
            m = race_metrics(g, col)
            m.update(model=name, race_id=rid, season=int(g.season.iloc[0]),
                     race_name=g.race_name.iloc[0])
            rows.append(m)
    pd.DataFrame(rows).to_csv(REPORTS / "backtest_per_race.csv", index=False)
    json.dump({"temperature_model": T_model, "temperature_grid": T_grid,
               "tune_seasons": TUNE_SEASONS, "holdout_seasons": HOLDOUT_SEASONS,
               "prob_metrics": prob}, open(REPORTS / "backtest_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
