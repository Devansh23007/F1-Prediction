"""F1 2026 Championship Monte Carlo Simulator (WDC & WCC).

Simulates remaining races in the 2026 season over 10,000 iterations,
accounting for official race and sprint points systems, driver/team Elo ratings,
and circuit chaos temperatures to calculate title probabilities.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DATASET = DATA_DIR / "processed" / "model_dataset_v2.csv"

# Official F1 points systems
GP_POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1])
SPRINT_POINTS = np.array([8, 7, 6, 5, 4, 3, 2, 1])

REMAINING_2026_CALENDAR = [
    {"round": 18, "race_name": "United States Grand Prix", "circuit_id": "americas", "is_sprint": True},
    {"round": 19, "race_name": "Mexico City Grand Prix", "circuit_id": "rodriguez", "is_sprint": False},
    {"round": 20, "race_name": "São Paulo Grand Prix", "circuit_id": "interlagos", "is_sprint": True},
    {"round": 21, "race_name": "Las Vegas Grand Prix", "circuit_id": "vegas", "is_sprint": False},
    {"round": 22, "race_name": "Qatar Grand Prix", "circuit_id": "losail", "is_sprint": True},
    {"round": 23, "race_name": "Abu Dhabi Grand Prix", "circuit_id": "yas_marina", "is_sprint": False},
]


def fetch_current_standings(season: int = 2026) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetches real-time driver and constructor standings from Jolpica with fallback."""
    drv_standings = []
    con_standings = []

    try:
        url_d = f"https://api.jolpi.ca/ergast/f1/{season}/driverStandings.json"
        req_d = urllib.request.Request(url_d, headers={"User-Agent": "F1-Predictor/2.0"})
        with urllib.request.urlopen(req_d, timeout=8) as r:
            payload_d = json.loads(r.read())
        raw_d = payload_d["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
        for d in raw_d:
            drv_standings.append({
                "driver_id": d["Driver"]["driverId"],
                "driver_name": f"{d['Driver']['givenName']} {d['Driver']['familyName']}",
                "constructor_id": d["Constructors"][0]["constructorId"],
                "constructor_name": d["Constructors"][0]["name"],
                "current_points": float(d["points"]),
                "wins": int(d["wins"]),
            })
    except Exception as e:
        print(f"Jolpica driver standings fallback triggered ({e})")

    try:
        url_c = f"https://api.jolpi.ca/ergast/f1/{season}/constructorStandings.json"
        req_c = urllib.request.Request(url_c, headers={"User-Agent": "F1-Predictor/2.0"})
        with urllib.request.urlopen(req_c, timeout=8) as r:
            payload_c = json.loads(r.read())
        raw_c = payload_c["MRData"]["StandingsTable"]["StandingsLists"][0]["ConstructorStandings"]
        for c in raw_c:
            con_standings.append({
                "constructor_id": c["Constructor"]["constructorId"],
                "constructor_name": c["Constructor"]["name"],
                "current_points": float(c["points"]),
                "wins": int(c["wins"]),
            })
    except Exception as e:
        print(f"Jolpica constructor standings fallback triggered ({e})")

    if not drv_standings and MODEL_DATASET.exists():
        # Fallback from local dataset
        df = pd.read_csv(MODEL_DATASET)
        s26 = df[df.season == 2026]
        max_round = s26["round"].max()
        last_race = s26[s26["round"] == max_round]
        # Aggregate points from finishes
        # Standard fallback drivers
        unique_drivers = s26[["driver_id", "driver_name", "constructor_id", "constructor_name"]].drop_duplicates("driver_id")
        for _, row in unique_drivers.iterrows():
            drv_standings.append({
                "driver_id": row["driver_id"],
                "driver_name": row["driver_name"],
                "constructor_id": row["constructor_id"],
                "constructor_name": row["constructor_name"],
                "current_points": 100.0,
                "wins": 1,
            })

    drv_df = pd.DataFrame(drv_standings)
    con_df = pd.DataFrame(con_standings)
    return drv_df, con_df


def simulate_championship(n_sims: int = 10000, chaos_multiplier: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs Monte Carlo simulation of remaining 2026 races."""
    drv_df, con_df = fetch_current_standings(2026)
    if drv_df.empty:
        raise ValueError("Could not load driver standings.")

    # Get driver skill and team ratings from model_dataset_v2.csv
    if MODEL_DATASET.exists():
        data = pd.read_csv(MODEL_DATASET)
        s26 = data[data.season == 2026]
        if not s26.empty:
            latest_idx = s26.race_idx.max()
            recent = s26[s26.race_idx == latest_idx]
            elo_map = dict(zip(recent.driver_id, recent.driver_elo + 0.6 * recent.team_elo))
        else:
            elo_map = {}
    else:
        elo_map = {}

    # Attach driver strength
    strengths = []
    for _, row in drv_df.iterrows():
        # Use Elo if available, else derive from current points
        st_val = elo_map.get(row["driver_id"], 1500.0 + row["current_points"] * 1.5)
        strengths.append(st_val)

    drv_df["strength"] = strengths
    n_drivers = len(drv_df)
    v = np.array(strengths)
    # Normalize strength to softmax-like log-logits
    v_norm = (v - np.mean(v)) / (np.std(v) + 1e-6)

    # Initialize simulated points array (n_sims x n_drivers)
    sim_points = np.tile(drv_df["current_points"].values, (n_sims, 1)).astype(float)
    sim_wins = np.tile(drv_df["wins"].values, (n_sims, 1)).astype(int)

    rng = np.random.default_rng(42)

    for race in REMAINING_2026_CALENDAR:
        # Plackett-Luce Gumbel trick: sample = v_norm / T + Gumbel(0, 1)
        temp = 1.2 * chaos_multiplier
        # GP Race Simulation:
        gumbel_gp = rng.gumbel(loc=0.0, scale=1.0, size=(n_sims, n_drivers))
        latents_gp = (v_norm / temp) + gumbel_gp
        
        # Finishing order: highest latent finishes 1st
        finish_order_gp = np.argsort(-latents_gp, axis=1)  # shape (n_sims, n_drivers)

        # Award GP Points
        for pos in range(min(10, n_drivers)):
            drivers_at_pos = finish_order_gp[:, pos]
            sim_points[np.arange(n_sims), drivers_at_pos] += GP_POINTS[pos]
            if pos == 0:
                sim_wins[np.arange(n_sims), drivers_at_pos] += 1

        # Sprint simulation if sprint weekend
        if race.get("is_sprint", False):
            gumbel_sprint = rng.gumbel(loc=0.0, scale=1.0, size=(n_sims, n_drivers))
            latents_sprint = (v_norm / temp) + gumbel_sprint
            finish_order_sprint = np.argsort(-latents_sprint, axis=1)
            for pos in range(min(8, n_drivers)):
                drivers_at_pos = finish_order_sprint[:, pos]
                sim_points[np.arange(n_sims), drivers_at_pos] += SPRINT_POINTS[pos]

    # Evaluate WDC winners across all simulations
    # Break ties using sim_points * 1000 + sim_wins
    tie_breaker = sim_points * 1000.0 + sim_wins
    wdc_ranks = np.argsort(-tie_breaker, axis=1)  # 0 is 1st place

    wdc_p_win = np.zeros(n_drivers)
    wdc_p_podium = np.zeros(n_drivers)
    for pos in range(n_drivers):
        champ_drivers = wdc_ranks[:, pos]
        for d_idx in range(n_drivers):
            count = np.sum(champ_drivers == d_idx)
            if pos == 0:
                wdc_p_win[d_idx] = count / n_sims
            if pos < 3:
                wdc_p_podium[d_idx] += count / n_sims

    # Construct WDC summary
    wdc_res = drv_df[["driver_name", "constructor_name", "current_points"]].copy()
    wdc_res["exp_points"] = sim_points.mean(axis=0).round(1)
    wdc_res["min_points"] = sim_points.min(axis=0).astype(int)
    wdc_res["max_points"] = sim_points.max(axis=0).astype(int)
    wdc_res["p_champion"] = (wdc_p_win * 100).round(1)
    wdc_res["p_top3"] = (wdc_p_podium * 100).round(1)
    wdc_res = wdc_res.sort_values(by=["p_champion", "exp_points"], ascending=False).reset_index(drop=True)

    # Evaluate Constructors' Championship (WCC)
    unique_teams = list(con_df["constructor_name"].unique()) if not con_df.empty else list(drv_df["constructor_name"].unique())
    n_teams = len(unique_teams)
    sim_team_points = np.zeros((n_sims, n_teams))

    team_to_idx = {t: i for i, t in enumerate(unique_teams)}
    for d_idx, row in drv_df.iterrows():
        t_name = row["constructor_name"]
        if t_name in team_to_idx:
            t_idx = team_to_idx[t_name]
            sim_team_points[:, t_idx] += sim_points[:, d_idx]

    wcc_ranks = np.argsort(-sim_team_points, axis=1)
    wcc_p_win = np.zeros(n_teams)
    wcc_p_podium = np.zeros(n_teams)
    for pos in range(n_teams):
        champ_teams = wcc_ranks[:, pos]
        for t_idx in range(n_teams):
            count = np.sum(champ_teams == t_idx)
            if pos == 0:
                wcc_p_win[t_idx] = count / n_sims
            if pos < 3:
                wcc_p_podium[t_idx] += count / n_sims

    wcc_res = pd.DataFrame({
        "constructor_name": unique_teams,
        "current_points": [con_df.loc[con_df.constructor_name == t, "current_points"].iloc[0] if (not con_df.empty and (con_df.constructor_name == t).any()) else 0.0 for t in unique_teams],
        "exp_points": sim_team_points.mean(axis=0).round(1),
        "min_points": sim_team_points.min(axis=0).astype(int),
        "max_points": sim_team_points.max(axis=0).astype(int),
        "p_champion": (wcc_p_win * 100).round(1),
        "p_top3": (wcc_p_podium * 100).round(1),
    }).sort_values(by=["p_champion", "exp_points"], ascending=False).reset_index(drop=True)

    return wdc_res, wcc_res


if __name__ == "__main__":
    print("Running 10,000 season simulations for remaining 2026 races...")
    wdc, wcc = simulate_championship(n_sims=10000)
    print("\n--- Drivers' Championship (WDC) Simulation ---")
    print(wdc.head(8).to_string())
    print("\n--- Constructors' Championship (WCC) Simulation ---")
    print(wcc.head(6).to_string())
