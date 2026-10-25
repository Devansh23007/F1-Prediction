"""Head-to-Head analytics and Circuit Profiles for F1 Race Predictor.

Provides:
- Driver vs Driver head-to-head comparison (finishes, qualifying, podiums, Elo evolution).
- Constructor vs Constructor comparison (wins, podiums, Elo evolution).
- Detailed Circuit Profiles (DNF rates, chaos index, pit stop loss, pole-to-win conversions).
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def get_driver_h2h(df: pd.DataFrame, driver_a: str, driver_b: str) -> Dict[str, Any]:
    """Calculates comprehensive head-to-head metrics and Elo trajectories for two drivers."""
    sub_a = df[df.driver_name == driver_a].copy()
    sub_b = df[df.driver_name == driver_b].copy()

    # Find common races
    common_races = set(sub_a.race_id).intersection(set(sub_b.race_id))
    
    # Filter to common races
    m_a = sub_a[sub_a.race_id.isin(common_races)].sort_values(["season", "round"]).reset_index(drop=True)
    m_b = sub_b[sub_b.race_id.isin(common_races)].sort_values(["season", "round"]).reset_index(drop=True)

    merged = pd.merge(
        m_a[["race_id", "race_idx", "season", "round", "race_name", "date", "finish_position",
             "grid_position", "qualifying_position", "points", "dnf", "driver_elo", "constructor_name"]],
        m_b[["race_id", "finish_position", "grid_position", "qualifying_position", "points",
             "dnf", "driver_elo", "constructor_name"]],
        on="race_id",
        suffixes=("_a", "_b")
    )

    total_races = len(merged)
    if total_races == 0:
        return {"total_races": 0}

    # Race finish battle
    # When both finished
    both_finished = merged[(merged.dnf_a == 0) & (merged.dnf_b == 0)]
    finishes_a_ahead = int((merged.finish_position_a < merged.finish_position_b).sum())
    finishes_b_ahead = int((merged.finish_position_b < merged.finish_position_a).sum())

    # Qualifying battle
    quali_a_ahead = int((merged.qualifying_position_a < merged.qualifying_position_b).sum())
    quali_b_ahead = int((merged.qualifying_position_b < merged.qualifying_position_a).sum())

    # Wins and podiums in common races
    wins_a = int((merged.finish_position_a == 1).sum())
    wins_b = int((merged.finish_position_b == 1).sum())
    podiums_a = int((merged.finish_position_a <= 3).sum())
    podiums_b = int((merged.finish_position_b <= 3).sum())

    # Points scored in common races
    pts_a = float(merged.points_a.sum())
    pts_b = float(merged.points_b.sum())

    # Check if they were teammates
    teammate_races = merged[merged.constructor_name_a == merged.constructor_name_b]
    were_teammates = len(teammate_races) > 0
    teammate_count = len(teammate_races)

    # Elo timeseries
    elo_ts = merged[["season", "round", "race_name", "race_idx", "driver_elo_a", "driver_elo_b"]].copy()

    return {
        "total_races": total_races,
        "finishes_a_ahead": finishes_a_ahead,
        "finishes_b_ahead": finishes_b_ahead,
        "quali_a_ahead": quali_a_ahead,
        "quali_b_ahead": quali_b_ahead,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "podiums_a": podiums_a,
        "podiums_b": podiums_b,
        "pts_a": pts_a,
        "pts_b": pts_b,
        "were_teammates": were_teammates,
        "teammate_count": teammate_count,
        "elo_ts": elo_ts,
        "merged_races": merged
    }


def get_constructor_h2h(df: pd.DataFrame, con_a: str, con_b: str) -> Dict[str, Any]:
    """Calculates head-to-head metrics and Elo trajectories for two constructors."""
    sub_a = df[df.constructor_name == con_a].copy()
    sub_b = df[df.constructor_name == con_b].copy()

    # Aggregate by race for team
    team_a_races = sub_a.groupby(["race_id", "season", "round", "race_name", "race_idx"], as_index=False).agg(
        team_elo=("team_elo", "first"),
        points=("points", "sum"),
        best_finish=("finish_position", "min"),
        wins=("finish_position", lambda s: (s == 1).sum()),
        podiums=("finish_position", lambda s: (s <= 3).sum())
    )
    team_b_races = sub_b.groupby(["race_id", "season", "round", "race_name", "race_idx"], as_index=False).agg(
        team_elo=("team_elo", "first"),
        points=("points", "sum"),
        best_finish=("finish_position", "min"),
        wins=("finish_position", lambda s: (s == 1).sum()),
        podiums=("finish_position", lambda s: (s <= 3).sum())
    )

    common = pd.merge(team_a_races, team_b_races, on="race_id", suffixes=("_a", "_b"))
    total_races = len(common)
    if total_races == 0:
        return {"total_races": 0}

    ahead_a = int((common.points_a > common.points_b).sum())
    ahead_b = int((common.points_b > common.points_a).sum())
    wins_a = int(common.wins_a.sum())
    wins_b = int(common.wins_b.sum())
    podiums_a = int(common.podiums_a.sum())
    podiums_b = int(common.podiums_b.sum())
    pts_a = float(common.points_a.sum())
    pts_b = float(common.points_b.sum())

    return {
        "total_races": total_races,
        "ahead_a": ahead_a,
        "ahead_b": ahead_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "podiums_a": podiums_a,
        "podiums_b": podiums_b,
        "pts_a": pts_a,
        "pts_b": pts_b,
        "elo_ts": common[["season_a", "round_a", "race_name_a", "race_idx_a", "team_elo_a", "team_elo_b"]].rename(
            columns={"season_a": "season", "round_a": "round", "race_name_a": "race_name", "race_idx_a": "race_idx"}
        )
    }


def get_all_circuits_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes chaos, DNF rates, pit stop times, and overtaking stats for all circuits."""
    circuits = []
    for c_id, g in df.groupby("circuit_id"):
        races_count = g.race_id.nunique()
        dnf_rate = g.dnf.mean() * 100
        chaos = g.circuit_chaos.iloc[0] if "circuit_chaos" in g.columns else 0.0
        pit_time = g.pit_loss_time.iloc[0] if "pit_loss_time" in g.columns else 21.5
        high_deg = int(g.high_deg_circuit.iloc[0]) if "high_deg_circuit" in g.columns else 0
        
        # Pole to win conversion
        pole_sitters = g[g.grid_position == 1]
        pole_wins = (pole_sitters.finish_position == 1).sum()
        pole_to_win = (pole_wins / len(pole_sitters) * 100) if len(pole_sitters) > 0 else 0.0

        # Avg position movement (overtaking friendliness)
        avg_pos_change = (g.finish_position - g.grid_position).abs().mean()

        # Circuit friendly name
        c_name = g.race_name.iloc[-1]

        circuits.append({
            "circuit_id": c_id,
            "circuit_name": c_name,
            "races_held": races_count,
            "dnf_rate_pct": round(dnf_rate, 1),
            "chaos_index": round(float(chaos), 2),
            "pit_loss_sec": round(float(pit_time), 1),
            "high_degradation": "Yes" if high_deg else "No",
            "pole_to_win_pct": round(pole_to_win, 1),
            "avg_pos_change": round(avg_pos_change, 2)
        })

    res = pd.DataFrame(circuits).sort_values("chaos_index", ascending=False).reset_index(drop=True)
    return res


def get_circuit_details(df: pd.DataFrame, circuit_id: str) -> Dict[str, Any]:
    """Returns granular race history and top drivers for a specific circuit."""
    c_df = df[df.circuit_id == circuit_id].copy()
    if c_df.empty:
        return {}

    # Top drivers at this circuit
    top_drivers = c_df.groupby("driver_name").agg(
        races=("race_id", "nunique"),
        wins=("finish_position", lambda s: (s == 1).sum()),
        podiums=("finish_position", lambda s: (s <= 3).sum()),
        avg_finish=("finish_position", "mean")
    ).reset_index()
    top_drivers = top_drivers.sort_values(by=["wins", "podiums"], ascending=False).reset_index(drop=True)

    # Race-by-race winners & pole sitters
    race_history = []
    for r_id, g in c_df.groupby("race_id", sort=False):
        winner_row = g[g.finish_position == 1]
        pole_row = g[g.grid_position == 1]
        winner = winner_row.driver_name.iloc[0] if not winner_row.empty else "Unknown"
        winner_team = winner_row.constructor_name.iloc[0] if not winner_row.empty else "Unknown"
        pole = pole_row.driver_name.iloc[0] if not pole_row.empty else "Unknown"
        dnf_count = int(g.dnf.sum())
        is_wet = int(g.rainfall.iloc[0]) if "rainfall" in g.columns else 0

        race_history.append({
            "season": int(g.season.iloc[0]),
            "round": int(g.round.iloc[0]),
            "race_name": g.race_name.iloc[0],
            "winner": winner,
            "winner_team": winner_team,
            "pole_sitter": pole,
            "dnfs": dnf_count,
            "weather": "Rain / Wet" if is_wet else "Dry"
        })

    return {
        "top_drivers": top_drivers,
        "race_history": pd.DataFrame(race_history).sort_values(["season", "round"], ascending=False).reset_index(drop=True)
    }
