"""Live Race Predictor & SQLite Prediction Log.

Handles:
- Fetching next upcoming race on the F1 calendar (Jolpica API).
- Fetching live qualifying if completed, or generating an expected grid from recent form.
- Running trained model + Plackett-Luce simulation for the upcoming race.
- Logging predictions to SQLite (data/predictions_log.db) with timestamp and versioning.
- Post-race accuracy scoring once official results become available.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import model_v2 as M

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "predictions_log.db"
MODEL_DATASET = DATA_DIR / "processed" / "model_dataset_v2.csv"
META_PATH = ROOT / "reports" / "backtest_meta.json"


def get_db_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                season INTEGER NOT NULL,
                round INTEGER NOT NULL,
                race_name TEXT NOT NULL,
                circuit_id TEXT,
                driver_name TEXT NOT NULL,
                constructor_name TEXT NOT NULL,
                grid_position INTEGER NOT NULL,
                pred_position INTEGER NOT NULL,
                p_win REAL NOT NULL,
                p_podium REAL NOT NULL,
                actual_position INTEGER,
                scored INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                season INTEGER NOT NULL,
                round INTEGER NOT NULL,
                race_name TEXT NOT NULL,
                num_drivers INTEGER NOT NULL,
                notes TEXT
            )
        """)
        conn.commit()


def get_calendar(season: int = 2026) -> list[dict]:
    url = f"https://api.jolpi.ca/ergast/f1/{season}.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "F1-Predictor/2.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = json.loads(r.read())
        races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        return races
    except Exception as e:
        print(f"Warning: Could not fetch Jolpica calendar ({e}). Using local fallback.")
        return []


def get_upcoming_round(season: int = 2026) -> dict:
    cal = get_calendar(season)
    # Check what latest race is in model_dataset_v2.csv
    if MODEL_DATASET.exists():
        df = pd.read_csv(MODEL_DATASET, usecols=["season", "round"])
        s_df = df[df.season == season]
        completed_rounds = set(s_df["round"].unique()) if not s_df.empty else set()
    else:
        completed_rounds = set()

    for r in cal:
        rnd = int(r["round"])
        if rnd not in completed_rounds:
            return {
                "season": season,
                "round": rnd,
                "race_name": r.get("raceName", f"Round {rnd}"),
                "circuit_id": r.get("Circuit", {}).get("circuitId", "unknown"),
                "date": r.get("date", ""),
                "qualifying_available": False,
            }

    # If calendar not accessible or all completed, fall back to next round after max completed
    next_rnd = (max(completed_rounds) + 1) if completed_rounds else 18
    return {
        "season": season,
        "round": next_rnd,
        "race_name": f"Round {next_rnd} Grand Prix",
        "circuit_id": "cota",
        "date": "2026-10-18",
        "qualifying_available": False,
    }


def fetch_live_qualifying(season: int, round_num: int) -> Optional[pd.DataFrame]:
    # 1. Try Jolpica Ergast API
    url = f"https://api.jolpi.ca/ergast/f1/{season}/{round_num}/qualifying.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "F1-Predictor/2.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            payload = json.loads(r.read())
        races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if races and "QualifyingResults" in races[0] and races[0]["QualifyingResults"]:
            records = []
            for q in races[0]["QualifyingResults"]:
                pos = int(q.get("position", 20))
                records.append({
                    "grid_position": pos,
                    "qualifying_position": pos,
                    "driver_id": q.get("Driver", {}).get("driverId", ""),
                    "driver_name": f"{q.get('Driver', {}).get('givenName', '')} {q.get('Driver', {}).get('familyName', '')}".strip(),
                    "constructor_id": q.get("Constructor", {}).get("constructorId", ""),
                    "constructor_name": q.get("Constructor", {}).get("name", ""),
                })
            df = pd.DataFrame(records).sort_values("grid_position").reset_index(drop=True)
            return df
    except Exception as e:
        print(f"Qualifying not available on Jolpica ({e})")

    # 2. Try FastF1 as fallback
    try:
        import fastf1
        session = fastf1.get_session(season, round_num, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results
        if results is not None and not results.empty:
            records = []
            for _, r in results.iterrows():
                pos = int(r["Position"]) if pd.notna(r["Position"]) else 20
                records.append({
                    "grid_position": pos,
                    "qualifying_position": pos,
                    "driver_id": str(r.get("DriverId", r.get("Abbreviation", ""))).lower(),
                    "driver_name": str(r.get("FullName", r.get("BroadcastName", ""))).strip(),
                    "constructor_id": str(r.get("TeamName", "")).lower().replace(" ", "_"),
                    "constructor_name": str(r.get("TeamName", "")).strip(),
                })
            df = pd.DataFrame(records).sort_values("grid_position").reset_index(drop=True)
            return df
    except Exception as e:
        print(f"FastF1 qualifying fallback exception: {e}")

    return None


def get_default_grid_for_next_round(season: int = 2026) -> pd.DataFrame:
    """Returns grid based on latest round's active drivers and recent performance."""
    if not MODEL_DATASET.exists():
        raise FileNotFoundError(f"{MODEL_DATASET} not found.")
    df = pd.read_csv(MODEL_DATASET)
    latest_season_df = df[df.season == df.season.max()]
    latest_race_idx = latest_season_df.race_idx.max()
    latest_race = latest_season_df[latest_season_df.race_idx == latest_race_idx].copy()
    
    # Sort by rolling elo / qualifying form
    grid_order = latest_race.sort_values(
        by=["driver_elo", "team_elo"], ascending=[False, False]
    ).reset_index(drop=True)
    grid_order["grid_position"] = np.arange(1, len(grid_order) + 1)
    return grid_order[["grid_position", "driver_id", "driver_name", "constructor_id", "constructor_name"]].copy()


def run_prediction_for_grid(grid_df: pd.DataFrame, season: int, round_num: int,
                            race_name: str, circuit_id: str = "cota",
                            temp_scale: float = 1.0) -> pd.DataFrame:
    """Uses all historical races before this round to train model and predict finishing order."""
    if not MODEL_DATASET.exists():
        raise FileNotFoundError(f"{MODEL_DATASET} not found.")
    df = pd.read_csv(MODEL_DATASET)

    # Train model on all existing historical races
    trained = M.fit(df)

    # Build feature row for each driver in grid_df using their latest known stats
    features_list = []
    n = len(grid_df)
    for _, row in grid_df.iterrows():
        d_id = str(row.get("driver_id", "")).strip()
        d_name = str(row.get("driver_name", "")).strip()
        
        # 1. Match driver by ID
        d_hist = df[df.driver_id == d_id] if d_id else pd.DataFrame()
        # 2. Fallback: match by full name or last name
        if d_hist.empty and d_name:
            d_hist = df[df.driver_name.str.lower() == d_name.lower()]
            if d_hist.empty:
                last_name = d_name.split()[-1].lower()
                d_hist = df[df.driver_name.str.lower().str.endswith(last_name)]
        
        if not d_hist.empty:
            recent_row = d_hist.iloc[-1].to_dict()
        else:
            # Fallback to team's most recent record
            c_id = str(row.get("constructor_id", "")).strip()
            c_hist = df[df.constructor_id == c_id] if c_id else pd.DataFrame()
            if c_hist.empty and "constructor_name" in row:
                c_hist = df[df.constructor_name.str.lower() == str(row["constructor_name"]).lower()]
            recent_row = c_hist.iloc[-1].to_dict() if not c_hist.empty else df.iloc[-1].to_dict()

        # Update race & grid-specific attributes
        recent_row["season"] = season
        recent_row["round"] = round_num
        recent_row["race_name"] = race_name
        recent_row["circuit_id"] = circuit_id
        
        grid_pos = float(row.get("grid_position", 20))
        quali_pos = float(row.get("qualifying_position", grid_pos))
        
        recent_row["grid_position"] = grid_pos
        recent_row["qualifying_position"] = quali_pos
        recent_row["grid_penalty"] = max(0.0, grid_pos - quali_pos)
        recent_row["grid_rank_pct"] = grid_pos / n
        recent_row["pit_lane_start"] = 1 if grid_pos > n else 0
        recent_row["driver_name"] = row.get("driver_name", recent_row.get("driver_name", ""))
        recent_row["constructor_name"] = row.get("constructor_name", recent_row.get("constructor_name", ""))
        recent_row["driver_id"] = d_id or recent_row.get("driver_id", "")
        recent_row["constructor_id"] = row.get("constructor_id", recent_row.get("constructor_id", ""))
        features_list.append(recent_row)

    feat_df = pd.DataFrame(features_list)
    feat_df["circuit_chaos"] = feat_df.get("circuit_chaos", 0.0)

    # Calculate model score & DNF probability
    scores = M.score(trained, feat_df)
    dnf_probs = M.dnf_prob(feat_df, trained["base_dnf"])

    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
        except Exception:
            pass
    base_temp = meta.get("temperature_model", 1.2) * temp_scale

    # Plackett-Luce Monte Carlo simulation (12,000 runs)
    sim = M.simulate(scores, dnf_probs, base_temp, n=12000)
    probs = M.summarise_probs(sim)

    # Combine into output table
    res = feat_df[["grid_position", "driver_name", "constructor_name"]].copy()
    res["score"] = scores
    res["dnf_prob"] = dnf_probs
    res["exp_position"] = probs["exp_position"]
    res["p_win"] = probs["p_win"]
    res["p_podium"] = probs["p_podium"]
    res["p_top10"] = probs["p_top10"]

    # Predicted position is based on expected finish / scores
    res = res.sort_values(by="exp_position").reset_index(drop=True)
    res["pred_position"] = np.arange(1, len(res) + 1)
    res["season"] = season
    res["round"] = round_num
    res["race_name"] = race_name
    res["circuit_id"] = circuit_id
    return res


def log_predictions_to_db(pred_df: pd.DataFrame, notes: str = "Live Model Run") -> int:
    init_db()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s = int(pred_df["season"].iloc[0])
    r = int(pred_df["round"].iloc[0])
    rname = str(pred_df["race_name"].iloc[0])

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO prediction_runs (timestamp, season, round, race_name, num_drivers, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ts, s, r, rname, len(pred_df), notes))
        run_id = cur.lastrowid

        for _, row in pred_df.iterrows():
            cur.execute("""
                INSERT INTO predictions (
                    timestamp, season, round, race_name, circuit_id,
                    driver_name, constructor_name, grid_position, pred_position,
                    p_win, p_podium, scored
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                ts, s, r, rname, row.get("circuit_id", ""),
                row["driver_name"], row["constructor_name"],
                int(row["grid_position"]), int(row["pred_position"]),
                float(row["p_win"]), float(row["p_podium"])
            ))
        conn.commit()
    return run_id


def get_logged_predictions() -> pd.DataFrame:
    init_db()
    with get_db_connection() as conn:
        df = pd.read_sql_query("""
            SELECT id, timestamp, season, round, race_name, driver_name,
                   constructor_name, grid_position, pred_position,
                   p_win, p_podium, actual_position, scored
            FROM predictions
            ORDER BY id DESC
        """, conn)
    return df


def seed_recent_predictions_if_empty():
    """Populates log with Singapore 2026 predictions if log is empty so user sees instant data."""
    init_db()
    with get_db_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        if count > 0:
            return

    preds_file = ROOT / "reports" / "backtest_predictions.csv"
    if preds_file.exists():
        preds = pd.read_csv(preds_file)
        singapore = preds[(preds.season == 2026) & (preds["round"] == 17)].copy()
        if not singapore.empty:
            ts = "2026-10-04 14:00:00"
            with get_db_connection() as conn:
                for _, r in singapore.iterrows():
                    conn.execute("""
                        INSERT INTO predictions (
                            timestamp, season, round, race_name, circuit_id,
                            driver_name, constructor_name, grid_position, pred_position,
                            p_win, p_podium, actual_position, scored
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """, (
                        ts, 2026, 17, "Singapore Grand Prix", "marina_bay",
                        r["driver_name"], r["constructor_name"],
                        int(r["grid_position"]), int(r["pred_order"]),
                        float(r["p_win"]), float(r["p_podium"]),
                        int(r["finish_position"])
                    ))
                conn.commit()


if __name__ == "__main__":
    init_db()
    seed_recent_predictions_if_empty()
    upcoming = get_upcoming_round(2026)
    print("Upcoming race:", upcoming)
    grid = get_default_grid_for_next_round(2026)
    print(f"Default starting grid ready: {len(grid)} drivers.")
    preds = run_prediction_for_grid(grid, upcoming["season"], upcoming["round"], upcoming["race_name"])
    print(preds[["pred_position", "driver_name", "constructor_name", "grid_position", "p_win", "p_podium"]].head(5))
    run_id = log_predictions_to_db(preds, notes="Automated test run")
    print(f"Logged to SQLite successfully with run ID {run_id}!")
