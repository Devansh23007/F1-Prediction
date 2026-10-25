"""Leaderboard management for 'Check Yourself' game.
Stores player scores, rounds played, and model vs user stats in SQLite.
"""
from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "predictions_log.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_leaderboard():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS game_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                player_name TEXT NOT NULL,
                race_id INTEGER NOT NULL,
                race_name TEXT NOT NULL,
                user_picks TEXT NOT NULL,
                actual_podium TEXT NOT NULL,
                user_score INTEGER NOT NULL,
                model_score INTEGER NOT NULL,
                grid_score INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard (
                player_name TEXT PRIMARY KEY,
                rounds_played INTEGER DEFAULT 0,
                total_user_score INTEGER DEFAULT 0,
                total_model_score INTEGER DEFAULT 0,
                total_grid_score INTEGER DEFAULT 0,
                user_wins INTEGER DEFAULT 0,
                last_played TEXT NOT NULL
            )
        """)
        conn.commit()


def record_round(player_name: str, race_id: int, race_name: str,
                 user_picks: list[str], actual_podium: list[str],
                 user_score: int, model_score: int, grid_score: int):
    init_leaderboard()
    player_name = player_name.strip() or "Anonymous Player"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO game_rounds (
                timestamp, player_name, race_id, race_name,
                user_picks, actual_podium, user_score, model_score, grid_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, player_name, race_id, race_name,
            ", ".join(user_picks), ", ".join(actual_podium),
            user_score, model_score, grid_score
        ))

        # Update or insert player in leaderboard
        cur.execute("SELECT * FROM leaderboard WHERE player_name = ?", (player_name,))
        existing = cur.fetchone()

        user_won = 1 if user_score > max(model_score, grid_score) else 0

        if existing:
            conn.execute("""
                UPDATE leaderboard
                SET rounds_played = rounds_played + 1,
                    total_user_score = total_user_score + ?,
                    total_model_score = total_model_score + ?,
                    total_grid_score = total_grid_score + ?,
                    user_wins = user_wins + ?,
                    last_played = ?
                WHERE player_name = ?
            """, (user_score, model_score, grid_score, user_won, ts, player_name))
        else:
            conn.execute("""
                INSERT INTO leaderboard (
                    player_name, rounds_played, total_user_score,
                    total_model_score, total_grid_score, user_wins, last_played
                ) VALUES (?, 1, ?, ?, ?, ?, ?)
            """, (player_name, user_score, model_score, grid_score, user_won, ts))
        conn.commit()


def get_leaderboard_df() -> pd.DataFrame:
    init_leaderboard()
    with get_conn() as conn:
        df = pd.read_sql_query("""
            SELECT player_name, rounds_played, total_user_score,
                   total_model_score, total_grid_score, user_wins, last_played
            FROM leaderboard
            ORDER BY total_user_score DESC, rounds_played DESC
        """, conn)
    if not df.empty:
        df["avg_score"] = (df.total_user_score / df.rounds_played).round(2)
        df["win_pct"] = ((df.user_wins / df.rounds_played) * 100).round(1)
    return df


def get_recent_game_rounds(limit: int = 20) -> pd.DataFrame:
    init_leaderboard()
    with get_conn() as conn:
        df = pd.read_sql_query(f"""
            SELECT timestamp, player_name, race_name, user_picks,
                   actual_podium, user_score, model_score, grid_score
            FROM game_rounds
            ORDER BY id DESC LIMIT {limit}
        """, conn)
    return df
