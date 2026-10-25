"""Feature engineering v2.

Every feature for a race uses ONLY information available before that race
starts (previous races, plus this race's qualifying / grid).  Elo ratings
are stored *before* the race is used to update them.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

FINISHED = {"Finished", "Lapped"}


def _is_finished(status: str) -> bool:
    s = str(status)
    return s in FINISHED or s.startswith("+")


def _parse_time(t):
    """'1:22.824' -> seconds (float) ; NaN if missing."""
    if pd.isna(t) or str(t).strip() == "":
        return np.nan
    t = str(t)
    try:
        if ":" in t:
            m, s = t.split(":")
            return int(m) * 60 + float(s)
        return float(t)
    except ValueError:
        return np.nan


def load_raw():
    res = pd.read_csv(RAW / "race_results.csv")
    qual = pd.read_csv(RAW / "qualifying_results.csv")
    races = pd.read_csv(RAW / "races.csv")
    for d in (res, qual, races):
        d["season"] = pd.to_numeric(d["season"], errors="coerce")
        d["round"] = pd.to_numeric(d["round"], errors="coerce")
        d.dropna(subset=["season", "round"], inplace=True)
        d[["season", "round"]] = d[["season", "round"]].astype(int)
    for c in ["grid_position", "finish_position", "points", "laps_completed"]:
        res[c] = pd.to_numeric(res[c], errors="coerce")
    qual["qualifying_position"] = pd.to_numeric(
        qual["qualifying_position"], errors="coerce")
    res = res.drop_duplicates(["season", "round", "driver_id"])
    qual = qual.drop_duplicates(["season", "round", "driver_id"])
    return res, qual, races


def quali_pace_features(qual: pd.DataFrame) -> pd.DataFrame:
    q = qual.copy()
    for c in ["q1_time", "q2_time", "q3_time"]:
        q[c + "_s"] = q[c].map(_parse_time)
    keys = ["season", "round"]
    for c in ["q1", "q2", "q3"]:
        best = q.groupby(keys)[c + "_time_s"].transform("min")
        q[c + "_gap"] = (q[c + "_time_s"] - best) / best * 100  # % off pole
    # deepest session gap = most informative pace number
    q["quali_gap_pct"] = q["q3_gap"].fillna(q["q2_gap"]).fillna(q["q1_gap"])
    q["reached_q3"] = q["q3_time_s"].notna().astype(int)
    q["reached_q2"] = q["q2_time_s"].notna().astype(int)
    return q[keys + ["driver_id", "quali_gap_pct", "q1_gap",
                     "reached_q2", "reached_q3"]]


def run_elo(df: pd.DataFrame, pos_col: str, prefix: str,
            entity: str, k=10.0, season_regress=0.25,
            rules_reset_regress=0.55,
            reset_seasons=(2022, 2026)) -> pd.DataFrame:
    """Pairwise Elo within each race.  Stores PRE-race rating.

    entity: 'driver_id' or 'constructor_id'.  For constructors the
    team's mean position in that race is used.

    At major technical regulation resets (e.g. 2022 ground effect,
    2026 active aero / power unit overhaul), team ratings regress more strongly
    toward the 1500 mean (55% regression vs standard 25%).
    """
    df = df.sort_values(["season", "round"]).copy()
    ratings: dict = {}
    pre = {}
    last_season = None
    reset_set = set(reset_seasons or [])
    for (season, rnd), race in df.groupby(["season", "round"], sort=True):
        if last_season is not None and season != last_season:
            if season in reset_set and entity == "constructor_id":
                regress = rules_reset_regress
            elif season in reset_set and entity == "driver_id":
                regress = min(rules_reset_regress, 0.35)
            else:
                regress = season_regress

            for e in ratings:  # pull toward the mean between seasons
                ratings[e] = 1500 + (ratings[e] - 1500) * (1 - regress)
        last_season = season

        pos = race.groupby(entity)[pos_col].mean().dropna()
        ents = pos.index.tolist()
        r = np.array([ratings.get(e, 1500.0) for e in ents])
        for e, v in zip(ents, r):
            pre[(season, rnd, e)] = v
        n = len(ents)
        if n < 2:
            continue
        p = pos.values
        # expected score of i vs j
        diff = r[None, :] - r[:, None]  # r_j - r_i
        exp = 1.0 / (1.0 + 10 ** (diff / 400.0))
        actual = (p[:, None] < p[None, :]).astype(float)
        actual[p[:, None] == p[None, :]] = 0.5
        np.fill_diagonal(exp, 0)
        np.fill_diagonal(actual, 0)
        delta = k * (actual - exp).sum(axis=1) / (n - 1) * 2
        for e, d in zip(ents, delta):
            ratings[e] = ratings.get(e, 1500.0) + d
    out = df[["season", "round", entity]].copy()
    out[prefix] = [pre.get((s, r_, e), 1500.0)
                   for s, r_, e in zip(out.season, out["round"], out[entity])]
    return out.drop_duplicates(["season", "round", entity])


def build(save=True) -> pd.DataFrame:
    res, qual, races = load_raw()
    pace = quali_pace_features(qual)

    df = res.merge(qual[["season", "round", "driver_id", "qualifying_position"]],
                   on=["season", "round", "driver_id"], how="left")
    df = df.merge(pace, on=["season", "round", "driver_id"], how="left")
    df = df.merge(
        races[["season", "round", "circuit_id", "country"]]
        .drop_duplicates(["season", "round"]),
        on=["season", "round"], how="left", suffixes=("", "_sched"))

    # ---------- FastF1 practice (FP2 / FP1 long-run pace) ----------
    fp_path = RAW / "fastf1_practice_pace.csv"
    if fp_path.exists():
        fp = pd.read_csv(fp_path)
        fp["season"] = fp["season"].astype(int)
        fp["round"] = fp["round"].astype(int)
        fp["driver_code"] = fp["driver_code"].astype(str).str.strip().str.upper()
        df["driver_code_clean"] = df["driver_code"].astype(str).str.strip().str.upper()
        cols = ["season", "round", "driver_code", "fp_longrun_gap_pct",
                "fp_race_vs_quali_pace", "fp_laps_count"]
        df = df.merge(fp[cols],
                      left_on=["season", "round", "driver_code_clean"],
                      right_on=["season", "round", "driver_code"],
                      how="left", suffixes=("", "_fp"))
        df.drop(columns=["driver_code_clean"], inplace=True)
        if "driver_code_fp" in df.columns:
            df.drop(columns=["driver_code_fp"], inplace=True)
    else:
        df["fp_longrun_gap_pct"] = np.nan
        df["fp_race_vs_quali_pace"] = np.nan
        df["fp_laps_count"] = np.nan

    df["fp_has_data"] = df["fp_longrun_gap_pct"].notna().astype(int)
    # If no long-run practice pace is available, use qualifying pace gap as baseline
    df["fp_longrun_gap_pct"] = df["fp_longrun_gap_pct"].fillna(df["quali_gap_pct"])
    df["fp_race_vs_quali_pace"] = df["fp_race_vs_quali_pace"].fillna(0.0)
    df["fp_laps_count"] = df["fp_laps_count"].fillna(25.0)

    # Teammate long-run comparison
    team_fp_sum = df.groupby(["season", "round", "constructor_id"])["fp_longrun_gap_pct"].transform("sum")
    team_fp_n = df.groupby(["season", "round", "constructor_id"])["fp_longrun_gap_pct"].transform("count")
    df["fp_vs_teammate"] = np.where(
        team_fp_n > 1,
        df["fp_longrun_gap_pct"] - (team_fp_sum - df["fp_longrun_gap_pct"]) / (team_fp_n - 1).clip(lower=1),
        0.0)

    df["grid_position"] = df["grid_position"].replace(0, np.nan)
    df["pit_lane_start"] = df["grid_position"].isna().astype(int)
    n_in_race = df.groupby(["season", "round"])["driver_id"].transform("count")
    df["qualifying_position"] = df["qualifying_position"].fillna(
        df["grid_position"]).fillna(n_in_race)
    df["grid_position"] = df["grid_position"].fillna(n_in_race)
    df["grid_penalty"] = df["grid_position"] - df["qualifying_position"]
    df["dnf"] = (~df["status"].map(_is_finished)).astype(int)
    df["pos_gain"] = df["grid_position"] - df["finish_position"]

    df = df.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    df["race_id"] = df["season"].astype(str) + "_" + df["round"].astype(str).str.zfill(2)
    df["race_idx"] = df.groupby(["season", "round"]).ngroup()

    # ---------- Elo (pre-race) ----------
    for pos_col, pref in [("finish_position", "elo"), ("qualifying_position", "qelo")]:
        d = run_elo(df, pos_col, f"driver_{pref}", "driver_id")
        df = df.merge(d, on=["season", "round", "driver_id"], how="left")
        c = run_elo(df, pos_col, f"team_{pref}", "constructor_id", k=12.0)
        df = df.merge(c, on=["season", "round", "constructor_id"], how="left")
    df["driver_vs_car_elo"] = df["driver_elo"] - df["team_elo"]
    df["driver_vs_car_qelo"] = df["driver_qelo"] - df["team_qelo"]

    # ---------- driver rolling form (shifted -> no leakage) ----------
    g = df.groupby("driver_id", sort=False)

    def roll(col, w, how="mean"):
        return g[col].transform(
            lambda s: getattr(s.shift(1).rolling(w, min_periods=1), how)())

    df["drv_fin_l3"] = roll("finish_position", 3)
    df["drv_fin_l5"] = roll("finish_position", 5)
    df["drv_fin_l10"] = roll("finish_position", 10)
    df["drv_pts_l5"] = roll("points", 5)
    df["drv_gain_l5"] = roll("pos_gain", 5)
    df["drv_dnf_l10"] = roll("dnf", 10)
    df["drv_quali_l5"] = roll("qualifying_position", 5)
    df["drv_career_races"] = g.cumcount()

    # ---------- constructor rolling form ----------
    team_race = (df.groupby(["constructor_id", "season", "round"], as_index=False)
                 .agg(t_fin=("finish_position", "mean"),
                      t_pts=("points", "sum"),
                      t_dnf=("dnf", "mean"),
                      t_q=("qualifying_position", "mean"))
                 .sort_values(["season", "round"]))
    tg = team_race.groupby("constructor_id", sort=False)
    team_race["team_fin_l5"] = tg["t_fin"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    team_race["team_pts_l5"] = tg["t_pts"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    team_race["team_dnf_l10"] = tg["t_dnf"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    team_race["team_quali_l5"] = tg["t_q"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df = df.merge(
        team_race[["constructor_id", "season", "round", "team_fin_l5",
                   "team_pts_l5", "team_dnf_l10", "team_quali_l5"]],
        on=["constructor_id", "season", "round"], how="left")

    # ---------- teammate comparison (pre-race: this race's qualifying) ----------
    team_q_sum = df.groupby(["season", "round", "constructor_id"])[
        "qualifying_position"].transform("sum")
    team_n = df.groupby(["season", "round", "constructor_id"])[
        "qualifying_position"].transform("count")
    df["quali_vs_teammate"] = np.where(
        team_n > 1,
        df["qualifying_position"] - (team_q_sum - df["qualifying_position"]) / (team_n - 1).clip(lower=1),
        0.0)

    # ---------- circuit history ----------
    df["drv_circuit_avg_fin"] = df.groupby(["driver_id", "circuit_id"])[
        "finish_position"].transform(lambda s: s.shift(1).expanding().mean())
    race_move = (df.groupby(["circuit_id", "season", "round"], as_index=False)
                 .agg(move=("pos_gain", lambda s: s.abs().mean()))
                 .sort_values(["season", "round"]))
    race_move["circuit_overtaking"] = race_move.groupby("circuit_id")["move"].transform(
        lambda s: s.shift(1).expanding().mean())
    df = df.merge(race_move[["circuit_id", "season", "round", "circuit_overtaking"]],
                  on=["circuit_id", "season", "round"], how="left")

    # ---------- within-race relative features ----------
    grp = df.groupby(["season", "round"])
    df["grid_rank_pct"] = df["grid_position"] / grp["grid_position"].transform("max")
    df["elo_rank"] = grp["driver_elo"].rank(ascending=False)
    df["team_elo_rank"] = grp["team_elo"].rank(ascending=False)
    df["combined_elo"] = df["driver_elo"] + df["team_elo"]
    df["combined_elo_rank"] = grp["combined_elo"].rank(ascending=False)

    df["podium"] = (df["finish_position"] <= 3).astype(int)
    df["win"] = (df["finish_position"] == 1).astype(int)

    if save:
        out = PROCESSED / "model_dataset_v2.csv"
        df.to_csv(out, index=False)
        print(f"saved {out}  rows={len(df)}  cols={df.shape[1]}")
    return df


FEATURES = [
    "grid_position", "qualifying_position", "grid_penalty", "pit_lane_start",
    "quali_gap_pct", "q1_gap", "reached_q2", "reached_q3", "quali_vs_teammate",
    "grid_rank_pct",
    "driver_elo", "driver_qelo", "team_elo", "team_qelo",
    "driver_vs_car_elo", "driver_vs_car_qelo", "elo_rank", "team_elo_rank",
    "combined_elo", "combined_elo_rank",
    "drv_fin_l3", "drv_fin_l5", "drv_fin_l10", "drv_pts_l5", "drv_gain_l5",
    "drv_dnf_l10", "drv_quali_l5", "drv_career_races",
    "team_fin_l5", "team_pts_l5", "team_dnf_l10", "team_quali_l5",
    "drv_circuit_avg_fin", "circuit_overtaking",
    "fp_longrun_gap_pct", "fp_race_vs_quali_pace", "fp_vs_teammate",
    "fp_has_data", "fp_laps_count",
]

if __name__ == "__main__":
    build()
