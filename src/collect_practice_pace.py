"""Extract representative practice (FP2 / FP1 fallback) pace from FastF1.

Captures long-run race trim pace that qualifying times conceal.
Saves to data/raw/fastf1_practice_pace.csv with incremental caching.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "data" / "fastf1_cache"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Configure FastF1 cache
try:
    fastf1.Cache.enable_cache(str(CACHE_DIR))
except Exception:
    pass


def extract_session_pace(year: int, round_num: int):
    """Load practice session (FP2 or FP1 fallback) and compute long-run metrics."""
    session = None
    session_name = "FP2"
    try:
        session = fastf1.get_session(year, round_num, "FP2")
        session.load(telemetry=False, weather=False)
    except Exception:
        try:
            session = fastf1.get_session(year, round_num, "FP1")
            session.load(telemetry=False, weather=False)
            session_name = "FP1"
        except Exception as e:
            return []

    if session is None or session.laps is None or len(session.laps) == 0:
        return []

    laps = session.laps.copy()
    try:
        accurate_laps = laps.pick_accurate()
    except Exception:
        accurate_laps = laps[laps["LapTime"].notna()]

    if len(accurate_laps) == 0:
        return []

    # Calculate single lap flying pace
    single_lap = accurate_laps.groupby("Driver")["LapTime"].apply(
        lambda s: s.dt.total_seconds().min()
    ).dropna()

    best_single = single_lap.min() if len(single_lap) > 0 else np.nan

    # Calculate long-run pace: stints with >= 3 consecutive accurate laps
    stints = accurate_laps.groupby(["Driver", "Stint"]).filter(lambda x: len(x) >= 3)
    longruns = stints.groupby("Driver")["LapTime"].apply(
        lambda s: s.dt.total_seconds().median()
    ).dropna()

    best_longrun = longruns.min() if len(longruns) > 0 else np.nan

    rows = []
    drivers = set(accurate_laps["Driver"].unique()) | set(session.results["Abbreviation"].dropna())
    for drv in drivers:
        drv_str = str(drv).strip().upper()
        if not drv_str:
            continue

        lap_count = int((laps["Driver"] == drv).sum())
        
        # Long run gap %
        lr_gap = np.nan
        if drv in longruns and pd.notna(best_longrun) and best_longrun > 0:
            lr_gap = (longruns[drv] - best_longrun) / best_longrun * 100.0

        # Quali sim gap %
        qs_gap = np.nan
        if drv in single_lap and pd.notna(best_single) and best_single > 0:
            qs_gap = (single_lap[drv] - best_single) / best_single * 100.0

        # Race vs Quali pace bias (negative = better race pace than single lap pace)
        race_bias = np.nan
        if pd.notna(lr_gap) and pd.notna(qs_gap):
            race_bias = lr_gap - qs_gap

        rows.append({
            "season": year,
            "round": round_num,
            "driver_code": drv_str,
            "fp_session": session_name,
            "fp_laps_count": lap_count,
            "fp_longrun_gap_pct": lr_gap,
            "fp_quali_sim_gap_pct": qs_gap,
            "fp_race_vs_quali_pace": race_bias,
        })
    return rows


def collect_practice_pace(start_year=2024, end_year=2026, max_workers=4, force=False):
    """Collect practice pace features across specified seasons."""
    output_path = RAW_DIR / "fastf1_practice_pace.csv"
    existing_rows = []
    cached_rounds = set()

    if not force and output_path.exists():
        try:
            old_df = pd.read_csv(output_path)
            existing_rows = old_df.to_dict("records")
            cached_rounds = set(zip(old_df["season"].astype(int), old_df["round"].astype(int)))
            print(f"Loaded existing practice pace: {len(cached_rounds)} races cached.")
        except Exception as e:
            print(f"Warning reading existing practice pace: {e}")

    # Determine races to fetch from races.csv
    races_csv = RAW_DIR / "races.csv"
    if not races_csv.exists():
        raise FileNotFoundError("races.csv not found in data/raw")

    races = pd.read_csv(races_csv)
    sub = races[(races["season"] >= start_year) & (races["season"] <= end_year)]
    targets = []
    for _, r in sub.iterrows():
        y, rnd = int(r["season"]), int(r["round"])
        if (y, rnd) not in cached_rounds or force:
            targets.append((y, rnd))

    print(f"Need to fetch practice pace for {len(targets)} races ({start_year}-{end_year})...")
    if not targets:
        return pd.DataFrame(existing_rows)

    new_rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(extract_session_pace, y, r): (y, r) for y, r in targets}
        for future in as_completed(future_map):
            y, r = future_map[future]
            try:
                res = future.result()
                if res:
                    new_rows.extend(res)
                    print(f"Fetched {y} Round {r} ({len(res)} drivers)")
                else:
                    print(f"No practice data for {y} Round {r}")
            except Exception as e:
                print(f"Error fetching {y} Round {r}: {e}")

    all_rows = existing_rows + new_rows
    df = pd.DataFrame(all_rows).drop_duplicates(
        subset=["season", "round", "driver_code"]
    ).sort_values(["season", "round", "driver_code"])

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} practice pace records to {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect FastF1 practice pace metrics")
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    collect_practice_pace(
        start_year=args.start_year,
        end_year=args.end_year,
        max_workers=args.workers,
        force=args.force
    )


if __name__ == "__main__":
    main()
