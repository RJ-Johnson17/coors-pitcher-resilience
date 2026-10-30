"""
Coors Pitcher Project — Step 1: pull Statcast data and build a DuckDB database.

What it does
  1. Downloads pitch-level Statcast data week by week (resumable: finished weeks
     are saved as parquet files and skipped on re-runs).
  2. Loads everything into a DuckDB file (coors.duckdb) with a clean `pitches`
     table, a `pitchers` name lookup, and a starter `pitcher_coors_splits` view.

Usage
  pip install pybaseball duckdb pandas pyarrow
  python pull_statcast.py                      # pull 2025-2026, then build the DB
  python pull_statcast.py --start-year 2026    # 2026 only (quick test)
  python pull_statcast.py --build-only         # rebuild DB from saved files

2026 is the main season; 2025 is included to give visiting pitchers enough
pitches at Coors. If the pull crashes or you stop it, just run it again;
it picks up where it left off.
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

RAW_DIR = Path("data/raw")
DB_PATH = Path("coors.duckdb")

# Regular season roughly falls inside this window every year; non-regular-season
# games are filtered out later using game_type.
SEASON_START = (3, 15)
SEASON_END = (10, 5)

# Columns kept from Statcast (keeps files smaller). Missing ones (e.g. arm_angle
# in older seasons) are added as empty columns so every file has the same shape.
KEEP_COLS = [
    # game context
    "game_date", "game_pk", "game_type", "game_year", "home_team", "away_team",
    "inning", "inning_topbot", "at_bat_number", "pitch_number", "balls", "strikes",
    # who
    "pitcher", "batter", "stand", "p_throws",
    # pitch characteristics
    "pitch_type", "pitch_name", "release_speed", "effective_speed",
    "release_spin_rate", "spin_axis", "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_z", "release_extension", "arm_angle",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "plate_x", "plate_z", "sz_top", "sz_bot",
    # outcomes
    "type", "description", "events", "launch_speed", "launch_angle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "delta_run_exp",
]


def week_ranges(year: int):
    """Yield (start, end) date pairs covering the season window, one week each."""
    start = date(year, *SEASON_START)
    end = date(year, *SEASON_END)
    while start <= end:
        stop = min(start + timedelta(days=6), end)
        yield start, stop
        start = stop + timedelta(days=1)


def pull_week(start: date, end: date, retries: int = 3) -> pd.DataFrame | None:
    from pybaseball import statcast  # imported here so --build-only works without it

    for attempt in range(1, retries + 1):
        try:
            df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat(), verbose=False)
            return df
        except Exception as e:  # network hiccups / Savant timeouts are common
            wait = 20 * attempt
            print(f"   attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    return None


def pull_years(start_year: int, end_year: int):
    try:
        from pybaseball import cache
        cache.enable()
    except Exception:
        pass

    failed = []
    for year in range(start_year, end_year + 1):
        out_dir = RAW_DIR / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        for start, end in week_ranges(year):
            out = out_dir / f"{start.isoformat()}.parquet"
            if out.exists():
                continue
            print(f"Pulling {start} to {end} ...")
            df = pull_week(start, end)
            if df is None:
                print(f"   FAILED {start} to {end} (re-run the script later to retry)")
                failed.append((start, end))
                continue
            if df.empty:
                # Offseason / all-star break edges; save an empty marker so it is skipped
                pd.DataFrame(columns=KEEP_COLS).to_parquet(out, index=False)
                print("   no games")
                continue
            df = df.reindex(columns=KEEP_COLS)
            df.to_parquet(out, index=False)
            print(f"   saved {len(df):,} pitches")
            time.sleep(2)  # be polite to Baseball Savant

    if failed:
        print(f"\n{len(failed)} week(s) failed. Run the script again to retry them.")


def build_db():
    files = list(RAW_DIR.glob("*/*.parquet"))
    if not files:
        raise SystemExit("No parquet files found in data/raw. Run the pull first.")

    con = duckdb.connect(str(DB_PATH))
    print(f"\nBuilding {DB_PATH} from {len(files)} files ...")

    # Main table: regular season only, with derived fields used throughout the project.
    con.execute(f"""
        CREATE OR REPLACE TABLE pitches AS
        SELECT
            *,
            CAST(game_date AS DATE)                              AS gdate,
            home_team = 'COL'                                    AS is_coors,
            -- top of inning = home team pitching
            CASE WHEN inning_topbot = 'Top' THEN home_team
                 ELSE away_team END                              AS pitching_team,
            pfx_x * 12                                           AS hmov_in,
            pfx_z * 12                                           AS ivb_in,
            sqrt(pfx_x * pfx_x + pfx_z * pfx_z) * 12             AS total_mov_in,
            description IN ('swinging_strike', 'swinging_strike_blocked',
                            'foul_tip')                          AS is_whiff,
            description IN ('swinging_strike', 'swinging_strike_blocked',
                            'foul_tip', 'foul', 'foul_bunt', 'hit_into_play',
                            'missed_bunt')                        AS is_swing
        FROM read_parquet('{RAW_DIR.as_posix()}/*/*.parquet', union_by_name = true)
        WHERE game_type = 'R'
          AND pitch_type IS NOT NULL
          AND pfx_x IS NOT NULL
    """)

    # Pitcher name lookup
    ids = [r[0] for r in con.execute("SELECT DISTINCT pitcher FROM pitches").fetchall()]
    try:
        from pybaseball import playerid_reverse_lookup
        names = playerid_reverse_lookup(ids, key_type="mlbam")
        names["pitcher_name"] = (names["name_first"].str.title() + " "
                                 + names["name_last"].str.title())
        names = names.rename(columns={"key_mlbam": "pitcher"})[["pitcher", "pitcher_name"]]
        con.register("names_df", names)
        con.execute("CREATE OR REPLACE TABLE pitchers AS SELECT * FROM names_df")
    except Exception as e:
        print(f"   name lookup skipped ({e}); IDs still work, re-run --build-only later")
        con.execute("CREATE OR REPLACE TABLE pitchers AS "
                    "SELECT DISTINCT pitcher, NULL::VARCHAR AS pitcher_name FROM pitches")

    # Starter view for Part 1: each pitcher's pitch types at Coors vs. elsewhere.
    con.execute("""
        CREATE OR REPLACE VIEW pitcher_coors_splits AS
        WITH agg AS (
            SELECT
                pitcher, pitch_type, p_throws, is_coors,
                COUNT(*)                   AS n,
                AVG(release_speed)         AS velo,
                AVG(release_spin_rate)     AS spin,
                AVG(hmov_in)               AS hmov,
                AVG(ivb_in)                AS ivb,
                AVG(total_mov_in)          AS total_mov,
                SUM(is_whiff::INT) * 1.0 / NULLIF(SUM(is_swing::INT), 0) AS whiff_rate
            FROM pitches
            GROUP BY ALL
        )
        SELECT
            c.pitcher, p.pitcher_name, c.pitch_type, c.p_throws,
            c.n                      AS n_coors,
            o.n                      AS n_other,
            c.velo - o.velo          AS velo_diff,
            c.spin - o.spin          AS spin_diff,
            c.hmov - o.hmov          AS hmov_diff_in,
            c.ivb  - o.ivb           AS ivb_diff_in,
            c.total_mov - o.total_mov AS total_mov_diff_in,
            c.whiff_rate - o.whiff_rate AS whiff_diff
        FROM agg c
        JOIN agg o
          ON c.pitcher = o.pitcher AND c.pitch_type = o.pitch_type
         AND c.is_coors AND NOT o.is_coors
        LEFT JOIN pitchers p ON p.pitcher = c.pitcher
    """)

    # Sanity checks
    print(con.execute("""
        SELECT game_year, COUNT(*) AS pitches, SUM(is_coors::INT) AS coors_pitches
        FROM pitches GROUP BY 1 ORDER BY 1
    """).df().to_string(index=False))

    print("\nLeague-wide movement change at Coors by pitch type (min 25 Coors pitches):")
    print(con.execute("""
        SELECT pitch_type,
               COUNT(*)                          AS pitcher_samples,
               ROUND(AVG(total_mov_diff_in), 2)  AS avg_total_mov_diff_in,
               ROUND(AVG(ivb_diff_in), 2)        AS avg_ivb_diff_in,
               ROUND(AVG(velo_diff), 2)          AS avg_velo_diff
        FROM pitcher_coors_splits
        WHERE n_coors >= 25
        GROUP BY 1
        HAVING COUNT(*) >= 20
        ORDER BY avg_total_mov_diff_in
    """).df().to_string(index=False))

    con.close()
    print(f"\nDone. Open it with: duckdb.connect('{DB_PATH}')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2025)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()

    if not args.build_only:
        pull_years(args.start_year, args.end_year)
    build_db()


if __name__ == "__main__":
    main()
