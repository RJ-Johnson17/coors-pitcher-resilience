"""
Coors Pitcher Project — Part 3: one Coors score per pitcher.

Combines Part 2's per-pitch predictions into a single number per pitcher:
how much movement his whole arsenal is projected to lose at Coors, compared
with the average MLB pitcher.

How it works
  1. Each pitch type gets a projected Coors movement change.
     - No real Coors history: use the Part 2 model prediction.
     - Some Coors history: blend actual and predicted, trusting the actual
       result more as Coors pitches pile up:
         blended = (n_coors * actual + K * predicted) / (n_coors + K)
  2. Weight each pitch type by how often he throws it (usage away from Coors).
  3. Coors score = his projected change minus the league-average pitcher's.
     Positive = his arsenal keeps more movement at Coors than average.
  4. The score splits into two parts that add up exactly:
     - Mix:   what his pitch selection alone implies (sliders good, sinkers bad)
     - Stuff: how his versions of those pitches compare with typical ones

Reads:   coors.duckdb, output/all_pitch_predictions.csv (from Part 2)
Writes:  output/pitcher_scores.csv, output/chart7_coors_scores.png,
         app_data/pitcher_scores.csv, app_data/pitch_level.csv (for the app)

Run:  python part3_score_pitchers.py
"""

from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DB_PATH = "coors.duckdb"
OUT = Path("output")
APP = Path("app_data")

BLEND_K = 100           # Coors pitches at which actual and predicted count equally
MIN_2026_PITCHES = 300  # to appear on the chart / default leaderboard

PITCH_NAMES = {
    "FF": "Four-seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider",
    "ST": "Sweeper", "CU": "Curveball", "KC": "Knuckle curve", "SV": "Slurve",
    "CH": "Changeup", "FS": "Splitter",
}

INFO_SQL = """
WITH t AS (
    SELECT pitcher, game_year, pitching_team, COUNT(*) AS n, MAX(gdate) AS last_game
    FROM pitches GROUP BY ALL
),
basic AS (
    SELECT pitcher,
           arg_max(pitching_team, last_game)                    AS team,
           COALESCE(SUM(n) FILTER (WHERE game_year = 2026), 0)  AS pitches_2026,
           SUM(n)                                               AS pitches_total
    FROM t GROUP BY pitcher
),
-- starter = whoever threw his team's first pitch of the game
starters AS (
    SELECT game_year, game_pk, pitching_team,
           arg_min(pitcher, inning * 100000 + at_bat_number * 100 + pitch_number) AS pitcher
    FROM pitches GROUP BY ALL
),
gs AS (
    SELECT pitcher, COUNT(*) FILTER (WHERE game_year = 2026) AS gs_2026
    FROM starters GROUP BY pitcher
),
g AS (
    SELECT pitcher, COUNT(DISTINCT game_pk) FILTER (WHERE game_year = 2026) AS g_2026
    FROM pitches GROUP BY pitcher
)
SELECT b.*, h.throws, o.n_other_all,
       COALESCE(gs.gs_2026, 0) AS gs_2026,
       COALESCE(g.g_2026, 0)   AS g_2026
FROM basic b
JOIN (SELECT pitcher, mode(p_throws) AS throws FROM pitches GROUP BY pitcher) h
  ON h.pitcher = b.pitcher
JOIN (SELECT pitcher, COUNT(*) AS n_other_all FROM pitches WHERE NOT is_coors
      GROUP BY pitcher) o
  ON o.pitcher = b.pitcher
LEFT JOIN gs ON gs.pitcher = b.pitcher
LEFT JOIN g  ON g.pitcher  = b.pitcher
"""


def ip_to_outs(ip) -> float:
    """MLB writes innings as 45.2 meaning 45 and 2/3. Convert to outs."""
    try:
        whole, _, frac = str(ip).partition(".")
        return int(whole) * 3 + (int(frac) if frac else 0)
    except (ValueError, TypeError):
        return np.nan


def fetch_mlb_stats(season: int = 2026) -> pd.DataFrame:
    """2026 ERA and innings from the official MLB Stats API (same player IDs as Statcast).
    Returns an empty table if the site can't be reached; the rest still works."""
    try:
        import requests
        r = requests.get("https://statsapi.mlb.com/api/v1/stats", timeout=30, params={
            "stats": "season", "group": "pitching", "season": season,
            "sportId": 1, "playerPool": "ALL", "limit": 5000})
        r.raise_for_status()
        rows = []
        for block in r.json().get("stats", []):
            for s in block.get("splits", []):
                stat, player = s.get("stat", {}), s.get("player", {})
                if player.get("id") is None:
                    continue
                rows.append({"pitcher": int(player["id"]),
                             "combined": "team" not in s,  # traded players may have per-team rows
                             "outs": ip_to_outs(stat.get("inningsPitched")),
                             "er": pd.to_numeric(stat.get("earnedRuns"), errors="coerce")})
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("no rows returned")
        # use a player's combined row if there is one, otherwise add up his team rows
        has_combined = df.groupby("pitcher")["combined"].transform("any")
        df = df[df["combined"] | ~has_combined]
        df = df.groupby("pitcher", as_index=False)[["outs", "er"]].sum()
        df["ip_2026"] = (df["outs"] // 3) + (df["outs"] % 3) / 10   # back to 45.2 style
        df["era_2026"] = np.where(df["outs"] > 0, 27 * df["er"] / df["outs"], np.nan)
        print(f"Got 2026 ERA/IP for {len(df):,} pitchers from MLB Stats API.")
        return df[["pitcher", "ip_2026", "era_2026"]]
    except Exception as e:
        print(f"   Could not get ERA/IP from MLB Stats API ({e}); those columns will be blank.")
        return pd.DataFrame(columns=["pitcher", "ip_2026", "era_2026"])


def main():
    OUT.mkdir(exist_ok=True)
    APP.mkdir(exist_ok=True)

    pred = pd.read_csv(OUT / "all_pitch_predictions.csv")
    con = duckdb.connect(DB_PATH, read_only=True)
    info = con.execute(INFO_SQL).df()
    con.close()

    # ---- 1. projected change per pitch type ----------------------------
    has_actual = pred["actual_total_diff"].notna()
    n_c = pred["n_coors"].where(has_actual, 0)
    pred["blended_diff"] = np.where(
        has_actual,
        (n_c * pred["actual_total_diff"].fillna(0) + BLEND_K * pred["predicted_total_diff"])
        / (n_c + BLEND_K),
        pred["predicted_total_diff"],
    )

    # ---- 2. usage weights ----------------------------------------------
    pred["usage"] = pred["n_other"] / pred.groupby("pitcher")["n_other"].transform("sum")
    pred["pitch_name"] = pred["pitch_type"].map(PITCH_NAMES).fillna(pred["pitch_type"])

    # ---- 3/4. pitcher-level score and its two components ---------------
    pred["w_blended"] = pred["usage"] * pred["blended_diff"]
    pred["w_type_avg"] = pred["usage"] * pred["type_avg_diff"]
    agg = pred.groupby(["pitcher", "pitcher_name"], as_index=False).agg(
        projected_change_in=("w_blended", "sum"),
        mix_expected_in=("w_type_avg", "sum"),
        coors_pitches=("n_coors", "sum"),
        modeled_pitches=("n_other", "sum"),
    )
    mix_str = (pred.sort_values("usage", ascending=False)
               .groupby("pitcher")
               .apply(lambda g: ", ".join(f"{t} {u:.0%}" for t, u in
                                          zip(g["pitch_type"], g["usage"]) if u >= 0.05),
                      include_groups=False)
               .rename("pitch_mix"))
    scores = agg.merge(mix_str, on="pitcher").merge(info, on="pitcher", how="left")
    scores = scores.merge(fetch_mlb_stats(2026), on="pitcher", how="left")
    # starter if he started at least half of his 2026 games
    scores["role"] = np.select(
        [scores["g_2026"].fillna(0) == 0,
         scores["gs_2026"].fillna(0) >= 0.5 * scores["g_2026"].fillna(0)],
        ["—", "SP"], default="RP")
    scores["coverage"] = scores["modeled_pitches"] / scores["n_other_all"]

    regulars = scores["pitches_2026"] >= MIN_2026_PITCHES
    league_avg = scores.loc[regulars, "projected_change_in"].mean()
    if np.isnan(league_avg):
        league_avg = scores["projected_change_in"].mean()
    scores["coors_score_in"] = scores["projected_change_in"] - league_avg
    scores["mix_component_in"] = scores["mix_expected_in"] - league_avg
    scores["stuff_component_in"] = scores["projected_change_in"] - scores["mix_expected_in"]
    scores["rank_2026"] = (scores["coors_score_in"].where(regulars)
                           .rank(ascending=False, method="min"))
    scores = scores.sort_values("coors_score_in", ascending=False)
    scores["pitches_2026"] = scores["pitches_2026"].fillna(0).astype(int)
    regulars = scores["pitches_2026"] >= MIN_2026_PITCHES  # realign after sorting

    keep = ["pitcher", "pitcher_name", "team", "throws", "role", "gs_2026", "g_2026",
            "ip_2026", "era_2026", "pitches_2026", "coors_pitches",
            "pitch_mix", "coverage", "projected_change_in", "coors_score_in",
            "mix_component_in", "stuff_component_in", "rank_2026"]
    scores[keep].to_csv(OUT / "pitcher_scores.csv", index=False)
    scores[keep].to_csv(APP / "pitcher_scores.csv", index=False)

    pitch_cols = ["pitcher", "pitcher_name", "pitch_type", "pitch_name", "usage", "n_other",
                  "n_coors", "velo", "spin", "total_mov", "mov_per_krpm",
                  "predicted_total_diff", "actual_total_diff", "blended_diff", "type_avg_diff"]
    pitch_cols = [c for c in pitch_cols if c in pred.columns]
    pred[pitch_cols].to_csv(APP / "pitch_level.csv", index=False)

    # ---- chart 7 -------------------------------------------------------
    pool = scores[regulars]
    if len(pool) >= 6:
        n = min(15, len(pool) // 2)
        d = pd.concat([pool.head(n), pool.tail(n)]).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 0.36 * len(d) + 2))
        y = np.arange(len(d))
        ax.barh(y, d["mix_component_in"], color="#9aa5b1", label="Pitch mix")
        ax.barh(y, d["stuff_component_in"], left=np.where(
            np.sign(d["stuff_component_in"]) == np.sign(d["mix_component_in"]),
            d["mix_component_in"], 0), color="#2a7de1", label="Pitch characteristics")
        ax.scatter(d["coors_score_in"], y, color="#111", s=18, zorder=3, label="Coors score")
        ax.set_yticks(y, [f"{r.pitcher_name} ({r.team})" for r in d.itertuples()], fontsize=8)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_title("Projected Coors resilience, 2026 pitchers", loc="left",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("inches of movement kept at Coors vs. average pitcher")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.text(0.01, 0.005, f"Top and bottom {n} of pitchers with {MIN_2026_PITCHES}+ pitches "
                 "in 2026. Black dot = total score (mix + characteristics).",
                 fontsize=8, color="#555")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(OUT / "chart7_coors_scores.png", dpi=160)
        plt.close(fig)

    # ---- summary -------------------------------------------------------
    fmt = {c: "{:+.2f}".format for c in ["coors_score_in", "mix_component_in",
                                         "stuff_component_in"]}
    fmt["era_2026"] = lambda v: "—" if pd.isna(v) else f"{v:.2f}"
    fmt["ip_2026"] = lambda v: "—" if pd.isna(v) else f"{v:.1f}"
    cols = ["pitcher_name", "team", "role", "ip_2026", "era_2026", "pitch_mix",
            "coors_score_in", "mix_component_in", "stuff_component_in"]
    pd.set_option("display.width", 180)
    pd.set_option("display.max_colwidth", 34)
    print(f"\nScored {len(scores):,} pitchers; {int(regulars.sum()):,} with "
          f"{MIN_2026_PITCHES}+ pitches in 2026 "
          f"({int((pool['role'] == 'SP').sum())} starters, "
          f"{int((pool['role'] == 'RP').sum())} relievers).")
    print(f"League-average projected movement change at Coors: {league_avg:+.2f} in")
    for role, name in [("SP", "starters"), ("RP", "relievers")]:
        sub = pool[pool["role"] == role]
        print(f"\nMost Coors-resilient {name} (2026):")
        print(sub.head(10)[cols].to_string(index=False, formatters=fmt))
        print(f"\nLeast Coors-resilient {name}:")
        print(sub.tail(5)[cols].to_string(index=False, formatters=fmt))
    print(f"\nSaved output/pitcher_scores.csv, chart7, and app data in {APP.resolve()}")


if __name__ == "__main__":
    main()
