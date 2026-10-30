"""
Coors Pitcher Project — Part 1: measure the Coors effect on pitch movement.

Reads coors.duckdb (built by pull_statcast.py) and writes to an output/ folder:
  chart1_movement_by_pitch_type.png  league-wide movement change at Coors
  chart2_rockies_vs_visitors.png     same, split Rockies pitchers vs. visitors
  chart3_resilience_leaders.png      pitchers who held movement best / worst
  pitch_type_summary.csv             numbers behind chart 1
  rockies_vs_visitors.csv            numbers behind chart 2
  pitcher_resilience.csv             full pitcher leaderboard
  pitcher_pitch_splits.csv           every pitcher x pitch type comparison

Method
  Each pitcher is compared with himself: his pitches at Coors vs. the same
  pitch type everywhere else. Averages are weighted by Coors pitch count.
  "Resilience" = how much movement a pitcher kept at Coors compared with what
  is typical for his pitch mix, shrunk toward zero for small samples.

Run:  python part1_coors_effect.py
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

MIN_COORS_PITCHES = 25      # per pitcher x pitch type, to be included at all
MIN_PITCHERS_PER_TYPE = 15  # pitch types with fewer pitchers are left off charts
MIN_LEADERBOARD = 100       # total Coors pitches to appear on the leaderboard
SHRINK_K = 100              # shrinkage strength (pitches); bigger = more cautious

PITCH_NAMES = {
    "FF": "Four-seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider",
    "ST": "Sweeper", "CU": "Curveball", "KC": "Knuckle curve", "SV": "Slurve",
    "CH": "Changeup", "FS": "Splitter", "FO": "Forkball", "SC": "Screwball",
    "KN": "Knuckleball", "CS": "Slow curve",
}
HIGHLIGHT = {"SL": "#2a7de1", "ST": "#e1662a"}  # slider vs. sweeper
BASE_COLOR = "#9aa5b1"

DIFF_COLS = ["total_diff", "ivb_diff", "hb_diff", "velo_diff", "spin_diff", "whiff_diff"]

SPLITS_SQL = """
WITH base AS (
    SELECT pitcher, pitch_type, is_coors, pitching_team,
           release_speed, release_spin_rate, hmov_in, ivb_in, is_whiff, is_swing
    FROM pitches
),
grp AS (   -- "Rockies pitcher" = most of his Coors pitches were thrown for COL
    SELECT pitcher, AVG((pitching_team = 'COL')::INT) >= 0.5 AS is_rockies
    FROM base WHERE is_coors GROUP BY pitcher
),
agg AS (
    SELECT pitcher, pitch_type, is_coors,
           COUNT(*)                 AS n,
           AVG(release_speed)       AS velo,
           AVG(release_spin_rate)   AS spin,
           ABS(AVG(hmov_in))        AS hb,    -- size of horizontal break, inches
           AVG(ivb_in)              AS ivb,   -- induced vertical break, inches
           SUM(is_whiff::INT) * 1.0 / NULLIF(SUM(is_swing::INT), 0) AS whiff
    FROM base
    GROUP BY ALL
)
SELECT c.pitcher,
       COALESCE(p.pitcher_name, CAST(c.pitcher AS VARCHAR)) AS pitcher_name,
       c.pitch_type, g.is_rockies,
       c.n AS n_coors, o.n AS n_other,
       c.velo - o.velo AS velo_diff,
       c.spin - o.spin AS spin_diff,
       c.hb   - o.hb   AS hb_diff,
       c.ivb  - o.ivb  AS ivb_diff,
       sqrt(c.hb * c.hb + c.ivb * c.ivb) - sqrt(o.hb * o.hb + o.ivb * o.ivb) AS total_diff,
       c.whiff - o.whiff AS whiff_diff
FROM agg c
JOIN agg o ON c.pitcher = o.pitcher AND c.pitch_type = o.pitch_type
          AND c.is_coors AND NOT o.is_coors
JOIN grp g ON g.pitcher = c.pitcher
LEFT JOIN pitchers p ON p.pitcher = c.pitcher
"""


def wavg(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna()
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(keys):
        row = dict(zip(keys, key))
        row["pitchers"] = len(g)
        row["coors_pitches"] = int(g["n_coors"].sum())
        for col in DIFF_COLS:
            row[col] = wavg(g[col], g["n_coors"])
        rows.append(row)
    return pd.DataFrame(rows)


def label(pt: str) -> str:
    return f"{PITCH_NAMES.get(pt, pt)} ({pt})"


def style(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.axvline(0, color="#333", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)


def chart_pitch_types(by_type: pd.DataFrame):
    d = by_type.sort_values("total_diff")
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * len(d) + 2), sharey=True)

    colors = [HIGHLIGHT.get(pt, BASE_COLOR) for pt in d["pitch_type"]]
    ylabels = [label(pt) for pt in d["pitch_type"]]
    y = np.arange(len(d))

    ax = axes[0]
    ax.barh(y, d["total_diff"], color=colors)
    for yi, v, n in zip(y, d["total_diff"], d["pitchers"]):
        ax.text(v - 0.05, yi, f"{v:+.2f}\"  (n={n})", va="center", ha="right", fontsize=9)
    lo = min(d["total_diff"].min(), 0)
    ax.set_xlim(lo * 1.45 - 0.2, max(d["total_diff"].max(), 0) + 0.3)  # room for labels
    ax.set_yticks(y, ylabels)
    style(ax, "Total movement change at Coors", "inches vs. same pitcher elsewhere")

    ax = axes[1]
    h = 0.38
    ax.barh(y - h / 2, d["ivb_diff"], h, label="Vertical (IVB)", color="#4c5b6b")
    ax.barh(y + h / 2, d["hb_diff"], h, label="Horizontal", color="#b8c2cc")
    style(ax, "Where the movement goes", "inches")
    ax.legend(frameon=False, loc="lower right")

    fig.suptitle("Every pitch type moves less at Coors — but not equally",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.005, "Within-pitcher comparison, weighted by Coors pitches. "
             f"Pitcher x pitch type needs {MIN_COORS_PITCHES}+ Coors pitches. "
             "Curveball IVB is negative, so a positive change means less drop.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "chart1_movement_by_pitch_type.png", dpi=160)
    plt.close(fig)


def chart_groups(by_group: pd.DataFrame, order: list[str]):
    piv = by_group.pivot(index="pitch_type", columns="group", values="total_diff")
    cnt = by_group.pivot(index="pitch_type", columns="group", values="pitchers")
    keep = [pt for pt in order if pt in piv.index and (cnt.loc[pt] >= 5).all()]
    piv, cnt = piv.loc[keep], cnt.loc[keep]

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(piv) + 2))
    y = np.arange(len(piv))
    h = 0.38
    for i, (grp, color) in enumerate([("Rockies", "#33006f"), ("Visitors", "#b8c2cc")]):
        if grp in piv:
            ax.barh(y + (i - 0.5) * h, piv[grp], h, label=grp, color=color)
    ax.set_yticks(y, [label(pt) for pt in piv.index])
    style(ax, "Rockies pitchers vs. visiting pitchers",
          "total movement change at Coors (inches)")
    ax.legend(frameon=False, loc="lower right")
    fig.text(0.01, 0.005, "Only pitch types with 5+ pitchers in both groups.",
             fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "chart2_rockies_vs_visitors.png", dpi=160)
    plt.close(fig)


def pitcher_resilience(splits: pd.DataFrame, by_type: pd.DataFrame) -> pd.DataFrame:
    expected = by_type.set_index("pitch_type")["total_diff"]
    s = splits[splits["pitch_type"].isin(expected.index)].copy()
    # positive residual = kept more movement than is typical for that pitch type
    s["resid"] = s["total_diff"] - s["pitch_type"].map(expected)

    rows = []
    for (pid, name, rockies), g in s.groupby(["pitcher", "pitcher_name", "is_rockies"]):
        n = int(g["n_coors"].sum())
        raw = wavg(g["resid"], g["n_coors"])
        main = g.sort_values("n_coors", ascending=False)["pitch_type"].head(3)
        rows.append({
            "pitcher": pid, "pitcher_name": name,
            "group": "Rockies" if rockies else "Visitors",
            "coors_pitches": n,
            "pitch_types": ", ".join(main),
            "raw_vs_expected_in": raw,
            "resilience_in": raw * n / (n + SHRINK_K),
            "total_diff_in": wavg(g["total_diff"], g["n_coors"]),
        })
    return pd.DataFrame(rows).sort_values("resilience_in", ascending=False)


def chart_leaders(res: pd.DataFrame):
    pool = res[res["coors_pitches"] >= MIN_LEADERBOARD]
    if len(pool) < 4:
        print("   not enough pitchers for the leaderboard chart; skipping")
        return
    n = min(10, len(pool) // 2)
    top, bottom = pool.head(n), pool.tail(n)
    d = pd.concat([top, bottom]).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 2))
    y = np.arange(len(d))
    colors = ["#2f8f5b" if v > 0 else "#c0392b" for v in d["resilience_in"]]
    ax.barh(y, d["resilience_in"], color=colors)
    ax.set_yticks(y, [f"{r.pitcher_name}  ({r.pitch_types})" for r in d.itertuples()],
                  fontsize=9)
    style(ax, "Who keeps their stuff at Coors?",
          "inches of movement kept vs. expected for pitch mix")
    fig.text(0.01, 0.005, f"Pitchers with {MIN_LEADERBOARD}+ Coors pitches, 2025-26. "
             "Shrunk toward 0 for smaller samples.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "chart3_resilience_leaders.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    con = duckdb.connect(DB_PATH, read_only=True)
    splits = con.execute(SPLITS_SQL).df()
    con.close()
    splits.to_csv(OUT / "pitcher_pitch_splits.csv", index=False)

    qual = splits[splits["n_coors"] >= MIN_COORS_PITCHES].copy()
    qual["group"] = np.where(qual["is_rockies"], "Rockies", "Visitors")

    by_type = summarize(qual, ["pitch_type"])
    by_type = by_type[by_type["pitchers"] >= MIN_PITCHERS_PER_TYPE].sort_values("total_diff")
    by_type.to_csv(OUT / "pitch_type_summary.csv", index=False)

    by_group = summarize(qual[qual["pitch_type"].isin(by_type["pitch_type"])],
                         ["pitch_type", "group"])
    by_group.to_csv(OUT / "rockies_vs_visitors.csv", index=False)

    res = pitcher_resilience(qual, by_type)
    res.to_csv(OUT / "pitcher_resilience.csv", index=False)

    chart_pitch_types(by_type)
    chart_groups(by_group, list(by_type["pitch_type"]))
    chart_leaders(res)

    pd.set_option("display.width", 140)
    fmt = {c: "{:+.2f}".format for c in DIFF_COLS + ["resilience_in", "raw_vs_expected_in"]}
    print("\nMovement change at Coors by pitch type (inches, weighted):")
    print(by_type[["pitch_type", "pitchers", "coors_pitches", "total_diff", "ivb_diff",
                   "hb_diff", "whiff_diff"]].to_string(index=False, formatters=fmt))

    print("\nRockies vs. visitors (total movement change):")
    print(by_group.pivot(index="pitch_type", columns="group", values="total_diff")
          .reindex(by_type["pitch_type"]).round(2).to_string())

    pool = res[res["coors_pitches"] >= MIN_LEADERBOARD]
    cols = ["pitcher_name", "group", "coors_pitches", "pitch_types", "resilience_in"]
    print(f"\nMost resilient ({MIN_LEADERBOARD}+ Coors pitches):")
    print(pool.head(10)[cols].to_string(index=False, formatters=fmt))
    print("\nLeast resilient:")
    print(pool.tail(10)[cols].to_string(index=False, formatters=fmt))

    print(f"\nCharts and CSVs saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
