"""
Coors Pitcher Project — interactive app.

Run locally:   pip install streamlit
               streamlit run app.py

Reads only the small CSVs in app_data/ (made by part3_score_pitchers.py),
so it can be deployed for free on Streamlit Community Cloud.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "app_data"

st.set_page_config(page_title="Coors Pitcher Resilience", layout="wide")


@st.cache_data
def load():
    scores = pd.read_csv(DATA / "pitcher_scores.csv")
    pitches = pd.read_csv(DATA / "pitch_level.csv")
    scores["team"] = scores["team"].fillna("—").astype(str)
    scores["pitcher_name"] = scores["pitcher_name"].astype(str)
    pitches["pitch_name"] = pitches["pitch_name"].astype(str)
    for col, default in [("role", "—"), ("era_2026", float("nan")),
                         ("ip_2026", float("nan")), ("gs_2026", 0), ("g_2026", 0)]:
        if col not in scores:
            scores[col] = default
    scores["role"] = scores["role"].fillna("—").astype(str)
    scores["label"] = (scores["pitcher_name"] + " (" + scores["team"] + ", "
                       + scores["role"] + ")")
    # rank among regulars of the same role (starters vs. starters, relievers vs. relievers)
    ranked = scores["rank_2026"].notna()
    scores["role_rank"] = (scores["coors_score_in"].where(ranked)
                           .groupby(scores["role"]).rank(ascending=False, method="min"))
    return scores, pitches


scores, pitches = load()
n_ranked = int(scores["rank_2026"].notna().sum())

st.title("Who keeps their stuff at Coors?")
st.caption("Projected pitch movement change at Coors Field for every MLB pitcher, "
           "built from 2025–26 Statcast data. Positive score = keeps more movement "
           "than the average pitcher.")

tab_lookup, tab_board, tab_method = st.tabs(["Pitcher lookup", "Leaderboard", "Method"])

# ------------------------------------------------------------------ lookup
with tab_lookup:
    options = scores.sort_values("pitches_2026", ascending=False)
    choice = st.selectbox("Pitcher", options["label"], index=0,
                          help="Type to search. Sorted by 2026 pitch count.")
    p = options[options["label"] == choice].iloc[0]
    arsenal = pitches[pitches["pitcher"] == p["pitcher"]].sort_values("usage", ascending=False)

    role_name = {"SP": "starters", "RP": "relievers"}.get(p["role"], "pitchers")
    n_role = int(((scores["role"] == p["role"]) & scores["rank_2026"].notna()).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coors score", f"{p['coors_score_in']:+.2f} in")
    rank = (f"{int(p['role_rank'])} of {n_role}" if pd.notna(p["role_rank"]) else "unranked")
    c2.metric(f"Rank among 2026 {role_name}", rank)
    c3.metric("Projected movement change", f"{p['projected_change_in']:+.2f} in")
    c4.metric("Coors pitches thrown", f"{int(p['coors_pitches']):,}")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Role", {"SP": "Starter", "RP": "Reliever"}.get(p["role"], "—"))
    d2.metric("2026 ERA", "—" if pd.isna(p["era_2026"]) else f"{p['era_2026']:.2f}")
    d3.metric("2026 IP", "—" if pd.isna(p["ip_2026"]) else f"{p['ip_2026']:.1f}")
    d4.metric("2026 games (starts)", f"{int(p['g_2026'])} ({int(p['gs_2026'])})")

    st.markdown(
        f"**Score breakdown:** pitch mix {p['mix_component_in']:+.2f} in, "
        f"pitch characteristics {p['stuff_component_in']:+.2f} in.  "
        f"Pitch mix: {p['pitch_mix']}."
    )
    if pd.notna(p.get("coverage")) and p["coverage"] < 0.9:
        st.info(f"Only {p['coverage']:.0%} of his pitches are pitch types the model covers; "
                "rarer pitches are left out of the score.")

    chart_df = arsenal.assign(label=arsenal["pitch_name"] + " (" + arsenal["pitch_type"] + ")")
    bars = alt.Chart(chart_df).mark_bar(color="#33006f").encode(
        x=alt.X("blended_diff:Q", title="projected movement change at Coors (in)"),
        y=alt.Y("label:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("label:N", title="Pitch"),
                 alt.Tooltip("usage:Q", format=".0%", title="Usage"),
                 alt.Tooltip("blended_diff:Q", format="+.2f", title="Projected"),
                 alt.Tooltip("type_avg_diff:Q", format="+.2f", title="Typical for type"),
                 alt.Tooltip("n_coors:Q", title="Coors pitches")],
    )
    ticks = alt.Chart(chart_df).mark_tick(color="#e1662a", thickness=3, size=22).encode(
        x="type_avg_diff:Q", y=alt.Y("label:N", sort="-x"))
    st.altair_chart((bars + ticks).properties(height=60 * len(chart_df) + 40),
                    width="stretch")
    st.caption("Bars = this pitcher's projected change. Orange tick = typical change for that "
               "pitch type. A bar shorter than the tick means his version holds up better.")

    table = arsenal[["pitch_name", "usage", "velo", "spin", "mov_per_krpm", "n_coors",
                     "actual_total_diff", "predicted_total_diff", "blended_diff",
                     "type_avg_diff"]].rename(columns={
        "pitch_name": "Pitch", "usage": "Usage", "velo": "Velo", "spin": "Spin",
        "mov_per_krpm": "Mov / 1k rpm", "n_coors": "Coors pitches",
        "actual_total_diff": "Actual at Coors", "predicted_total_diff": "Model",
        "blended_diff": "Projected", "type_avg_diff": "Type avg"})
    st.dataframe(table.style.format({
        "Usage": "{:.0%}", "Velo": "{:.1f}", "Spin": "{:.0f}", "Mov / 1k rpm": "{:.2f}",
        "Actual at Coors": "{:+.2f}", "Model": "{:+.2f}", "Projected": "{:+.2f}",
        "Type avg": "{:+.2f}"}, na_rep="—"), hide_index=True, width="stretch")

# ------------------------------------------------------------- leaderboard
with tab_board:
    f0, f1, f2, f3, f4 = st.columns([1.3, 2, 2, 1, 1])
    role = f0.radio("Role", ["All", "Starters", "Relievers"], horizontal=False)
    min_ip = f1.slider("Minimum 2026 innings", 0, 200, 20, step=5,
                       help="If ERA/IP couldn't be loaded, this filter uses pitch count instead.")
    teams = f2.multiselect("Teams", sorted(scores["team"].unique()))
    throws = f3.selectbox("Throws", ["Both", "R", "L"])
    no_col = f4.checkbox("Exclude Rockies", value=False)

    if scores["ip_2026"].notna().any():
        board = scores[scores["ip_2026"].fillna(0) >= min_ip]
    else:  # fallback: roughly 15 pitches per inning
        board = scores[scores["pitches_2026"] >= min_ip * 15]
    if role != "All":
        board = board[board["role"] == ("SP" if role == "Starters" else "RP")]
    if teams:
        board = board[board["team"].isin(teams)]
    if throws != "Both":
        board = board[board["throws"] == throws]
    if no_col:
        board = board[board["team"] != "COL"]

    st.caption(f"{len(board):,} pitchers. Click a column header to sort.")
    st.dataframe(
        board[["pitcher_name", "team", "role", "throws", "ip_2026", "era_2026", "pitch_mix",
               "coors_score_in", "mix_component_in", "stuff_component_in",
               "coors_pitches"]].rename(columns={
            "pitcher_name": "Pitcher", "team": "Team", "role": "Role", "throws": "T",
            "ip_2026": "2026 IP", "era_2026": "2026 ERA", "pitch_mix": "Pitch mix",
            "coors_score_in": "Coors score", "mix_component_in": "From mix",
            "stuff_component_in": "From stuff", "coors_pitches": "Coors pitches"})
        .style.format({"Coors score": "{:+.2f}", "From mix": "{:+.2f}",
                       "From stuff": "{:+.2f}", "2026 IP": "{:.1f}", "2026 ERA": "{:.2f}"},
                      na_rep="—")
        .background_gradient(subset=["Coors score"], cmap="RdYlGn", vmin=-1.5, vmax=1.5),
        hide_index=True, width="stretch", height=600)

# ------------------------------------------------------------------ method
with tab_method:
    st.markdown("""
**Question.** Thin air at Coors Field weakens the spin-driven part of pitch movement.
Which pitchers' arsenals hold up, including pitchers who have never pitched there?

**Data.** Every regular-season pitch from 2025 and 2026 (Statcast via pybaseball),
stored in DuckDB and queried with SQL.

**Step 1: measure the effect.** Compare each pitcher with himself: the same pitch type at
Coors vs. everywhere else. Every pitch type loses movement. Tight sliders lose the least
and sweepers lose roughly three times as much; four-seamers lose mostly ride, while sinkers
and changeups lose mostly run.

**Step 2: predict it.** Using only how a pitch looks away from Coors (velocity, spin,
movement per 1,000 rpm, release, extension), a weighted regression with pitch-type
baselines predicts Coors movement loss. Validated with cross-validation grouped by
pitcher. Pitches that get more movement per rpm, and higher-spin versions of a pitch,
lose the most, consistent with the physics.

**Step 3: score everyone.** Each pitch type's projection blends the model with actual
Coors results (trusting actual results more as the sample grows), weighted by usage.
The score is relative to the average 2026 pitcher and splits into pitch mix and pitch
characteristics.

**Limitations.**
- Movement is not run prevention: Coors also inflates offense through the large outfield
  and carry on batted balls.
- One or two Coors appearances per visiting pitcher per year keeps samples small.
- Weather and the humidor also affect the ball.
- Rare pitch types are left out when there were too few pitchers to model them.
""")
    st.caption("Built by RJ Johnson · Python, DuckDB, scikit-learn, statsmodels, Streamlit")
