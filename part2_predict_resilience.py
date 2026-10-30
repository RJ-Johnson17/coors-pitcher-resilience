"""
Coors Pitcher Project — Part 2: what predicts who keeps their movement at Coors?

Question
  Using only how a pitch looks AWAY from Coors (velocity, spin, movement,
  release), can we predict how much movement it will lose AT Coors? If so, we
  can score pitchers who have never pitched there (Part 3).

Key idea being tested
  Thin air weakens the spin-driven (Magnus) part of movement. Pitches that get
  a lot of movement per unit of spin are mostly spin-driven, so they should
  lose more. "Movement per 1,000 rpm" is a rough proxy for spin efficiency:
  tight gyro sliders score low, sweepers and ride fastballs score high.

Method
  - One row per pitcher x pitch type with 25+ Coors pitches and 100+ pitches elsewhere.
  - Features are z-scored WITHIN pitch type ("faster than other sliders"),
    plus a baseline for each pitch type.
  - Models compared with 5-fold cross-validation grouped by pitcher, so a
    pitcher is never in both the training and test fold:
      1. Baseline: pitch-type average only
      2. Weighted linear regression
      3. Gradient boosting
  - Rows are weighted by Coors pitches (capped), since small samples are noisy.

Outputs (output/ folder)
  chart4_feature_effects.png       what drives movement loss (regression coefficients)
  chart5_predicted_vs_actual.png   cross-validated predictions vs. reality
  chart6_spin_efficiency.png       movement per rpm vs. movement lost at Coors
  part2_model_comparison.csv       model accuracy table
  part2_coefficients.csv           full regression output
  all_pitch_predictions.csv        predicted Coors movement change for EVERY
                                   pitcher x pitch type (input to Part 3)

Run:  pip install scikit-learn statsmodels
      python part2_predict_resilience.py
"""

from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DB_PATH = "coors.duckdb"
OUT = Path("output")

MIN_COORS = 25          # Coors pitches needed for a row to be used in training
MIN_OTHER = 100         # pitches elsewhere needed to trust the feature averages
MIN_PITCHERS_TYPE = 15  # pitch types with fewer training pitchers are dropped
WEIGHT_CAP = 300        # cap weights so a few Rockies starters don't dominate
MAX_MISSING = 0.30      # drop a feature if more than 30% of rows lack it

FEATURES = {
    "velo": "Velocity",
    "spin": "Spin rate",
    # total movement = movement per rpm x spin, so it is left out to avoid
    # two nearly identical features splitting the effect between them
    "mov_per_krpm": "Movement per 1,000 rpm",
    "extension": "Extension",
    "release_height": "Release height",
    "arm_angle": "Arm angle",
}
PITCH_NAMES = {
    "FF": "Four-seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider",
    "ST": "Sweeper", "CU": "Curveball", "KC": "Knuckle curve", "SV": "Slurve",
    "CH": "Changeup", "FS": "Splitter",
}
TYPE_COLORS = {"SL": "#2a7de1", "ST": "#e1662a", "FF": "#33006f", "SI": "#7a5195",
               "CU": "#2f8f5b", "CH": "#c49a00", "FC": "#888888", "FS": "#bc5090",
               "KC": "#1b5e3b"}

DATA_SQL = """
WITH base AS (
    SELECT pitcher, pitch_type, is_coors, release_speed, release_spin_rate,
           hmov_in, ivb_in, release_extension, release_pos_z, arm_angle
    FROM pitches
    WHERE release_spin_rate IS NOT NULL
),
agg AS (
    SELECT pitcher, pitch_type, is_coors,
           COUNT(*)                AS n,
           AVG(release_speed)      AS velo,
           AVG(release_spin_rate)  AS spin,
           ABS(AVG(hmov_in))       AS hb,
           AVG(ivb_in)             AS ivb,
           AVG(release_extension)  AS extension,
           AVG(release_pos_z)      AS release_height,
           AVG(arm_angle)          AS arm_angle
    FROM base
    GROUP BY ALL
)
SELECT o.pitcher,
       COALESCE(p.pitcher_name, CAST(o.pitcher AS VARCHAR)) AS pitcher_name,
       o.pitch_type,
       o.n                  AS n_other,
       COALESCE(c.n, 0)     AS n_coors,
       o.velo, o.spin, o.extension, o.release_height, o.arm_angle,
       sqrt(o.hb * o.hb + o.ivb * o.ivb) AS total_mov,
       sqrt(c.hb * c.hb + c.ivb * c.ivb) - sqrt(o.hb * o.hb + o.ivb * o.ivb) AS total_diff
FROM agg o
LEFT JOIN agg c ON c.pitcher = o.pitcher AND c.pitch_type = o.pitch_type AND c.is_coors
LEFT JOIN pitchers p ON p.pitcher = o.pitcher
WHERE NOT o.is_coors
"""


# ---------------------------------------------------------------- helpers
def wrmse(y, p, w):
    return float(np.sqrt(mean_squared_error(y, p, sample_weight=w)))


def zscore_within_type(df, feats, stats=None):
    """Z-score each feature within pitch type. Returns (z DataFrame, stats used)."""
    if stats is None:
        stats = df.groupby("pitch_type")[feats].agg(["mean", "std"])
    z = pd.DataFrame(index=df.index)
    for f in feats:
        mu = df["pitch_type"].map(stats[(f, "mean")])
        sd = df["pitch_type"].map(stats[(f, "std")]).replace(0, np.nan)
        z[f] = ((df[f] - mu) / sd).fillna(0.0)  # missing/constant -> average for type
    return z, stats


def design(z, pitch_types, types):
    """Pitch-type indicator columns (no intercept) + z-scored features."""
    dummies = pd.DataFrame({f"type_{t}": (pitch_types == t).astype(float) for t in types},
                           index=z.index)
    return pd.concat([dummies, z], axis=1)


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(DATA_SQL).df()
    con.close()

    df = df[df["n_other"] >= MIN_OTHER].copy()
    df["mov_per_krpm"] = df["total_mov"] / (df["spin"] / 1000.0)

    train = df[(df["n_coors"] >= MIN_COORS) & df["total_diff"].notna()].copy()
    counts = train["pitch_type"].value_counts()
    types = sorted(counts[counts >= MIN_PITCHERS_TYPE].index)
    train = train[train["pitch_type"].isin(types)].reset_index(drop=True)
    everyone = df[df["pitch_type"].isin(types)].reset_index(drop=True)

    feats = [f for f in FEATURES
             if train[f].isna().mean() <= MAX_MISSING and train[f].notna().sum() > 0]
    dropped = [f for f in FEATURES if f not in feats]
    for f in feats:  # fill the occasional gap with the pitch-type median
        med = train.groupby("pitch_type")[f].median()
        train[f] = train[f].fillna(train["pitch_type"].map(med))
        everyone[f] = everyone[f].fillna(everyone["pitch_type"].map(med))

    print(f"Training rows: {len(train)} pitcher x pitch types, "
          f"{train['pitcher'].nunique()} pitchers, pitch types: {', '.join(types)}")
    if dropped:
        print(f"Features dropped for missing data: {', '.join(dropped)}")

    y = train["total_diff"].to_numpy()
    w = np.minimum(train["n_coors"].to_numpy(), WEIGHT_CAP).astype(float)
    groups = train["pitcher"].to_numpy()

    # ---- cross-validation (grouped by pitcher) -------------------------
    n_splits = min(5, train["pitcher"].nunique())
    cv = {"Baseline (pitch type only)": np.zeros(len(y)),
          "Linear regression": np.zeros(len(y)),
          "Gradient boosting": np.zeros(len(y))}
    for tr, te in GroupKFold(n_splits=n_splits).split(train, y, groups):
        trd, ted = train.iloc[tr], train.iloc[te]
        z_tr, stats = zscore_within_type(trd, feats)
        z_te, _ = zscore_within_type(ted, feats, stats)
        X_tr = design(z_tr, trd["pitch_type"], types).to_numpy()
        X_te = design(z_te, ted["pitch_type"], types).to_numpy()

        type_mean = (trd.assign(w=w[tr], wy=w[tr] * y[tr]).groupby("pitch_type")
                     .apply(lambda g: g["wy"].sum() / g["w"].sum(), include_groups=False))
        cv["Baseline (pitch type only)"][te] = ted["pitch_type"].map(type_mean).fillna(
            np.average(y[tr], weights=w[tr])).to_numpy()

        lin = LinearRegression(fit_intercept=False).fit(X_tr, y[tr], sample_weight=w[tr])
        cv["Linear regression"][te] = lin.predict(X_te)

        gbm = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                                        subsample=0.8, min_samples_leaf=10, random_state=0)
        gbm.fit(X_tr, y[tr], sample_weight=w[tr])
        cv["Gradient boosting"][te] = gbm.predict(X_te)

    comp = pd.DataFrame([{"model": m, "cv_rmse_in": wrmse(y, p, w),
                          "cv_r2": float(r2_score(y, p, sample_weight=w))}
                         for m, p in cv.items()])
    base_rmse = comp.loc[0, "cv_rmse_in"]
    comp["improvement_vs_baseline"] = 1 - comp["cv_rmse_in"] / base_rmse
    comp.to_csv(OUT / "part2_model_comparison.csv", index=False)

    # ---- full-sample regression for interpretation ---------------------
    z_all, stats = zscore_within_type(train, feats)
    X_df = design(z_all, train["pitch_type"], types)
    ols = sm.WLS(y, X_df, weights=w).fit()
    ci = ols.conf_int()
    coefs = pd.DataFrame({"term": X_df.columns, "coef": ols.params.values,
                          "ci_low": ci[0].values, "ci_high": ci[1].values,
                          "p_value": ols.pvalues.values})
    coefs.to_csv(OUT / "part2_coefficients.csv", index=False)
    feat_coefs = coefs[coefs["term"].isin(feats)].copy()
    feat_coefs["label"] = feat_coefs["term"].map(FEATURES)

    # ---- pick final model and score every pitcher x pitch type ---------
    lin_rmse = comp.set_index("model").loc["Linear regression", "cv_rmse_in"]
    gbm_rmse = comp.set_index("model").loc["Gradient boosting", "cv_rmse_in"]
    use_gbm = gbm_rmse < lin_rmse * 0.97  # boosting must clearly win to give up interpretability
    final_name = "Gradient boosting" if use_gbm else "Linear regression"

    z_every, _ = zscore_within_type(everyone, feats, stats)
    X_every = design(z_every, everyone["pitch_type"], types).to_numpy()
    if use_gbm:
        final = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                                          subsample=0.8, min_samples_leaf=10, random_state=0)
        final.fit(X_df.to_numpy(), y, sample_weight=w)
    else:
        final = LinearRegression(fit_intercept=False).fit(X_df.to_numpy(), y, sample_weight=w)
    everyone["predicted_total_diff"] = final.predict(X_every)

    type_avg = (train.assign(w=w, wy=w * y).groupby("pitch_type")
                .apply(lambda g: g["wy"].sum() / g["w"].sum(), include_groups=False))
    everyone["type_avg_diff"] = everyone["pitch_type"].map(type_avg)
    everyone["predicted_vs_type_avg"] = everyone["predicted_total_diff"] - everyone["type_avg_diff"]
    everyone["actual_total_diff"] = np.where(everyone["n_coors"] >= MIN_COORS,
                                             everyone["total_diff"], np.nan)
    cols = ["pitcher", "pitcher_name", "pitch_type", "n_other", "n_coors", "total_mov", *feats,
            "predicted_total_diff", "type_avg_diff", "predicted_vs_type_avg", "actual_total_diff"]
    everyone[cols].to_csv(OUT / "all_pitch_predictions.csv", index=False)

    # ---- charts --------------------------------------------------------
    # chart 4: feature effects
    fc = feat_coefs.sort_values("coef")
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(fc) + 2))
    yy = np.arange(len(fc))
    colors = ["#2f8f5b" if c > 0 else "#c0392b" for c in fc["coef"]]
    ax.errorbar(fc["coef"], yy, xerr=[fc["coef"] - fc["ci_low"], fc["ci_high"] - fc["coef"]],
                fmt="none", ecolor="#999", capsize=3)
    ax.scatter(fc["coef"], yy, color=colors, s=60, zorder=3)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_yticks(yy, fc["label"])
    style(ax, "What predicts keeping movement at Coors?",
          xlabel="inches of movement change per 1 SD (within pitch type)")
    fig.text(0.01, 0.005, "Weighted regression with pitch-type baselines; bars = 95% CI. "
             "Right of 0 = keeps more movement at Coors.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "chart4_feature_effects.png", dpi=160)
    plt.close(fig)

    # chart 5: CV predicted vs actual
    best_cv = cv[final_name]
    fig, ax = plt.subplots(figsize=(7, 7))
    for t in types:
        m = (train["pitch_type"] == t).to_numpy()
        ax.scatter(best_cv[m], y[m], s=w[m] / 4, alpha=0.6, label=PITCH_NAMES.get(t, t),
                   color=TYPE_COLORS.get(t, "#999"))
    lo, hi = np.nanmin([best_cv.min(), y.min()]), np.nanmax([best_cv.max(), y.max()])
    ax.plot([lo, hi], [lo, hi], color="#333", lw=0.8, ls="--")
    r2 = comp.set_index("model").loc[final_name, "cv_r2"]
    style(ax, f"{final_name}: out-of-sample predictions (R² = {r2:.2f})",
          xlabel="predicted movement change at Coors (in)",
          ylabel="actual movement change at Coors (in)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "chart5_predicted_vs_actual.png", dpi=160)
    plt.close(fig)

    # chart 6: spin efficiency proxy vs movement lost
    fig, ax = plt.subplots(figsize=(9, 6))
    for t in types:
        d = train[train["pitch_type"] == t]
        ax.scatter(d["mov_per_krpm"], d["total_diff"], s=np.minimum(d["n_coors"], WEIGHT_CAP) / 4,
                   alpha=0.6, color=TYPE_COLORS.get(t, "#999"), label=PITCH_NAMES.get(t, t))
    slope, intercept = np.polyfit(train["mov_per_krpm"], y, 1, w=np.sqrt(w))
    xs = np.linspace(train["mov_per_krpm"].min(), train["mov_per_krpm"].max(), 50)
    ax.plot(xs, slope * xs + intercept, color="#333", lw=1.2)
    ax.axhline(0, color="#999", lw=0.8)
    style(ax, "More movement per rpm = more movement lost at Coors?",
          xlabel="movement per 1,000 rpm away from Coors (inches)",
          ylabel="movement change at Coors (inches)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.text(0.01, 0.005, "Each dot = one pitcher's pitch type; size = Coors pitches. "
             "Low movement per rpm suggests gyro (non-Magnus) spin.", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "chart6_spin_efficiency.png", dpi=160)
    plt.close(fig)

    # ---- print summary -------------------------------------------------
    pd.set_option("display.width", 140)
    print("\nModel comparison (cross-validated, grouped by pitcher):")
    print(comp.to_string(index=False, formatters={
        "cv_rmse_in": "{:.3f}".format, "cv_r2": "{:.3f}".format,
        "improvement_vs_baseline": "{:+.1%}".format}))
    print(f"\nFinal model used for predictions: {final_name}")

    print("\nFeature effects (inches per 1 SD within pitch type; + = keeps more movement):")
    print(feat_coefs[["label", "coef", "ci_low", "ci_high", "p_value"]]
          .sort_values("coef", ascending=False)
          .to_string(index=False, formatters={"coef": "{:+.3f}".format,
                                              "ci_low": "{:+.3f}".format,
                                              "ci_high": "{:+.3f}".format,
                                              "p_value": "{:.3f}".format}))

    never = everyone[everyone["n_coors"] < MIN_COORS]
    print(f"\nScored {len(everyone):,} pitcher x pitch types "
          f"({len(never):,} with little or no Coors history).")
    print(f"Charts and CSVs saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
