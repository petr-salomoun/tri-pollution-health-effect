"""
Step 3: Statistical analysis of environmental justice patterns.

- Correlations between pollution and demographics
- Temporal trends
- Chemical-specific analysis
- Facility relocation detection
"""
import json
import logging

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.config import ANALYSIS_FILE, SCORED_FILE

logger = logging.getLogger(__name__)


def load_scored_data(filepath=None):
    """Load the scored facility dataset."""
    filepath = filepath or SCORED_FILE
    return pd.read_csv(filepath, low_memory=False)


def correlation_analysis(df):
    """Compute correlations between releases and demographic/health variables."""
    results = {}

    # Core correlation: releases vs demographics
    pairs = [
        ("TOTAL_RELEASES", "poverty_pct", "Releases vs Poverty"),
        ("TOTAL_RELEASES", "minority_pct", "Releases vs Minority %"),
        ("TOTAL_RELEASES", "median_income", "Releases vs Median Income"),
        ("weighted_releases", "poverty_pct", "Weighted Releases vs Poverty"),
        ("weighted_releases", "minority_pct", "Weighted Releases vs Minority %"),
    ]

    # Health correlations
    health_cols = [c for c in df.columns if c.endswith("_crude") or c == "pct_no_insurance"]
    for hcol in health_cols:
        pairs.append(("TOTAL_RELEASES", hcol, f"Releases vs {hcol}"))

    correlations = []
    for col1, col2, label in pairs:
        if col1 in df.columns and col2 in df.columns:
            mask = df[[col1, col2]].notna().all(axis=1)
            x, y = df.loc[mask, col1], df.loc[mask, col2]
            if len(x) > 30:
                # Spearman (rank-based, better for skewed distributions)
                rho, p_val = stats.spearmanr(x, y)
                correlations.append({
                    "variable_1": col1,
                    "variable_2": col2,
                    "label": label,
                    "spearman_rho": round(rho, 4),
                    "p_value": float(f"{p_val:.2e}"),
                    "n_observations": int(len(x)),
                    "significant": p_val < 0.05,
                })

    results["correlations"] = correlations
    logger.info(f"Computed {len(correlations)} correlation pairs")
    return results


def demographic_disparity_analysis(df):
    """Compare pollution burden across demographic groups."""
    results = {}

    # Split facilities by poverty quintile
    df["poverty_quintile"] = pd.qcut(
        df["poverty_pct"].fillna(df["poverty_pct"].median()),
        5, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"],
        duplicates="drop",
    )

    poverty_stats = df.groupby("poverty_quintile", observed=True).agg({
        "TOTAL_RELEASES": ["mean", "median", "sum"],
        "weighted_releases": ["mean", "median"],
        "ej_score": "mean",
        "TRI_FACILITY_ID": "nunique",
    }).round(2)

    poverty_stats.columns = ["_".join(c) for c in poverty_stats.columns]
    results["by_poverty_quintile"] = poverty_stats.reset_index().to_dict(orient="records")

    # Split by minority percentage
    df["minority_quintile"] = pd.qcut(
        df["minority_pct"].fillna(df["minority_pct"].median()),
        5, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"],
        duplicates="drop",
    )

    minority_stats = df.groupby("minority_quintile", observed=True).agg({
        "TOTAL_RELEASES": ["mean", "median", "sum"],
        "weighted_releases": ["mean", "median"],
        "ej_score": "mean",
        "TRI_FACILITY_ID": "nunique",
    }).round(2)

    minority_stats.columns = ["_".join(c) for c in minority_stats.columns]
    results["by_minority_quintile"] = minority_stats.reset_index().to_dict(orient="records")

    # Mann-Whitney U test: top vs bottom quintile
    for var, quintile_col in [("poverty_pct", "poverty_quintile"), ("minority_pct", "minority_quintile")]:
        q1 = df[df[quintile_col] == "Q1 (Lowest)"]["TOTAL_RELEASES"].dropna()
        q5 = df[df[quintile_col] == "Q5 (Highest)"]["TOTAL_RELEASES"].dropna()
        if len(q1) > 10 and len(q5) > 10:
            u_stat, p_val = stats.mannwhitneyu(q5, q1, alternative="greater")
            results[f"mannwhitney_{var}"] = {
                "u_statistic": round(float(u_stat), 2),
                "p_value": float(f"{p_val:.2e}"),
                "significant": p_val < 0.05,
                "q5_median": round(float(q5.median()), 2),
                "q1_median": round(float(q1.median()), 2),
                "ratio": round(float(q5.median() / max(q1.median(), 1)), 2),
                "interpretation": (
                    f"Communities in highest {var.replace('_pct', '')} quintile have "
                    f"{round(float(q5.median() / max(q1.median(), 1)), 1)}x higher "
                    f"median releases than lowest quintile"
                ),
            }

    logger.info("Demographic disparity analysis complete")
    return results


def temporal_trend_analysis(df):
    """Analyze how EJ patterns change over time."""
    results = {}

    if "REPORTING_YEAR" not in df.columns:
        return results

    # Overall trend
    yearly = df.groupby("REPORTING_YEAR").agg({
        "TOTAL_RELEASES": ["sum", "mean", "median"],
        "TRI_FACILITY_ID": "nunique",
        "ej_score": "mean",
    }).round(2)
    yearly.columns = ["_".join(c) for c in yearly.columns]
    yearly = yearly.reset_index()
    results["yearly_totals"] = yearly.to_dict(orient="records")

    # Trend by demographic quintile over time
    for quintile_col in ["poverty_quintile", "minority_quintile"]:
        if quintile_col in df.columns:
            trend = df.groupby(["REPORTING_YEAR", quintile_col], observed=True).agg({
                "TOTAL_RELEASES": "mean",
            }).reset_index()
            results[f"trend_by_{quintile_col}"] = trend.to_dict(orient="records")

    # Calculate year-over-year change
    if len(yearly) > 1:
        total_change = (
            (yearly["TOTAL_RELEASES_sum"].iloc[-1] - yearly["TOTAL_RELEASES_sum"].iloc[0])
            / yearly["TOTAL_RELEASES_sum"].iloc[0] * 100
        )
        results["total_change_pct"] = round(float(total_change), 2)

    logger.info("Temporal trend analysis complete")
    return results


def chemical_analysis(df):
    """Analyze which chemicals are most concentrated in vulnerable communities."""
    results = {}

    if "CHEMICAL_NAME" not in df.columns:
        return results

    # For facilities reporting single chemicals, analyze by chemical
    # Since CHEMICAL_NAME may be concatenated with ";", split
    chem_rows = []
    for _, row in df.iterrows():
        chems = str(row.get("CHEMICAL_NAME", "")).split(";")
        for chem in chems:
            chem = chem.strip()
            if chem and chem != "nan":
                chem_rows.append({
                    "chemical": chem,
                    "releases": row["TOTAL_RELEASES"],
                    "poverty_pct": row.get("poverty_pct", np.nan),
                    "minority_pct": row.get("minority_pct", np.nan),
                    "ej_score": row.get("ej_score", np.nan),
                })

    if not chem_rows:
        return results

    chem_df = pd.DataFrame(chem_rows)

    # Top chemicals by total releases
    top_chems = chem_df.groupby("chemical").agg({
        "releases": ["sum", "mean", "count"],
        "poverty_pct": "mean",
        "minority_pct": "mean",
        "ej_score": "mean",
    }).round(2)
    top_chems.columns = ["_".join(c) for c in top_chems.columns]
    top_chems = top_chems.sort_values("releases_sum", ascending=False).head(20)
    results["top_chemicals"] = top_chems.reset_index().to_dict(orient="records")

    # Chemicals with highest EJ disparity
    chem_ej = chem_df.groupby("chemical").agg({
        "ej_score": ["mean", "count"],
        "poverty_pct": "mean",
    }).round(2)
    chem_ej.columns = ["_".join(c) for c in chem_ej.columns]
    chem_ej = chem_ej[chem_ej["ej_score_count"] > 50]  # Minimum sample size
    chem_ej = chem_ej.sort_values("ej_score_mean", ascending=False).head(10)
    results["highest_ej_chemicals"] = chem_ej.reset_index().to_dict(orient="records")

    logger.info(f"Chemical analysis: {len(top_chems)} top chemicals analyzed")
    return results


def state_analysis(df):
    """Analyze EJ patterns by state."""
    results = {}

    if "ST" not in df.columns:
        return results

    state_stats = df.groupby("ST").agg({
        "TOTAL_RELEASES": ["sum", "mean"],
        "ej_score": "mean",
        "TRI_FACILITY_ID": "nunique",
        "poverty_pct": "mean",
        "minority_pct": "mean",
    }).round(2)
    state_stats.columns = ["_".join(c) for c in state_stats.columns]
    state_stats = state_stats.sort_values("ej_score_mean", ascending=False)
    results["by_state"] = state_stats.reset_index().to_dict(orient="records")

    logger.info(f"State analysis: {len(state_stats)} states")
    return results


def summary_statistics(df):
    """Compute summary statistics for the report."""
    results = {
        "total_records": int(len(df)),
        "unique_facilities": int(df["TRI_FACILITY_ID"].nunique()),
        "years_covered": sorted(df["REPORTING_YEAR"].dropna().unique().astype(int).tolist()),
        "states_covered": int(df["ST"].nunique()) if "ST" in df.columns else 0,
        "total_releases_lbs": round(float(df["TOTAL_RELEASES"].sum()), 0),
        "mean_ej_score": round(float(df["ej_score"].mean()), 2),
        "median_ej_score": round(float(df["ej_score"].median()), 2),
        "severity_counts": df["severity_tier"].value_counts().to_dict(),
        "pct_critical": round(float((df["severity_tier"] == "Critical").mean() * 100), 1),
        "pct_in_high_poverty": round(
            float((df["poverty_pct"] > 20).mean() * 100), 1
        ) if "poverty_pct" in df.columns else None,
        "pct_in_high_minority": round(
            float((df["minority_pct"] > 50).mean() * 100), 1
        ) if "minority_pct" in df.columns else None,
        "mean_poverty_pct": round(float(df["poverty_pct"].mean()), 2) if "poverty_pct" in df.columns else None,
        "mean_minority_pct": round(float(df["minority_pct"].mean()), 2) if "minority_pct" in df.columns else None,
        "carcinogen_pct": round(
            float(df["IS_CARCINOGEN"].mean() * 100), 1
        ) if "IS_CARCINOGEN" in df.columns else None,
    }

    # Top 10 worst facilities
    latest_year = df["REPORTING_YEAR"].max()
    latest = df[df["REPORTING_YEAR"] == latest_year]
    top_facilities = latest.nlargest(10, "ej_score")[
        ["TRI_FACILITY_ID", "FACILITY_NAME", "ST", "TOTAL_RELEASES",
         "ej_score", "severity_tier", "poverty_pct", "minority_pct"]
    ].to_dict(orient="records")
    results["top_10_worst_facilities"] = top_facilities

    return results


def analyze_all(df=None):
    """Run all analyses."""
    logger.info("=" * 60)
    logger.info("STEP 3: ANALYZING DATA")
    logger.info("=" * 60)

    if df is None:
        df = load_scored_data()

    # Need quintiles for some analyses
    df["poverty_quintile"] = pd.qcut(
        df["poverty_pct"].fillna(df["poverty_pct"].median()),
        5, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"],
        duplicates="drop",
    )
    df["minority_quintile"] = pd.qcut(
        df["minority_pct"].fillna(df["minority_pct"].median()),
        5, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"],
        duplicates="drop",
    )

    all_results = {}
    all_results["summary"] = summary_statistics(df)
    all_results["correlations"] = correlation_analysis(df)
    all_results["disparities"] = demographic_disparity_analysis(df)
    all_results["temporal"] = temporal_trend_analysis(df)
    all_results["chemicals"] = chemical_analysis(df)
    all_results["states"] = state_analysis(df)

    # Save results
    with open(ANALYSIS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Analysis results saved to {ANALYSIS_FILE}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyze_all()
