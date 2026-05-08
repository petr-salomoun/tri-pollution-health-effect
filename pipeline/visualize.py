"""
Step 4: Generate visualizations for the report.

Produces static charts saved as PNG files for embedding in README/DETAILS.
"""
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pipeline.config import (
    ANALYSIS_FILE,
    FIGURES_DIR,
    SCORED_FILE,
    SEVERITY_COLORS,
    VIZ_DPI,
    VIZ_FONT_SIZE,
    VIZ_PALETTE,
    VIZ_SCATTER_MAX,
    VIZ_STYLE,
)

logger = logging.getLogger(__name__)

# Style
sns.set_theme(style=VIZ_STYLE, palette=VIZ_PALETTE)
plt.rcParams["figure.dpi"] = VIZ_DPI
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = VIZ_FONT_SIZE


def load_data():
    df = pd.read_csv(SCORED_FILE, low_memory=False)
    with open(ANALYSIS_FILE) as f:
        analysis = json.load(f)
    return df, analysis


def plot_releases_vs_poverty(df, output_dir=None):
    """Scatter plot: total releases vs poverty rate."""
    output_dir = output_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use log scale for releases
    mask = (df["TOTAL_RELEASES"] > 0) & df["poverty_pct"].notna()
    data = df[mask].copy()

    # Sample for readability if too many points
    if len(data) > VIZ_SCATTER_MAX:
        data = data.sample(VIZ_SCATTER_MAX, random_state=42)

    scatter = ax.scatter(
        data["poverty_pct"],
        data["TOTAL_RELEASES"],
        c=data["ej_score"],
        cmap="RdYlGn_r",
        alpha=0.4,
        s=15,
        edgecolors="none",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Poverty Rate (%)", fontsize=13)
    ax.set_ylabel("Total Releases (lbs, log scale)", fontsize=13)
    ax.set_title("Toxic Releases vs. Community Poverty Rate", fontsize=15, fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax, label="EJ Score")

    # Add trend line
    from scipy import stats as sp_stats
    log_releases = np.log1p(data["TOTAL_RELEASES"])
    slope, intercept, r, p, se = sp_stats.linregress(data["poverty_pct"], log_releases)
    x_line = np.linspace(data["poverty_pct"].min(), data["poverty_pct"].max(), 100)
    ax.plot(x_line, np.expm1(slope * x_line + intercept), "r-", linewidth=2, alpha=0.8,
            label=f"Trend (r={r:.3f}, p={p:.1e})")
    ax.legend(fontsize=11)

    fig.savefig(output_dir / "releases_vs_poverty.png")
    plt.close(fig)
    logger.info("Generated: releases_vs_poverty.png")


def plot_releases_vs_minority(df, output_dir=None):
    """Scatter plot: total releases vs minority percentage."""
    output_dir = output_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(10, 7))

    mask = (df["TOTAL_RELEASES"] > 0) & df["minority_pct"].notna()
    data = df[mask].copy()
    if len(data) > VIZ_SCATTER_MAX:
        data = data.sample(VIZ_SCATTER_MAX, random_state=42)

    scatter = ax.scatter(
        data["minority_pct"],
        data["TOTAL_RELEASES"],
        c=data["ej_score"],
        cmap="RdYlGn_r",
        alpha=0.4,
        s=15,
        edgecolors="none",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Minority Population (%)", fontsize=13)
    ax.set_ylabel("Total Releases (lbs, log scale)", fontsize=13)
    ax.set_title("Toxic Releases vs. Minority Population Share", fontsize=15, fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="EJ Score")

    from scipy import stats as sp_stats
    log_releases = np.log1p(data["TOTAL_RELEASES"])
    slope, intercept, r, p, se = sp_stats.linregress(data["minority_pct"], log_releases)
    x_line = np.linspace(data["minority_pct"].min(), data["minority_pct"].max(), 100)
    ax.plot(x_line, np.expm1(slope * x_line + intercept), "r-", linewidth=2, alpha=0.8,
            label=f"Trend (r={r:.3f}, p={p:.1e})")
    ax.legend(fontsize=11)

    fig.savefig(output_dir / "releases_vs_minority.png")
    plt.close(fig)
    logger.info("Generated: releases_vs_minority.png")


def plot_severity_distribution(df, output_dir=None):
    """Bar chart of severity tier distribution."""
    output_dir = output_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = SEVERITY_COLORS
    order = ["Low", "Moderate", "High", "Critical"]
    counts = df["severity_tier"].value_counts()

    bars = ax.bar(
        order,
        [counts.get(t, 0) for t in order],
        color=[colors[t] for t in order],
        edgecolor="white",
        linewidth=1.5,
    )

    for bar, tier in zip(bars, order):
        count = counts.get(tier, 0)
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df) * 0.01,
                f"{count:,}\n({pct:.1f}%)", ha="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Severity Tier", fontsize=13)
    ax.set_ylabel("Number of Facility-Year Records", fontsize=13)
    ax.set_title("Environmental Justice Severity Distribution", fontsize=15, fontweight="bold")

    fig.savefig(output_dir / "severity_distribution.png")
    plt.close(fig)
    logger.info("Generated: severity_distribution.png")


def plot_temporal_trends(df, output_dir=None):
    """Line plot of emissions over time by demographic quintile."""
    output_dir = output_dir or FIGURES_DIR

    if "REPORTING_YEAR" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Overall trend
    yearly = df.groupby("REPORTING_YEAR")["TOTAL_RELEASES"].agg(["sum", "mean"]).reset_index()
    axes[0].plot(yearly["REPORTING_YEAR"], yearly["sum"] / 1e6, "b-o", linewidth=2, markersize=5)
    axes[0].set_xlabel("Year", fontsize=12)
    axes[0].set_ylabel("Total Releases (millions of lbs)", fontsize=12)
    axes[0].set_title("Total TRI Releases Over Time", fontsize=14, fontweight="bold")

    # By poverty quintile
    df["poverty_quintile"] = pd.qcut(
        df["poverty_pct"].fillna(df["poverty_pct"].median()),
        5, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"],
        duplicates="drop",
    )
    trend = df.groupby(["REPORTING_YEAR", "poverty_quintile"], observed=True)["TOTAL_RELEASES"].mean().reset_index()

    for quintile in ["Q1 (Lowest)", "Q3", "Q5 (Highest)"]:
        q_data = trend[trend["poverty_quintile"] == quintile]
        axes[1].plot(q_data["REPORTING_YEAR"], q_data["TOTAL_RELEASES"],
                     "-o", linewidth=2, markersize=4, label=quintile)

    axes[1].set_xlabel("Year", fontsize=12)
    axes[1].set_ylabel("Mean Releases per Facility (lbs)", fontsize=12)
    axes[1].set_title("Mean Releases by Poverty Quintile Over Time", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "temporal_trends.png")
    plt.close(fig)
    logger.info("Generated: temporal_trends.png")


def plot_boxplot_disparity(df, output_dir=None):
    """Box plots comparing releases in high vs low poverty/minority areas."""
    output_dir = output_dir or FIGURES_DIR

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, (col, label) in enumerate([("poverty_pct", "Poverty"), ("minority_pct", "Minority")]):
        if col not in df.columns:
            continue
        data = df[df["TOTAL_RELEASES"] > 0].copy()
        data["group"] = pd.qcut(
            data[col].fillna(data[col].median()),
            5, labels=["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"],
            duplicates="drop",
        )
        data["log_releases"] = np.log10(data["TOTAL_RELEASES"] + 1)

        sns.boxplot(data=data, x="group", y="log_releases", ax=axes[i],
                    palette="RdYlGn_r", showfliers=False)
        axes[i].set_xlabel(f"{label} Rate Quintile", fontsize=12)
        axes[i].set_ylabel("log10(Releases + 1)", fontsize=12)
        axes[i].set_title(f"Release Distribution by {label} Quintile", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "disparity_boxplots.png")
    plt.close(fig)
    logger.info("Generated: disparity_boxplots.png")


def plot_top_chemicals(df, output_dir=None):
    """Horizontal bar chart of top chemicals by total releases."""
    output_dir = output_dir or FIGURES_DIR

    if "CHEMICAL_NAME" not in df.columns:
        return

    # Expand concatenated chemical names
    chem_releases = {}
    for _, row in df.iterrows():
        chems = str(row.get("CHEMICAL_NAME", "")).split(";")
        for chem in chems:
            chem = chem.strip()
            if chem and chem != "nan":
                chem_releases[chem] = chem_releases.get(chem, 0) + row["TOTAL_RELEASES"]

    top = sorted(chem_releases.items(), key=lambda x: x[1], reverse=True)[:15]

    fig, ax = plt.subplots(figsize=(10, 7))
    names = [t[0] for t in reversed(top)]
    values = [t[1] / 1e6 for t in reversed(top)]

    bars = ax.barh(names, values, color=sns.color_palette("viridis", len(names)))
    ax.set_xlabel("Total Releases (millions of lbs)", fontsize=12)
    ax.set_title("Top 15 Chemicals by Total Releases", fontsize=14, fontweight="bold")

    fig.savefig(output_dir / "top_chemicals.png")
    plt.close(fig)
    logger.info("Generated: top_chemicals.png")


def plot_state_ej_scores(df, output_dir=None):
    """Bar chart of mean EJ score by state."""
    output_dir = output_dir or FIGURES_DIR

    if "ST" not in df.columns:
        return

    state_scores = df.groupby("ST")["ej_score"].mean().sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(state_scores)))
    ax.bar(state_scores.index, state_scores.values, color=colors, edgecolor="white")
    ax.set_xlabel("State", fontsize=12)
    ax.set_ylabel("Mean EJ Score", fontsize=12)
    ax.set_title("Top 20 States by Mean Environmental Justice Score", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    fig.savefig(output_dir / "state_ej_scores.png")
    plt.close(fig)
    logger.info("Generated: state_ej_scores.png")


def plot_ej_score_histogram(df, output_dir=None):
    """Histogram of EJ score distribution."""
    output_dir = output_dir or FIGURES_DIR

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["ej_score"], bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(df["ej_score"].median(), color="red", linestyle="--", linewidth=2, label=f"Median: {df['ej_score'].median():.1f}")
    ax.axvline(df["ej_score"].mean(), color="orange", linestyle="--", linewidth=2, label=f"Mean: {df['ej_score'].mean():.1f}")
    ax.set_xlabel("Environmental Justice Score", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Distribution of EJ Scores Across All Facility-Year Records", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    fig.savefig(output_dir / "ej_score_histogram.png")
    plt.close(fig)
    logger.info("Generated: ej_score_histogram.png")


def visualize_all(df=None):
    """Generate all visualizations."""
    logger.info("=" * 60)
    logger.info("STEP 4: GENERATING VISUALIZATIONS")
    logger.info("=" * 60)

    if df is None:
        df, analysis = load_data()

    plot_releases_vs_poverty(df)
    plot_releases_vs_minority(df)
    plot_severity_distribution(df)
    plot_temporal_trends(df)
    plot_boxplot_disparity(df)
    plot_top_chemicals(df)
    plot_state_ej_scores(df)
    plot_ej_score_histogram(df)

    logger.info(f"All visualizations saved to {FIGURES_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    visualize_all()
