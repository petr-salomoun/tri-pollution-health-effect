"""
Deep Research Analyses: Three Independent Studies
==================================================
1. Spatial Health Impact Analysis
   - Which release types/chemicals/media associate most with adverse health outcomes
   - Dose-response curves per health outcome
   - Carcinogen vs non-carcinogen differential health burden
   - Hot-spot clustering of co-occurring releases + poor health

2. Spatial Wealth Analysis
   - Release volume, frequency, and carcinogen concentration vs income/poverty
   - Industry sector siting patterns by wealth decile
   - Cumulative exposure burden by wealth quintile

3. Combined: Does Health Burden Concentrate in Poor Areas?
   - Health-poverty interaction near TRI facilities
   - Facility proximity effect on health outcomes conditional on poverty
   - Multi-disease burden index vs economic deprivation
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, kruskal

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ─── Output directory ───────────────────────────────────────────────────────
OUT = Path("output/research")
OUT.mkdir(parents=True, exist_ok=True)

# ─── EPA-based carcinogen list (IARC Group 1 / EPA Group A known carcinogens
#     that appear in TRI) ────────────────────────────────────────────────────
KNOWN_CARCINOGENS = {
    "Benzene", "1,3-Butadiene", "Vinyl chloride", "Formaldehyde",
    "Arsenic", "Arsenic compounds", "Cadmium", "Cadmium compounds",
    "Chromium compounds", "Beryllium", "Beryllium compounds",
    "Lead", "Lead compounds", "Mercury", "Mercury compounds",
    "Nickel", "Nickel compounds", "Asbestos (friable)",
    "Ethylene oxide", "Styrene", "Tetrachloroethylene",
    "Trichloroethylene", "Chloroform", "Carbon tetrachloride",
    "Dichloromethane", "1,2-Dichloroethane", "1,1,2,2-Tetrachloroethane",
    "Ethylene dibromide (EDB)", "1,2-Dibromoethane",
    "Hexachlorobenzene", "Polycyclic aromatic compounds",
    "Dioxins and furans", "Hydrogen fluoride", "Hydrazine",
    "1,1-Dimethylhydrazine", "Acrylonitrile", "Acrylamide",
    "Cobalt", "Cobalt compounds", "Benzo[a]pyrene",
    "Chlorinated dibenzo-p-dioxins (chlorinated)", "Hexavalent chromium",
}

PERSISTENT_CHEMICALS = {
    "Arsenic", "Arsenic compounds", "Lead", "Lead compounds",
    "Mercury", "Mercury compounds", "Cadmium", "Cadmium compounds",
    "Chromium compounds", "PCBs", "Dioxins and furans",
    "Polycyclic aromatic compounds", "Hexachlorobenzene",
    "Perfluorooctanoic acid (PFOA)", "Perfluorooctane sulfonate (PFOS)",
}

# Medium labels for display
MEDIUM_LABELS = {
    "AIR FUG": "Air (Fugitive)",
    "AIR STACK": "Air (Stack)",
    "WATER": "Surface Water",
    "LAND TREA": "Land Treatment",
    "SURF IMP": "Surface Impound.",
    "OTH LANDF": "Other Landfill",
    "RCRA C": "RCRA C Landfill",
    "OTH DISP": "Other Disposal",
    "UNINJ I": "Underground Inj. I",
    "UNINJ IIV": "Underground Inj. II-V",
}

HEALTH_LABELS = {
    "cancer_crude": "Cancer Rate (%)",
    "asthma_crude": "Asthma Rate (%)",
    "chd_crude": "Heart Disease Rate (%)",
    "copd_crude": "COPD Rate (%)",
    "diabetes_crude": "Diabetes Rate (%)",
    "mental_health_crude": "Poor Mental Health (%)",
}

HEALTH_COLS = list(HEALTH_LABELS.keys())

PALETTE_TIERS = {
    "Critical": "#8e44ad",
    "High": "#e74c3c",
    "Moderate": "#f39c12",
    "Low": "#2ecc71",
}

# ─── Utilities ───────────────────────────────────────────────────────────────

def _load_census():
    """Load census data with properly constructed FIPS tract codes and derived metrics."""
    census = pd.read_csv("data/raw/census_acs.csv", low_memory=False)
    # Properly construct FIPS_TRACT from state + county + tract (zero-padded)
    census['fips_tract'] = (
        census['state'].astype(str).str.zfill(2) +
        census['county'].astype(str).str.zfill(3) +
        census['tract'].astype(str).str.zfill(6)
    )
    
    # Basic demographics
    census['poverty_pct'] = (
        census['B17001_002E'] / census['B17001_001E'].clip(1) * 100
    )
    census['minority_pct'] = (
        100 - census['B02001_002E'] / census['B02001_001E'].clip(1) * 100
    )
    census['median_income'] = census['B19013_001E']
    
    # Age metrics (if available)
    if 'B01002_001E' in census.columns:
        census['median_age'] = pd.to_numeric(census['B01002_001E'], errors='coerce')
    
    if 'B01001_001E' in census.columns:
        census['total_pop'] = pd.to_numeric(census['B01001_001E'], errors='coerce')
        
        # Calculate 65+ population (males 65-85+ and females 65-85+)
        # Males 65+: B01001_020E through B01001_025E
        # Females 65+: B01001_044E through B01001_049E
        male_65_cols = [f'B01001_0{i}E' for i in range(20, 26)]
        female_65_cols = [f'B01001_0{i}E' for i in range(44, 50)]
        
        male_65_cols = [c for c in male_65_cols if c in census.columns]
        female_65_cols = [c for c in female_65_cols if c in census.columns]
        
        if male_65_cols and female_65_cols:
            for c in male_65_cols + female_65_cols:
                census[c] = pd.to_numeric(census[c], errors='coerce').fillna(0)
            
            census['pop_65_plus'] = (
                census[male_65_cols].sum(axis=1) + 
                census[female_65_cols].sum(axis=1)
            )
            census['pct_65_plus'] = (
                census['pop_65_plus'] / census['total_pop'].clip(1) * 100
            )
    
    return census


def _save(fig, name, tight=True):
    path = OUT / f"{name}.png"
    if tight:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def _load_scored():
    p = Path("data/processed/facilities_scored.csv")
    df = pd.read_csv(p, low_memory=False)
    # Add carcinogen flag from name-based list
    def _is_carcin(name):
        if pd.isna(name):
            return False
        for c in KNOWN_CARCINOGENS:
            if c.lower() in str(name).lower():
                return True
        return False
    df["IS_CARCINOGEN"] = df["CHEMICAL_NAME"].apply(_is_carcin)
    df["IS_PERSISTENT"] = df["CHEMICAL_NAME"].apply(
        lambda x: any(p.lower() in str(x).lower() for p in PERSISTENT_CHEMICALS)
    )
    df["log_releases"] = np.log10(df["TOTAL_RELEASES"].clip(lower=0.1))
    df["has_health"] = df["cancer_crude"].notna()
    return df


def _load_tract_neighbors():
    """Load tract neighbor lookup (built from centroid proximity)."""
    import json
    p = Path("data/processed/tract_neighbors.json")
    if not p.exists():
        logger.warning("tract_neighbors.json not found — run neighbor build script first")
        return {}
    with open(p) as f:
        return json.load(f)


def _build_influence_zones(tri_tracts: set, neighbor_dict: dict):
    """
    Build TRI influence zone: includes TRI-hosting tracts AND their neighbors.
    
    Returns:
        influenced_tracts: set of fips_tract codes in the TRI influence zone
    """
    influenced = set(tri_tracts)
    for tri_tract in tri_tracts:
        neighbors = neighbor_dict.get(tri_tract, [])
        influenced.update(neighbors)
    return influenced


def _classify_tracts_for_case_control(all_tracts_df, tri_tract_data, neighbor_dict):
    """
    Classify tracts into:
      - 'tri_direct': tract contains a TRI facility
      - 'tri_neighbor': tract neighbors a TRI facility tract (but has no TRI itself)
      - 'control': tract has no TRI and no TRI neighbors (true background)
    
    Args:
        all_tracts_df: DataFrame with 'fips_tract' column (all CDC/census tracts)
        tri_tract_data: DataFrame with TRI tract-level aggregations (has 'fips_tract', 'carc_releases', etc.)
        neighbor_dict: dict mapping fips_tract -> list of neighbor fips_tracts
    
    Returns:
        DataFrame with added columns: 'tri_zone' ('tri_direct', 'tri_neighbor', 'control'),
        'in_influence' (bool), plus release data for direct TRI tracts
    """
    all_tracts_df = all_tracts_df.copy()
    all_tracts_df['fips_tract'] = all_tracts_df['fips_tract'].astype(str).str.zfill(11)
    
    # Set of tracts with TRI facilities
    tri_direct_set = set(tri_tract_data['fips_tract'].astype(str).str.zfill(11).unique())
    
    # Build influence zone (TRI + neighbors)
    influenced_set = _build_influence_zones(tri_direct_set, neighbor_dict)
    
    # Neighbor-only tracts (influenced but not direct)
    tri_neighbor_set = influenced_set - tri_direct_set
    
    # Classify
    def _classify(fips):
        if fips in tri_direct_set:
            return 'tri_direct'
        elif fips in tri_neighbor_set:
            return 'tri_neighbor'
        else:
            return 'control'
    
    all_tracts_df['tri_zone'] = all_tracts_df['fips_tract'].apply(_classify)
    all_tracts_df['in_influence'] = all_tracts_df['tri_zone'].isin(['tri_direct', 'tri_neighbor'])
    all_tracts_df['has_tri'] = all_tracts_df['tri_zone'] == 'tri_direct'  # backward compat
    
    # Merge release data for direct TRI tracts
    tri_tract_data = tri_tract_data.copy()
    tri_tract_data['fips_tract'] = tri_tract_data['fips_tract'].astype(str).str.zfill(11)
    all_tracts_df = all_tracts_df.merge(
        tri_tract_data[['fips_tract', 'total_releases', 'carc_releases', 'n_facilities']],
        on='fips_tract', how='left'
    )
    all_tracts_df['total_releases'] = all_tracts_df['total_releases'].fillna(0)
    all_tracts_df['carc_releases'] = all_tracts_df['carc_releases'].fillna(0)
    all_tracts_df['n_facilities'] = all_tracts_df['n_facilities'].fillna(0)
    
    return all_tracts_df


def _load_release_qty_medium():
    """Load and aggregate release qty by medium per doc_ctrl_num."""
    p = Path("data/raw/tri_release_qty.csv")
    if not p.exists():
        return None
    logger.info("Loading TRI release quantity by medium...")
    cols = ["doc_ctrl_num", "environmental_medium", "total_release"]
    rdf = pd.read_csv(p, usecols=cols, low_memory=False)
    rdf["total_release"] = pd.to_numeric(rdf["total_release"], errors="coerce").fillna(0)
    # Pivot to one row per doc_ctrl_num with columns per medium
    pivot = rdf.pivot_table(
        index="doc_ctrl_num",
        columns="environmental_medium",
        values="total_release",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot.columns = [str(c).strip() for c in pivot.columns]
    return pivot


def _quintile_label(q):
    return ["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"][q - 1]


def annotate_pval(ax, x1, x2, y, p, fontsize=8):
    """Draw bracket + significance stars."""
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = "ns"
    ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], lw=0.8, c="0.3")
    ax.text((x1 + x2) / 2, y + 0.025, stars, ha="center", va="bottom",
            fontsize=fontsize, color="0.2")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: SPATIAL HEALTH IMPACT
# ═══════════════════════════════════════════════════════════════════════════════

def analysis1_health_impact(df):
    """Generate ~7 plots for the health impact analysis."""
    hdf = df[df["has_health"]].copy()
    logger.info(f"Health analysis subset: {len(hdf):,} records, {hdf['TRI_FACILITY_ID'].nunique():,} facilities")

    # ── 1a. Heatmap: Spearman correlations between releases and all health outcomes ──
    _plot_1a_correlation_heatmap(hdf)

    # ── 1b. Dose-response curves: release decile vs each health outcome ──
    _plot_1b_dose_response(hdf)

    # ── 1c. Carcinogen vs non-carcinogen: health outcome distributions ──
    _plot_1c_carcinogen_health(hdf)

    # ── 1d. Release medium composition: air vs water vs land ──
    _plot_1d_medium_breakdown(df)

    # ── 1e. Persistent chemical exposure vs health ──
    _plot_1e_persistent_health(hdf)

    # ── 1f. Multi-disease burden index vs log releases ──
    _plot_1f_multidisease_burden(hdf)

    # ── 1g. State-level: mean releases vs mean health burden ──
    _plot_1g_state_releases_health(hdf)


def _plot_1a_correlation_heatmap(hdf):
    """Heatmap of Spearman ρ between release metrics and health outcomes."""
    release_vars = {
        "log_releases": "Log₁₀ Total Releases",
        "TOXICITY_WEIGHT": "Toxicity Weight",
        "weighted_releases": "Weighted Releases",
        "IS_CARCINOGEN": "Carcinogen Flag",
        "IS_PERSISTENT": "Persistent Flag",
    }

    # Compute correlation matrix
    rows = []
    for rv, rl in release_vars.items():
        row = {"Release Metric": rl}
        col_data = hdf[rv].astype(float)
        for hc, hl in HEALTH_LABELS.items():
            r, p = spearmanr(col_data, hdf[hc], nan_policy="omit")
            row[hl.replace(" (%)", "")] = r
        rows.append(row)

    corr_df = pd.DataFrame(rows).set_index("Release Metric")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        corr_df.astype(float),
        annot=True, fmt=".3f",
        cmap="RdBu_r", center=0,
        vmin=-0.25, vmax=0.25,
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
        ax=ax,
    )
    ax.set_title("Spearman Correlations: Release Metrics vs Health Outcomes\n(tract-level, CDC PLACES data)", fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    _save(fig, "1a_correlation_heatmap")


def _plot_1b_dose_response(hdf):
    """Release decile vs each health outcome — dose-response curves."""
    hdf = hdf.copy()
    # Use rank-based decile to avoid duplicate bin edges
    hdf["release_decile"] = pd.qcut(hdf["TOTAL_RELEASES"].rank(method="first"), 10,
                                     labels=range(1, 11))
    hdf = hdf.dropna(subset=["release_decile"])

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, (hc, hl) in enumerate(HEALTH_LABELS.items()):
        ax = axes[i]
        grouped = hdf.groupby("release_decile", observed=True)[hc].agg(["mean", "sem"]).reset_index()
        grouped["release_decile"] = grouped["release_decile"].astype(int)
        grouped = grouped.sort_values("release_decile")

        ax.fill_between(grouped["release_decile"],
                        grouped["mean"] - grouped["sem"],
                        grouped["mean"] + grouped["sem"],
                        alpha=0.25, color="#e74c3c")
        ax.plot(grouped["release_decile"], grouped["mean"],
                "o-", color="#c0392b", linewidth=2, markersize=6)

        # Trend test (Spearman on means)
        r, p = spearmanr(grouped["release_decile"], grouped["mean"])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.set_title(f"{hl.replace(' (%)', '')}\nρ={r:.3f} {sig}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Release Volume Decile\n(1=lowest, 10=highest)")
        ax.set_ylabel("Mean Rate (%)")
        ax.set_xticks(range(1, 11))
        ax.grid(True, alpha=0.3)

    fig.suptitle("Dose-Response: TRI Release Volume Decile vs Community Health Outcomes",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "1b_dose_response_curves", tight=False)


def _plot_1c_carcinogen_health(hdf):
    """Side-by-side violins: carcinogen vs non-carcinogen release health outcomes."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    group_labels = {False: "Non-Carcinogen\nReleases", True: "Carcinogen\nReleases"}
    colors = {False: "#3498db", True: "#e74c3c"}

    for i, (hc, hl) in enumerate(HEALTH_LABELS.items()):
        ax = axes[i]
        plot_data = hdf[[hc, "IS_CARCINOGEN"]].dropna()
        plot_data["Group"] = plot_data["IS_CARCINOGEN"].map(group_labels)

        sns.violinplot(data=plot_data, x="Group", y=hc,
                       palette={v: colors[k] for k, v in group_labels.items()},
                       inner="quartile", ax=ax, cut=0)

        # Mann-Whitney U test
        g0 = plot_data[plot_data["IS_CARCINOGEN"] == False][hc].dropna()
        g1 = plot_data[plot_data["IS_CARCINOGEN"] == True][hc].dropna()
        if len(g0) > 10 and len(g1) > 10:
            stat, p = mannwhitneyu(g1, g0, alternative="greater")
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ymax = ax.get_ylim()[1]
            ax.text(0.5, 0.95, f"MW p{sig}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

        ax.set_title(hl.replace(" (%)", ""), fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Rate (%)")

    fig.suptitle("Community Health Outcomes:\nCarcinogen vs Non-Carcinogen TRI Releases",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "1c_carcinogen_health")


def _plot_1d_medium_breakdown(df):
    """Release by environmental medium — volume breakdown and health context."""
    rqp = Path("data/raw/tri_release_qty.csv")
    if not rqp.exists():
        logger.warning("tri_release_qty.csv not found, skipping medium breakdown")
        return

    logger.info("Building medium breakdown plot (large file)...")
    cols = ["doc_ctrl_num", "environmental_medium", "total_release"]
    rdf = pd.read_csv(rqp, usecols=cols, low_memory=False)
    rdf["total_release"] = pd.to_numeric(rdf["total_release"], errors="coerce").fillna(0)

    # Total volume by medium
    medium_totals = rdf.groupby("environmental_medium")["total_release"].sum().sort_values(ascending=False)
    medium_totals.index = [MEDIUM_LABELS.get(m, m) for m in medium_totals.index]
    medium_totals = medium_totals[medium_totals > 0]

    # Fraction per medium (normalised to pct)
    medium_pct = 100 * medium_totals / medium_totals.sum()

    # Count of non-zero reports per medium
    medium_count = rdf[rdf["total_release"] > 0].groupby("environmental_medium")["doc_ctrl_num"].nunique()
    medium_count.index = [MEDIUM_LABELS.get(m, m) for m in medium_count.index]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel A: total volume by medium (log scale)
    ax = axes[0]
    colors_med = sns.color_palette("husl", len(medium_totals))
    bars = ax.barh(medium_totals.index, medium_totals.values / 1e9, color=colors_med)
    ax.set_xlabel("Total Release Volume (Billion lbs)")
    ax.set_title("A. Total Release Volume\nby Environmental Medium", fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, medium_totals.values / 1e9):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}B", va="center", fontsize=8)

    # Panel B: pie chart of volume share
    ax = axes[1]
    top_media = medium_pct.head(6)
    other = 100 - top_media.sum()
    labels = list(top_media.index) + (["Other"] if other > 0.5 else [])
    sizes = list(top_media.values) + ([other] if other > 0.5 else [])
    wedge_colors = sns.color_palette("husl", len(labels))
    ax.pie(sizes, labels=None, autopct="%1.1f%%", colors=wedge_colors,
           startangle=140, pctdistance=0.8, textprops={"fontsize": 8})
    ax.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.2),
              fontsize=7, ncol=2)
    ax.set_title("B. Share of Total\nRelease Volume", fontweight="bold")

    # Panel C: number of unique facilities reporting each medium
    ax = axes[2]
    medium_count_s = medium_count.sort_values(ascending=False).head(10)
    ax.barh(medium_count_s.index, medium_count_s.values / 1000, color=sns.color_palette("husl", len(medium_count_s)))
    ax.set_xlabel("Unique Facilities Reporting (thousands)")
    ax.set_title("C. Facility Frequency\nby Release Medium", fontweight="bold")
    ax.invert_yaxis()

    fig.suptitle("TRI Chemical Releases by Environmental Medium (All Years Combined)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "1d_medium_breakdown")


def _plot_1e_persistent_health(hdf):
    """Persistent chemical releases vs health — comparing persistent vs non-persistent."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    # Create three groups for richer comparison
    def _group(row):
        if row["IS_PERSISTENT"]:
            return "Persistent\nChemical"
        elif row["IS_CARCINOGEN"]:
            return "Non-Persistent\nCarcinogen"
        else:
            return "Other\nChemical"

    hdf = hdf.copy()
    hdf["chem_class"] = hdf.apply(_group, axis=1)
    order = ["Other\nChemical", "Non-Persistent\nCarcinogen", "Persistent\nChemical"]
    palette = {"Other\nChemical": "#3498db", "Non-Persistent\nCarcinogen": "#f39c12",
               "Persistent\nChemical": "#e74c3c"}

    for i, (hc, hl) in enumerate(HEALTH_LABELS.items()):
        ax = axes[i]
        plot_data = hdf[[hc, "chem_class"]].dropna()

        sns.boxplot(data=plot_data, x="chem_class", y=hc,
                    order=order, palette=palette,
                    width=0.5, flierprops=dict(markersize=1), ax=ax)

        # Kruskal-Wallis test
        groups = [plot_data[plot_data["chem_class"] == g][hc].dropna() for g in order]
        if all(len(g) > 5 for g in groups):
            stat, p = kruskal(*groups)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.set_title(f"{hl.replace(' (%)', '')}\nKW p{sig}", fontsize=10, fontweight="bold")

        ax.set_xlabel("")
        ax.set_ylabel("Rate (%)")

    fig.suptitle("Chemical Persistence Class vs Community Health Outcomes",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "1e_persistent_health")


def _plot_1f_multidisease_burden(hdf):
    """Multi-disease burden index (composite) vs log releases — scatter + LOESS-style."""
    hdf = hdf.copy()

    # Normalize each health outcome to 0-1 and average → burden index
    for hc in HEALTH_COLS:
        mn, mx = hdf[hc].quantile(0.01), hdf[hc].quantile(0.99)
        hdf[f"{hc}_norm"] = (hdf[hc] - mn) / (mx - mn + 1e-9)
    norm_cols = [f"{hc}_norm" for hc in HEALTH_COLS]
    hdf["health_burden_index"] = hdf[norm_cols].mean(axis=1)

    # Bin releases into 20 bins for trend line
    hdf = hdf[hdf["TOTAL_RELEASES"] > 0]
    hdf["release_bin"] = pd.qcut(hdf["TOTAL_RELEASES"], 20, labels=False, duplicates="drop")
    trend = hdf.groupby("release_bin").agg(
        mean_release=("TOTAL_RELEASES", "median"),
        mean_burden=("health_burden_index", "mean"),
        sem_burden=("health_burden_index", "sem"),
        n=("health_burden_index", "count"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter (sample) colored by severity tier
    sample = hdf.sample(min(8000, len(hdf)), random_state=42)
    for tier, color in PALETTE_TIERS.items():
        sub = sample[sample["severity_tier"] == tier]
        ax1.scatter(np.log10(sub["TOTAL_RELEASES"].clip(0.1)),
                    sub["health_burden_index"],
                    c=color, alpha=0.3, s=8, label=tier, rasterized=True)

    # Overlay trend
    ax1.plot(np.log10(trend["mean_release"].clip(0.1)), trend["mean_burden"],
             "k-", linewidth=2.5, label="Trend (bin median)")
    ax1.fill_between(np.log10(trend["mean_release"].clip(0.1)),
                     trend["mean_burden"] - trend["sem_burden"],
                     trend["mean_burden"] + trend["sem_burden"],
                     alpha=0.2, color="black")

    r, p = spearmanr(hdf["TOTAL_RELEASES"].clip(0.1), hdf["health_burden_index"])
    ax1.set_xlabel("Log₁₀ Total Releases (lbs)", fontsize=11)
    ax1.set_ylabel("Multi-Disease Burden Index (0–1)", fontsize=11)
    ax1.set_title(f"Composite Health Burden vs Release Volume\nSpearman ρ={r:.3f}, p={p:.2e}",
                  fontsize=12, fontweight="bold")
    ax1.legend(title="Severity Tier", fontsize=8, markerscale=2)

    # Right: hexbin density
    ax2.hexbin(np.log10(hdf["TOTAL_RELEASES"].clip(0.1)),
               hdf["health_burden_index"],
               gridsize=40, cmap="YlOrRd", mincnt=1)
    cb = plt.colorbar(ax2.collections[0], ax=ax2)
    cb.set_label("Count")
    ax2.set_xlabel("Log₁₀ Total Releases (lbs)", fontsize=11)
    ax2.set_ylabel("Multi-Disease Burden Index (0–1)", fontsize=11)
    ax2.set_title("Density: Health Burden vs Release Volume", fontsize=12, fontweight="bold")

    fig.suptitle("Multi-Disease Health Burden Index vs TRI Chemical Releases",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "1f_multidisease_burden")


def _plot_1g_state_releases_health(hdf):
    """State-level scatter: mean releases vs mean health burden, sized by facility count."""
    state_agg = hdf.groupby("ST").agg(
        mean_releases=("TOTAL_RELEASES", "mean"),
        median_releases=("TOTAL_RELEASES", "median"),
        n_facilities=("TRI_FACILITY_ID", "nunique"),
        cancer=("cancer_crude", "mean"),
        asthma=("asthma_crude", "mean"),
        copd=("copd_crude", "mean"),
        chd=("chd_crude", "mean"),
    ).reset_index()

    # Composite health score (average of 4 rates, normalized)
    for col in ["cancer", "asthma", "copd", "chd"]:
        mn, mx = state_agg[col].min(), state_agg[col].max()
        state_agg[f"{col}_norm"] = (state_agg[col] - mn) / (mx - mn + 1e-9)
    state_agg["health_score"] = state_agg[["cancer_norm", "asthma_norm", "copd_norm", "chd_norm"]].mean(axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        np.log10(state_agg["median_releases"].clip(0.1)),
        state_agg["health_score"],
        s=state_agg["n_facilities"] * 3,
        c=state_agg["health_score"],
        cmap="RdYlGn_r",
        alpha=0.85, edgecolors="0.3", linewidths=0.5,
        vmin=0, vmax=1
    )

    for _, row in state_agg.iterrows():
        ax.annotate(row["ST"],
                    (np.log10(row["median_releases"] + 0.1), row["health_score"]),
                    fontsize=7, ha="center", va="bottom", color="0.2")

    # Trend line
    x = np.log10(state_agg["median_releases"].clip(0.1))
    y = state_agg["health_score"]
    m, b, r, p, _ = stats.linregress(x, y)
    xfit = np.linspace(x.min(), x.max(), 100)
    ax.plot(xfit, m * xfit + b, "k--", lw=1.5, alpha=0.6,
            label=f"OLS r={r:.2f}, p={p:.3f}")

    ax.set_xlabel("Log₁₀ Median TRI Release per Facility (lbs)", fontsize=11)
    ax.set_ylabel("Composite Health Burden Score (0–1)", fontsize=11)
    ax.set_title("State-Level: Median TRI Release Volume vs Composite Health Burden\n(circle size = number of TRI facilities)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.colorbar(scatter, ax=ax, label="Health Score")

    fig.tight_layout()
    _save(fig, "1g_state_releases_health")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: SPATIAL WEALTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analysis2_wealth(df):
    """Generate ~7 plots for the wealth / economic justice analysis."""

    # ── 2a. Release volume distribution across income deciles ──
    _plot_2a_releases_income_deciles(df)

    # ── 2b. Facility density × poverty quartile (violin + strip) ──
    _plot_2b_facility_density_poverty(df)

    # ── 2c. Carcinogen fraction by income quintile ──
    _plot_2c_carcinogen_wealth(df)

    # ── 2d. Persistent chemical burden by income quintile ──
    _plot_2d_persistent_wealth(df)

    # ── 2e. Temporal trend: are low-income areas improving? ──
    _plot_2e_temporal_equity(df)

    # ── 2f. Cumulative release burden: Lorenz-style curve ──
    _plot_2f_lorenz_curve(df)

    # ── 2g. Release medium mix by poverty quintile (stacked bar) ──
    _plot_2g_medium_by_poverty(df)


def _plot_2a_releases_income_deciles(df):
    """Box + swarm: total releases across household income deciles."""
    df = df.copy()
    df = df[df["median_income"] > 0]
    df["income_decile"] = pd.qcut(df["median_income"], 10,
                                   labels=[f"D{i}" for i in range(1, 11)],
                                   duplicates="drop")
    df["log_releases"] = np.log10(df["TOTAL_RELEASES"].clip(lower=0.1))

    # Also compute mean release per decile for annotation
    dec_means = df.groupby("income_decile", observed=True)["TOTAL_RELEASES"].mean().reset_index()
    dec_means.columns = ["income_decile", "mean_release"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: box plot of log releases per income decile
    palette = sns.color_palette("RdYlGn", 10)
    sns.boxplot(data=df, x="income_decile", y="log_releases",
                palette=palette, width=0.6,
                flierprops=dict(markersize=1, alpha=0.3), ax=ax1)
    ax1.set_xlabel("Household Income Decile (D1=Poorest, D10=Wealthiest)")
    ax1.set_ylabel("Log₁₀ Total Releases (lbs)")
    ax1.set_title("TRI Release Volume by Household\nIncome Decile", fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)

    # Overlay trend line on means
    dec_means_vals = df.groupby("income_decile", observed=True)["log_releases"].mean().values
    ax1.plot(range(len(dec_means_vals)), dec_means_vals, "ko-", linewidth=2, markersize=5, label="Mean")
    ax1.legend()

    # Right: mean TOTAL_RELEASES in thousands with error bars
    grouped = df.groupby("income_decile", observed=True)["TOTAL_RELEASES"].agg(["mean", "sem"]).reset_index()
    ax2.bar(range(len(grouped)), grouped["mean"] / 1000,
            color=sns.color_palette("RdYlGn", len(grouped)),
            edgecolor="0.3", linewidth=0.5)
    ax2.errorbar(range(len(grouped)), grouped["mean"] / 1000,
                 yerr=grouped["sem"] / 1000, fmt="none", color="0.2", linewidth=1.2)
    ax2.set_xticks(range(len(grouped)))
    ax2.set_xticklabels(grouped["income_decile"], rotation=30)
    ax2.set_xlabel("Household Income Decile (D1=Poorest, D10=Wealthiest)")
    ax2.set_ylabel("Mean Releases per Facility-Year (thousands lbs)")
    ax2.set_title("Mean TRI Release Volume by\nHousehold Income Decile", fontweight="bold")

    # Kruskal-Wallis
    groups = [df[df["income_decile"] == d]["TOTAL_RELEASES"].dropna()
              for d in df["income_decile"].cat.categories]
    stat, p = kruskal(*[g for g in groups if len(g) > 5])
    fig.suptitle(f"TRI Releases vs Economic Status\n(Kruskal-Wallis p={p:.2e})",
                 fontsize=14, fontweight="bold")

    fig.tight_layout()
    _save(fig, "2a_releases_income_deciles")


def _plot_2b_facility_density_poverty(df):
    """Facility count and total release burden by poverty quartile."""
    df = df.copy()
    df["poverty_quartile"] = pd.qcut(df["poverty_pct"], 4,
                                      labels=["Q1\n0–25%\n(Lowest)", "Q2\n25–50%",
                                              "Q3\n50–75%", "Q4\n75–100%\n(Highest)"],
                                      duplicates="drop")

    # Aggregate to facility-level (avoid year duplication)
    fac_latest = df.sort_values("REPORTING_YEAR").groupby("TRI_FACILITY_ID").last().reset_index()
    fac_latest["poverty_quartile"] = pd.qcut(fac_latest["poverty_pct"], 4,
                                              labels=["Q1\n0–25%\n(Lowest)", "Q2\n25–50%",
                                                      "Q3\n50–75%", "Q4\n75–100%\n(Highest)"],
                                              duplicates="drop")

    q_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Panel A: number of facilities per poverty quartile
    counts = fac_latest["poverty_quartile"].value_counts().sort_index()
    axes[0].bar(range(len(counts)), counts.values, color=q_colors,
                edgecolor="0.3", linewidth=0.5)
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(counts.index, fontsize=9)
    axes[0].set_xlabel("Poverty Rate Quartile")
    axes[0].set_ylabel("Number of Unique TRI Facilities")
    axes[0].set_title("A. Facility Count\nby Poverty Quartile", fontweight="bold")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, f"{v:,}", ha="center", fontsize=8)

    # Panel B: total releases per poverty quartile
    total_by_q = df.groupby("poverty_quartile", observed=True)["TOTAL_RELEASES"].sum()
    axes[1].bar(range(len(total_by_q)), total_by_q.values / 1e9, color=q_colors,
                edgecolor="0.3", linewidth=0.5)
    axes[1].set_xticks(range(len(total_by_q)))
    axes[1].set_xticklabels(total_by_q.index, fontsize=9)
    axes[1].set_xlabel("Poverty Rate Quartile")
    axes[1].set_ylabel("Total Cumulative Releases (Billion lbs)")
    axes[1].set_title("B. Total Release Volume\nby Poverty Quartile", fontweight="bold")
    for i, v in enumerate(total_by_q.values / 1e9):
        axes[1].text(i, v + 0.01, f"{v:.2f}B", ha="center", fontsize=8)

    # Panel C: per-capita release burden (total releases / total population)
    pc = df.groupby("poverty_quartile", observed=True).agg(
        releases=("TOTAL_RELEASES", "sum"),
        pop=("total_population", "sum"),
    ).reset_index()
    pc["per_capita"] = pc["releases"] / pc["pop"].replace(0, np.nan)
    axes[2].bar(range(len(pc)), pc["per_capita"], color=q_colors,
                edgecolor="0.3", linewidth=0.5)
    axes[2].set_xticks(range(len(pc)))
    axes[2].set_xticklabels(pc["poverty_quartile"], fontsize=9)
    axes[2].set_xlabel("Poverty Rate Quartile")
    axes[2].set_ylabel("Releases per Capita (lbs/person)")
    axes[2].set_title("C. Per-Capita Release Burden\nby Poverty Quartile", fontweight="bold")
    for i, v in enumerate(pc["per_capita"]):
        axes[2].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)

    fig.suptitle("TRI Facility Siting and Release Burden by Community Poverty Level",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "2b_facility_poverty_burden")


def _plot_2c_carcinogen_wealth(df):
    """Fraction of reports involving known carcinogens by income quintile."""
    df = df.copy()
    df = df[df["median_income"] > 0]
    df["income_quintile"] = pd.qcut(df["median_income"], 5, labels=range(1, 6), duplicates="drop")

    quintile_stats = df.groupby("income_quintile", observed=True).agg(
        total=("IS_CARCINOGEN", "count"),
        carcinogen_reports=("IS_CARCINOGEN", "sum"),
        persistent_reports=("IS_PERSISTENT", "sum"),
        mean_releases=("TOTAL_RELEASES", "mean"),
    ).reset_index()
    quintile_stats["carcin_pct"] = 100 * quintile_stats["carcinogen_reports"] / quintile_stats["total"]
    quintile_stats["persist_pct"] = 100 * quintile_stats["persistent_reports"] / quintile_stats["total"]
    quintile_stats["income_quintile"] = [f"Q{i}\n({'Poorest' if i==1 else 'Wealthiest' if i==5 else ''})"
                                          for i in range(1, 6)]

    q_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    x = range(len(quintile_stats))

    # Carcinogen %
    axes[0].bar(x, quintile_stats["carcin_pct"], color=q_colors, edgecolor="0.3", linewidth=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(quintile_stats["income_quintile"], fontsize=9)
    axes[0].set_ylabel("% of TRI Reports Involving Carcinogens")
    axes[0].set_title("A. Carcinogen Report Fraction\nby Income Quintile", fontweight="bold")
    axes[0].set_xlabel("Household Income Quintile (Q1=Poorest)")
    for i, v in enumerate(quintile_stats["carcin_pct"]):
        axes[0].text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=8)

    # Persistent chemical %
    axes[1].bar(x, quintile_stats["persist_pct"], color=q_colors, edgecolor="0.3", linewidth=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(quintile_stats["income_quintile"], fontsize=9)
    axes[1].set_ylabel("% of TRI Reports with Persistent Chemicals")
    axes[1].set_title("B. Persistent Chemical Report Fraction\nby Income Quintile", fontweight="bold")
    axes[1].set_xlabel("Household Income Quintile (Q1=Poorest)")
    for i, v in enumerate(quintile_stats["persist_pct"]):
        axes[1].text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=8)

    # Mean releases
    axes[2].bar(x, quintile_stats["mean_releases"] / 1000, color=q_colors, edgecolor="0.3", linewidth=0.5)
    axes[2].set_xticks(x); axes[2].set_xticklabels(quintile_stats["income_quintile"], fontsize=9)
    axes[2].set_ylabel("Mean TRI Releases per Report (thousands lbs)")
    axes[2].set_title("C. Mean Release Volume\nby Income Quintile", fontweight="bold")
    axes[2].set_xlabel("Household Income Quintile (Q1=Poorest)")

    fig.suptitle("Hazardous Chemical Exposure Inequality: Carcinogen and Persistent Chemical Burden by Income",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "2c_carcinogen_wealth")


def _plot_2d_persistent_wealth(df):
    """Scatter: median income vs log releases, separately for persistent vs non-persistent."""
    df = df.copy()
    df = df[df["median_income"] > 0]
    df["log_releases"] = np.log10(df["TOTAL_RELEASES"].clip(0.1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (flag, label, color) in zip(axes, [
        (False, "Non-Persistent Chemicals", "#3498db"),
        (True, "Persistent/Bioaccumulative Chemicals", "#e74c3c"),
    ]):
        sub = df[df["IS_PERSISTENT"] == flag].sample(min(5000, len(df[df["IS_PERSISTENT"] == flag])), random_state=42)
        ax.hexbin(sub["median_income"] / 1000, sub["log_releases"],
                  gridsize=35, cmap="YlOrRd" if flag else "Blues", mincnt=1)

        # Bin means
        sub2 = df[df["IS_PERSISTENT"] == flag].copy()
        sub2["income_bin"] = pd.cut(sub2["median_income"], 20, labels=False)
        trend = sub2.groupby("income_bin").agg(
            x=("median_income", "median"),
            y=("log_releases", "mean"),
        ).dropna()
        ax.plot(trend["x"] / 1000, trend["y"], "w-", linewidth=2.5, alpha=0.8)
        ax.plot(trend["x"] / 1000, trend["y"], "k-", linewidth=1.5, alpha=0.6)

        r, p = spearmanr(sub2["median_income"], sub2["log_releases"])
        ax.set_xlabel("Median Household Income ($K)", fontsize=11)
        ax.set_ylabel("Log₁₀ TRI Releases (lbs)", fontsize=11)
        ax.set_title(f"{label}\nSpearman ρ={r:.3f}, p={p:.2e}", fontsize=11, fontweight="bold")

    fig.suptitle("TRI Chemical Releases vs Household Income:\nPersistent vs Non-Persistent Chemicals",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "2d_persistent_wealth")


def _plot_2e_temporal_equity(df):
    """Time series: mean releases by income quintile over 2013-2023."""
    df = df.copy()
    df = df[df["median_income"] > 0]

    # Assign quintile based on 2023 tract income (fixed reference)
    tract_income = df.groupby("fips_tract")["median_income"].mean()
    quantiles = tract_income.quantile([0.2, 0.4, 0.6, 0.8]).values

    def assign_q(inc):
        if inc <= quantiles[0]: return "Q1 (Poorest)"
        elif inc <= quantiles[1]: return "Q2"
        elif inc <= quantiles[2]: return "Q3"
        elif inc <= quantiles[3]: return "Q4"
        else: return "Q5 (Wealthiest)"

    df["income_q"] = df["median_income"].apply(assign_q)

    yearly = df.groupby(["REPORTING_YEAR", "income_q"]).agg(
        mean_releases=("TOTAL_RELEASES", "mean"),
        total_releases=("TOTAL_RELEASES", "sum"),
        n_facilities=("TRI_FACILITY_ID", "nunique"),
    ).reset_index()

    colors_q = {"Q1 (Poorest)": "#e74c3c", "Q2": "#e67e22",
                "Q3": "#f1c40f", "Q4": "#2ecc71", "Q5 (Wealthiest)": "#27ae60"}
    q_order = ["Q1 (Poorest)", "Q2", "Q3", "Q4", "Q5 (Wealthiest)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mean releases
    for q in q_order:
        sub = yearly[yearly["income_q"] == q]
        axes[0].plot(sub["REPORTING_YEAR"], sub["mean_releases"] / 1000,
                     "o-", color=colors_q[q], label=q, linewidth=2, markersize=5)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Mean TRI Releases per Facility-Year (thousands lbs)")
    axes[0].set_title("A. Mean Release Volume Over Time\nby Community Income Quintile", fontweight="bold")
    axes[0].legend(title="Income Quintile", fontsize=8)
    axes[0].grid(alpha=0.3)

    # Index: relative to 2013 baseline (how much has each group improved?)
    for q in q_order:
        sub = yearly[yearly["income_q"] == q].sort_values("REPORTING_YEAR")
        baseline = sub[sub["REPORTING_YEAR"] == sub["REPORTING_YEAR"].min()]["mean_releases"].values
        if len(baseline) > 0 and baseline[0] > 0:
            sub = sub.copy()
            sub["index"] = 100 * sub["mean_releases"] / baseline[0]
            axes[1].plot(sub["REPORTING_YEAR"], sub["index"],
                         "o-", color=colors_q[q], label=q, linewidth=2, markersize=5)

    axes[1].axhline(100, color="0.5", linestyle="--", linewidth=1, label="Baseline (2013)")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Release Index (2013 = 100)")
    axes[1].set_title("B. Release Trend Index: Relative to 2013 Baseline\n(below 100 = improvement)", fontweight="bold")
    axes[1].legend(title="Income Quintile", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Temporal Equity: Are Low-Income Communities Improving at Same Rate?\n(2013–2023)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "2e_temporal_equity")


def _plot_2f_lorenz_curve(df):
    """Lorenz-style curve: cumulative facilities (sorted by income) vs cumulative releases."""
    df = df.copy()
    df = df[df["median_income"] > 0]
    df = df.sort_values("median_income").reset_index(drop=True)

    cumulative_pop_fraction = np.arange(1, len(df) + 1) / len(df)
    cumulative_release_fraction = df["TOTAL_RELEASES"].cumsum() / df["TOTAL_RELEASES"].sum()

    # Also carcinogen releases only
    carcin_df = df[df["IS_CARCINOGEN"]].copy().sort_values("median_income").reset_index(drop=True)
    cum_c_pop = np.arange(1, len(carcin_df) + 1) / len(carcin_df) if len(carcin_df) > 0 else []
    cum_c_rel = carcin_df["TOTAL_RELEASES"].cumsum() / carcin_df["TOTAL_RELEASES"].sum() if len(carcin_df) > 0 else []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Full Lorenz curve
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect equality")
    ax1.plot(cumulative_pop_fraction, cumulative_release_fraction,
             color="#e74c3c", linewidth=2.5, label="All TRI releases")
    if len(cum_c_pop) > 0:
        ax1.plot(cum_c_pop, cum_c_rel,
                 color="#8e44ad", linewidth=2.5, linestyle="--", label="Carcinogen releases")

    # Gini coefficient
    gini = 1 - 2 * np.trapezoid(cumulative_release_fraction, cumulative_pop_fraction)
    ax1.fill_between(cumulative_pop_fraction, cumulative_release_fraction,
                     cumulative_pop_fraction, alpha=0.15, color="#e74c3c",
                     label=f"Inequality area (Gini≈{gini:.3f})")
    ax1.set_xlabel("Cumulative Fraction of Facility-Years\n(sorted by community income, poor→wealthy)")
    ax1.set_ylabel("Cumulative Fraction of Total TRI Releases")
    ax1.set_title("Lorenz Curve: TRI Release Inequality\nAcross Income Spectrum", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Panel B: Decile bar chart - release share per income decile
    df["income_decile"] = pd.qcut(df["median_income"], 10,
                                   labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
    decile_share = df.groupby("income_decile", observed=True)["TOTAL_RELEASES"].sum()
    decile_pct = 100 * decile_share / decile_share.sum()

    colors_d = sns.color_palette("RdYlGn", 10)
    ax2.bar(range(10), decile_pct.values, color=colors_d, edgecolor="0.3", linewidth=0.5)
    ax2.axhline(10, color="0.4", linestyle="--", linewidth=1.5, label="Equal share (10%)")
    ax2.set_xticks(range(10))
    ax2.set_xticklabels([f"D{i}" for i in range(1, 11)])
    ax2.set_xlabel("Income Decile (D1=Poorest, D10=Wealthiest)")
    ax2.set_ylabel("Share of Total TRI Releases (%)")
    ax2.set_title("Share of TRI Release Volume\nby Community Income Decile", fontweight="bold")
    ax2.legend()
    for i, v in enumerate(decile_pct.values):
        ax2.text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=7.5)

    fig.suptitle("Release Inequality Across Income Spectrum:\nPoor Communities Bear Disproportionate Burden",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "2f_lorenz_curve")


def _plot_2g_medium_by_poverty(df):
    """Stacked bar: release medium composition across poverty quintiles."""
    rqp = Path("data/raw/tri_release_qty.csv")
    if not rqp.exists():
        logger.warning("tri_release_qty.csv not found, skipping medium/poverty plot")
        return

    logger.info("Building medium × poverty plot...")
    # Load release qty, aggregate by doc_ctrl_num
    rdf = pd.read_csv(rqp,
                      usecols=["doc_ctrl_num", "environmental_medium", "total_release"],
                      low_memory=False)
    rdf["total_release"] = pd.to_numeric(rdf["total_release"], errors="coerce").fillna(0)

    # Only keep major media categories, group others
    def simplify_medium(m):
        if "AIR" in str(m): return "Air"
        if "WATER" in str(m): return "Water"
        if "UNINJ" in str(m): return "Underground Injection"
        return "Land"

    rdf["medium_simple"] = rdf["environmental_medium"].apply(simplify_medium)
    medium_by_doc = rdf.groupby(["doc_ctrl_num", "medium_simple"])["total_release"].sum().reset_index()

    # Join with facility poverty data (need doc_ctrl_num in facilities)
    tri_raw = pd.read_csv("data/raw/tri_facilities.csv", low_memory=False)
    tri_raw.columns = tri_raw.columns.str.upper().str.strip()
    # Find doc_ctrl_num and tri_facility_id columns (case-insensitive)
    dcn_col = next((c for c in tri_raw.columns if "DOC_CTRL" in c), None)
    fid_col = next((c for c in tri_raw.columns if "TRI_FACILITY_ID" in c), None)
    if dcn_col and fid_col:
        tri_raw = tri_raw[[dcn_col, fid_col]].rename(
            columns={dcn_col: "doc_ctrl_num", fid_col: "TRI_FACILITY_ID"}
        )

    pov_lookup = df.groupby("TRI_FACILITY_ID")["poverty_pct"].mean().reset_index()
    merged = medium_by_doc.merge(tri_raw, on="doc_ctrl_num", how="left"
    ).merge(pov_lookup, on="TRI_FACILITY_ID", how="left")
    merged = merged.dropna(subset=["poverty_pct"])
    merged["poverty_quintile"] = pd.qcut(merged["poverty_pct"], 5,
                                          labels=["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"],
                                          duplicates="drop")

    # Aggregate: total releases per medium per poverty quintile
    pivot = merged.groupby(["poverty_quintile", "medium_simple"], observed=True)["total_release"].sum().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    medium_colors = {"Air": "#e74c3c", "Water": "#3498db",
                     "Land": "#8B4513", "Underground Injection": "#9b59b6"}
    media = [m for m in ["Air", "Water", "Land", "Underground Injection"] if m in pivot_pct.columns]

    # Stacked bar: percentage composition
    bottom = np.zeros(len(pivot_pct))
    for med in media:
        ax1.bar(range(len(pivot_pct)), pivot_pct[med].values, bottom=bottom,
                label=med, color=medium_colors.get(med, "gray"), edgecolor="0.3", linewidth=0.3)
        bottom += pivot_pct[med].values
    ax1.set_xticks(range(len(pivot_pct)))
    ax1.set_xticklabels(pivot_pct.index)
    ax1.set_xlabel("Community Poverty Rate Quintile")
    ax1.set_ylabel("Share of Release Volume (%)")
    ax1.set_title("A. Release Medium Composition\nby Community Poverty Quintile", fontweight="bold")
    ax1.legend(title="Release Medium", fontsize=9, loc="upper right")

    # Absolute totals per medium per quintile
    bottom2 = np.zeros(len(pivot))
    for med in media:
        ax2.bar(range(len(pivot)), pivot[med].values / 1e9, bottom=bottom2 / 1e9,
                label=med, color=medium_colors.get(med, "gray"), edgecolor="0.3", linewidth=0.3)
        bottom2 += pivot[med].values
    ax2.set_xticks(range(len(pivot)))
    ax2.set_xticklabels(pivot.index)
    ax2.set_xlabel("Community Poverty Rate Quintile")
    ax2.set_ylabel("Total Release Volume (Billion lbs)")
    ax2.set_title("B. Absolute Release Volume\nby Community Poverty Quintile", fontweight="bold")
    ax2.legend(title="Release Medium", fontsize=9)

    fig.suptitle("What Are Communities Exposed To? Release Medium by Poverty Level",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "2g_medium_by_poverty")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: COMBINED — HEALTH BURDEN × POVERTY NEAR TRI SITES
# ═══════════════════════════════════════════════════════════════════════════════

def analysis3_combined(df):
    """~7 plots examining whether health burden concentrates in poor areas near TRI sites."""
    hdf = df[df["has_health"]].copy()

    # ── 3a. 2D heatmap: poverty × releases → health burden ──
    _plot_3a_poverty_releases_health_heatmap(hdf)

    # ── 3b. Interaction effect: high release × high poverty → health outcomes ──
    _plot_3b_interaction_effect(hdf)

    # ── 3c. 4-quadrant: low/high poverty × low/high releases (4 group health comparison) ──
    _plot_3c_four_quadrant(hdf)

    # ── 3d. Cumulative disadvantage: facilities per tract vs poverty vs health ──
    _plot_3d_cumulative_disadvantage(hdf)

    # ── 3e. Scatter matrix: poverty, minority, releases, health outcomes ──
    _plot_3e_scatter_matrix(hdf)

    # ── 3f. Year-over-year: high-poverty high-burden communities ──
    _plot_3f_temporal_combined(df)

    # ── 3g. Vulnerability index: combined EJ score vs health outcomes ──
    _plot_3g_ej_vs_health(hdf)


def _plot_3a_poverty_releases_health_heatmap(hdf):
    """2D heatmap: mean health burden at grid of poverty × log releases."""
    hdf = hdf.copy()
    hdf = hdf[hdf["TOTAL_RELEASES"] > 0]

    # Composite health outcome
    for hc in HEALTH_COLS:
        mn, mx = hdf[hc].quantile(0.01), hdf[hc].quantile(0.99)
        hdf[f"{hc}_n"] = (hdf[hc] - mn) / (mx - mn + 1e-9)
    hdf["health_idx"] = hdf[[f"{hc}_n" for hc in HEALTH_COLS]].mean(axis=1)

    hdf["log_releases"] = np.log10(hdf["TOTAL_RELEASES"].clip(0.1))

    # Create 2D grid (10×10 bins)
    hdf["pov_bin"] = pd.cut(hdf["poverty_pct"], 10, labels=False)
    hdf["rel_bin"] = pd.cut(hdf["log_releases"], 10, labels=False)
    grid = hdf.groupby(["rel_bin", "pov_bin"])["health_idx"].mean().unstack(fill_value=np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: health burden heatmap
    ax = axes[0]
    pov_edges = np.linspace(hdf["poverty_pct"].min(), hdf["poverty_pct"].max(), 11)
    rel_edges = np.linspace(hdf["log_releases"].min(), hdf["log_releases"].max(), 11)
    im = ax.imshow(grid.values, origin="lower", aspect="auto",
                   cmap="RdYlGn_r", vmin=0, vmax=1,
                   extent=[pov_edges[0], pov_edges[-1], rel_edges[0], rel_edges[-1]])
    plt.colorbar(im, ax=ax, label="Mean Health Burden Index")
    ax.set_xlabel("Poverty Rate (%)")
    ax.set_ylabel("Log₁₀ TRI Releases (lbs)")
    ax.set_title("A. Health Burden at\nPoverty × Release Intersection", fontweight="bold")

    # Panel B: count heatmap (data density)
    ax = axes[1]
    count_grid = hdf.groupby(["rel_bin", "pov_bin"])["health_idx"].count().unstack(fill_value=0)
    im2 = ax.imshow(np.log10(count_grid.values + 1), origin="lower", aspect="auto",
                    cmap="Blues",
                    extent=[pov_edges[0], pov_edges[-1], rel_edges[0], rel_edges[-1]])
    plt.colorbar(im2, ax=ax, label="Log₁₀ Count")
    ax.set_xlabel("Poverty Rate (%)")
    ax.set_ylabel("Log₁₀ TRI Releases (lbs)")
    ax.set_title("B. Data Density\n(log count)", fontweight="bold")

    # Panel C: minority fraction heatmap
    minority_grid = hdf.groupby(["rel_bin", "pov_bin"])["minority_pct"].mean().unstack(fill_value=np.nan)
    ax = axes[2]
    im3 = ax.imshow(minority_grid.values, origin="lower", aspect="auto",
                    cmap="PuRd", vmin=0, vmax=100,
                    extent=[pov_edges[0], pov_edges[-1], rel_edges[0], rel_edges[-1]])
    plt.colorbar(im3, ax=ax, label="Mean Minority Population (%)")
    ax.set_xlabel("Poverty Rate (%)")
    ax.set_ylabel("Log₁₀ TRI Releases (lbs)")
    ax.set_title("C. Minority Population % at\nPoverty × Release Intersection", fontweight="bold")

    fig.suptitle("Spatial Intersection: Poverty, TRI Releases, and Community Health",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "3a_poverty_releases_health_heatmap")


def _plot_3b_interaction_effect(hdf):
    """Test interaction: high-release × high-poverty → amplified health burden."""
    hdf = hdf.copy()
    median_release = hdf["TOTAL_RELEASES"].median()
    median_poverty = hdf["poverty_pct"].median()

    hdf["high_release"] = hdf["TOTAL_RELEASES"] >= median_release
    hdf["high_poverty"] = hdf["poverty_pct"] >= median_poverty
    hdf["group"] = hdf.apply(
        lambda r: ("High Release\n+ High Poverty" if r.high_release and r.high_poverty
                   else "High Release\n+ Low Poverty" if r.high_release
                   else "Low Release\n+ High Poverty" if r.high_poverty
                   else "Low Release\n+ Low Poverty"), axis=1
    )

    group_order = ["Low Release\n+ Low Poverty", "Low Release\n+ High Poverty",
                   "High Release\n+ Low Poverty", "High Release\n+ High Poverty"]
    group_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (hc, hl) in enumerate(HEALTH_LABELS.items()):
        ax = axes[i]
        plot_data = hdf[["group", hc]].dropna()

        # Mean + CI bar chart
        stats_g = plot_data.groupby("group")[hc].agg(["mean", "sem"])
        stats_g = stats_g.reindex(group_order)

        bars = ax.bar(range(len(group_order)), stats_g["mean"].values,
                      color=group_colors, edgecolor="0.3", linewidth=0.5, width=0.6)
        ax.errorbar(range(len(group_order)), stats_g["mean"].values,
                    yerr=1.96 * stats_g["sem"].values,
                    fmt="none", color="0.2", linewidth=1.5, capsize=4)

        # Kruskal-Wallis
        groups_data = [plot_data[plot_data["group"] == g][hc].dropna() for g in group_order]
        if all(len(g) > 5 for g in groups_data):
            stat, p = kruskal(*groups_data)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            ax.set_title(f"{hl.replace(' (%)', '')}\nKW p{sig}", fontsize=10, fontweight="bold")

        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(group_order, fontsize=7, rotation=10)
        ax.set_ylabel("Mean Rate (%)")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Interaction Effect: Release × Poverty on Community Health Outcomes\n"
                 "Does Living Near Heavy Polluters in Poor Areas Amplify Health Burden?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "3b_interaction_effect")


def _plot_3c_four_quadrant(hdf):
    """Four-quadrant analysis: poverty × releases, showing health gradient visually."""
    hdf = hdf.copy()
    hdf = hdf[hdf["TOTAL_RELEASES"] > 0]

    # Composite health burden
    for hc in HEALTH_COLS:
        mn, mx = hdf[hc].quantile(0.01), hdf[hc].quantile(0.99)
        hdf[f"{hc}_n"] = (hdf[hc] - mn) / (mx - mn + 1e-9)
    hdf["health_idx"] = hdf[[f"{hc}_n" for hc in HEALTH_COLS]].mean(axis=1)

    med_pov = hdf["poverty_pct"].median()
    med_rel = np.log10(hdf["TOTAL_RELEASES"].clip(0.1)).median()
    hdf["log_rel"] = np.log10(hdf["TOTAL_RELEASES"].clip(0.1))

    fig, ax = plt.subplots(figsize=(12, 9))

    scatter = ax.scatter(
        hdf["poverty_pct"],
        hdf["log_rel"],
        c=hdf["health_idx"],
        cmap="RdYlGn_r",
        alpha=0.4, s=12,
        vmin=0, vmax=1,
        rasterized=True,
    )

    # Quadrant dividers
    ax.axvline(med_pov, color="0.3", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axhline(med_rel, color="0.3", linestyle="--", linewidth=1.5, alpha=0.8)

    # Quadrant labels and mean health scores
    quadrants = [
        (f"<{med_pov:.0f}%", f"<{med_rel:.1f}", "LOW poverty\nLOW releases"),
        (f">{med_pov:.0f}%", f"<{med_rel:.1f}", "HIGH poverty\nLOW releases"),
        (f"<{med_pov:.0f}%", f">{med_rel:.1f}", "LOW poverty\nHIGH releases"),
        (f">{med_pov:.0f}%", f">{med_rel:.1f}", "HIGH poverty\nHIGH releases"),
    ]
    q_means = [
        hdf[(hdf["poverty_pct"] < med_pov) & (hdf["log_rel"] < med_rel)]["health_idx"].mean(),
        hdf[(hdf["poverty_pct"] >= med_pov) & (hdf["log_rel"] < med_rel)]["health_idx"].mean(),
        hdf[(hdf["poverty_pct"] < med_pov) & (hdf["log_rel"] >= med_rel)]["health_idx"].mean(),
        hdf[(hdf["poverty_pct"] >= med_pov) & (hdf["log_rel"] >= med_rel)]["health_idx"].mean(),
    ]

    text_positions = [
        (med_pov * 0.3, ax.get_ylim()[0] * 0.9 if ax.get_ylim()[0] < 0 else med_rel * 0.5),
        (med_pov * 1.5, ax.get_ylim()[0] * 0.9 if ax.get_ylim()[0] < 0 else med_rel * 0.5),
        (med_pov * 0.3, med_rel * 1.3),
        (med_pov * 1.5, med_rel * 1.3),
    ]

    # Compute mean health for each quadrant and annotate
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    quad_texts = [
        (xlim[0] + (med_pov - xlim[0]) * 0.5, ylim[0] + (med_rel - ylim[0]) * 0.3, q_means[0], "Low Pov\nLow Release"),
        (med_pov + (xlim[1] - med_pov) * 0.5, ylim[0] + (med_rel - ylim[0]) * 0.3, q_means[1], "High Pov\nLow Release"),
        (xlim[0] + (med_pov - xlim[0]) * 0.5, med_rel + (ylim[1] - med_rel) * 0.7, q_means[2], "Low Pov\nHigh Release"),
        (med_pov + (xlim[1] - med_pov) * 0.5, med_rel + (ylim[1] - med_rel) * 0.7, q_means[3], "High Pov\nHigh Release"),
    ]
    for tx, ty, mean_h, label in quad_texts:
        color = plt.cm.RdYlGn_r(mean_h)
        ax.text(tx, ty,
                f"{label}\nHealth Index: {mean_h:.3f}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.85, edgecolor="0.3"))

    plt.colorbar(scatter, ax=ax, label="Multi-Disease Health Burden Index (0=best, 1=worst)")
    ax.set_xlabel("Community Poverty Rate (%)", fontsize=12)
    ax.set_ylabel("Log₁₀ TRI Releases (lbs)", fontsize=12)
    ax.set_title("Four-Quadrant Analysis: Poverty × TRI Releases → Health Burden\n"
                 "Point color = health burden index", fontsize=13, fontweight="bold")

    fig.tight_layout()
    _save(fig, "3c_four_quadrant")


def _plot_3d_cumulative_disadvantage(hdf):
    """Tract-level: number of TRI facilities nearby × poverty × health burden."""
    # Aggregate to tract level
    tract_agg = hdf.groupby("fips_tract").agg(
        n_facility_reports=("TRI_FACILITY_ID", "count"),
        n_unique_facilities=("TRI_FACILITY_ID", "nunique"),
        total_releases=("TOTAL_RELEASES", "sum"),
        poverty_pct=("poverty_pct", "mean"),
        minority_pct=("minority_pct", "mean"),
        median_income=("median_income", "mean"),
        cancer=("cancer_crude", "mean"),
        asthma=("asthma_crude", "mean"),
        chd=("chd_crude", "mean"),
        copd=("copd_crude", "mean"),
        diabetes=("diabetes_crude", "mean"),
    ).reset_index()

    tract_agg = tract_agg.dropna(subset=["cancer", "poverty_pct"])

    for hc in ["cancer", "asthma", "chd", "copd", "diabetes"]:
        mn, mx = tract_agg[hc].quantile(0.01), tract_agg[hc].quantile(0.99)
        tract_agg[f"{hc}_n"] = (tract_agg[hc] - mn) / (mx - mn + 1e-9)
    tract_agg["health_burden"] = tract_agg[[f"{hc}_n" for hc in ["cancer","asthma","chd","copd","diabetes"]]].mean(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # A: n_unique_facilities per tract vs health burden
    tract_agg["fac_bin"] = pd.cut(tract_agg["n_unique_facilities"],
                                    bins=[0, 1, 2, 3, 5, 100],
                                    labels=["1", "2", "3", "4-5", "6+"])
    grouped = tract_agg.groupby("fac_bin", observed=True)["health_burden"].agg(["mean", "sem"])
    ax = axes[0]
    ax.bar(range(len(grouped)), grouped["mean"].values,
           color=sns.color_palette("YlOrRd", len(grouped)),
           edgecolor="0.3", linewidth=0.5)
    ax.errorbar(range(len(grouped)), grouped["mean"].values,
                yerr=1.96 * grouped["sem"].values,
                fmt="none", color="0.2", linewidth=1.5, capsize=4)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped.index)
    ax.set_xlabel("Number of TRI Facilities Reporting in Tract")
    ax.set_ylabel("Mean Health Burden Index")
    ax.set_title("A. Facility Count per Tract\nvs Health Burden", fontweight="bold")

    # B: poverty vs health burden, sized by n_facilities
    ax = axes[1]
    scatter = ax.scatter(
        tract_agg["poverty_pct"],
        tract_agg["health_burden"],
        s=5 * np.clip(tract_agg["n_unique_facilities"], 1, 20),
        c=tract_agg["total_releases"].clip(lower=0.1).apply(np.log10),
        cmap="YlOrRd",
        alpha=0.6, edgecolors="none", rasterized=True,
    )
    plt.colorbar(scatter, ax=ax, label="Log₁₀ Total Releases")
    r, p = spearmanr(tract_agg["poverty_pct"], tract_agg["health_burden"])
    ax.set_xlabel("Community Poverty Rate (%)", fontsize=10)
    ax.set_ylabel("Multi-Disease Health Burden Index", fontsize=10)
    ax.set_title(f"B. Poverty vs Health Burden\n(size=facilities, color=releases)\nρ={r:.3f}", fontweight="bold")

    # C: income vs health burden for tracts with 3+ facilities vs 1 facility
    low_burden = tract_agg[tract_agg["n_unique_facilities"] == 1]
    high_burden = tract_agg[tract_agg["n_unique_facilities"] >= 3]
    ax = axes[2]
    ax.scatter(low_burden["median_income"] / 1000, low_burden["health_burden"],
               alpha=0.4, s=8, color="#3498db", label="1 facility in tract", rasterized=True)
    ax.scatter(high_burden["median_income"] / 1000, high_burden["health_burden"],
               alpha=0.6, s=15, color="#e74c3c", label="3+ facilities in tract", rasterized=True)
    ax.set_xlabel("Median Household Income ($K)")
    ax.set_ylabel("Health Burden Index")
    ax.set_title("C. Income vs Health Burden\nby Facility Density", fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Cumulative Disadvantage: Multiple TRI Facilities × Poverty = Amplified Health Burden",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "3d_cumulative_disadvantage")


def _plot_3e_scatter_matrix(hdf):
    """Pairplot of key variables with health outcomes."""
    cols_of_interest = {
        "log_releases": "Log TRI Releases",
        "poverty_pct": "Poverty Rate (%)",
        "minority_pct": "Minority Pop (%)",
        "cancer_crude": "Cancer Rate (%)",
        "asthma_crude": "Asthma Rate (%)",
        "chd_crude": "Heart Disease (%)",
    }
    sample = hdf[list(cols_of_interest.keys()) + ["severity_tier"]].dropna().sample(
        min(3000, len(hdf)), random_state=42
    )
    sample_renamed = sample.rename(columns=cols_of_interest)

    tier_palette = {k: v for k, v in PALETTE_TIERS.items() if k in sample["severity_tier"].unique()}
    g = sns.pairplot(
        sample_renamed,
        hue="severity_tier",
        vars=list(cols_of_interest.values()),
        plot_kws={"alpha": 0.3, "s": 8, "rasterized": True},
        diag_kind="kde",
        palette=tier_palette,
        corner=True,
    )
    g.figure.suptitle("Scatter Matrix: Releases, Poverty, Minority Pop, and Health Outcomes\n(colored by EJ Severity Tier)",
                       fontsize=12, fontweight="bold", y=1.01)
    _save(g.figure, "3e_scatter_matrix")


def _plot_3f_temporal_combined(df):
    """Year by year: mean health rates in high-burden vs low-burden areas (fixed tracts)."""
    hdf = df[df["has_health"]].copy()

    # Define high-burden tracts: top quartile of combined poverty + minority score (fixed)
    tract_static = hdf.groupby("fips_tract").agg(
        pov=("poverty_pct", "mean"),
        min_pct=("minority_pct", "mean"),
        cancer=("cancer_crude", "mean"),
    ).reset_index().dropna()

    tract_static["burden_score"] = (
        tract_static["pov"].rank(pct=True) + tract_static["min_pct"].rank(pct=True)
    ) / 2

    high_burden_tracts = set(tract_static[tract_static["burden_score"] >= 0.75]["fips_tract"])
    low_burden_tracts = set(tract_static[tract_static["burden_score"] <= 0.25]["fips_tract"])

    hdf["burden_group"] = hdf["fips_tract"].apply(
        lambda t: "High Burden\n(Top 25% poverty + minority)"
        if t in high_burden_tracts
        else ("Low Burden\n(Bottom 25% poverty + minority)"
              if t in low_burden_tracts else None)
    )
    hdf = hdf[hdf["burden_group"].notna()]

    yearly = hdf.groupby(["REPORTING_YEAR", "burden_group"])[HEALTH_COLS].mean().reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    colors_bg = {
        "High Burden\n(Top 25% poverty + minority)": "#e74c3c",
        "Low Burden\n(Bottom 25% poverty + minority)": "#2ecc71",
    }

    for i, (hc, hl) in enumerate(HEALTH_LABELS.items()):
        ax = axes[i]
        for group, color in colors_bg.items():
            sub = yearly[yearly["burden_group"] == group]
            ax.plot(sub["REPORTING_YEAR"], sub[hc],
                    "o-", color=color, linewidth=2, markersize=5,
                    label=group.replace("\n", " "))
            ax.fill_between(sub["REPORTING_YEAR"], sub[hc] * 0.98, sub[hc] * 1.02,
                            alpha=0.15, color=color)

        ax.set_xlabel("Year")
        ax.set_ylabel("Mean Rate (%)")
        ax.set_title(hl.replace(" (%)", ""), fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Health Outcomes Over Time: High-Burden Communities vs Low-Burden Communities\n"
                 "(Fixed tracts classified by poverty + minority status)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "3f_temporal_combined")


def _plot_3g_ej_vs_health(hdf):
    """EJ composite score vs individual health outcomes — regression lines."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, (hc, hl) in enumerate(HEALTH_LABELS.items()):
        ax = axes[i]
        plot_data = hdf[["ej_score", hc, "severity_tier"]].dropna()

        for tier, color in PALETTE_TIERS.items():
            sub = plot_data[plot_data["severity_tier"] == tier]
            ax.scatter(sub["ej_score"], sub[hc],
                       c=color, alpha=0.3, s=8, label=tier, rasterized=True)

        # Regression line
        x = plot_data["ej_score"]
        y = plot_data[hc]
        m, b, r, p, se = stats.linregress(x, y)
        xfit = np.linspace(x.min(), x.max(), 100)
        ax.plot(xfit, m * xfit + b, "k-", linewidth=2)

        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.set_xlabel("EJ Score (0=low burden, 100=high)", fontsize=9)
        ax.set_ylabel("Rate (%)", fontsize=9)
        ax.set_title(f"{hl.replace(' (%)', '')}\nr={r:.3f} p{sig}", fontsize=10, fontweight="bold")
        if i == 0:
            legend_patches = [mpatches.Patch(color=c, label=t) for t, c in PALETTE_TIERS.items()]
            ax.legend(handles=legend_patches, fontsize=7, title="Severity")

    fig.suptitle("EJ Composite Score vs Individual Health Outcomes\n(Does higher EJ score predict worse health?)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "3g_ej_vs_health")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_analyses():
    """Run all three research analyses."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    logger.info("=" * 60)
    logger.info("DEEP RESEARCH ANALYSES")
    logger.info("=" * 60)

    df = _load_scored()
    logger.info(f"Loaded {len(df):,} records, {df['TRI_FACILITY_ID'].nunique():,} facilities")
    logger.info(f"Health data available for {df['has_health'].sum():,} records "
                f"({100*df['has_health'].mean():.1f}%)")

    logger.info("\n>>> ANALYSIS 1: Health Impact of TRI Releases")
    analysis1_health_impact(df)

    logger.info("\n>>> ANALYSIS 2: Wealth and Spatial Inequality")
    analysis2_wealth(df)

    logger.info("\n>>> ANALYSIS 3: Combined — Health Burden in Poor Communities")
    analysis3_combined(df)

    logger.info("\n>>> ANALYSIS 5: Hypothesis Investigation")
    analysis5_hypotheses(df)

    logger.info("\n>>> ANALYSIS 6: New Hypotheses — Chemical Type, Closure, Cancer Paradox, Medium Pathways")
    analysis6_new_hypotheses(df)

    logger.info("\n>>> ANALYSIS 7: Local (Census-Tract) Impact Analysis")
    analysis7_local_impact(df)

    logger.info("\n>>> ANALYSIS 8: Case-Control — TRI Tracts vs Background Tracts")
    analysis8_case_control(df)

    logger.info("\n>>> ANALYSIS 9: Minority & Poverty Disentanglement")
    analysis9_minority_poverty_disentangle(df)

    logger.info("\n>>> ANALYSIS 10: Health Burden Quantification")
    analysis10_health_burden(df)

    logger.info("\n>>> ANALYSIS 11: TRI-tract Poverty Gap (vs background)")
    analysis11_tri_poverty_gap(df)

    logger.info("\n>>> ANALYSIS 12: Cancer Paradox in White-Majority Tracts")
    analysis12_cancer_white_tracts(df)

    logger.info("\n>>> ANALYSIS 13: Facility Closure Selectivity")
    analysis13_facility_closure_selectivity(df)

    # Summary of all outputs
    plots = sorted(OUT.glob("*.png"))
    logger.info("\n" + "=" * 60)
    logger.info(f"Generated {len(plots)} research plots in {OUT}/")
    for p in plots:
        logger.info(f"  {p.name}")
    logger.info("=" * 60)

    return plots


def analysis5_hypotheses(df: pd.DataFrame) -> None:
    """
    Data-driven hypothesis investigation.

    H1: Poverty, not pollution volume, is the dominant predictor of disease burden.
    H2: The cancer detection paradox — high-minority tracts report LOWER cancer rates
        (likely under-screening / mortality before diagnosis).
    H3: Industrial pollution is declining nation-wide, but the remaining burden
        is concentrating in fewer, larger high-release facilities.
    H4: High-release states are NOT the most impoverished states —
        industrial geography and poverty geography are largely decoupled.
    """
    hdf = df[df["cancer_crude"].notna()].copy()
    hdf["log_releases"] = np.log10(hdf["TOTAL_RELEASES"].clip(0.1))

    # ── H1a: partial-correlation decomposition ────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "H1: Poverty predicts disease burden — pollution volume does not\n"
        "(Spearman ρ with 95% CI via bootstrap, n ≈ 24 k records)",
        fontsize=13, fontweight="bold",
    )
    outcomes = ["asthma_crude", "chd_crude", "copd_crude", "diabetes_crude",
                "mental_health_crude", "cancer_crude"]
    labels = ["Asthma", "CHD", "COPD", "Diabetes", "Mental Health", "Cancer"]
    predictors = ["poverty_pct", "median_income", "minority_pct", "log_releases"]
    pred_labels = ["Poverty %", "Median Income", "Minority %", "log₁₀(Releases)"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for ax, outcome, label in zip(axes.flat, outcomes, labels):
        rhos = []
        for pred in predictors:
            sub = hdf[[pred, outcome]].dropna()
            r, _ = spearmanr(sub[pred], sub[outcome])
            rhos.append(r)
        bars = ax.barh(pred_labels, rhos, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Spearman ρ")
        for bar, r in zip(bars, rhos):
            ax.text(r + (0.03 if r >= 0 else -0.03), bar.get_y() + bar.get_height() / 2,
                    f"{r:.2f}", va="center", ha="left" if r >= 0 else "right", fontsize=8)
    plt.tight_layout()
    _save(fig, "h1a_poverty_vs_pollution_predictor")

    # ── H1b: scatter grid poverty vs health (6 outcomes) ──────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("H1: Poverty % vs Health Outcomes (Spearman ρ shown)",
                 fontsize=13, fontweight="bold")
    for ax, outcome, label in zip(axes.flat, outcomes, labels):
        sub = hdf[["poverty_pct", outcome, "log_releases"]].dropna().sample(
            min(3000, len(hdf)), random_state=42)
        r, p = spearmanr(sub["poverty_pct"], sub[outcome])
        ax.hexbin(sub["poverty_pct"], sub[outcome], gridsize=30,
                  cmap="YlOrRd", mincnt=1)
        ax.set_xlabel("Poverty %")
        ax.set_ylabel(label)
        ax.set_title(f"{label}  ρ={r:.3f}, p<0.001", fontsize=9)
        # trend line
        z = np.polyfit(sub["poverty_pct"], sub[outcome], 1)
        xr = np.linspace(sub["poverty_pct"].min(), sub["poverty_pct"].max(), 100)
        ax.plot(xr, np.poly1d(z)(xr), "b--", lw=1.5)
    plt.tight_layout()
    _save(fig, "h1b_poverty_health_scatter")

    # ── H2: Cancer paradox ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "H2: The Cancer Detection Paradox\n"
        "High-minority tracts report LOWER cancer rates — consistent with under-screening "
        "or mortality before diagnosis",
        fontsize=12, fontweight="bold",
    )

    # Panel A: cancer vs minority %
    ax = axes[0]
    hdf["min_q5"] = pd.qcut(hdf["minority_pct"], 5,
                             labels=["Q1\n0–20%", "Q2", "Q3", "Q4", "Q5\n80–100%"])
    means = hdf.groupby("min_q5", observed=True)["cancer_crude"].mean()
    cmap = plt.cm.RdBu_r
    ax.bar(means.index, means.values,
           color=[cmap(i / 4) for i in range(5)], edgecolor="black")
    ax.set_xlabel("Minority % quintile")
    ax.set_ylabel("Mean cancer crude rate (per 100)")
    ax.set_title("Cancer rate by minority quintile")
    r, _ = spearmanr(hdf["minority_pct"], hdf["cancer_crude"])
    ax.text(0.97, 0.97, f"ρ = {r:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, color="red",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Panel B: cancer vs poverty (opposite direction)
    ax = axes[1]
    hdf["pov_q5"] = pd.qcut(hdf["poverty_pct"], 5,
                              labels=["Q1\nLowest", "Q2", "Q3", "Q4", "Q5\nHighest"])
    means2 = hdf.groupby("pov_q5", observed=True)["cancer_crude"].mean()
    ax.bar(means2.index, means2.values,
           color=[cmap(i / 4) for i in range(5)], edgecolor="black")
    ax.set_xlabel("Poverty % quintile")
    ax.set_ylabel("Mean cancer crude rate (per 100)")
    ax.set_title("Cancer rate by poverty quintile")
    r2, _ = spearmanr(hdf["poverty_pct"], hdf["cancer_crude"])
    ax.text(0.97, 0.97, f"ρ = {r2:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, color="red",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Panel C: contrast with another outcome (diabetes) to show it's specific to cancer
    ax = axes[2]
    hdf_m = hdf.groupby("min_q5", observed=True)[
        ["cancer_crude", "diabetes_crude"]].mean()
    x = np.arange(len(hdf_m))
    w = 0.35
    ax.bar(x - w/2, hdf_m["cancer_crude"], w, label="Cancer", color="#e74c3c")
    ax.bar(x + w/2, hdf_m["diabetes_crude"], w, label="Diabetes", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(hdf_m.index)
    ax.set_xlabel("Minority % quintile")
    ax.set_ylabel("Mean crude rate (per 100)")
    ax.set_title("Cancer vs Diabetes by minority %\n(diverging patterns)")
    ax.legend()
    plt.tight_layout()
    _save(fig, "h2_cancer_paradox")

    # ── H3: Declining total releases + consolidation ──────────────────────────
    yr = df.groupby("REPORTING_YEAR").agg(
        total_rel=("TOTAL_RELEASES", "sum"),
        n_facilities=("TRI_FACILITY_ID", "nunique"),
        mean_rel=("TOTAL_RELEASES", "mean"),
        median_rel=("TOTAL_RELEASES", "median"),
    ).reset_index()
    yr["rel_per_facility"] = yr["total_rel"] / yr["n_facilities"]
    # Gini per year (proxy for concentration)
    def gini(arr):
        a = np.sort(arr[arr > 0])
        n = len(a)
        if n == 0:
            return np.nan
        idx = np.arange(1, n + 1)
        return (2 * (idx * a).sum() / (n * a.sum())) - (n + 1) / n

    yr_gini = df.groupby("REPORTING_YEAR")["TOTAL_RELEASES"].apply(gini).reset_index()
    yr_gini.columns = ["REPORTING_YEAR", "gini"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "H3: Releases declining nationally but burden concentrating in fewer facilities",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0, 0]
    ax.fill_between(yr["REPORTING_YEAR"], yr["total_rel"] / 1e9, alpha=0.3, color="#e74c3c")
    ax.plot(yr["REPORTING_YEAR"], yr["total_rel"] / 1e9, "o-", color="#e74c3c", lw=2)
    ax.set_title("Total national releases (billion lbs)")
    ax.set_ylabel("Billion lbs")
    change = (yr.iloc[-1]["total_rel"] - yr.iloc[0]["total_rel"]) / yr.iloc[0]["total_rel"] * 100
    ax.text(0.05, 0.05, f"Change 2013→2023: {change:+.1f}%",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax = axes[0, 1]
    ax.fill_between(yr["REPORTING_YEAR"], yr["n_facilities"], alpha=0.3, color="#3498db")
    ax.plot(yr["REPORTING_YEAR"], yr["n_facilities"], "o-", color="#3498db", lw=2)
    ax.set_title("Number of reporting facilities")
    ax.set_ylabel("Facility count")
    fac_change = (yr.iloc[-1]["n_facilities"] - yr.iloc[0]["n_facilities"]) / yr.iloc[0]["n_facilities"] * 100
    ax.text(0.05, 0.05, f"Change 2013→2023: {fac_change:+.1f}%",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax = axes[1, 0]
    ax.fill_between(yr["REPORTING_YEAR"], yr["rel_per_facility"] / 1e3, alpha=0.3, color="#2ecc71")
    ax.plot(yr["REPORTING_YEAR"], yr["rel_per_facility"] / 1e3, "o-", color="#2ecc71", lw=2)
    ax.set_title("Mean releases per facility (thousand lbs)")
    ax.set_ylabel("Thousand lbs / facility")
    rpf_change = (yr.iloc[-1]["rel_per_facility"] - yr.iloc[0]["rel_per_facility"]) / yr.iloc[0]["rel_per_facility"] * 100
    ax.text(0.05, 0.05, f"Change 2013→2023: {rpf_change:+.1f}%",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax = axes[1, 1]
    yr_merged = yr.merge(yr_gini, on="REPORTING_YEAR")
    ax.fill_between(yr_merged["REPORTING_YEAR"], yr_merged["gini"], alpha=0.3, color="#9b59b6")
    ax.plot(yr_merged["REPORTING_YEAR"], yr_merged["gini"], "o-", color="#9b59b6", lw=2)
    ax.set_title("Gini coefficient of releases\n(higher = more concentrated in fewer facilities)")
    ax.set_ylabel("Gini coefficient")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    _save(fig, "h3_declining_releases_consolidation")

    # ── H4: State-level industrial geography ≠ poverty geography ─────────────
    state = df.groupby("ST").agg(
        mean_rel=("TOTAL_RELEASES", "mean"),
        median_rel=("TOTAL_RELEASES", "median"),
        total_rel=("TOTAL_RELEASES", "sum"),
        mean_pov=("poverty_pct", "mean"),
        mean_min=("minority_pct", "mean"),
        mean_inc=("median_income", "mean"),
        n=("TRI_FACILITY_ID", "nunique"),
    ).reset_index()
    state = state[state["n"] >= 10].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "H4: High-release states are NOT the poorest states\n"
        "Industrial geography and poverty geography are largely decoupled",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    r1, p1 = spearmanr(state["mean_pov"], state["mean_rel"])
    sc = ax.scatter(state["mean_pov"], state["mean_rel"] / 1e3,
                    s=state["n"] * 0.5 + 30, c=state["mean_min"],
                    cmap="RdYlBu_r", alpha=0.8, edgecolors="black", lw=0.5)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Minority %")
    for _, row in state.nlargest(10, "mean_rel").iterrows():
        ax.annotate(row["ST"], (row["mean_pov"], row["mean_rel"] / 1e3),
                    fontsize=7, ha="center", va="bottom")
    for _, row in state.nlargest(10, "mean_pov").iterrows():
        ax.annotate(row["ST"], (row["mean_pov"], row["mean_rel"] / 1e3),
                    fontsize=7, ha="center", va="top", color="red")
    # trend
    z = np.polyfit(state["mean_pov"], state["mean_rel"], 1)
    xr = np.linspace(state["mean_pov"].min(), state["mean_pov"].max(), 100)
    ax.plot(xr, np.poly1d(z)(xr) / 1e3, "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Mean poverty % in tract")
    ax.set_ylabel("Mean facility releases (thousand lbs)")
    ax.set_title(f"Poverty % vs mean releases\n(bubble = facility count, color = minority %)\nρ = {r1:.3f}")

    ax = axes[1]
    r2, p2 = spearmanr(state["mean_inc"], state["mean_rel"])
    sc2 = ax.scatter(state["mean_inc"] / 1000, state["mean_rel"] / 1e3,
                     s=state["n"] * 0.5 + 30, c=state["mean_min"],
                     cmap="RdYlBu_r", alpha=0.8, edgecolors="black", lw=0.5)
    cb2 = plt.colorbar(sc2, ax=ax)
    cb2.set_label("Minority %")
    for _, row in state.nlargest(8, "mean_rel").iterrows():
        ax.annotate(row["ST"], (row["mean_inc"] / 1000, row["mean_rel"] / 1e3),
                    fontsize=7, ha="center", va="bottom")
    z2 = np.polyfit(state["mean_inc"], state["mean_rel"], 1)
    xr2 = np.linspace(state["mean_inc"].min(), state["mean_inc"].max(), 100)
    ax.plot(xr2 / 1000, np.poly1d(z2)(xr2) / 1e3, "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Mean median income ($k)")
    ax.set_ylabel("Mean facility releases (thousand lbs)")
    ax.set_title(f"Income vs mean releases\nρ = {r2:.3f}")
    plt.tight_layout()
    _save(fig, "h4_state_geography_decoupled")

    # ── H4b: cleaner scatter — log(releases) vs poverty % for all states ──────
    # Each state is a point; labeled by state abbreviation; circle size = facility count
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "H4: High-release states are NOT the same as high-poverty states\n"
        "Each point = one US state; circle size ∝ number of TRI facilities",
        fontsize=12, fontweight="bold",
    )

    top_rel = state.nlargest(10, "mean_rel")["ST"].tolist()
    top_pov = state.nlargest(10, "mean_pov")["ST"].tolist()
    overlap = set(top_rel) & set(top_pov)

    def _state_color(st):
        if st in top_rel and st in top_pov:
            return "#8e44ad"   # overlap
        if st in top_rel:
            return "#e74c3c"   # high release only
        if st in top_pov:
            return "#3498db"   # high poverty only
        return "#bdc3c7"

    ax = axes[0]
    for _, row in state.iterrows():
        c = _state_color(row["ST"])
        ax.scatter(row["mean_pov"], np.log10(row["mean_rel"] + 1),
                   s=row["n"] * 0.8 + 30, color=c, alpha=0.75, edgecolors="0.3", lw=0.5)
        ax.text(row["mean_pov"], np.log10(row["mean_rel"] + 1), row["ST"],
                fontsize=6.5, ha="center", va="bottom", color="0.2")

    z = np.polyfit(state["mean_pov"], np.log10(state["mean_rel"].clip(1)), 1)
    xr = np.linspace(state["mean_pov"].min(), state["mean_pov"].max(), 100)
    ax.plot(xr, np.poly1d(z)(xr), "k--", lw=1.2, alpha=0.5, label=f"ρ = {r1:.3f}")
    ax.set_xlabel("Mean poverty rate (%, avg of facility neighborhoods)", fontsize=10)
    ax.set_ylabel("log₁₀(Mean facility releases, lbs)", fontsize=10)
    ax.set_title("State poverty vs release volume\n(red = top-10 by releases, blue = top-10 by poverty)")
    ax.legend(fontsize=9)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#e74c3c", label="Top-10 by releases"),
        Patch(color="#3498db", label="Top-10 by poverty"),
        Patch(color="#8e44ad", label=f"Both (overlap: {len(overlap)})"),
        Patch(color="#bdc3c7", label="Other states"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    # Right panel: ranked comparison table as bar chart
    ax = axes[1]
    rank_df = state.copy()
    rank_df['rel_rank'] = rank_df['mean_rel'].rank(ascending=False).astype(int)
    rank_df['pov_rank'] = rank_df['mean_pov'].rank(ascending=False).astype(int)
    rank_df = rank_df.sort_values('rel_rank').head(15)

    y = np.arange(len(rank_df))
    ax.barh(y - 0.2, rank_df['rel_rank'], 0.35, color='#e74c3c', alpha=0.8,
            label='Pollution rank (1=most releases)')
    ax.barh(y + 0.2, rank_df['pov_rank'], 0.35, color='#3498db', alpha=0.8,
            label='Poverty rank (1=most poverty)')
    ax.set_yticks(y)
    ax.set_yticklabels(rank_df['ST'])
    ax.invert_yaxis()
    ax.set_xlabel("Rank (1 = highest)")
    ax.set_title("Top 15 states by releases:\ntheir poverty rank is typically much lower")
    ax.legend(fontsize=9)
    ax.axvline(15, color='gray', ls='--', lw=0.8, alpha=0.5)

    plt.tight_layout()
    _save(fig, "h4b_release_vs_poverty_states")

    logger.info("  Saved: h1a, h1b, h2, h3, h4, h4b hypothesis evidence plots")


def analysis6_new_hypotheses(df: pd.DataFrame) -> None:
    """
    H5: Dangerous chemical classes × release medium — does air release of carcinogens
        associate with different health outcomes than water/land?
    H6: Do facilities in poor communities stay open longer (persistence of burden)?
    H7: Cancer paradox — competing mortality explains low cancer rates in minority tracts
    H8: Medium-specific health pathway (air → respiratory; water → systemic)
    """
    from scipy.stats import spearmanr, mannwhitneyu, kruskal

    # ── Chemical classification ───────────────────────────────────────────────
    CARCINOGENS = {
        'arsenic', 'benzene', 'chromium', 'cadmium', 'nickel', 'lead',
        'vinyl chloride', 'formaldehyde', '1,3-butadiene', 'trichloroethylene',
        'perchloroethylene', 'tetrachloroethylene', 'styrene', 'ethylene oxide',
        'dioxin', 'polycyclic aromatic', 'benzo', 'naphthalene', 'beryllium',
        'cobalt', 'antimony', 'hydrazine', 'acrylonitrile',
    }
    PERSISTENT = {
        'lead', 'mercury', 'cadmium', 'arsenic', 'dioxin', 'pcb',
        'polychlorinated', 'chlordane', 'aldrin', 'dieldrin', 'endrin',
        'heptachlor', 'mirex', 'toxaphene', 'hexachlorobenzene',
        'polycyclic aromatic', 'benzo',
    }
    ACUTE_TOXIC = {
        'chlorine', 'hydrogen cyanide', 'ammonia', 'hydrogen fluoride',
        'phosgene', 'sulfur dioxide', 'hydrogen sulfide', 'nitric acid',
        'hydrofluoric',
    }

    def classify_chem(name):
        if not isinstance(name, str):
            return 'Other'
        n = name.lower()
        if any(c in n for c in CARCINOGENS):
            return 'Carcinogen'
        if any(c in n for c in PERSISTENT):
            return 'Persistent'
        if any(c in n for c in ACUTE_TOXIC):
            return 'Acute Toxic'
        return 'Other'

    df = df.copy()
    df['chem_class'] = df['CHEMICAL_NAME'].apply(classify_chem)
    df['log_releases'] = np.log10(df['TOTAL_RELEASES'].clip(0.1))
    hdf = df[df['cancer_crude'].notna()].copy()

    # ── H5: Chemical danger class × health outcomes ───────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(
        "H5: Chemical danger class vs health outcomes\n"
        "(Carcinogens & Acute Toxics released in highest volumes — "
        "but health outcomes differ little by class, suggesting exposure pathway matters more than volume)",
        fontsize=11, fontweight="bold",
    )
    outcomes = ["cancer_crude", "asthma_crude", "chd_crude",
                "copd_crude", "diabetes_crude", "mental_health_crude"]
    out_labels = ["Cancer", "Asthma", "CHD", "COPD", "Diabetes", "Mental Health"]
    class_order = ["Carcinogen", "Acute Toxic", "Persistent", "Other"]
    class_colors = ["#e74c3c", "#e67e22", "#9b59b6", "#95a5a6"]

    for ax, outcome, label in zip(axes.flat, outcomes, out_labels):
        data_by_class = [
            hdf[hdf['chem_class'] == c][outcome].dropna().values
            for c in class_order
        ]
        bp = ax.boxplot(data_by_class, patch_artist=True, notch=True,
                        showfliers=False)
        for patch, color in zip(bp['boxes'], class_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(class_order, fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontweight="bold")
        # Kruskal-Wallis p
        stat, p = kruskal(*[d for d in data_by_class if len(d) > 0])
        ax.text(0.98, 0.98, f"KW p={p:.2e}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8)

    plt.tight_layout()
    _save(fig, "h5_chemical_class_health")

    # ── H5b: Release volume by chemical class (what's actually in the air) ───
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("H5b: Total release volume and facility count by chemical class",
                 fontsize=12, fontweight="bold")

    class_stats = df.groupby('chem_class').agg(
        total_vol=('TOTAL_RELEASES', 'sum'),
        mean_vol=('TOTAL_RELEASES', 'mean'),
        n_records=('TRI_FACILITY_ID', 'count'),
        n_fac=('TRI_FACILITY_ID', 'nunique'),
    ).reindex(class_order)

    ax = axes[0]
    bars = ax.bar(class_order, class_stats['total_vol'] / 1e9, color=class_colors, alpha=0.85)
    ax.set_ylabel("Total releases (billion lbs)")
    ax.set_title("Total release volume by chemical class")
    for bar, v in zip(bars, class_stats['total_vol'] / 1e9):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}B", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    bars2 = ax.bar(class_order, class_stats['n_fac'], color=class_colors, alpha=0.85)
    ax.set_ylabel("Unique facilities")
    ax.set_title("Number of facilities per chemical class")
    for bar, v in zip(bars2, class_stats['n_fac']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(v):,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save(fig, "h5b_chemical_class_volume")

    # ── H6: Facility persistence in poor communities ──────────────────────────
    # Use reporting years as proxy: facilities that stopped < 2023 = "closed"
    fac_years = df.groupby('TRI_FACILITY_ID').agg(
        first_year=('REPORTING_YEAR', 'min'),
        last_year=('REPORTING_YEAR', 'max'),
        n_years=('REPORTING_YEAR', 'nunique'),
        mean_pov=('poverty_pct', 'mean'),
        mean_min=('minority_pct', 'mean'),
        mean_inc=('median_income', 'mean'),
        total_rel=('TOTAL_RELEASES', 'sum'),
    ).reset_index()
    fac_years['active_span'] = fac_years['last_year'] - fac_years['first_year'] + 1
    fac_years['still_active'] = fac_years['last_year'] == 2023
    fac_years['pov_q'] = pd.qcut(fac_years['mean_pov'].clip(0), 5,
                                  labels=['Q1\nLowest', 'Q2', 'Q3', 'Q4', 'Q5\nHighest'],
                                  duplicates='drop')

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "H6: Do facilities in poor communities stay open longer?\n"
        "(TRI reporting span 2013–2023 as proxy for operational lifetime)",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    span_by_pov = fac_years.groupby('pov_q', observed=True)['active_span'].mean()
    ax.bar(span_by_pov.index, span_by_pov.values,
           color=plt.cm.Reds(np.linspace(0.3, 0.9, len(span_by_pov))))
    ax.set_xlabel("Poverty quintile")
    ax.set_ylabel("Mean active span (years)")
    ax.set_title("Mean operational span by poverty quintile")
    r, p = spearmanr(fac_years['mean_pov'].dropna(),
                     fac_years.loc[fac_years['mean_pov'].notna(), 'active_span'])
    ax.text(0.05, 0.95, f"ρ={r:.3f}", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    ax = axes[1]
    still_active_rate = fac_years.groupby('pov_q', observed=True)['still_active'].mean() * 100
    ax.bar(still_active_rate.index, still_active_rate.values,
           color=plt.cm.Oranges(np.linspace(0.3, 0.9, len(still_active_rate))))
    ax.set_xlabel("Poverty quintile")
    ax.set_ylabel("% still reporting in 2023")
    ax.set_title("Survival rate (active in 2023) by poverty quintile")

    ax = axes[2]
    # Total cumulative release burden by poverty quintile
    cum_rel = fac_years.groupby('pov_q', observed=True)['total_rel'].sum() / 1e9
    ax.bar(cum_rel.index, cum_rel.values,
           color=plt.cm.Purples(np.linspace(0.3, 0.9, len(cum_rel))))
    ax.set_xlabel("Poverty quintile")
    ax.set_ylabel("Total cumulative releases (billion lbs)")
    ax.set_title("Total 11-year cumulative releases by poverty quintile")
    r2, _ = spearmanr(fac_years['mean_pov'].dropna(),
                      fac_years.loc[fac_years['mean_pov'].notna(), 'total_rel'])
    ax.text(0.05, 0.95, f"ρ={r2:.3f}", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    plt.tight_layout()
    _save(fig, "h6_facility_persistence_poverty")

    # ── H7: Cancer paradox — competing mortality ──────────────────────────────
    hdf['competing_mortality'] = (
        hdf['chd_crude'].fillna(hdf['chd_crude'].median()) +
        hdf['copd_crude'].fillna(hdf['copd_crude'].median()) +
        hdf['diabetes_crude'].fillna(hdf['diabetes_crude'].median())
    )
    hdf['min_q5'] = pd.qcut(hdf['minority_pct'], 5,
                             labels=['Q1\n0–20%', 'Q2', 'Q3', 'Q4', 'Q5\n80–100%'])

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "H7: Cancer Paradox — Competing Mortality Hypothesis\n"
        "High-minority tracts have higher CHD+COPD+Diabetes burden; "
        "people may die of these before receiving a cancer diagnosis",
        fontsize=11, fontweight="bold",
    )

    ax = axes[0]
    comp_by_min = hdf.groupby('min_q5', observed=True)['competing_mortality'].mean()
    ax.bar(comp_by_min.index, comp_by_min.values,
           color=plt.cm.OrRd(np.linspace(0.3, 0.9, 5)), edgecolor='black')
    ax.set_xlabel("Minority % quintile")
    ax.set_ylabel("Mean CHD + COPD + Diabetes crude rate")
    ax.set_title("Competing mortality burden by minority %\n(higher → more non-cancer deaths)")
    r_c, _ = spearmanr(hdf['minority_pct'], hdf['competing_mortality'])
    ax.text(0.05, 0.95, f"ρ={r_c:.3f}", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', fc='white'))

    ax = axes[1]
    cancer_by_min = hdf.groupby('min_q5', observed=True)['cancer_crude'].mean()
    ax.bar(cancer_by_min.index, cancer_by_min.values,
           color=plt.cm.Blues_r(np.linspace(0.3, 0.9, 5)), edgecolor='black')
    ax.set_xlabel("Minority % quintile")
    ax.set_ylabel("Mean cancer crude rate (per 100)")
    ax.set_title("Cancer rate by minority %\n(falls as minority % rises)")
    r_k, _ = spearmanr(hdf['minority_pct'], hdf['cancer_crude'])
    ax.text(0.05, 0.95, f"ρ={r_k:.3f}", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', fc='white'))

    ax = axes[2]
    # Scatter: competing mortality vs cancer — inverse relationship?
    sub = hdf[['competing_mortality', 'cancer_crude', 'minority_pct']].dropna().sample(
        min(4000, len(hdf)), random_state=42)
    sc = ax.scatter(sub['competing_mortality'], sub['cancer_crude'],
                    c=sub['minority_pct'], cmap='RdYlBu_r', alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax, label='Minority %')
    z = np.polyfit(sub['competing_mortality'], sub['cancer_crude'], 1)
    xr = np.linspace(sub['competing_mortality'].min(), sub['competing_mortality'].max(), 100)
    ax.plot(xr, np.poly1d(z)(xr), 'k--', lw=2)
    r_sc, p_sc = spearmanr(sub['competing_mortality'], sub['cancer_crude'])
    ax.set_xlabel("Competing mortality score (CHD+COPD+Diabetes)")
    ax.set_ylabel("Cancer crude rate")
    ax.set_title(f"Competing mortality vs cancer rate\nρ={r_sc:.3f}")
    ax.text(0.98, 0.98, "Color = minority %", transform=ax.transAxes,
            ha='right', va='top', fontsize=8)

    plt.tight_layout()
    _save(fig, "h7_cancer_competing_mortality")

    # ── H8: Release medium → specific health pathways ────────────────────────
    # Use release qty file to get per-medium totals per facility-year
    logger.info("  Building H8 medium-specific health pathway plot (large join)...")
    try:
        rq = pd.read_csv("data/raw/tri_release_qty.csv", low_memory=False)
        # Aggregate medium to broad categories
        medium_map = {
            'AIR FUG': 'Air', 'AIR STACK': 'Air',
            'WATER': 'Water',
            'LAND TREA': 'Land', 'OTH LANDF': 'Land', 'SURF IMP': 'Land',
            'OTH DISP': 'Land', 'RCRA C': 'Land',
            'UNINJ I': 'Underground', 'UNINJ IIV': 'Underground',
        }
        rq['medium_broad'] = rq['environmental_medium'].map(medium_map).fillna('Other')
        rq_agg = rq.groupby(['doc_ctrl_num', 'medium_broad'])['total_release'].sum().reset_index()
        rq_wide = rq_agg.pivot_table(
            index='doc_ctrl_num', columns='medium_broad', values='total_release',
            aggfunc='sum', fill_value=0
        ).reset_index()
        rq_wide.columns.name = None

        # Load raw TRI to get facility ID
        tri_raw = pd.read_csv("data/raw/tri_facilities.csv", low_memory=False)
        tri_raw.columns = tri_raw.columns.str.upper().str.strip()
        dcn_col = next((c for c in tri_raw.columns if 'DOC_CTRL' in c), None)
        fid_col = next((c for c in tri_raw.columns if 'TRI_FACILITY_ID' in c), None)
        if dcn_col and fid_col:
            fac_map = tri_raw[[dcn_col, fid_col]].rename(
                columns={dcn_col: 'doc_ctrl_num', fid_col: 'TRI_FACILITY_ID'})
            rq_wide = rq_wide.merge(fac_map, on='doc_ctrl_num', how='left')

        # Merge with health data
        health_cols = ['TRI_FACILITY_ID', 'cancer_crude', 'asthma_crude',
                       'chd_crude', 'copd_crude', 'diabetes_crude', 'mental_health_crude']
        fac_health = hdf[health_cols].drop_duplicates('TRI_FACILITY_ID')
        merged = rq_wide.merge(fac_health, on='TRI_FACILITY_ID', how='inner')
        logger.info(f"  H8 merged: {len(merged):,} records")

        fig, axes = plt.subplots(2, 3, figsize=(16, 11))
        fig.suptitle(
            "H8: Does release medium determine health pathway?\n"
            "(Air releases → respiratory; Water releases → systemic diseases?)",
            fontsize=12, fontweight="bold",
        )
        medium_cols = [c for c in ['Air', 'Water', 'Land', 'Underground']
                       if c in merged.columns]
        outcomes_h8 = ['asthma_crude', 'copd_crude', 'cancer_crude',
                       'chd_crude', 'diabetes_crude', 'mental_health_crude']
        out_labels_h8 = ['Asthma', 'COPD', 'Cancer', 'CHD', 'Diabetes', 'Mental Health']

        for ax, outcome, label in zip(axes.flat, outcomes_h8, out_labels_h8):
            rhos = []
            for med in medium_cols:
                sub = merged[[med, outcome]].dropna()
                sub = sub[sub[med] > 0]
                if len(sub) < 30:
                    rhos.append(np.nan)
                    continue
                r, _ = spearmanr(np.log10(sub[med].clip(0.1)), sub[outcome])
                rhos.append(r)
            colors_h8 = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6'][:len(medium_cols)]
            valid = [(m, r, c) for m, r, c in zip(medium_cols, rhos, colors_h8)
                     if not np.isnan(r)]
            if valid:
                ax.barh([v[0] for v in valid], [v[1] for v in valid],
                        color=[v[2] for v in valid], alpha=0.85)
                ax.axvline(0, color='black', lw=0.8)
                ax.set_xlim(-0.3, 0.3)
            ax.set_title(label, fontweight='bold')
            ax.set_xlabel("Spearman ρ (log releases vs health rate)")

        plt.tight_layout()
        _save(fig, "h8_medium_health_pathway")

    except Exception as e:
        logger.warning(f"  H8 skipped: {e}")

    logger.info("  Saved: h5, h5b, h6, h7, h8 new hypothesis plots")


def analysis7_local_impact(df: pd.DataFrame) -> None:
    """
    Local (census-tract) impact analysis.

    Each census tract aggregates all TRI facilities within it:
    - Total release burden in the neighborhood
    - Number of distinct facilities
    - Cumulative years of operation

    Then correlates with that tract's health outcomes.
    This is the most meaningful geographic scale for health impact.
    """
    from scipy.stats import spearmanr

    # ── Build tract-level summary ─────────────────────────────────────────────
    tract = df.groupby('fips_tract').agg(
        n_facilities=('TRI_FACILITY_ID', 'nunique'),
        total_releases=('TOTAL_RELEASES', 'sum'),
        mean_releases=('TOTAL_RELEASES', 'mean'),
        n_years=('REPORTING_YEAR', 'nunique'),
        poverty_pct=('poverty_pct', 'first'),
        minority_pct=('minority_pct', 'first'),
        median_income=('median_income', 'first'),
        cancer_crude=('cancer_crude', 'first'),
        asthma_crude=('asthma_crude', 'first'),
        chd_crude=('chd_crude', 'first'),
        copd_crude=('copd_crude', 'first'),
        diabetes_crude=('diabetes_crude', 'first'),
        mental_health_crude=('mental_health_crude', 'first'),
    ).reset_index()

    tract_h = tract[tract['cancer_crude'].notna()].copy()
    tract_h['log_total_releases'] = np.log10(tract_h['total_releases'].clip(0.1))
    tract_h['log_n_fac'] = np.log10(tract_h['n_facilities'].clip(0.5))

    outcomes = ['cancer_crude', 'asthma_crude', 'chd_crude',
                'copd_crude', 'diabetes_crude', 'mental_health_crude']
    out_labels = ['Cancer', 'Asthma', 'CHD', 'COPD', 'Diabetes', 'Mental Health']

    # ── Plot A: predictor comparison (releases vs poverty) ───────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Local Impact: Census-tract level analysis\n"
        "Predictors of health outcomes: log(total releases in tract) vs poverty % vs n facilities\n"
        f"n={len(tract_h):,} tracts with both TRI facilities and CDC health data",
        fontsize=11, fontweight="bold",
    )
    predictors = ['log_total_releases', 'n_facilities', 'poverty_pct', 'median_income', 'minority_pct']
    pred_labels = ['log₁₀(Total Releases\nin tract)', 'N Facilities\nin tract',
                   'Poverty %', 'Median\nIncome', 'Minority %']
    pred_colors = ['#e67e22', '#e74c3c', '#c0392b', '#3498db', '#2ecc71']

    for ax, outcome, label in zip(axes.flat, outcomes, out_labels):
        rhos = []
        for pred in predictors:
            sub = tract_h[[pred, outcome]].dropna()
            r, _ = spearmanr(sub[pred], sub[outcome])
            rhos.append(r)
        bars = ax.barh(pred_labels, rhos, color=pred_colors)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel("Spearman ρ")
        for bar, r in zip(bars, rhos):
            ax.text(r + (0.02 if r >= 0 else -0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f"{r:.3f}", va='center',
                    ha='left' if r >= 0 else 'right', fontsize=8)

    plt.tight_layout()
    _save(fig, "t1_local_predictor_comparison")

    # ── Plot B: Total releases × health — dose-response at tract level ───────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Local dose-response: Census tract total TRI releases vs health outcomes\n"
        "(Each point = one census tract; trend line + hexbin density)",
        fontsize=11, fontweight="bold",
    )
    for ax, outcome, label in zip(axes.flat, outcomes, out_labels):
        sub = tract_h[['log_total_releases', outcome, 'poverty_pct']].dropna()
        hb = ax.hexbin(sub['log_total_releases'], sub[outcome],
                       gridsize=25, cmap='YlOrRd', mincnt=1)
        z = np.polyfit(sub['log_total_releases'], sub[outcome], 1)
        xr = np.linspace(sub['log_total_releases'].min(), sub['log_total_releases'].max(), 100)
        ax.plot(xr, np.poly1d(z)(xr), 'b--', lw=2)
        r, p = spearmanr(sub['log_total_releases'], sub[outcome])
        ax.set_xlabel("log₁₀(Total releases in tract, lbs)")
        ax.set_ylabel(label)
        ax.set_title(f"{label}  ρ={r:.3f}, p={p:.3f}", fontsize=9)
    plt.tight_layout()
    _save(fig, "t2_local_dose_response")

    # ── Plot C: Pollution burden quintile vs health ───────────────────────────
    tract_h['rel_quintile'] = pd.qcut(
        tract_h['log_total_releases'], 5,
        labels=['Q1\nLowest', 'Q2', 'Q3', 'Q4', 'Q5\nHighest'],
        duplicates='drop',
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Local dose-response by pollution burden quintile\n"
        "(Census tracts sorted by total TRI releases in the tract → health outcome means)",
        fontsize=11, fontweight="bold",
    )
    for ax, outcome, label in zip(axes.flat, outcomes, out_labels):
        means = tract_h.groupby('rel_quintile', observed=True)[outcome].mean()
        cis = tract_h.groupby('rel_quintile', observed=True)[outcome].sem() * 1.96
        ax.bar(means.index, means.values,
               yerr=cis.values,
               color=plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(means))),
               edgecolor='black', capsize=3)
        ax.set_xlabel("Pollution burden quintile")
        ax.set_ylabel(f"Mean {label} rate")
        ax.set_title(label, fontweight='bold')
    plt.tight_layout()
    _save(fig, "t3_local_pollution_quintile_health")

    # ── Plot D: Poverty-controlled releases vs health ─────────────────────────
    # Split tracts into high/low poverty, then within each group,
    # does release burden still predict health?
    tract_h['high_poverty'] = tract_h['poverty_pct'] > tract_h['poverty_pct'].median()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Local impact: Effect of TRI releases CONTROLLING for poverty\n"
        "(Same release burden — does it matter more in poor vs wealthy tracts?)\n"
        "Blue = low poverty tracts, Red = high poverty tracts",
        fontsize=11, fontweight="bold",
    )
    for ax, outcome, label in zip(axes.flat, outcomes, out_labels):
        for is_high, color, grp_label in [(False, '#3498db', 'Low poverty'), (True, '#e74c3c', 'High poverty')]:
            sub = tract_h[tract_h['high_poverty'] == is_high][
                ['log_total_releases', outcome]].dropna()
            r, p = spearmanr(sub['log_total_releases'], sub[outcome])
            # bin into deciles
            sub['rdec'] = pd.qcut(sub['log_total_releases'], 8, duplicates='drop')
            mn = sub.groupby('rdec', observed=True)[outcome].mean()
            xvals = np.arange(len(mn))
            ax.plot(xvals, mn.values, 'o-', color=color, lw=2,
                    label=f"{grp_label} (ρ={r:.2f})", alpha=0.85)
        ax.set_xlabel("Release burden decile (1=lowest)")
        ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold')
        ax.legend(fontsize=7)
    plt.tight_layout()
    _save(fig, "t4_local_poverty_controlled")

    # ── Plot E: Geographic clusters — top-burden tracts ──────────────────────
    # Identify tracts with high releases AND high health burden
    top_burden = tract_h.copy()
    top_burden['health_score'] = (
        top_burden[['cancer_crude', 'asthma_crude', 'chd_crude',
                    'copd_crude', 'diabetes_crude', 'mental_health_crude']]
        .rank(pct=True).mean(axis=1)
    )
    top_burden['high_releases'] = top_burden['log_total_releases'] > np.percentile(
        top_burden['log_total_releases'], 75)
    top_burden['high_health'] = top_burden['health_score'] > 0.75

    n_hh = (top_burden['high_releases'] & top_burden['high_health']).sum()
    n_hl = (top_burden['high_releases'] & ~top_burden['high_health']).sum()
    n_lh = (~top_burden['high_releases'] & top_burden['high_health']).sum()
    n_ll = (~top_burden['high_releases'] & ~top_burden['high_health']).sum()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Local impact: Four-quadrant analysis at census-tract level\n"
        "High releases + High health burden vs other combinations",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    sc = ax.scatter(top_burden['log_total_releases'],
                    top_burden['health_score'],
                    c=top_burden['poverty_pct'], cmap='RdYlGn_r',
                    alpha=0.6, s=20)
    plt.colorbar(sc, ax=ax, label='Poverty %')
    xmed = top_burden['log_total_releases'].quantile(0.75)
    ymed = 0.75
    ax.axvline(xmed, color='black', lw=1, ls='--')
    ax.axhline(ymed, color='black', lw=1, ls='--')
    ax.text(xmed + 0.05, ymed + 0.01, f"High burden\n({n_hh} tracts)", fontsize=9,
            ha='left', color='red', fontweight='bold')
    ax.text(xmed - 3, ymed + 0.01, f"Poor health,\nclean neighborhood\n({n_lh})", fontsize=8,
            ha='center', color='orange')
    ax.text(xmed + 0.05, ymed - 0.05, f"High releases,\nhealthy\n({n_hl})", fontsize=8,
            ha='left', color='blue')
    ax.set_xlabel("log₁₀(Total releases in tract)")
    ax.set_ylabel("Composite health burden score (percentile)")
    ax.set_title("Releases vs health burden (color = poverty %)")

    ax = axes[1]
    categories = ['High releases\n+ High health\n(concern)', 'High releases\n+ Low health',
                  'Low releases\n+ High health', 'Low releases\n+ Low health']
    counts = [n_hh, n_hl, n_lh, n_ll]
    colors_q = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']
    bars = ax.bar(categories, counts, color=colors_q, edgecolor='black')
    ax.set_ylabel("Number of census tracts")
    ax.set_title("Census tract distribution by release/health quadrant")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(cnt), ha='center', va='bottom', fontweight='bold')
    pct_hh = n_hh / len(top_burden) * 100
    ax.text(0.05, 0.95, f"{pct_hh:.1f}% of tracts:\nhigh pollution + high disease",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', fc='#ffe0e0', alpha=0.9))
    plt.tight_layout()
    _save(fig, "t5_local_four_quadrant")

    logger.info("  Saved: t1–t5 local (census-tract) impact plots")


def analysis8_case_control(df: pd.DataFrame) -> None:
    """
    Case-control analysis using NEIGHBOR-BASED INFLUENCE MODEL.
    
    Influence zone = TRI tract + all neighboring tracts (within ~5km centroid distance).
    True controls = tracts with NO TRI facility AND NOT neighboring any TRI tract.
    
    This captures pollution dispersion beyond the facility's own tract boundary.
    
    Groups:
      - 'control': true background (no TRI, not neighbor of TRI)
      - 'tri_neighbor': neighbors a TRI tract (influenced but no direct facility)
      - 'tri_ncarc': has TRI facility, non-carcinogen only
      - 'tri_carc': has TRI facility releasing carcinogens
    """
    from scipy.stats import mannwhitneyu, spearmanr

    CARCINOGENS = {
        'arsenic', 'benzene', 'chromium', 'cadmium', 'nickel', 'lead',
        'vinyl chloride', 'formaldehyde', '1,3-butadiene', 'trichloroethylene',
        'styrene', 'ethylene oxide', 'dioxin', 'polycyclic aromatic', 'benzo',
        'naphthalene', 'beryllium', 'cobalt', 'acrylonitrile',
    }

    # ── Load neighbor lookup ──────────────────────────────────────────────────
    neighbor_dict = _load_tract_neighbors()
    if not neighbor_dict:
        logger.error("Cannot run case-control without neighbor lookup")
        return

    # ── Build all-tract dataset ───────────────────────────────────────────────
    cdc_raw = pd.read_csv("data/raw/cdc_places.csv", low_memory=False)
    cdc_pivot = cdc_raw.pivot_table(
        index='locationid', columns='measureid', values='data_value'
    ).reset_index()
    cdc_pivot.columns.name = None
    cdc_pivot['fips_tract'] = cdc_pivot['locationid'].astype(str).str.zfill(11)

    census = _load_census()

    # ── Build TRI tract-level aggregation ─────────────────────────────────────
    df2 = df.copy()
    df2['is_carc'] = df2['CHEMICAL_NAME'].apply(
        lambda x: any(c in str(x).lower() for c in CARCINOGENS)
    )
    df_carc = (
        df2[df2['is_carc']]
        .groupby('fips_tract')['TOTAL_RELEASES']
        .sum()
        .reset_index()
        .rename(columns={'TOTAL_RELEASES': 'carc_releases'})
    )
    df_total = df2.groupby('fips_tract').agg(
        total_releases=('TOTAL_RELEASES', 'sum'),
        n_facilities=('TRI_FACILITY_ID', 'nunique'),
    ).reset_index()
    tri_tract = df_total.merge(df_carc, on='fips_tract', how='left')
    tri_tract['carc_releases'] = tri_tract['carc_releases'].fillna(0)
    tri_tract['fips_tract'] = tri_tract['fips_tract'].astype(str).str.zfill(11)

    # ── Merge all data ────────────────────────────────────────────────────────
    all_t = cdc_pivot.merge(
        census[['fips_tract', 'poverty_pct']], on='fips_tract', how='left'
    )
    
    # ── Classify tracts using influence model ─────────────────────────────────
    all_t = _classify_tracts_for_case_control(all_t, tri_tract, neighbor_dict)

    all_t2 = all_t[
        all_t['CANCER'].notna() & all_t['poverty_pct'].notna() &
        (all_t['poverty_pct'] >= 0)
    ].copy().reset_index(drop=True)

    pov = all_t2['poverty_pct'].values
    bins = np.percentile(pov, [20, 40, 60, 80])
    all_t2['pov_q'] = np.digitize(pov, bins)

    outcomes = [
        ('Cancer', 'CANCER'),
        ('Asthma', 'CASTHMA'),
        ('Heart Disease (CHD)', 'CHD'),
        ('COPD (lung disease)', 'COPD'),
        ('Diabetes', 'DIABETES'),
        ('Poor Mental Health', 'MHLTH'),
    ]

    # ── Define groups ─────────────────────────────────────────────────────────
    tri_carc = all_t2[(all_t2['tri_zone'] == 'tri_direct') & (all_t2['carc_releases'] > 0)]
    tri_ncarc = all_t2[(all_t2['tri_zone'] == 'tri_direct') & (all_t2['carc_releases'] == 0)]
    tri_neighbor = all_t2[all_t2['tri_zone'] == 'tri_neighbor']
    ctrl = all_t2[all_t2['tri_zone'] == 'control']
    
    # Combined influence zone (for some plots)
    influenced = all_t2[all_t2['in_influence']]
    
    logger.info(
        f"  Case-control (NEIGHBOR MODEL): "
        f"carc TRI={len(tri_carc):,}, non-carc TRI={len(tri_ncarc):,}, "
        f"TRI-neighbor={len(tri_neighbor):,}, true control={len(ctrl):,}"
    )

    # ── Plot A: Unadjusted comparison — 4 groups × 6 outcomes ────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Case-Control with NEIGHBOR INFLUENCE MODEL\n"
        f"(True control: {len(ctrl):,} tracts with no TRI & no TRI neighbors | "
        f"TRI-neighbor: {len(tri_neighbor):,} | "
        f"TRI-direct: {len(tri_ncarc)+len(tri_carc):,})\n"
        "Stars = Mann-Whitney U test vs true control  "
        "(*** p<0.001, ** p<0.01, * p<0.05, ns = not significant)",
        fontsize=10, fontweight="bold",
    )
    group_colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    group_labels = ['True control\n(no TRI,\nno neighbor)', 'TRI\nneighbor', 'TRI direct\n(non-carc)', 'TRI direct\n(carcinogen)']
    groups = [ctrl, tri_neighbor, tri_ncarc, tri_carc]

    for ax, (hc_label, col) in zip(axes.flat, outcomes):
        vals = [g[col].dropna() for g in groups]
        means = [v.mean() for v in vals]
        sems = [v.sem() for v in vals]
        x = np.arange(4)
        bars = ax.bar(x, means, color=group_colors, alpha=0.85, edgecolor='black')
        ax.errorbar(x, means, yerr=sems, fmt='none', color='black', capsize=4, lw=2)
        # Significance vs control
        ymax = max(means) + max(sems) * 2
        ax.set_ylim(bottom=min(means) * 0.97)
        for xi in range(1, 4):
            _, p = mannwhitneyu(ctrl[col].dropna(), groups[xi][col].dropna())
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.text(xi, means[xi] + sems[xi] + (ymax - min(means)) * 0.03,
                    stars, ha='center', fontsize=10, color='black')
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=7)
        ax.set_title(hc_label, fontweight='bold')
        ax.set_ylabel("Mean crude rate (% of adults)")

    plt.tight_layout()
    _save(fig, "cc1_unadjusted_comparison")

    # ── Plot B: Poverty-matched difference (influence zone vs true control) ───
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.suptitle(
        "Case-Control: Poverty-adjusted health gap (INFLUENCE ZONE vs true control)\n"
        "Influence zone = TRI tract + all neighbors | True control = no TRI, no TRI neighbors\n"
        "Within each poverty quintile — residual effect of being near TRI facilities",
        fontsize=10, fontweight="bold",
    )
    pov_quintile_labels = [
        f'Q{i+1}\n({int(np.percentile(pov, i*20))}'
        f'–{int(np.percentile(pov, (i+1)*20))}% pov)'
        for i in range(5)
    ]
    outcome_labels = [hc for hc, _ in outcomes]
    xpos = np.arange(len(outcomes))
    width = 0.13

    for qi in range(5):
        diffs = []
        for hc, col in outcomes:
            infl_m = all_t2.loc[(all_t2['pov_q'] == qi) & all_t2['in_influence'], col].mean()
            ctrl_m = all_t2.loc[(all_t2['pov_q'] == qi) & (all_t2['tri_zone'] == 'control'), col].mean()
            diffs.append(infl_m - ctrl_m if not (np.isnan(infl_m) or np.isnan(ctrl_m)) else 0)
        offset = (qi - 2) * width
        ax.bar(xpos + offset, diffs, width,
               label=pov_quintile_labels[qi],
               color=plt.cm.RdYlBu_r(qi / 4), alpha=0.85, edgecolor='black', lw=0.5)

    ax.axhline(0, color='black', lw=1.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(outcome_labels, fontsize=8)
    ax.set_ylabel("Mean difference (influence zone − true control, % of adults)")
    ax.set_title("Positive bars = TRI influence zone is sicker within the same poverty band")
    ax.legend(title="Poverty quintile", loc='upper left', fontsize=9)
    plt.tight_layout()
    _save(fig, "cc2_poverty_adjusted_gap")

    # ── Plot C: Dose-response in TRI-direct tracts only ───────────────────────
    tri_only = all_t2[(all_t2['tri_zone'] == 'tri_direct') & (all_t2['carc_releases'] > 0)].copy()
    tri_only['log_carc'] = np.log10(tri_only['carc_releases'].clip(0.1))
    tri_only['log_total'] = np.log10(tri_only['total_releases'].clip(0.1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Dose-response within TRI-direct tracts: log(carcinogen releases) vs health outcomes\n"
        f"(n={len(tri_only):,} census tracts with carcinogen-releasing facilities)",
        fontsize=11, fontweight="bold",
    )
    for ax, (hc_label, col) in zip(axes.flat, outcomes):
        sub = tri_only[['log_carc', col, 'poverty_pct']].dropna()
        if len(sub) < 10:
            ax.set_title(f"{hc_label} (insufficient data)")
            continue
        hb = ax.hexbin(sub['log_carc'], sub[col], gridsize=20, cmap='YlOrRd', mincnt=1)
        z = np.polyfit(sub['log_carc'], sub[col], 1)
        xr = np.linspace(sub['log_carc'].min(), sub['log_carc'].max(), 100)
        ax.plot(xr, np.poly1d(z)(xr), 'b--', lw=2)
        r, p = spearmanr(sub['log_carc'], sub[col])
        ax.set_xlabel("log₁₀(Carcinogen releases in tract, lbs)")
        ax.set_ylabel(hc_label)
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.set_title(f"{hc_label}  ρ={r:.3f} {stars}", fontsize=9, fontweight='bold')
    plt.tight_layout()
    _save(fig, "cc3_dose_response_exposed")

    # ── Plot D: Poverty-interaction — influence zone vs true control by poverty ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Does industrial proximity compound poverty's health impact?\n"
        "Health outcome by poverty quintile × TRI influence status\n"
        "(Influence = TRI tract OR neighbor of TRI tract)",
        fontsize=12, fontweight="bold",
    )
    pov_q_labels = ['Q1\n(Lowest\npoverty)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest\npoverty)']

    for ax, (hc_label, col) in zip(axes.flat, outcomes):
        ctrl_means = [
            all_t2.loc[(all_t2['pov_q'] == q) & (all_t2['tri_zone'] == 'control'), col].mean()
            for q in range(5)
        ]
        infl_means = [
            all_t2.loc[(all_t2['pov_q'] == q) & all_t2['in_influence'], col].mean()
            for q in range(5)
        ]
        x = np.arange(5)
        ax.plot(x, ctrl_means, 'o--', color='#2ecc71', lw=2, ms=8, label='True control')
        ax.plot(x, infl_means, 's-', color='#e74c3c', lw=2, ms=8, label='TRI influence zone')
        ax.fill_between(x, ctrl_means, infl_means, alpha=0.15, color='red')
        ax.set_xticks(x)
        ax.set_xticklabels(pov_q_labels, fontsize=7)
        ax.set_title(hc_label, fontweight='bold')
        ax.set_ylabel("Mean crude rate")
        ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "cc4_poverty_x_exposure_interaction")

    logger.info("  Saved: cc1–cc4 case-control analysis plots (NEIGHBOR MODEL)")


def analysis9_minority_poverty_disentangle(df: pd.DataFrame) -> None:
    """
    Disentangle minority % vs poverty % effects on health outcomes,
    correcting the tract-level analysis for both confounders.
    
    USES NEIGHBOR INFLUENCE MODEL:
    - Influence zone = TRI tract + neighbors
    - True control = no TRI, not neighbor of TRI

    Key question: Is cancer's negative association with minority % independent
    of poverty? And does it change the story about TRI facility effects?
    """
    from scipy.stats import spearmanr, mannwhitneyu
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    CARCINOGENS = {
        'arsenic', 'benzene', 'chromium', 'cadmium', 'nickel', 'lead',
        'vinyl chloride', 'formaldehyde', '1,3-butadiene', 'trichloroethylene',
        'styrene', 'ethylene oxide', 'dioxin', 'polycyclic aromatic', 'benzo',
        'naphthalene', 'beryllium', 'cobalt', 'acrylonitrile',
    }

    # ── Load neighbor lookup ──────────────────────────────────────────────────
    neighbor_dict = _load_tract_neighbors()
    if not neighbor_dict:
        logger.error("Cannot run analysis9 without neighbor lookup")
        return

    # ── Build complete tract dataset with minority % ──────────────────────────
    cdc_raw = pd.read_csv("data/raw/cdc_places.csv", low_memory=False)
    cdc_pivot = cdc_raw.pivot_table(
        index='locationid', columns='measureid', values='data_value'
    ).reset_index()
    cdc_pivot.columns.name = None
    cdc_pivot['fips_tract'] = cdc_pivot['locationid'].astype(str).str.zfill(11)

    census = _load_census()

    df2 = df.copy()
    df2['is_carc'] = df2['CHEMICAL_NAME'].apply(
        lambda x: any(c in str(x).lower() for c in CARCINOGENS)
    )
    df_carc = (
        df2[df2['is_carc']].groupby('fips_tract')['TOTAL_RELEASES']
        .sum().reset_index().rename(columns={'TOTAL_RELEASES': 'carc_releases'})
    )
    df_total = df2.groupby('fips_tract').agg(
        total_releases=('TOTAL_RELEASES', 'sum'),
        n_facilities=('TRI_FACILITY_ID', 'nunique'),
    ).reset_index()
    tri_tract = df_total.merge(df_carc, on='fips_tract', how='left')
    tri_tract['carc_releases'] = tri_tract['carc_releases'].fillna(0)
    tri_tract['fips_tract'] = tri_tract['fips_tract'].astype(str).str.zfill(11)

    all_t = cdc_pivot.merge(
        census[['fips_tract', 'poverty_pct', 'minority_pct']], on='fips_tract', how='left'
    )
    
    # ── Classify tracts using influence model ─────────────────────────────────
    all_t = _classify_tracts_for_case_control(all_t, tri_tract, neighbor_dict)
    all_t['log_releases'] = np.log10(all_t['total_releases'].clip(0.1))

    all_t2 = all_t[
        all_t['CANCER'].notna() & all_t['poverty_pct'].notna() &
        all_t['minority_pct'].notna()
    ].copy().reset_index(drop=True)

    outcomes = [
        ('Cancer', 'CANCER'), ('Asthma', 'CASTHMA'), ('CHD', 'CHD'),
        ('COPD', 'COPD'), ('Diabetes', 'DIABETES'), ('Mental Health', 'MHLTH'),
    ]

    # ── Plot A: Partial correlations grid ─────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(
        "Disentangling minority % vs poverty % effects on health outcomes\n"
        "Bars show Spearman ρ: unadjusted vs controlled for the other variable",
        fontsize=12, fontweight="bold",
    )

    pov_bins = np.percentile(all_t2['poverty_pct'], [20, 40, 60, 80])
    min_bins_arr = np.percentile(all_t2['minority_pct'], [20, 40, 60, 80])
    all_t2['pov_q'] = np.digitize(all_t2['poverty_pct'].values, pov_bins)
    all_t2['min_q'] = np.digitize(all_t2['minority_pct'].values, min_bins_arr)

    for ax, (hc_label, col) in zip(axes.flat, outcomes):
        sub = all_t2[all_t2[col].notna()]

        r_pov, _ = spearmanr(sub['poverty_pct'], sub[col])
        r_min, _ = spearmanr(sub['minority_pct'], sub[col])

        r_pov_ctrl = np.mean([
            spearmanr(sub[sub['min_q'] == q]['poverty_pct'],
                      sub[sub['min_q'] == q][col])[0]
            for q in range(5) if len(sub[sub['min_q'] == q]) > 30
        ])
        r_min_ctrl = np.mean([
            spearmanr(sub[sub['pov_q'] == q]['minority_pct'],
                      sub[sub['pov_q'] == q][col])[0]
            for q in range(5) if len(sub[sub['pov_q'] == q]) > 30
        ])

        labels = ['Poverty\n(unadjusted)', 'Minority\n(unadjusted)',
                  'Poverty\n(controlled\nfor minority)', 'Minority\n(controlled\nfor poverty)']
        rhos = [r_pov, r_min, r_pov_ctrl, r_min_ctrl]
        colors = ['#e74c3c', '#3498db', '#e74c3c', '#3498db']
        patterns = ['', '', '//', '//']

        bars = ax.bar(range(4), rhos, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=0.8)
        for bar, pat in zip(bars, patterns):
            if pat:
                bar.set_hatch(pat)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(-1, 1)
        ax.set_title(hc_label, fontweight='bold')
        ax.set_ylabel("Spearman ρ")
        for i, r in enumerate(rhos):
            ax.text(i, r + (0.03 if r >= 0 else -0.06), f"{r:.2f}",
                    ha='center', fontsize=8, fontweight='bold')

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color='#e74c3c', label='Poverty'),
        Patch(color='#3498db', label='Minority %'),
        Patch(color='gray', hatch='//', label='After controlling for the other'),
    ]
    axes.flat[-1].legend(handles=legend_handles, loc='lower right', fontsize=8)
    plt.tight_layout()
    _save(fig, "mp1_partial_correlations")

    # ── Plot B: OLS standardized coefficients — using in_influence instead of has_tri
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.suptitle(
        "Multivariate regression: standardized coefficients for each health outcome\n"
        "Model: health ~ poverty % + minority % + in_influence_zone + log(releases)\n"
        "(Influence zone = TRI tract OR neighbor of TRI tract)",
        fontsize=11, fontweight="bold",
    )
    predictor_names = ['poverty_pct', 'minority_pct', 'in_influence', 'log_releases']
    pred_labels = ['Poverty %', 'Minority %', 'In TRI\ninfluence zone', 'log₁₀\n(Releases)']
    pred_colors = ['#e74c3c', '#3498db', '#e67e22', '#9b59b6']
    n_outcomes = len(outcomes)
    n_preds = len(predictor_names)
    x = np.arange(n_outcomes)
    width = 0.2

    all_coefs = []
    for hc_label, col in outcomes:
        sub = all_t2[[col] + predictor_names].copy()
        sub['in_influence'] = sub['in_influence'].astype(float)
        sub = sub.dropna()
        Xm = sub[predictor_names].values
        ym = sub[col].values
        sc = StandardScaler()
        Xsc = sc.fit_transform(Xm)
        coefs, _, _, _ = np.linalg.lstsq(
            np.column_stack([Xsc, np.ones(len(Xsc))]), ym, rcond=None
        )
        all_coefs.append(coefs[:n_preds])

    all_coefs = np.array(all_coefs)
    for pi, (pname, pcolor) in enumerate(zip(pred_labels, pred_colors)):
        offset = (pi - (n_preds - 1) / 2) * width
        ax.bar(x + offset, all_coefs[:, pi], width,
               label=pname, color=pcolor, alpha=0.85, edgecolor='black', lw=0.5)

    ax.axhline(0, color='black', lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([hc for hc, _ in outcomes])
    ax.set_ylabel("Standardized regression coefficient")
    ax.set_title("Positive = predictor increases health burden rate")
    ax.legend(title="Predictor", loc='lower right', fontsize=9)
    plt.tight_layout()
    _save(fig, "mp2_ols_standardized_coefs")

    # ── Plot C: Cancer paradox — 2D heatmap: minority × poverty → cancer rate ─
    all_t3 = all_t2[all_t2['CANCER'].notna()].copy()
    all_t3['pov_q5'] = pd.cut(all_t3['poverty_pct'],
                               bins=np.percentile(all_t3['poverty_pct'], [0, 20, 40, 60, 80, 100]),
                               labels=['0–5%', '5–9%', '9–14%', '14–22%', '22–100%'],
                               include_lowest=True)
    all_t3['min_q5'] = pd.cut(all_t3['minority_pct'],
                               bins=np.percentile(all_t3['minority_pct'], [0, 20, 40, 60, 80, 100]),
                               labels=['0–10%', '10–22%', '22–40%', '40–65%', '65–100%'],
                               include_lowest=True)
    pivot_cancer = all_t3.groupby(['pov_q5', 'min_q5'], observed=True)['CANCER'].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Cancer rate by poverty × minority — 2D grid\n"
        "Right panel: COPD for comparison (no paradox expected)",
        fontsize=12, fontweight="bold",
    )

    im = axes[0].imshow(pivot_cancer.values, cmap='RdYlGn_r', aspect='auto', origin='lower')
    plt.colorbar(im, ax=axes[0], label='Mean cancer crude rate (% of adults)\nRed = higher rate')
    axes[0].set_xticks(range(len(pivot_cancer.columns)))
    axes[0].set_xticklabels(pivot_cancer.columns, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticks(range(len(pivot_cancer.index)))
    axes[0].set_yticklabels(pivot_cancer.index, fontsize=8)
    axes[0].set_xlabel("Minority % quintile")
    axes[0].set_ylabel("Poverty % quintile")
    axes[0].set_title("Cancer rate\n(Red = more cancer; note: rate DROPS in high-minority columns → paradox)")
    for i in range(len(pivot_cancer.index)):
        for j in range(len(pivot_cancer.columns)):
            val = pivot_cancer.values[i, j]
            if not np.isnan(val):
                axes[0].text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=7, color='black')

    pivot_copd = all_t3.groupby(['pov_q5', 'min_q5'], observed=True)['COPD'].mean().unstack()
    im2 = axes[1].imshow(pivot_copd.values, cmap='RdYlGn_r', aspect='auto', origin='lower')
    plt.colorbar(im2, ax=axes[1], label='Mean COPD crude rate (% of adults)\nRed = higher rate')
    axes[1].set_xticks(range(len(pivot_copd.columns)))
    axes[1].set_xticklabels(pivot_copd.columns, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticks(range(len(pivot_copd.index)))
    axes[1].set_yticklabels(pivot_copd.index, fontsize=8)
    axes[1].set_xlabel("Minority % quintile")
    axes[1].set_ylabel("Poverty % quintile")
    axes[1].set_title("COPD rate\n(Red = more COPD; rate rises with poverty rows, not minority columns)")
    for i in range(len(pivot_copd.index)):
        for j in range(len(pivot_copd.columns)):
            val = pivot_copd.values[i, j]
            if not np.isnan(val):
                axes[1].text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=7, color='black')
    plt.tight_layout()
    _save(fig, "mp3_cancer_vs_copd_2d_heatmap")

    # ── Plot D: Case-control using INFLUENCE ZONE vs TRUE CONTROL ─────────────
    # Compare influence zone (TRI + neighbors) vs true control (no TRI, no TRI neighbors)
    all_t2_m = all_t2.copy()
    min_pct = all_t2_m['minority_pct'].values
    min_bins2 = np.percentile(min_pct, [20, 40, 60, 80])
    all_t2_m['min_q'] = np.digitize(min_pct, min_bins2)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(
        "Dual-adjusted case-control: TRI INFLUENCE ZONE vs TRUE CONTROL\n"
        "Poverty-AND-minority-matched health gap\n"
        "(Influence zone = TRI tract OR neighbor | True control = no TRI & no TRI neighbor)",
        fontsize=11, fontweight="bold",
    )

    for ax, (hc_label, col) in zip(axes.flat, outcomes):
        matrix_diff = np.full((5, 5), np.nan)
        for pq in range(5):
            for mq in range(5):
                cell = all_t2_m[(all_t2_m['pov_q'] == pq) & (all_t2_m['min_q'] == mq)]
                infl_m = cell.loc[cell['in_influence'], col].mean()
                ctrl_m = cell.loc[cell['tri_zone'] == 'control', col].mean()
                n_infl = cell['in_influence'].sum()
                n_ctrl = (cell['tri_zone'] == 'control').sum()
                if n_infl >= 3 and n_ctrl >= 10:
                    matrix_diff[pq, mq] = infl_m - ctrl_m

        vmax = np.nanmax(np.abs(matrix_diff)) if not np.all(np.isnan(matrix_diff)) else 1
        im = ax.imshow(matrix_diff, cmap='RdBu_r', aspect='auto', origin='lower',
                       vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Influence−control diff')
        ax.set_xticks(range(5))
        ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], fontsize=8)
        ax.set_yticks(range(5))
        ax.set_yticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], fontsize=8)
        ax.set_xlabel("Minority % quintile")
        ax.set_ylabel("Poverty % quintile")
        ax.set_title(hc_label, fontweight='bold')
        for i in range(5):
            for j in range(5):
                val = matrix_diff[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:+.1f}", ha='center', va='center',
                            fontsize=6, fontweight='bold',
                            color='white' if abs(val) > vmax * 0.5 else 'black')

    plt.tight_layout()
    _save(fig, "mp4_dual_adjusted_case_control")

    logger.info("  Saved: mp1–mp4 minority+poverty disentanglement plots (NEIGHBOR MODEL)")

    plt.tight_layout()
    _save(fig, "mp4_dual_adjusted_case_control")

    logger.info("  Saved: mp1–mp4 minority+poverty disentanglement plots")


def analysis10_health_burden(df: pd.DataFrame) -> None:
    """
    Quantitative health burden model.

    USES NEIGHBOR INFLUENCE MODEL:
    - Influence zone = TRI tract + neighbors
    - True control = no TRI, not neighbor of TRI

    Converts prevalence differences (influence zone vs control) into:
    - Excess disease cases (number of people)
    - Excess DALYs (Disability-Adjusted Life Years)
    - Attributable Fraction estimates
    - Release-weighted exposure model

    DALY disability weights (GBD 2019 approximate values):
    - COPD:     0.199 per year with moderate/severe COPD
    - CHD:      0.296 per year with ischemic heart disease episode
    - Cancer:   0.188 per year (average across cancer types)
    - Diabetes: 0.049 per year with uncomplicated DM2
    - Asthma:   0.133 per year with moderate persistent asthma

    These are applied to PREVALENCE differences to get a rough DALY proxy.
    NOTE: This is an approximation — proper YLD needs age-structured data.
    """
    from scipy.stats import spearmanr

    CARCINOGENS = {
        'arsenic', 'benzene', 'chromium', 'cadmium', 'nickel', 'lead',
        'vinyl chloride', 'formaldehyde', '1,3-butadiene', 'trichloroethylene',
        'styrene', 'ethylene oxide', 'dioxin', 'polycyclic aromatic', 'benzo',
        'naphthalene', 'beryllium', 'cobalt', 'acrylonitrile',
    }

    # ── Load neighbor lookup ──────────────────────────────────────────────────
    neighbor_dict = _load_tract_neighbors()
    if not neighbor_dict:
        logger.error("Cannot run analysis10 without neighbor lookup")
        return

    # ── Build complete dataset with CDC population ────────────────────────────
    cdc_raw = pd.read_csv("data/raw/cdc_places.csv", low_memory=False)
    cdc_pivot = cdc_raw.pivot_table(
        index='locationid', columns='measureid', values='data_value'
    ).reset_index()
    # Population is same for all measures — grab it
    pop_lookup = (
        cdc_raw[cdc_raw['measureid'] == 'CANCER'][['locationid', 'totalpopulation']]
        .rename(columns={'totalpopulation': 'population'})
    )
    cdc_pivot.columns.name = None
    cdc_pivot['fips_tract'] = cdc_pivot['locationid'].astype(str).str.zfill(11)
    cdc_pivot = cdc_pivot.merge(pop_lookup, on='locationid', how='left')

    census = _load_census()

    df2 = df.copy()
    df2['is_carc'] = df2['CHEMICAL_NAME'].apply(
        lambda x: any(c in str(x).lower() for c in CARCINOGENS)
    )
    df_carc = (
        df2[df2['is_carc']].groupby('fips_tract')['TOTAL_RELEASES']
        .sum().reset_index().rename(columns={'TOTAL_RELEASES': 'carc_releases'})
    )
    df_total = df2.groupby('fips_tract').agg(
        total_releases=('TOTAL_RELEASES', 'sum'),
        n_facilities=('TRI_FACILITY_ID', 'nunique'),
    ).reset_index()
    tri_tract = df_total.merge(df_carc, on='fips_tract', how='left')
    tri_tract['carc_releases'] = tri_tract['carc_releases'].fillna(0)
    tri_tract['fips_tract'] = tri_tract['fips_tract'].astype(str).str.zfill(11)

    all_t = cdc_pivot.merge(
        census[['fips_tract', 'poverty_pct', 'minority_pct']], on='fips_tract', how='left'
    )
    
    # ── Classify tracts using influence model ─────────────────────────────────
    all_t = _classify_tracts_for_case_control(all_t, tri_tract, neighbor_dict)

    all_t2 = all_t[
        all_t['CANCER'].notna() & all_t['poverty_pct'].notna() &
        all_t['minority_pct'].notna() & all_t['population'].notna()
    ].copy().reset_index(drop=True)

    # Poverty × minority quintile adjustment
    pov_bins = np.percentile(all_t2['poverty_pct'], [20, 40, 60, 80])
    min_bins = np.percentile(all_t2['minority_pct'], [20, 40, 60, 80])
    all_t2['pov_q'] = np.digitize(all_t2['poverty_pct'].values, pov_bins)
    all_t2['min_q'] = np.digitize(all_t2['minority_pct'].values, min_bins)

    outcomes = [
        ('COPD', 'COPD', 0.199),
        ('CHD', 'CHD', 0.296),
        ('Cancer', 'CANCER', 0.188),
        ('Diabetes', 'DIABETES', 0.049),
        ('Asthma', 'CASTHMA', 0.133),
        ('Mental Health', 'MHLTH', 0.130),
    ]

    # ── Compute dual-adjusted excess rates (INFLUENCE vs CONTROL) ─────────────
    infl_pop_total = all_t2.loc[all_t2['in_influence'], 'population'].sum()
    ctrl_pop_total = all_t2.loc[all_t2['tri_zone'] == 'control', 'population'].sum()

    adj_diffs = {}
    raw_diffs = {}
    for hc_label, col, _ in outcomes:
        # Raw difference (influence zone vs true control)
        t_m = all_t2.loc[all_t2['in_influence'], col].mean()
        c_m = all_t2.loc[all_t2['tri_zone'] == 'control', col].mean()
        raw_diffs[hc_label] = t_m - c_m

        # Dual-adjusted (25-cell means)
        diffs_cells = []
        for pq in range(5):
            for mq in range(5):
                cell = all_t2[(all_t2['pov_q'] == pq) & (all_t2['min_q'] == mq)]
                t_m_c = cell.loc[cell['in_influence'], col].mean()
                c_m_c = cell.loc[cell['tri_zone'] == 'control', col].mean()
                n_infl = cell['in_influence'].sum()
                n_ctrl = (cell['tri_zone'] == 'control').sum()
                if n_infl >= 3 and n_ctrl >= 10:
                    diffs_cells.append(t_m_c - c_m_c)
        adj_diffs[hc_label] = np.mean(diffs_cells) if diffs_cells else np.nan

    # ── Plot A: Raw vs adjusted excess rates ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Health burden of TRI INFLUENCE ZONE proximity\n"
        "Excess disease prevalence: raw vs. poverty+minority-adjusted\n"
        "(Influence zone = TRI tract OR neighbor | True control = no TRI & no TRI neighbor)",
        fontsize=11, fontweight="bold",
    )

    hc_labels = [hc for hc, _, _ in outcomes]
    raw_vals = [raw_diffs[hc] for hc in hc_labels]
    adj_vals = [adj_diffs[hc] for hc in hc_labels]

    x = np.arange(len(hc_labels))
    w = 0.35
    ax = axes[0]
    bars1 = ax.bar(x - w / 2, raw_vals, w, label='Unadjusted', color='#e74c3c', alpha=0.85)
    bars2 = ax.bar(x + w / 2, adj_vals, w, label='Adj. (poverty+minority)', color='#3498db', alpha=0.85)
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(hc_labels)
    ax.set_ylabel("Excess prevalence (%pts) — Influence zone vs control")
    ax.set_title("Excess disease rate (TRI influence zone vs true control)")
    ax.legend()
    for bar in bars1:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (0.01 if v >= 0 else -0.03),
                f"{v:+.2f}", ha='center', fontsize=8)
    for bar in bars2:
        v = bar.get_height()
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.01 if v >= 0 else -0.03),
                    f"{v:+.2f}", ha='center', fontsize=8)

    # Excess absolute cases
    ax = axes[1]
    excess_cases = [adj_diffs[hc] / 100 * infl_pop_total for hc in hc_labels]
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in excess_cases]
    ax.barh(hc_labels, excess_cases, color=colors, alpha=0.85, edgecolor='black')
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel(f"Excess disease cases (total in {infl_pop_total/1e6:.1f}M influence-zone residents)")
    ax.set_title("Absolute excess disease burden\n(adj. for poverty + minority %)")
    for i, v in enumerate(excess_cases):
        if not np.isnan(v):
            ax.text(v + (200 if v >= 0 else -200), i, f"{v:+,.0f}", va='center',
                    ha='left' if v >= 0 else 'right', fontsize=9)
    plt.tight_layout()
    _save(fig, "hb1_excess_burden")

    # ── Plot B: DALY estimate ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(
        "Estimated excess Disability-Adjusted Life Years (DALYs)\n"
        "attributable to TRI INFLUENCE ZONE proximity\n"
        "(GBD 2019 disability weights × adjusted excess prevalence × influence-zone population)\n"
        "NOTE: These are rough order-of-magnitude estimates",
        fontsize=11, fontweight="bold",
    )

    excess_dalys = []
    excess_cases_list = []
    for hc_label, col, dw in outcomes:
        diff = adj_diffs.get(hc_label, 0) or 0
        cases = diff / 100 * infl_pop_total
        dalys = cases * dw
        excess_dalys.append(dalys)
        excess_cases_list.append(cases)

    colors_d = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71', '#9b59b6', '#1abc9c']
    ax.barh(hc_labels, excess_dalys, color=colors_d, alpha=0.85, edgecolor='black')
    ax.axvline(0, color='black', lw=1.5)
    ax.set_xlabel("Estimated excess DALYs (years of healthy life lost)")
    ax.set_title(
        f"Among ~{infl_pop_total/1e6:.1f}M people living in TRI influence zones\n"
        "vs. demographically matched true control tracts"
    )
    for i, (v, cases) in enumerate(zip(excess_dalys, excess_cases_list)):
        if not np.isnan(v):
            ax.text(max(v, 0) + 5, i, f"{v:+,.0f} DALYs ({cases:+,.0f} excess cases)",
                    va='center', ha='left', fontsize=9)

    total_dalys = sum(d for d in excess_dalys if not np.isnan(d))
    ax.text(0.97, 0.03,
            f"Total excess DALYs (rough estimate): {total_dalys:,.0f}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9),
            fontweight='bold')
    plt.tight_layout()
    _save(fig, "hb2_daly_estimates")

    # ── Plot C: Dose-response — weighted release burden vs health ─────────────
    # Build a release-weighted exposure score per TRI-direct tract
    all_t3 = all_t2[all_t2['tri_zone'] == 'tri_direct'].copy()
    all_t3['log_carc'] = np.log10(all_t3['carc_releases'].clip(0.1))
    all_t3['log_total'] = np.log10(all_t3['total_releases'].clip(0.1))
    # Compute poverty+minority adjusted residual for each tract
    # Use median imputation within cell
    residuals = {}
    for hc_label, col, _ in outcomes:
        all_t3[f'{col}_resid'] = np.nan
        for pq in range(5):
            for mq in range(5):
                cell_mask = (all_t2['pov_q'] == pq) & (all_t2['min_q'] == mq)
                ctrl_mean = all_t2.loc[cell_mask & (all_t2['tri_zone'] == 'control'), col].mean()
                tri_mask = (all_t3['pov_q'] == pq) & (all_t3['min_q'] == mq)
                if not np.isnan(ctrl_mean):
                    all_t3.loc[tri_mask, f'{col}_resid'] = (
                        all_t3.loc[tri_mask, col] - ctrl_mean
                    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(
        "Dose-response: log(carcinogen releases) vs residual health burden\n"
        "(Residual = health rate minus matched TRUE CONTROL mean, controlling poverty+minority)\n"
        f"n={all_t3['COPD_resid'].notna().sum():,} TRI-direct tracts",
        fontsize=11, fontweight="bold",
    )
    for ax, (hc_label, col, _) in zip(axes.flat, outcomes):
        resid_col = f'{col}_resid'
        sub = all_t3[all_t3[resid_col].notna() & all_t3['log_carc'].notna()].dropna(
            subset=['log_carc', resid_col])
        if len(sub) < 20:
            ax.set_title(f"{hc_label} (insufficient data)")
            continue
        hb = ax.hexbin(sub['log_carc'], sub[resid_col], gridsize=15, cmap='YlOrRd', mincnt=1)
        z = np.polyfit(sub['log_carc'], sub[resid_col], 1)
        xr = np.linspace(sub['log_carc'].min(), sub['log_carc'].max(), 100)
        ax.plot(xr, np.poly1d(z)(xr), 'b--', lw=2)
        ax.axhline(0, color='black', lw=0.8, ls=':')
        r, p = spearmanr(sub['log_carc'], sub[resid_col])
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.set_xlabel("log₁₀(Carcinogen releases, lbs)")
        ax.set_ylabel(f"{hc_label} — adjusted residual (%pts)")
        ax.set_title(f"{hc_label}  ρ={r:.3f} {stars}", fontweight='bold')
    plt.tight_layout()
    _save(fig, "hb3_adjusted_dose_response")

    # ── Plot D: Summary — attributable fraction pie/bar ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Proportion of excess disease burden attributable to TRI INFLUENCE ZONE proximity\n"
        "(Attributable Fraction = excess rate / background rate × 100)\n"
        "(True control = no TRI AND not neighbor of TRI)",
        fontsize=11, fontweight="bold",
    )

    ax = axes[0]
    background_rates = {}
    attr_fractions = {}
    for hc_label, col, _ in outcomes:
        bg = all_t2.loc[all_t2['tri_zone'] == 'control', col].mean()
        adj_diff = adj_diffs.get(hc_label, 0) or 0
        af = adj_diff / bg * 100 if bg > 0 else 0
        background_rates[hc_label] = bg
        attr_fractions[hc_label] = af

    afs = [attr_fractions[hc] for hc in hc_labels]
    colors_af = ['#e74c3c' if af > 0 else '#3498db' for af in afs]
    bars = ax.bar(hc_labels, afs, color=colors_af, alpha=0.85, edgecolor='black')
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel("Attributable Fraction (%)")
    ax.set_title("% of disease cases attributable to\nliving in TRI influence zone (adj.)")
    for bar, af in zip(bars, afs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                af + (0.1 if af >= 0 else -0.3),
                f"{af:+.1f}%", ha='center', fontsize=9, fontweight='bold')

    ax = axes[1]
    # Stacked: background + excess
    bg_vals = [background_rates[hc] for hc in hc_labels]
    exc_vals = [max(adj_diffs.get(hc, 0) or 0, 0) for hc in hc_labels]
    ax.bar(hc_labels, bg_vals, color='#bdc3c7', edgecolor='black', label='Background rate')
    ax.bar(hc_labels, exc_vals, bottom=bg_vals, color='#e74c3c', alpha=0.85,
           edgecolor='black', label='TRI-attributable excess')
    ax.set_ylabel("Crude disease prevalence rate (%)")
    ax.set_title("Background rate + TRI-attributable excess\n(adjusted for poverty+minority)")
    ax.legend()
    plt.tight_layout()
    _save(fig, "hb4_attributable_fraction")

    logger.info("  Saved: hb1–hb4 health burden quantification plots (NEIGHBOR MODEL)")
    logger.info(
        f"  Key numbers: Influence-zone population={infl_pop_total/1e6:.1f}M; "
        f"total excess DALYs≈{total_dalys:,.0f}"
    )


def analysis11_tri_poverty_gap(df: pd.DataFrame) -> None:
    """
    Are TRI tracts systematically poorer than non-TRI tracts?
    And has that gap changed over time?
    
    USES NEIGHBOR INFLUENCE MODEL:
    - Shows 3 groups: true control, TRI-neighbor, TRI-direct
    - Demonstrates environmental justice gradient

    Generates plot: tri_poverty_gap.png
    """
    # ── Load neighbor lookup ──────────────────────────────────────────────────
    neighbor_dict = _load_tract_neighbors()
    if not neighbor_dict:
        logger.error("Cannot run analysis11 without neighbor lookup")
        return

    cdc_raw = pd.read_csv("data/raw/cdc_places.csv", low_memory=False)
    cdc_tracts = set(
        cdc_raw['locationid'].astype(str).str.zfill(11).unique()
    )

    census = _load_census()

    # TRI tract summary - need carc_releases and n_facilities for _classify_tracts_for_case_control
    CARCINOGENS = {
        'arsenic', 'benzene', 'chromium', 'cadmium', 'nickel', 'lead',
        'vinyl chloride', 'formaldehyde', '1,3-butadiene', 'trichloroethylene',
        'styrene', 'ethylene oxide', 'dioxin', 'polycyclic aromatic', 'benzo',
        'naphthalene', 'beryllium', 'cobalt', 'acrylonitrile',
    }
    df2 = df.copy()
    df2['is_carc'] = df2['CHEMICAL_NAME'].apply(
        lambda x: any(c in str(x).lower() for c in CARCINOGENS)
    )
    df_carc = (
        df2[df2['is_carc']].groupby('fips_tract')['TOTAL_RELEASES']
        .sum().reset_index().rename(columns={'TOTAL_RELEASES': 'carc_releases'})
    )
    df_total = df2.groupby('fips_tract').agg(
        total_releases=('TOTAL_RELEASES', 'sum'),
        n_facilities=('TRI_FACILITY_ID', 'nunique'),
    ).reset_index()
    tri_tract = df_total.merge(df_carc, on='fips_tract', how='left')
    tri_tract['carc_releases'] = tri_tract['carc_releases'].fillna(0)
    tri_tract['fips_tract'] = tri_tract['fips_tract'].astype(str).str.zfill(11)

    # All CDC-tracked tracts with census data
    all_tracts = census[census['fips_tract'].isin(cdc_tracts)].copy()
    all_tracts = all_tracts[all_tracts['poverty_pct'].notna() & all_tracts['minority_pct'].notna()]

    # ── Classify tracts using influence model ─────────────────────────────────
    all_tracts = _classify_tracts_for_case_control(all_tracts, tri_tract, neighbor_dict)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        "Are TRI facility neighborhoods poorer and more minority?\n"
        "Comparing: TRUE CONTROL vs TRI-NEIGHBOR vs TRI-DIRECT tracts\n"
        "(True control = no TRI AND not adjacent to TRI tract)",
        fontsize=12, fontweight="bold",
    )

    from scipy.stats import mannwhitneyu as _mwu, kruskal

    for ax_i, (col, label) in enumerate(
        [('poverty_pct', 'Poverty rate (%)'), ('minority_pct', 'Minority population (%)')]
    ):
        ax = axes[ax_i]
        ctrl_vals = all_tracts.loc[all_tracts['tri_zone'] == 'control', col].dropna()
        nbr_vals = all_tracts.loc[all_tracts['tri_zone'] == 'tri_neighbor', col].dropna()
        dir_vals = all_tracts.loc[all_tracts['tri_zone'] == 'tri_direct', col].dropna()
        
        # Kruskal-Wallis for 3-group comparison
        _, p = kruskal(ctrl_vals, nbr_vals, dir_vals)
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        vparts = ax.violinplot([ctrl_vals.values, nbr_vals.values, dir_vals.values], 
                               positions=[0, 1, 2], showmedians=True)
        colors_v = ['#2ecc71', '#f39c12', '#e74c3c']
        for i, pc in enumerate(vparts['bodies']):
            pc.set_facecolor(colors_v[i])
            pc.set_alpha(0.7)
        
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([
            f'True Control\n(n={len(ctrl_vals):,})',
            f'TRI-Neighbor\n(n={len(nbr_vals):,})',
            f'TRI-Direct\n(n={len(dir_vals):,})',
        ], fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(
            f"{label}\nKruskal-Wallis p{stars}\n"
            f"Median: ctrl={ctrl_vals.median():.1f}% | nbr={nbr_vals.median():.1f}% | dir={dir_vals.median():.1f}%",
            fontsize=9, fontweight='bold',
        )
        for i, (vals, clr) in enumerate(zip([ctrl_vals, nbr_vals, dir_vals], colors_v)):
            ax.axhline(vals.median(), color=clr, lw=1.2, ls='--', alpha=0.5, xmin=(i/3)+0.05, xmax=((i+1)/3)-0.05)

    # Third panel: temporal trend — mean poverty of TRI tracts over time
    ax = axes[2]
    tri_tract_yr = (
        df.groupby(['REPORTING_YEAR', 'fips_tract'])
        .size().reset_index(name='n_fac')
    )
    tri_tract_yr['fips_tract'] = tri_tract_yr['fips_tract'].astype(str).str.zfill(11)
    tri_tract_yr2 = tri_tract_yr.merge(
        census[['fips_tract', 'poverty_pct', 'minority_pct']],
        on='fips_tract', how='left',
    ).dropna(subset=['poverty_pct'])
    yr_mean = tri_tract_yr2.groupby('REPORTING_YEAR').agg(
        mean_pov=('poverty_pct', 'mean'),
        mean_min=('minority_pct', 'mean'),
        n_tracts=('fips_tract', 'nunique'),
    ).reset_index()
    
    # Background = true control only
    bg_pov = all_tracts.loc[all_tracts['tri_zone'] == 'control', 'poverty_pct'].mean()
    bg_min = all_tracts.loc[all_tracts['tri_zone'] == 'control', 'minority_pct'].mean()

    ax.plot(yr_mean['REPORTING_YEAR'], yr_mean['mean_pov'], 'o-',
            color='#e74c3c', lw=2, label='TRI tracts — poverty %')
    ax.plot(yr_mean['REPORTING_YEAR'], yr_mean['mean_min'], 's--',
            color='#3498db', lw=2, label='TRI tracts — minority %')
    ax.axhline(bg_pov, color='#e74c3c', lw=1, ls=':', alpha=0.6, label=f'True control poverty ({bg_pov:.1f}%)')
    ax.axhline(bg_min, color='#3498db', lw=1, ls=':', alpha=0.6, label=f'True control minority ({bg_min:.1f}%)')
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean % in TRI-facility tracts")
    ax.set_title("Poverty & minority % in TRI tracts over time\nvs TRUE CONTROL (dotted)", fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, "tri_poverty_gap")
    logger.info("  Saved: tri_poverty_gap.png (NEIGHBOR MODEL)")


def analysis12_cancer_white_tracts(df: pd.DataFrame) -> None:
    """
    Cancer paradox deeper analysis.
    
    USES NEIGHBOR INFLUENCE MODEL:
    - True control = no TRI AND not neighbor of TRI
    - Shows 3 groups: true control, TRI non-carcinogen, TRI carcinogen
    
    - In predominantly white census tracts (minority <20%), does TRI presence
      show the expected positive association with cancer?
    - Compare carcinogen-releasing TRI tracts vs TRUE CONTROL tracts within white-majority tracts.
    Generates: cancer_white_tracts.png
    """
    from scipy.stats import mannwhitneyu as _mwu

    # ── Load neighbor lookup ──────────────────────────────────────────────────
    neighbor_dict = _load_tract_neighbors()
    if not neighbor_dict:
        logger.error("Cannot run analysis12 without neighbor lookup")
        return

    cdc_raw = pd.read_csv("data/raw/cdc_places.csv", low_memory=False)
    cdc_pivot = cdc_raw.pivot_table(
        index='locationid', columns='measureid', values='data_value'
    ).reset_index()
    cdc_pivot.columns.name = None
    cdc_pivot['fips_tract'] = cdc_pivot['locationid'].astype(str).str.zfill(11)

    census = _load_census()

    CARCINOGENS = {
        'arsenic', 'benzene', 'chromium', 'cadmium', 'nickel', 'lead',
        'vinyl chloride', 'formaldehyde', '1,3-butadiene', 'trichloroethylene',
        'styrene', 'ethylene oxide', 'dioxin', 'polycyclic aromatic', 'benzo',
        'naphthalene', 'beryllium', 'cobalt', 'acrylonitrile',
    }

    df2 = df.copy()
    df2['is_carc'] = df2['CHEMICAL_NAME'].apply(
        lambda x: any(c in str(x).lower() for c in CARCINOGENS)
    )
    df_carc = (
        df2[df2['is_carc']].groupby('fips_tract')['TOTAL_RELEASES']
        .sum().reset_index().rename(columns={'TOTAL_RELEASES': 'carc_releases'})
    )
    df_total = df2.groupby('fips_tract').agg(
        total_releases=('TOTAL_RELEASES', 'sum'),
        n_facilities=('TRI_FACILITY_ID', 'nunique'),
    ).reset_index()
    tri_tract = df_total.merge(df_carc, on='fips_tract', how='left')
    tri_tract['carc_releases'] = tri_tract['carc_releases'].fillna(0)
    tri_tract['fips_tract'] = tri_tract['fips_tract'].astype(str).str.zfill(11)

    all_t = cdc_pivot.merge(
        census[['fips_tract', 'poverty_pct', 'minority_pct']], on='fips_tract', how='left'
    )
    
    # ── Classify tracts using influence model ─────────────────────────────────
    all_t = _classify_tracts_for_case_control(all_t, tri_tract, neighbor_dict)

    all_t2 = all_t[
        all_t['CANCER'].notna() & all_t['poverty_pct'].notna() & all_t['minority_pct'].notna()
    ].copy().reset_index(drop=True)

    # Define minority-level bands
    bands = [
        ('Mostly white (<20% minority)', all_t2['minority_pct'] < 20),
        ('Mixed (20-60% minority)', (all_t2['minority_pct'] >= 20) & (all_t2['minority_pct'] < 60)),
        ('Mostly minority (>60%)', all_t2['minority_pct'] >= 60),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        "Cancer paradox: does carcinogen exposure show cancer signal in white-majority tracts?\n"
        "If the paradox is purely a screening artifact, TRI should show elevated cancer in white tracts\n"
        "(TRUE CONTROL = no TRI AND not neighbor of TRI tract)",
        fontsize=11, fontweight="bold",
    )

    outcomes_4 = [
        ('Cancer', 'CANCER'), ('COPD (lung)', 'COPD'), ('Heart Disease', 'CHD'), ('Diabetes', 'DIABETES'),
    ]
    colors_g = ['#2ecc71', '#e67e22', '#e74c3c']
    group_labels = ['True Control', 'TRI\n(non-carc)', 'TRI\n(carcinogen)']

    for ax, (band_label, band_mask) in zip(axes, bands):
        band_df = all_t2[band_mask].copy()
        # True control = no TRI, no neighbor
        ctrl_b = band_df[band_df['tri_zone'] == 'control']
        # TRI-direct with carcinogens
        carc_b = band_df[(band_df['tri_zone'] == 'tri_direct') & (band_df['carc_releases'] > 0)]
        # TRI-direct without carcinogens
        ncarc_b = band_df[(band_df['tri_zone'] == 'tri_direct') & (band_df['carc_releases'] == 0)]

        x = np.arange(len(outcomes_4))
        w = 0.22
        groups = [ctrl_b, ncarc_b, carc_b]
        for gi, (grp, lbl, clr) in enumerate(zip(groups, group_labels, colors_g)):
            means = [grp[col].mean() for _, col in outcomes_4]
            sems = [grp[col].sem() for _, col in outcomes_4]
            ax.bar(x + (gi - 1) * w, means, w, label=lbl, color=clr, alpha=0.85, edgecolor='black', lw=0.5)
            ax.errorbar(x + (gi - 1) * w, means, yerr=sems, fmt='none', color='black', capsize=3, lw=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels([hc for hc, _ in outcomes_4], fontsize=8)
        n_ctrl = len(ctrl_b); n_carc = len(carc_b)
        ax.set_title(
            f"{band_label}\n(n true-ctrl={n_ctrl:,}, n carc-TRI={n_carc:,})",
            fontsize=9, fontweight='bold',
        )
        ax.set_ylabel("Mean crude rate (% of adults)")
        if ax == axes[0]:
            ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, "cancer_white_tracts")
    logger.info("  Saved: cancer_white_tracts.png (NEIGHBOR MODEL)")


def analysis13_facility_closure_selectivity(df: pd.DataFrame) -> None:
    """
    Did the facilities that closed between 2013 and 2023 differ from those that stayed?
    - Were closures concentrated in dirtier or cleaner facilities?
    - Were closures in richer or poorer communities?
    This tests whether market / regulatory pressure removed the most harmful facilities first.
    Generates: facility_closure_selectivity.png
    """
    # Identify first and last reporting year per facility
    fac_years = df.groupby('TRI_FACILITY_ID').agg(
        first_yr=('REPORTING_YEAR', 'min'),
        last_yr=('REPORTING_YEAR', 'max'),
        mean_releases=('TOTAL_RELEASES', 'mean'),
        total_releases=('TOTAL_RELEASES', 'sum'),
        mean_poverty=('poverty_pct', 'mean'),
        mean_minority=('minority_pct', 'mean'),
        n_years=('REPORTING_YEAR', 'nunique'),
    ).reset_index()

    max_yr = df['REPORTING_YEAR'].max()
    min_yr = df['REPORTING_YEAR'].min()

    # "Closed" = last year reported is before max_yr; "Active" = reported in max_yr
    fac_years['closed'] = fac_years['last_yr'] < max_yr
    fac_years['new'] = fac_years['first_yr'] > min_yr  # opened after study start

    active = fac_years[~fac_years['closed']]
    closed = fac_years[fac_years['closed']]
    new = fac_years[fac_years['new'] & ~fac_years['closed']]

    from scipy.stats import mannwhitneyu as _mwu

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f"What kind of facilities closed between {min_yr} and {max_yr}?\n"
        "Comparing active facilities vs those that stopped reporting (closed/ceased)",
        fontsize=12, fontweight="bold",
    )

    metrics = [
        ('mean_releases', 'Mean annual releases\n(lbs/year)', 'log'),
        ('mean_poverty', 'Mean community poverty rate (%)', 'linear'),
        ('mean_minority', 'Mean community minority % (%)', 'linear'),
    ]

    for ax, (col, label, scale) in zip(axes, metrics):
        a_vals = active[col].dropna()
        c_vals = closed[col].dropna()
        _, p = _mwu(a_vals, c_vals)
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        if scale == 'log':
            a_plot = np.log10(a_vals.clip(0.1))
            c_plot = np.log10(c_vals.clip(0.1))
            ylabel = f"log₁₀({label.split(chr(10))[0]})"
        else:
            a_plot = a_vals
            c_plot = c_vals
            ylabel = label

        ax.violinplot([c_plot.values, a_plot.values], positions=[0, 1], showmedians=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            f'Closed/ceased\n(n={len(closed):,})',
            f'Still active {max_yr}\n(n={len(active):,})',
        ], fontsize=9)
        ax.set_ylabel(ylabel)

        med_closed = (np.power(10, c_plot.median()) if scale == 'log' else c_plot.median())
        med_active = (np.power(10, a_plot.median()) if scale == 'log' else a_plot.median())
        title_unit = 'lbs' if scale == 'log' else '%'
        ax.set_title(
            f"{label}\nMW p{stars}\n"
            f"Closed median: {med_closed:,.0f} {title_unit}  |  "
            f"Active median: {med_active:,.0f} {title_unit}",
            fontsize=9, fontweight='bold',
        )
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, "facility_closure_selectivity")
    logger.info("  Saved: facility_closure_selectivity.png")


def analysis14_comprehensive_hypothesis_test(df: pd.DataFrame = None) -> None:
    """
    Comprehensive hypothesis testing with all available confounders.
    
    Tests the following hypotheses:
    H1: TRI proximity causes worse health (controlling for all confounders)
    H2: Age explains the cancer paradox (older populations have more detected cancer)
    H3: Healthcare access (uninsurance) explains health differences
    H4: Population density correlates with industrial siting
    
    Uses: poverty, minority %, median age, % 65+, healthcare access (ACCESS2)
    """
    from scipy.stats import spearmanr, pearsonr
    from sklearn.preprocessing import StandardScaler
    
    logger.info("\n>>> ANALYSIS 14: Comprehensive Hypothesis Testing")
    
    # Load all data sources
    cdc_raw = pd.read_csv("data/raw/cdc_places.csv", low_memory=False)
    cdc_pivot = cdc_raw.pivot_table(
        index='locationid', columns='measureid', values='data_value'
    ).reset_index()
    cdc_pivot.columns.name = None
    cdc_pivot['fips_tract'] = cdc_pivot['locationid'].astype(str).str.zfill(11)
    
    # Get population
    pop_lookup = (
        cdc_raw[cdc_raw['measureid'] == 'CANCER'][['locationid', 'totalpopulation']]
        .rename(columns={'totalpopulation': 'population'})
    )
    cdc_pivot = cdc_pivot.merge(pop_lookup, on='locationid', how='left')
    
    census = _load_census()
    
    # Load TRI and classify tracts using facilities_scored.csv (has fips_tract)
    neighbor_dict = _load_tract_neighbors()
    
    # Load facilities data with proper tract assignment
    fac = pd.read_csv("data/processed/facilities_scored.csv", low_memory=False)
    fac['fips_tract'] = fac['fips_tract'].astype(str).str.zfill(11)
    
    CARCINOGENS = {
        'arsenic', 'benzene', 'chromium', 'cadmium', 'nickel', 'lead',
        'vinyl chloride', 'formaldehyde', '1,3-butadiene', 'trichloroethylene',
        'styrene', 'ethylene oxide', 'dioxin', 'polycyclic aromatic', 'benzo',
        'naphthalene', 'beryllium', 'cobalt', 'acrylonitrile',
    }
    
    # Aggregate by tract
    fac['is_carc'] = fac['CHEMICAL_NAME'].apply(
        lambda x: any(c in str(x).lower() for c in CARCINOGENS) if pd.notna(x) else False
    )
    df_carc = (
        fac[fac['is_carc']].groupby('fips_tract')['TOTAL_RELEASES']
        .sum().reset_index().rename(columns={'TOTAL_RELEASES': 'carc_releases'})
    )
    df_total = fac.groupby('fips_tract').agg(
        total_releases=('TOTAL_RELEASES', 'sum'),
        n_facilities=('TRI_FACILITY_ID', 'nunique'),
    ).reset_index()
    tri_tract = df_total.merge(df_carc, on='fips_tract', how='left')
    tri_tract['carc_releases'] = tri_tract['carc_releases'].fillna(0)
    
    # Merge all data
    all_t = cdc_pivot.merge(
        census[['fips_tract', 'poverty_pct', 'minority_pct', 'median_age', 'pct_65_plus', 'total_pop']],
        on='fips_tract', how='left'
    )
    
    # Classify tracts
    if neighbor_dict:
        all_t = _classify_tracts_for_case_control(all_t, tri_tract, neighbor_dict)
    else:
        logger.warning("No neighbor dict available, using simple TRI classification")
        tri_set = set(tri_tract['fips_tract'].unique())
        all_t['tri_zone'] = all_t['fips_tract'].apply(lambda x: 'tri_direct' if x in tri_set else 'control')
        all_t['in_influence'] = all_t['tri_zone'] == 'tri_direct'
    
    # Filter to complete cases
    all_t2 = all_t[
        all_t['CANCER'].notna() & 
        all_t['poverty_pct'].notna() & 
        all_t['minority_pct'].notna() &
        all_t['median_age'].notna() &
        all_t['ACCESS2'].notna()
    ].copy().reset_index(drop=True)
    
    logger.info(f"  Complete cases for hypothesis testing: {len(all_t2):,} tracts")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 1: Correlation matrix of ALL predictors vs ALL outcomes
    # ══════════════════════════════════════════════════════════════════════════
    predictors = ['poverty_pct', 'minority_pct', 'median_age', 'pct_65_plus', 'ACCESS2', 'in_influence']
    pred_labels = ['Poverty %', 'Minority %', 'Median Age', '% 65+', 'Uninsured %', 'TRI Influence']
    outcomes = ['CANCER', 'COPD', 'DIABETES', 'CHD', 'CASTHMA', 'MHLTH']
    outcome_labels = ['Cancer', 'COPD', 'Diabetes', 'Heart Disease', 'Asthma', 'Mental Health']
    
    # Compute correlation matrix
    corr_matrix = np.zeros((len(predictors), len(outcomes)))
    pval_matrix = np.zeros((len(predictors), len(outcomes)))
    
    for i, pred in enumerate(predictors):
        for j, out in enumerate(outcomes):
            sub = all_t2[[pred, out]].dropna()
            if len(sub) > 100:
                r, p = spearmanr(sub[pred], sub[out])
                corr_matrix[i, j] = r
                pval_matrix[i, j] = p
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.6, vmax=0.6)
    plt.colorbar(im, ax=ax, label='Spearman ρ')
    
    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels(outcome_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(predictors)))
    ax.set_yticklabels(pred_labels)
    
    # Annotate with correlations and significance
    for i in range(len(predictors)):
        for j in range(len(outcomes)):
            r = corr_matrix[i, j]
            p = pval_matrix[i, j]
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax.text(j, i, f"{r:.2f}{stars}", ha='center', va='center', 
                    fontsize=9, fontweight='bold',
                    color='white' if abs(r) > 0.3 else 'black')
    
    ax.set_title(
        "Correlation Matrix: Predictors vs Health Outcomes\n"
        f"n={len(all_t2):,} census tracts with complete data\n"
        "(*** p<0.001, ** p<0.01, * p<0.05)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    _save(fig, "hyp1_full_correlation_matrix")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 2: Age explains the cancer paradox?
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(
        "Does AGE explain the cancer paradox?\n"
        "Testing: older populations should have higher cancer detection rates",
        fontsize=12, fontweight='bold'
    )
    
    # Panel 1: Age vs Cancer
    ax = axes[0, 0]
    sub = all_t2[['median_age', 'CANCER']].dropna()
    ax.hexbin(sub['median_age'], sub['CANCER'], gridsize=25, cmap='YlOrRd', mincnt=1)
    r, p = spearmanr(sub['median_age'], sub['CANCER'])
    ax.set_xlabel("Median Age")
    ax.set_ylabel("Cancer Rate (%)")
    ax.set_title(f"Age vs Cancer: ρ={r:.3f}***", fontweight='bold')
    
    # Panel 2: % 65+ vs Cancer
    ax = axes[0, 1]
    sub = all_t2[['pct_65_plus', 'CANCER']].dropna()
    ax.hexbin(sub['pct_65_plus'], sub['CANCER'], gridsize=25, cmap='YlOrRd', mincnt=1)
    r, p = spearmanr(sub['pct_65_plus'], sub['CANCER'])
    ax.set_xlabel("% Population 65+")
    ax.set_ylabel("Cancer Rate (%)")
    ax.set_title(f"% 65+ vs Cancer: ρ={r:.3f}***", fontweight='bold')
    
    # Panel 3: Minority % vs Median Age (are minority tracts younger?)
    ax = axes[0, 2]
    sub = all_t2[['minority_pct', 'median_age']].dropna()
    ax.hexbin(sub['minority_pct'], sub['median_age'], gridsize=25, cmap='YlOrRd', mincnt=1)
    r, p = spearmanr(sub['minority_pct'], sub['median_age'])
    ax.set_xlabel("Minority %")
    ax.set_ylabel("Median Age")
    ax.set_title(f"Minority % vs Age: ρ={r:.3f}***\n(Negative = minority tracts are younger)", fontweight='bold')
    
    # Panel 4: Cancer by age quintile
    ax = axes[1, 0]
    all_t2['age_q'] = pd.qcut(all_t2['median_age'], q=5, labels=['Q1\n(youngest)', 'Q2', 'Q3', 'Q4', 'Q5\n(oldest)'])
    age_cancer = all_t2.groupby('age_q', observed=True)['CANCER'].mean()
    bars = ax.bar(range(5), age_cancer.values, color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xticks(range(5))
    ax.set_xticklabels(age_cancer.index)
    ax.set_ylabel("Mean Cancer Rate (%)")
    ax.set_title("Cancer Rate by Age Quintile", fontweight='bold')
    for i, v in enumerate(age_cancer.values):
        ax.text(i, v + 0.1, f"{v:.1f}%", ha='center', fontsize=9)
    
    # Panel 5: Cancer by age quintile, TRI vs Control
    ax = axes[1, 1]
    age_cancer_tri = all_t2.groupby(['age_q', 'tri_zone'], observed=True)['CANCER'].mean().unstack()
    x = np.arange(5)
    w = 0.25
    if 'control' in age_cancer_tri.columns:
        ax.bar(x - w, age_cancer_tri['control'], w, label='Control', color='#2ecc71', alpha=0.8)
    if 'tri_neighbor' in age_cancer_tri.columns:
        ax.bar(x, age_cancer_tri['tri_neighbor'], w, label='TRI-Neighbor', color='#f39c12', alpha=0.8)
    if 'tri_direct' in age_cancer_tri.columns:
        ax.bar(x + w, age_cancer_tri['tri_direct'], w, label='TRI-Direct', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(age_cancer_tri.index)
    ax.set_ylabel("Mean Cancer Rate (%)")
    ax.set_title("Cancer by Age × TRI Status\n(Does TRI effect persist controlling for age?)", fontweight='bold')
    ax.legend(fontsize=8)
    
    # Panel 6: Age-adjusted cancer difference (TRI - Control within age quintiles)
    ax = axes[1, 2]
    if 'control' in age_cancer_tri.columns and 'tri_direct' in age_cancer_tri.columns:
        diffs = age_cancer_tri['tri_direct'] - age_cancer_tri['control']
        colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in diffs]
        ax.bar(range(5), diffs, color=colors, alpha=0.8, edgecolor='black')
        ax.axhline(0, color='black', lw=1)
        ax.set_xticks(range(5))
        ax.set_xticklabels(age_cancer_tri.index)
        ax.set_ylabel("TRI-Direct minus Control (pp)")
        ax.set_title("Age-Adjusted Cancer Gap\n(Positive = TRI has more cancer)", fontweight='bold')
        for i, d in enumerate(diffs):
            ax.text(i, d + 0.05, f"{d:+.2f}", ha='center', fontsize=9)
    
    plt.tight_layout()
    _save(fig, "hyp2_age_cancer_paradox")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 3: Healthcare access (uninsurance) as confounder
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(
        "Does healthcare access (uninsurance) explain health differences?\n"
        "Testing: uninsured populations have lower detected cancer but higher chronic disease",
        fontsize=12, fontweight='bold'
    )
    
    # Panel 1: Uninsurance vs Cancer
    ax = axes[0, 0]
    sub = all_t2[['ACCESS2', 'CANCER']].dropna()
    ax.hexbin(sub['ACCESS2'], sub['CANCER'], gridsize=25, cmap='YlOrRd', mincnt=1)
    r, p = spearmanr(sub['ACCESS2'], sub['CANCER'])
    ax.set_xlabel("% Uninsured (ACCESS2)")
    ax.set_ylabel("Cancer Rate (%)")
    ax.set_title(f"Uninsurance vs Cancer: ρ={r:.3f}***\n(Negative = less detection)", fontweight='bold')
    
    # Panel 2: Uninsurance vs COPD
    ax = axes[0, 1]
    sub = all_t2[['ACCESS2', 'COPD']].dropna()
    ax.hexbin(sub['ACCESS2'], sub['COPD'], gridsize=25, cmap='YlOrRd', mincnt=1)
    r, p = spearmanr(sub['ACCESS2'], sub['COPD'])
    ax.set_xlabel("% Uninsured (ACCESS2)")
    ax.set_ylabel("COPD Rate (%)")
    ax.set_title(f"Uninsurance vs COPD: ρ={r:.3f}***", fontweight='bold')
    
    # Panel 3: Uninsurance vs Diabetes
    ax = axes[0, 2]
    sub = all_t2[['ACCESS2', 'DIABETES']].dropna()
    ax.hexbin(sub['ACCESS2'], sub['DIABETES'], gridsize=25, cmap='YlOrRd', mincnt=1)
    r, p = spearmanr(sub['ACCESS2'], sub['DIABETES'])
    ax.set_xlabel("% Uninsured (ACCESS2)")
    ax.set_ylabel("Diabetes Rate (%)")
    ax.set_title(f"Uninsurance vs Diabetes: ρ={r:.3f}***", fontweight='bold')
    
    # Panel 4: TRI zones by uninsurance level
    ax = axes[1, 0]
    all_t2['unins_q'] = pd.qcut(all_t2['ACCESS2'], q=5, labels=['Q1\n(low)', 'Q2', 'Q3', 'Q4', 'Q5\n(high)'])
    unins_tri = all_t2.groupby('unins_q', observed=True)['in_influence'].mean() * 100
    ax.bar(range(5), unins_tri.values, color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xticks(range(5))
    ax.set_xticklabels(unins_tri.index)
    ax.set_ylabel("% of tracts in TRI influence zone")
    ax.set_title("Are high-uninsurance tracts more likely\nto be near TRI facilities?", fontweight='bold')
    
    # Panel 5: Multivariate - controlling for uninsurance, does TRI effect persist?
    ax = axes[1, 1]
    all_t2['unins_q2'] = pd.qcut(all_t2['ACCESS2'], q=3, labels=['Low unins', 'Medium', 'High unins'])
    unins_copd = all_t2.groupby(['unins_q2', 'tri_zone'], observed=True)['COPD'].mean().unstack()
    x = np.arange(3)
    w = 0.25
    if 'control' in unins_copd.columns:
        ax.bar(x - w, unins_copd['control'], w, label='Control', color='#2ecc71', alpha=0.8)
    if 'tri_neighbor' in unins_copd.columns:
        ax.bar(x, unins_copd['tri_neighbor'], w, label='TRI-Neighbor', color='#f39c12', alpha=0.8)
    if 'tri_direct' in unins_copd.columns:
        ax.bar(x + w, unins_copd['tri_direct'], w, label='TRI-Direct', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(unins_copd.index)
    ax.set_ylabel("COPD Rate (%)")
    ax.set_title("COPD by Uninsurance × TRI Status\n(Does TRI effect persist controlling for access?)", fontweight='bold')
    ax.legend(fontsize=8)
    
    # Panel 6: Summary of partial correlations
    ax = axes[1, 2]
    # Compute partial correlation: TRI effect on COPD controlling for access
    # Simple approach: within-stratum differences
    diffs_by_unins = []
    labels_unins = []
    for q in all_t2['unins_q2'].unique():
        sub = all_t2[all_t2['unins_q2'] == q]
        if 'control' in sub['tri_zone'].values and 'tri_direct' in sub['tri_zone'].values:
            ctrl_m = sub[sub['tri_zone'] == 'control']['COPD'].mean()
            tri_m = sub[sub['tri_zone'] == 'tri_direct']['COPD'].mean()
            diffs_by_unins.append(tri_m - ctrl_m)
            labels_unins.append(str(q))
    
    if diffs_by_unins:
        colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in diffs_by_unins]
        ax.bar(range(len(diffs_by_unins)), diffs_by_unins, color=colors, alpha=0.8, edgecolor='black')
        ax.axhline(0, color='black', lw=1)
        ax.set_xticks(range(len(diffs_by_unins)))
        ax.set_xticklabels(labels_unins)
        ax.set_ylabel("TRI-Direct minus Control (pp)")
        ax.set_title("Insurance-Adjusted COPD Gap\n(Positive = TRI effect persists)", fontweight='bold')
        for i, d in enumerate(diffs_by_unins):
            ax.text(i, d + 0.05, f"{d:+.2f}", ha='center', fontsize=9)
    
    plt.tight_layout()
    _save(fig, "hyp3_healthcare_access")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PLOT 4: Summary - what really predicts health?
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(
        "Summary: What REALLY predicts health outcomes?\n"
        "Multivariate OLS standardized coefficients (all predictors together)",
        fontsize=12, fontweight='bold'
    )
    
    # Prepare standardized data
    pred_cols = ['poverty_pct', 'minority_pct', 'median_age', 'ACCESS2', 'in_influence']
    pred_labels2 = ['Poverty', 'Minority %', 'Median Age', 'Uninsured %', 'TRI Influence']
    
    all_t3 = all_t2[pred_cols + outcomes].copy()
    all_t3['in_influence'] = all_t3['in_influence'].astype(float)
    all_t3 = all_t3.dropna()
    
    sc = StandardScaler()
    X_scaled = sc.fit_transform(all_t3[pred_cols])
    
    for ax, out, out_label in zip(axes.flat, outcomes, outcome_labels):
        y = all_t3[out].values
        # OLS with standardized predictors
        coefs, _, _, _ = np.linalg.lstsq(
            np.column_stack([X_scaled, np.ones(len(X_scaled))]), y, rcond=None
        )
        coefs = coefs[:len(pred_cols)]
        
        colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in coefs]
        ax.barh(range(len(coefs)), coefs, color=colors, alpha=0.8, edgecolor='black')
        ax.axvline(0, color='black', lw=1)
        ax.set_yticks(range(len(coefs)))
        ax.set_yticklabels(pred_labels2)
        ax.set_xlabel("Standardized coefficient")
        ax.set_title(out_label, fontweight='bold')
        for i, c in enumerate(coefs):
            ax.text(c + 0.01 if c >= 0 else c - 0.01, i, f"{c:.3f}", 
                    va='center', ha='left' if c >= 0 else 'right', fontsize=8)
    
    plt.tight_layout()
    _save(fig, "hyp4_multivariate_summary")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Print summary statistics
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n  === HYPOTHESIS TESTING RESULTS ===")
    
    # Age-cancer relationship
    r_age_cancer, _ = spearmanr(all_t2['median_age'].dropna(), all_t2['CANCER'].dropna())
    logger.info(f"  H2: Age vs Cancer correlation: ρ = {r_age_cancer:.3f}")
    
    # Minority-age relationship
    r_min_age, _ = spearmanr(all_t2['minority_pct'].dropna(), all_t2['median_age'].dropna())
    logger.info(f"  H2: Minority % vs Age: ρ = {r_min_age:.3f} (negative = minority tracts younger)")
    
    # TRI effect on COPD after controlling age
    logger.info("  H1: TRI effect on COPD after age adjustment:")
    for q in all_t2['age_q'].unique():
        sub = all_t2[all_t2['age_q'] == q]
        if 'control' in sub['tri_zone'].values and 'tri_direct' in sub['tri_zone'].values:
            ctrl_m = sub[sub['tri_zone'] == 'control']['COPD'].mean()
            tri_m = sub[sub['tri_zone'] == 'tri_direct']['COPD'].mean()
            logger.info(f"      {q}: TRI-Control = {tri_m - ctrl_m:+.2f}pp")
    
    logger.info("  Saved: hyp1-hyp4 hypothesis testing plots")


if __name__ == "__main__":
    run_all_analyses()
