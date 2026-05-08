"""
New Hypothesis Testing Module: Novel Research Questions
========================================================
This module tests hypotheses NOT covered in the original research.py:

H9:  Age structure explains the cancer paradox - younger populations in TRI areas have 
     fewer cancer diagnoses (cancer is age-related), masking the true pollution effect.
     
H10: Temporal lag - communities with facilities that STARTED reporting earlier have 
     worse health outcomes (cumulative historical exposure).
     
H11: Facility clustering creates synergistic burden - tracts with 3+ facilities show 
     more-than-additive health effects.
     
H12: Industry sector (SIC/NAICS) predicts health pathway specificity - chemicals plants
     → respiratory; metal processing → neurological/developmental.
     
H13: Urban vs rural TRI impacts differ - rural facilities may have greater per-capita 
     impact due to less dilution and fewer healthcare resources.
     
H14: Carcinogen exposure timing matters - facilities that INCREASED carcinogen releases
     over time show different health patterns than those that decreased.

H15: Healthcare access moderates pollution-health link - the health gap between TRI and 
     control tracts is LARGER in areas with poor healthcare access.
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, kruskal, ttest_ind

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

OUT = Path("output/research")
OUT.mkdir(parents=True, exist_ok=True)

HEALTH_COLS = ["cancer_crude", "asthma_crude", "chd_crude", "copd_crude", 
               "diabetes_crude", "mental_health_crude"]
HEALTH_LABELS = {
    "cancer_crude": "Cancer Rate (%)",
    "asthma_crude": "Asthma Rate (%)",
    "chd_crude": "Heart Disease Rate (%)",
    "copd_crude": "COPD Rate (%)",
    "diabetes_crude": "Diabetes Rate (%)",
    "mental_health_crude": "Poor Mental Health (%)",
}

def _save(fig, name, tight=True):
    path = OUT / f"{name}.png"
    if tight:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def _load_all_data():
    """Load and merge all data sources with age metrics."""
    # Load facilities
    fac = pd.read_csv("data/processed/facilities_scored.csv", low_memory=False)
    fac["has_health"] = fac["cancer_crude"].notna()
    
    # Load census with age data
    census = pd.read_csv("data/raw/census_acs.csv", low_memory=False)
    census['fips_tract'] = (
        census['state'].astype(str).str.zfill(2) +
        census['county'].astype(str).str.zfill(3) +
        census['tract'].astype(str).str.zfill(6)
    )
    
    # Compute age metrics
    census['median_age'] = pd.to_numeric(census['B01002_001E'], errors='coerce')
    census['total_pop'] = pd.to_numeric(census['B01001_001E'], errors='coerce')
    
    # 65+ population (males + females)
    male_65_cols = ['B01001_020E', 'B01001_021E', 'B01001_022E', 
                    'B01001_023E', 'B01001_024E', 'B01001_025E']
    female_65_cols = ['B01001_044E', 'B01001_045E', 'B01001_046E',
                      'B01001_047E', 'B01001_048E', 'B01001_049E']
    
    for c in male_65_cols + female_65_cols:
        if c in census.columns:
            census[c] = pd.to_numeric(census[c], errors='coerce').fillna(0)
    
    census['pop_65_plus'] = (
        census[male_65_cols].sum(axis=1) + 
        census[female_65_cols].sum(axis=1)
    )
    census['pct_65_plus'] = (
        census['pop_65_plus'] / census['total_pop'].clip(1) * 100
    )
    
    # Merge age into facilities
    age_cols = ['fips_tract', 'median_age', 'pct_65_plus', 'total_pop']
    # Ensure fips_tract types match
    fac['fips_tract'] = fac['fips_tract'].astype(str).str.zfill(11)
    census['fips_tract'] = census['fips_tract'].astype(str).str.zfill(11)
    fac = fac.merge(census[age_cols], on='fips_tract', how='left')
    
    # Load TRI raw for industry info
    tri_raw = pd.read_csv("data/raw/tri_facilities.csv", low_memory=False)
    tri_raw.columns = tri_raw.columns.str.upper().str.strip()
    
    return fac, census, tri_raw


def _load_tract_classification():
    """Load tract neighbor data for case-control."""
    import json
    p = Path("data/processed/tract_neighbors.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# H9: AGE STRUCTURE EXPLAINS CANCER PARADOX
# ═══════════════════════════════════════════════════════════════════════════════

def hypothesis_9_age_cancer(fac, census):
    """
    H9: Age structure explains the cancer paradox.
    
    Cancer is strongly age-related. If TRI tracts have younger populations
    (industrial workers, families), this could explain lower cancer rates
    even if pollution increases cancer risk.
    """
    logger.info("Testing H9: Age structure and cancer paradox...")
    
    # Get tract-level aggregates
    tract_data = fac.groupby('fips_tract').agg({
        'cancer_crude': 'mean',
        'asthma_crude': 'mean',
        'copd_crude': 'mean',
        'diabetes_crude': 'mean',
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
        'TOTAL_RELEASES': 'sum',
        'median_age': 'first',
        'pct_65_plus': 'first',
    }).reset_index()
    
    tract_data = tract_data.dropna(subset=['cancer_crude', 'median_age'])
    tract_data['has_tri'] = tract_data['TOTAL_RELEASES'] > 0
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    # Panel A: Age distribution TRI vs non-TRI tracts
    ax = axes[0, 0]
    tri_ages = tract_data[tract_data['has_tri']]['median_age'].dropna()
    control_ages = tract_data[~tract_data['has_tri']]['median_age'].dropna()
    
    ax.hist(control_ages, bins=30, alpha=0.6, label=f'Non-TRI (n={len(control_ages):,})', 
            color='#3498db', density=True)
    ax.hist(tri_ages, bins=30, alpha=0.6, label=f'TRI tracts (n={len(tri_ages):,})', 
            color='#e74c3c', density=True)
    
    ax.axvline(control_ages.mean(), color='#2980b9', linestyle='--', linewidth=2,
               label=f'Control mean: {control_ages.mean():.1f}')
    ax.axvline(tri_ages.mean(), color='#c0392b', linestyle='--', linewidth=2,
               label=f'TRI mean: {tri_ages.mean():.1f}')
    
    stat, p = mannwhitneyu(tri_ages, control_ages)
    ax.set_title(f"A. Median Age Distribution\nMann-Whitney p={p:.2e}", fontweight='bold')
    ax.set_xlabel("Median Age (years)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    
    # Panel B: 65+ population comparison
    ax = axes[0, 1]
    tri_65 = tract_data[tract_data['has_tri']]['pct_65_plus'].dropna()
    control_65 = tract_data[~tract_data['has_tri']]['pct_65_plus'].dropna()
    
    data_box = [control_65, tri_65]
    bp = ax.boxplot(data_box, labels=['Non-TRI', 'TRI'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    stat, p = mannwhitneyu(tri_65, control_65)
    ax.set_title(f"B. % Population 65+\nMann-Whitney p={p:.2e}", fontweight='bold')
    ax.set_ylabel("% of Population Age 65+")
    
    diff = control_65.mean() - tri_65.mean()
    ax.text(0.5, 0.95, f"Difference: {diff:.1f}pp\n(TRI tracts younger)",
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # Panel C: Age vs Cancer (overall)
    ax = axes[0, 2]
    sample = tract_data.sample(min(3000, len(tract_data)), random_state=42)
    scatter = ax.scatter(sample['median_age'], sample['cancer_crude'],
                        c=sample['has_tri'].map({True: '#e74c3c', False: '#3498db'}),
                        alpha=0.3, s=10, rasterized=True)
    
    r, p = spearmanr(tract_data['median_age'], tract_data['cancer_crude'])
    ax.set_title(f"C. Age vs Cancer Rate\nSpearman ρ={r:.3f}, p={p:.2e}", fontweight='bold')
    ax.set_xlabel("Median Age (years)")
    ax.set_ylabel("Cancer Rate (%)")
    
    # Trend line
    z = np.polyfit(sample['median_age'], sample['cancer_crude'], 1)
    xr = np.linspace(sample['median_age'].min(), sample['median_age'].max(), 100)
    ax.plot(xr, np.poly1d(z)(xr), 'k--', lw=2, alpha=0.7)
    
    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color='#3498db', label='Non-TRI tract'),
        Patch(color='#e74c3c', label='TRI tract'),
    ]
    ax.legend(handles=legend_handles, fontsize=8)
    
    # Panel D: Cancer rates by age quintile, split by TRI
    ax = axes[1, 0]
    tract_data['age_quintile'] = pd.qcut(tract_data['median_age'], 5, 
                                          labels=['Q1\nYoungest', 'Q2', 'Q3', 'Q4', 'Q5\nOldest'])
    
    grouped = tract_data.groupby(['age_quintile', 'has_tri'], observed=True)['cancer_crude'].mean().unstack()
    x = np.arange(5)
    width = 0.35
    
    ax.bar(x - width/2, grouped[False], width, label='Non-TRI', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, grouped[True], width, label='TRI', color='#e74c3c', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.set_xlabel("Median Age Quintile")
    ax.set_ylabel("Mean Cancer Rate (%)")
    ax.set_title("D. Cancer Rate by Age Quintile\n(TRI vs Non-TRI)", fontweight='bold')
    ax.legend()
    
    # Panel E: Age-adjusted cancer comparison
    ax = axes[1, 1]
    # Residualize cancer on age
    from scipy.stats import linregress
    valid = tract_data.dropna(subset=['median_age', 'cancer_crude'])
    slope, intercept, _, _, _ = linregress(valid['median_age'], valid['cancer_crude'])
    valid['cancer_age_adj'] = valid['cancer_crude'] - (slope * valid['median_age'] + intercept)
    
    tri_adj = valid[valid['has_tri']]['cancer_age_adj']
    control_adj = valid[~valid['has_tri']]['cancer_age_adj']
    
    data_box = [control_adj, tri_adj]
    bp = ax.boxplot(data_box, labels=['Non-TRI', 'TRI'], patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    stat, p = mannwhitneyu(tri_adj, control_adj)
    ax.set_title(f"E. Age-Adjusted Cancer Residuals\nMann-Whitney p={p:.2e}", fontweight='bold')
    ax.set_ylabel("Cancer Rate (age-adjusted)")
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    diff_adj = tri_adj.mean() - control_adj.mean()
    direction = "higher" if diff_adj > 0 else "lower"
    ax.text(0.5, 0.95, f"After age adjustment:\nTRI tracts {direction} ({diff_adj:+.3f}pp)",
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', fc='yellow' if diff_adj > 0 else 'lightgreen', alpha=0.8))
    
    # Panel F: Summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Summary stats
    summary_data = [
        ['Metric', 'Non-TRI Tracts', 'TRI Tracts', 'Difference'],
        ['Mean Median Age', f'{control_ages.mean():.1f}', f'{tri_ages.mean():.1f}', 
         f'{tri_ages.mean() - control_ages.mean():.1f}'],
        ['Mean % 65+', f'{control_65.mean():.1f}%', f'{tri_65.mean():.1f}%',
         f'{tri_65.mean() - control_65.mean():.1f}pp'],
        ['Raw Cancer Rate', f'{tract_data[~tract_data["has_tri"]]["cancer_crude"].mean():.2f}%',
         f'{tract_data[tract_data["has_tri"]]["cancer_crude"].mean():.2f}%', 
         f'{tract_data[tract_data["has_tri"]]["cancer_crude"].mean() - tract_data[~tract_data["has_tri"]]["cancer_crude"].mean():.2f}pp'],
        ['Age-Adj Cancer', f'{control_adj.mean():.3f}', f'{tri_adj.mean():.3f}',
         f'{diff_adj:.3f}pp'],
    ]
    
    table = ax.table(cellText=summary_data, loc='center', cellLoc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title("F. Summary: Age Effect on Cancer Paradox", fontweight='bold', pad=20)
    
    fig.suptitle("HYPOTHESIS 9: Does Age Structure Explain the Cancer Paradox?\n"
                 "TRI tracts have younger populations — after age adjustment, TRI effect may emerge",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, "h9_age_cancer_paradox")
    
    return {
        'mean_age_diff': tri_ages.mean() - control_ages.mean(),
        'pct_65_diff': tri_65.mean() - control_65.mean(),
        'raw_cancer_diff': tract_data[tract_data["has_tri"]]["cancer_crude"].mean() - tract_data[~tract_data["has_tri"]]["cancer_crude"].mean(),
        'age_adj_cancer_diff': diff_adj,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# H10: TEMPORAL LAG - LONGER EXPOSURE = WORSE OUTCOMES
# ═══════════════════════════════════════════════════════════════════════════════

def hypothesis_10_temporal_lag(fac):
    """
    H10: Facilities that started reporting earlier create worse health outcomes.
    
    This tests cumulative exposure - communities with longer industrial history
    should show worse health even if current emissions are similar.
    """
    logger.info("Testing H10: Temporal lag and cumulative exposure...")
    
    # Calculate facility history per tract
    tract_history = fac.groupby('fips_tract').agg({
        'REPORTING_YEAR': ['min', 'max', 'nunique'],
        'TRI_FACILITY_ID': 'nunique',
        'TOTAL_RELEASES': 'sum',
        'cancer_crude': 'mean',
        'asthma_crude': 'mean',
        'copd_crude': 'mean',
        'diabetes_crude': 'mean',
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
    })
    tract_history.columns = ['first_year', 'last_year', 'years_active', 'n_facilities',
                              'total_releases', 'cancer', 'asthma', 'copd', 'diabetes',
                              'poverty', 'minority']
    tract_history = tract_history.reset_index()
    tract_history['exposure_years'] = 2024 - tract_history['first_year']  # years since first facility
    tract_history = tract_history.dropna(subset=['cancer', 'copd'])
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    # Panel A: Distribution of exposure years
    ax = axes[0, 0]
    ax.hist(tract_history['exposure_years'], bins=11, color='#e74c3c', 
            edgecolor='black', alpha=0.7)
    ax.set_xlabel("Years Since First TRI Report (2024 - first_year)")
    ax.set_ylabel("Number of Tracts")
    ax.set_title("A. Distribution of Exposure Duration", fontweight='bold')
    ax.axvline(tract_history['exposure_years'].mean(), color='black', linestyle='--',
               label=f"Mean: {tract_history['exposure_years'].mean():.1f} years")
    ax.legend()
    
    # Panel B-E: Health outcomes vs exposure years
    outcomes = [('copd', 'COPD Rate (%)', '#e74c3c'),
                ('diabetes', 'Diabetes Rate (%)', '#f39c12'),
                ('asthma', 'Asthma Rate (%)', '#3498db'),
                ('cancer', 'Cancer Rate (%)', '#9b59b6')]
    
    for idx, (outcome, label, color) in enumerate(outcomes):
        ax = axes.flatten()[idx + 1]
        
        # Group by exposure years
        tract_history['exp_bin'] = pd.cut(tract_history['exposure_years'], 
                                          bins=[0, 5, 8, 11, 12],
                                          labels=['1-5 years', '6-8 years', '9-11 years', '12+ years'])
        grouped = tract_history.groupby('exp_bin', observed=True)[outcome].agg(['mean', 'sem', 'count'])
        
        x = np.arange(len(grouped))
        bars = ax.bar(x, grouped['mean'], color=color, alpha=0.8, edgecolor='black')
        ax.errorbar(x, grouped['mean'], yerr=1.96*grouped['sem'], 
                   fmt='none', color='black', capsize=4)
        
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, rotation=15)
        ax.set_ylabel(label)
        
        # Correlation
        r, p = spearmanr(tract_history['exposure_years'], tract_history[outcome])
        ax.set_title(f"{'BCDE'[idx]}. {label.replace(' (%)', '')}\nρ={r:.3f}, p={p:.2e}",
                    fontweight='bold')
        
        # Add trend line annotation
        direction = "↑" if r > 0 else "↓" if r < 0 else "→"
        ax.text(0.95, 0.95, f"{direction} Longer exposure", 
               transform=ax.transAxes, ha='right', va='top',
               fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # Panel F: Controlling for total releases
    ax = axes[1, 2]
    
    # Residualize COPD on total releases, then correlate with exposure years
    from scipy.stats import linregress
    valid = tract_history.dropna(subset=['copd', 'total_releases', 'exposure_years'])
    valid['log_releases'] = np.log10(valid['total_releases'].clip(1))
    
    slope, intercept, _, _, _ = linregress(valid['log_releases'], valid['copd'])
    valid['copd_adj'] = valid['copd'] - (slope * valid['log_releases'] + intercept)
    
    # Scatter: exposure years vs adjusted COPD
    ax.scatter(valid['exposure_years'], valid['copd_adj'], 
              alpha=0.3, s=10, c='#e74c3c', rasterized=True)
    
    r, p = spearmanr(valid['exposure_years'], valid['copd_adj'])
    ax.set_xlabel("Years Since First TRI Report")
    ax.set_ylabel("COPD Rate (release-adjusted)")
    ax.set_title(f"F. Release-Adjusted COPD vs Exposure Duration\nρ={r:.3f}, p={p:.2e}",
                fontweight='bold')
    
    # Trend line
    z = np.polyfit(valid['exposure_years'], valid['copd_adj'], 1)
    xr = np.linspace(valid['exposure_years'].min(), valid['exposure_years'].max(), 100)
    ax.plot(xr, np.poly1d(z)(xr), 'k--', lw=2)
    
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    fig.suptitle("HYPOTHESIS 10: Does Longer Industrial Exposure Create Worse Health Outcomes?\n"
                 "Testing cumulative burden hypothesis (years since first TRI facility)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, "h10_temporal_lag_exposure")
    
    return tract_history


# ═══════════════════════════════════════════════════════════════════════════════
# H11: FACILITY CLUSTERING CREATES SYNERGISTIC BURDEN
# ═══════════════════════════════════════════════════════════════════════════════

def hypothesis_11_clustering_synergy(fac):
    """
    H11: Multiple facilities create more-than-additive health effects.
    
    Test whether tracts with 3+ facilities show disproportionately worse
    health compared to what we'd expect from linear extrapolation.
    """
    logger.info("Testing H11: Facility clustering synergy...")
    
    tract_agg = fac.groupby('fips_tract').agg({
        'TRI_FACILITY_ID': 'nunique',
        'TOTAL_RELEASES': 'sum',
        'cancer_crude': 'mean',
        'asthma_crude': 'mean',
        'copd_crude': 'mean',
        'diabetes_crude': 'mean',
        'chd_crude': 'mean',
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
    }).reset_index()
    tract_agg.columns = ['fips_tract', 'n_facilities', 'total_releases', 
                          'cancer', 'asthma', 'copd', 'diabetes', 'chd',
                          'poverty', 'minority']
    tract_agg = tract_agg.dropna(subset=['copd', 'diabetes'])
    
    # Create facility count categories
    def fac_category(n):
        if n == 1: return '1 facility'
        elif n == 2: return '2 facilities'
        elif n <= 4: return '3-4 facilities'
        else: return '5+ facilities'
    
    tract_agg['fac_category'] = tract_agg['n_facilities'].apply(fac_category)
    cat_order = ['1 facility', '2 facilities', '3-4 facilities', '5+ facilities']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    # Panel A: Number of tracts by facility count
    ax = axes[0, 0]
    counts = tract_agg['fac_category'].value_counts().reindex(cat_order)
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    ax.bar(cat_order, counts.values, color=colors, edgecolor='black')
    ax.set_ylabel("Number of Census Tracts")
    ax.set_title("A. Tract Distribution by Facility Count", fontweight='bold')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, f'{v:,}', ha='center', fontsize=9)
    
    # Panels B-E: Health outcomes by facility count
    outcomes = [('copd', 'COPD Rate (%)'), ('diabetes', 'Diabetes Rate (%)'),
                ('chd', 'CHD Rate (%)'), ('asthma', 'Asthma Rate (%)')]
    
    for idx, (outcome, label) in enumerate(outcomes):
        ax = axes.flatten()[idx + 1]
        
        grouped = tract_agg.groupby('fac_category', observed=True)[outcome].agg(['mean', 'sem', 'count'])
        grouped = grouped.reindex(cat_order)
        
        x = np.arange(len(cat_order))
        bars = ax.bar(x, grouped['mean'], color=colors, edgecolor='black', alpha=0.8)
        ax.errorbar(x, grouped['mean'], yerr=1.96*grouped['sem'],
                   fmt='none', color='black', capsize=4)
        
        ax.set_xticks(x)
        ax.set_xticklabels(cat_order, rotation=15)
        ax.set_ylabel(label)
        
        # Test for synergy: is the 5+ group higher than expected from linear trend?
        # Expected from linear: extrapolate from 1-facility rate
        base_rate = grouped.loc['1 facility', 'mean']
        rate_2 = grouped.loc['2 facilities', 'mean']
        increment = rate_2 - base_rate  # per-facility increment
        
        expected_5plus = base_rate + 5 * increment
        actual_5plus = grouped.loc['5+ facilities', 'mean']
        synergy = actual_5plus - expected_5plus
        
        # Kruskal-Wallis
        groups = [tract_agg[tract_agg['fac_category'] == c][outcome].dropna() for c in cat_order]
        stat, p = kruskal(*[g for g in groups if len(g) > 5])
        
        ax.set_title(f"{'BCDE'[idx]}. {label.replace(' (%)', '')}\nKW p={p:.2e}",
                    fontweight='bold')
        
        # Annotate synergy
        if synergy > 0:
            ax.text(0.95, 0.95, f"Synergy: +{synergy:.2f}pp\nabove linear", 
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   color='red', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
    
    # Panel F: Per-facility release burden
    ax = axes[1, 2]
    tract_agg['releases_per_fac'] = tract_agg['total_releases'] / tract_agg['n_facilities']
    
    grouped_rel = tract_agg.groupby('fac_category', observed=True)['releases_per_fac'].agg(['mean', 'sem'])
    grouped_rel = grouped_rel.reindex(cat_order)
    
    ax.bar(np.arange(4), grouped_rel['mean']/1000, color='#9b59b6', edgecolor='black', alpha=0.8)
    ax.errorbar(np.arange(4), grouped_rel['mean']/1000, 
               yerr=1.96*grouped_rel['sem']/1000, fmt='none', color='black', capsize=4)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(cat_order, rotation=15)
    ax.set_ylabel("Mean Releases per Facility (thousand lbs)")
    ax.set_title("F. Release Volume per Facility\n(Larger facilities cluster?)", fontweight='bold')
    
    fig.suptitle("HYPOTHESIS 11: Does Facility Clustering Create Synergistic Health Burden?\n"
                 "Testing whether 3+ facilities create more-than-additive effects",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, "h11_facility_clustering_synergy")


# ═══════════════════════════════════════════════════════════════════════════════
# H12: INDUSTRY SECTOR PREDICTS HEALTH PATHWAY
# ═══════════════════════════════════════════════════════════════════════════════

def hypothesis_12_industry_health_pathway(fac, tri_raw):
    """
    H12: Different industry sectors cause different health effects.
    
    Chemical plants → respiratory
    Metal processing → cardiovascular/neurological
    Paper/pulp → respiratory
    """
    logger.info("Testing H12: Industry sector predicts health pathway...")
    
    # Find industry column in TRI raw
    industry_col = None
    for col in tri_raw.columns:
        if 'INDUSTRY' in col or 'NAICS' in col or 'SIC' in col:
            industry_col = col
            break
    
    if industry_col is None:
        # Try primary NAICS code pattern
        for col in tri_raw.columns:
            if 'PRIMARY' in col and 'NAICS' in col:
                industry_col = col
                break
    
    if industry_col is None:
        logger.warning("No industry column found in TRI data, skipping H12")
        return
    
    logger.info(f"Using industry column: {industry_col}")
    
    # Merge industry into facilities
    fac_id_col = [c for c in tri_raw.columns if 'TRI_FACILITY_ID' in c][0]
    industry_lookup = tri_raw[[fac_id_col, industry_col]].drop_duplicates()
    industry_lookup.columns = ['TRI_FACILITY_ID', 'industry_code']
    
    fac_ind = fac.merge(industry_lookup, on='TRI_FACILITY_ID', how='left')
    
    # Define industry categories based on NAICS codes
    def classify_industry(code):
        if pd.isna(code):
            return 'Unknown'
        code = str(code)[:2]  # First 2 digits of NAICS
        industry_map = {
            '11': 'Agriculture',
            '21': 'Mining',
            '22': 'Utilities',
            '23': 'Construction',
            '31': 'Manufacturing (Food/Textile)',
            '32': 'Manufacturing (Chemical/Petroleum)',
            '33': 'Manufacturing (Metal/Machinery)',
            '42': 'Wholesale Trade',
            '44': 'Retail Trade',
            '45': 'Retail Trade',
            '48': 'Transportation',
            '49': 'Transportation',
            '51': 'Information',
            '52': 'Finance',
            '53': 'Real Estate',
            '54': 'Professional Services',
            '55': 'Management',
            '56': 'Admin Services',
            '61': 'Education',
            '62': 'Healthcare',
            '71': 'Arts/Entertainment',
            '72': 'Accommodation/Food',
            '81': 'Other Services',
            '92': 'Public Admin',
        }
        return industry_map.get(code, 'Other')
    
    fac_ind['industry_cat'] = fac_ind['industry_code'].apply(classify_industry)
    
    # Filter to main industry categories with enough data
    hdf = fac_ind[fac_ind['cancer_crude'].notna()].copy()
    industry_counts = hdf['industry_cat'].value_counts()
    main_industries = industry_counts[industry_counts >= 100].index.tolist()
    hdf = hdf[hdf['industry_cat'].isin(main_industries)]
    
    if len(main_industries) < 3:
        logger.warning("Not enough industry categories for meaningful analysis")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    # Panel A: Facility count by industry
    ax = axes[0, 0]
    industry_summary = hdf.groupby('industry_cat').agg({
        'TRI_FACILITY_ID': 'nunique',
        'TOTAL_RELEASES': 'sum',
    }).sort_values('TRI_FACILITY_ID', ascending=False)
    
    y_pos = np.arange(len(industry_summary))
    ax.barh(y_pos, industry_summary['TRI_FACILITY_ID'], color='#3498db', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(industry_summary.index, fontsize=8)
    ax.set_xlabel("Number of Facilities")
    ax.set_title("A. Facility Count by Industry Sector", fontweight='bold')
    ax.invert_yaxis()
    
    # Panels B-E: Health outcomes by industry
    outcomes = [('copd_crude', 'COPD (%)'), ('asthma_crude', 'Asthma (%)'),
                ('chd_crude', 'Heart Disease (%)'), ('diabetes_crude', 'Diabetes (%)')]
    
    for idx, (outcome, label) in enumerate(outcomes):
        ax = axes.flatten()[idx + 1]
        
        grouped = hdf.groupby('industry_cat')[outcome].agg(['mean', 'sem'])
        grouped = grouped.sort_values('mean', ascending=True)
        
        y_pos = np.arange(len(grouped))
        ax.barh(y_pos, grouped['mean'], 
               xerr=1.96*grouped['sem'], capsize=3,
               color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(grouped))),
               alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(grouped.index, fontsize=8)
        ax.set_xlabel(label)
        ax.set_title(f"{'BCDE'[idx]}. {label.replace(' (%)', '')} by Industry", fontweight='bold')
        
        # Kruskal-Wallis
        groups = [hdf[hdf['industry_cat'] == c][outcome].dropna() for c in grouped.index]
        if len([g for g in groups if len(g) > 5]) >= 2:
            stat, p = kruskal(*[g for g in groups if len(g) > 5])
            ax.text(0.95, 0.05, f"KW p={p:.2e}", transform=ax.transAxes, 
                   ha='right', va='bottom', fontsize=8)
    
    # Panel F: Chemical vs Metal manufacturing comparison
    ax = axes[1, 2]
    
    chemical = hdf[hdf['industry_cat'].str.contains('Chemical', na=False)]
    metal = hdf[hdf['industry_cat'].str.contains('Metal', na=False)]
    
    outcomes_compare = ['copd_crude', 'asthma_crude', 'chd_crude', 'diabetes_crude']
    labels_compare = ['COPD', 'Asthma', 'CHD', 'Diabetes']
    
    chem_means = [chemical[o].mean() for o in outcomes_compare]
    metal_means = [metal[o].mean() for o in outcomes_compare]
    
    x = np.arange(len(outcomes_compare))
    width = 0.35
    
    ax.bar(x - width/2, chem_means, width, label='Chemical Manufacturing', color='#e74c3c')
    ax.bar(x + width/2, metal_means, width, label='Metal Manufacturing', color='#3498db')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels_compare)
    ax.set_ylabel("Mean Health Rate (%)")
    ax.set_title("F. Chemical vs Metal Manufacturing\nHealth Profile Comparison", fontweight='bold')
    ax.legend(fontsize=8)
    
    fig.suptitle("HYPOTHESIS 12: Does Industry Sector Predict Specific Health Pathways?\n"
                 "Different industries may cause different disease profiles",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, "h12_industry_health_pathway")


# ═══════════════════════════════════════════════════════════════════════════════
# H13: URBAN VS RURAL INDUSTRIAL IMPACT
# ═══════════════════════════════════════════════════════════════════════════════

def hypothesis_13_urban_rural(fac, census):
    """
    H13: Rural industrial facilities have greater per-capita health impact.
    
    Less pollution dilution, fewer healthcare resources, greater visibility
    of individual facility effects.
    """
    logger.info("Testing H13: Urban vs rural industrial impact...")
    
    # Define urban/rural based on population density proxy
    # Use total population as proxy (tracts with <3000 pop are rural-ish)
    hdf = fac[fac['cancer_crude'].notna()].copy()
    hdf = hdf[hdf['total_pop'].notna()]
    
    # Define categories
    def urbanicity(pop):
        if pop < 2000:
            return 'Rural (<2K)'
        elif pop < 5000:
            return 'Suburban (2-5K)'
        else:
            return 'Urban (5K+)'
    
    hdf['urbanicity'] = hdf['total_pop'].apply(urbanicity)
    urban_order = ['Rural (<2K)', 'Suburban (2-5K)', 'Urban (5K+)']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    # Panel A: Tract distribution by urbanicity
    ax = axes[0, 0]
    urban_counts = hdf['urbanicity'].value_counts().reindex(urban_order)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax.bar(urban_order, urban_counts.values, color=colors, edgecolor='black')
    ax.set_ylabel("Number of TRI Tracts")
    ax.set_title("A. TRI Tract Distribution by Urbanicity", fontweight='bold')
    for i, v in enumerate(urban_counts.values):
        ax.text(i, v + 20, f'{v:,}', ha='center', fontsize=9)
    
    # Panel B: Releases per capita by urbanicity
    ax = axes[0, 1]
    hdf['releases_per_capita'] = hdf['TOTAL_RELEASES'] / hdf['total_pop'].clip(1)
    
    grouped_rpc = hdf.groupby('urbanicity', observed=True)['releases_per_capita'].agg(['mean', 'sem'])
    grouped_rpc = grouped_rpc.reindex(urban_order)
    
    ax.bar(urban_order, grouped_rpc['mean'], color=colors, edgecolor='black', alpha=0.8)
    ax.errorbar(range(3), grouped_rpc['mean'], 
               yerr=1.96*grouped_rpc['sem'], fmt='none', color='black', capsize=4)
    ax.set_ylabel("Releases per Capita (lbs/person)")
    ax.set_title("B. Per-Capita Pollution Burden\n(Higher in rural areas)", fontweight='bold')
    
    # Annotate
    rural_rpc = grouped_rpc.loc['Rural (<2K)', 'mean']
    urban_rpc = grouped_rpc.loc['Urban (5K+)', 'mean']
    ratio = rural_rpc / urban_rpc if urban_rpc > 0 else 0
    ax.text(0.95, 0.95, f"Rural/Urban ratio: {ratio:.1f}x", 
           transform=ax.transAxes, ha='right', va='top', fontsize=10,
           bbox=dict(boxstyle='round', fc='yellow', alpha=0.8))
    
    # Panels C-F: Health outcomes by urbanicity
    outcomes = [('copd_crude', 'COPD Rate (%)'), ('diabetes_crude', 'Diabetes Rate (%)'),
                ('asthma_crude', 'Asthma Rate (%)'), ('chd_crude', 'Heart Disease Rate (%)')]
    
    for idx, (outcome, label) in enumerate(outcomes):
        ax = axes.flatten()[idx + 2]
        
        grouped = hdf.groupby('urbanicity', observed=True)[outcome].agg(['mean', 'sem'])
        grouped = grouped.reindex(urban_order)
        
        ax.bar(urban_order, grouped['mean'], color=colors, edgecolor='black', alpha=0.8)
        ax.errorbar(range(3), grouped['mean'], 
                   yerr=1.96*grouped['sem'], fmt='none', color='black', capsize=4)
        ax.set_ylabel(label)
        
        # Kruskal-Wallis
        groups = [hdf[hdf['urbanicity'] == c][outcome].dropna() for c in urban_order]
        stat, p = kruskal(*[g for g in groups if len(g) > 5])
        ax.set_title(f"{'CDEF'[idx]}. {label.replace(' (%)', '')}\nKW p={p:.2e}",
                    fontweight='bold')
        
        # Annotate direction
        rural_mean = grouped.loc['Rural (<2K)', 'mean']
        urban_mean = grouped.loc['Urban (5K+)', 'mean']
        if rural_mean > urban_mean:
            ax.text(0.95, 0.95, "Rural > Urban", transform=ax.transAxes,
                   ha='right', va='top', fontsize=9, color='green',
                   bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.8))
    
    fig.suptitle("HYPOTHESIS 13: Do Rural Industrial Areas Have Greater Health Impact?\n"
                 "Testing dilution and healthcare access effects",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, "h13_urban_rural_impact")


# ═══════════════════════════════════════════════════════════════════════════════
# H14: CARCINOGEN TRAJECTORY MATTERS
# ═══════════════════════════════════════════════════════════════════════════════

def hypothesis_14_carcinogen_trajectory(fac):
    """
    H14: Facilities that INCREASED carcinogen releases over time show
    different health patterns than those that decreased.
    """
    logger.info("Testing H14: Carcinogen release trajectory...")
    
    # Need carcinogen flag - check existing columns
    if 'IS_CARCINOGEN' not in fac.columns:
        # Try to infer from chemical names
        from pipeline.research import KNOWN_CARCINOGENS
        def is_carcin(name):
            if pd.isna(name):
                return False
            for c in KNOWN_CARCINOGENS:
                if c.lower() in str(name).lower():
                    return True
            return False
        fac['IS_CARCINOGEN'] = fac['CHEMICAL_NAME'].apply(is_carcin)
    
    # Calculate trajectory per facility
    carc_only = fac[fac['IS_CARCINOGEN']].copy()
    
    # Early period (2013-2017) vs late period (2019-2023)
    early = carc_only[carc_only['REPORTING_YEAR'] <= 2017].groupby('TRI_FACILITY_ID').agg({
        'TOTAL_RELEASES': 'mean',
        'fips_tract': 'first',
    })
    early.columns = ['early_releases', 'fips_tract']
    
    late = carc_only[carc_only['REPORTING_YEAR'] >= 2019].groupby('TRI_FACILITY_ID').agg({
        'TOTAL_RELEASES': 'mean',
    })
    late.columns = ['late_releases']
    
    trajectory = early.merge(late, left_index=True, right_index=True, how='inner').reset_index()
    trajectory['pct_change'] = (trajectory['late_releases'] - trajectory['early_releases']) / trajectory['early_releases'].clip(1) * 100
    
    # Classify trajectory
    def classify_traj(pct):
        if pct > 50:
            return 'Increased 50%+'
        elif pct > 10:
            return 'Increased 10-50%'
        elif pct > -10:
            return 'Stable (-10% to +10%)'
        elif pct > -50:
            return 'Decreased 10-50%'
        else:
            return 'Decreased 50%+'
    
    trajectory['trajectory'] = trajectory['pct_change'].apply(classify_traj)
    
    # Merge health data
    health_by_tract = fac.groupby('fips_tract').agg({
        'cancer_crude': 'mean',
        'copd_crude': 'mean',
        'diabetes_crude': 'mean',
        'poverty_pct': 'mean',
    }).reset_index()
    
    trajectory = trajectory.merge(health_by_tract, on='fips_tract', how='left')
    trajectory = trajectory.dropna(subset=['cancer_crude'])
    
    traj_order = ['Decreased 50%+', 'Decreased 10-50%', 'Stable (-10% to +10%)', 
                  'Increased 10-50%', 'Increased 50%+']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Trajectory distribution
    ax = axes[0, 0]
    traj_counts = trajectory['trajectory'].value_counts().reindex(traj_order)
    colors = ['#2ecc71', '#82e0aa', '#f7dc6f', '#f5b041', '#e74c3c']
    ax.bar(range(len(traj_order)), traj_counts.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(traj_order)))
    ax.set_xticklabels(traj_order, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Number of Facilities")
    ax.set_title("A. Carcinogen Release Trajectory\n(2013-2017 vs 2019-2023)", fontweight='bold')
    
    # Panel B: Cancer rates by trajectory
    ax = axes[0, 1]
    grouped = trajectory.groupby('trajectory', observed=True)['cancer_crude'].agg(['mean', 'sem'])
    grouped = grouped.reindex(traj_order)
    
    ax.bar(range(len(traj_order)), grouped['mean'], color=colors, edgecolor='black', alpha=0.8)
    ax.errorbar(range(len(traj_order)), grouped['mean'], 
               yerr=1.96*grouped['sem'], fmt='none', color='black', capsize=4)
    ax.set_xticks(range(len(traj_order)))
    ax.set_xticklabels(traj_order, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Mean Cancer Rate (%)")
    ax.set_title("B. Cancer Rate by Carcinogen Trajectory", fontweight='bold')
    
    # Correlation
    r, p = spearmanr(trajectory['pct_change'], trajectory['cancer_crude'])
    ax.text(0.95, 0.95, f"ρ={r:.3f}, p={p:.3f}", transform=ax.transAxes,
           ha='right', va='top', fontsize=10, 
           bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # Panel C: COPD rates by trajectory
    ax = axes[1, 0]
    grouped_copd = trajectory.groupby('trajectory', observed=True)['copd_crude'].agg(['mean', 'sem'])
    grouped_copd = grouped_copd.reindex(traj_order)
    
    ax.bar(range(len(traj_order)), grouped_copd['mean'], color=colors, edgecolor='black', alpha=0.8)
    ax.errorbar(range(len(traj_order)), grouped_copd['mean'], 
               yerr=1.96*grouped_copd['sem'], fmt='none', color='black', capsize=4)
    ax.set_xticks(range(len(traj_order)))
    ax.set_xticklabels(traj_order, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Mean COPD Rate (%)")
    ax.set_title("C. COPD Rate by Carcinogen Trajectory", fontweight='bold')
    
    r, p = spearmanr(trajectory['pct_change'], trajectory['copd_crude'])
    ax.text(0.95, 0.95, f"ρ={r:.3f}, p={p:.3f}", transform=ax.transAxes,
           ha='right', va='top', fontsize=10,
           bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # Panel D: Summary scatter
    ax = axes[1, 1]
    scatter = ax.scatter(trajectory['pct_change'].clip(-200, 200), 
                        trajectory['cancer_crude'],
                        c=trajectory['poverty_pct'],
                        cmap='RdYlBu_r', alpha=0.5, s=20, rasterized=True)
    plt.colorbar(scatter, ax=ax, label='Poverty %')
    
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("% Change in Carcinogen Releases (2013-17 → 2019-23)")
    ax.set_ylabel("Cancer Rate (%)")
    ax.set_title("D. Carcinogen Change vs Cancer Rate\n(color = poverty)", fontweight='bold')
    ax.set_xlim(-200, 200)
    
    fig.suptitle("HYPOTHESIS 14: Do Carcinogen Trajectories Predict Cancer Rates?\n"
                 "Testing whether increasing/decreasing emissions matter for health",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, "h14_carcinogen_trajectory")


# ═══════════════════════════════════════════════════════════════════════════════
# H15: HEALTHCARE ACCESS MODERATES POLLUTION-HEALTH LINK
# ═══════════════════════════════════════════════════════════════════════════════

def hypothesis_15_healthcare_moderation(fac):
    """
    H15: Healthcare access moderates the pollution-health relationship.
    
    The health gap between TRI and control areas should be LARGER in 
    areas with poor healthcare access.
    """
    logger.info("Testing H15: Healthcare access moderates pollution effect...")
    
    # pct_no_insurance as proxy for healthcare access
    hdf = fac[fac['cancer_crude'].notna() & fac['pct_no_insurance'].notna()].copy()
    
    # Split by insurance status
    hdf['insurance_quintile'] = pd.qcut(hdf['pct_no_insurance'], 5, 
                                         labels=['Q1\nBest Access', 'Q2', 'Q3', 'Q4', 'Q5\nWorst Access'])
    
    # Also need TRI vs non-TRI comparison - use release level
    hdf['high_releases'] = hdf['TOTAL_RELEASES'] > hdf['TOTAL_RELEASES'].median()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    
    # Panel A: Uninsured distribution
    ax = axes[0, 0]
    ax.hist(hdf['pct_no_insurance'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    ax.set_xlabel("% Uninsured Population")
    ax.set_ylabel("Number of Tracts")
    ax.set_title("A. Healthcare Access Distribution\n(% uninsured as proxy)", fontweight='bold')
    ax.axvline(hdf['pct_no_insurance'].median(), color='red', linestyle='--',
               label=f"Median: {hdf['pct_no_insurance'].median():.1f}%")
    ax.legend()
    
    # Panels B-E: Health outcomes stratified by insurance quintile
    outcomes = [('copd_crude', 'COPD'), ('diabetes_crude', 'Diabetes'),
                ('asthma_crude', 'Asthma'), ('cancer_crude', 'Cancer')]
    
    for idx, (outcome, label) in enumerate(outcomes):
        ax = axes.flatten()[idx + 1]
        
        # Calculate mean for each insurance quintile × release level
        grouped = hdf.groupby(['insurance_quintile', 'high_releases'], observed=True)[outcome].mean().unstack()
        
        x = np.arange(5)
        width = 0.35
        
        ax.bar(x - width/2, grouped[False], width, label='Lower releases', color='#3498db', alpha=0.8)
        ax.bar(x + width/2, grouped[True], width, label='Higher releases', color='#e74c3c', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Q1\nBest', 'Q2', 'Q3', 'Q4', 'Q5\nWorst'], fontsize=8)
        ax.set_xlabel("Healthcare Access Quintile")
        ax.set_ylabel(f"{label} Rate (%)")
        ax.set_title(f"{'BCDE'[idx]}. {label} by Insurance Access", fontweight='bold')
        
        if idx == 0:
            ax.legend(fontsize=8)
        
        # Calculate the GAP (high - low releases) at each quintile
        gap = grouped[True] - grouped[False]
        
        # Is the gap larger in Q5 (worst access) vs Q1 (best access)?
        if len(gap) >= 5:
            gap_ratio = gap.iloc[4] / gap.iloc[0] if gap.iloc[0] != 0 else np.inf
            direction = "amplified" if gap_ratio > 1.2 else "reduced" if gap_ratio < 0.8 else "similar"
            ax.text(0.95, 0.95, f"Gap in worst access:\n{direction}", 
                   transform=ax.transAxes, ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', fc='lightyellow' if gap_ratio > 1 else 'lightgreen', alpha=0.8))
    
    # Panel F: Summary - gap size by access quintile
    ax = axes[1, 2]
    
    gaps = {}
    for outcome, label in outcomes:
        grouped = hdf.groupby(['insurance_quintile', 'high_releases'], observed=True)[outcome].mean().unstack()
        gaps[label] = (grouped[True] - grouped[False]).values
    
    gap_df = pd.DataFrame(gaps)
    gap_df.index = ['Q1\nBest', 'Q2', 'Q3', 'Q4', 'Q5\nWorst']
    
    gap_df.plot(kind='bar', ax=ax, width=0.8, alpha=0.8, edgecolor='black')
    ax.set_xlabel("Healthcare Access Quintile")
    ax.set_ylabel("Health Gap (High vs Low Release Tracts, pp)")
    ax.set_title("F. Pollution-Health Gap by Access Level\n(Positive = high releases worse)", fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    plt.xticks(rotation=0)
    
    fig.suptitle("HYPOTHESIS 15: Does Poor Healthcare Access Amplify Pollution's Health Impact?\n"
                 "Testing whether the TRI health gap is larger in underserved areas",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, "h15_healthcare_moderation")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_new_hypotheses():
    """Run all new hypothesis tests."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    
    logger.info("=" * 60)
    logger.info("NEW HYPOTHESIS TESTING MODULE")
    logger.info("=" * 60)
    
    fac, census, tri_raw = _load_all_data()
    logger.info(f"Loaded {len(fac):,} records")
    
    results = {}
    
    logger.info("\n>>> H9: Age Structure and Cancer Paradox")
    results['h9'] = hypothesis_9_age_cancer(fac, census)
    
    logger.info("\n>>> H10: Temporal Lag and Cumulative Exposure")
    hypothesis_10_temporal_lag(fac)
    
    logger.info("\n>>> H11: Facility Clustering Synergy")
    hypothesis_11_clustering_synergy(fac)
    
    logger.info("\n>>> H12: Industry Sector Health Pathways")
    hypothesis_12_industry_health_pathway(fac, tri_raw)
    
    logger.info("\n>>> H13: Urban vs Rural Impact")
    hypothesis_13_urban_rural(fac, census)
    
    logger.info("\n>>> H14: Carcinogen Release Trajectory")
    hypothesis_14_carcinogen_trajectory(fac)
    
    logger.info("\n>>> H15: Healthcare Access Moderation")
    hypothesis_15_healthcare_moderation(fac)
    
    logger.info("\n" + "=" * 60)
    logger.info("New hypothesis testing complete!")
    plots = sorted(OUT.glob("h9*.png")) + sorted(OUT.glob("h10*.png")) + \
            sorted(OUT.glob("h11*.png")) + sorted(OUT.glob("h12*.png")) + \
            sorted(OUT.glob("h13*.png")) + sorted(OUT.glob("h14*.png")) + \
            sorted(OUT.glob("h15*.png"))
    for p in plots:
        logger.info(f"  {p.name}")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    run_new_hypotheses()
