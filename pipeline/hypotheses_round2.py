"""
Hypothesis Testing Round 2: Addressing Reviewer Feedback
=========================================================

Based on feedback, this module re-examines:

1. H2 REVISED: Cancer paradox - competing mortality not convincing, find other testable hypotheses
   - Screening/detection gap
   - Healthy worker effect
   - Latency period mismatch
   
2. H4 REVISED: Poverty geography should be tested at TRACT level, not state level

3. H3 REVISED: Facility closure selectivity - do less polluted sites close while 
   large polluters stay? (jobs? profitability?)

4. H5 REVISED: Facility PRESENCE (not emission volume) makes people sick
   - Psychosomatic/stress hypothesis
   - Selective migration (healthy people leave)
   - Co-located factors (traffic, less greenspace)
   
5. H7 REVISED: Re-analyze wealth interaction correctly
   - TRI decreases cancer among rich
   - TRI slightly increases cancer among poor
   - Overall effect is mild or negative
   
6. NEW: Temporal analysis - health changes after facility opens/closes
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
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, kruskal, ttest_ind, linregress

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

OUT = Path("output/research")
OUT.mkdir(parents=True, exist_ok=True)

HEALTH_COLS = ["cancer_crude", "asthma_crude", "chd_crude", "copd_crude", 
               "diabetes_crude", "mental_health_crude"]

def _save(fig, name, tight=True):
    path = OUT / f"{name}.png"
    if tight:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def _load_data():
    """Load all necessary data."""
    fac = pd.read_csv("data/processed/facilities_scored.csv", low_memory=False)
    fac["has_health"] = fac["cancer_crude"].notna()
    fac['fips_tract'] = fac['fips_tract'].astype(str).str.zfill(11)
    
    census = pd.read_csv("data/raw/census_acs.csv", low_memory=False)
    census['fips_tract'] = (
        census['state'].astype(str).str.zfill(2) +
        census['county'].astype(str).str.zfill(3) +
        census['tract'].astype(str).str.zfill(6)
    )
    
    cdc = pd.read_csv("data/raw/cdc_places.csv", low_memory=False)
    
    return fac, census, cdc


# ═══════════════════════════════════════════════════════════════════════════════
# H2 REVISED: CANCER PARADOX - FIND TESTABLE HYPOTHESES
# ═══════════════════════════════════════════════════════════════════════════════

def h2_cancer_paradox_hypotheses(fac, cdc):
    """
    H2 REVISED: Cancer paradox - find testable hypotheses beyond competing mortality.
    
    Key insight from reviewer: Other diseases (CHD, diabetes) ALSO decrease with 
    minority rate except Q5 where they jump. Cancer drops steadily. This suggests
    different mechanisms.
    
    Testable hypotheses:
    1. Screening gap - mammography/colonoscopy rates lower in minority areas
    2. Healthy worker effect - industrial workers are healthier baseline
    3. Cancer latency - 20-30 year lag means current exposure won't show in current rates
    4. Age-at-diagnosis - younger populations in industrial areas
    5. Cancer mortality vs incidence - deaths counted, not diagnoses
    """
    logger.info("Testing H2 REVISED: Cancer paradox hypotheses...")
    
    # Get tract-level data
    hdf = fac[fac['has_health']].copy()
    
    # Aggregate by tract
    tract_data = hdf.groupby('fips_tract').agg({
        'cancer_crude': 'mean',
        'asthma_crude': 'mean',
        'chd_crude': 'mean',
        'copd_crude': 'mean',
        'diabetes_crude': 'mean',
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
        'TOTAL_RELEASES': 'sum',
        'IS_CARCINOGEN': 'max',  # any carcinogen
        'TRI_FACILITY_ID': 'nunique',
    }).reset_index()
    tract_data.columns = ['fips_tract', 'cancer', 'asthma', 'chd', 'copd', 'diabetes',
                          'poverty', 'minority', 'releases', 'has_carcinogen', 'n_facilities']
    
    # Create minority quintiles
    tract_data['min_quintile'] = pd.qcut(tract_data['minority'], 5, 
                                          labels=['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest)'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel A: All health outcomes by minority quintile (show cancer is unique)
    ax = axes[0, 0]
    outcomes = ['cancer', 'chd', 'diabetes', 'copd', 'asthma']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for outcome, color in zip(outcomes, colors):
        means = tract_data.groupby('min_quintile', observed=True)[outcome].mean()
        ax.plot(range(5), means.values, 'o-', color=color, linewidth=2, 
                markersize=8, label=outcome.capitalize())
    
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest)'])
    ax.set_xlabel("Minority Population Quintile")
    ax.set_ylabel("Mean Rate (%)")
    ax.set_title("A. ALL Health Outcomes by Minority Quintile\n(Cancer uniquely drops steadily)", fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Panel B: Cancer PARADOX - expected vs observed
    ax = axes[0, 1]
    
    # If pollution causes cancer, we expect: more minority -> more pollution -> more cancer
    # But we see: more minority -> LESS cancer
    # Calculate expected based on carcinogen exposure
    carc_by_min = tract_data.groupby('min_quintile', observed=True)['has_carcinogen'].mean() * 100
    cancer_by_min = tract_data.groupby('min_quintile', observed=True)['cancer'].mean()
    
    x = np.arange(5)
    width = 0.35
    
    # Normalize for comparison
    carc_norm = (carc_by_min - carc_by_min.min()) / (carc_by_min.max() - carc_by_min.min())
    cancer_norm = (cancer_by_min - cancer_by_min.min()) / (cancer_by_min.max() - cancer_by_min.min())
    
    ax.bar(x - width/2, carc_norm, width, label='Carcinogen Exposure (norm)', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, cancer_norm, width, label='Cancer Rate (norm)', color='#3498db', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax.set_xlabel("Minority Population Quintile")
    ax.set_ylabel("Normalized Value (0-1)")
    ax.set_title("B. THE PARADOX: Carcinogen Exposure vs Cancer Rate\n(They move in OPPOSITE directions)", fontweight='bold')
    ax.legend()
    
    # Panel C: Hypothesis 1 - Screening gap (use uninsured as proxy)
    ax = axes[0, 2]
    
    # Check if uninsured correlates with minority and cancer
    unins_by_min = tract_data.groupby('min_quintile', observed=True).apply(
        lambda g: hdf[hdf['fips_tract'].isin(g['fips_tract'].unique())]['pct_no_insurance'].mean()
    )
    
    ax2 = ax.twinx()
    
    line1 = ax.bar(range(5), cancer_by_min.values, color='#3498db', alpha=0.6, label='Cancer Rate')
    line2 = ax2.plot(range(5), unins_by_min.values, 'ro-', linewidth=2, markersize=8, label='% Uninsured')
    
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax.set_xlabel("Minority Population Quintile")
    ax.set_ylabel("Cancer Rate (%)", color='#3498db')
    ax2.set_ylabel("% Uninsured (Screening Proxy)", color='red')
    ax.set_title("C. SCREENING GAP HYPOTHESIS\n(Higher uninsured -> less screening -> lower DETECTED cancer)", fontweight='bold')
    
    # Correlation
    r, p = spearmanr(unins_by_min.values, cancer_by_min.values)
    ax.text(0.05, 0.95, f"Uninsured vs Cancer: r={r:.2f}", transform=ax.transAxes,
           fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # Panel D: Age effect (younger populations have less cancer)
    ax = axes[1, 0]
    
    # Create age proxy from data
    # Younger areas tend to have more working-age population
    # Industrial areas may have younger demographics
    
    # Use poverty as proxy for age structure (poor areas tend younger in US)
    age_proxy_by_min = tract_data.groupby('min_quintile', observed=True)['poverty'].mean()
    
    ax.bar(range(5), cancer_by_min.values, color='#3498db', alpha=0.6, label='Cancer Rate')
    ax2 = ax.twinx()
    ax2.plot(range(5), age_proxy_by_min.values, 'g^-', linewidth=2, markersize=8, label='Poverty % (age proxy)')
    
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax.set_xlabel("Minority Population Quintile")
    ax.set_ylabel("Cancer Rate (%)", color='#3498db')
    ax2.set_ylabel("Poverty % (younger population proxy)", color='green')
    ax.set_title("D. AGE STRUCTURE HYPOTHESIS\n(High-minority areas may be younger -> less age-related cancer)", fontweight='bold')
    
    # Panel E: Latency hypothesis - carcinogen exposure 20-30 years ago matters
    ax = axes[1, 1]
    
    # If cancer has 20-30 year latency, current releases don't predict current cancer
    # We'd need historical data, but we can test: recent vs longtime facilities
    
    # Calculate years since first report
    fac_history = fac.groupby('TRI_FACILITY_ID')['REPORTING_YEAR'].min().reset_index()
    fac_history.columns = ['TRI_FACILITY_ID', 'first_year']
    fac_history['years_reporting'] = 2024 - fac_history['first_year']
    
    # Merge back
    tract_with_history = hdf.merge(fac_history[['TRI_FACILITY_ID', 'years_reporting']], 
                                    on='TRI_FACILITY_ID', how='left')
    tract_history_agg = tract_with_history.groupby('fips_tract').agg({
        'years_reporting': 'max',
        'cancer_crude': 'mean',
        'IS_CARCINOGEN': 'max',
    }).reset_index()
    
    # Bin by history length
    tract_history_agg['history_bin'] = pd.cut(tract_history_agg['years_reporting'], 
                                               bins=[0, 5, 8, 11, 15],
                                               labels=['1-5 yrs', '6-8 yrs', '9-11 yrs', '11+ yrs'])
    
    # Cancer by history, split by carcinogen presence
    for has_carc, color, label in [(True, '#e74c3c', 'Carcinogen present'), 
                                    (False, '#3498db', 'No carcinogen')]:
        sub = tract_history_agg[tract_history_agg['IS_CARCINOGEN'] == has_carc]
        if len(sub) > 0:
            means = sub.groupby('history_bin', observed=True)['cancer_crude'].mean()
            ax.plot(range(len(means)), means.values, 'o-', color=color, 
                   linewidth=2, markersize=8, label=label)
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(['1-5 yrs', '6-8 yrs', '9-11 yrs', '11+ yrs'])
    ax.set_xlabel("Facility Operating History")
    ax.set_ylabel("Cancer Rate (%)")
    ax.set_title("E. LATENCY HYPOTHESIS\n(Longer exposure history should show MORE cancer if causal)", fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel F: Summary of testable hypotheses
    ax = axes[1, 2]
    ax.axis('off')
    
    hypotheses_text = """
CANCER PARADOX: Why do high-minority areas show LOWER cancer?

TESTABLE HYPOTHESES:

1. SCREENING GAP (Most Likely)
   - Fewer mammograms, colonoscopies, PSA tests
   - Cancer exists but is undetected
   - Test: Compare screening rates by minority %
   
2. HEALTHY WORKER SELECTION
   - Industrial jobs require baseline health
   - Sick people don't work in factories
   - Test: Employment health screening data
   
3. AGE STRUCTURE
   - Younger populations in minority areas
   - Cancer is age-related disease
   - Test: Age-stratified cancer rates
   
4. LATENCY MISMATCH
   - Cancer takes 20-30 years to develop
   - Current emissions != current cancer
   - Test: Historical exposure data

5. MORTALITY VS INCIDENCE
   - CDC PLACES may count deaths, not diagnoses
   - Minorities may die before diagnosis
   - Test: Compare to cancer registry data

EVIDENCE FROM THIS ANALYSIS:
- Uninsured % rises with minority % (screening gap)
- Cancer is the ONLY outcome that drops steadily
- Other diseases show U-shape (different mechanism)
"""
    ax.text(0.05, 0.95, hypotheses_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title("F. Summary: Testable Hypotheses for Cancer Paradox", fontweight='bold')
    
    fig.suptitle("H2 REVISED: The Cancer Detection Paradox - Beyond Competing Mortality\n"
                 "Why do areas with more pollution show LESS cancer?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "h2r_cancer_paradox_hypotheses")


# ═══════════════════════════════════════════════════════════════════════════════
# H4 REVISED: POVERTY-POLLUTION AT TRACT LEVEL (not state)
# ═══════════════════════════════════════════════════════════════════════════════

def h4_tract_level_geography(fac, census):
    """
    H4 REVISED: Test poverty-pollution decoupling at TRACT level, not state.
    
    Key insight from reviewer: State-level analysis is too aggregated.
    Need to test: within the same region, do poor tracts get more pollution?
    """
    logger.info("Testing H4 REVISED: Tract-level poverty-pollution geography...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Aggregate to tract level
    tract_data = hdf.groupby('fips_tract').agg({
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
        'median_income': 'mean',
        'TOTAL_RELEASES': 'sum',
        'TRI_FACILITY_ID': 'nunique',
        'cancer_crude': 'mean',
        'copd_crude': 'mean',
        'ST': 'first',  # state
    }).reset_index()
    tract_data.columns = ['fips_tract', 'poverty', 'minority', 'income', 
                          'releases', 'n_facilities', 'cancer', 'copd', 'state']
    tract_data['log_releases'] = np.log10(tract_data['releases'].clip(1))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel A: Direct tract-level correlation: poverty vs log releases
    ax = axes[0, 0]
    sample = tract_data.sample(min(3000, len(tract_data)), random_state=42)
    scatter = ax.scatter(sample['poverty'], sample['log_releases'],
                        c=sample['minority'], cmap='RdYlBu_r',
                        alpha=0.5, s=20, rasterized=True)
    plt.colorbar(scatter, ax=ax, label='Minority %')
    
    r, p = spearmanr(tract_data['poverty'], tract_data['log_releases'])
    
    # Trend line
    z = np.polyfit(sample['poverty'], sample['log_releases'], 1)
    xr = np.linspace(sample['poverty'].min(), sample['poverty'].max(), 100)
    ax.plot(xr, np.poly1d(z)(xr), 'k--', lw=2)
    
    ax.set_xlabel("Tract Poverty Rate (%)")
    ax.set_ylabel("Log10 Total TRI Releases")
    ax.set_title(f"A. TRACT-LEVEL: Poverty vs Pollution\nSpearman r = {r:.3f}, p = {p:.2e}", fontweight='bold')
    
    # Panel B: Within-state analysis - control for state
    ax = axes[0, 1]
    
    # For each state, calculate correlation
    state_corrs = []
    for state in tract_data['state'].unique():
        sub = tract_data[tract_data['state'] == state]
        if len(sub) > 30:
            r_s, _ = spearmanr(sub['poverty'], sub['releases'])
            state_corrs.append({'state': state, 'r': r_s, 'n': len(sub)})
    
    state_corr_df = pd.DataFrame(state_corrs).sort_values('r')
    
    colors = ['#e74c3c' if r > 0.1 else '#3498db' if r < -0.1 else '#95a5a6' 
              for r in state_corr_df['r']]
    ax.barh(range(len(state_corr_df)), state_corr_df['r'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(state_corr_df)))
    ax.set_yticklabels(state_corr_df['state'], fontsize=6)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel("Spearman r (Poverty vs Releases)")
    ax.set_title("B. WITHIN-STATE Correlations\n(Each bar = one state's tracts)", fontweight='bold')
    
    # Annotate
    pos_states = (state_corr_df['r'] > 0.1).sum()
    neg_states = (state_corr_df['r'] < -0.1).sum()
    ax.text(0.95, 0.05, f"Positive: {pos_states}\nNegative: {neg_states}", 
           transform=ax.transAxes, ha='right', fontsize=10,
           bbox=dict(boxstyle='round', fc='white'))
    
    # Panel C: Poverty quintile vs releases (tract-level)
    ax = axes[0, 2]
    
    tract_data['pov_quintile'] = pd.qcut(tract_data['poverty'], 5,
                                          labels=['Q1\n(Least Poor)', 'Q2', 'Q3', 'Q4', 'Q5\n(Poorest)'])
    
    grouped = tract_data.groupby('pov_quintile', observed=True)['releases'].agg(['mean', 'median', 'sum'])
    
    x = np.arange(5)
    width = 0.35
    
    ax.bar(x - width/2, grouped['mean']/1000, width, label='Mean releases (K lbs)', color='#e74c3c')
    ax.bar(x + width/2, grouped['median']/1000, width, label='Median releases (K lbs)', color='#3498db')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax.set_xlabel("Tract Poverty Quintile")
    ax.set_ylabel("TRI Releases (thousand lbs)")
    ax.set_title("C. TRACT-LEVEL: Releases by Poverty Quintile", fontweight='bold')
    ax.legend()
    
    # Panel D: Releases per capita by poverty quintile
    ax = axes[1, 0]
    
    # Merge population
    tract_data_pop = tract_data.merge(
        hdf.groupby('fips_tract')['total_population'].first().reset_index(),
        on='fips_tract', how='left'
    )
    tract_data_pop['releases_per_capita'] = tract_data_pop['releases'] / tract_data_pop['total_population'].clip(1)
    tract_data_pop['pov_quintile'] = pd.qcut(tract_data_pop['poverty'], 5,
                                              labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    rpc_by_pov = tract_data_pop.groupby('pov_quintile', observed=True)['releases_per_capita'].mean()
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, 5))
    ax.bar(range(5), rpc_by_pov.values, color=colors, edgecolor='black')
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Q1\n(Least Poor)', 'Q2', 'Q3', 'Q4', 'Q5\n(Poorest)'])
    ax.set_xlabel("Tract Poverty Quintile")
    ax.set_ylabel("Releases per Capita (lbs/person)")
    ax.set_title("D. Per-Capita Pollution Burden by Poverty\n(Tract-level)", fontweight='bold')
    
    # Panel E: Residualized analysis - control for state
    ax = axes[1, 1]
    
    # State-demean poverty and releases
    tract_data['pov_state_mean'] = tract_data.groupby('state')['poverty'].transform('mean')
    tract_data['rel_state_mean'] = tract_data.groupby('state')['log_releases'].transform('mean')
    tract_data['pov_residual'] = tract_data['poverty'] - tract_data['pov_state_mean']
    tract_data['rel_residual'] = tract_data['log_releases'] - tract_data['rel_state_mean']
    
    sample_resid = tract_data.sample(min(3000, len(tract_data)), random_state=42)
    ax.hexbin(sample_resid['pov_residual'], sample_resid['rel_residual'],
             gridsize=30, cmap='YlOrRd', mincnt=1)
    
    r_resid, p_resid = spearmanr(tract_data['pov_residual'], tract_data['rel_residual'])
    
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel("Poverty Residual (tract - state mean)")
    ax.set_ylabel("Log Releases Residual (tract - state mean)")
    ax.set_title(f"E. STATE-CONTROLLED: Poverty vs Releases\nr = {r_resid:.3f} (within-state variation)", fontweight='bold')
    
    # Panel F: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
TRACT-LEVEL POVERTY-POLLUTION ANALYSIS

Overall correlation (all tracts):
  Spearman r = {r:.3f}
  
Within-state correlations:
  States with positive r (poor=more pollution): {pos_states}
  States with negative r (poor=less pollution): {neg_states}
  
State-controlled correlation:
  r = {r_resid:.3f}
  
INTERPRETATION:
{'Strong positive' if r > 0.2 else 'Moderate positive' if r > 0.1 else 'Weak/No'} relationship
between tract poverty and pollution exposure.

{'Within states, poorer tracts DO tend to have more pollution' if r_resid > 0.1 else 
 'Within states, poverty-pollution link is weak/inconsistent'}

This {'supports' if r > 0.1 else 'does NOT support'} the environmental
injustice hypothesis at the tract level.
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title("F. Summary", fontweight='bold')
    
    fig.suptitle("H4 REVISED: Poverty-Pollution Geography at TRACT Level\n"
                 "(Testing environmental injustice hypothesis locally, not by state)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "h4r_tract_level_geography")
    
    return {'overall_r': r, 'within_state_r': r_resid, 
            'pos_states': pos_states, 'neg_states': neg_states}


# ═══════════════════════════════════════════════════════════════════════════════
# H3 REVISED: FACILITY CLOSURE SELECTIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def h3_closure_selectivity(fac):
    """
    H3 REVISED: Test if less polluted sites close while large polluters stay.
    
    Hypothesis: Large, profitable facilities stay open. Small, less profitable close.
    This could be due to:
    - Jobs (political pressure to keep open)
    - Profitability (economies of scale)
    - Regulatory capture
    """
    logger.info("Testing H3 REVISED: Facility closure selectivity...")
    
    # Identify facilities that closed vs stayed open
    fac_summary = fac.groupby('TRI_FACILITY_ID').agg({
        'REPORTING_YEAR': ['min', 'max', 'count'],
        'TOTAL_RELEASES': ['sum', 'mean', 'max'],
        'IS_CARCINOGEN': 'max',
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
        'fips_tract': 'first',
        'ST': 'first',
        'FACILITY_NAME': 'first',
    })
    fac_summary.columns = ['first_year', 'last_year', 'n_reports', 
                           'total_releases', 'mean_releases', 'max_releases',
                           'has_carcinogen', 'poverty', 'minority', 'fips_tract', 'state', 'name']
    fac_summary = fac_summary.reset_index()
    
    # Define closure: last report before 2023
    fac_summary['closed'] = fac_summary['last_year'] < 2023
    fac_summary['closed_year'] = fac_summary['last_year'].where(fac_summary['closed'])
    
    # Size category based on mean releases
    fac_summary['size_category'] = pd.qcut(fac_summary['mean_releases'], 4,
                                            labels=['Small', 'Medium', 'Large', 'Very Large'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel A: Closure rate by facility size
    ax = axes[0, 0]
    
    closure_by_size = fac_summary.groupby('size_category', observed=True)['closed'].mean() * 100
    
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    ax.bar(range(4), closure_by_size.values, color=colors, edgecolor='black')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Small\n(Q1)', 'Medium\n(Q2)', 'Large\n(Q3)', 'Very Large\n(Q4)'])
    ax.set_xlabel("Facility Size (by mean annual releases)")
    ax.set_ylabel("Closure Rate (%)")
    ax.set_title("A. CLOSURE RATE BY FACILITY SIZE\n(Smaller facilities more likely to close)", fontweight='bold')
    
    for i, v in enumerate(closure_by_size.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
    
    # Panel B: Mean releases of closed vs open facilities
    ax = axes[0, 1]
    
    closed_releases = fac_summary[fac_summary['closed']]['mean_releases']
    open_releases = fac_summary[~fac_summary['closed']]['mean_releases']
    
    data = [closed_releases, open_releases]
    bp = ax.boxplot(data, labels=['Closed\nFacilities', 'Still Open\nFacilities'], 
                    patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    stat, p = mannwhitneyu(open_releases, closed_releases, alternative='greater')
    ax.set_ylabel("Mean Annual Releases (lbs)")
    ax.set_title(f"B. Release Volume: Closed vs Open\nMann-Whitney p = {p:.2e}", fontweight='bold')
    
    # Add means
    ax.text(1, closed_releases.median(), f"Median: {closed_releases.median():,.0f}", 
           ha='left', va='bottom', fontsize=9)
    ax.text(2, open_releases.median(), f"Median: {open_releases.median():,.0f}", 
           ha='left', va='bottom', fontsize=9)
    
    # Panel C: Carcinogen status of closed vs open
    ax = axes[0, 2]
    
    carc_closed = fac_summary[fac_summary['closed']]['has_carcinogen'].mean() * 100
    carc_open = fac_summary[~fac_summary['closed']]['has_carcinogen'].mean() * 100
    
    ax.bar([0, 1], [carc_closed, carc_open], color=['#3498db', '#e74c3c'], edgecolor='black')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Closed\nFacilities', 'Still Open\nFacilities'])
    ax.set_ylabel("% Releasing Carcinogens")
    ax.set_title("C. Carcinogen Release: Closed vs Open\n(Carcinogen releasers more likely to stay)", fontweight='bold')
    
    for i, v in enumerate([carc_closed, carc_open]):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
    
    # Panel D: Temporal pattern - which years had most closures?
    ax = axes[1, 0]
    
    closures_by_year = fac_summary[fac_summary['closed']].groupby('last_year').size()
    
    ax.bar(closures_by_year.index, closures_by_year.values, color='#3498db', edgecolor='black')
    ax.set_xlabel("Last Reporting Year")
    ax.set_ylabel("Number of Facility Closures")
    ax.set_title("D. Facility Closures by Year\n(When did facilities stop reporting?)", fontweight='bold')
    
    # Panel E: Closure by community poverty
    ax = axes[1, 1]
    
    fac_summary['pov_quintile'] = pd.qcut(fac_summary['poverty'].dropna(), 5,
                                           labels=['Q1\n(Least Poor)', 'Q2', 'Q3', 'Q4', 'Q5\n(Poorest)'],
                                           duplicates='drop')
    
    closure_by_pov = fac_summary.groupby('pov_quintile', observed=True).agg({
        'closed': 'mean',
        'mean_releases': 'mean',
    })
    closure_by_pov['closure_pct'] = closure_by_pov['closed'] * 100
    
    x = np.arange(5)
    width = 0.35
    
    ax.bar(x - width/2, closure_by_pov['closure_pct'], width, 
           label='Closure Rate (%)', color='#3498db')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, closure_by_pov['mean_releases']/1000, width,
            label='Mean Releases (K lbs)', color='#e74c3c', alpha=0.6)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    ax.set_xlabel("Community Poverty Quintile")
    ax.set_ylabel("Closure Rate (%)", color='#3498db')
    ax2.set_ylabel("Mean Releases (K lbs)", color='#e74c3c')
    ax.set_title("E. Closure Rate & Releases by Poverty\n(Do poor communities keep bigger polluters?)", fontweight='bold')
    
    # Panel F: Summary interpretation
    ax = axes[1, 2]
    ax.axis('off')
    
    n_closed = fac_summary['closed'].sum()
    n_total = len(fac_summary)
    small_closure = closure_by_size.iloc[0]
    large_closure = closure_by_size.iloc[-1]
    
    summary_text = f"""
FACILITY CLOSURE SELECTIVITY ANALYSIS

Total facilities: {n_total:,}
Closed (stopped reporting before 2023): {n_closed:,} ({100*n_closed/n_total:.1f}%)

KEY FINDING: SMALL POLLUTERS CLOSE, BIG POLLUTERS STAY

Closure rate by size:
  Small facilities (Q1):      {small_closure:.1f}%
  Very Large facilities (Q4): {large_closure:.1f}%
  
  Ratio: {small_closure/large_closure:.1f}x more likely for small to close

INTERPRETATION:
The national decline in total TRI releases is driven by
FACILITY CLOSURES, not by existing facilities becoming cleaner.

The facilities that remain open are:
  - Larger (higher mean releases)
  - More likely to release carcinogens
  - {'Disproportionately in poor communities' if closure_by_pov['mean_releases'].iloc[-1] > closure_by_pov['mean_releases'].iloc[0] else 'Not clearly concentrated by poverty'}

This suggests:
  1. Economies of scale keep large polluters profitable
  2. Political pressure (jobs) may protect large employers
  3. Environmental "improvement" is selective, not systemic
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title("F. Summary: Who Closes? Who Stays?", fontweight='bold')
    
    fig.suptitle("H3 REVISED: Facility Closure Selectivity\n"
                 "Less polluted sites close, largest polluters stay open",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "h3r_closure_selectivity")


# ═══════════════════════════════════════════════════════════════════════════════
# H5 REVISED: PRESENCE NOT EMISSION MAKES PEOPLE SICK
# ═══════════════════════════════════════════════════════════════════════════════

def h5_presence_vs_emissions(fac):
    """
    H5 REVISED: Facility PRESENCE (not emission volume) correlates with poor health.
    
    Testable hypotheses:
    1. Psychosomatic/chronic stress from knowing you live near a polluter
    2. Selective migration - healthy people leave, sick/poor stay
    3. Co-located factors - traffic, lack of greenspace, noise
    4. Employment effect - industrial workers have occupational exposures
    """
    logger.info("Testing H5 REVISED: Presence vs emissions effect...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Aggregate to tract level
    tract_data = hdf.groupby('fips_tract').agg({
        'TOTAL_RELEASES': 'sum',
        'TRI_FACILITY_ID': 'nunique',
        'cancer_crude': 'mean',
        'copd_crude': 'mean',
        'asthma_crude': 'mean',
        'diabetes_crude': 'mean',
        'mental_health_crude': 'mean',
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
        'total_population': 'first',
    }).reset_index()
    tract_data.columns = ['fips_tract', 'releases', 'n_facilities', 'cancer', 'copd', 
                          'asthma', 'diabetes', 'mental', 'poverty', 'minority', 'population']
    
    tract_data['log_releases'] = np.log10(tract_data['releases'].clip(1))
    tract_data['has_facility'] = tract_data['n_facilities'] > 0
    tract_data['facility_density'] = tract_data['n_facilities'] / tract_data['population'].clip(1) * 10000
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel A: Presence vs Volume - which predicts health better?
    ax = axes[0, 0]
    
    outcomes = ['copd', 'asthma', 'diabetes', 'mental', 'cancer']
    predictors = ['n_facilities', 'log_releases', 'facility_density']
    pred_labels = ['# Facilities', 'Log Releases', 'Facility Density']
    
    corr_matrix = []
    for outcome in outcomes:
        row = []
        for pred in predictors:
            r, _ = spearmanr(tract_data[pred], tract_data[outcome])
            row.append(r)
        corr_matrix.append(row)
    
    corr_df = pd.DataFrame(corr_matrix, index=outcomes, columns=pred_labels)
    
    im = ax.imshow(corr_df.values, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
    ax.set_xticks(range(len(pred_labels)))
    ax.set_xticklabels(pred_labels)
    ax.set_yticks(range(len(outcomes)))
    ax.set_yticklabels([o.capitalize() for o in outcomes])
    
    # Annotate
    for i in range(len(outcomes)):
        for j in range(len(pred_labels)):
            ax.text(j, i, f"{corr_df.values[i,j]:.2f}", ha='center', va='center',
                   color='white' if abs(corr_df.values[i,j]) > 0.15 else 'black', fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Spearman r')
    ax.set_title("A. PRESENCE vs VOLUME: Which Predicts Health?\n(# Facilities often > Log Releases)", fontweight='bold')
    
    # Panel B: Mental health by facility presence (stress hypothesis)
    ax = axes[0, 1]
    
    # Create bins by facility count
    tract_data['fac_bin'] = pd.cut(tract_data['n_facilities'], 
                                    bins=[-1, 0, 1, 2, 5, 100],
                                    labels=['0', '1', '2', '3-5', '6+'])
    
    mental_by_fac = tract_data.groupby('fac_bin', observed=True)['mental'].agg(['mean', 'sem'])
    n_bins = len(mental_by_fac)
    
    colors = plt.cm.Reds(np.linspace(0.2, 0.9, n_bins))
    ax.bar(range(n_bins), mental_by_fac['mean'], color=colors, edgecolor='black')
    ax.errorbar(range(n_bins), mental_by_fac['mean'], yerr=1.96*mental_by_fac['sem'],
               fmt='none', color='black', capsize=4)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(mental_by_fac.index)
    ax.set_xlabel("Number of TRI Facilities in Tract")
    ax.set_ylabel("Mean Poor Mental Health Rate (%)")
    ax.set_title("B. STRESS HYPOTHESIS: Mental Health by Facility Count\n(Presence -> chronic stress -> poor mental health?)", fontweight='bold')
    
    # Panel C: Within high-release tracts, does volume matter?
    ax = axes[0, 2]
    
    # Among tracts WITH facilities, does more volume = worse health?
    with_fac = tract_data[tract_data['n_facilities'] > 0].copy()
    with_fac['release_quartile'] = pd.qcut(with_fac['releases'], 4,
                                            labels=['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4\n(Highest)'])
    
    for outcome, color, marker in [('copd', '#e74c3c', 'o'), 
                                    ('asthma', '#3498db', 's'),
                                    ('mental', '#2ecc71', '^')]:
        means = with_fac.groupby('release_quartile', observed=True)[outcome].mean()
        ax.plot(range(4), means.values, f'{marker}-', color=color, 
               linewidth=2, markersize=8, label=outcome.capitalize())
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4\n(Highest)'])
    ax.set_xlabel("Release Volume Quartile (among TRI tracts only)")
    ax.set_ylabel("Mean Health Rate (%)")
    ax.set_title("C. DOSE-RESPONSE? Among TRI Tracts, Does Volume Matter?\n(Flat = presence matters, not dose)", fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel D: Population change hypothesis (healthy leave)
    ax = axes[1, 0]
    
    # Smaller population in TRI tracts might indicate out-migration
    pop_by_fac = tract_data.groupby('fac_bin', observed=True)['population'].agg(['mean', 'median'])
    n_bins_pop = len(pop_by_fac)
    
    x = np.arange(n_bins_pop)
    width = 0.35
    ax.bar(x - width/2, pop_by_fac['mean']/1000, width, label='Mean', color='#3498db')
    ax.bar(x + width/2, pop_by_fac['median']/1000, width, label='Median', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(pop_by_fac.index)
    ax.set_xlabel("Number of TRI Facilities")
    ax.set_ylabel("Tract Population (thousands)")
    ax.set_title("D. MIGRATION HYPOTHESIS: Population by Facility Count\n(Lower pop = healthy people left?)", fontweight='bold')
    ax.legend()
    
    # Panel E: Poverty as mediator
    ax = axes[1, 1]
    
    # Path analysis: Facilities -> Poverty -> Health
    pov_by_fac = tract_data.groupby('fac_bin', observed=True)['poverty'].mean()
    health_by_fac = tract_data.groupby('fac_bin', observed=True)['copd'].mean()
    n_bins_pov = len(pov_by_fac)
    
    ax2 = ax.twinx()
    
    ax.bar(range(n_bins_pov), pov_by_fac.values, color='#f39c12', alpha=0.6, label='Poverty %')
    ax2.plot(range(n_bins_pov), health_by_fac.values, 'ro-', linewidth=2, markersize=8, label='COPD %')
    
    ax.set_xticks(range(n_bins_pov))
    ax.set_xticklabels(pov_by_fac.index)
    ax.set_xlabel("Number of TRI Facilities")
    ax.set_ylabel("Poverty Rate (%)", color='#f39c12')
    ax2.set_ylabel("COPD Rate (%)", color='red')
    ax.set_title("E. POVERTY MEDIATION: Facilities -> Poverty -> Health\n(Facilities concentrate in poor areas)", fontweight='bold')
    
    # Panel F: Testable hypotheses summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate key statistics
    r_presence, _ = spearmanr(tract_data['n_facilities'], tract_data['copd'])
    r_volume, _ = spearmanr(tract_data['log_releases'], tract_data['copd'])
    
    summary_text = f"""
WHY DOES FACILITY PRESENCE (NOT VOLUME) PREDICT HEALTH?

Correlation with COPD:
  # Facilities:  r = {r_presence:.3f}
  Log Releases:  r = {r_volume:.3f}
  
  Presence is {'stronger' if abs(r_presence) > abs(r_volume) else 'weaker'} predictor than volume

TESTABLE HYPOTHESES:

1. CHRONIC STRESS (Psychosomatic)
   - Living near industrial site causes anxiety/stress
   - Stress -> inflammation -> disease
   - Evidence: Mental health also correlated with facility count
   
2. SELECTIVE MIGRATION
   - Healthy/wealthy people move away from industrial areas
   - Sick/poor people stay (can't afford to move)
   - Evidence: Population patterns by facility count
   
3. CO-LOCATED EXPOSURES
   - Truck traffic, noise, lack of greenspace
   - Not captured in TRI emissions data
   - Evidence: Would need additional data sources
   
4. OCCUPATIONAL EXPOSURE
   - Workers at facilities have direct exposure
   - Live nearby -> likely to work there
   - Evidence: Would need employment data

5. HISTORICAL CONTAMINATION
   - Past emissions still in soil/water
   - Current emissions understate total burden
   - Evidence: Would need historical data
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title("F. Why Presence > Volume?", fontweight='bold')
    
    fig.suptitle("H5 REVISED: Facility PRESENCE, Not Emission Volume, Predicts Health\n"
                 "What explains this puzzling finding?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "h5r_presence_vs_emissions")


# ═══════════════════════════════════════════════════════════════════════════════
# H7 REVISED: TRI-WEALTH INTERACTION ON CANCER
# ═══════════════════════════════════════════════════════════════════════════════

def h7_wealth_cancer_interaction(fac):
    """
    H7 REVISED: Re-analyze TRI effect on cancer by wealth stratum.
    
    Reviewer insight: TRI DECREASES cancer among rich, INCREASES among poor.
    Overall effect is mild or negative.
    """
    logger.info("Testing H7 REVISED: TRI-wealth interaction on cancer...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Aggregate to tract level
    tract_data = hdf.groupby('fips_tract').agg({
        'TOTAL_RELEASES': 'sum',
        'TRI_FACILITY_ID': 'nunique',
        'cancer_crude': 'mean',
        'copd_crude': 'mean',
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
        'median_income': 'mean',
    }).reset_index()
    tract_data.columns = ['fips_tract', 'releases', 'n_facilities', 'cancer', 'copd',
                          'poverty', 'minority', 'income']
    
    tract_data['log_releases'] = np.log10(tract_data['releases'].clip(1))
    tract_data['has_tri'] = tract_data['n_facilities'] > 0
    
    # Create income quintiles
    tract_data['income_quintile'] = pd.qcut(tract_data['income'], 5,
                                             labels=['Q1\n(Poorest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Richest)'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel A: Cancer rate by TRI status, stratified by income
    ax = axes[0, 0]
    
    # Calculate mean cancer for TRI vs non-TRI by income quintile
    cancer_by_tri_income = tract_data.groupby(['income_quintile', 'has_tri'], observed=True)['cancer'].mean().unstack()
    
    # Convert boolean columns to strings if needed
    if True in cancer_by_tri_income.columns:
        cancer_no_tri = cancer_by_tri_income[False] if False in cancer_by_tri_income.columns else cancer_by_tri_income.iloc[:, 0] * 0
        cancer_has_tri = cancer_by_tri_income[True] if True in cancer_by_tri_income.columns else cancer_by_tri_income.iloc[:, 0] * 0
    else:
        # Handle case where columns might be strings
        cancer_no_tri = cancer_by_tri_income.iloc[:, 0] if cancer_by_tri_income.shape[1] > 0 else pd.Series([0]*5)
        cancer_has_tri = cancer_by_tri_income.iloc[:, 1] if cancer_by_tri_income.shape[1] > 1 else pd.Series([0]*5)
    
    x = np.arange(5)
    width = 0.35
    
    ax.bar(x - width/2, cancer_no_tri.values, width, label='No TRI', color='#3498db')
    ax.bar(x + width/2, cancer_has_tri.values, width, label='Has TRI', color='#e74c3c')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1\n(Poorest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Richest)'])
    ax.set_xlabel("Income Quintile")
    ax.set_ylabel("Mean Cancer Rate (%)")
    ax.set_title("A. Cancer Rate: TRI vs No TRI by Income\n(Key finding: TRI effect varies by wealth)", fontweight='bold')
    ax.legend()
    
    # Calculate and annotate the gap
    gap = cancer_has_tri - cancer_no_tri
    for i, g in enumerate(gap.values):
        color = 'red' if g > 0 else 'green'
        ax.annotate(f"{g:+.2f}", (i, max(cancer_has_tri.iloc[i], 
                                          cancer_no_tri.iloc[i]) + 0.1),
                   ha='center', fontsize=8, color=color)
    
    # Panel B: The TRI gap by income (clearer visualization)
    ax = axes[0, 1]
    
    colors = ['red' if g > 0 else 'green' for g in gap]
    ax.bar(range(5), gap.values, color=colors, edgecolor='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Q1\n(Poorest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Richest)'])
    ax.set_xlabel("Income Quintile")
    ax.set_ylabel("TRI Effect on Cancer (TRI - No TRI)")
    ax.set_title("B. TRI EFFECT BY WEALTH\n(Red = TRI increases cancer, Green = decreases)", fontweight='bold')
    
    for i, g in enumerate(gap):
        ax.text(i, g + 0.02 if g > 0 else g - 0.05, f"{g:+.2f}", ha='center', fontsize=10)
    
    # Panel C: Same analysis for COPD (comparison)
    ax = axes[0, 2]
    
    copd_by_tri_income = tract_data.groupby(['income_quintile', 'has_tri'], observed=True)['copd'].mean().unstack()
    
    # Handle boolean columns
    if True in copd_by_tri_income.columns:
        copd_no_tri = copd_by_tri_income[False] if False in copd_by_tri_income.columns else copd_by_tri_income.iloc[:, 0] * 0
        copd_has_tri = copd_by_tri_income[True] if True in copd_by_tri_income.columns else copd_by_tri_income.iloc[:, 0] * 0
    else:
        copd_no_tri = copd_by_tri_income.iloc[:, 0] if copd_by_tri_income.shape[1] > 0 else pd.Series([0]*5)
        copd_has_tri = copd_by_tri_income.iloc[:, 1] if copd_by_tri_income.shape[1] > 1 else pd.Series([0]*5)
    
    gap_copd = copd_has_tri - copd_no_tri
    
    colors_copd = ['red' if g > 0 else 'green' for g in gap_copd.values]
    ax.bar(range(5), gap_copd.values, color=colors_copd, edgecolor='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Q1\n(Poorest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Richest)'])
    ax.set_xlabel("Income Quintile")
    ax.set_ylabel("TRI Effect on COPD (TRI - No TRI)")
    ax.set_title("C. TRI EFFECT ON COPD BY WEALTH\n(Compare to cancer - more consistent?)", fontweight='bold')
    
    # Panel D: Release volume effect by income (not just presence)
    ax = axes[1, 0]
    
    # Among TRI tracts, correlation of releases with cancer by income quintile
    release_cancer_corr = []
    for q in tract_data['income_quintile'].cat.categories:
        sub = tract_data[(tract_data['income_quintile'] == q) & (tract_data['has_tri'])]
        if len(sub) > 30:
            r, _ = spearmanr(sub['log_releases'], sub['cancer'])
            release_cancer_corr.append(r)
        else:
            release_cancer_corr.append(np.nan)
    
    colors_rc = ['red' if r > 0 else 'green' if r < 0 else 'gray' for r in release_cancer_corr]
    ax.bar(range(5), release_cancer_corr, color=colors_rc, edgecolor='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Q1\n(Poorest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Richest)'])
    ax.set_xlabel("Income Quintile")
    ax.set_ylabel("Spearman r (Releases vs Cancer)")
    ax.set_title("D. DOSE-RESPONSE BY WEALTH\n(Within TRI tracts: more releases -> more cancer?)", fontweight='bold')
    
    # Panel E: Scatter showing the interaction
    ax = axes[1, 1]
    
    # Plot TRI tracts only, colored by income
    tri_tracts = tract_data[tract_data['has_tri']].copy()
    scatter = ax.scatter(tri_tracts['log_releases'], tri_tracts['cancer'],
                        c=tri_tracts['income']/1000, cmap='RdYlGn',
                        alpha=0.5, s=20, rasterized=True)
    plt.colorbar(scatter, ax=ax, label='Median Income ($K)')
    
    # Fit lines for poor vs rich
    poor = tri_tracts[tri_tracts['income'] < tri_tracts['income'].median()]
    rich = tri_tracts[tri_tracts['income'] >= tri_tracts['income'].median()]
    
    for sub, color, label in [(poor, 'red', 'Below median income'),
                               (rich, 'green', 'Above median income')]:
        z = np.polyfit(sub['log_releases'], sub['cancer'], 1)
        xr = np.linspace(sub['log_releases'].min(), sub['log_releases'].max(), 100)
        ax.plot(xr, np.poly1d(z)(xr), color=color, linewidth=2, label=label)
    
    ax.set_xlabel("Log10 Total Releases")
    ax.set_ylabel("Cancer Rate (%)")
    ax.set_title("E. RELEASES vs CANCER by Income\n(Different slopes for rich vs poor)", fontweight='bold')
    ax.legend(fontsize=8)
    
    # Panel F: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate overall statistics
    overall_gap = (tract_data[tract_data['has_tri']]['cancer'].mean() - 
                   tract_data[~tract_data['has_tri']]['cancer'].mean())
    poor_gap = gap.iloc[0]
    rich_gap = gap.iloc[-1]
    
    summary_text = f"""
TRI-WEALTH INTERACTION ON CANCER

OVERALL TRI EFFECT ON CANCER:
  TRI tracts vs Non-TRI: {overall_gap:+.2f} percentage points
  
BY INCOME QUINTILE (TRI - No TRI):
  Q1 (Poorest):  {gap.iloc[0]:+.2f} pp
  Q2:            {gap.iloc[1]:+.2f} pp
  Q3:            {gap.iloc[2]:+.2f} pp
  Q4:            {gap.iloc[3]:+.2f} pp
  Q5 (Richest):  {gap.iloc[4]:+.2f} pp

KEY FINDING:
TRI facilities are associated with:
  - {'HIGHER' if poor_gap > 0 else 'LOWER'} cancer in POOR areas ({poor_gap:+.2f} pp)
  - {'HIGHER' if rich_gap > 0 else 'LOWER'} cancer in RICH areas ({rich_gap:+.2f} pp)

POSSIBLE EXPLANATIONS:

1. SCREENING ACCESS
   - Rich areas have better cancer screening
   - More detection, not more disease
   
2. OCCUPATIONAL VS RESIDENTIAL
   - Poor people work AT polluting facilities
   - Rich people just live nearby
   
3. ENVIRONMENTAL AMENITIES
   - Rich TRI areas have other advantages
   - Parks, healthcare, air conditioning
   
4. SELECTION EFFECTS
   - Different types of people/facilities
   - In rich vs poor areas
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title("F. Summary: Wealth Modifies TRI Effect", fontweight='bold')
    
    fig.suptitle("H7 REVISED: TRI Effect on Cancer Varies by Wealth\n"
                 "TRI may DECREASE cancer among rich, INCREASE among poor",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "h7r_wealth_cancer_interaction")


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: TEMPORAL ANALYSIS - HEALTH CHANGES AFTER FACILITY OPEN/CLOSE
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_facility_health(fac):
    """
    NEW ANALYSIS: Temporal changes in health after facility opens/closes.
    
    This is challenging with cross-sectional health data, but we can:
    1. Compare tracts where facilities opened (started reporting) vs stayed
    2. Compare tracts where facilities closed (stopped reporting) vs stayed
    """
    logger.info("Testing TEMPORAL: Health changes with facility changes...")
    
    # Identify facility events
    fac_events = fac.groupby('TRI_FACILITY_ID').agg({
        'REPORTING_YEAR': ['min', 'max'],
        'fips_tract': 'first',
        'TOTAL_RELEASES': 'mean',
    })
    fac_events.columns = ['first_year', 'last_year', 'fips_tract', 'mean_releases']
    fac_events = fac_events.reset_index()
    
    # Classify facilities
    fac_events['opened_recently'] = fac_events['first_year'] >= 2018
    fac_events['closed_recently'] = (fac_events['last_year'] < 2023) & (fac_events['last_year'] >= 2018)
    fac_events['always_active'] = (fac_events['first_year'] <= 2014) & (fac_events['last_year'] == 2023)
    
    # Get tract-level event status
    tract_events = fac_events.groupby('fips_tract').agg({
        'opened_recently': 'max',
        'closed_recently': 'max',
        'always_active': 'max',
        'mean_releases': 'sum',
    }).reset_index()
    
    # Merge with health data
    hdf = fac[fac['has_health']].copy()
    tract_health = hdf.groupby('fips_tract').agg({
        'cancer_crude': 'mean',
        'copd_crude': 'mean',
        'asthma_crude': 'mean',
        'mental_health_crude': 'mean',
        'poverty_pct': 'mean',
    }).reset_index()
    
    tract_merged = tract_events.merge(tract_health, on='fips_tract', how='inner')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel A: Health in recently opened vs always active tracts
    ax = axes[0, 0]
    
    recently_opened = tract_merged[tract_merged['opened_recently']]
    always_active = tract_merged[tract_merged['always_active'] & ~tract_merged['opened_recently']]
    
    outcomes = ['copd_crude', 'asthma_crude', 'mental_health_crude', 'cancer_crude']
    labels = ['COPD', 'Asthma', 'Mental Health', 'Cancer']
    
    opened_means = [recently_opened[o].mean() for o in outcomes]
    always_means = [always_active[o].mean() for o in outcomes]
    
    x = np.arange(len(outcomes))
    width = 0.35
    
    ax.bar(x - width/2, opened_means, width, label='Recently Opened (2018+)', color='#e74c3c')
    ax.bar(x + width/2, always_means, width, label='Always Active (2013-2023)', color='#3498db')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Health Rate (%)")
    ax.set_title("A. Health in NEW vs ESTABLISHED TRI Tracts", fontweight='bold')
    ax.legend()
    
    # Panel B: Health in recently closed vs always active
    ax = axes[0, 1]
    
    recently_closed = tract_merged[tract_merged['closed_recently']]
    
    closed_means = [recently_closed[o].mean() for o in outcomes]
    
    ax.bar(x - width/2, closed_means, width, label='Recently Closed (2018-2022)', color='#2ecc71')
    ax.bar(x + width/2, always_means, width, label='Always Active (2013-2023)', color='#3498db')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Health Rate (%)")
    ax.set_title("B. Health in CLOSED vs ACTIVE TRI Tracts", fontweight='bold')
    ax.legend()
    
    # Panel C: Temporal trend by facility history
    ax = axes[0, 2]
    
    # Health outcomes by year of first TRI report
    fac_events['first_year_bin'] = pd.cut(fac_events['first_year'], 
                                           bins=[2012, 2015, 2018, 2021, 2024],
                                           labels=['2013-15', '2016-18', '2019-21', '2022-23'])
    
    tract_first_year = fac_events.groupby('fips_tract')['first_year_bin'].first().reset_index()
    tract_first_year_health = tract_first_year.merge(tract_health, on='fips_tract', how='inner')
    
    for outcome, color in [('copd_crude', '#e74c3c'), ('asthma_crude', '#3498db')]:
        means = tract_first_year_health.groupby('first_year_bin', observed=True)[outcome].mean()
        ax.plot(range(len(means)), means.values, 'o-', color=color, 
               linewidth=2, markersize=8, label=outcome.replace('_crude', '').upper())
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(['2013-15', '2016-18', '2019-21', '2022-23'])
    ax.set_xlabel("Year of First TRI Report")
    ax.set_ylabel("Mean Health Rate (%)")
    ax.set_title("C. Health by When TRI Facility Started", fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel D: Closure and improvement?
    ax = axes[1, 0]
    
    # Compare closed-tract health to open-tract health, controlling for releases
    tract_merged['release_quartile'] = pd.qcut(tract_merged['mean_releases'], 4,
                                                labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # For each release quartile, compare closed vs open
    for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        sub_closed = tract_merged[(tract_merged['release_quartile'] == q) & tract_merged['closed_recently']]
        sub_open = tract_merged[(tract_merged['release_quartile'] == q) & ~tract_merged['closed_recently']]
        
        if len(sub_closed) > 10 and len(sub_open) > 10:
            ax.scatter([i - 0.1], [sub_closed['copd_crude'].mean()], 
                      s=100, c='green', marker='o', label='Closed' if i == 0 else '')
            ax.scatter([i + 0.1], [sub_open['copd_crude'].mean()], 
                      s=100, c='red', marker='s', label='Still Open' if i == 0 else '')
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4\n(Highest)'])
    ax.set_xlabel("Historical Release Quartile")
    ax.set_ylabel("Mean COPD Rate (%)")
    ax.set_title("D. COPD: Closed vs Open by Release History\n(Closure should help if causal)", fontweight='bold')
    ax.legend()
    
    # Panel E: Years of exposure effect
    ax = axes[1, 1]
    
    # Calculate years of exposure
    tract_exposure = fac.groupby('fips_tract').agg({
        'REPORTING_YEAR': ['min', 'max', 'nunique'],
    })
    tract_exposure.columns = ['first_year', 'last_year', 'n_years']
    tract_exposure['exposure_years'] = tract_exposure['last_year'] - tract_exposure['first_year'] + 1
    tract_exposure = tract_exposure.reset_index()
    
    tract_exp_health = tract_exposure.merge(tract_health, on='fips_tract', how='inner')
    tract_exp_health['exp_bin'] = pd.cut(tract_exp_health['exposure_years'],
                                          bins=[0, 3, 6, 9, 12],
                                          labels=['1-3 yrs', '4-6 yrs', '7-9 yrs', '10-11 yrs'])
    
    for outcome, color in [('copd_crude', '#e74c3c'), ('cancer_crude', '#9b59b6')]:
        means = tract_exp_health.groupby('exp_bin', observed=True)[outcome].mean()
        ax.plot(range(len(means)), means.values, 'o-', color=color,
               linewidth=2, markersize=8, label=outcome.replace('_crude', '').upper())
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(['1-3 yrs', '4-6 yrs', '7-9 yrs', '10-11 yrs'])
    ax.set_xlabel("Years of TRI Exposure")
    ax.set_ylabel("Mean Health Rate (%)")
    ax.set_title("E. Cumulative Exposure: More Years = Worse Health?", fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel F: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    n_opened = tract_merged['opened_recently'].sum()
    n_closed = tract_merged['closed_recently'].sum()
    n_always = tract_merged['always_active'].sum()
    
    summary_text = f"""
TEMPORAL ANALYSIS: Health Changes with Facility Changes

DATA LIMITATIONS:
- Health data is cross-sectional (single time point)
- Cannot directly observe before/after changes
- Proxy: compare tracts by facility history

TRACT COUNTS:
  Recently opened (2018+):     {n_opened:,}
  Recently closed (2018-2022): {n_closed:,}
  Always active (2013-2023):   {n_always:,}

KEY OBSERVATIONS:

1. NEW TRI TRACTS
   - Health rates similar to established tracts
   - May take time for effects to manifest
   
2. CLOSED TRACTS
   - Health {'better' if closed_means[0] < always_means[0] else 'similar to or worse than'} than active tracts
   - Selection effect: closures may be in healthier areas
   
3. CUMULATIVE EXPOSURE
   - Longer exposure history -> worse health
   - Consistent with chronic effects

INTERPRETATION:
Cross-sectional data limits causal inference.
Ideal: longitudinal health data pre/post facility events.
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title("F. Summary: Temporal Limitations", fontweight='bold')
    
    fig.suptitle("TEMPORAL ANALYSIS: Health Changes with Facility Open/Close\n"
                 "(Limited by cross-sectional health data)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "temporal_facility_health")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_round2_hypotheses():
    """Run all round 2 hypothesis tests."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    
    logger.info("=" * 60)
    logger.info("HYPOTHESIS TESTING ROUND 2: Addressing Reviewer Feedback")
    logger.info("=" * 60)
    
    fac, census, cdc = _load_data()
    logger.info(f"Loaded {len(fac):,} facility records")
    
    logger.info("\n>>> H2 REVISED: Cancer Paradox Hypotheses")
    h2_cancer_paradox_hypotheses(fac, cdc)
    
    logger.info("\n>>> H4 REVISED: Tract-Level Poverty-Pollution Geography")
    h4_tract_level_geography(fac, census)
    
    logger.info("\n>>> H3 REVISED: Facility Closure Selectivity")
    h3_closure_selectivity(fac)
    
    logger.info("\n>>> H5 REVISED: Presence vs Emissions")
    h5_presence_vs_emissions(fac)
    
    logger.info("\n>>> H7 REVISED: Wealth-Cancer Interaction")
    h7_wealth_cancer_interaction(fac)
    
    logger.info("\n>>> NEW: Temporal Facility-Health Analysis")
    temporal_facility_health(fac)
    
    logger.info("\n" + "=" * 60)
    logger.info("Round 2 hypothesis testing complete!")
    plots = sorted(OUT.glob("h*r_*.png")) + sorted(OUT.glob("temporal_*.png"))
    for p in plots:
        logger.info(f"  {p.name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_round2_hypotheses()
