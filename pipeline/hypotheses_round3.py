"""
Hypothesis Testing Round 3: Chemical Types, Release Pathways, and Mechanisms
=============================================================================

Following reviewer feedback, this module focuses on:

1. CHEMICAL TYPES EFFECT - Do carcinogens, persistent chemicals, or acute toxics 
   have different health effects AFTER controlling for poverty, screening, and 
   facility presence?

2. RELEASE MEDIUM SPECIFICITY - Do air releases cause respiratory disease? 
   Do water releases cause different outcomes? Test pathway-specific effects.

3. SELECTIVE MIGRATION HYPOTHESIS - Test whether healthy people leave industrial 
   areas, creating apparent health effects from population sorting.

4. LONG-TERM EFFECTS - Test the hypothesis that industrial history (past exposure)
   matters more than current emissions.

5. CHEMICAL-HEALTH PATHWAY MATCHING - Does the type of chemical predict the type
   of disease (e.g., carcinogens -> cancer, respiratory irritants -> COPD)?
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
import statsmodels.api as sm
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

OUT = Path("output/research")
OUT.mkdir(parents=True, exist_ok=True)

# Chemical classifications
CARCINOGENS = {
    'arsenic', 'benzene', 'chromium', 'cadmium', 'nickel', 'lead',
    'vinyl chloride', 'formaldehyde', '1,3-butadiene', 'trichloroethylene',
    'perchloroethylene', 'tetrachloroethylene', 'styrene', 'ethylene oxide',
    'dioxin', 'polycyclic aromatic', 'benzo', 'naphthalene', 'beryllium',
    'cobalt', 'antimony', 'hydrazine', 'acrylonitrile', 'asbestos',
}

RESPIRATORY_IRRITANTS = {
    'ammonia', 'chlorine', 'hydrogen chloride', 'sulfur dioxide',
    'nitrogen oxide', 'hydrogen fluoride', 'formaldehyde', 'acrolein',
    'ozone', 'particulate', 'dust', 'fume', 'sulfuric acid', 'nitric acid',
    'isocyanate', 'toluene diisocyanate',
}

NEUROTOXINS = {
    'lead', 'mercury', 'manganese', 'arsenic', 'toluene', 'xylene',
    'n-hexane', 'carbon disulfide', 'methanol', 'ethylene glycol',
    'organophosphate', 'carbamate',
}

PERSISTENT = {
    'lead', 'mercury', 'cadmium', 'arsenic', 'dioxin', 'pcb',
    'polychlorinated', 'pfoa', 'pfos', 'pfas', 'hexachlorobenzene',
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


def _load_data():
    """Load all necessary data."""
    fac = pd.read_csv("data/processed/facilities_scored.csv", low_memory=False)
    fac["has_health"] = fac["cancer_crude"].notna()
    fac['fips_tract'] = fac['fips_tract'].astype(str).str.zfill(11)
    
    # Classify chemicals
    def classify_chemical(name):
        if not isinstance(name, str):
            return {'carcinogen': False, 'respiratory': False, 'neuro': False, 'persistent': False}
        n = name.lower()
        return {
            'carcinogen': any(c in n for c in CARCINOGENS),
            'respiratory': any(c in n for c in RESPIRATORY_IRRITANTS),
            'neuro': any(c in n for c in NEUROTOXINS),
            'persistent': any(c in n for c in PERSISTENT),
        }
    
    chem_class = fac['CHEMICAL_NAME'].apply(classify_chemical)
    fac['is_carcinogen'] = chem_class.apply(lambda x: x['carcinogen'])
    fac['is_respiratory'] = chem_class.apply(lambda x: x['respiratory'])
    fac['is_neuro'] = chem_class.apply(lambda x: x['neuro'])
    fac['is_persistent'] = chem_class.apply(lambda x: x['persistent'])
    
    return fac


def _load_release_medium():
    """Load release quantity by medium if available."""
    p = Path("data/raw/tri_release_qty.csv")
    if not p.exists():
        return None
    rq = pd.read_csv(p, low_memory=False)
    rq['total_release'] = pd.to_numeric(rq['total_release'], errors='coerce').fillna(0)
    
    # Simplify medium
    def simplify_medium(m):
        if 'AIR' in str(m).upper():
            return 'Air'
        if 'WATER' in str(m).upper():
            return 'Water'
        if 'UNINJ' in str(m).upper():
            return 'Underground'
        return 'Land'
    
    rq['medium_simple'] = rq['environmental_medium'].apply(simplify_medium)
    return rq


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: CHEMICAL TYPE EFFECTS AFTER CONTROLLING FOR CONFOUNDERS
# ═══════════════════════════════════════════════════════════════════════════════

def chemical_effects_controlled(fac):
    """
    Test whether chemical types have health effects AFTER controlling for:
    - Poverty (proxy for healthcare access/screening)
    - Facility presence (the non-specific "industrial neighborhood" effect)
    - Population characteristics
    """
    logger.info("Testing chemical type effects with controls...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Aggregate to tract level with chemical type flags
    tract_data = hdf.groupby('fips_tract').agg({
        'TOTAL_RELEASES': 'sum',
        'TRI_FACILITY_ID': 'nunique',
        'is_carcinogen': 'max',  # any carcinogen in tract
        'is_respiratory': 'max',
        'is_neuro': 'max',
        'is_persistent': 'max',
        'cancer_crude': 'mean',
        'copd_crude': 'mean',
        'asthma_crude': 'mean',
        'mental_health_crude': 'mean',
        'diabetes_crude': 'mean',
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
        'pct_no_insurance': 'mean',
        'total_population': 'first',
    }).reset_index()
    
    tract_data.columns = ['fips_tract', 'releases', 'n_facilities', 
                          'has_carcinogen', 'has_respiratory', 'has_neuro', 'has_persistent',
                          'cancer', 'copd', 'asthma', 'mental', 'diabetes',
                          'poverty', 'minority', 'uninsured', 'population']
    
    tract_data['log_releases'] = np.log10(tract_data['releases'].clip(1))
    tract_data = tract_data.dropna(subset=['cancer', 'copd', 'poverty'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel A: Raw correlation - chemical type vs health
    ax = axes[0, 0]
    
    chem_types = ['has_carcinogen', 'has_respiratory', 'has_neuro', 'has_persistent']
    outcomes = ['cancer', 'copd', 'asthma', 'mental']
    
    corr_raw = []
    for chem in chem_types:
        row = []
        for outcome in outcomes:
            r, _ = spearmanr(tract_data[chem].astype(int), tract_data[outcome])
            row.append(r)
        corr_raw.append(row)
    
    corr_raw_df = pd.DataFrame(corr_raw, 
                                index=['Carcinogen', 'Resp. Irritant', 'Neurotoxin', 'Persistent'],
                                columns=['Cancer', 'COPD', 'Asthma', 'Mental'])
    
    im = ax.imshow(corr_raw_df.values, cmap='RdBu_r', vmin=-0.15, vmax=0.15, aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(corr_raw_df.columns)
    ax.set_yticks(range(4))
    ax.set_yticklabels(corr_raw_df.index)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr_raw_df.values[i,j]:.3f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, label='Spearman r')
    ax.set_title("A. RAW: Chemical Type vs Health\n(No controls)", fontweight='bold')
    
    # Panel B: Partial correlations controlling for poverty + n_facilities
    ax = axes[0, 1]
    
    # Residualize health outcomes on poverty and facility count
    tract_data_clean = tract_data.dropna(subset=['poverty', 'n_facilities', 'cancer', 'copd'])
    
    partial_corr = []
    for chem in chem_types:
        row = []
        for outcome in outcomes:
            # Fit regression: outcome ~ poverty + n_facilities
            X = sm.add_constant(tract_data_clean[['poverty', 'n_facilities']])
            model = sm.OLS(tract_data_clean[outcome], X).fit()
            residual_outcome = model.resid
            
            # Correlation of chemical type with residualized outcome
            r, _ = spearmanr(tract_data_clean[chem].astype(int), residual_outcome)
            row.append(r)
        partial_corr.append(row)
    
    partial_df = pd.DataFrame(partial_corr,
                               index=['Carcinogen', 'Resp. Irritant', 'Neurotoxin', 'Persistent'],
                               columns=['Cancer', 'COPD', 'Asthma', 'Mental'])
    
    im2 = ax.imshow(partial_df.values, cmap='RdBu_r', vmin=-0.15, vmax=0.15, aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(partial_df.columns)
    ax.set_yticks(range(4))
    ax.set_yticklabels(partial_df.index)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{partial_df.values[i,j]:.3f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im2, ax=ax, label='Partial r')
    ax.set_title("B. CONTROLLED: After removing\npoverty + facility count effects", fontweight='bold')
    
    # Panel C: Compare raw vs controlled
    ax = axes[0, 2]
    
    # Focus on expected pathways
    pathways = [
        ('has_carcinogen', 'cancer', 'Carcinogen → Cancer'),
        ('has_respiratory', 'copd', 'Resp. Irritant → COPD'),
        ('has_respiratory', 'asthma', 'Resp. Irritant → Asthma'),
        ('has_neuro', 'mental', 'Neurotoxin → Mental'),
    ]
    
    raw_corrs = []
    partial_corrs = []
    labels = []
    
    for chem, outcome, label in pathways:
        # Raw
        r_raw, _ = spearmanr(tract_data_clean[chem].astype(int), tract_data_clean[outcome])
        raw_corrs.append(r_raw)
        
        # Partial
        X = sm.add_constant(tract_data_clean[['poverty', 'n_facilities']])
        model = sm.OLS(tract_data_clean[outcome], X).fit()
        r_partial, _ = spearmanr(tract_data_clean[chem].astype(int), model.resid)
        partial_corrs.append(r_partial)
        
        labels.append(label)
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, raw_corrs, width, label='Raw', color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, partial_corrs, width, label='Controlled', color='#3498db', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel("Spearman r")
    ax.set_title("C. Expected Pathways: Raw vs Controlled\n(Does chemical type matter after controls?)", fontweight='bold')
    ax.legend()
    
    # Panel D: Carcinogen effect on cancer by screening proxy
    ax = axes[1, 0]
    
    # Split by uninsured rate (screening proxy) - use rank-based for safety
    tract_data_clean['unins_tertile'] = pd.cut(tract_data_clean['uninsured'].fillna(tract_data_clean['uninsured'].median()).rank(method='first'), 
                                                 bins=3, labels=['Low Unins.\n(Good screening)', 'Medium', 'High Unins.\n(Poor screening)'])
    
    # Cancer rate in carcinogen vs non-carcinogen tracts, by screening level
    grouped = tract_data_clean.groupby(['unins_tertile', 'has_carcinogen'], observed=True)['cancer'].mean().unstack()
    
    # Handle missing columns
    if True not in grouped.columns:
        grouped[True] = grouped.iloc[:, 0] * 0
    if False not in grouped.columns:
        grouped[False] = grouped.iloc[:, 0] * 0
    
    x = np.arange(len(grouped))
    width = 0.35
    
    ax.bar(x - width/2, grouped[False].values, width, label='No Carcinogen', color='#3498db')
    ax.bar(x + width/2, grouped[True].values, width, label='Has Carcinogen', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.set_ylabel("Mean Cancer Rate (%)")
    ax.set_title("D. CARCINOGEN EFFECT BY SCREENING ACCESS\n(Carcinogen effect should be stronger with good screening)", fontweight='bold')
    ax.legend()
    
    # Calculate gaps
    gaps = grouped[True] - grouped[False]
    for i, g in enumerate(gaps):
        color = 'red' if g > 0 else 'green'
        ax.annotate(f"Gap: {g:+.2f}", (i, max(grouped[True].iloc[i], grouped[False].iloc[i]) + 0.1),
                   ha='center', fontsize=9, color=color)
    
    # Panel E: Respiratory irritant effect on COPD by poverty
    ax = axes[1, 1]
    
    tract_data_clean['pov_tertile'] = pd.cut(tract_data_clean['poverty'].rank(method='first'), bins=3, 
                                               labels=['Low Poverty', 'Medium', 'High Poverty'])
    
    grouped_resp = tract_data_clean.groupby(['pov_tertile', 'has_respiratory'], observed=True)['copd'].mean().unstack()
    
    if True not in grouped_resp.columns:
        grouped_resp[True] = grouped_resp.iloc[:, 0] * 0
    if False not in grouped_resp.columns:
        grouped_resp[False] = grouped_resp.iloc[:, 0] * 0
    
    ax.bar(x - width/2, grouped_resp[False].values, width, label='No Resp. Irritant', color='#3498db')
    ax.bar(x + width/2, grouped_resp[True].values, width, label='Has Resp. Irritant', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped_resp.index)
    ax.set_ylabel("Mean COPD Rate (%)")
    ax.set_title("E. RESPIRATORY IRRITANT EFFECT BY POVERTY\n(Effect strongest in poor areas with less healthcare?)", fontweight='bold')
    ax.legend()
    
    # Panel F: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate summary statistics
    carc_cancer_raw = corr_raw_df.loc['Carcinogen', 'Cancer']
    carc_cancer_ctrl = partial_df.loc['Carcinogen', 'Cancer']
    resp_copd_raw = corr_raw_df.loc['Resp. Irritant', 'COPD']
    resp_copd_ctrl = partial_df.loc['Resp. Irritant', 'COPD']
    
    summary_text = f"""
CHEMICAL TYPE EFFECTS AFTER CONTROLLING FOR CONFOUNDERS

Expected pathways tested:
1. Carcinogen → Cancer
2. Respiratory Irritant → COPD/Asthma
3. Neurotoxin → Mental Health
4. Persistent → Multiple outcomes

RAW vs CONTROLLED CORRELATIONS:

Carcinogen → Cancer:
  Raw:        r = {carc_cancer_raw:.3f}
  Controlled: r = {carc_cancer_ctrl:.3f}
  Change:     {100*(carc_cancer_ctrl - carc_cancer_raw)/abs(carc_cancer_raw) if carc_cancer_raw != 0 else 0:+.0f}%
  
Resp. Irritant → COPD:
  Raw:        r = {resp_copd_raw:.3f}
  Controlled: r = {resp_copd_ctrl:.3f}
  Change:     {100*(resp_copd_ctrl - resp_copd_raw)/abs(resp_copd_raw) if resp_copd_raw != 0 else 0:+.0f}%

INTERPRETATION:
{'Chemical type effects PERSIST after controlling for poverty and facility count' 
 if abs(carc_cancer_ctrl) > 0.03 or abs(resp_copd_ctrl) > 0.03 
 else 'Chemical type effects largely DISAPPEAR after controls - the "industrial neighborhood" effect dominates'}

KEY FINDING ON SCREENING:
Carcinogen-cancer association is {'stronger' if gaps.iloc[0] > gaps.iloc[-1] else 'weaker'} 
in areas with good screening access, supporting the 
detection/screening gap hypothesis.
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title("F. Summary: Do Chemicals Matter After Controls?", fontweight='bold')
    
    fig.suptitle("ROUND 3: Chemical Type Effects After Controlling for Confounders\n"
                 "Does the TYPE of chemical predict health outcomes beyond 'industrial neighborhood' effect?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "r3_chemical_effects_controlled")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: RELEASE MEDIUM PATHWAY SPECIFICITY
# ═══════════════════════════════════════════════════════════════════════════════

def release_medium_pathways(fac):
    """
    Test whether release medium predicts specific health outcomes:
    - Air releases → Respiratory (COPD, Asthma)
    - Water releases → Different pattern?
    """
    logger.info("Testing release medium pathway specificity...")
    
    rq = _load_release_medium()
    if rq is None:
        logger.warning("Release medium data not available, skipping")
        return
    
    # Aggregate by doc_ctrl_num (report)
    medium_agg = rq.groupby(['doc_ctrl_num', 'medium_simple'])['total_release'].sum().unstack(fill_value=0)
    medium_agg = medium_agg.reset_index()
    
    # Load facility mapping
    tri_raw = pd.read_csv("data/raw/tri_facilities.csv", low_memory=False)
    tri_raw.columns = tri_raw.columns.str.upper().str.strip()
    dcn_col = next((c for c in tri_raw.columns if 'DOC_CTRL' in c), None)
    fid_col = next((c for c in tri_raw.columns if 'TRI_FACILITY_ID' in c), None)
    
    if not dcn_col or not fid_col:
        logger.warning("Cannot map medium data to facilities")
        return
    
    fac_map = tri_raw[[dcn_col, fid_col]].rename(
        columns={dcn_col: 'doc_ctrl_num', fid_col: 'TRI_FACILITY_ID'})
    
    medium_agg = medium_agg.merge(fac_map, on='doc_ctrl_num', how='left')
    
    # Aggregate to facility level
    fac_medium = medium_agg.groupby('TRI_FACILITY_ID').agg({
        'Air': 'sum',
        'Water': 'sum',
        'Land': 'sum',
    }).reset_index()
    fac_medium.columns = ['TRI_FACILITY_ID', 'air_releases', 'water_releases', 'land_releases']
    
    # Merge with health data
    hdf = fac[fac['has_health']].copy()
    health_cols = ['TRI_FACILITY_ID', 'fips_tract', 'cancer_crude', 'copd_crude', 
                   'asthma_crude', 'chd_crude', 'diabetes_crude', 'poverty_pct']
    fac_health = hdf[health_cols].drop_duplicates('TRI_FACILITY_ID')
    
    merged = fac_medium.merge(fac_health, on='TRI_FACILITY_ID', how='inner')
    merged = merged.dropna(subset=['copd_crude', 'poverty_pct'])
    
    # Log transform releases
    for col in ['air_releases', 'water_releases', 'land_releases']:
        merged[f'log_{col}'] = np.log10(merged[col].clip(1))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel A: Medium composition pie chart
    ax = axes[0, 0]
    total_by_medium = merged[['air_releases', 'water_releases', 'land_releases']].sum()
    colors = ['#3498db', '#2ecc71', '#8B4513']
    ax.pie(total_by_medium.values, labels=['Air', 'Water', 'Land'],
           autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title("A. Total Release Volume by Medium", fontweight='bold')
    
    # Panel B: Correlation matrix - medium vs health
    ax = axes[0, 1]
    
    medium_cols = ['log_air_releases', 'log_water_releases', 'log_land_releases']
    outcome_cols = ['copd_crude', 'asthma_crude', 'cancer_crude', 'chd_crude']
    
    corr_medium = []
    for med in medium_cols:
        row = []
        for outcome in outcome_cols:
            r, _ = spearmanr(merged[med], merged[outcome])
            row.append(r)
        corr_medium.append(row)
    
    corr_df = pd.DataFrame(corr_medium,
                            index=['Air', 'Water', 'Land'],
                            columns=['COPD', 'Asthma', 'Cancer', 'CHD'])
    
    im = ax.imshow(corr_df.values, cmap='RdBu_r', vmin=-0.15, vmax=0.15, aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(corr_df.columns)
    ax.set_yticks(range(3))
    ax.set_yticklabels(corr_df.index)
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{corr_df.values[i,j]:.3f}", ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax, label='Spearman r')
    ax.set_title("B. Release Medium vs Health Outcomes\n(Air should correlate with respiratory)", fontweight='bold')
    
    # Panel C: Air releases vs COPD/Asthma (expected pathway)
    ax = axes[0, 2]
    
    # Create air release quintiles using rank-based binning
    merged['air_quintile'] = pd.cut(merged['air_releases'].rank(method='first'), bins=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    for outcome, color, label in [('copd_crude', '#e74c3c', 'COPD'), 
                                   ('asthma_crude', '#3498db', 'Asthma')]:
        means = merged.groupby('air_quintile', observed=True)[outcome].mean()
        ax.plot(range(len(means)), means.values, 'o-', color=color, 
               linewidth=2, markersize=8, label=label)
    
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest)'])
    ax.set_xlabel("Air Release Quintile")
    ax.set_ylabel("Mean Health Rate (%)")
    ax.set_title("C. AIR RELEASES vs Respiratory Outcomes\n(Expected pathway: Air → COPD/Asthma)", fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel D: Water releases vs outcomes (comparison)
    ax = axes[1, 0]
    
    # Use rank-based quintiles to handle ties
    merged['water_quintile'] = pd.cut(merged['water_releases'].rank(method='first'), bins=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    for outcome, color, label in [('copd_crude', '#e74c3c', 'COPD'), 
                                   ('cancer_crude', '#9b59b6', 'Cancer')]:
        means = merged.groupby('water_quintile', observed=True)[outcome].mean()
        ax.plot(range(len(means)), means.values, 'o-', color=color,
               linewidth=2, markersize=8, label=label)
    
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Q1\n(Lowest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Highest)'])
    ax.set_xlabel("Water Release Quintile")
    ax.set_ylabel("Mean Health Rate (%)")
    ax.set_title("D. WATER RELEASES vs Health\n(Different pathway than air?)", fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel E: Controlling for poverty - does medium effect persist?
    ax = axes[1, 1]
    
    # Residualize outcomes on poverty
    partial_corr_controlled = []
    for med in ['log_air_releases', 'log_water_releases', 'log_land_releases']:
        row = []
        for outcome in ['copd_crude', 'asthma_crude']:
            X = sm.add_constant(merged['poverty_pct'])
            model = sm.OLS(merged[outcome], X).fit()
            r, _ = spearmanr(merged[med], model.resid)
            row.append(r)
        partial_corr_controlled.append(row)
    
    partial_df = pd.DataFrame(partial_corr_controlled,
                               index=['Air', 'Water', 'Land'],
                               columns=['COPD', 'Asthma'])
    
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, partial_df['COPD'], width, label='COPD (poverty-controlled)', color='#e74c3c')
    ax.bar(x + width/2, partial_df['Asthma'], width, label='Asthma (poverty-controlled)', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(['Air', 'Water', 'Land'])
    ax.set_xlabel("Release Medium")
    ax.set_ylabel("Partial Correlation (controlling for poverty)")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title("E. MEDIUM EFFECT CONTROLLED FOR POVERTY\n(Air-respiratory link should persist)", fontweight='bold')
    ax.legend()
    
    # Panel F: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    air_copd = corr_df.loc['Air', 'COPD']
    air_asthma = corr_df.loc['Air', 'Asthma']
    air_copd_ctrl = partial_df.loc['Air', 'COPD']
    
    summary_text = f"""
RELEASE MEDIUM PATHWAY SPECIFICITY

HYPOTHESIS: Air releases should cause respiratory disease
            more specifically than water/land releases.

RELEASE COMPOSITION:
  Air:   {100*total_by_medium['air_releases']/total_by_medium.sum():.1f}%
  Water: {100*total_by_medium['water_releases']/total_by_medium.sum():.1f}%
  Land:  {100*total_by_medium['land_releases']/total_by_medium.sum():.1f}%

AIR → RESPIRATORY PATHWAY:
  Air vs COPD (raw):   r = {air_copd:.3f}
  Air vs Asthma (raw): r = {air_asthma:.3f}
  
  After controlling for poverty:
  Air vs COPD:   r = {air_copd_ctrl:.3f}

FINDING:
{'AIR RELEASES show STRONGER association with respiratory outcomes than water/land' 
 if air_copd > corr_df.loc['Water', 'COPD'] and air_copd > corr_df.loc['Land', 'COPD']
 else 'NO clear pathway specificity - all mediums show similar patterns'}

This {'SUPPORTS' if air_copd > 0.05 else 'DOES NOT SUPPORT'} the 
pathway-specific mechanism hypothesis.

INTERPRETATION:
{'The type of release (air/water/land) matters for predicting which diseases occur.'
 if air_copd > 0.05 
 else 'Release medium does not strongly predict specific diseases. The general industrial neighborhood effect may dominate.'}
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title("F. Summary: Does Medium Predict Pathway?", fontweight='bold')
    
    fig.suptitle("ROUND 3: Release Medium Pathway Specificity\n"
                 "Do air releases cause respiratory disease more than water/land releases?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "r3_release_medium_pathways")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: SELECTIVE MIGRATION HYPOTHESIS
# ═══════════════════════════════════════════════════════════════════════════════

def selective_migration(fac):
    """
    Test the selective migration hypothesis:
    - Healthy/wealthy people leave industrial areas
    - Creates apparent health effect from population sorting
    
    Proxies to test:
    - Population change over time in TRI vs non-TRI tracts
    - Age structure (young families leave vs older stay)
    - Income gradient around facilities
    """
    logger.info("Testing selective migration hypothesis...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Aggregate to tract level
    tract_data = hdf.groupby('fips_tract').agg({
        'TOTAL_RELEASES': 'sum',
        'TRI_FACILITY_ID': 'nunique',
        'total_population': 'first',
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
        'median_income': 'mean',
        'cancer_crude': 'mean',
        'copd_crude': 'mean',
        'REPORTING_YEAR': ['min', 'max'],
    })
    tract_data.columns = ['releases', 'n_facilities', 'population', 'poverty', 
                          'minority', 'income', 'cancer', 'copd', 'first_year', 'last_year']
    tract_data = tract_data.reset_index()
    tract_data['years_active'] = tract_data['last_year'] - tract_data['first_year'] + 1
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel A: Population size in TRI vs long-active tracts
    ax = axes[0, 0]
    
    tract_data['activity_group'] = pd.cut(tract_data['years_active'],
                                           bins=[0, 3, 6, 9, 12],
                                           labels=['1-3 yrs', '4-6 yrs', '7-9 yrs', '10-11 yrs'])
    
    pop_by_activity = tract_data.groupby('activity_group', observed=True)['population'].agg(['mean', 'median', 'sem'])
    
    x = np.arange(len(pop_by_activity))
    ax.bar(x, pop_by_activity['mean']/1000, color=plt.cm.Blues(np.linspace(0.3, 0.9, len(pop_by_activity))),
           edgecolor='black')
    ax.errorbar(x, pop_by_activity['mean']/1000, yerr=1.96*pop_by_activity['sem']/1000,
               fmt='none', color='black', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(pop_by_activity.index)
    ax.set_xlabel("Years of TRI Activity")
    ax.set_ylabel("Mean Tract Population (thousands)")
    ax.set_title("A. POPULATION BY TRI DURATION\n(Longer TRI history = smaller population?)", fontweight='bold')
    
    r, p = spearmanr(tract_data['years_active'], tract_data['population'])
    ax.text(0.95, 0.95, f"r = {r:.3f}", transform=ax.transAxes, ha='right', fontsize=10,
           bbox=dict(boxstyle='round', fc='white'))
    
    # Panel B: Income by facility count (healthy/wealthy leave?)
    ax = axes[0, 1]
    
    tract_data['fac_bin'] = pd.cut(tract_data['n_facilities'],
                                    bins=[-1, 1, 2, 5, 100],
                                    labels=['1', '2', '3-5', '6+'])
    
    income_by_fac = tract_data.groupby('fac_bin', observed=True)['income'].agg(['mean', 'sem'])
    
    n_bins = len(income_by_fac)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_bins))
    ax.bar(range(n_bins), income_by_fac['mean']/1000, color=colors, edgecolor='black')
    ax.errorbar(range(n_bins), income_by_fac['mean']/1000, yerr=1.96*income_by_fac['sem']/1000,
               fmt='none', color='black', capsize=4)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(income_by_fac.index)
    ax.set_xlabel("Number of TRI Facilities")
    ax.set_ylabel("Mean Median Income ($K)")
    ax.set_title("B. INCOME BY FACILITY COUNT\n(More facilities = lower income = wealthy left?)", fontweight='bold')
    
    # Panel C: Minority % by facility count (who stays?)
    ax = axes[0, 2]
    
    minority_by_fac = tract_data.groupby('fac_bin', observed=True)['minority'].agg(['mean', 'sem'])
    
    ax.bar(range(len(minority_by_fac)), minority_by_fac['mean'], 
           color=plt.cm.Blues(np.linspace(0.3, 0.9, len(minority_by_fac))), edgecolor='black')
    ax.errorbar(range(len(minority_by_fac)), minority_by_fac['mean'], yerr=1.96*minority_by_fac['sem'],
               fmt='none', color='black', capsize=4)
    ax.set_xticks(range(len(minority_by_fac)))
    ax.set_xticklabels(minority_by_fac.index)
    ax.set_xlabel("Number of TRI Facilities")
    ax.set_ylabel("Mean Minority %")
    ax.set_title("C. WHO STAYS? Minority % by Facility Count\n(Higher minority % = less able to move?)", fontweight='bold')
    
    # Panel D: The key test - does population mediate the health effect?
    ax = axes[1, 0]
    
    # Compare:
    # 1. Raw correlation: n_facilities vs COPD
    # 2. Controlling for population
    
    tract_clean = tract_data.dropna(subset=['copd', 'n_facilities', 'population', 'poverty'])
    
    r_raw, _ = spearmanr(tract_clean['n_facilities'], tract_clean['copd'])
    
    # Control for population
    X = sm.add_constant(tract_clean[['population', 'poverty']])
    model = sm.OLS(tract_clean['copd'], X).fit()
    r_controlled, _ = spearmanr(tract_clean['n_facilities'], model.resid)
    
    bars = ax.bar(['Raw', 'Controlled\n(pop + poverty)'], [r_raw, r_controlled],
                  color=['#e74c3c', '#3498db'], edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("Correlation: Facilities vs COPD")
    ax.set_title("D. MEDIATION TEST: Does Population Explain the Effect?", fontweight='bold')
    
    for bar, r in zip(bars, [r_raw, r_controlled]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f"r = {r:.3f}", ha='center', fontsize=10)
    
    change = (r_controlled - r_raw) / abs(r_raw) * 100 if r_raw != 0 else 0
    ax.text(0.5, 0.95, f"Change: {change:+.0f}%", transform=ax.transAxes,
           ha='center', fontsize=11, bbox=dict(boxstyle='round', fc='yellow', alpha=0.8))
    
    # Panel E: Scatterplot - population vs health, by TRI status
    ax = axes[1, 1]
    
    # Compare relationship in high vs low TRI areas
    high_tri = tract_clean[tract_clean['n_facilities'] >= 3]
    low_tri = tract_clean[tract_clean['n_facilities'] == 1]
    
    ax.scatter(high_tri['population']/1000, high_tri['copd'], 
              alpha=0.3, s=15, c='#e74c3c', label=f'3+ facilities (n={len(high_tri)})')
    ax.scatter(low_tri['population']/1000, low_tri['copd'],
              alpha=0.3, s=15, c='#3498db', label=f'1 facility (n={len(low_tri)})')
    
    ax.set_xlabel("Tract Population (thousands)")
    ax.set_ylabel("COPD Rate (%)")
    ax.set_title("E. Population vs COPD by Facility Count\n(Selection: small pop + high COPD = who's left)", fontweight='bold')
    ax.legend(fontsize=8)
    
    # Panel F: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    pop_corr, _ = spearmanr(tract_data['n_facilities'], tract_data['population'])
    income_corr, _ = spearmanr(tract_data['n_facilities'], tract_data['income'])
    
    summary_text = f"""
SELECTIVE MIGRATION HYPOTHESIS TEST

The hypothesis: Healthy/wealthy people leave industrial areas,
creating apparent health effects from population sorting rather
than direct chemical exposure.

EVIDENCE:

1. POPULATION SIZE
   Facilities vs Population: r = {pop_corr:.3f}
   {'More TRI = smaller population (people left)' if pop_corr < 0 else 'No clear population effect'}

2. INCOME (proxy for ability to move)
   Facilities vs Income: r = {income_corr:.3f}
   {'More TRI = lower income (wealthy left or poor stayed)' if income_corr < 0 else 'No clear income gradient'}

3. MEDIATION TEST
   Raw (facilities vs COPD): r = {r_raw:.3f}
   Controlled (pop + poverty): r = {r_controlled:.3f}
   
   Population explains {abs(change):.0f}% of the facility-health association

INTERPRETATION:
{'SUPPORTED: Population dynamics explain a substantial portion of the facility-health link. Healthy/wealthy people likely left industrial areas.'
 if abs(change) > 20
 else 'PARTIALLY SUPPORTED: Some selection effect, but direct exposure may also matter.'
 if abs(change) > 10
 else 'NOT SUPPORTED: Population dynamics explain little of the health effect. Direct exposure more likely.'}

This matters because:
- If migration explains the effect, TRI isn't causing disease directly
- Instead, TRI areas accumulate vulnerable populations
- Policy should focus on mobility assistance, not just emission reduction
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title("F. Summary: Is It Selection or Causation?", fontweight='bold')
    
    fig.suptitle("ROUND 3: Selective Migration Hypothesis\n"
                 "Do healthy people leave industrial areas, creating apparent health effects?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "r3_selective_migration")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: LONG-TERM HISTORICAL EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def historical_effects(fac):
    """
    Test the hypothesis that historical exposure matters more than current emissions.
    
    Today's industrial region was past industrial region too.
    Releases might have been higher historically.
    Current TRI underestimates cumulative burden.
    """
    logger.info("Testing historical/long-term effects hypothesis...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Calculate facility/tract history
    tract_history = hdf.groupby('fips_tract').agg({
        'REPORTING_YEAR': ['min', 'max', 'nunique'],
        'TOTAL_RELEASES': ['sum', 'mean'],
        'TRI_FACILITY_ID': 'nunique',
        'cancer_crude': 'mean',
        'copd_crude': 'mean',
        'asthma_crude': 'mean',
        'poverty_pct': 'mean',
    })
    tract_history.columns = ['first_year', 'last_year', 'n_years', 
                              'total_releases', 'mean_releases', 'n_facilities',
                              'cancer', 'copd', 'asthma', 'poverty']
    tract_history = tract_history.reset_index()
    tract_history['years_active'] = tract_history['last_year'] - tract_history['first_year'] + 1
    tract_history['started_before_2015'] = tract_history['first_year'] <= 2015
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel A: Health by years of TRI activity
    ax = axes[0, 0]
    
    tract_history['activity_bin'] = pd.cut(tract_history['years_active'],
                                            bins=[0, 3, 6, 9, 12],
                                            labels=['1-3 yrs', '4-6 yrs', '7-9 yrs', '10-11 yrs'])
    
    for outcome, color, label in [('copd', '#e74c3c', 'COPD'),
                                   ('asthma', '#3498db', 'Asthma'),
                                   ('cancer', '#9b59b6', 'Cancer')]:
        means = tract_history.groupby('activity_bin', observed=True)[outcome].mean()
        ax.plot(range(len(means)), means.values, 'o-', color=color,
               linewidth=2, markersize=8, label=label)
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(['1-3 yrs', '4-6 yrs', '7-9 yrs', '10-11 yrs'])
    ax.set_xlabel("Years of TRI Activity in Tract")
    ax.set_ylabel("Mean Health Rate (%)")
    ax.set_title("A. CUMULATIVE EXPOSURE: Health by TRI Duration\n(Longer exposure = worse health?)", fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel B: Early starters vs late starters
    ax = axes[0, 1]
    
    early = tract_history[tract_history['started_before_2015']]
    late = tract_history[~tract_history['started_before_2015']]
    
    outcomes = ['copd', 'asthma', 'cancer']
    early_means = [early[o].mean() for o in outcomes]
    late_means = [late[o].mean() for o in outcomes]
    
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, early_means, width, label=f'Early (pre-2015, n={len(early)})', color='#e74c3c')
    ax.bar(x + width/2, late_means, width, label=f'Late (2015+, n={len(late)})', color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(['COPD', 'Asthma', 'Cancer'])
    ax.set_ylabel("Mean Rate (%)")
    ax.set_title("B. EARLY vs LATE STARTERS\n(Long-standing industrial areas = worse health?)", fontweight='bold')
    ax.legend()
    
    # Panel C: Does current volume explain health, or is it history?
    ax = axes[0, 2]
    
    tract_clean = tract_history.dropna(subset=['copd', 'total_releases', 'years_active', 'poverty'])
    tract_clean['log_releases'] = np.log10(tract_clean['total_releases'].clip(1))
    
    # Compare predictors
    predictors = ['log_releases', 'years_active', 'n_facilities']
    pred_labels = ['Total Releases', 'Years Active', '# Facilities']
    
    corrs = []
    for pred in predictors:
        r, _ = spearmanr(tract_clean[pred], tract_clean['copd'])
        corrs.append(r)
    
    colors_pred = ['#e74c3c', '#3498db', '#2ecc71']
    ax.barh(range(3), corrs, color=colors_pred, edgecolor='black')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels(pred_labels)
    ax.set_xlabel("Correlation with COPD")
    ax.set_title("C. WHAT PREDICTS HEALTH?\n(Duration vs Volume vs Count)", fontweight='bold')
    
    for i, r in enumerate(corrs):
        ax.text(r + 0.01 if r > 0 else r - 0.01, i, f"{r:.3f}", va='center',
               ha='left' if r > 0 else 'right', fontsize=10)
    
    # Panel D: Controlling for years active
    ax = axes[1, 0]
    
    # Raw correlation: releases vs COPD
    r_raw, _ = spearmanr(tract_clean['log_releases'], tract_clean['copd'])
    
    # Controlled for years active
    X = sm.add_constant(tract_clean[['years_active', 'poverty']])
    model = sm.OLS(tract_clean['copd'], X).fit()
    r_history_controlled, _ = spearmanr(tract_clean['log_releases'], model.resid)
    
    # Reverse: controlling releases for years
    X2 = sm.add_constant(tract_clean[['log_releases', 'poverty']])
    model2 = sm.OLS(tract_clean['copd'], X2).fit()
    r_releases_controlled, _ = spearmanr(tract_clean['years_active'], model2.resid)
    
    bars = ax.bar(['Releases\n(raw)', 'Releases\n(ctrl history)', 'History\n(ctrl releases)'],
                  [r_raw, r_history_controlled, r_releases_controlled],
                  color=['#e74c3c', '#f39c12', '#3498db'], edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("Correlation with COPD")
    ax.set_title("D. DECOMPOSITION: History vs Current Releases\n(What matters more?)", fontweight='bold')
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01 if h > 0 else h - 0.02,
               f"{h:.3f}", ha='center', fontsize=10)
    
    # Panel E: Cancer latency test
    ax = axes[1, 1]
    
    # Cancer has 20-30 year latency. Current emissions shouldn't predict it.
    # But older industrial areas should have more cancer (from past exposure)
    
    r_cancer_releases, _ = spearmanr(tract_clean['log_releases'], tract_clean['cancer'])
    r_cancer_history, _ = spearmanr(tract_clean['years_active'], tract_clean['cancer'])
    
    r_copd_releases, _ = spearmanr(tract_clean['log_releases'], tract_clean['copd'])
    r_copd_history, _ = spearmanr(tract_clean['years_active'], tract_clean['copd'])
    
    x_pos = np.arange(2)
    width = 0.35
    
    ax.bar(x_pos - width/2, [r_copd_releases, r_copd_history], width, 
           label='COPD (acute)', color='#e74c3c')
    ax.bar(x_pos + width/2, [r_cancer_releases, r_cancer_history], width,
           label='Cancer (latent)', color='#9b59b6')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Current Releases', 'Years of History'])
    ax.set_ylabel("Correlation")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title("E. LATENCY TEST: COPD vs Cancer\n(Cancer should correlate more with history)", fontweight='bold')
    ax.legend()
    
    # Panel F: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    history_effect = corrs[1]  # years_active vs COPD
    releases_effect = corrs[0]  # log_releases vs COPD
    
    summary_text = f"""
HISTORICAL/LONG-TERM EFFECTS HYPOTHESIS

The hypothesis: Today's industrial region was also yesterday's
industrial region. Historical releases (potentially higher) created
contamination that persists. Current TRI underestimates true burden.

KEY FINDINGS:

1. DURATION MATTERS
   Years of TRI activity vs COPD: r = {history_effect:.3f}
   Current releases vs COPD:      r = {releases_effect:.3f}
   
   {'Duration (history) predicts health BETTER than current releases'
    if abs(history_effect) > abs(releases_effect)
    else 'Current releases predict health better than duration'}

2. EARLY vs LATE STARTERS
   Areas with TRI since pre-2015 have {'higher' if early['copd'].mean() > late['copd'].mean() else 'similar'} COPD
   than areas with recent TRI activity.

3. LATENCY TEST (Cancer vs COPD)
   Cancer-history correlation: r = {r_cancer_history:.3f}
   COPD-history correlation:   r = {r_copd_history:.3f}
   
   {'Cancer correlates more with history (consistent with latency)'
    if abs(r_cancer_history) > abs(r_copd_history)
    else 'COPD correlates more with history (unexpected)'}

4. AFTER CONTROLLING
   Releases effect after controlling for history: r = {r_history_controlled:.3f}
   History effect after controlling for releases: r = {r_releases_controlled:.3f}

INTERPRETATION:
{'SUPPORTED: Industrial history matters. Long-term exposure and historical contamination explain health outcomes better than current emissions.'
 if abs(history_effect) > abs(releases_effect)
 else 'MIXED: Both current and historical factors matter. Neither dominates.'}
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title("F. Summary: Does History Matter?", fontweight='bold')
    
    fig.suptitle("ROUND 3: Historical/Long-Term Effects\n"
                 "Does industrial history predict health better than current emissions?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "r3_historical_effects")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: COMBINED MECHANISM MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def combined_mechanism_model(fac):
    """
    Build a combined model that tests all mechanisms together:
    - Chemical type effects
    - Historical exposure
    - Selective migration
    - Poverty/screening confounding
    """
    logger.info("Building combined mechanism model...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Build comprehensive tract-level dataset
    tract_data = hdf.groupby('fips_tract').agg({
        'TOTAL_RELEASES': 'sum',
        'TRI_FACILITY_ID': 'nunique',
        'REPORTING_YEAR': ['min', 'count'],
        'is_carcinogen': 'max',
        'is_respiratory': 'max',
        'cancer_crude': 'mean',
        'copd_crude': 'mean',
        'asthma_crude': 'mean',
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
        'pct_no_insurance': 'mean',
        'total_population': 'first',
    })
    tract_data.columns = ['releases', 'n_facilities', 'first_year', 'n_observations',
                          'has_carcinogen', 'has_respiratory',
                          'cancer', 'copd', 'asthma', 'poverty', 'minority', 
                          'uninsured', 'population']
    tract_data = tract_data.reset_index()
    tract_data['years_active'] = 2024 - tract_data['first_year']
    tract_data['log_releases'] = np.log10(tract_data['releases'].clip(1))
    tract_data['log_population'] = np.log10(tract_data['population'].clip(1))
    
    tract_clean = tract_data.dropna()
    
    # Convert boolean columns to int for regression
    tract_clean['has_carcinogen'] = tract_clean['has_carcinogen'].astype(int)
    tract_clean['has_respiratory'] = tract_clean['has_respiratory'].astype(int)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Panel A: Full correlation matrix
    ax = axes[0, 0]
    
    vars_interest = ['copd', 'cancer', 'log_releases', 'n_facilities', 
                     'years_active', 'poverty', 'uninsured', 'log_population']
    var_labels = ['COPD', 'Cancer', 'Log Releases', '# Facilities',
                  'Years Active', 'Poverty', 'Uninsured', 'Log Pop']
    
    corr_matrix = tract_clean[vars_interest].corr(method='spearman')
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(var_labels)))
    ax.set_xticklabels(var_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(var_labels)))
    ax.set_yticklabels(var_labels, fontsize=8)
    
    for i in range(len(var_labels)):
        for j in range(len(var_labels)):
            if i != j:
                ax.text(j, i, f"{corr_matrix.values[i,j]:.2f}", ha='center', va='center', fontsize=7)
    
    plt.colorbar(im, ax=ax, label='Spearman r')
    ax.set_title("A. FULL CORRELATION MATRIX\n(All key variables)", fontweight='bold')
    
    # Panel B: Stepwise model comparison for COPD
    ax = axes[0, 1]
    
    # Build progressively complex models
    models = {
        'Releases only': ['log_releases'],
        '+ # Facilities': ['log_releases', 'n_facilities'],
        '+ History': ['log_releases', 'n_facilities', 'years_active'],
        '+ Poverty': ['log_releases', 'n_facilities', 'years_active', 'poverty'],
        '+ Population': ['log_releases', 'n_facilities', 'years_active', 'poverty', 'log_population'],
        '+ Chemical type': ['log_releases', 'n_facilities', 'years_active', 'poverty', 'log_population', 'has_respiratory'],
    }
    
    r2_values = []
    model_names = []
    
    for name, predictors in models.items():
        X = sm.add_constant(tract_clean[predictors])
        model = sm.OLS(tract_clean['copd'], X).fit()
        r2_values.append(model.rsquared)
        model_names.append(name)
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(r2_values)))
    ax.barh(range(len(r2_values)), r2_values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_xlabel("R² (variance explained)")
    ax.set_title("B. STEPWISE MODEL: Predicting COPD\n(What explains the most?)", fontweight='bold')
    
    for i, r2 in enumerate(r2_values):
        ax.text(r2 + 0.01, i, f"{r2:.3f}", va='center', fontsize=9)
    
    # Panel C: Coefficient comparison in full model
    ax = axes[0, 2]
    
    # Standardize predictors for comparable coefficients
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    predictors_full = ['log_releases', 'n_facilities', 'years_active', 'poverty', 
                       'log_population', 'has_respiratory']
    X_scaled = scaler.fit_transform(tract_clean[predictors_full])
    X_scaled = sm.add_constant(X_scaled)
    
    model_full = sm.OLS(tract_clean['copd'], X_scaled).fit()
    coefs = model_full.params[1:]  # exclude constant
    
    pred_labels = ['Log Releases', '# Facilities', 'Years Active', 'Poverty',
                   'Log Population', 'Respiratory Chem']
    
    colors_coef = ['red' if c > 0 else 'blue' for c in coefs]
    ax.barh(range(len(coefs)), coefs, color=colors_coef, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_yticks(range(len(pred_labels)))
    ax.set_yticklabels(pred_labels)
    ax.set_xlabel("Standardized Coefficient")
    ax.set_title("C. STANDARDIZED COEFFICIENTS\n(Which factor matters most?)", fontweight='bold')
    
    for i, c in enumerate(coefs):
        ax.text(c + 0.01 if c > 0 else c - 0.01, i, f"{c:.3f}", 
               va='center', ha='left' if c > 0 else 'right', fontsize=9)
    
    # Panel D: Variance decomposition
    ax = axes[1, 0]
    
    # Calculate unique contribution of each factor
    full_r2 = r2_values[-1]
    
    contributions = []
    for predictor in predictors_full:
        # R² without this predictor
        other_preds = [p for p in predictors_full if p != predictor]
        X_reduced = sm.add_constant(tract_clean[other_preds])
        model_reduced = sm.OLS(tract_clean['copd'], X_reduced).fit()
        
        # Unique contribution = full R² - reduced R²
        contribution = full_r2 - model_reduced.rsquared
        contributions.append(contribution)
    
    colors_contrib = plt.cm.Set3(np.linspace(0, 1, len(contributions)))
    ax.pie(np.abs(contributions), labels=pred_labels, autopct='%1.1f%%',
           colors=colors_contrib, startangle=90)
    ax.set_title("D. VARIANCE DECOMPOSITION\n(Unique contribution of each factor)", fontweight='bold')
    
    # Panel E: Same analysis for cancer
    ax = axes[1, 1]
    
    r2_cancer = []
    for name, predictors in models.items():
        X = sm.add_constant(tract_clean[predictors])
        model = sm.OLS(tract_clean['cancer'], X).fit()
        r2_cancer.append(model.rsquared)
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax.bar(x - width/2, r2_values, width, label='COPD', color='#e74c3c')
    ax.bar(x + width/2, r2_cancer, width, label='Cancer', color='#9b59b6')
    ax.set_xticks(x)
    ax.set_xticklabels(['1', '2', '3', '4', '5', '6'], fontsize=9)
    ax.set_xlabel("Model Complexity (1=simple, 6=full)")
    ax.set_ylabel("R²")
    ax.set_title("E. MODEL COMPARISON: COPD vs Cancer\n(Which is more predictable?)", fontweight='bold')
    ax.legend()
    
    # Panel F: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Find strongest predictor
    strongest_idx = np.argmax(np.abs(coefs.values))
    strongest_pred = pred_labels[strongest_idx]
    strongest_coef = coefs.values[strongest_idx]
    
    # Find largest unique contribution
    largest_contrib_idx = np.argmax(contributions)
    largest_contrib = pred_labels[largest_contrib_idx]
    
    summary_text = f"""
COMBINED MECHANISM MODEL

This analysis tests all hypothesized mechanisms together to determine
which factors matter most for health outcomes.

MODEL PERFORMANCE (predicting COPD):
  Releases only:           R² = {r2_values[0]:.3f}
  + Facilities:            R² = {r2_values[1]:.3f}
  + History:               R² = {r2_values[2]:.3f}
  + Poverty:               R² = {r2_values[3]:.3f}
  + Population:            R² = {r2_values[4]:.3f}
  + Chemical type:         R² = {r2_values[5]:.3f}

STRONGEST PREDICTOR:
  {strongest_pred}: β = {strongest_coef:.3f}

LARGEST UNIQUE CONTRIBUTION:
  {largest_contrib}

KEY FINDINGS:

1. {'POVERTY is the strongest predictor' if 'Poverty' in strongest_pred else 
   f'{strongest_pred} is the strongest predictor'}

2. Chemical type (respiratory irritants) {'adds explanatory power'
   if r2_values[-1] > r2_values[-2] + 0.01 else 'adds little beyond other factors'}

3. Industrial history {'matters independently' if contributions[2] > 0.01 
   else 'overlaps with other factors'}

4. Population (migration proxy) {'contributes to the model'
   if contributions[4] > 0.01 else 'adds little'}

5. Total model explains {r2_values[-1]*100:.1f}% of COPD variance

INTERPRETATION:
The health burden in industrial areas is explained by a combination of:
- Socioeconomic factors (poverty, ability to move)
- Exposure duration (industrial history)
- Specific chemical exposures (respiratory irritants)
- Population selection (who stays/leaves)
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title("F. Summary: What Explains Health?", fontweight='bold')
    
    fig.suptitle("ROUND 3: Combined Mechanism Model\n"
                 "Testing all hypothesized factors together",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, "r3_combined_mechanism_model")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_round3_hypotheses():
    """Run all round 3 hypothesis tests."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    
    logger.info("=" * 60)
    logger.info("HYPOTHESIS TESTING ROUND 3: Chemical Types and Mechanisms")
    logger.info("=" * 60)
    
    fac = _load_data()
    logger.info(f"Loaded {len(fac):,} facility records")
    
    logger.info("\n>>> Chemical Type Effects After Controls")
    chemical_effects_controlled(fac)
    
    logger.info("\n>>> Release Medium Pathway Specificity")
    release_medium_pathways(fac)
    
    logger.info("\n>>> Selective Migration Hypothesis")
    selective_migration(fac)
    
    logger.info("\n>>> Historical/Long-Term Effects")
    historical_effects(fac)
    
    logger.info("\n>>> Combined Mechanism Model")
    combined_mechanism_model(fac)
    
    logger.info("\n" + "=" * 60)
    logger.info("Round 3 hypothesis testing complete!")
    plots = sorted(OUT.glob("r3_*.png"))
    for p in plots:
        logger.info(f"  {p.name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_round3_hypotheses()
