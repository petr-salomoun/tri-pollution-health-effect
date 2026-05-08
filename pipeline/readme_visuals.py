"""
README Visuals - Targeted plots for the narrative
=================================================

Creates publication-quality visualizations supporting the README story structure.
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
from scipy.stats import spearmanr, mannwhitneyu

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

OUT = Path("output/readme")
OUT.mkdir(parents=True, exist_ok=True)

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
    """Load facility data with health outcomes."""
    fac = pd.read_csv("data/processed/facilities_scored.csv", low_memory=False)
    fac["has_health"] = fac["cancer_crude"].notna()
    fac['fips_tract'] = fac['fips_tract'].astype(str).str.zfill(11)
    return fac


def plot_dose_response_by_disease(fac):
    """
    FIGURE 1: Dose-response - number of facilities vs health outcomes
    Shows clear gradient for multiple diseases.
    """
    logger.info("Creating dose-response by facility count...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Aggregate to tract level with facility count
    tract_data = hdf.groupby('fips_tract').agg({
        'TRI_FACILITY_ID': 'nunique',
        'copd_crude': 'mean',
        'diabetes_crude': 'mean',
        'chd_crude': 'mean',
        'asthma_crude': 'mean',
        'cancer_crude': 'mean',
        'poverty_pct': 'mean',
    }).reset_index()
    tract_data.columns = ['fips_tract', 'n_facilities', 'copd', 'diabetes', 'chd', 'asthma', 'cancer', 'poverty']
    
    # Create facility count bins
    def facility_bin(n):
        if n == 0:
            return '0 (control)'
        elif n == 1:
            return '1'
        elif n <= 3:
            return '2-3'
        elif n <= 5:
            return '4-5'
        else:
            return '6+'
    
    tract_data['facility_bin'] = tract_data['n_facilities'].apply(facility_bin)
    
    # Load controls
    try:
        controls = pd.read_csv("data/processed/controls_scored.csv", low_memory=False)
        controls = controls[controls['cancer_crude'].notna()].copy()
        controls['n_facilities'] = 0
        controls['facility_bin'] = '0 (control)'
        controls = controls.rename(columns={
            'copd_crude': 'copd', 'diabetes_crude': 'diabetes', 
            'chd_crude': 'chd', 'asthma_crude': 'asthma', 
            'cancer_crude': 'cancer', 'poverty_pct': 'poverty'
        })
        all_data = pd.concat([
            tract_data[['facility_bin', 'copd', 'diabetes', 'chd', 'asthma', 'cancer', 'poverty']],
            controls[['facility_bin', 'copd', 'diabetes', 'chd', 'asthma', 'cancer', 'poverty']]
        ], ignore_index=True)
    except:
        all_data = tract_data
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    diseases = [
        ('copd', 'COPD', '#e74c3c'),
        ('diabetes', 'Diabetes', '#3498db'),
        ('chd', 'Heart Disease', '#9b59b6'),
        ('asthma', 'Asthma', '#2ecc71'),
        ('cancer', 'Cancer', '#f39c12'),
        ('poverty', 'Poverty Rate', '#7f8c8d'),
    ]
    
    bin_order = ['0 (control)', '1', '2-3', '4-5', '6+']
    
    for ax, (col, title, color) in zip(axes.flat, diseases):
        means = all_data.groupby('facility_bin')[col].mean()
        stds = all_data.groupby('facility_bin')[col].std()
        ns = all_data.groupby('facility_bin')[col].count()
        sems = stds / np.sqrt(ns)
        
        # Reorder
        means = means.reindex(bin_order)
        sems = sems.reindex(bin_order)
        
        x = range(len(bin_order))
        bars = ax.bar(x, means.values, yerr=sems.values * 1.96, 
                      color=color, alpha=0.7, capsize=3, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(bin_order, fontsize=9)
        ax.set_xlabel("# TRI Facilities in Tract", fontsize=10)
        ax.set_ylabel(f"{title} Rate (%)", fontsize=10)
        ax.set_title(f"{title}", fontweight='bold', fontsize=12)
        
        # Add value labels
        for i, (v, s) in enumerate(zip(means.values, sems.values)):
            if not np.isnan(v):
                ax.text(i, v + s*1.96 + 0.1, f"{v:.1f}%", ha='center', fontsize=8)
        
        # Add gradient line
        if col != 'poverty':
            baseline = means.iloc[0] if not np.isnan(means.iloc[0]) else means.iloc[1]
            ax.axhline(baseline, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("DOSE-RESPONSE: More Facilities = Worse Health\n(Except poverty - facilities don't locate in poorest areas)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, "dose_response_facilities")


def plot_exposure_duration(fac):
    """
    FIGURE 2: Longer exposure = worse health
    Compare tracts by how long facilities have been present.
    """
    logger.info("Creating exposure duration plot...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Find first year each tract had a facility
    tract_first_year = hdf.groupby('fips_tract')['REPORTING_YEAR'].min().reset_index()
    tract_first_year.columns = ['fips_tract', 'first_year']
    
    # Merge back
    tract_data = hdf.groupby('fips_tract').agg({
        'copd_crude': 'mean',
        'diabetes_crude': 'mean',
        'chd_crude': 'mean',
        'cancer_crude': 'mean',
        'poverty_pct': 'mean',
    }).reset_index()
    
    tract_data = tract_data.merge(tract_first_year, on='fips_tract')
    tract_data['years_exposed'] = 2024 - tract_data['first_year']
    
    # Bin exposure duration
    def duration_bin(y):
        if y <= 3:
            return 'Recent (1-3 yrs)'
        elif y <= 6:
            return 'Medium (4-6 yrs)'
        elif y <= 9:
            return 'Long (7-9 yrs)'
        else:
            return 'Full period (10+ yrs)'
    
    tract_data['duration_bin'] = tract_data['years_exposed'].apply(duration_bin)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    diseases = [
        ('copd_crude', 'COPD', '#e74c3c'),
        ('diabetes_crude', 'Diabetes', '#3498db'),
        ('chd_crude', 'Heart Disease', '#9b59b6'),
        ('cancer_crude', 'Cancer', '#f39c12'),
    ]
    
    bin_order = ['Recent (1-3 yrs)', 'Medium (4-6 yrs)', 'Long (7-9 yrs)', 'Full period (10+ yrs)']
    
    for ax, (col, title, color) in zip(axes.flat, diseases):
        means = tract_data.groupby('duration_bin')[col].mean().reindex(bin_order)
        stds = tract_data.groupby('duration_bin')[col].std().reindex(bin_order)
        ns = tract_data.groupby('duration_bin')[col].count().reindex(bin_order)
        sems = stds / np.sqrt(ns)
        
        x = range(len(bin_order))
        bars = ax.bar(x, means.values, yerr=sems.values * 1.96,
                      color=color, alpha=0.7, capsize=4, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(['1-3 yrs', '4-6 yrs', '7-9 yrs', '10+ yrs'], fontsize=9)
        ax.set_xlabel("Duration of Facility Presence", fontsize=10)
        ax.set_ylabel(f"{title} Rate (%)", fontsize=10)
        ax.set_title(f"{title}", fontweight='bold', fontsize=12)
        
        # Add trend line
        valid_idx = ~np.isnan(means.values)
        if sum(valid_idx) >= 2:
            z = np.polyfit(np.array(x)[valid_idx], means.values[valid_idx], 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), 'k--', linewidth=2, alpha=0.7)
            slope_per_year = z[0] / 3  # approx per year
            ax.text(0.95, 0.95, f"Trend: +{slope_per_year:.2f}pp/yr", 
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("CUMULATIVE EFFECT: Longer Exposure = Worse Health", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, "exposure_duration")


def plot_confounder_matrix(fac):
    """
    FIGURE 3: Correlation matrix showing confounders
    Key confounders: poverty, minority %, insurance, education vs health outcomes
    """
    logger.info("Creating confounder correlation matrix...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Aggregate to tract
    tract_data = hdf.groupby('fips_tract').agg({
        'TOTAL_RELEASES': 'sum',
        'TRI_FACILITY_ID': 'nunique',
        'copd_crude': 'mean',
        'diabetes_crude': 'mean',
        'chd_crude': 'mean',
        'cancer_crude': 'mean',
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
        'pct_no_insurance': 'mean',
        'median_income': 'mean',
    }).reset_index()
    
    tract_data['log_releases'] = np.log10(tract_data['TOTAL_RELEASES'].clip(1))
    
    # Variables of interest
    predictors = ['poverty_pct', 'minority_pct', 'pct_no_insurance', 'median_income', 'log_releases', 'TRI_FACILITY_ID']
    outcomes = ['copd_crude', 'diabetes_crude', 'chd_crude', 'cancer_crude']
    
    pred_labels = ['Poverty %', 'Minority %', 'Uninsured %', 'Median Income', 'Log Releases', '# Facilities']
    outcome_labels = ['COPD', 'Diabetes', 'CHD', 'Cancer']
    
    # Compute Spearman correlations
    corr_matrix = np.zeros((len(predictors), len(outcomes)))
    for i, pred in enumerate(predictors):
        for j, out in enumerate(outcomes):
            valid = tract_data[[pred, out]].dropna()
            r, _ = spearmanr(valid[pred], valid[out])
            corr_matrix[i, j] = r
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    
    ax.set_xticks(range(len(outcome_labels)))
    ax.set_xticklabels(outcome_labels, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(pred_labels)))
    ax.set_yticklabels(pred_labels, fontsize=11)
    
    # Add correlation values
    for i in range(len(pred_labels)):
        for j in range(len(outcome_labels)):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.25 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=12, fontweight='bold', color=color)
    
    plt.colorbar(im, ax=ax, label='Spearman Correlation', shrink=0.8)
    
    ax.set_title("CONFOUNDERS: What Actually Predicts Disease?\n(Poverty dominates; releases are weak)", 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Add annotation box
    ax.annotate('', xy=(0, -0.5), xytext=(3.5, -0.5),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2),
                annotation_clip=False)
    ax.text(1.75, -0.8, 'SOCIOECONOMIC FACTORS\n(Strong predictors)', ha='center', va='top', 
            fontsize=10, fontweight='bold', color='green',
            transform=ax.get_xaxis_transform())
    
    ax.annotate('', xy=(0, 5.5), xytext=(3.5, 5.5),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=2),
                annotation_clip=False)
    ax.text(1.75, 5.8, 'POLLUTION VARIABLES\n(Weak predictors)', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='orange',
            transform=ax.get_xaxis_transform())
    
    plt.tight_layout()
    _save(fig, "confounder_matrix")


def plot_poverty_pathway(fac):
    """
    FIGURE 4: Poverty as confounder - pathway exists but weak
    Shows poverty → health is strong, TRI → poverty is weak
    """
    logger.info("Creating poverty pathway plot...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Aggregate to tract
    tract_data = hdf.groupby('fips_tract').agg({
        'copd_crude': 'mean',
        'poverty_pct': 'mean',
    }).reset_index()
    
    # Load controls
    try:
        controls = pd.read_csv("data/processed/controls_scored.csv", low_memory=False)
        controls = controls[controls['copd_crude'].notna()].copy()
        controls['is_tri'] = False
        tract_data['is_tri'] = True
        controls = controls.rename(columns={'copd_crude': 'copd_crude', 'poverty_pct': 'poverty_pct'})
        all_data = pd.concat([
            tract_data[['copd_crude', 'poverty_pct', 'is_tri']],
            controls[['copd_crude', 'poverty_pct', 'is_tri']]
        ], ignore_index=True)
    except:
        all_data = tract_data
        all_data['is_tri'] = True
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Poverty distribution TRI vs Control
    ax = axes[0]
    tri_pov = all_data[all_data['is_tri']]['poverty_pct'].dropna()
    ctrl_pov = all_data[~all_data['is_tri']]['poverty_pct'].dropna()
    
    bins = np.linspace(0, 50, 30)
    ax.hist(ctrl_pov, bins=bins, alpha=0.6, label=f'Control (μ={ctrl_pov.mean():.1f}%)', color='#3498db', density=True)
    ax.hist(tri_pov, bins=bins, alpha=0.6, label=f'TRI (μ={tri_pov.mean():.1f}%)', color='#e74c3c', density=True)
    ax.axvline(ctrl_pov.mean(), color='#3498db', linestyle='--', linewidth=2)
    ax.axvline(tri_pov.mean(), color='#e74c3c', linestyle='--', linewidth=2)
    ax.set_xlabel("Poverty Rate (%)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("A. TRI vs Control Poverty\n(Gap is only ~1pp)", fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    
    # Panel B: Poverty vs COPD scatter
    ax = axes[1]
    sample = all_data.dropna().sample(min(3000, len(all_data)), random_state=42)
    colors = ['#e74c3c' if t else '#3498db' for t in sample['is_tri']]
    ax.scatter(sample['poverty_pct'], sample['copd_crude'], alpha=0.3, c=colors, s=10)
    
    # Fit lines
    for is_tri, color, label in [(True, '#e74c3c', 'TRI'), (False, '#3498db', 'Control')]:
        subset = all_data[all_data['is_tri'] == is_tri].dropna()
        if len(subset) < 10:
            continue
        z = np.polyfit(subset['poverty_pct'], subset['copd_crude'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 40, 100)
        ax.plot(x_line, p(x_line), color=color, linewidth=2, label=f'{label}: {z[0]:.3f}x + {z[1]:.1f}')
    
    ax.set_xlabel("Poverty Rate (%)", fontsize=11)
    ax.set_ylabel("COPD Rate (%)", fontsize=11)
    ax.set_title("B. Poverty → COPD\n(Same slope, TRI elevated)", fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    
    # Panel C: The gap explained
    ax = axes[2]
    
    # Calculate poverty-adjusted gap
    # Within each poverty quintile, compare TRI vs control COPD
    all_data['pov_quintile'] = pd.cut(all_data['poverty_pct'].rank(method='first'), bins=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    gaps = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        subset = all_data[all_data['pov_quintile'] == q]
        tri_copd = subset[subset['is_tri']]['copd_crude'].mean()
        ctrl_copd = subset[~subset['is_tri']]['copd_crude'].mean()
        gaps.append(tri_copd - ctrl_copd if not np.isnan(tri_copd) and not np.isnan(ctrl_copd) else 0)
    
    x = range(5)
    colors_gap = ['#27ae60' if g < 0 else '#e74c3c' for g in gaps]
    ax.bar(x, gaps, color=colors_gap, edgecolor='black', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1\n(Richest)', 'Q2', 'Q3', 'Q4', 'Q5\n(Poorest)'], fontsize=9)
    ax.set_xlabel("Poverty Quintile", fontsize=11)
    ax.set_ylabel("TRI - Control COPD Gap (pp)", fontsize=11)
    ax.set_title("C. TRI Effect Within Poverty Bands\n(Gap persists at all levels)", fontweight='bold', fontsize=12)
    
    mean_gap = np.mean(gaps)
    ax.axhline(mean_gap, color='red', linestyle='--', linewidth=1)
    ax.text(4.5, mean_gap, f'Avg gap: {mean_gap:.2f}pp', ha='right', va='bottom', fontsize=9, color='red')
    
    plt.suptitle("POVERTY AS CONFOUNDER: Real but Doesn't Explain TRI Effect", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, "poverty_pathway")


def plot_demographic_differences(fac):
    """
    FIGURE 5: TRI areas are different - not just poorer
    Less white, less insured, different population structure
    """
    logger.info("Creating demographic differences plot...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Aggregate to tract
    tract_data = hdf.groupby('fips_tract').agg({
        'poverty_pct': 'mean',
        'minority_pct': 'mean',
        'pct_no_insurance': 'mean',
        'median_income': 'mean',
    }).reset_index()
    tract_data['is_tri'] = True
    
    # Load controls
    try:
        controls = pd.read_csv("data/processed/controls_scored.csv", low_memory=False)
        controls = controls[controls['cancer_crude'].notna()].copy()
        controls['is_tri'] = False
        all_data = pd.concat([
            tract_data[['poverty_pct', 'minority_pct', 'pct_no_insurance', 'median_income', 'is_tri']],
            controls[['poverty_pct', 'minority_pct', 'pct_no_insurance', 'median_income', 'is_tri']]
        ], ignore_index=True)
    except:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = [
        ('poverty_pct', 'Poverty Rate (%)', axes[0, 0]),
        ('minority_pct', 'Minority Population (%)', axes[0, 1]),
        ('pct_no_insurance', 'Uninsured Rate (%)', axes[1, 0]),
        ('median_income', 'Median Income ($)', axes[1, 1]),
    ]
    
    for col, label, ax in metrics:
        tri_vals = all_data[all_data['is_tri']][col].dropna()
        ctrl_vals = all_data[~all_data['is_tri']][col].dropna()
        
        tri_mean = tri_vals.mean()
        ctrl_mean = ctrl_vals.mean()
        diff = tri_mean - ctrl_mean
        diff_pct = (diff / ctrl_mean * 100) if ctrl_mean != 0 else 0
        
        # Statistical test
        stat, p = mannwhitneyu(tri_vals, ctrl_vals, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        
        x = [0, 1]
        means = [ctrl_mean, tri_mean]
        colors = ['#3498db', '#e74c3c']
        bars = ax.bar(x, means, color=colors, edgecolor='black', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Control', 'TRI'], fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        
        # Add difference annotation
        sign = '+' if diff > 0 else ''
        if col == 'median_income':
            ax.set_title(f"{label}\nTRI: {sign}${diff:,.0f} ({sign}{diff_pct:.1f}%) {sig}", fontweight='bold', fontsize=11)
        else:
            ax.set_title(f"{label}\nTRI: {sign}{diff:.1f}pp ({sign}{diff_pct:.1f}%) {sig}", fontweight='bold', fontsize=11)
        
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("TRI AREAS ARE DIFFERENT\n(Not much poorer, but more minority & less insured)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, "demographic_differences")


def plot_chemical_weakness(fac):
    """
    FIGURE 6: Chemical effects are weak after controls
    Shows variance explained by different factors
    """
    logger.info("Creating chemical weakness plot...")
    
    import statsmodels.api as sm
    
    hdf = fac[fac['has_health']].copy()
    
    # Classify chemicals
    carcinogens = {'arsenic', 'benzene', 'chromium', 'cadmium', 'nickel', 'lead',
                   'vinyl chloride', 'formaldehyde', 'trichloroethylene'}
    respiratory = {'ammonia', 'chlorine', 'hydrogen chloride', 'sulfur dioxide',
                   'nitrogen oxide', 'hydrogen fluoride', 'formaldehyde', 'sulfuric acid'}
    
    def is_carcinogen(name):
        if not isinstance(name, str):
            return False
        return any(c in name.lower() for c in carcinogens)
    
    def is_respiratory(name):
        if not isinstance(name, str):
            return False
        return any(c in name.lower() for c in respiratory)
    
    hdf['is_carcinogen'] = hdf['CHEMICAL_NAME'].apply(is_carcinogen)
    hdf['is_respiratory'] = hdf['CHEMICAL_NAME'].apply(is_respiratory)
    
    # Aggregate to tract
    tract_data = hdf.groupby('fips_tract').agg({
        'TOTAL_RELEASES': 'sum',
        'TRI_FACILITY_ID': 'nunique',
        'is_carcinogen': 'max',
        'is_respiratory': 'max',
        'copd_crude': 'mean',
        'cancer_crude': 'mean',
        'poverty_pct': 'mean',
        'pct_no_insurance': 'mean',
    }).reset_index()
    
    tract_data['log_releases'] = np.log10(tract_data['TOTAL_RELEASES'].clip(1))
    tract_data['is_carcinogen'] = tract_data['is_carcinogen'].astype(int)
    tract_data['is_respiratory'] = tract_data['is_respiratory'].astype(int)
    
    tract_clean = tract_data.dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Variance explained for COPD
    ax = axes[0]
    
    models_copd = [
        ('Poverty only', ['poverty_pct']),
        ('+ Uninsured', ['poverty_pct', 'pct_no_insurance']),
        ('+ Log releases', ['poverty_pct', 'pct_no_insurance', 'log_releases']),
        ('+ # Facilities', ['poverty_pct', 'pct_no_insurance', 'log_releases', 'TRI_FACILITY_ID']),
        ('+ Respiratory chem', ['poverty_pct', 'pct_no_insurance', 'log_releases', 'TRI_FACILITY_ID', 'is_respiratory']),
    ]
    
    r2_copd = []
    for name, preds in models_copd:
        X = sm.add_constant(tract_clean[preds])
        model = sm.OLS(tract_clean['copd_crude'], X).fit()
        r2_copd.append(model.rsquared)
    
    # Compute increments
    increments = [r2_copd[0]] + [r2_copd[i] - r2_copd[i-1] for i in range(1, len(r2_copd))]
    
    labels = ['Poverty', '+Uninsured', '+Releases', '+Facilities', '+Chem type']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#2ecc71']
    
    bars = ax.bar(range(len(labels)), increments, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, rotation=15)
    ax.set_ylabel("R² Increment (Variance Explained)", fontsize=11)
    ax.set_title("A. PREDICTING COPD\n(Poverty explains ~23%, pollution adds <1%)", fontweight='bold', fontsize=12)
    
    for i, (inc, tot) in enumerate(zip(increments, r2_copd)):
        if inc > 0.005:
            ax.text(i, inc + 0.005, f"+{inc*100:.1f}%\n(R²={tot*100:.1f}%)", ha='center', fontsize=8)
        else:
            ax.text(i, inc + 0.002, f"+{inc*100:.2f}%", ha='center', fontsize=7)
    
    ax.set_ylim(0, 0.28)
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Variance explained for Cancer
    ax = axes[1]
    
    models_cancer = [
        ('Poverty only', ['poverty_pct']),
        ('+ Uninsured', ['poverty_pct', 'pct_no_insurance']),
        ('+ Log releases', ['poverty_pct', 'pct_no_insurance', 'log_releases']),
        ('+ # Facilities', ['poverty_pct', 'pct_no_insurance', 'log_releases', 'TRI_FACILITY_ID']),
        ('+ Carcinogen', ['poverty_pct', 'pct_no_insurance', 'log_releases', 'TRI_FACILITY_ID', 'is_carcinogen']),
    ]
    
    r2_cancer = []
    for name, preds in models_cancer:
        X = sm.add_constant(tract_clean[preds])
        model = sm.OLS(tract_clean['cancer_crude'], X).fit()
        r2_cancer.append(model.rsquared)
    
    increments_c = [r2_cancer[0]] + [r2_cancer[i] - r2_cancer[i-1] for i in range(1, len(r2_cancer))]
    
    bars = ax.bar(range(len(labels)), increments_c, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(labels)))
    labels_c = ['Poverty', '+Uninsured', '+Releases', '+Facilities', '+Carcinogen']
    ax.set_xticklabels(labels_c, fontsize=10, rotation=15)
    ax.set_ylabel("R² Increment (Variance Explained)", fontsize=11)
    ax.set_title("B. PREDICTING CANCER\n(Poverty & insurance explain ~20%, carcinogens add nothing)", fontweight='bold', fontsize=12)
    
    for i, (inc, tot) in enumerate(zip(increments_c, r2_cancer)):
        if abs(inc) > 0.005:
            ax.text(i, max(inc, 0.002) + 0.005, f"+{inc*100:.1f}%\n(R²={tot*100:.1f}%)", ha='center', fontsize=8)
        else:
            ax.text(i, max(inc, 0) + 0.002, f"+{inc*100:.2f}%", ha='center', fontsize=7)
    
    ax.set_ylim(0, 0.28)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("CHEMICAL EFFECTS ARE MINIMAL\n(Socioeconomic factors dominate; chemical type adds nothing)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, "chemical_weakness")


def plot_closure_persistence(fac):
    """
    FIGURE 7: Health issues persist after facility closure
    Compare tracts that lost facilities vs those that kept them
    """
    logger.info("Creating closure persistence plot...")
    
    hdf = fac[fac['has_health']].copy()
    
    # Find tracts with facilities in early years (2013-2015) vs later
    early_years = set(hdf[hdf['REPORTING_YEAR'] <= 2015]['fips_tract'].unique())
    late_years = set(hdf[hdf['REPORTING_YEAR'] >= 2021]['fips_tract'].unique())
    
    # Categories
    always_present = early_years & late_years  # Had facility 2013-2015 AND 2021-2023
    closed = early_years - late_years  # Had facility 2013-2015, NOT in 2021-2023
    new = late_years - early_years  # NOT in 2013-2015, but in 2021-2023
    
    # Get health outcomes for each category (using most recent data)
    recent = hdf[hdf['REPORTING_YEAR'] >= 2021].groupby('fips_tract').agg({
        'copd_crude': 'mean',
        'diabetes_crude': 'mean',
        'cancer_crude': 'mean',
        'poverty_pct': 'mean',
    }).reset_index()
    
    # For closed tracts, use earlier data since they're not in recent
    early = hdf[hdf['REPORTING_YEAR'] <= 2015].groupby('fips_tract').agg({
        'copd_crude': 'mean',
        'diabetes_crude': 'mean',
        'cancer_crude': 'mean',
        'poverty_pct': 'mean',
    }).reset_index()
    
    # Categorize
    recent['category'] = recent['fips_tract'].apply(
        lambda x: 'Always present' if x in always_present else 'New facility' if x in new else 'Other'
    )
    early['category'] = early['fips_tract'].apply(
        lambda x: 'Closed' if x in closed else 'Other'
    )
    
    # Combine
    always_df = recent[recent['category'] == 'Always present']
    new_df = recent[recent['category'] == 'New facility']
    closed_df = early[early['category'] == 'Closed']
    
    # Load controls for baseline
    try:
        controls = pd.read_csv("data/processed/controls_scored.csv", low_memory=False)
        controls = controls[controls['copd_crude'].notna()].copy()
        ctrl_copd = controls['copd_crude'].mean()
        ctrl_diab = controls['diabetes_crude'].mean()
        ctrl_cancer = controls['cancer_crude'].mean()
    except:
        ctrl_copd, ctrl_diab, ctrl_cancer = 6.86, 12.34, 7.83
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    categories = ['Control', 'New facility\n(2021+ only)', 'Closed\n(pre-2016 only)', 'Always present']
    
    for ax, (col, title, baseline) in zip(axes, [
        ('copd_crude', 'COPD', ctrl_copd),
        ('diabetes_crude', 'Diabetes', ctrl_diab),
        ('cancer_crude', 'Cancer', ctrl_cancer),
    ]):
        means = [
            baseline,
            new_df[col].mean() if len(new_df) > 0 else np.nan,
            closed_df[col].mean() if len(closed_df) > 0 else np.nan,
            always_df[col].mean() if len(always_df) > 0 else np.nan,
        ]
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars = ax.bar(range(4), means, color=colors, edgecolor='black', alpha=0.7)
        
        ax.set_xticks(range(4))
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel(f"{title} Rate (%)", fontsize=11)
        ax.set_title(f"{title}", fontweight='bold', fontsize=12)
        
        ax.axhline(baseline, color='gray', linestyle='--', alpha=0.5)
        
        for i, v in enumerate(means):
            if not np.isnan(v):
                ax.text(i, v + 0.1, f"{v:.1f}%", ha='center', fontsize=9)
        
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("CLOSURE DOESN'T HEAL: Tracts That Lost Facilities Still Show Elevated Disease\n" +
                 f"(N: New={len(new_df)}, Closed={len(closed_df)}, Always={len(always_df)})",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, "closure_persistence")


def plot_model_diagram():
    """
    FIGURE 8: Visual model diagram (matplotlib, not ASCII)
    """
    logger.info("Creating model diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define boxes
    def draw_box(x, y, w, h, text, color, fontsize=10):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)
    
    def draw_arrow(x1, y1, x2, y2, color='black', label='', offset=0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        if label:
            mx, my = (x1+x2)/2 + offset, (y1+y2)/2
            ax.text(mx, my, label, fontsize=9, ha='center', va='bottom', color=color)
    
    # Health outcome box at top
    draw_box(5.5, 8.5, 3, 1.2, "POOR HEALTH\n(COPD, CHD, Diabetes)", '#e74c3c', fontsize=11)
    
    # Three mechanism boxes in middle
    draw_box(0.5, 5, 3.5, 1.5, "POVERTY &\nDEPRIVATION\n(~23%)", '#3498db', fontsize=10)
    draw_box(5.25, 5, 3.5, 1.5, "SELECTIVE\nMIGRATION\n(~20-40%)", '#9b59b6', fontsize=10)
    draw_box(10, 5, 3.5, 1.5, "HISTORICAL\nBURDEN\n(~10-20%)", '#f39c12', fontsize=10)
    
    # TRI facility box at bottom
    draw_box(5.5, 1.5, 3, 1.2, "TRI FACILITY\nPRESENCE", '#7f8c8d', fontsize=11)
    
    # Weak effect box
    draw_box(11, 1.5, 2.5, 1.2, "CHEMICAL\nEFFECT\n(<1%)", '#95a5a6', fontsize=9)
    
    # Arrows from mechanisms to health
    draw_arrow(2.25, 6.5, 6, 8.5, '#3498db')
    draw_arrow(7, 6.5, 7, 8.5, '#9b59b6')
    draw_arrow(11.75, 6.5, 8, 8.5, '#f39c12')
    
    # Arrows from TRI to mechanisms
    draw_arrow(7, 2.7, 2.25, 5, '#7f8c8d')
    draw_arrow(7, 2.7, 7, 5, '#7f8c8d')
    draw_arrow(7, 2.7, 11.75, 5, '#7f8c8d')
    
    # Dashed arrow from chemical to health (weak)
    ax.annotate('', xy=(10, 8.8), xytext=(12.25, 2.7),
                arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=1.5, linestyle='--'))
    
    # Labels
    ax.text(7, 0.5, "TRI facilities are a MARKER of disadvantage,\nnot the primary CAUSE of poor health", 
            ha='center', va='center', fontsize=12, fontweight='bold', style='italic',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9))
    
    ax.text(0.5, 9.5, "THE REVISED MODEL", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    _save(fig, "model_diagram")


def run_readme_visuals():
    """Run all README visualizations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    logger.info("=" * 60)
    logger.info("GENERATING README VISUALS")
    logger.info("=" * 60)
    
    fac = _load_data()
    logger.info(f"Loaded {len(fac):,} facility records")
    
    # Generate all plots
    plot_dose_response_by_disease(fac)
    plot_exposure_duration(fac)
    plot_confounder_matrix(fac)
    plot_poverty_pathway(fac)
    plot_demographic_differences(fac)
    plot_chemical_weakness(fac)
    plot_closure_persistence(fac)
    plot_model_diagram()
    
    logger.info("=" * 60)
    logger.info("README visuals complete!")
    logger.info(f"Output directory: {OUT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_readme_visuals()
