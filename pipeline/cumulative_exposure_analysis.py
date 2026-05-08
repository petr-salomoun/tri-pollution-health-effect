"""
Cumulative Exposure Analysis
============================

Tests whether health outcomes are better explained by:
1. Current year releases
2. Cumulative releases over past N years
3. Permanent cumulative model (all historical releases have lasting effect)

This helps distinguish between:
- Direct acute exposure effects (current releases matter)
- Cumulative chronic exposure (total historical exposure matters)
- Selective migration (facility presence duration matters, not tonnage)
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

OUT = Path("output/research")
OUT.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all necessary data."""
    logger.info("Loading data for cumulative exposure analysis...")
    
    # Health data
    cdc = pd.read_csv("data/raw/cdc_places.csv")
    cdc['fips_tract'] = cdc['locationid'].astype(str).str.zfill(11)
    health = cdc.pivot_table(index='fips_tract', columns='measureid', values='data_value', aggfunc='first').reset_index()
    health = health.rename(columns={
        'CANCER': 'cancer_crude', 'CASTHMA': 'asthma_crude', 'CHD': 'chd_crude',
        'COPD': 'copd_crude', 'DIABETES': 'diabetes_crude', 'ACCESS2': 'pct_no_insurance'
    })
    
    # Demographics
    census = pd.read_csv("data/raw/census_acs.csv")
    census['fips_tract'] = census['FIPS_TRACT'].astype(str).str.zfill(11)
    census['poverty_pct'] = (census['B17001_002E'] / census['B17001_001E'].clip(1)) * 100
    census['minority_pct'] = ((census['B02001_001E'] - census['B02001_002E']) / census['B02001_001E'].clip(1)) * 100
    
    # Facility data with yearly releases
    fac = pd.read_csv("data/processed/facilities_scored.csv", low_memory=False)
    fac['fips_tract'] = fac['fips_tract'].astype(str).str.zfill(11)
    
    return health, census, fac


def compute_cumulative_metrics(fac):
    """
    Compute various cumulative exposure metrics per tract.
    
    Returns DataFrame with:
    - total_cumulative: Sum of all releases 2013-2023
    - recent_3yr: Sum of 2021-2023 releases
    - recent_5yr: Sum of 2019-2023 releases
    - early_3yr: Sum of 2013-2015 releases
    - years_active: Number of years with any releases
    - first_year: First year of releases
    - latest_releases: 2023 releases only
    """
    logger.info("Computing cumulative exposure metrics...")
    
    # Total cumulative
    total_cum = fac.groupby('fips_tract')['TOTAL_RELEASES'].sum().reset_index()
    total_cum.columns = ['fips_tract', 'total_cumulative']
    
    # Recent 3 years (2021-2023)
    recent_3yr = fac[fac['REPORTING_YEAR'] >= 2021].groupby('fips_tract')['TOTAL_RELEASES'].sum().reset_index()
    recent_3yr.columns = ['fips_tract', 'recent_3yr']
    
    # Recent 5 years (2019-2023)
    recent_5yr = fac[fac['REPORTING_YEAR'] >= 2019].groupby('fips_tract')['TOTAL_RELEASES'].sum().reset_index()
    recent_5yr.columns = ['fips_tract', 'recent_5yr']
    
    # Early 3 years (2013-2015)
    early_3yr = fac[fac['REPORTING_YEAR'] <= 2015].groupby('fips_tract')['TOTAL_RELEASES'].sum().reset_index()
    early_3yr.columns = ['fips_tract', 'early_3yr']
    
    # Years active
    years = fac.groupby('fips_tract')['REPORTING_YEAR'].agg(['min', 'max', 'nunique']).reset_index()
    years.columns = ['fips_tract', 'first_year', 'last_year', 'years_active']
    
    # Latest (2023 only)
    latest = fac[fac['REPORTING_YEAR'] == 2023].groupby('fips_tract')['TOTAL_RELEASES'].sum().reset_index()
    latest.columns = ['fips_tract', 'latest_releases']
    
    # Merge all
    metrics = total_cum.merge(recent_3yr, on='fips_tract', how='outer')
    metrics = metrics.merge(recent_5yr, on='fips_tract', how='outer')
    metrics = metrics.merge(early_3yr, on='fips_tract', how='outer')
    metrics = metrics.merge(years, on='fips_tract', how='outer')
    metrics = metrics.merge(latest, on='fips_tract', how='outer')
    
    # Fill NAs with 0 for release amounts
    for col in ['total_cumulative', 'recent_3yr', 'recent_5yr', 'early_3yr', 'latest_releases']:
        metrics[col] = metrics[col].fillna(0)
    
    # Add log versions
    for col in ['total_cumulative', 'recent_3yr', 'recent_5yr', 'early_3yr', 'latest_releases']:
        metrics[f'log_{col}'] = np.log10(metrics[col].clip(1))
    
    # Duration metric
    metrics['duration'] = 2024 - metrics['first_year']
    
    logger.info(f"Computed metrics for {len(metrics):,} tracts")
    return metrics


def compare_exposure_models(all_tracts):
    """
    Compare different exposure models for predicting health outcomes.
    """
    logger.info("Comparing exposure models...")
    
    # Only TRI tracts for within-TRI comparison
    tri_data = all_tracts[all_tracts['total_cumulative'] > 0].copy()
    
    results = []
    
    for disease in ['copd_crude', 'diabetes_crude', 'chd_crude', 'asthma_crude', 'cancer_crude']:
        logger.info(f"  Analyzing {disease}...")
        
        clean = tri_data.dropna(subset=[disease, 'poverty_pct', 'pct_no_insurance'])
        
        # Model 1: Current releases only
        X1 = sm.add_constant(clean[['poverty_pct', 'pct_no_insurance', 'log_latest_releases']])
        m1 = sm.OLS(clean[disease], X1).fit()
        
        # Model 2: Recent 3 years
        X2 = sm.add_constant(clean[['poverty_pct', 'pct_no_insurance', 'log_recent_3yr']])
        m2 = sm.OLS(clean[disease], X2).fit()
        
        # Model 3: Total cumulative
        X3 = sm.add_constant(clean[['poverty_pct', 'pct_no_insurance', 'log_total_cumulative']])
        m3 = sm.OLS(clean[disease], X3).fit()
        
        # Model 4: Duration (proxy for selective migration)
        X4 = sm.add_constant(clean[['poverty_pct', 'pct_no_insurance', 'duration']])
        m4 = sm.OLS(clean[disease], X4).fit()
        
        # Model 5: Early vs Late (test for historical burden)
        X5 = sm.add_constant(clean[['poverty_pct', 'pct_no_insurance', 'log_early_3yr', 'log_recent_3yr']])
        m5 = sm.OLS(clean[disease], X5).fit()
        
        results.append({
            'disease': disease,
            'model_current_r2': m1.rsquared,
            'model_recent3yr_r2': m2.rsquared,
            'model_cumulative_r2': m3.rsquared,
            'model_duration_r2': m4.rsquared,
            'model_early_late_r2': m5.rsquared,
            'current_coef': m1.params['log_latest_releases'],
            'current_pval': m1.pvalues['log_latest_releases'],
            'cumulative_coef': m3.params['log_total_cumulative'],
            'cumulative_pval': m3.pvalues['log_total_cumulative'],
            'duration_coef': m4.params['duration'],
            'duration_pval': m4.pvalues['duration'],
        })
    
    results_df = pd.DataFrame(results)
    return results_df


def plot_model_comparison(results_df):
    """Create visualization comparing model performance."""
    logger.info("Creating model comparison plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: R² comparison across models
    ax = axes[0]
    
    diseases = ['COPD', 'Diabetes', 'CHD', 'Asthma', 'Cancer']
    models = ['Current', 'Recent 3yr', 'Cumulative', 'Duration', 'Early+Late']
    
    r2_data = results_df[['model_current_r2', 'model_recent3yr_r2', 'model_cumulative_r2', 
                           'model_duration_r2', 'model_early_late_r2']].values
    
    x = np.arange(len(diseases))
    width = 0.15
    
    for i, (model, color) in enumerate(zip(models, ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#27ae60'])):
        ax.bar(x + i*width, r2_data[:, i], width, label=model, color=color, alpha=0.8)
    
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(diseases, fontsize=10)
    ax.set_ylabel("Model R²", fontsize=11)
    ax.set_title("A. MODEL FIT BY EXPOSURE METRIC\n(After controlling for poverty and insurance)", fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Duration coefficient significance
    ax = axes[1]
    
    duration_coefs = results_df['duration_coef'].values
    duration_pvals = results_df['duration_pval'].values
    
    colors = ['#27ae60' if p < 0.05 else '#e74c3c' for p in duration_pvals]
    
    bars = ax.bar(x, duration_coefs, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(diseases, fontsize=10)
    ax.set_ylabel("Duration Coefficient", fontsize=11)
    ax.set_title("B. DURATION EFFECT\n(Green=significant p<0.05, Red=not significant)", fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (v, p) in enumerate(zip(duration_coefs, duration_pvals)):
        ax.text(i, v + 0.002 if v > 0 else v - 0.004, f"p={p:.3f}", ha='center', fontsize=8)
    
    plt.suptitle("CUMULATIVE EXPOSURE ANALYSIS: Does Historical Burden Matter?", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = OUT / "cumulative_exposure_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def print_summary(results_df):
    """Print summary of findings."""
    print()
    print("=" * 70)
    print("CUMULATIVE EXPOSURE ANALYSIS SUMMARY")
    print("=" * 70)
    print()
    
    print("MODEL R² COMPARISON (within TRI tracts, controlling for poverty/insurance):")
    print("-" * 70)
    print(f"{'Disease':<15} {'Current':<10} {'Recent3yr':<10} {'Cumulative':<12} {'Duration':<10}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['disease']:<15} {row['model_current_r2']:.4f}    {row['model_recent3yr_r2']:.4f}     "
              f"{row['model_cumulative_r2']:.4f}      {row['model_duration_r2']:.4f}")
    
    print()
    print("KEY FINDINGS:")
    print("-" * 70)
    
    # Check which model performs best on average
    avg_r2 = results_df[['model_current_r2', 'model_recent3yr_r2', 'model_cumulative_r2', 'model_duration_r2']].mean()
    best_model = avg_r2.idxmax()
    
    print(f"1. Best performing model on average: {best_model} (R² = {avg_r2[best_model]:.4f})")
    
    # Check duration significance
    sig_duration = (results_df['duration_pval'] < 0.05).sum()
    print(f"2. Duration significant for {sig_duration}/5 diseases")
    
    # Check if cumulative > current
    cum_better = (results_df['model_cumulative_r2'] > results_df['model_current_r2']).sum()
    print(f"3. Cumulative better than current for {cum_better}/5 diseases")
    
    print()
    print("INTERPRETATION:")
    print("-" * 70)
    
    if avg_r2['model_duration_r2'] > avg_r2['model_cumulative_r2']:
        print("- DURATION matters more than TONNAGE")
        print("  → Supports SELECTIVE MIGRATION hypothesis")
        print("  → Longer exposure time = more population self-selection")
    else:
        print("- CUMULATIVE RELEASES matter more than DURATION")
        print("  → Supports CHEMICAL EXPOSURE hypothesis")
        print("  → Historical burden of pollution has lasting effects")
    
    if avg_r2['model_current_r2'] > avg_r2['model_cumulative_r2']:
        print("- CURRENT releases explain more than CUMULATIVE")
        print("  → Suggests acute exposure effects dominate")
    else:
        print("- CUMULATIVE releases explain more than CURRENT")
        print("  → Suggests chronic/historical exposure matters")
    
    print()


def run_analysis():
    """Main analysis function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    logger.info("=" * 60)
    logger.info("CUMULATIVE EXPOSURE ANALYSIS")
    logger.info("=" * 60)
    
    # Load data
    health, census, fac = load_data()
    
    # Compute cumulative metrics
    cum_metrics = compute_cumulative_metrics(fac)
    
    # Merge all data
    all_tracts = health.merge(census[['fips_tract', 'poverty_pct', 'minority_pct']], on='fips_tract', how='left')
    all_tracts = all_tracts.merge(cum_metrics, on='fips_tract', how='left')
    
    # Fill NAs for non-TRI tracts
    for col in ['total_cumulative', 'recent_3yr', 'recent_5yr', 'early_3yr', 'latest_releases']:
        all_tracts[col] = all_tracts[col].fillna(0)
        all_tracts[f'log_{col}'] = np.log10(all_tracts[col].clip(1))
    
    all_tracts['duration'] = all_tracts['duration'].fillna(0)
    
    logger.info(f"Total tracts: {len(all_tracts):,}")
    logger.info(f"TRI tracts: {(all_tracts['total_cumulative'] > 0).sum():,}")
    
    # Compare models
    results_df = compare_exposure_models(all_tracts)
    
    # Create plot
    plot_model_comparison(results_df)
    
    # Print summary
    print_summary(results_df)
    
    # Save results
    results_df.to_csv(OUT / "cumulative_exposure_results.csv", index=False)
    logger.info(f"Results saved to {OUT / 'cumulative_exposure_results.csv'}")
    
    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)
    
    return results_df


if __name__ == "__main__":
    run_analysis()
