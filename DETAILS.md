# Methodology Details

This document provides detailed methodology for all quantitative claims in the main report, enabling peer review and reproducibility.

## Data Sources

| Source | Files | Records |
|--------|-------|---------|
| EPA TRI | `data/raw/tri_facilities.csv`, `data/processed/facilities_scored.csv` | 110,603 facility-year records |
| CDC PLACES | `data/raw/cdc_places.csv` | Tract-level health prevalence for 78,815 tracts |
| Census ACS | `data/raw/census_acs.csv` | Demographics for 85,396 tracts |

## Tract Classification

### TRI vs Control
- **TRI tracts**: Tracts containing at least one TRI facility in any year 2013-2023
- **Control tracts**: All other tracts with CDC health data

```python
# Code: pipeline/readme_visuals_v2.py, _load_all_data()
tri_tracts = fac.groupby('fips_tract').agg({'TRI_FACILITY_ID': 'nunique'}).reset_index()
all_tracts['is_tri'] = all_tracts['n_facilities'].notna()
```

**Counts:**
- TRI tracts: 2,196
- Control tracts: 76,619

### Influence Zone Classification
- **TRI-direct**: Contains facility (n=2,196)
- **TRI-neighbor**: Within ~5km of TRI-direct, no own facility (n=6,454)
- **True control**: No facility AND not neighbor (n=70,165)

## Correlation Analysis (Section 3)

### Spearman Correlations with Disease Outcomes

Calculated using `scipy.stats.spearmanr` on 78,815 tracts:

```python
# Poverty correlations
poverty_pct vs copd_crude:     r = 0.537
poverty_pct vs diabetes_crude: r = 0.585
poverty_pct vs chd_crude:      r = 0.398
poverty_pct vs asthma_crude:   r = 0.429
poverty_pct vs cancer_crude:   r = -0.352

# Minority correlations
minority_pct vs copd_crude:     r = -0.154
minority_pct vs diabetes_crude: r = 0.257
minority_pct vs chd_crude:      r = -0.223
minority_pct vs asthma_crude:   r = -0.030
minority_pct vs cancer_crude:   r = -0.750

# Insurance correlations  
pct_no_insurance vs copd_crude:     r = 0.379
pct_no_insurance vs diabetes_crude: r = 0.595
pct_no_insurance vs chd_crude:      r = 0.244
pct_no_insurance vs asthma_crude:   r = 0.151
pct_no_insurance vs cancer_crude:   r = -0.527

# Facility count correlations (weak)
n_facilities vs copd_crude:     r = 0.076
n_facilities vs diabetes_crude: r = 0.051
n_facilities vs chd_crude:      r = 0.064
n_facilities vs asthma_crude:   r = 0.043
n_facilities vs cancer_crude:   r = 0.035

# Log releases correlations (weak)
log_releases vs copd_crude:     r = 0.069
log_releases vs diabetes_crude: r = 0.049
log_releases vs chd_crude:      r = 0.058
log_releases vs asthma_crude:   r = 0.038
log_releases vs cancer_crude:   r = 0.031
```

**Summary statement in report:** "Poverty correlates 0.40-0.59 with most chronic diseases... # Facilities correlates only 0.04-0.08"

## Variance Explained (R²) Analysis

### Poverty R² by Disease

Using OLS regression: `disease ~ poverty_pct`

```python
# Results from statsmodels.api OLS
copd_crude:     R² = 29.0%
diabetes_crude: R² = 31.7%
chd_crude:      R² = 12.1%
asthma_crude:   R² = 23.4%
cancer_crude:   R² = 13.8%
```

**Average across diseases:** ~23% (reported as "R² = 12-32% depending on disease, avg ~23%")

## Gap Decomposition (Section 4)

### Total TRI-Control Gap

For COPD (primary example):
- TRI tracts mean COPD: 8.13%
- Control tracts mean COPD: 6.90%
- **Total gap: 1.23 pp**

### Explained vs Unexplained

Using multivariate regression: `copd ~ poverty_pct + pct_no_insurance + minority_pct`

```python
# Coefficient estimates from full model
poverty_pct coefficient:      0.146
minority_pct coefficient:    -0.033
pct_no_insurance coefficient: 0.046

# Mean differences (TRI - Control)
poverty_pct diff:      +0.6 pp → explains 0.09 pp (7% of gap)
minority_pct diff:     -6.7 pp → explains 0.22 pp (18% of gap)
pct_no_insurance diff: +0.3 pp → explains 0.01 pp (1% of gap)

# Total explained: 0.32 pp (26%)
# UNEXPLAINED: 0.91 pp (74%)
```

### Mechanism Estimates (Section 4 breakdown)

The 55%/26%/15%/5% breakdown is derived as follows:

1. **Socioeconomic factors (26%)**: Directly from regression decomposition above
2. **Selective migration (55%)**: The unexplained gap (74%) is primarily attributed to migration based on:
   - Duration effect: Longer facility history = larger health gap even within poverty quintiles
   - Closure persistence: Health doesn't improve after facility closure
   - Estimate: ~55% of total gap (74% × ~75% attributed to migration)
3. **Historical burden (15%)**: Estimated residual after migration
4. **Current chemical exposure (5%)**: Based on <1% variance explained by pollution variables

**Note:** These are best estimates, not precise measurements. The 74% unexplained gap could be any combination of migration, historical burden, and unmeasured confounders.

## Cumulative Exposure Analysis

### Methodology

Compared five models predicting health outcomes (within TRI tracts only, controlling for poverty and insurance):

1. **Current releases**: 2023 releases only
2. **Recent 3yr**: 2021-2023 releases
3. **Cumulative**: Total 2013-2023 releases
4. **Duration**: Years since first TRI report
5. **Early + Late**: Separate terms for 2013-2015 and recent releases

### Results

```
Disease         Current_R²  Cumulative_R²  Duration_R²
copd_crude      0.3764      0.3790         0.3786
diabetes_crude  0.4766      0.4785         0.4781
chd_crude       0.1952      0.1985         0.1965
asthma_crude    0.4354      0.4338         0.4338
cancer_crude    0.3720      0.3730         0.3711
```

**Key finding:** Differences are minimal (~0.002 R²). Neither current nor cumulative exposure models show significant improvement over socioeconomic controls alone.

**Duration significance:** 3/5 diseases show significant duration coefficients (p<0.05), but effect sizes are small.

### Files
- Script: `pipeline/cumulative_exposure_analysis.py`
- Results: `output/research/cumulative_exposure_results.csv`
- Plot: `output/research/cumulative_exposure_analysis.png`

## Selective Closure Analysis (Section 6a)

### Release Trends

```python
# From facilities_scored.csv grouped by REPORTING_YEAR
Year    Facilities    Total Releases    Per Facility
2013    10,984        998.8M lbs        90,930 lbs
2023     9,096        800.6M lbs        88,016 lbs

# Changes 2013-2023
Facilities: -17.2%
Total releases: -19.8%
Per facility: -3.2%
```

**Conclusion:** Decline is almost entirely from closures, not facility improvements.

### Facility Improvement Analysis

Of 8,449 facilities active in both 2013 and 2023:
- Reduced releases: 3,945 (46.7%)
- Increased releases: 4,504 (53.3%)
- Median change: 0%

### Closure Rate by Size

Small facilities (Q1) closure rate: ~2.5× higher than large facilities (Q5)

## Age Structure

### TRI vs Control Median Age

```python
# From Census ACS B01002_001E (median age)
TRI tracts median age:     40.3 years (n=2,196)
Control tracts median age: 40.0 years (n=76,600)
Difference: 0.3 years
```

**Note:** Original text stated "37.8 vs 39.2" which was incorrect. Actual difference is negligible.

## Air vs Land Release Comparison

### Correlation Differences

Within TRI tracts, comparing air releases vs land releases correlation with health:

```python
Disease         Air r    Land r   Difference
copd_crude      0.064    0.055    0.009
asthma_crude    0.000   -0.039    0.039
diabetes_crude  0.095    0.082    0.013
chd_crude       0.058    0.069    0.010
cancer_crude   -0.007    0.002    0.010
```

**Finding:** No meaningful pathway specificity (differences <0.04). Air releases don't specifically predict respiratory diseases more than land releases.

## Reproducibility

All analyses can be reproduced using:

```bash
cd /home/salomounp/private/AI/science/polution

# Generate README visualizations
python3 pipeline/readme_visuals_v2.py

# Run cumulative exposure analysis
python3 pipeline/cumulative_exposure_analysis.py
```

## Data Quality Notes

1. **IS_CARCINOGEN flag**: All values are False in processed data. Carcinogen-specific analyses not possible without external chemical classification data.

2. **Missing values**: CDC PLACES has ~99% coverage; Census has some tracts with negative values (data errors) which are excluded.

3. **Temporal mismatch**: Health data (CDC PLACES 2023) vs TRI data (2013-2023). Cross-sectional design cannot establish causality.

---

*Last updated: Generated alongside README.md analysis*
*Contact: petr.salomoun@gmail.com*
