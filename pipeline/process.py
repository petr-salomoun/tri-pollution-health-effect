"""
Step 2: Process and merge TRI, CDC PLACES, and Census ACS data.

- Clean and normalize facility data
- Spatial join: TRI facilities → census tracts via FIPS codes
- Merge demographics and health data
- Compute environmental justice scores
"""
import logging

import numpy as np
import pandas as pd

from pipeline.config import (
    CARCINOGEN_WEIGHT,
    CDC_RAW_FILE,
    CENSUS_RAW_FILE,
    DEFAULT_TOXICITY_WEIGHT,
    EJ_WEIGHTS,
    HIGH_TOXICITY_WEIGHT,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MERGED_FILE,
    RAW_DIR,
    SCORED_FILE,
    SEVERITY_THRESHOLDS,
    TRI_RAW_FILE,
)

logger = logging.getLogger(__name__)


def load_and_clean_tri(filepath=None):
    """Load and clean TRI facility/release data.

    Handles two data formats:
    1. Fallback/synthetic: single CSV with all columns (LATITUDE, TOTAL_RELEASES, etc.)
    2. Real EPA data: TRI_REPORTING_FORM (no lat/lon, no releases) needs joining with
       TRI_FACILITY (lat/lon) and TRI_RELEASE_QTY (releases)

    If the combined file doesn't exist, reads from per-year files in data/raw/tri_years/.
    """
    filepath = filepath or TRI_RAW_FILE

    if filepath.exists():
        df = pd.read_csv(filepath, low_memory=False)
    else:
        # Try reading from per-year files
        year_dir = RAW_DIR / "tri_years"
        year_files = sorted(year_dir.glob("tri_*.csv")) if year_dir.exists() else []
        if year_files:
            logger.info(f"Combined file not found; reading {len(year_files)} per-year files from {year_dir}")
            dfs = [pd.read_csv(f, low_memory=False) for f in year_files]
            df = pd.concat(dfs, ignore_index=True)
            # Save combined for next time
            df.to_csv(filepath, index=False)
            logger.info(f"Merged {len(year_files)} year files → {len(df)} records, saved to {filepath}")
        else:
            raise FileNotFoundError(
                f"No TRI data found. Run the download step first: python run_pipeline.py download"
            )

    # Standardize column names to uppercase
    df.columns = [c.upper().strip().replace(" ", "_") for c in df.columns]

    # Detect data format
    has_coords = "LATITUDE" in df.columns and "LONGITUDE" in df.columns
    has_releases = "TOTAL_RELEASES" in df.columns or any(
        "ON_SITE" in c and "RELEASE" in c for c in df.columns
    )

    if not has_coords or not has_releases:
        logger.info("Detected real EPA TRI format — joining with facility and release tables")
        df = _enrich_real_tri_data(df)

    # Ensure required columns exist with fallback alternatives
    _ensure_column(df, "TRI_FACILITY_ID", ["TRIFD", "FACILITY_ID"])
    _ensure_column(df, "REPORTING_YEAR", ["YEAR"])

    if "TOTAL_RELEASES" not in df.columns:
        # Try to compute from component columns
        on_cols = [c for c in df.columns if "ON_SITE" in c and "RELEASE" in c and "TOTAL" in c]
        off_cols = [c for c in df.columns if "OFF_SITE" in c and "RELEASE" in c and "TOTAL" in c]
        if on_cols:
            df["TOTAL_RELEASES"] = pd.to_numeric(df[on_cols[0]], errors="coerce").fillna(0)
            if off_cols:
                df["TOTAL_RELEASES"] += pd.to_numeric(df[off_cols[0]], errors="coerce").fillna(0)
        elif "TOTAL_RELEASE" in df.columns:
            df["TOTAL_RELEASES"] = pd.to_numeric(df["TOTAL_RELEASE"], errors="coerce").fillna(0)
        elif "ONE_TIME_RELEASE_QTY" in df.columns:
            df["TOTAL_RELEASES"] = pd.to_numeric(df["ONE_TIME_RELEASE_QTY"], errors="coerce").fillna(0)
        else:
            logger.warning("No release quantity columns found. Setting TOTAL_RELEASES to 0.")
            df["TOTAL_RELEASES"] = 0

    # Clean coordinates
    if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
        df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
        df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
        df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

        # Filter valid US coordinates
        df = df[
            (df["LATITUDE"] > LAT_MIN) & (df["LATITUDE"] < LAT_MAX) &
            (df["LONGITUDE"] > LON_MIN) & (df["LONGITUDE"] < LON_MAX)
        ]
    else:
        logger.warning("No LATITUDE/LONGITUDE columns — facility coordinates unavailable")

    # Clean releases
    df["TOTAL_RELEASES"] = pd.to_numeric(df["TOTAL_RELEASES"], errors="coerce").fillna(0)
    df["REPORTING_YEAR"] = pd.to_numeric(df["REPORTING_YEAR"], errors="coerce")

    # Chemical info
    if "CHEMICAL_NAME" not in df.columns:
        for alt in ["CAS_CHEM_NAME", "GENERIC_CHEM_NAME", "CHEM_NAME"]:
            if alt in df.columns:
                df["CHEMICAL_NAME"] = df[alt]
                break

    # Carcinogen flag
    if "CARCINOGEN" in df.columns:
        df["IS_CARCINOGEN"] = df["CARCINOGEN"].astype(str).str.upper().isin(["YES", "Y", "TRUE"])
    else:
        df["IS_CARCINOGEN"] = False

    # Toxicity weight
    if "TOXICITY_WEIGHT" not in df.columns:
        df["TOXICITY_WEIGHT"] = np.where(
            df["IS_CARCINOGEN"], CARCINOGEN_WEIGHT, DEFAULT_TOXICITY_WEIGHT
        )

    # State
    if "ST" not in df.columns:
        for alt in ["STATE", "STATE_ABBR", "FACILITY_STATE"]:
            if alt in df.columns:
                df["ST"] = df[alt]
                break

    # Facility name
    if "FACILITY_NAME" not in df.columns:
        for alt in ["FAC_NAME", "FACILITY_NAME"]:
            if alt in df.columns:
                df["FACILITY_NAME"] = df[alt]
                break
        if "FACILITY_NAME" not in df.columns:
            df["FACILITY_NAME"] = df.get("TRI_FACILITY_ID", "Unknown")

    logger.info(f"Cleaned TRI data: {len(df)} records, {df['TRI_FACILITY_ID'].nunique() if 'TRI_FACILITY_ID' in df.columns else '?'} facilities")
    return df


def _enrich_real_tri_data(df):
    """Enrich real EPA TRI_REPORTING_FORM data with facility info and release quantities."""

    # Try to load and merge TRI_FACILITY table
    facility_file = RAW_DIR / "tri_facility_info.csv"
    if facility_file.exists():
        logger.info(f"Loading TRI_FACILITY from {facility_file}")
        fac = pd.read_csv(facility_file, low_memory=False)
        fac.columns = [c.upper().strip().replace(" ", "_") for c in fac.columns]

        # Use PREF_LATITUDE/LONGITUDE (proper decimal degrees) over FAC_ (encoded format)
        if "PREF_LATITUDE" in fac.columns:
            fac["LATITUDE"] = fac["PREF_LATITUDE"]
        elif "FAC_LATITUDE" in fac.columns:
            fac["LATITUDE"] = fac["FAC_LATITUDE"]
        if "PREF_LONGITUDE" in fac.columns:
            fac["LONGITUDE"] = fac["PREF_LONGITUDE"]
        elif "FAC_LONGITUDE" in fac.columns:
            fac["LONGITUDE"] = fac["FAC_LONGITUDE"]
        if "STATE_ABBR" in fac.columns and "ST" not in fac.columns:
            fac["ST"] = fac["STATE_ABBR"]

        # Find join key
        join_key = "TRI_FACILITY_ID"
        if join_key not in fac.columns:
            for alt in ["FACILITY_ID", "TRIFD"]:
                if alt in fac.columns:
                    fac = fac.rename(columns={alt: join_key})
                    break

        if join_key in fac.columns and join_key in df.columns:
            useful = [join_key, "LATITUDE", "LONGITUDE", "FACILITY_NAME",
                      "STREET_ADDRESS", "CITY_NAME", "COUNTY_NAME", "ST",
                      "ZIP_CODE", "PRIMARY_NAICS", "INDUSTRY_SECTOR_CODE",
                      "STATE_COUNTY_FIPS_CODE"]
            keep = [c for c in useful if c in fac.columns]
            fac_subset = fac[keep].drop_duplicates(subset=[join_key])
            df = df.merge(fac_subset, on=join_key, how="left")
            # EPA stores US longitudes as positive; negate for standard convention
            if "LONGITUDE" in df.columns:
                pos_lon = (df["LONGITUDE"] > 0) & (df["LONGITUDE"] < 180)
                if pos_lon.sum() > pos_lon.size * 0.5:
                    df.loc[pos_lon, "LONGITUDE"] = -df.loc[pos_lon, "LONGITUDE"]
                    logger.info("Negated positive LONGITUDE values to standard convention")
            n_with_coords = df["LATITUDE"].notna().sum() if "LATITUDE" in df.columns else 0
            logger.info(f"Merged facility info: {n_with_coords}/{len(df)} records got coordinates")
    else:
        logger.warning(f"TRI_FACILITY file not found at {facility_file}. Run download with live API to fetch it.")

    # Try to load and merge TRI_RELEASE_QTY table
    release_file = RAW_DIR / "tri_release_qty.csv"
    if release_file.exists():
        logger.info(f"Loading TRI_RELEASE_QTY from {release_file}")
        rel = pd.read_csv(release_file, low_memory=False)
        rel.columns = [c.upper().strip().replace(" ", "_") for c in rel.columns]

        if "DOC_CTRL_NUM" in rel.columns and "DOC_CTRL_NUM" in df.columns:
            rel["TOTAL_RELEASE"] = pd.to_numeric(rel.get("TOTAL_RELEASE", pd.Series(dtype=float)), errors="coerce")
            rel_agg = rel.groupby("DOC_CTRL_NUM")["TOTAL_RELEASE"].sum().reset_index()
            rel_agg = rel_agg.rename(columns={"TOTAL_RELEASE": "TOTAL_RELEASES"})
            df = df.merge(rel_agg, on="DOC_CTRL_NUM", how="left")
            logger.info("Merged release quantities from TRI_RELEASE_QTY")
    else:
        # Use max_amount_of_chem as proxy for releases
        # This is the maximum amount of the reported chemical on-site at any time
        # It's not identical to total releases but correlates and is available
        if "MAX_AMOUNT_OF_CHEM" in df.columns and "TOTAL_RELEASES" not in df.columns:
            # EPA codes: 1=<1lb, 2=1-99, 3=100-999, 4=1000-9999, 5=10000-99999,
            # 6=100000-999999, 7=1000000-9999999, 8=10000000-99999999, 9=100000000-999999999, 10=1B+
            amount_map = {1: 0.5, 2: 50, 3: 550, 4: 5500, 5: 55000,
                          6: 550000, 7: 5500000, 8: 55000000, 9: 550000000, 10: 5000000000}
            df["MAX_AMOUNT_OF_CHEM"] = pd.to_numeric(df["MAX_AMOUNT_OF_CHEM"], errors="coerce")
            df["TOTAL_RELEASES"] = df["MAX_AMOUNT_OF_CHEM"].map(amount_map).fillna(0)
            logger.info("Using MAX_AMOUNT_OF_CHEM range midpoints as release proxy (TRI_RELEASE_QTY not available)")

    return df


def _ensure_column(df, target, alternatives):
    """Ensure target column exists, trying alternatives."""
    if target not in df.columns:
        for alt in alternatives:
            if alt in df.columns:
                df[target] = df[alt]
                return
        logger.warning(f"Column {target} not found (tried: {alternatives})")


def load_and_clean_cdc(filepath=None):
    """Load and pivot CDC PLACES health data to one row per tract."""
    filepath = filepath or CDC_RAW_FILE
    df = pd.read_csv(filepath, low_memory=False)

    # Standardize columns
    df.columns = [c.lower().strip() for c in df.columns]

    # Pivot: one row per tract, columns per measure
    if "measureid" in df.columns and "data_value" in df.columns:
        df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")

        pivot = df.pivot_table(
            index="locationid",
            columns="measureid",
            values="data_value",
            aggfunc="mean",
        ).reset_index()

        pivot.columns = [c.lower() for c in pivot.columns]
        pivot = pivot.rename(columns={"locationid": "fips_tract"})

        # Standardize measure names
        rename_map = {
            "casthma": "asthma_crude",
            "cancer": "cancer_crude",
            "copd": "copd_crude",
            "chd": "chd_crude",
            "diabetes": "diabetes_crude",
            "access2": "pct_no_insurance",
            "mhlth": "mental_health_crude",
        }
        pivot = pivot.rename(columns=rename_map)

        # Also grab population if available
        if "totalpopulation" in df.columns:
            pop = df.groupby("locationid")["totalpopulation"].first().reset_index()
            pop.columns = ["fips_tract", "tract_population"]
            pivot = pivot.merge(pop, on="fips_tract", how="left")

        logger.info(f"CDC PLACES: {len(pivot)} tracts with health data")
        return pivot

    logger.warning("CDC data format unexpected")
    return pd.DataFrame()


def load_and_clean_census(filepath=None):
    """Load and process Census ACS demographics."""
    filepath = filepath or CENSUS_RAW_FILE
    df = pd.read_csv(filepath, low_memory=False)

    result = pd.DataFrame()

    if "FIPS_TRACT" in df.columns:
        result["fips_tract"] = df["FIPS_TRACT"].astype(str)
    elif "state" in df.columns and "county" in df.columns and "tract" in df.columns:
        result["fips_tract"] = (
            df["state"].astype(str).str.zfill(2) +
            df["county"].astype(str).str.zfill(3) +
            df["tract"].astype(str).str.zfill(6)
        )
    else:
        logger.warning("Cannot construct FIPS tract ID from Census data")
        return pd.DataFrame()

    # Poverty rate
    total_pov = pd.to_numeric(df.get("B17001_001E", pd.Series(dtype=float)), errors="coerce")
    below_pov = pd.to_numeric(df.get("B17001_002E", pd.Series(dtype=float)), errors="coerce")
    result["poverty_pct"] = (below_pov / total_pov * 100).round(2)

    # Minority percentage (100% - white alone %)
    total_pop = pd.to_numeric(df.get("B02001_001E", pd.Series(dtype=float)), errors="coerce")
    white = pd.to_numeric(df.get("B02001_002E", pd.Series(dtype=float)), errors="coerce")
    result["minority_pct"] = ((total_pop - white) / total_pop * 100).round(2)

    # Median income
    result["median_income"] = pd.to_numeric(df.get("B19013_001E", pd.Series(dtype=float)), errors="coerce")

    result["total_population"] = total_pop

    # Clean
    result = result.dropna(subset=["poverty_pct", "minority_pct"])
    result["poverty_pct"] = result["poverty_pct"].clip(0, 100)
    result["minority_pct"] = result["minority_pct"].clip(0, 100)

    logger.info(f"Census ACS: {len(result)} tracts with demographics")
    return result


def assign_tracts_to_facilities(tri_df, census_df):
    """Assign census tract FIPS to TRI facilities.

    Uses STATE_COUNTY_FIPS_CODE from TRI_FACILITY to match county-level,
    then picks a random tract within that county. Falls back to state-level matching.
    """
    census_df["state_fips"] = census_df["fips_tract"].str[:2]
    census_df["county_fips"] = census_df["fips_tract"].str[:5]

    # Build full state FIPS → abbreviation map
    _fips_map = {
        "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
        "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
        "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
        "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
        "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
        "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
        "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
        "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
        "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
        "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
        "56": "WY", "72": "PR",
    }
    abbrev_to_fips = {v: k for k, v in _fips_map.items()}

    # Build lookup tables: county_fips -> tracts, state_fips -> tracts
    county_tracts = census_df.groupby("county_fips")["fips_tract"].apply(list).to_dict()
    state_tracts = census_df.groupby("state_fips")["fips_tract"].apply(list).to_dict()
    all_tracts = census_df["fips_tract"].values

    np.random.seed(42)

    # Try matching via STATE_COUNTY_FIPS_CODE first (5-digit county FIPS)
    tri_df = tri_df.copy()
    tri_df["_county_fips"] = None

    if "STATE_COUNTY_FIPS_CODE" in tri_df.columns:
        tri_df["_county_fips"] = tri_df["STATE_COUNTY_FIPS_CODE"].astype(str).str.zfill(5)
    elif "ST" in tri_df.columns:
        tri_df["_state_fips"] = tri_df["ST"].map(abbrev_to_fips)

    # Vectorized assignment: county match → state match → random
    def assign_tract(county_fips, state_abbr):
        if pd.notna(county_fips) and county_fips in county_tracts:
            tracts = county_tracts[county_fips]
            return tracts[np.random.randint(len(tracts))]
        sfips = abbrev_to_fips.get(state_abbr, "")
        if sfips in state_tracts:
            tracts = state_tracts[sfips]
            return tracts[np.random.randint(len(tracts))]
        return all_tracts[np.random.randint(len(all_tracts))] if len(all_tracts) > 0 else None

    # Use vectorized operations where possible
    logger.info("Assigning census tracts to facilities...")
    county_col = tri_df.get("_county_fips", pd.Series(dtype=str))
    st_col = tri_df.get("ST", pd.Series(dtype=str))

    # For large datasets, use a facility-level assignment then broadcast
    # (same facility always gets the same tract)
    fac_key = tri_df.groupby("TRI_FACILITY_ID").first()[["_county_fips", "ST"]].reset_index()
    fac_key["fips_tract"] = [
        assign_tract(row["_county_fips"], row["ST"])
        for _, row in fac_key.iterrows()
    ]
    tri_df = tri_df.merge(fac_key[["TRI_FACILITY_ID", "fips_tract"]], on="TRI_FACILITY_ID", how="left")

    # Clean up temp columns
    tri_df = tri_df.drop(columns=["_county_fips", "_state_fips"], errors="ignore")
    logger.info(f"Assigned tracts to {tri_df['fips_tract'].notna().sum()}/{len(tri_df)} records")

    return tri_df


def merge_datasets(tri_df, cdc_df, census_df):
    """Merge TRI facilities with census demographics and health data."""
    # Assign tracts to facilities
    tri_df = assign_tracts_to_facilities(tri_df, census_df)

    # Aggregate TRI to facility-year level
    facility_agg = tri_df.groupby(
        ["TRI_FACILITY_ID", "REPORTING_YEAR", "fips_tract"]
    ).agg({
        "FACILITY_NAME": "first",
        "LATITUDE": "first",
        "LONGITUDE": "first",
        "ST": "first",
        "TOTAL_RELEASES": "sum",
        "IS_CARCINOGEN": "any",
        "TOXICITY_WEIGHT": "max",
        "CHEMICAL_NAME": lambda x: "; ".join(x.unique()[:5]),
    }).reset_index()

    if "INDUSTRY_SECTOR" in tri_df.columns:
        sector = tri_df.groupby("TRI_FACILITY_ID")["INDUSTRY_SECTOR"].first().reset_index()
        facility_agg = facility_agg.merge(sector, on="TRI_FACILITY_ID", how="left")

    # Ensure fips_tract is string for matching
    facility_agg["fips_tract"] = facility_agg["fips_tract"].astype(str)
    census_df["fips_tract"] = census_df["fips_tract"].astype(str)
    cdc_df["fips_tract"] = cdc_df["fips_tract"].astype(str)

    # Merge with census
    merged = facility_agg.merge(census_df, on="fips_tract", how="left")

    # Merge with CDC health data
    merged = merged.merge(cdc_df, on="fips_tract", how="left")

    logger.info(f"Merged dataset: {len(merged)} facility-year records")
    return merged


def compute_ej_score(df):
    """Compute Environmental Justice score (0-100) for each facility-year."""
    # 1. Toxicity component: log-transformed weighted releases
    df["weighted_releases"] = df["TOTAL_RELEASES"] * df["TOXICITY_WEIGHT"]
    log_releases = np.log1p(df["weighted_releases"])
    toxicity_score = _percentile_rank(log_releases) * 100

    # 2. Demographic vulnerability: poverty + minority percentage
    poverty_norm = _percentile_rank(df["poverty_pct"].fillna(0)) * 100
    minority_norm = _percentile_rank(df["minority_pct"].fillna(0)) * 100
    demographic_score = (poverty_norm + minority_norm) / 2

    # 3. Health burden: asthma + cancer prevalence
    asthma = _percentile_rank(df["asthma_crude"].fillna(0)) * 100 if "asthma_crude" in df.columns else 50
    cancer = _percentile_rank(df["cancer_crude"].fillna(0)) * 100 if "cancer_crude" in df.columns else 50
    health_score = (asthma + cancer) / 2

    # 4. Persistence: number of years facility has reported
    years_reporting = df.groupby("TRI_FACILITY_ID")["REPORTING_YEAR"].transform("nunique")
    max_years = df["REPORTING_YEAR"].nunique()
    persistence_score = (years_reporting / max_years) * 100

    # Composite score
    df["ej_score"] = (
        EJ_WEIGHTS["toxicity"] * toxicity_score +
        EJ_WEIGHTS["demographic"] * demographic_score +
        EJ_WEIGHTS["health"] * health_score +
        EJ_WEIGHTS["persistence"] * persistence_score
    ).round(2)

    # Severity tiers
    df["severity_tier"] = pd.cut(
        df["ej_score"],
        bins=[-1, 25, 50, 75, 101],
        labels=["Low", "Moderate", "High", "Critical"],
    )

    # Component scores for analysis
    df["toxicity_component"] = toxicity_score.round(2)
    df["demographic_component"] = demographic_score.round(2)
    df["health_component"] = health_score.round(2) if isinstance(health_score, pd.Series) else health_score
    df["persistence_component"] = persistence_score.round(2)

    logger.info(f"EJ scores computed. Distribution:")
    logger.info(f"  Critical: {(df['severity_tier'] == 'Critical').sum()}")
    logger.info(f"  High: {(df['severity_tier'] == 'High').sum()}")
    logger.info(f"  Moderate: {(df['severity_tier'] == 'Moderate').sum()}")
    logger.info(f"  Low: {(df['severity_tier'] == 'Low').sum()}")

    return df


def _percentile_rank(series):
    """Compute percentile rank (0-1) for a series."""
    return series.rank(pct=True, method="average").fillna(0.5)


def process_all():
    """Run the full processing pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 2: PROCESSING DATA")
    logger.info("=" * 60)

    tri_df = load_and_clean_tri()
    cdc_df = load_and_clean_cdc()
    census_df = load_and_clean_census()

    merged = merge_datasets(tri_df, cdc_df, census_df)
    merged.to_csv(MERGED_FILE, index=False)
    logger.info(f"Saved merged data to {MERGED_FILE}")

    scored = compute_ej_score(merged)
    scored.to_csv(SCORED_FILE, index=False)
    logger.info(f"Saved scored data to {SCORED_FILE}")

    return scored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_all()
