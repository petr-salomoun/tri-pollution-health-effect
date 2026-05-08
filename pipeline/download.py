"""
Step 1: Download data from EPA TRI, CDC PLACES, and Census ACS.

Uses public APIs and bulk download endpoints. No API keys required
for TRI and CDC. Census API works without key (limited rate).
"""
import csv
import io
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from pipeline.config import (
    CDC_BATCH_SIZE,
    CDC_MAX_RECORDS,
    CDC_MEASURES,
    CDC_PLACES_URL,
    CDC_RAW_FILE,
    CENSUS_ACS_YEAR,
    CENSUS_API_BASE,
    CENSUS_RAW_FILE,
    CENSUS_STATE_FIPS,
    CENSUS_VARIABLES,
    FALLBACK_CENSUS_TRACTS,
    FALLBACK_FACILITIES_PER_YEAR,
    FALLBACK_RANDOM_SEED,
    RAW_DIR,
    RESUME_DOWNLOADS,
    TRI_API_LIMIT,
    TRI_API_TIMEOUT,
    TRI_BASE_URL,
    TRI_RAW_FILE,
    TRI_YEARS,
    USE_FALLBACK,
)

logger = logging.getLogger(__name__)


def _fetch_release_chunk(offset, chunk_file):
    """Fetch a single TRI_RELEASE_QTY chunk. Returns (success, records_count)."""
    url = f"{TRI_BASE_URL}/TRI/TRI_RELEASE_QTY/rows/{offset}:{offset + TRI_API_LIMIT}/CSV"
    for attempt in range(1, 6):
        try:
            resp = requests.get(url, timeout=TRI_API_TIMEOUT)
            resp.raise_for_status()
            if len(resp.content.strip()) < 50:
                return "empty", 0
            df_chunk = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            if len(df_chunk) == 0:
                return "empty", 0
            df_chunk.to_csv(chunk_file, index=False)
            return "ok", len(df_chunk)
        except Exception as e:
            if attempt < 5:
                time.sleep(5 * attempt)
            else:
                logger.debug(f"  offset {offset} failed after 5 attempts: {e}")
                return "failed", 0
    return "failed", 0


def _download_tri_release_qty(release_dir, release_qty_file):
    """Download TRI_RELEASE_QTY table with gap-filling and progress reporting."""
    logger.info("=" * 60)
    logger.info("TRI_RELEASE_QTY DOWNLOAD (~32M rows)")
    logger.info("=" * 60)

    # --- Phase 1: Inventory existing chunks ---
    existing_chunks = sorted(release_dir.glob("tri_release_chunk_*.csv"))
    existing_nums = set()
    for f in existing_chunks:
        try:
            num = int(f.stem.split("_")[-1])
            if f.stat().st_size > 100:
                existing_nums.add(num)
        except ValueError:
            pass

    if existing_nums:
        max_existing = max(existing_nums)
        gaps = [i for i in range(max_existing + 1) if i not in existing_nums]
        logger.info(f"Existing: {len(existing_nums)} chunks downloaded "
                    f"(up to chunk {max_existing}, offset {max_existing * TRI_API_LIMIT})")
        if gaps:
            logger.info(f"Gaps found: {len(gaps)} missing chunks: {gaps[:30]}{'...' if len(gaps) > 30 else ''}")
    else:
        max_existing = -1
        gaps = []
        logger.info("No existing chunks found, starting fresh")

    # --- Phase 2: Fill gaps (missing chunks within already-explored range) ---
    if gaps:
        logger.info(f"Phase 1: Filling {len(gaps)} gaps...")
        filled = 0
        still_missing = []
        for gap_num in gaps:
            chunk_file = release_dir / f"tri_release_chunk_{gap_num:06d}.csv"
            offset = gap_num * TRI_API_LIMIT
            status, count = _fetch_release_chunk(offset, chunk_file)
            if status == "ok":
                filled += 1
                if filled % 10 == 0:
                    logger.info(f"  Filled {filled}/{len(gaps)} gaps")
            elif status == "failed":
                still_missing.append(gap_num)
            time.sleep(1.0)
        logger.info(f"Gap-fill complete: {filled} recovered, {len(still_missing)} still missing")
        if still_missing:
            logger.warning(f"Permanently missing chunks: {still_missing[:50]}")

    # --- Phase 3: Continue downloading new chunks beyond max_existing ---
    logger.info(f"Phase 2: Downloading new chunks from chunk {max_existing + 1}...")
    chunk_num = max_existing + 1
    consecutive_empty = 0
    max_consecutive_empty = 50
    downloaded = 0
    failed_offsets = []

    while consecutive_empty < max_consecutive_empty:
        chunk_file = release_dir / f"tri_release_chunk_{chunk_num:06d}.csv"
        offset = chunk_num * TRI_API_LIMIT

        if chunk_file.exists() and chunk_file.stat().st_size > 100:
            chunk_num += 1
            consecutive_empty = 0
            continue

        status, count = _fetch_release_chunk(offset, chunk_file)

        if status == "ok":
            downloaded += 1
            consecutive_empty = 0
            if downloaded % 50 == 0:
                logger.info(f"  Downloaded {downloaded} new chunks "
                            f"(chunk {chunk_num}, offset {offset/1e6:.1f}M)")
            time.sleep(1.0)
        elif status == "empty":
            consecutive_empty = max_consecutive_empty  # End of data
        elif status == "failed":
            failed_offsets.append(offset)
            consecutive_empty += 1
            time.sleep(2)

        chunk_num += 1

    logger.info(f"Download complete: {downloaded} new chunks downloaded")
    if failed_offsets:
        logger.warning(f"{len(failed_offsets)} offsets failed permanently")

    # --- Phase 4: Summary and merge ---
    all_chunks = sorted(release_dir.glob("tri_release_chunk_*.csv"))
    valid_chunks = [f for f in all_chunks if f.stat().st_size > 100]

    if valid_chunks:
        chunk_nums = sorted(int(f.stem.split("_")[-1]) for f in valid_chunks)
        total_range = chunk_nums[-1] + 1 if chunk_nums else 0
        missing = total_range - len(chunk_nums)

        logger.info("=" * 60)
        logger.info(f"TRI_RELEASE_QTY SUMMARY:")
        logger.info(f"  Total chunks: {len(valid_chunks)}")
        logger.info(f"  Chunk range: 0 to {chunk_nums[-1]}")
        logger.info(f"  Missing in range: {missing}")
        logger.info(f"  Est. records: ~{len(valid_chunks) * TRI_API_LIMIT / 1e6:.1f}M")
        logger.info("=" * 60)

        logger.info("Merging chunks into single file...")
        rel_dfs = []
        for f in valid_chunks:
            try:
                rel_dfs.append(pd.read_csv(f, low_memory=False))
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")
        if rel_dfs:
            rel_all = pd.concat(rel_dfs, ignore_index=True)
            rel_all.to_csv(release_qty_file, index=False)
            logger.info(f"Saved {len(rel_all):,} total TRI release records to {release_qty_file}")
    else:
        logger.warning("No TRI_RELEASE_QTY chunks downloaded")


def download_tri_data(years=None, output_file=None):
    """Download TRI facility and release data via EPA Envirofacts API.

    Downloads two tables:
    - TRI_FACILITY: facility info (lat/lon, address, industry)
    - TRI_REPORTING_FORM: chemical releases per year

    Resumable: saves per-year and per-table CSVs.
    Falls back to a simulated dataset if the API is unavailable.
    """
    years = years or TRI_YEARS
    output_file = output_file or TRI_RAW_FILE

    if output_file.exists() and not RESUME_DOWNLOADS:
        logger.info(f"TRI data already exists at {output_file}, skipping download")
        return pd.read_csv(output_file)

    # Skip slow API if USE_FALLBACK is set (config.yml or env var)
    if USE_FALLBACK or os.environ.get("USE_FALLBACK", "0") == "1":
        logger.info("USE_FALLBACK=1, generating representative TRI data")
        return _generate_tri_fallback(years, output_file)

    year_dir = RAW_DIR / "tri_years"
    year_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download TRI_FACILITY table (facility metadata with lat/lon)
    facility_file = RAW_DIR / "tri_facility_info.csv"
    if not (RESUME_DOWNLOADS and facility_file.exists() and facility_file.stat().st_size > 100):
        logger.info("Downloading TRI_FACILITY table (coordinates, addresses)...")
        facility_records = []
        offset = 0
        while True:
            url = f"{TRI_BASE_URL}/TRI/TRI_FACILITY/rows/{offset}:{offset + TRI_API_LIMIT}/CSV"
            try:
                resp = requests.get(url, timeout=TRI_API_TIMEOUT)
                resp.raise_for_status()
                if len(resp.content.strip()) < 50:
                    break
                df_chunk = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                if len(df_chunk) == 0:
                    break
                facility_records.append(df_chunk)
                logger.info(f"TRI_FACILITY offset {offset}: got {len(df_chunk)} records")
                if len(df_chunk) < TRI_API_LIMIT:
                    break
                offset += TRI_API_LIMIT
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"TRI_FACILITY API failed at offset {offset}: {e}")
                break

        if facility_records:
            fac_df = pd.concat(facility_records, ignore_index=True)
            fac_df.to_csv(facility_file, index=False)
            logger.info(f"Saved {len(fac_df)} TRI facility records")
    else:
        logger.info(f"TRI_FACILITY already downloaded, skipping")

    # Step 2: Download TRI_REPORTING_FORM per year (metadata)
    for year in tqdm(years, desc="Downloading TRI reporting forms by year"):
        year_file = year_dir / f"tri_{year}.csv"

        if RESUME_DOWNLOADS and year_file.exists() and year_file.stat().st_size > 100:
            logger.info(f"Year {year} reporting form: already downloaded, skipping")
            continue

        year_records = []
        offset = 0
        while True:
            url = (
                f"{TRI_BASE_URL}/TRI/TRI_REPORTING_FORM"
                f"/REPORTING_YEAR/{year}"
                f"/rows/{offset}:{offset + TRI_API_LIMIT}/CSV"
            )
            try:
                resp = requests.get(url, timeout=TRI_API_TIMEOUT)
                resp.raise_for_status()
                if len(resp.content.strip()) < 50:
                    break
                df_chunk = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                if len(df_chunk) == 0:
                    break
                year_records.append(df_chunk)
                logger.info(f"Year {year} form, offset {offset}: got {len(df_chunk)} records")
                if len(df_chunk) < TRI_API_LIMIT:
                    break
                offset += TRI_API_LIMIT
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"API failed for year {year} offset {offset}: {e}")
                break

        if year_records:
            year_df = pd.concat(year_records, ignore_index=True)
            year_df.to_csv(year_file, index=False)
            logger.info(f"Year {year}: saved {len(year_df)} form records")

    # Step 3: Download TRI_RELEASE_QTY (actual release amounts)
    # This table has no REPORTING_YEAR column - must download all rows (~32M)
    # and join via DOC_CTRL_NUM to TRI_REPORTING_FORM
    release_dir = RAW_DIR / "tri_release_qty"
    release_dir.mkdir(parents=True, exist_ok=True)

    release_qty_file = RAW_DIR / "tri_release_qty.csv"
    if not (RESUME_DOWNLOADS and release_qty_file.exists() and release_qty_file.stat().st_size > 100):
        _download_tri_release_qty(release_dir, release_qty_file)
    else:
        logger.info("TRI_RELEASE_QTY already downloaded, skipping")

    # Step 4: Merge all reporting form year files into final output
    release_files = sorted(year_dir.glob("tri_*.csv"))
    if release_files:
        releases = pd.concat([pd.read_csv(f, low_memory=False) for f in release_files], ignore_index=True)

        # Merge with facility table if available
        if facility_file.exists():
            fac_df = pd.read_csv(facility_file, low_memory=False)
            # Standardize join key
            fac_cols = fac_df.columns.str.upper().str.strip()
            fac_df.columns = fac_cols

            rel_cols = releases.columns.str.upper().str.strip()
            releases.columns = rel_cols

            # Find the join key (TRI_FACILITY_ID or FACILITY_ID)
            fac_id_col = None
            for candidate in ["TRI_FACILITY_ID", "FACILITY_ID"]:
                if candidate in fac_df.columns:
                    fac_id_col = candidate
                    break

            rel_id_col = None
            for candidate in ["TRI_FACILITY_ID", "FACILITY_ID"]:
                if candidate in releases.columns:
                    rel_id_col = candidate
                    break

            if fac_id_col and rel_id_col:
                # Select useful facility columns
                fac_useful = ["LATITUDE", "LONGITUDE", "FACILITY_NAME", "STREET_ADDRESS",
                              "CITY_NAME", "COUNTY_NAME", "ST", "ZIP_CODE",
                              "PRIMARY_NAICS", "INDUSTRY_SECTOR_CODE"]
                fac_keep = [fac_id_col] + [c for c in fac_useful if c in fac_df.columns]
                fac_subset = fac_df[fac_keep].drop_duplicates(subset=[fac_id_col])

                # Rename to match if needed
                if fac_id_col != rel_id_col:
                    fac_subset = fac_subset.rename(columns={fac_id_col: rel_id_col})

                releases = releases.merge(fac_subset, on=rel_id_col, how="left")
                logger.info(f"Merged facility info. Columns now: {list(releases.columns)}")

        releases.to_csv(output_file, index=False)
        logger.info(f"Merged {len(release_files)} year files → {len(releases)} TRI records")
        return releases

    logger.warning("EPA API unavailable. Generating representative TRI dataset.")
    return _generate_tri_fallback(years, output_file)


def _generate_tri_fallback(years, output_file):
    """Generate a representative TRI dataset based on published EPA statistics.

    This uses known distributions from EPA TRI National Analysis reports
    to create a statistically representative dataset for analysis.
    """
    import numpy as np

    np.random.seed(42)

    # Known top TRI states by facility count (from EPA TRI National Analysis)
    state_weights = {
        "TX": 0.08, "OH": 0.06, "PA": 0.06, "IN": 0.05, "IL": 0.05,
        "CA": 0.05, "LA": 0.04, "MI": 0.04, "WI": 0.03, "NC": 0.03,
        "TN": 0.03, "GA": 0.03, "AL": 0.03, "KY": 0.03, "MO": 0.03,
        "NY": 0.03, "SC": 0.02, "VA": 0.02, "NJ": 0.02, "MN": 0.02,
        "AR": 0.02, "MS": 0.02, "IA": 0.02, "OK": 0.02, "KS": 0.02,
        "WV": 0.01, "FL": 0.02, "WA": 0.01, "OR": 0.01, "CO": 0.01,
        "NE": 0.01, "CT": 0.01, "MA": 0.01, "NM": 0.01, "AZ": 0.01,
        "UT": 0.01, "NV": 0.01, "ID": 0.005, "MT": 0.005, "ND": 0.005,
    }

    # State approximate center coordinates
    state_coords = {
        "TX": (31.0, -99.0), "OH": (40.4, -82.7), "PA": (41.2, -77.2),
        "IN": (40.3, -86.1), "IL": (40.6, -89.4), "CA": (36.8, -119.4),
        "LA": (30.5, -91.2), "MI": (44.3, -85.6), "WI": (43.8, -88.8),
        "NC": (35.8, -79.0), "TN": (35.5, -86.6), "GA": (33.0, -83.5),
        "AL": (32.3, -86.9), "KY": (37.8, -84.3), "MO": (38.6, -92.6),
        "NY": (43.0, -75.5), "SC": (33.8, -81.2), "VA": (37.4, -79.5),
        "NJ": (40.1, -74.4), "MN": (46.4, -94.7), "AR": (34.7, -92.4),
        "MS": (32.3, -89.4), "IA": (42.0, -93.2), "OK": (35.0, -97.1),
        "KS": (38.5, -98.8), "WV": (38.9, -80.5), "FL": (27.7, -81.7),
        "WA": (47.4, -120.7), "OR": (43.8, -120.6), "CO": (39.0, -105.8),
        "NE": (41.1, -98.3), "CT": (41.6, -72.7), "MA": (42.4, -71.4),
        "NM": (34.2, -105.6), "AZ": (34.0, -111.1), "UT": (39.3, -111.1),
        "NV": (38.8, -116.4), "ID": (44.1, -114.7), "MT": (46.8, -110.4),
        "ND": (47.5, -100.5),
    }

    # Top TRI chemicals (from EPA TRI National Analysis)
    chemicals = [
        ("Nitrate compounds", False, 1.0),
        ("Hydrochloric acid", False, 1.5),
        ("Sulfuric acid", False, 1.5),
        ("Toluene", False, 1.5),
        ("Methanol", False, 1.0),
        ("Xylene (mixed isomers)", False, 1.5),
        ("Ammonia", False, 1.0),
        ("N-Hexane", False, 1.5),
        ("Styrene", True, 3.0),
        ("Lead", True, 3.0),
        ("Barium compounds", False, 2.0),
        ("Manganese compounds", False, 2.0),
        ("Zinc compounds", False, 1.0),
        ("Copper compounds", False, 1.5),
        ("Chromium compounds", True, 3.0),
        ("Formaldehyde", True, 3.0),
        ("Benzene", True, 3.0),
        ("Ethylene glycol", False, 1.0),
        ("Hydrogen fluoride", False, 2.0),
        ("Nickel compounds", True, 3.0),
    ]

    industry_sectors = [
        "Chemical Manufacturing", "Metal Mining", "Electric Utilities",
        "Primary Metals", "Paper Manufacturing", "Food Processing",
        "Petroleum Refining", "Plastics & Rubber", "Wood Products",
        "Transportation Equipment",
    ]

    n_facilities_per_year = 2000  # Representative sample
    records = []

    states = list(state_weights.keys())
    weights = list(state_weights.values())
    weights = [w / sum(weights) for w in weights]  # Normalize

    # Generate facility IDs that persist across years
    facility_ids = []
    for i in range(n_facilities_per_year):
        st = np.random.choice(states, p=weights)
        fid = f"{st}{i:05d}FACILITY"
        lat, lon = state_coords[st]
        lat += np.random.normal(0, 1.5)
        lon += np.random.normal(0, 1.5)
        sector = np.random.choice(industry_sectors)
        facility_ids.append((fid, st, lat, lon, sector))

    for year in years:
        # Simulate gradual emission reduction trend (known from EPA data)
        year_factor = 1.0 - 0.02 * (year - 2013)  # ~2% annual reduction

        for fid, st, lat, lon, sector in facility_ids:
            # Each facility reports 1-3 chemicals
            n_chems = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            selected_chems = [chemicals[i] for i in np.random.choice(len(chemicals), n_chems, replace=False)]

            for chem_name, is_carcinogen, tox_weight in selected_chems:
                # Log-normal distribution for releases (known from TRI data)
                base_release = np.random.lognormal(mean=8, sigma=2) * year_factor

                # Some facilities have very high releases (heavy tail)
                if np.random.random() < 0.05:
                    base_release *= 10

                county_fips = f"{np.random.randint(1, 200):03d}"
                tract_suffix = f"{np.random.randint(100, 999999):06d}"

                records.append({
                    "TRI_FACILITY_ID": fid,
                    "FACILITY_NAME": f"{sector} Facility {fid[:7]}",
                    "STREET_ADDRESS": f"{np.random.randint(100, 9999)} Industrial Blvd",
                    "CITY_NAME": f"City_{st}_{county_fips}",
                    "COUNTY_NAME": f"County_{county_fips}",
                    "ST": st,
                    "ZIP_CODE": f"{np.random.randint(10000, 99999)}",
                    "LATITUDE": round(lat, 6),
                    "LONGITUDE": round(lon, 6),
                    "FEDERAL_FACILITY": "NO",
                    "INDUSTRY_SECTOR": sector,
                    "CHEMICAL_NAME": chem_name,
                    "CARCINOGEN": "YES" if is_carcinogen else "NO",
                    "UNIT_OF_MEASURE": "Pounds",
                    "ON_SITE_RELEASE_TOTAL": round(max(0, base_release), 2),
                    "OFF_SITE_RELEASE_TOTAL": round(max(0, base_release * np.random.uniform(0, 0.3)), 2),
                    "TOTAL_RELEASES": round(max(0, base_release * (1 + np.random.uniform(0, 0.3))), 2),
                    "REPORTING_YEAR": year,
                    "COUNTY_FIPS": f"{np.random.randint(1, 56):02d}{county_fips}",
                    "TOXICITY_WEIGHT": tox_weight,
                })

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    logger.info(f"Generated {len(df)} representative TRI records to {output_file}")
    return df


def download_cdc_places(output_file=None):
    """Download CDC PLACES health data at census tract level.

    Resumable: saves batch files, skips already-downloaded batches.
    Uses Socrata Open Data API. No API key required.
    """
    output_file = output_file or CDC_RAW_FILE

    if output_file.exists() and not RESUME_DOWNLOADS:
        logger.info(f"CDC PLACES data already exists at {output_file}, skipping")
        return pd.read_csv(output_file)

    if USE_FALLBACK or os.environ.get("USE_FALLBACK", "0") == "1":
        logger.info("USE_FALLBACK=1, generating representative CDC data")
        return _generate_cdc_fallback(output_file)

    batch_dir = RAW_DIR / "cdc_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    measures_filter = ",".join(f"'{m}'" for m in CDC_MEASURES)
    offset = 0
    batch_num = 0

    while True:  # Paginate until no more records (no artificial cap)
        batch_file = batch_dir / f"cdc_batch_{batch_num:04d}.csv"

        if RESUME_DOWNLOADS and batch_file.exists() and batch_file.stat().st_size > 100:
            logger.info(f"CDC batch {batch_num} already downloaded, skipping")
            offset += CDC_BATCH_SIZE
            batch_num += 1
            continue

        url = (
            f"{CDC_PLACES_URL}"
            f"?$limit={CDC_BATCH_SIZE}"
            f"&$offset={offset}"
            f"&$where=measureid in({measures_filter})"
            f"&$select=locationid,locationname,stateabbr,measureid,data_value,totalpopulation,geolocation"
        )

        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            df_chunk = pd.read_csv(io.StringIO(resp.text))

            if len(df_chunk) == 0:
                break

            df_chunk.to_csv(batch_file, index=False)
            logger.info(f"CDC batch {batch_num}, offset {offset}: got {len(df_chunk)} records")

            if len(df_chunk) < CDC_BATCH_SIZE:
                break

            offset += CDC_BATCH_SIZE
            batch_num += 1
            time.sleep(1)

        except Exception as e:
            logger.warning(f"CDC API request failed at offset {offset}: {e}")
            break

    # Merge all batch files
    all_files = sorted(batch_dir.glob("cdc_batch_*.csv"))
    if all_files:
        dfs = [pd.read_csv(f) for f in all_files]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Merged {len(all_files)} CDC batches → {len(df)} records to {output_file}")
        return df

    logger.warning("CDC API unavailable. Generating representative health data.")
    return _generate_cdc_fallback(output_file)


def _generate_cdc_fallback(output_file):
    """Generate representative CDC PLACES-like health data."""
    import numpy as np
    np.random.seed(43)

    # Generate tract-level health data for ~5000 tracts
    n_tracts = 5000
    states = [
        "TX", "OH", "PA", "IN", "IL", "CA", "LA", "MI", "WI", "NC",
        "TN", "GA", "AL", "KY", "MO", "NY", "SC", "VA", "NJ", "MN",
    ]

    measures = {
        "CASTHMA": (9.5, 2.5),    # National avg ~9.5%, std ~2.5
        "CANCER": (6.2, 1.8),
        "COPD": (6.5, 2.5),
        "CHD": (5.8, 2.0),
        "DIABETES": (10.5, 3.5),
        "ACCESS2": (12.0, 6.0),
        "MHLTH": (14.0, 4.0),
    }

    records = []
    for i in range(n_tracts):
        st = np.random.choice(states)
        state_fips = {"TX": "48", "OH": "39", "PA": "42", "IN": "18", "IL": "17",
                      "CA": "06", "LA": "22", "MI": "26", "WI": "55", "NC": "37",
                      "TN": "47", "GA": "13", "AL": "01", "KY": "21", "MO": "29",
                      "NY": "36", "SC": "45", "VA": "51", "NJ": "34", "MN": "27"}
        sfips = state_fips[st]
        county_fips = f"{np.random.randint(1, 200):03d}"
        tract_fips = f"{np.random.randint(100, 999999):06d}"
        location_id = f"{sfips}{county_fips}{tract_fips}"

        # Correlated health outcomes — poorer tracts have worse health
        # This models the known relationship
        deprivation_factor = np.random.uniform(0, 1)

        pop = np.random.randint(1000, 10000)

        for measure, (mean, std) in measures.items():
            # Health values correlate with deprivation
            value = mean + deprivation_factor * std * 2 + np.random.normal(0, std * 0.5)
            value = max(0.5, min(value, mean + 4 * std))

            records.append({
                "locationid": location_id,
                "locationname": f"Tract {location_id}",
                "stateabbr": st,
                "measureid": measure,
                "data_value": round(value, 1),
                "totalpopulation": pop,
                "geolocation": "",
            })

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    logger.info(f"Generated {len(df)} CDC PLACES-like records to {output_file}")
    return df


def download_census_acs(output_file=None):
    """Download Census ACS demographics at tract level.

    Uses Census Bureau API. Free, no key required for small requests.
    """
    output_file = output_file or CENSUS_RAW_FILE

    if output_file.exists():
        logger.info(f"Census ACS data already exists at {output_file}, skipping")
        return pd.read_csv(output_file)

    if USE_FALLBACK or os.environ.get("USE_FALLBACK", "0") == "1":
        logger.info("USE_FALLBACK=1, generating representative Census data")
        return _generate_census_fallback(output_file)

    # Variables: B17001_002E = below poverty, B17001_001E = total for poverty
    # B02001_001E = total pop, B02001_002E = white alone
    # B19013_001E = median household income
    variables = CENSUS_VARIABLES

    state_dir = RAW_DIR / "census_states"
    state_dir.mkdir(parents=True, exist_ok=True)

    for sfips in tqdm(CENSUS_STATE_FIPS, desc="Downloading Census ACS by state"):
        state_file = state_dir / f"census_{sfips}.csv"

        if RESUME_DOWNLOADS and state_file.exists() and state_file.stat().st_size > 100:
            logger.info(f"State {sfips}: already downloaded, skipping")
            continue

        url = (
            f"{CENSUS_API_BASE}/{CENSUS_ACS_YEAR}/acs/acs5"
            f"?get={variables},NAME"
            f"&for=tract:*"
            f"&in=state:{sfips}"
        )

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if len(data) > 1:
                header = data[0]
                rows = data[1:]
                df_chunk = pd.DataFrame(rows, columns=header)
                df_chunk.to_csv(state_file, index=False)
                logger.info(f"State {sfips}: got {len(df_chunk)} tracts")

            time.sleep(0.5)

        except Exception as e:
            logger.warning(f"Census API failed for state {sfips}: {e}")
            continue

    # Merge all state files
    all_files = sorted(state_dir.glob("census_*.csv"))
    if all_files:
        dfs = [pd.read_csv(f) for f in all_files]
        df = pd.concat(dfs, ignore_index=True)
        df["FIPS_TRACT"] = (
            df["state"].astype(str).str.zfill(2) + 
            df["county"].astype(str).str.zfill(3) + 
            df["tract"].astype(str).str.zfill(6)
        )
        df.to_csv(output_file, index=False)
        logger.info(f"Merged {len(all_files)} state files → {len(df)} Census tracts to {output_file}")
        return df

    logger.warning("Census API unavailable. Generating representative demographics.")
    return _generate_census_fallback(output_file)


def _generate_census_fallback(output_file):
    """Generate representative Census ACS-like data."""
    import numpy as np
    np.random.seed(44)

    n_tracts = 5000
    state_fips_map = {
        "TX": "48", "OH": "39", "PA": "42", "IN": "18", "IL": "17",
        "CA": "06", "LA": "22", "MI": "26", "WI": "55", "NC": "37",
        "TN": "47", "GA": "13", "AL": "01", "KY": "21", "MO": "29",
        "NY": "36", "SC": "45", "VA": "51", "NJ": "34", "MN": "27",
    }

    records = []
    for i in range(n_tracts):
        st, sfips = list(state_fips_map.items())[i % len(state_fips_map)]
        county = f"{np.random.randint(1, 200):03d}"
        tract = f"{np.random.randint(100, 999999):06d}"
        fips_tract = f"{sfips}{county}{tract}"

        total_pop = np.random.randint(1000, 10000)
        # National poverty rate ~12.4%, with wide variation
        poverty_rate = np.random.beta(2, 12) * 100  # Right-skewed
        poverty_count = int(total_pop * poverty_rate / 100)

        # National ~57.8% white alone, varies widely
        white_pct = np.clip(np.random.beta(5, 3) * 100, 5, 98)
        white_count = int(total_pop * white_pct / 100)

        # Median income ~$75k nationally, log-normal
        median_income = int(np.random.lognormal(mean=10.9, sigma=0.5))
        median_income = max(15000, min(median_income, 250000))

        records.append({
            "B17001_001E": total_pop,
            "B17001_002E": poverty_count,
            "B02001_001E": total_pop,
            "B02001_002E": white_count,
            "B19013_001E": median_income,
            "NAME": f"Tract {fips_tract}",
            "state": sfips,
            "county": county,
            "tract": tract,
            "FIPS_TRACT": fips_tract,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    logger.info(f"Generated {len(df)} Census-like tract records to {output_file}")
    return df


def download_all():
    """Download all datasets."""
    logger.info("=" * 60)
    logger.info("STEP 1: DOWNLOADING DATA")
    logger.info("=" * 60)

    tri = download_tri_data()
    logger.info(f"TRI data: {len(tri)} records")

    cdc = download_cdc_places()
    logger.info(f"CDC PLACES data: {len(cdc)} records")

    census = download_census_acs()
    logger.info(f"Census ACS data: {len(census)} records")

    return tri, cdc, census


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all()
