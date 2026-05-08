"""Environmental Justice Pipeline - Configuration

Loads all settings from config.yml at project root.
Falls back to defaults if config.yml is missing.
"""
import os
from pathlib import Path

import yaml

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yml"


def _load_yaml():
    """Load config.yml or return empty dict."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    return {}


_cfg = _load_yaml()

# --- Pipeline Stages ---
_stages = _cfg.get("stages", {})
STAGE_DOWNLOAD = _stages.get("download", True)
STAGE_PROCESS = _stages.get("process", True)
STAGE_ANALYZE = _stages.get("analyze", True)
STAGE_VISUALIZE = _stages.get("visualize", True)
STAGE_MAP = _stages.get("map", True)

# --- Paths ---
_paths = _cfg.get("paths", {})
DATA_DIR = PROJECT_ROOT / _paths.get("data_dir", "data")
RAW_DIR = PROJECT_ROOT / _paths.get("raw_dir", "data/raw")
PROCESSED_DIR = PROJECT_ROOT / _paths.get("processed_dir", "data/processed")
OUTPUT_DIR = PROJECT_ROOT / _paths.get("output_dir", "output")
FIGURES_DIR = PROJECT_ROOT / _paths.get("figures_dir", "output/figures")

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Data source: TRI ---
_tri = _cfg.get("data", {}).get("tri", {})
TRI_BASE_URL = _tri.get("api_base_url", "https://data.epa.gov/efservice")
TRI_YEARS = _tri.get("years", list(range(2013, 2024)))
TRI_API_LIMIT = _tri.get("api_limit", 10000)
TRI_API_TIMEOUT = _tri.get("api_timeout", 30)
TRI_RAW_FILE = RAW_DIR / "tri_facilities.csv"

# --- Data source: CDC PLACES ---
_cdc = _cfg.get("data", {}).get("cdc_places", {})
CDC_PLACES_URL = _cdc.get("api_url", "https://data.cdc.gov/resource/cwsq-ngmh.csv")
CDC_BATCH_SIZE = _cdc.get("batch_size", 50000)
CDC_MAX_RECORDS = _cdc.get("max_records", 500000)
CDC_MEASURES = _cdc.get("measures", ["CASTHMA", "CANCER", "COPD", "CHD", "DIABETES", "ACCESS2", "MHLTH"])
CDC_RAW_FILE = RAW_DIR / "cdc_places.csv"

# --- Data source: Census ACS ---
_census = _cfg.get("data", {}).get("census_acs", {})
CENSUS_API_BASE = _census.get("api_base_url", "https://api.census.gov/data")
CENSUS_ACS_YEAR = _census.get("year", 2022)
# Variables: poverty, race, income, total pop, median age, 65+ population
CENSUS_VARIABLES = _census.get("variables", 
    "B17001_001E,B17001_002E,"
    "B02001_001E,B02001_002E,"
    "B19013_001E,"
    "B01001_001E,"
    "B01002_001E,"
    "B01001_020E,B01001_021E,B01001_022E,B01001_023E,B01001_024E,B01001_025E,"
    "B01001_044E,B01001_045E,B01001_046E,B01001_047E,B01001_048E,B01001_049E"
)
CENSUS_STATE_FIPS = _census.get("state_fips", [
    "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19","20",
    "21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36",
    "37","38","39","40","41","42","44","45","46","47","48","49","50","51","53","54","55","56",
    "72",
])
CENSUS_RAW_FILE = RAW_DIR / "census_acs.csv"

# --- Use fallback ---
_data_cfg = _cfg.get("data", {})
USE_FALLBACK = _data_cfg.get("use_fallback", False)
RESUME_DOWNLOADS = _data_cfg.get("resume_downloads", True)

# Tract shapefiles
TRACT_SHAPEFILE = RAW_DIR / "tracts.geojson"

# Processed files
MERGED_FILE = PROCESSED_DIR / "facilities_merged.csv"
SCORED_FILE = PROCESSED_DIR / "facilities_scored.csv"
ANALYSIS_FILE = PROCESSED_DIR / "analysis_results.json"

# Output files
MAP_FILE = OUTPUT_DIR / "map.html"
DATASET_FILE = OUTPUT_DIR / "facilities_scored.csv"

# --- Processing ---
_proc = _cfg.get("processing", {})
LAT_MIN = _proc.get("lat_min", 24)
LAT_MAX = _proc.get("lat_max", 50)
LON_MIN = _proc.get("lon_min", -125)
LON_MAX = _proc.get("lon_max", -66)

# --- Scoring ---
_scoring = _cfg.get("scoring", {})
_tox_weights = _scoring.get("toxicity_weights", {})
CARCINOGEN_WEIGHT = _tox_weights.get("carcinogen", 3.0)
HIGH_TOXICITY_WEIGHT = _tox_weights.get("high_toxicity", 2.0)
DEFAULT_TOXICITY_WEIGHT = _tox_weights.get("default", 1.0)

EJ_WEIGHTS = _scoring.get("ej_weights", {
    "toxicity": 0.40,
    "demographic": 0.30,
    "health": 0.20,
    "persistence": 0.10,
})

SEVERITY_THRESHOLDS = _scoring.get("severity_tiers", {
    "Critical": 75,
    "High": 50,
    "Moderate": 25,
    "Low": 0,
})

# --- Visualization ---
_viz = _cfg.get("visualization", {})
VIZ_DPI = _viz.get("dpi", 150)
VIZ_FONT_SIZE = _viz.get("font_size", 11)
VIZ_STYLE = _viz.get("style", "whitegrid")
VIZ_PALETTE = _viz.get("palette", "muted")
VIZ_SCATTER_MAX = _viz.get("scatter_max_points", 5000)
SEVERITY_COLORS = _viz.get("severity_colors", {
    "Critical": "#8e44ad",
    "High": "#e74c3c",
    "Moderate": "#f39c12",
    "Low": "#2ecc71",
})

# --- Map ---
_map = _cfg.get("map", {})
MAP_CENTER_LAT = _map.get("center_lat", 39.5)
MAP_CENTER_LON = _map.get("center_lon", -98.35)
MAP_ZOOM_START = _map.get("zoom_start", 5)
MAP_TILES = _map.get("tiles", "CartoDB positron")
MAP_MARKER_SCALE = _map.get("marker_radius_scale", 0.8)
MAP_MARKER_MIN = _map.get("marker_radius_min", 3)
MAP_MARKER_MAX = _map.get("marker_radius_max", 20)

# --- Fallback ---
_fallback = _cfg.get("fallback", {})
FALLBACK_FACILITIES_PER_YEAR = _fallback.get("facilities_per_year", 2000)
FALLBACK_CENSUS_TRACTS = _fallback.get("census_tracts", 5000)
FALLBACK_RANDOM_SEED = _fallback.get("random_seed", 42)

# Legacy aliases
TRI_TABLE = "TRI_FACILITY"
TRI_RELEASE_TABLE = "TRI_REPORTING_FORM"
