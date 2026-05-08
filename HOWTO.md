# Reproducing This Analysis

This document explains how to reproduce the figures and analysis from scratch using publicly available data.

## Prerequisites

- Python 3.10+
- ~2 GB disk space

```bash
pip install -r requirements.txt
```

## Quick Start (Synthetic Data)

For a fast end-to-end test using statistically representative synthetic data:

```bash
USE_FALLBACK=1 python run_pipeline.py
```

Completes in ~16 seconds. Useful for validating the pipeline and code.

## Full Reproduction (Real Data)

To reproduce with the actual EPA TRI, CDC PLACES, and Census ACS data:

```bash
python run_pipeline.py
```

**Warning:** The EPA Envirofacts API is slow (~10K rows per request, ~2 min each).
Downloading 11 years of TRI data (~800K rows) takes **8+ hours**.
CDC and Census downloads are much faster.

## Pipeline Steps

Run individual steps:

```bash
python run_pipeline.py download    # Step 1: Download EPA/CDC/Census data
python run_pipeline.py process     # Step 2: Clean, merge, compute metrics
python run_pipeline.py analyze     # Step 3: Statistical analysis
python run_pipeline.py visualize   # Step 4: Generate EJ score charts
python run_pipeline.py map         # Step 5: Interactive map (output/map.html)
```

## Regenerate README Figures

```bash
python pipeline/readme_visuals_v2.py
```

Regenerates the 12 figures used in the main report from already-processed data.
Requires `data/processed/facilities_scored.csv` (output of `process` step).

## Data Sources

All source data is freely available:

| Source | URL |
|--------|-----|
| EPA TRI | https://www.epa.gov/toxics-release-inventory-tri-program |
| CDC PLACES | https://chronicdata.cdc.gov/browse?category=500+Cities+%26+Places |
| Census ACS | https://www.census.gov/programs-surveys/acs/data/summary-file.html |

The pipeline downloads these automatically via public APIs.

## Directory Structure

```
pipeline/           Analysis code
├── download.py     Data acquisition (EPA/CDC/Census APIs)
├── process.py      Cleaning, merging, EJ score computation
├── analyze.py      Statistical summaries
├── visualize.py    EJ score visualizations
├── map.py          Interactive map
├── readme_visuals_v2.py  Main report figures (12 plots)
├── research.py     Extended research analyses
├── hypotheses_round2.py  Hypothesis testing round 2
├── hypotheses_round3.py  Hypothesis testing round 3
├── new_hypotheses.py     Additional hypotheses
├── cumulative_exposure_analysis.py  Exposure duration analysis
└── config.py       Configuration constants

tests/
└── test_pipeline.py   Test suite (pytest)

output/
├── readme/         Figures for main report (README.md)
└── research/       Figures from extended research
```

## Running Tests

```bash
USE_FALLBACK=1 python -m pytest tests/test_pipeline.py -v
```

All 13 tests should pass. Tests cover: download, processing, analysis,
visualization, map generation, and end-to-end integration.

## Configuration

Edit `pipeline/config.py` to adjust:
- `TRI_YEARS`: Which years to download (default: 2013–2023)
- `EJ_WEIGHTS`: Weights for the composite EJ score
- `SEVERITY_THRESHOLDS`: Score cutoffs for severity tiers
- `CDC_MAX_RECORDS`: How much CDC PLACES data to fetch

## Troubleshooting

| Issue | Solution |
|-------|----------|
| EPA API timeout | Use `USE_FALLBACK=1` or increase timeout in `download.py` |
| Census API rate limit | Add API key or reduce states in `download.py` |
| Missing `rtree` | `pip install rtree` (needs libspatialindex) |
| Seaborn warnings | Cosmetic only, safe to ignore |
| Map file too large | Reduce facilities in `map.py` by filtering to fewer states |
