"""Tests for the Environmental Justice pipeline."""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    ANALYSIS_FILE,
    FIGURES_DIR,
    MAP_FILE,
    MERGED_FILE,
    PROCESSED_DIR,
    RAW_DIR,
    SCORED_FILE,
)


logging.basicConfig(level=logging.INFO)


class TestDownload:
    """Test data download and fallback generation."""

    def test_tri_download_or_fallback(self):
        from pipeline.download import download_tri_data
        df = download_tri_data()
        assert len(df) > 0, "TRI data should have records"
        assert "LATITUDE" in df.columns or "latitude" in df.columns.str.lower()
        assert "LONGITUDE" in df.columns or "longitude" in df.columns.str.lower()

    def test_cdc_download_or_fallback(self):
        from pipeline.download import download_cdc_places
        df = download_cdc_places()
        assert len(df) > 0, "CDC data should have records"

    def test_census_download_or_fallback(self):
        from pipeline.download import download_census_acs
        df = download_census_acs()
        assert len(df) > 0, "Census data should have records"


class TestProcess:
    """Test data processing and merging."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure data is downloaded first."""
        from pipeline.download import download_all
        download_all()

    def test_load_and_clean_tri(self):
        from pipeline.process import load_and_clean_tri
        df = load_and_clean_tri()
        assert len(df) > 0
        assert "LATITUDE" in df.columns
        assert "LONGITUDE" in df.columns
        assert "TOTAL_RELEASES" in df.columns
        # Check coordinates are valid US
        assert df["LATITUDE"].min() > 20
        assert df["LATITUDE"].max() < 55
        assert df["LONGITUDE"].min() > -130
        assert df["LONGITUDE"].max() < -60

    def test_load_and_clean_cdc(self):
        from pipeline.process import load_and_clean_cdc
        df = load_and_clean_cdc()
        assert len(df) > 0
        assert "fips_tract" in df.columns

    def test_load_and_clean_census(self):
        from pipeline.process import load_and_clean_census
        df = load_and_clean_census()
        assert len(df) > 0
        assert "fips_tract" in df.columns
        assert "poverty_pct" in df.columns
        assert "minority_pct" in df.columns
        # Check ranges
        assert df["poverty_pct"].min() >= 0
        assert df["poverty_pct"].max() <= 100
        assert df["minority_pct"].min() >= 0
        assert df["minority_pct"].max() <= 100

    def test_merge_and_score(self):
        from pipeline.process import process_all
        df = process_all()
        assert len(df) > 0
        assert "ej_score" in df.columns
        assert "severity_tier" in df.columns
        assert df["ej_score"].min() >= 0
        assert df["ej_score"].max() <= 100
        # All tiers should be present
        tiers = df["severity_tier"].unique()
        assert len(tiers) >= 2, f"Expected multiple severity tiers, got {tiers}"


class TestAnalyze:
    """Test statistical analysis."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not SCORED_FILE.exists():
            from pipeline.download import download_all
            from pipeline.process import process_all
            download_all()
            process_all()

    def test_analyze_all(self):
        from pipeline.analyze import analyze_all
        results = analyze_all()
        assert "summary" in results
        assert "correlations" in results
        assert "disparities" in results

        # Summary checks
        s = results["summary"]
        assert s["total_records"] > 0
        assert s["unique_facilities"] > 0
        assert len(s["years_covered"]) > 0

    def test_correlations_computed(self):
        from pipeline.analyze import analyze_all
        results = analyze_all()
        corr = results["correlations"]["correlations"]
        assert len(corr) > 0, "Should have correlation results"
        for c in corr:
            assert "spearman_rho" in c
            assert "p_value" in c
            assert -1 <= c["spearman_rho"] <= 1

    def test_analysis_file_saved(self):
        from pipeline.analyze import analyze_all
        analyze_all()
        assert ANALYSIS_FILE.exists()
        with open(ANALYSIS_FILE) as f:
            data = json.load(f)
        assert "summary" in data


class TestVisualize:
    """Test chart generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not SCORED_FILE.exists() or not ANALYSIS_FILE.exists():
            from pipeline.download import download_all
            from pipeline.process import process_all
            from pipeline.analyze import analyze_all
            download_all()
            process_all()
            analyze_all()

    def test_visualize_all(self):
        from pipeline.visualize import visualize_all
        visualize_all()

        expected_files = [
            "releases_vs_poverty.png",
            "releases_vs_minority.png",
            "severity_distribution.png",
            "temporal_trends.png",
            "disparity_boxplots.png",
            "top_chemicals.png",
            "state_ej_scores.png",
            "ej_score_histogram.png",
        ]
        for fname in expected_files:
            fpath = FIGURES_DIR / fname
            assert fpath.exists(), f"Expected figure {fname} not found"
            assert fpath.stat().st_size > 1000, f"Figure {fname} seems too small"


class TestMap:
    """Test interactive map generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not SCORED_FILE.exists():
            from pipeline.download import download_all
            from pipeline.process import process_all
            download_all()
            process_all()

    def test_create_map(self):
        from pipeline.map import create_map
        path = create_map()
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "leaflet" in content.lower() or "L.map" in content
        assert len(content) > 10000, "Map HTML seems too small"


class TestEndToEnd:
    """Full pipeline integration test."""

    def test_full_pipeline(self):
        """Run the complete pipeline and verify all outputs."""
        from run_pipeline import run_all
        results = run_all()

        # Check all output files exist
        assert SCORED_FILE.exists(), "Scored dataset not found"
        assert ANALYSIS_FILE.exists(), "Analysis results not found"
        assert MAP_FILE.exists(), "Map HTML not found"

        # Check dataset
        df = pd.read_csv(SCORED_FILE)
        assert len(df) > 0
        assert "ej_score" in df.columns
        assert "severity_tier" in df.columns

        # Check figures
        figures = list(FIGURES_DIR.glob("*.png"))
        assert len(figures) >= 5, f"Expected at least 5 figures, got {len(figures)}"

        # Check analysis results are valid JSON
        with open(ANALYSIS_FILE) as f:
            analysis = json.load(f)
        assert "summary" in analysis

        print(f"\nPipeline test passed!")
        print(f"  Records: {len(df)}")
        print(f"  Facilities: {df['TRI_FACILITY_ID'].nunique()}")
        print(f"  Figures: {len(figures)}")
        print(f"  Map size: {MAP_FILE.stat().st_size / 1024:.0f} KB")
