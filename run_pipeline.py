#!/usr/bin/env python3
"""
Environmental Justice Pipeline - Main Entry Point

Usage:
    python run_pipeline.py              # Run full pipeline
    python run_pipeline.py download     # Only download data
    python run_pipeline.py process      # Only process/merge
    python run_pipeline.py analyze      # Only analyze
    python run_pipeline.py visualize    # Only generate charts
    python run_pipeline.py map          # Only generate map
"""
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def run_download():
    from pipeline.download import download_all
    return download_all()


def run_process():
    from pipeline.process import process_all
    return process_all()


def run_analyze(df=None):
    from pipeline.analyze import analyze_all
    return analyze_all(df)


def run_visualize(df=None):
    from pipeline.visualize import visualize_all
    return visualize_all(df)


def run_map(df=None):
    from pipeline.map import create_map
    return create_map(df)


def run_all():
    """Run the complete pipeline end-to-end, respecting stage flags in config.yml."""
    from pipeline.config import (
        STAGE_DOWNLOAD, STAGE_PROCESS, STAGE_ANALYZE,
        STAGE_VISUALIZE, STAGE_MAP, SCORED_FILE, DATASET_FILE,
    )
    import pandas as pd

    start = time.time()
    logger.info("=" * 70)
    logger.info("ENVIRONMENTAL JUSTICE PIPELINE - FULL RUN")
    logger.info("=" * 70)

    scored_df = None
    results = None

    # Step 1: Download
    if STAGE_DOWNLOAD:
        tri, cdc, census = run_download()
    else:
        logger.info("SKIP: download (disabled in config.yml)")

    # Step 2: Process and merge
    if STAGE_PROCESS:
        scored_df = run_process()
    else:
        logger.info("SKIP: process (disabled in config.yml)")
        if SCORED_FILE.exists():
            scored_df = pd.read_csv(SCORED_FILE, low_memory=False)

    # Step 3: Analyze
    if STAGE_ANALYZE:
        results = run_analyze(scored_df)
    else:
        logger.info("SKIP: analyze (disabled in config.yml)")

    # Step 4: Visualize
    if STAGE_VISUALIZE:
        run_visualize(scored_df)
    else:
        logger.info("SKIP: visualize (disabled in config.yml)")

    # Step 5: Map
    if STAGE_MAP:
        run_map(scored_df)
    else:
        logger.info("SKIP: map (disabled in config.yml)")

    # Copy final dataset to output
    import shutil
    if SCORED_FILE.exists():
        shutil.copy2(SCORED_FILE, DATASET_FILE)
        logger.info(f"Final dataset copied to {DATASET_FILE}")

    elapsed = time.time() - start
    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    logger.info(f"  Dataset: {DATASET_FILE}")
    logger.info(f"  Map: output/map.html")
    logger.info(f"  Figures: output/figures/")
    logger.info("=" * 70)

    return results


def main():
    steps = {
        "download": run_download,
        "process": run_process,
        "analyze": run_analyze,
        "visualize": run_visualize,
        "map": run_map,
    }

    if len(sys.argv) > 1:
        step = sys.argv[1].lower()
        if step in steps:
            steps[step]()
        else:
            print(f"Unknown step: {step}")
            print(f"Available: {', '.join(steps.keys())}")
            sys.exit(1)
    else:
        run_all()


if __name__ == "__main__":
    main()
