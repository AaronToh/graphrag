#!/usr/bin/env python3
"""
GraphRAG Pruning Lab - Stage 1: Baseline Index Builder

This script runs Microsoft's GraphRAG indexing pipeline to create baseline artifacts.
It processes input documents and generates entities, relationships, communities, and embeddings.

Usage:
    python ingest/build_index.py [--config CONFIG_FILE] [--overwrite]

Arguments:
    --config: Path to GraphRAG configuration file (default: ../workspace/settings.yaml)
    --overwrite: Overwrite existing output files if they exist
"""

import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add the workspace directory to Python path for GraphRAG
workspace_dir = Path(__file__).parent.parent / "workspace"
sys.path.insert(0, str(workspace_dir))

def setup_logging():
    """Configure logging for the indexing process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('indexing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_input_data(input_dir: Path) -> bool:
    """Check if input data exists and is valid."""
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        return False

    # Check for text files
    text_files = list(input_dir.glob("*.txt")) + list(input_dir.glob("*.md"))
    if not text_files:
        print(f"❌ No .txt or .md files found in {input_dir}")
        return False

    print(f"✅ Found {len(text_files)} input files:")
    for file in text_files[:5]:  # Show first 5
        print(f"   - {file.name}")
    if len(text_files) > 5:
        print(f"   ... and {len(text_files) - 5} more")

    return True

import subprocess

def run_graphrag_index(config_path: Path, logger: logging.Logger) -> bool:
    """Run GraphRAG indexing pipeline via CLI."""
    try:
        logger.info("🚀 Starting GraphRAG indexing pipeline...")

        # Run the CLI command directly
        result = subprocess.run(
            ["pixi", "run", "graphrag", "index", "--config", str(config_path)],
            capture_output=True,
            text=True,
            check=False  # don't raise immediately
        )

        # Log stdout and stderr
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)

        if result.returncode == 0:
            logger.info("✅ GraphRAG indexing completed successfully!")
            return True
        else:
            logger.error(f"❌ GraphRAG indexing failed with exit code {result.returncode}")
            return False

    except FileNotFoundError:
        logger.error("❌ graphrag CLI not found. Did you install it in this pixi env?")
        logger.error("   Try: pixi add graphrag")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error running graphrag: {e}")
        return False

def verify_outputs(output_dir: Path, logger: logging.Logger) -> bool:
    """Verify that all expected output files were created."""
    expected_files = [
        "entities.parquet",
        "relationships.parquet",
        "communities.parquet",
        "community_reports.parquet",
        "text_units.parquet",
        "covariates.parquet"
    ]

    missing_files = []
    for file in expected_files:
        if not (output_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        logger.warning(f"⚠️  Missing expected output files: {missing_files}")
        return False

    # Check LanceDB directory
    lancedb_dir = output_dir / "lancedb"
    if not lancedb_dir.exists():
        logger.warning("⚠️  LanceDB directory not found")
        return False

    logger.info("✅ All expected output artifacts found:")
    for file in expected_files:
        file_path = output_dir / file
        size = file_path.stat().st_size / 1024 / 1024  # MB
        logger.info(".2f")

    return True

def main():
    parser = argparse.ArgumentParser(description="Build GraphRAG baseline index")
    parser.add_argument(
        "--config",
        type=str,
        default="workspace/settings.yaml",
        help="Path to GraphRAG configuration file"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_path = project_root / args.config
    input_dir = project_root / "data" / "input"
    output_dir = project_root / "workspace" / "output"

    # Setup logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("🎯 GraphRAG Pruning Lab - Stage 1: Baseline Index Builder")
    logger.info(f"📁 Project root: {project_root}")
    logger.info(f"⚙️  Config file: {config_path}")
    logger.info(f"📥 Input directory: {input_dir}")
    logger.info(f"📤 Output directory: {output_dir}")

    # Check if output already exists
    if output_dir.exists() and not args.overwrite:
        logger.info("📁 Output directory already exists")
        if verify_outputs(output_dir, logger):
            logger.info("✅ Baseline index already built and verified!")
            return 0
        else:
            logger.warning("⚠️  Existing output appears incomplete, rebuilding...")

    # Check input data
    if not check_input_data(input_dir):
        logger.error("❌ Input validation failed")
        return 1

    # Check config file
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return 1

    # Run GraphRAG indexing
    start_time = datetime.now()
    success = run_graphrag_index(config_path, logger)
    end_time = datetime.now()

    if success:
        duration = end_time - start_time
        logger.info(".1f")

        # Verify outputs
        if verify_outputs(output_dir, logger):
            logger.info("🎉 Stage 1 completed successfully!")
            logger.info("📊 Ready for Stage 2: Pruning Layer")
            return 0
        else:
            logger.error("❌ Output verification failed")
            return 1
    else:
        logger.error("❌ Stage 1 failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
