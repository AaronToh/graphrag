#!/usr/bin/env python3
"""
GraphRAG Pruning Lab - Stage 1: OpenAI-based Index Builder

This script runs Microsoft's GraphRAG indexing pipeline using OpenAI API to create baseline artifacts.
It processes input documents from data/text_input and generates entities, relationships, communities, and embeddings.

Usage:
    python -m ingest.build_index [--config CONFIG_FILE] [--overwrite] [--verbose]

Arguments:
    --config: Path to GraphRAG configuration file (default: workspace/settings.yaml)
    --overwrite: Overwrite existing output files if they exist
    --verbose: Enable verbose logging
"""

import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
from dotenv import load_dotenv

from ingest import PROJECT_ROOT, WORKSPACE_DIR


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure clean logging for the indexing process."""
    from ingest.logging_config import setup_detailed_logging
    
    # Set up logging in ingest/logs directory
    log_dir = PROJECT_ROOT / 'ingest' / 'logs'
    level = logging.DEBUG if verbose else logging.INFO
    
    return setup_detailed_logging(log_dir, level)


def check_input_data(input_dir: Path, logger: logging.Logger) -> bool:
    """Check if input data exists and is valid."""
    if not input_dir.exists():
        logger.error(f"❌ Input directory does not exist: {input_dir}")
        return False

    # Check for text files
    text_files = list(input_dir.glob("*.txt"))
    if not text_files:
        logger.error(f"❌ No .txt files found in {input_dir}")
        return False

    logger.info(f"✅ Found {len(text_files)} text files:")
    for file in text_files[:5]:  # Show first 5
        logger.info(f"   - {file.name}")
    if len(text_files) > 5:
        logger.info(f"   ... and {len(text_files) - 5} more")

    return True


def load_environment_variables(workspace_dir: Path, logger: logging.Logger) -> bool:
    """Load environment variables from .env file and verify API key."""
    env_file = workspace_dir / '.env'
    
    if not env_file.exists():
        logger.error(f"❌ .env file not found: {env_file}")
        logger.error("   Run 'graphrag init --root workspace' to create it")
        return False
    
    try:
        # Load environment variables from .env file
        logger.info("🔧 Loading environment variables...")
        load_dotenv(env_file)
        
        # Check if API key is loaded
        api_key = os.getenv('GRAPHRAG_API_KEY')
        if not api_key:
            logger.error("❌ GRAPHRAG_API_KEY not found in environment!")
            logger.error("   Check your workspace/.env file")
            return False
        
        # Check for placeholder values
        if api_key in ['<API_KEY>', 'your_openai_api_key_here']:
            logger.error("❌ Please set your actual OpenAI API key in workspace/.env")
            logger.error("   Replace the placeholder with your real API key")
            return False
        
        # Show first and last few characters of API key for verification
        key_start = api_key[:8] if len(api_key) >= 8 else api_key[:len(api_key)//2]
        key_end = api_key[-8:] if len(api_key) >= 8 else api_key[len(api_key)//2:]
        logger.info(f"✅ API key loaded: {key_start}...{key_end}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading .env file: {e}")
        return False


def check_openai_config(workspace_dir: Path, logger: logging.Logger) -> bool:
    """Check if OpenAI API configuration is properly set up."""
    env_file = workspace_dir / '.env'
    
    if not env_file.exists():
        logger.error(f"❌ .env file not found: {env_file}")
        logger.error("   Run 'graphrag init --root workspace' to create it")
        return False
    
    # Check if API key is set
    try:
        with open(env_file) as f:
            content = f.read()
            if 'GRAPHRAG_API_KEY=' not in content:
                logger.error("❌ GRAPHRAG_API_KEY not found in .env file")
                return False
            
            if 'GRAPHRAG_API_KEY=<API_KEY>' in content or 'your_openai_api_key_here' in content:
                logger.error("❌ Please set your actual OpenAI API key in workspace/.env")
                logger.error("   Replace the placeholder with your real API key")
                return False
                
        logger.info("✅ OpenAI API configuration appears to be set up")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking .env file: {e}")
        return False


def run_graphrag_index(workspace_dir: Path, logger: logging.Logger) -> bool:
    """Run GraphRAG indexing pipeline using the CLI."""
    try:
        logger.info("🚀 Starting GraphRAG indexing with GPT-4o Mini...")
        
        # Change to workspace directory for GraphRAG execution
        original_cwd = os.getcwd()
        
        try:
            os.chdir(workspace_dir)
            logger.info(f"📁 Working in: {workspace_dir.name}")
            
            # Import and run GraphRAG indexing
            import subprocess
            
            # Use pixi to run GraphRAG index command
            cmd = ["pixi", "run", "graphrag", "index"]
            logger.info("🔄 Running GraphRAG indexing...")
            
            # Run the indexing process
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=workspace_dir
            )
            
            # Log key output lines only
            if result.stdout:
                # Only log important progress lines
                for line in result.stdout.split('\n'):
                    if any(keyword in line.lower() for keyword in ['workflow', 'completed', 'error', 'failed', 'success']):
                        logger.info(f"   {line.strip()}")
            
            if result.stderr and result.returncode != 0:
                logger.warning("⚠️  GraphRAG errors:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        logger.warning(f"   {line}")
            
            if result.returncode == 0:
                logger.info("✅ GraphRAG indexing completed successfully!")
                return True
            else:
                logger.error(f"❌ GraphRAG indexing failed (exit code: {result.returncode})")
                if result.stderr:
                    logger.error("Error details logged above")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error running GraphRAG indexing: {e}")
            logger.exception("Full traceback:")
            return False
        finally:
            os.chdir(original_cwd)
            logger.info(f"📁 Restored working directory to: {original_cwd}")
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in indexing: {e}")
        return False


def verify_outputs(output_dir: Path, logger: logging.Logger) -> bool:
    """Verify that all expected output files were created."""
    expected_files = [
        "entities.parquet",
        "relationships.parquet", 
        "communities.parquet",
        "community_reports.parquet",
        "text_units.parquet"
    ]

    missing_files = []
    for file in expected_files:
        file_path = output_dir / file
        if not file_path.exists():
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
        if file_path.exists():
            size = file_path.stat().st_size / 1024 / 1024  # MB
            logger.info(f"   - {file}: {size:.2f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(description="Build GraphRAG baseline index using OpenAI API")
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
    config_path = PROJECT_ROOT / args.config
    workspace_dir = config_path.parent
    input_dir = PROJECT_ROOT / "data" / "text_input"
    output_dir = workspace_dir / "output"

    # Setup logging
    logger = setup_logging(args.verbose)

    logger.info("🎯 GraphRAG Index Builder - Using GPT-4o Mini")
    logger.info(f"📁 Project root: {PROJECT_ROOT}")
    logger.info(f"⚙️  Config: {config_path.name}")
    logger.info(f"📥 Input: {input_dir.name} ({len(list(input_dir.glob('*.txt')))} files)")
    logger.info(f"📤 Output: {output_dir}")

    # Load environment variables and check API key
    if not load_environment_variables(workspace_dir, logger):
        logger.error("❌ Environment setup failed")
        return 1

    # Check if output already exists
    if output_dir.exists() and not args.overwrite:
        logger.info("📁 Output directory already exists")
        if verify_outputs(output_dir, logger):
            logger.info("✅ Baseline index already built and verified!")
            return 0
        else:
            logger.warning("⚠️  Existing output appears incomplete, rebuilding...")

    # Check input data
    if not check_input_data(input_dir, logger):
        logger.error("❌ Input validation failed")
        return 1

    # Check config file
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return 1

    # Run GraphRAG indexing
    start_time = datetime.now()
    indexing_success = run_graphrag_index(workspace_dir, logger)
    end_time = datetime.now()

    success = indexing_success
    if indexing_success:
        duration = end_time - start_time
        logger.info(f"⏱️  Indexing completed in {duration.total_seconds():.1f} seconds")

        # Verify outputs
        if verify_outputs(output_dir, logger):
            logger.info("🎉 Stage 1 completed successfully!")
            logger.info("📊 Ready for Stage 2: Pruning Layer")
        else:
            logger.error("❌ Output verification failed")
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())