#!/usr/bin/env python3
"""
Run All Pruning Methods and Evaluate

This script:
1. Runs all pruning methods (KGTrimmer, POG, PathRAG, CrumbTrail)
2. Evaluates them using the evaluation framework
3. Generates comparison reports
"""

import sys
from pathlib import Path
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pruning_method(method_name: str, script_path: str):
    """Run a pruning method using its quickstart script."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running {method_name} Pruning")
    logger.info(f"{'='*80}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✓ {method_name} completed successfully")
            return True
        else:
            logger.error(f"✗ {method_name} failed:")
            logger.error(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {method_name} timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"✗ {method_name} error: {e}")
        return False


def main():
    """Run all pruning methods and evaluation."""
    logger.info("="*80)
    logger.info("Running All Pruning Methods and Evaluation")
    logger.info("="*80)
    
    # Check baseline exists
    baseline_dir = Path("workspace/output")
    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        logger.error("Please run Stage 1 (ingest/build_index.py) first!")
        return 1
    
    # Run pruning methods
    methods = [
        ("KGTrimmer", "examples/kgtrimmer_quickstart.py"),
        ("POG", "examples/pog_quickstart.py"),
        ("PathRAG", "examples/pathrag_quickstart.py"),
    ]
    
    # Note: CrumbTrail already exists, so we'll skip it
    
    results = {}
    for method_name, script_path in methods:
        if Path(script_path).exists():
            # Run with example 1 (basic)
            success = run_pruning_method(method_name, script_path)
            results[method_name] = success
        else:
            logger.warning(f"Script not found: {script_path}, skipping {method_name}")
            results[method_name] = False
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Pruning Summary")
    logger.info("="*80)
    for method, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {method}")
    
    # Run evaluation
    logger.info("\n" + "="*80)
    logger.info("Running Evaluation")
    logger.info("="*80)
    
    logger.info("Running evaluation comparison...")
    try:
        eval_result = subprocess.run(
            [
                sys.executable, "eval/run_eval.py",
                "--use-pubmedqa",
                "--pubmedqa-samples", "20",
                "--output-dir", "eval/results"
            ],
            capture_output=True,
            text=True,
            timeout=1800  # 30 min timeout
        )
        
        if eval_result.returncode == 0:
            logger.info("✓ Evaluation completed successfully")
            logger.info("\nEvaluation output:")
            logger.info(eval_result.stdout)
        else:
            logger.error("✗ Evaluation failed:")
            logger.error(eval_result.stderr)
    except Exception as e:
        logger.error(f"✗ Evaluation error: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("Complete!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("  1. Review pruning results in workspace/output/pruned_*/")
    logger.info("  2. Check evaluation results in eval/results/")
    logger.info("  3. Run comparison: python examples/compare_pruning_methods.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

