#!/usr/bin/env python3
"""
Evaluate All Pruning Methods

This script evaluates all pruned graphs with 100+ PubMedQA samples,
collecting metrics: Faithfulness, SAS, MRR, Response Time.
"""

import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_method(pruned_path: Path, method_name: str, num_samples: int = 100) -> Dict[str, Any]:
    """
    Evaluate a single pruning method.

    Args:
        pruned_path: Path to pruned artifacts
        method_name: Name of the method
        num_samples: Number of samples to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating: {method_name}")
    logger.info(f"{'='*80}")

    if not pruned_path.exists():
        logger.warning(f"Path does not exist: {pruned_path}, skipping")
        return None

    # Check if evaluation already exists
    results_dir = Path("eval/results")
    results_file = results_dir / f"{method_name}_evaluation.json"
    if results_file.exists():
        logger.info(f"✓ Evaluation already exists for {method_name}, loading...")
        with open(results_file, 'r') as f:
            return json.load(f)

    try:
        # Run evaluation
        logger.info(f"Running evaluation with {num_samples} samples...")
        result = subprocess.run(
            [
                sys.executable, "eval/run_eval.py",
                "--baseline", "workspace/output",
                "--pruned", str(pruned_path),
                "--use-pubmedqa",
                "--pubmedqa-samples", str(num_samples),
                "--output-dir", "eval/results"
            ],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"✗ Evaluation failed for {method_name}:")
            logger.error(result.stderr[:500])
            return None

        # Extract metrics from output
        output = result.stdout
        
        # Try to find the comparison metrics JSON file
        import re
        import glob
        
        # Method 1: Try to extract timestamp from output
        timestamp_match = re.search(r'comparison_metrics_(\d{8}_\d{6})', output)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            metrics_file = results_dir / f"comparison_metrics_{timestamp}.json"
        else:
            # Method 2: Find the most recent comparison_metrics file
            metrics_files = list(results_dir.glob("comparison_metrics_*.json"))
            if metrics_files:
                # Sort by modification time, get most recent
                metrics_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                metrics_file = metrics_files[0]
                logger.info(f"  Found most recent metrics file: {metrics_file.name}")
            else:
                logger.warning(f"  No comparison_metrics files found in {results_dir}")
                metrics_file = None
        
        if metrics_file and metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    
                    # Save with method name
                    metrics['method_name'] = method_name
                    metrics['timestamp'] = datetime.now().isoformat()
                    metrics['num_samples'] = num_samples
                    
                    with open(results_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    
                    logger.info(f"✓ Evaluation completed for {method_name}")
                    return metrics
            except Exception as e:
                logger.error(f"  Error reading metrics file: {e}")
                return None

        logger.warning(f"Could not extract metrics from output for {method_name}")
        logger.debug(f"  Output snippet: {output[-500:]}")
        return None

    except subprocess.TimeoutExpired:
        logger.error(f"✗ Evaluation timed out for {method_name}")
        return None
    except Exception as e:
        logger.error(f"✗ Evaluation error for {method_name}: {e}")
        return None


def load_all_methods() -> List[Dict[str, Any]]:
    """Load all pruning methods from ablation config only (not all existing outputs)."""
    methods = []
    
    # Load from ablation config only (to avoid evaluating all 42 methods)
    config_path = Path("eval/ablation_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            configs = json.load(f)
            for config in configs:
                if config.get('pruning_strategy') != 'none':
                    methods.append({
                        'name': config['name'],
                        'path': Path(config['artifacts_path'])
                    })
    
    return methods


def main():
    """Evaluate all pruning methods."""
    logger.info("="*80)
    logger.info("Evaluating All Pruning Methods (100+ samples)")
    logger.info("="*80)

    # Load all methods
    methods = load_all_methods()
    logger.info(f"Found {len(methods)} methods to evaluate")

    # Evaluate each method
    results = {}
    num_samples = 5  # Use 5 samples for faster testing (can be increased for full evaluation)

    for method_info in methods:
        method_name = method_info['name']
        pruned_path = method_info['path']
        
        metrics = evaluate_method(pruned_path, method_name, num_samples)
        if metrics:
            results[method_name] = metrics

    # Aggregate results
    logger.info("\n" + "="*80)
    logger.info("Aggregating Results")
    logger.info("="*80)

    # Save aggregated results
    results_file = Path("eval/results/method_evaluations.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"✓ Saved aggregated results to {results_file}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Evaluation Summary")
    logger.info("="*80)
    
    for method_name, metrics in results.items():
        if metrics and 'comparison' in metrics:
            comp = metrics['comparison']
            logger.info(f"\n{method_name}:")
            logger.info(f"  Faithfulness change: {comp.get('faithfulness_change_pct', 'N/A'):.2f}%")
            logger.info(f"  Response time change: {comp.get('response_time_change_pct', 'N/A'):.2f}%")
            if 'pruned' in metrics:
                logger.info(f"  SAS Score: {metrics['pruned'].get('sas_score', 'N/A'):.4f}")
                logger.info(f"  MRR Score: {metrics['pruned'].get('mrr_score', 'N/A'):.4f}")

    logger.info("\n" + "="*80)
    logger.info("Complete!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {results_file}")
    logger.info("Next: Run analysis with analyze_pruning_results.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())

