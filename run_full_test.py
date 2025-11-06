#!/usr/bin/env python3
"""
Full Test Suite: Run All Pruning Methods and Evaluate

This script:
1. Runs all pruning methods (KGTrimmer, POG, PathRAG)
2. Runs evaluation on each pruned graph
3. Generates comparison reports
"""

import sys
from pathlib import Path
import logging
import subprocess
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pruning_method(method_name: str, pruner, method_func, **kwargs):
    """Run a pruning method."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running {method_name} Pruning")
    logger.info(f"{'='*80}")
    
    try:
        output_dir = Path(f"workspace/output/pruned_{method_name.lower()}")
        
        # Check if already exists
        if (output_dir / "pruned_entities.parquet").exists():
            logger.info(f"✓ {method_name} already exists, skipping")
            return True
        
        artifacts = method_func(**kwargs)
        logger.info(f"✓ {method_name} completed: {len(artifacts['entities'])} entities")
        return True
    except Exception as e:
        logger.error(f"✗ {method_name} failed: {e}", exc_info=True)
        return False


def run_evaluation(pruned_path: Path, method_name: str, num_samples: int = 5):
    """Run evaluation on a pruned graph."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating {method_name}")
    logger.info(f"{'='*80}")
    
    try:
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
            timeout=1800  # 30 min timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✓ {method_name} evaluation completed")
            # Extract metrics from output
            output = result.stdout
            if "Faithfulness Score:" in output:
                logger.info(f"  Results extracted from output")
            return True
        else:
            logger.error(f"✗ {method_name} evaluation failed:")
            logger.error(result.stderr[:500])  # First 500 chars
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {method_name} evaluation timed out")
        return False
    except Exception as e:
        logger.error(f"✗ {method_name} evaluation error: {e}")
        return False


def main():
    """Run full test suite."""
    logger.info("="*80)
    logger.info("Full Test Suite: All Pruning Methods + Evaluation")
    logger.info("="*80)
    
    # Check baseline exists
    baseline_dir = Path("workspace/output")
    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        logger.error("Please run Stage 1 (ingest/build_index.py) first!")
        return 1
    
    # Import pruning functions
    from pruning.prune_graph import GraphPruner
    
    baseline_dir = Path("workspace/output")
    results = {}
    
    # Run pruning methods
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Running Pruning Methods")
    logger.info("="*80)
    
    # KGTrimmer
    pruner_kg = GraphPruner(baseline_dir, Path("workspace/output/pruned_kgtrimmer"))
    results['kgtrimmer'] = run_pruning_method(
        "KGTrimmer", pruner_kg, pruner_kg.apply_kgtrimmer_pipeline,
        min_importance_percentile=0.2, preserve_connectivity=False
    )
    
    # POG
    pruner_pog = GraphPruner(baseline_dir, Path("workspace/output/pruned_pog"))
    results['pog'] = run_pruning_method(
        "POG", pruner_pog, pruner_pog.apply_pog_pipeline,
        num_seeds=50, top_k_paths=100
    )
    
    # PathRAG
    pruner_pathrag = GraphPruner(baseline_dir, Path("workspace/output/pruned_pathrag"))
    results['pathrag'] = run_pruning_method(
        "PathRAG", pruner_pathrag, pruner_pathrag.apply_pathrag_pipeline,
        alpha=0.8, top_n_nodes=40, top_k_paths=15
    )
    
    # Print pruning summary
    logger.info("\n" + "="*80)
    logger.info("Pruning Summary")
    logger.info("="*80)
    for method, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {method.upper()}")
    
    # Run evaluations
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Running Evaluations")
    logger.info("="*80)
    
    eval_results = {}
    num_samples = 5  # Small sample for testing
    
    # Evaluate each method
    methods_to_eval = {
        'kgtrimmer': Path("workspace/output/pruned_kgtrimmer"),
        'pog': Path("workspace/output/pruned_pog"),
        'pathrag': Path("workspace/output/pruned_pathrag"),
    }
    
    for method_name, pruned_path in methods_to_eval.items():
        if results.get(method_name, False) and pruned_path.exists():
            eval_results[method_name] = run_evaluation(pruned_path, method_name, num_samples)
        else:
            logger.warning(f"Skipping {method_name} evaluation (pruning failed or path missing)")
            eval_results[method_name] = False
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    
    logger.info("\nPruning Results:")
    for method, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {method.upper()}")
    
    logger.info("\nEvaluation Results:")
    for method, success in eval_results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {method.upper()}")
    
    logger.info("\n" + "="*80)
    logger.info("Complete!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("  1. Review pruning results in workspace/output/pruned_*/")
    logger.info("  2. Check evaluation results in eval/results/")
    logger.info("  3. Run comparison: python examples/compare_pruning_methods.py")
    logger.info("  4. Run full evaluation: python eval/run_eval.py --use-pubmedqa --pubmedqa-samples 50")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

