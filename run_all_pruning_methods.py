#!/usr/bin/env python3
"""
Run All Pruning Methods from Ablation Config

This script loads all configurations from eval/ablation_config.json and runs
each pruning method, saving results with metadata.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from pruning.prune_graph import GraphPruner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pruning_method(config: Dict[str, Any], baseline_dir: Path) -> bool:
    """
    Run a single pruning method based on configuration.

    Args:
        config: Configuration dictionary from ablation_config.json
        baseline_dir: Path to baseline artifacts

    Returns:
        True if successful, False otherwise
    """
    method_name = config['name']
    artifacts_path = Path(config['artifacts_path'])
    strategy = config.get('pruning_strategy', 'none')

    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {method_name}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"{'='*80}")

    # Check if already exists
    if (artifacts_path / "pruning_metadata.json").exists():
        logger.info(f"✓ {method_name} already exists, skipping")
        return True

    try:
        # Create pruner
        pruner = GraphPruner(baseline_dir, artifacts_path)

        # Run appropriate method based on strategy
        if strategy == 'none':
            logger.info("Skipping baseline (no pruning)")
            return True

        elif strategy == 'top_k':
            k = config.get('k', 50)
            target = config.get('target', 'nodes')
            pruner.apply_top_k_pipeline(k=k, target=target)

        elif strategy == 'threshold':
            threshold = config.get('threshold', 0.5)
            target = config.get('target', 'nodes')
            pruner.apply_threshold_pipeline(threshold=threshold, target=target)

        elif strategy == 'crumbtrail':
            protected_fraction = config.get('protected_fraction', 0.2)
            protected_selection = config.get('protected_selection', 'degree_centrality')
            pruner.apply_crumbtrail_pipeline(
                protected_fraction=protected_fraction,
                protected_selection=protected_selection
            )

        elif strategy == 'kgtrimmer':
            collective_weight = config.get('collective_weight', 0.5)
            holistic_weight = config.get('holistic_weight', 0.5)
            min_importance_percentile = config.get('min_importance_percentile', 0.2)
            pruner.apply_kgtrimmer_pipeline(
                collective_weight=collective_weight,
                holistic_weight=holistic_weight,
                min_importance_percentile=min_importance_percentile,
                preserve_connectivity=True
            )

        elif strategy == 'pog':
            num_seeds = config.get('num_seeds', 50)
            top_k_paths = config.get('top_k_paths', 100)
            max_path_length = config.get('max_path_length', 5)
            pruner.apply_pog_pipeline(
                num_seeds=num_seeds,
                top_k_paths=top_k_paths,
                max_path_length=max_path_length
            )

        elif strategy == 'pathrag':
            alpha = config.get('alpha', 0.8)
            theta = config.get('theta', 0.05)
            top_n_nodes = config.get('top_n_nodes', 40)
            top_k_paths = config.get('top_k_paths', 15)
            max_path_length = config.get('max_path_length', 5)
            pruner.apply_pathrag_pipeline(
                alpha=alpha,
                theta=theta,
                top_n_nodes=top_n_nodes,
                top_k_paths=top_k_paths,
                max_path_length=max_path_length
            )

        elif strategy == 'pathrag_hybrid':
            top_n_nodes = config.get('top_n_nodes', 500)
            top_k_paths = config.get('top_k_paths', 5000)
            max_path_length = config.get('max_path_length', 6)
            node_retention_pct = config.get('node_retention_pct', 0.3)
            node_scoring_method = config.get('node_scoring_method', 'degree_centrality')
            alpha = config.get('alpha', 0.8)
            theta = config.get('theta', 0.05)
            pruner.apply_pathrag_hybrid_pipeline(
                top_n_nodes=top_n_nodes,
                top_k_paths=top_k_paths,
                max_path_length=max_path_length,
                node_retention_pct=node_retention_pct,
                node_scoring_method=node_scoring_method,
                alpha=alpha,
                theta=theta
            )

        elif strategy == 'pog_hybrid':
            num_seeds = config.get('num_seeds', 300)
            top_k_paths = config.get('top_k_paths', 5000)
            max_path_length = config.get('max_path_length', 7)
            node_retention_pct = config.get('node_retention_pct', 0.3)
            node_scoring_method = config.get('node_scoring_method', 'degree_centrality')
            pruner.apply_pog_hybrid_pipeline(
                num_seeds=num_seeds,
                top_k_paths=top_k_paths,
                max_path_length=max_path_length,
                node_retention_pct=node_retention_pct,
                node_scoring_method=node_scoring_method
            )

        elif strategy == 'adaptive_multi_strategy':
            target_reduction = config.get('target_reduction', 0.55)
            min_connectivity_pct = config.get('min_connectivity_pct', 0.90)
            protected_fraction = config.get('protected_fraction', 0.20)
            hub_degree_percentile = config.get('hub_degree_percentile', 0.75)
            pruner.apply_adaptive_multi_strategy_pipeline(
                target_reduction=target_reduction,
                min_connectivity_pct=min_connectivity_pct,
                protected_fraction=protected_fraction,
                hub_degree_percentile=hub_degree_percentile
            )

        elif strategy == 'combined':
            node_k = config.get('node_k', 70)
            edge_k = config.get('edge_k', 5)
            pruner.apply_combined_pipeline(node_k=node_k, edge_k=edge_k)

        elif strategy == 'edges_top_k':
            # Handle edges_top_5 case
            k = config.get('k', 5)
            pruner.apply_edges_top_k_pipeline(k=k)

        else:
            logger.error(f"Unknown strategy: {strategy}")
            return False

        logger.info(f"✓ {method_name} completed successfully")
        return True

    except Exception as e:
        logger.error(f"✗ {method_name} failed: {e}", exc_info=True)
        return False


def main():
    """Run all pruning methods from ablation config."""
    logger.info("="*80)
    logger.info("Running All Pruning Methods from Ablation Config")
    logger.info("="*80)

    # Load config
    config_path = Path("eval/ablation_config.json")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    with open(config_path, 'r') as f:
        configs = json.load(f)

    # Check baseline
    baseline_dir = Path("workspace/output")
    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        logger.error("Please run Stage 1 (ingest/build_index.py) first!")
        return 1

    # Run each method
    results = {}
    for config in configs:
        method_name = config['name']
        success = run_pruning_method(config, baseline_dir)
        results[method_name] = success

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Summary")
    logger.info("="*80)
    
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    logger.info(f"✓ Successful: {len(successful)}/{len(results)}")
    for name in successful:
        logger.info(f"  ✓ {name}")

    if failed:
        logger.info(f"\n✗ Failed: {len(failed)}/{len(results)}")
        for name in failed:
            logger.info(f"  ✗ {name}")

    logger.info("\n" + "="*80)
    logger.info("Complete!")
    logger.info("="*80)
    logger.info("\nNext: Run evaluation with eval_all_pruning_methods.py")

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

