#!/usr/bin/env python3
"""
Test All Pruning Methods and Run Evaluation

This script runs all pruning methods (KGTrimmer, POG, PathRAG, CrumbTrail)
and then evaluates them using the evaluation framework.
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pruning.prune_graph import GraphPruner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_kgtrimmer():
    """Run KGTrimmer pruning."""
    logger.info("\n" + "="*80)
    logger.info("Running KGTrimmer Pruning")
    logger.info("="*80)
    
    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_kgtrimmer")
    
    pruner = GraphPruner(baseline_dir, output_dir)
    artifacts = pruner.apply_kgtrimmer_pipeline(
        collective_weight=0.5,
        holistic_weight=0.5,
        min_importance_percentile=0.2,
        preserve_connectivity=True,
        max_iterations=10
    )
    
    logger.info(f"✓ KGTrimmer completed: {artifacts['metadata']['pruned_stats']['num_entities']} entities")
    return artifacts


def run_pog():
    """Run POG pruning."""
    logger.info("\n" + "="*80)
    logger.info("Running POG Pruning")
    logger.info("="*80)
    
    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_pog")
    
    pruner = GraphPruner(baseline_dir, output_dir)
    artifacts = pruner.apply_pog_pipeline(
        seed_method='degree_centrality',
        num_seeds=50,
        max_path_length=5,
        top_k_paths=100,
        llm_provider='openai',
        llm_model='gpt-4o-mini',
        sbert_model='sentence-transformers/all-MiniLM-L6-v2',
        semantic_threshold=0.7
    )
    
    logger.info(f"✓ POG completed: {artifacts['metadata']['pruned_stats']['num_entities']} entities")
    return artifacts


def run_pathrag():
    """Run PathRAG pruning."""
    logger.info("\n" + "="*80)
    logger.info("Running PathRAG Pruning")
    logger.info("="*80)
    
    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_pathrag")
    
    pruner = GraphPruner(baseline_dir, output_dir)
    artifacts = pruner.apply_pathrag_pipeline(
        alpha=0.8,
        theta=0.05,
        top_n_nodes=40,
        top_k_paths=15,
        max_path_length=5,
        seed_method='degree_centrality',
        path_scoring_method='avg_edge_flow'
    )
    
    logger.info(f"✓ PathRAG completed: {artifacts['metadata']['pruned_stats']['num_entities']} entities")
    return artifacts


def run_crumbtrail():
    """Run CrumbTrail pruning (if not already done)."""
    logger.info("\n" + "="*80)
    logger.info("Running CrumbTrail Pruning")
    logger.info("="*80)
    
    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_crumbtrail")
    
    # Check if already exists
    if (output_dir / "pruned_entities.parquet").exists():
        logger.info("✓ CrumbTrail already exists, skipping")
        return None
    
    pruner = GraphPruner(baseline_dir, output_dir)
    artifacts = pruner.apply_crumbtrail_pipeline(
        root_entity=None,
        protected_fraction=0.2,
        protected_selection='degree_centrality',
        max_iterations=1000
    )
    
    logger.info(f"✓ CrumbTrail completed: {artifacts['metadata']['pruned_stats']['num_entities']} entities")
    return artifacts


def main():
    """Run all pruning methods."""
    logger.info("="*80)
    logger.info("Testing All Pruning Methods")
    logger.info("="*80)
    
    baseline_dir = Path("workspace/output")
    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        logger.error("Please run Stage 1 (ingest/build_index.py) first!")
        return
    
    results = {}
    
    # Run all pruning methods
    try:
        results['crumbtrail'] = run_crumbtrail()
    except Exception as e:
        logger.error(f"CrumbTrail failed: {e}", exc_info=True)
    
    try:
        results['kgtrimmer'] = run_kgtrimmer()
    except Exception as e:
        logger.error(f"KGTrimmer failed: {e}", exc_info=True)
    
    try:
        results['pog'] = run_pog()
    except Exception as e:
        logger.error(f"POG failed: {e}", exc_info=True)
    
    try:
        results['pathrag'] = run_pathrag()
    except Exception as e:
        logger.error(f"PathRAG failed: {e}", exc_info=True)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Pruning Summary")
    logger.info("="*80)
    
    for method, artifacts in results.items():
        if artifacts:
            baseline = artifacts['metadata']['baseline_stats']
            pruned = artifacts['metadata']['pruned_stats']
            entity_reduction = 100 * (1 - pruned['num_entities'] / baseline['num_entities'])
            logger.info(f"{method.upper()}: {pruned['num_entities']} entities ({entity_reduction:.1f}% reduction)")
    
    logger.info("\n✓ All pruning methods completed!")
    logger.info("Next: Run evaluation with: python eval/run_eval.py --use-pubmedqa --pubmedqa-samples 20")


if __name__ == "__main__":
    main()

