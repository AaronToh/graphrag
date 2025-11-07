#!/usr/bin/env python3
"""
Run all pruning strategies and show summary statistics.

This script only runs pruning (no evaluation).
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pruning.prune_graph import GraphPruner
from pruning.scoring_utils import load_graphrag_artifacts
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_graph_stats(entities_df: pd.DataFrame, relationships_df: pd.DataFrame) -> dict:
    """Get statistics about the graph."""
    return {
        'num_entities': len(entities_df),
        'num_relationships': len(relationships_df),
        'unique_sources': relationships_df['source'].nunique() if len(relationships_df) > 0 else 0,
        'unique_targets': relationships_df['target'].nunique() if len(relationships_df) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run all pruning strategies and show summary")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace"),
        help="Path to workspace directory (default: workspace)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline artifacts (default: workspace/output)",
    )
    
    args = parser.parse_args()
    
    workspace_path = args.workspace
    baseline_dir = args.baseline or workspace_path / "output"
    
    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return 1
    
    # Load baseline stats
    baseline_entities, baseline_relationships, _ = load_graphrag_artifacts(baseline_dir)
    baseline_stats = get_graph_stats(baseline_entities, baseline_relationships)
    
    logger.info("\n" + "="*80)
    logger.info("BASELINE GRAPH STATISTICS")
    logger.info("="*80)
    logger.info(f"  Entities:        {baseline_stats['num_entities']:,}")
    logger.info(f"  Relationships:   {baseline_stats['num_relationships']:,}")
    logger.info(f"  Unique Sources:  {baseline_stats['unique_sources']:,}")
    logger.info(f"  Unique Targets:  {baseline_stats['unique_targets']:,}")
    logger.info("="*80 + "\n")
    
    strategies = [
        {
            'name': 'crumbtrail',
            'method': 'apply_crumbtrail_pipeline',
            'params': {}
        },
        {
            'name': 'kgtrimmer',
            'method': 'apply_kgtrimmer_pipeline',
            'params': {}
        },
        {
            'name': 'pathrag_hybrid',
            'method': 'apply_pathrag_hybrid_pipeline',
            'params': {}
        },
        {
            'name': 'pog_hybrid',
            'method': 'apply_pog_hybrid_pipeline',
            'params': {}
        },
        {
            'name': 'adaptive_multi_strategy',
            'method': 'apply_adaptive_multi_strategy_pipeline',
            'params': {}
        },
    ]
    
    pruning_results = {}
    
    for strategy in strategies:
        logger.info("\n" + "="*80)
        logger.info(f"PRUNING STRATEGY: {strategy['name'].upper()}")
        logger.info("="*80)
        
        output_dir = workspace_path / f"pruned_{strategy['name']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a new pruner instance for each strategy
        strategy_pruner = GraphPruner(baseline_dir, output_dir)
        
        # Get the method and call it
        method = getattr(strategy_pruner, strategy['method'])
        try:
            logger.info(f"\n📊 BEFORE PRUNING:")
            logger.info(f"  Entities:        {baseline_stats['num_entities']:,}")
            logger.info(f"  Relationships:   {baseline_stats['num_relationships']:,}")
            
            logger.info(f"\n🔄 Running {strategy['name']} pruning algorithm...")
            pruned_artifacts = method(**strategy['params'])
            
            # Get after stats
            pruned_entities = pruned_artifacts['entities']
            pruned_relationships = pruned_artifacts['relationships']
            after_stats = get_graph_stats(pruned_entities, pruned_relationships)
            
            # Calculate reductions
            entity_reduction = ((baseline_stats['num_entities'] - after_stats['num_entities']) / baseline_stats['num_entities'] * 100) if baseline_stats['num_entities'] > 0 else 0
            relationship_reduction = ((baseline_stats['num_relationships'] - after_stats['num_relationships']) / baseline_stats['num_relationships'] * 100) if baseline_stats['num_relationships'] > 0 else 0
            
            logger.info(f"\n📊 AFTER PRUNING:")
            logger.info(f"  Entities:        {after_stats['num_entities']:,} ({entity_reduction:.1f}% reduction)")
            logger.info(f"  Relationships:   {after_stats['num_relationships']:,} ({relationship_reduction:.1f}% reduction)")
            logger.info(f"  Unique Sources:  {after_stats['unique_sources']:,}")
            logger.info(f"  Unique Targets:  {after_stats['unique_targets']:,}")
            
            logger.info(f"\n✅ {strategy['name']} pruning completed successfully!")
            logger.info(f"   Results saved to: {output_dir}")
            logger.info("="*80)
            
            # Store results for summary
            pruning_results[strategy['name']] = {
                'num_entities': after_stats['num_entities'],
                'num_relationships': after_stats['num_relationships'],
                'unique_sources': after_stats['unique_sources'],
                'unique_targets': after_stats['unique_targets'],
                'entity_reduction': entity_reduction,
                'edge_reduction': relationship_reduction
            }
            
        except Exception as e:
            logger.error(f"\n❌ {strategy['name']} pruning failed: {e}")
            import traceback
            traceback.print_exc()
            logger.info("="*80)
    
    # Show final summary
    logger.info("\n" + "="*80)
    logger.info("PRUNING SUMMARY - ALL STRATEGIES")
    logger.info("="*80)
    logger.info(f"\n{'Strategy':<30} {'Entities':<15} {'Relationships':<15} {'Entity %':<12} {'Edge %':<12}")
    logger.info("-" * 80)
    
    for strategy_name, stats in pruning_results.items():
        entity_reduction = stats.get('entity_reduction', 0)
        edge_reduction = stats.get('edge_reduction', 0)
        logger.info(f"{strategy_name:<30} "
                   f"{stats['num_entities']:<15,} "
                   f"{stats['num_relationships']:<15,} "
                   f"{entity_reduction:<12.1f}% "
                   f"{edge_reduction:<12.1f}%")
    
    logger.info("\n" + "="*80)
    logger.info("BASELINE:")
    logger.info(f"  Entities:      {baseline_stats['num_entities']:,}")
    logger.info(f"  Relationships: {baseline_stats['num_relationships']:,}")
    logger.info("="*80)
    
    # Detailed summary
    logger.info("\n" + "="*80)
    logger.info("DETAILED SUMMARY")
    logger.info("="*80)
    for strategy_name, stats in pruning_results.items():
        logger.info(f"\n{strategy_name.upper()}:")
        logger.info(f"  Entities:        {stats['num_entities']:,} ({stats['entity_reduction']:.1f}% reduction)")
        logger.info(f"  Relationships:   {stats['num_relationships']:,} ({stats['edge_reduction']:.1f}% reduction)")
        logger.info(f"  Unique Sources:  {stats['unique_sources']:,}")
        logger.info(f"  Unique Targets:  {stats['unique_targets']:,}")
    
    logger.info("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

