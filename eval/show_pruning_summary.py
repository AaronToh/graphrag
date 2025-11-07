#!/usr/bin/env python3
"""
Display summary statistics from already-pruned graphs.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    parser = argparse.ArgumentParser(description="Show pruning summary from existing pruned graphs")
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
    print("Loading baseline statistics...")
    logger.info("Loading baseline statistics...")
    try:
        baseline_entities, baseline_relationships, _ = load_graphrag_artifacts(baseline_dir)
        baseline_stats = get_graph_stats(baseline_entities, baseline_relationships)
    except Exception as e:
        print(f"Error loading baseline: {e}")
        logger.error(f"Error loading baseline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*80)
    print("BASELINE GRAPH STATISTICS")
    print("="*80)
    print(f"  Entities:        {baseline_stats['num_entities']:,}")
    print(f"  Relationships:   {baseline_stats['num_relationships']:,}")
    print(f"  Unique Sources:  {baseline_stats['unique_sources']:,}")
    print(f"  Unique Targets:  {baseline_stats['unique_targets']:,}")
    print("="*80 + "\n")
    
    strategies = [
        'crumbtrail',
        'kgtrimmer',
        'pathrag_hybrid',
        'pog_hybrid',
        'adaptive_multi_strategy',
    ]
    
    pruning_results = {}
    
    for strategy_name in strategies:
        pruned_dir = workspace_path / f"pruned_{strategy_name}"
        
        if not pruned_dir.exists():
            logger.warning(f"⚠️  Pruned directory not found: {pruned_dir}, skipping {strategy_name}")
            continue
        
        logger.info(f"Loading {strategy_name}...")
        try:
            pruned_entities, pruned_relationships, _ = load_graphrag_artifacts(pruned_dir)
            pruned_stats = get_graph_stats(pruned_entities, pruned_relationships)
            
            # Calculate reductions
            entity_reduction = ((baseline_stats['num_entities'] - pruned_stats['num_entities']) / baseline_stats['num_entities'] * 100) if baseline_stats['num_entities'] > 0 else 0
            relationship_reduction = ((baseline_stats['num_relationships'] - pruned_stats['num_relationships']) / baseline_stats['num_relationships'] * 100) if baseline_stats['num_relationships'] > 0 else 0
            
            pruning_results[strategy_name] = {
                'num_entities': pruned_stats['num_entities'],
                'num_relationships': pruned_stats['num_relationships'],
                'unique_sources': pruned_stats['unique_sources'],
                'unique_targets': pruned_stats['unique_targets'],
                'entity_reduction': entity_reduction,
                'edge_reduction': relationship_reduction
            }
        except Exception as e:
            logger.error(f"❌ Failed to load {strategy_name}: {e}")
            continue
    
    # Show summary table
    print("\n" + "="*80)
    print("PRUNING SUMMARY - ALL STRATEGIES")
    print("="*80)
    print(f"\n{'Strategy':<30} {'Entities':<15} {'Relationships':<15} {'Entity %':<12} {'Edge %':<12}")
    print("-" * 80)
    
    for strategy_name, stats in pruning_results.items():
        print(f"{strategy_name:<30} "
              f"{stats['num_entities']:<15,} "
              f"{stats['num_relationships']:<15,} "
              f"{stats['entity_reduction']:<12.1f}% "
              f"{stats['edge_reduction']:<12.1f}%")
    
    print("\n" + "="*80)
    print("BASELINE:")
    print(f"  Entities:      {baseline_stats['num_entities']:,}")
    print(f"  Relationships: {baseline_stats['num_relationships']:,}")
    print("="*80)
    
    # Detailed summary
    print("\n" + "="*80)
    print("DETAILED SUMMARY")
    print("="*80)
    for strategy_name, stats in pruning_results.items():
        print(f"\n{strategy_name.upper()}:")
        print(f"  Entities:        {stats['num_entities']:,} ({stats['entity_reduction']:.1f}% reduction)")
        print(f"  Relationships:   {stats['num_relationships']:,} ({stats['edge_reduction']:.1f}% reduction)")
        print(f"  Unique Sources:  {stats['unique_sources']:,}")
        print(f"  Unique Targets:  {stats['unique_targets']:,}")
    
    print("\n" + "="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

