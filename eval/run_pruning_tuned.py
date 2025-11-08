#!/usr/bin/env python3
"""
Run all pruning strategies with tuned parameters for 50-60% entity reduction.

This script only runs pruning (no evaluation) with optimized parameters.
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

def log(msg: str = "", *, newline: bool = True):
    end = "\n" if newline else ""
    print(msg, end=end, flush=True)


def get_graph_stats(entities_df: pd.DataFrame, relationships_df: pd.DataFrame) -> dict:
    """Get statistics about the graph."""
    return {
        'num_entities': len(entities_df),
        'num_relationships': len(relationships_df),
        'unique_sources': relationships_df['source'].nunique() if len(relationships_df) > 0 else 0,
        'unique_targets': relationships_df['target'].nunique() if len(relationships_df) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run pruning with tuned parameters for 50-60% reduction")
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
    logger.info("Loading baseline statistics...")
    baseline_entities, baseline_relationships, _ = load_graphrag_artifacts(baseline_dir)
    baseline_stats = get_graph_stats(baseline_entities, baseline_relationships)
    
    log("\n" + "="*80)
    log("BASELINE GRAPH STATISTICS")
    log("="*80)
    log(f"  Entities:        {baseline_stats['num_entities']:,}")
    log(f"  Relationships:   {baseline_stats['num_relationships']:,}")
    log(f"  Unique Sources:  {baseline_stats['unique_sources']:,}")
    log(f"  Unique Targets:  {baseline_stats['unique_targets']:,}")
    log("="*80 + "\n")
    
    # Strategies with tuned parameters for 50-60% reduction
    strategies = [
        {
            'name': 'crumbtrail',
            'method': 'apply_crumbtrail_pipeline',
            'params': {
                'protected_fraction': 0.20,  # Increased from 0.2 to keep more nodes (target: ~50% reduction)
                'protected_selection': 'degree_centrality',
                'max_iterations': 1000
            }
        },
        {
            'name': 'kgtrimmer',
            'method': 'apply_kgtrimmer_pipeline',
            'params': {
                'collective_weight': 0.5,
                'holistic_weight': 0.5,
                'min_importance_percentile': 0.45,  # Target ~55% reduction with single-pass scoring
                'preserve_connectivity': True,
                'max_iterations': 1
            }
        },
        {
            'name': 'pathrag_hybrid',
            'method': 'apply_pathrag_hybrid_pipeline',
            'params': {
                'top_n_nodes': 500,
                'top_k_paths': 3000,  # Balanced path budget for ~55% retention
                'max_path_length': 6,
                'node_retention_pct': 0.45,  # Blend node retention with path signals (~55% target)
                'node_scoring_method': 'degree_centrality',
                'alpha': 0.8,
                'theta': 0.02,  # Looser flow cutoff to keep key transitions
                'seed_method': 'degree_centrality',
                'path_scoring_method': 'avg_edge_flow'
            }
        },
        {
            'name': 'pog_hybrid',
            'method': 'apply_pog_hybrid_pipeline',
            'params': {
                'num_seeds': 400,  # Expanded seed coverage for hybrid target
                'top_k_paths': 4000,  # Balanced path quota for hybrid target
                'max_path_length': 7,
                'node_retention_pct': 0.45,  # Blend node retention with path signals (~55% target)
                'node_scoring_method': 'degree_centrality',
                'seed_method': 'degree_centrality',
                'sbert_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'semantic_threshold': 0.35  # Lower threshold to avoid over-pruning
            }
        },
        {
            'name': 'adaptive_multi_strategy',
            'method': 'apply_adaptive_multi_strategy_pipeline',
            'params': {
                'target_reduction': 0.60,  # Stronger reduction target for adaptive strategy
                'min_connectivity_pct': 0.85,
                'protected_fraction': 0.20,  # Smaller protected set to enable deeper pruning
                'hub_degree_percentile': 0.75
            }
        },
    ]
    
    pruning_results = {}
    total_strategies = len(strategies)
    
    for idx, strategy in enumerate(strategies, 1):
        log("\n" + "="*80)
        log(f"PRUNING STRATEGY {idx}/{total_strategies}: {strategy['name'].upper()}")
        log("="*80)
        
        output_dir = workspace_path / f"pruned_{strategy['name']}_tuned"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log(f"📁 Output directory: {output_dir}")
        log(f"🔧 Method: {strategy['method']}")
        log(f"⚙️  Parameters: {strategy['params']}")
        
        # Create a new pruner instance for each strategy
        log(f"\n🔨 Initializing GraphPruner...")
        strategy_pruner = GraphPruner(baseline_dir, output_dir)
        
        # Get the method and call it
        method = getattr(strategy_pruner, strategy['method'])
        try:
            log(f"\n📊 BEFORE PRUNING:")
            log(f"  Entities:        {baseline_stats['num_entities']:,}")
            log(f"  Relationships:   {baseline_stats['num_relationships']:,}")
            
            log(f"\n🔄 Running {strategy['name']} pruning algorithm...")
            log(f"   Target: 50-60% entity reduction")
            pruned_artifacts = method(**strategy['params'])
            
            # Get after stats
            pruned_entities = pruned_artifacts['entities']
            pruned_relationships = pruned_artifacts['relationships']
            after_stats = get_graph_stats(pruned_entities, pruned_relationships)
            
            # Calculate reductions
            entity_reduction = ((baseline_stats['num_entities'] - after_stats['num_entities']) / baseline_stats['num_entities'] * 100) if baseline_stats['num_entities'] > 0 else 0
            relationship_reduction = ((baseline_stats['num_relationships'] - after_stats['num_relationships']) / baseline_stats['num_relationships'] * 100) if baseline_stats['num_relationships'] > 0 else 0
            
            log(f"\n📊 AFTER PRUNING:")
            log(f"  Entities:        {after_stats['num_entities']:,} ({entity_reduction:.1f}% reduction)")
            log(f"  Relationships:   {after_stats['num_relationships']:,} ({relationship_reduction:.1f}% reduction)")
            log(f"  Unique Sources:  {after_stats['unique_sources']:,}")
            log(f"  Unique Targets:  {after_stats['unique_targets']:,}")
            
            # Check if target achieved
            if 50 <= entity_reduction <= 60:
                log(f"\n✅ {strategy['name']} pruning completed successfully! (Target achieved: {entity_reduction:.1f}%)")
            elif entity_reduction < 50:
                log(f"\n⚠️  {strategy['name']} pruning completed but reduction is lower than target ({entity_reduction:.1f}% < 50%)")
            else:
                log(f"\n⚠️  {strategy['name']} pruning completed but reduction is higher than target ({entity_reduction:.1f}% > 60%)")
            
            log(f"   Results saved to: {output_dir}")
            log("="*80)
            
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
            log(f"\n❌ {strategy['name']} pruning failed: {e}")
            import traceback
            traceback.print_exc()
            log("="*80)
    
    # Show final summary
    log("\n" + "="*80)
    log("PRUNING SUMMARY - ALL STRATEGIES (TUNED FOR 50-60% REDUCTION)")
    log("="*80)
    log(f"\n{'Strategy':<30} {'Entities':<15} {'Relationships':<15} {'Entity %':<12} {'Edge %':<12} {'Target':<10}")
    log("-" * 95)
    
    for strategy_name, stats in pruning_results.items():
        entity_reduction = stats.get('entity_reduction', 0)
        edge_reduction = stats.get('edge_reduction', 0)
        target_status = "✅" if 50 <= entity_reduction <= 60 else "⚠️"
        log(f"{strategy_name:<30} "
              f"{stats['num_entities']:<15,} "
              f"{stats['num_relationships']:<15,} "
              f"{entity_reduction:<12.1f}% "
              f"{edge_reduction:<12.1f}% "
              f"{target_status:<10}")
    
    log("\n" + "="*80)
    log("BASELINE:")
    log(f"  Entities:      {baseline_stats['num_entities']:,}")
    log(f"  Relationships: {baseline_stats['num_relationships']:,}")
    log("="*80)
    
    # Detailed summary
    log("\n" + "="*80)
    log("DETAILED SUMMARY")
    log("="*80)
    for strategy_name, stats in pruning_results.items():
        entity_reduction = stats.get('entity_reduction', 0)
        target_status = "✅ Target achieved" if 50 <= entity_reduction <= 60 else f"⚠️  Target: 50-60%, Got: {entity_reduction:.1f}%"
        log(f"\n{strategy_name.upper()}:")
        log(f"  Entities:        {stats['num_entities']:,} ({stats['entity_reduction']:.1f}% reduction) - {target_status}")
        log(f"  Relationships:   {stats['num_relationships']:,} ({stats['edge_reduction']:.1f}% reduction)")
        log(f"  Unique Sources:  {stats['unique_sources']:,}")
        log(f"  Unique Targets:  {stats['unique_targets']:,}")
    
    log("\n" + "="*80)
    log("✅ Pruning completed! Results saved to workspace/pruned_*_tuned directories")
    log("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

