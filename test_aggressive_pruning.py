#!/usr/bin/env python3
"""
Test script for conservative aggressive pruning strategy targeting 20-30% reduction.

This script tests a conservative approach using the regular aggressive method
with carefully tuned parameters to achieve meaningful but not excessive reduction.
"""

import logging
from pathlib import Path
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_conservative_aggressive_pruning():
    """Test conservative aggressive pruning strategy targeting 20-30% reduction."""
    
    from pruning.prune_graph import GraphPruner
    
    baseline_dir = Path("workspace/output")
    
    # Test configurations for conservative aggressive pruning targeting 20-30% reduction
    test_configs = [
        {
            'name': 'conservative_aggressive_20pct',
            'output_dir': Path("workspace/output/pruned_conservative_20pct"),
            'protected_fraction': 0.15,  # 15% - much more conservative
            'protected_selection': 'degree_centrality',
            'connectivity_threshold': 0.3
        },
        {
            'name': 'conservative_aggressive_25pct',
            'output_dir': Path("workspace/output/pruned_conservative_25pct"),
            'protected_fraction': 0.12,  # 12%
            'protected_selection': 'degree_centrality',
            'connectivity_threshold': 0.25
        },
        {
            'name': 'conservative_aggressive_30pct',
            'output_dir': Path("workspace/output/pruned_conservative_30pct"),
            'protected_fraction': 0.10,  # 10%
            'protected_selection': 'degree_centrality',
            'connectivity_threshold': 0.2
        }
    ]
    
    results = []
    
    for config in test_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Testing {config['name']}")
        logger.info(f"🎯 Target: 20-30% entity reduction (conservative approach)")
        logger.info(f"{'='*60}")
        
        try:
            # Create pruner
            pruner = GraphPruner(baseline_dir, config['output_dir'])
            
            # Apply conservative aggressive pruning using the regular aggressive method
            artifacts = pruner.apply_aggressive_crumbtrail_pipeline(
                root_entity=None,
                protected_fraction=config['protected_fraction'],
                protected_selection=config['protected_selection'],
                connectivity_threshold=config['connectivity_threshold'],
                max_iterations=1000
            )
            
            # Extract results
            metadata = artifacts['metadata']
            baseline_stats = metadata['baseline_stats']
            pruned_stats = metadata['pruned_stats']
            reduction_pct = metadata['reduction_percentages']
            
            result = {
                'name': config['name'],
                'protected_fraction': config['protected_fraction'],
                'protected_selection': config['protected_selection'],
                'connectivity_threshold': config['connectivity_threshold'],
                'baseline_entities': baseline_stats['num_entities'],
                'pruned_entities': pruned_stats['num_entities'],
                'entity_reduction_pct': reduction_pct['entities'],
                'baseline_relationships': baseline_stats['num_relationships'],
                'pruned_relationships': pruned_stats['num_relationships'],
                'relationship_reduction_pct': reduction_pct['relationships'],
                'output_dir': str(config['output_dir']),
                'target_achieved': 20.0 <= reduction_pct['entities'] <= 30.0
            }
            
            results.append(result)
            
            if result['target_achieved']:
                logger.info(f"🎯 {config['name']} SUCCESS!")
                logger.info(f"   Entity reduction: {reduction_pct['entities']:.1f}% (20-30% target range)")
            else:
                logger.info(f"⚠️  {config['name']} outside target range")
                logger.info(f"   Entity reduction: {reduction_pct['entities']:.1f}% (target: 20-30%)")
            logger.info(f"   Relationship reduction: {reduction_pct['relationships']:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ {config['name']} failed: {str(e)}")
            result = {
                'name': config['name'],
                'error': str(e),
                'target_achieved': False
            }
            results.append(result)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("🎯 CONSERVATIVE AGGRESSIVE PRUNING TEST SUMMARY")
    logger.info(f"🎯 TARGET: 20-30% entity reduction (conservative approach)")
    logger.info(f"{'='*80}")
    
    successful_results = []
    for result in results:
        if 'error' in result:
            logger.info(f"❌ {result['name']}: FAILED - {result['error']}")
        else:
            successful_results.append(result)
            status = "🎯 TARGET ACHIEVED" if result['target_achieved'] else "⚠️  OUTSIDE TARGET"
            logger.info(f"{status} - {result['name']}:")
            logger.info(f"   Protected: {result['protected_fraction']*100:.1f}% ({result['protected_selection']})")
            logger.info(f"   Connectivity threshold: {result['connectivity_threshold']}")
            logger.info(f"   Entities: {result['baseline_entities']} → {result['pruned_entities']} ({result['entity_reduction_pct']:.1f}% reduction)")
            logger.info(f"   Relationships: {result['baseline_relationships']} → {result['pruned_relationships']} ({result['relationship_reduction_pct']:.1f}% reduction)")
            logger.info(f"   Output: {result['output_dir']}")
            logger.info("")
    
    # Find the best results
    target_achieved = [r for r in successful_results if r['target_achieved']]
    
    if target_achieved:
        # Find the one closest to 25% (middle of 20-30% range)
        best_result = min(target_achieved, key=lambda x: abs(x['entity_reduction_pct'] - 25.0))
        logger.info(f"🏆 BEST RESULT (closest to 25%): {best_result['name']}")
        logger.info(f"   Entity reduction: {best_result['entity_reduction_pct']:.1f}%")
        logger.info(f"   Use this for evaluation: {best_result['output_dir']}")
        
        # Show all that achieved target
        logger.info(f"\n🎯 ALL SUCCESSFUL CONFIGURATIONS:")
        for result in sorted(target_achieved, key=lambda x: x['entity_reduction_pct'], reverse=True):
            logger.info(f"   {result['name']}: {result['entity_reduction_pct']:.1f}% reduction")
        
        return best_result['output_dir']
    else:
        logger.warning("⚠️  No configuration achieved 20-30% target range!")
        if successful_results:
            # Find the one closest to 25%
            best_attempt = min(successful_results, key=lambda x: abs(x['entity_reduction_pct'] - 25.0))
            logger.info(f"   Best attempt: {best_attempt['name']} ({best_attempt['entity_reduction_pct']:.1f}%)")
            return best_attempt['output_dir']
        else:
            logger.error("❌ All conservative aggressive pruning attempts failed!")
            return None

if __name__ == "__main__":
    best_output_dir = test_conservative_aggressive_pruning()
    if best_output_dir:
        print(f"\n🎯 RECOMMENDED FOR EVALUATION: {best_output_dir}")
    else:
        print("\n❌ No successful pruning configuration found!")