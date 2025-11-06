#!/usr/bin/env python3
"""
KGTrimmer Quick Start Example

This script demonstrates how to use the KGTrimmer implementation to prune
a GraphRAG knowledge graph.

KGTrimmer evaluates node importance from both collective (community-based) and
holistic (global) perspectives.

Usage:
    python examples/kgtrimmer_quickstart.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pruning.prune_graph import GraphPruner
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_kgtrimmer():
    """Example 1: Basic KGTrimmer with default parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic KGTrimmer Pruning")
    print("="*80)

    # Define paths
    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_kgtrimmer")

    # Check baseline exists
    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        logger.error("Please run Stage 1 (ingest/build_index.py) first!")
        return

    # Initialize pruner
    logger.info("Initializing GraphPruner...")
    pruner = GraphPruner(baseline_dir, output_dir)

    # Run KGTrimmer with default parameters
    logger.info("Running KGTrimmer with default parameters...")
    artifacts = pruner.apply_kgtrimmer_pipeline(
        collective_weight=0.5,
        holistic_weight=0.5,
        min_importance_percentile=0.2,  # Keep top 20% of nodes
        preserve_connectivity=True,
        max_iterations=10
    )

    # Print results
    baseline_stats = artifacts['metadata']['baseline_stats']
    pruned_stats = artifacts['metadata']['pruned_stats']

    print("\n📊 Results:")
    print(f"  Baseline:")
    print(f"    - Entities: {baseline_stats['num_entities']:,}")
    print(f"    - Relationships: {baseline_stats['num_relationships']:,}")
    print(f"    - Avg Degree: {baseline_stats.get('avg_degree', 0):.2f}")
    print(f"  Pruned:")
    print(f"    - Entities: {pruned_stats['num_entities']:,}")
    print(f"    - Relationships: {pruned_stats['num_relationships']:,}")
    print(f"    - Avg Degree: {pruned_stats.get('avg_degree', 0):.2f}")

    entity_reduction = 100 * (1 - pruned_stats['num_entities'] / baseline_stats['num_entities'])
    edge_reduction = 100 * (1 - pruned_stats['num_relationships'] / baseline_stats['num_relationships'])

    print(f"  Reduction:")
    print(f"    - Entities: {entity_reduction:.1f}%")
    print(f"    - Relationships: {edge_reduction:.1f}%")

    print(f"\n✅ Pruned artifacts saved to: {output_dir}")


def example_2_aggressive_pruning():
    """Example 2: KGTrimmer with aggressive pruning (keep fewer nodes)."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Aggressive KGTrimmer Pruning")
    print("="*80)

    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_kgtrimmer_aggressive")

    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return

    pruner = GraphPruner(baseline_dir, output_dir)

    # More aggressive pruning parameters
    logger.info("Running KGTrimmer with aggressive parameters...")
    artifacts = pruner.apply_kgtrimmer_pipeline(
        collective_weight=0.3,  # Less weight on community
        holistic_weight=0.7,    # More weight on global importance
        min_importance_percentile=0.1,  # Keep only top 10%
        preserve_connectivity=True,
        max_iterations=15
    )

    # Print results
    baseline_stats = artifacts['metadata']['baseline_stats']
    pruned_stats = artifacts['metadata']['pruned_stats']
    entity_reduction = 100 * (1 - pruned_stats['num_entities'] / baseline_stats['num_entities'])

    print(f"\n📊 Results:")
    print(f"  Entity reduction: {entity_reduction:.1f}%")
    print(f"  Output: {output_dir}")


def example_3_conservative_pruning():
    """Example 3: KGTrimmer with conservative pruning (keep more nodes)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Conservative KGTrimmer Pruning")
    print("="*80)

    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_kgtrimmer_conservative")

    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return

    pruner = GraphPruner(baseline_dir, output_dir)

    # More conservative pruning parameters
    logger.info("Running KGTrimmer with conservative parameters...")
    artifacts = pruner.apply_kgtrimmer_pipeline(
        collective_weight=0.6,  # More weight on community
        holistic_weight=0.4,    # Less weight on global
        min_importance_percentile=0.3,  # Keep top 30%
        preserve_connectivity=True,
        max_iterations=5
    )

    # Print results
    baseline_stats = artifacts['metadata']['baseline_stats']
    pruned_stats = artifacts['metadata']['pruned_stats']
    entity_reduction = 100 * (1 - pruned_stats['num_entities'] / baseline_stats['num_entities'])

    print(f"\n📊 Results:")
    print(f"  Entity reduction: {entity_reduction:.1f}%")
    print(f"  Output: {output_dir}")


def example_4_compare_configurations():
    """Example 4: Run multiple KGTrimmer configurations and compare."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch KGTrimmer Configurations")
    print("="*80)

    baseline_dir = Path("workspace/output")

    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return

    # Define configurations to test
    configs = [
        {
            'name': 'aggressive',
            'collective_weight': 0.3,
            'holistic_weight': 0.7,
            'min_importance_percentile': 0.1,
            'output': 'workspace/output/batch_kgtrimmer_aggressive'
        },
        {
            'name': 'default',
            'collective_weight': 0.5,
            'holistic_weight': 0.5,
            'min_importance_percentile': 0.2,
            'output': 'workspace/output/batch_kgtrimmer_default'
        },
        {
            'name': 'conservative',
            'collective_weight': 0.6,
            'holistic_weight': 0.4,
            'min_importance_percentile': 0.3,
            'output': 'workspace/output/batch_kgtrimmer_conservative'
        }
    ]

    results = []

    for config in configs:
        logger.info(f"\nRunning configuration: {config['name']}")

        pruner = GraphPruner(baseline_dir, Path(config['output']))

        artifacts = pruner.apply_kgtrimmer_pipeline(
            collective_weight=config['collective_weight'],
            holistic_weight=config['holistic_weight'],
            min_importance_percentile=config['min_importance_percentile'],
            preserve_connectivity=True,
            max_iterations=10
        )

        # Collect results
        baseline_stats = artifacts['metadata']['baseline_stats']
        pruned_stats = artifacts['metadata']['pruned_stats']
        entity_reduction = 100 * (1 - pruned_stats['num_entities'] /
                                  baseline_stats['num_entities'])
        edge_reduction = 100 * (1 - pruned_stats['num_relationships'] /
                               baseline_stats['num_relationships'])

        results.append({
            'name': config['name'],
            'collective_weight': config['collective_weight'],
            'holistic_weight': config['holistic_weight'],
            'min_importance_percentile': config['min_importance_percentile'],
            'entity_reduction': entity_reduction,
            'edge_reduction': edge_reduction
        })

    # Print summary
    print("\n📊 Batch Results Summary:")
    print(f"{'Configuration':<15} {'Collective':<12} {'Holistic':<12} {'Keep%':<10} {'Entity Red%':<12} {'Edge Red%':<12}")
    print("-" * 80)
    for r in results:
        keep_pct = r['min_importance_percentile'] * 100
        print(f"{r['name']:<15} {r['collective_weight']:<12.1f} {r['holistic_weight']:<12.1f} "
              f"{keep_pct:<10.0f} {r['entity_reduction']:<12.1f} {r['edge_reduction']:<12.1f}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("KGTrimmer Quick Start Examples")
    print("="*80)

    examples = [
        ("Basic KGTrimmer", example_1_basic_kgtrimmer),
        ("Aggressive Pruning", example_2_aggressive_pruning),
        ("Conservative Pruning", example_3_conservative_pruning),
        ("Batch Configurations", example_4_compare_configurations),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. Run all")

    try:
        choice = input("\nSelect example (0-4): ").strip()

        if choice == '0':
            for name, func in examples:
                func()
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            examples[int(choice) - 1][1]()
        else:
            print("Invalid choice!")
            return

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return

    print("\n" + "="*80)
    print("Examples Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review pruned artifacts in workspace/output/pruned_kgtrimmer/")
    print("  2. Compare with other pruning methods")
    print("  3. Run evaluation to measure quality metrics")


if __name__ == "__main__":
    main()

