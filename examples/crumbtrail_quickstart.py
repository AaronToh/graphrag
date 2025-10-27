#!/usr/bin/env python3
"""
CrumbTrail Quick Start Example

This script demonstrates how to use the CrumbTrail implementation to prune
a GraphRAG knowledge graph.

CrumbTrail is a bottom-up layering algorithm that preserves connectivity
from a root to protected nodes while pruning unessential nodes.

Usage:
    python examples/crumbtrail_quickstart.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pruning.prune_graph import GraphPruner
from pruning.scoring_utils import load_graphrag_artifacts
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_crumbtrail():
    """Example 1: Basic CrumbTrail with default parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic CrumbTrail Pruning")
    print("="*80)

    # Define paths
    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_crumbtrail")

    # Check baseline exists
    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        logger.error("Please run Stage 1 (ingest/build_index.py) first!")
        return

    # Initialize pruner
    logger.info("Initializing GraphPruner...")
    pruner = GraphPruner(baseline_dir, output_dir)

    # Run CrumbTrail with default parameters
    logger.info("Running CrumbTrail with default parameters...")
    artifacts = pruner.apply_crumbtrail_pipeline(
        root_entity=None,  # Will create virtual root
        protected_fraction=0.2,  # Protect top 20% of nodes
        protected_selection='degree_centrality',
        max_iterations=1000
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
    """Example 2: CrumbTrail with aggressive pruning (fewer protected nodes)."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Aggressive CrumbTrail Pruning")
    print("="*80)

    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_crumbtrail_aggressive")

    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return

    pruner = GraphPruner(baseline_dir, output_dir)

    # More aggressive pruning parameters
    logger.info("Running CrumbTrail with aggressive parameters...")
    artifacts = pruner.apply_crumbtrail_pipeline(
        root_entity=None,
        protected_fraction=0.1,  # Only protect top 10%
        protected_selection='degree_centrality',
        max_iterations=1000
    )

    # Print results
    baseline_stats = artifacts['metadata']['baseline_stats']
    pruned_stats = artifacts['metadata']['pruned_stats']
    entity_reduction = 100 * (1 - pruned_stats['num_entities'] / baseline_stats['num_entities'])

    print(f"\n📊 Results:")
    print(f"  Protected: {artifacts['metadata']['parameters']['num_protected']} nodes "
          f"({100*artifacts['metadata']['parameters']['protected_fraction']:.0f}%)")
    print(f"  Entity reduction: {entity_reduction:.1f}%")
    print(f"  Output: {output_dir}")


def example_3_conservative_pruning():
    """Example 3: CrumbTrail with conservative pruning (more protected nodes)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Conservative CrumbTrail Pruning")
    print("="*80)

    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_crumbtrail_conservative")

    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return

    pruner = GraphPruner(baseline_dir, output_dir)

    # More conservative pruning parameters
    logger.info("Running CrumbTrail with conservative parameters...")
    artifacts = pruner.apply_crumbtrail_pipeline(
        root_entity=None,
        protected_fraction=0.3,  # Protect top 30%
        protected_selection='degree_centrality',
        max_iterations=1000
    )

    # Print results
    baseline_stats = artifacts['metadata']['baseline_stats']
    pruned_stats = artifacts['metadata']['pruned_stats']
    entity_reduction = 100 * (1 - pruned_stats['num_entities'] / baseline_stats['num_entities'])

    print(f"\n📊 Results:")
    print(f"  Protected: {artifacts['metadata']['parameters']['num_protected']} nodes "
          f"({100*artifacts['metadata']['parameters']['protected_fraction']:.0f}%)")
    print(f"  Entity reduction: {entity_reduction:.1f}%")
    print(f"  Output: {output_dir}")


def example_4_compare_configurations():
    """Example 4: Run multiple CrumbTrail configurations and compare."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch CrumbTrail Configurations")
    print("="*80)

    baseline_dir = Path("workspace/output")

    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return

    # Define configurations to test
    configs = [
        {
            'name': 'aggressive',
            'protected_fraction': 0.1,
            'output': 'workspace/output/batch_crumbtrail_aggressive'
        },
        {
            'name': 'default',
            'protected_fraction': 0.2,
            'output': 'workspace/output/batch_crumbtrail_default'
        },
        {
            'name': 'conservative',
            'protected_fraction': 0.3,
            'output': 'workspace/output/batch_crumbtrail_conservative'
        }
    ]

    results = []

    for config in configs:
        logger.info(f"\nRunning configuration: {config['name']}")

        pruner = GraphPruner(baseline_dir, Path(config['output']))

        artifacts = pruner.apply_crumbtrail_pipeline(
            root_entity=None,
            protected_fraction=config['protected_fraction'],
            protected_selection='degree_centrality',
            max_iterations=1000
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
            'protected_pct': 100 * config['protected_fraction'],
            'entity_reduction': entity_reduction,
            'edge_reduction': edge_reduction
        })

    # Print summary
    print("\n📊 Batch Results Summary:")
    print(f"{'Configuration':<15} {'Protected%':<12} {'Entity Red%':<12} {'Edge Red%':<12}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<15} {r['protected_pct']:<12.0f} "
              f"{r['entity_reduction']:<12.1f} {r['edge_reduction']:<12.1f}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("CrumbTrail Quick Start Examples")
    print("="*80)

    examples = [
        ("Basic CrumbTrail", example_1_basic_crumbtrail),
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
    print("  1. Review pruned artifacts in workspace/output/pruned_crumbtrail/")
    print("  2. Compare with PathRAG or other pruning methods")
    print("  3. Run evaluation to measure quality metrics")


if __name__ == "__main__":
    main()
