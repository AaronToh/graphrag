#!/usr/bin/env python3
"""
PathRAG Quick Start Example

This script demonstrates how to use the PathRAG implementation to prune
a GraphRAG knowledge graph.

Usage:
    python examples/pathrag_quickstart.py
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


def example_1_basic_pathrag():
    """Example 1: Basic PathRAG with default parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic PathRAG Pruning")
    print("="*80)

    # Define paths
    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_pathrag_example")

    # Check baseline exists
    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        logger.error("Please run Stage 1 (ingest/build_index.py) first!")
        return

    # Initialize pruner
    logger.info("Initializing GraphPruner...")
    pruner = GraphPruner(baseline_dir, output_dir)

    # Run PathRAG with default paper parameters
    logger.info("Running PathRAG with default parameters...")
    artifacts = pruner.apply_pathrag_pipeline(
        alpha=0.8,              # Flow decay factor (paper default)
        theta=0.05,             # Early stopping threshold (paper default)
        top_n_nodes=40,         # Top nodes to connect (N in paper)
        top_k_paths=15,         # Paths to keep (K in paper)
        max_path_length=5,      # Maximum path length
        seed_method='degree_centrality',
        path_scoring_method='min_edge_flow'
    )

    # Print results
    baseline_stats = artifacts['metadata']['baseline_stats']
    pruned_stats = artifacts['metadata']['pruned_stats']

    print("\n📊 Results:")
    print(f"  Baseline:")
    print(f"    - Entities: {baseline_stats['num_entities']}")
    print(f"    - Relationships: {baseline_stats['num_relationships']}")
    print(f"  Pruned:")
    print(f"    - Entities: {pruned_stats['num_entities']}")
    print(f"    - Relationships: {pruned_stats['num_relationships']}")

    entity_reduction = 100 * (1 - pruned_stats['num_entities'] / baseline_stats['num_entities'])
    edge_reduction = 100 * (1 - pruned_stats['num_relationships'] / baseline_stats['num_relationships'])

    print(f"  Reduction:")
    print(f"    - Entities: {entity_reduction:.1f}%")
    print(f"    - Relationships: {edge_reduction:.1f}%")

    print(f"\n✅ Pruned artifacts saved to: {output_dir}")


def example_2_custom_parameters():
    """Example 2: PathRAG with custom parameters for aggressive pruning."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Aggressive PathRAG Pruning")
    print("="*80)

    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_pathrag_aggressive_example")

    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return

    pruner = GraphPruner(baseline_dir, output_dir)

    # More aggressive pruning parameters
    logger.info("Running PathRAG with aggressive parameters...")
    artifacts = pruner.apply_pathrag_pipeline(
        alpha=0.7,              # Lower decay = more aggressive
        theta=0.1,              # Higher threshold = stop earlier
        top_n_nodes=30,         # Fewer nodes
        top_k_paths=10,         # Fewer paths
        max_path_length=4,      # Shorter paths
        seed_method='degree_centrality',
        path_scoring_method='min_edge_flow'
    )

    # Print results
    baseline_stats = artifacts['metadata']['baseline_stats']
    pruned_stats = artifacts['metadata']['pruned_stats']
    edge_reduction = 100 * (1 - pruned_stats['num_relationships'] / baseline_stats['num_relationships'])

    print(f"\n📊 Results:")
    print(f"  Edge reduction: {edge_reduction:.1f}%")
    print(f"  Output: {output_dir}")


def example_3_inspect_scores():
    """Example 3: Inspect node and edge flow scores."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Inspecting Flow Scores")
    print("="*80)

    baseline_dir = Path("workspace/output")

    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return

    # Load artifacts
    logger.info("Loading GraphRAG artifacts...")
    entities_df, relationships_df, communities_df = load_graphrag_artifacts(baseline_dir)

    # Import scorer
    from pruning.scoring_utils import GraphScorer

    # Initialize scorer
    scorer = GraphScorer(entities_df, relationships_df, communities_df)

    # Run flow propagation
    logger.info("Running flow propagation...")
    node_scores, edge_flows = scorer.score_nodes_flow_propagation(
        seed_nodes=None,        # Auto-select seeds
        alpha=0.8,
        theta=0.05,
        top_n_seeds=40,
        seed_method='degree_centrality'
    )

    # Show top nodes
    print("\n🔝 Top 10 Nodes by Flow Score:")
    top_nodes = node_scores.nlargest(10)
    for i, (node_id, score) in enumerate(top_nodes.items(), 1):
        entity = entities_df[entities_df['id'] == node_id].iloc[0]
        print(f"  {i}. {entity['title'][:40]:40s} | Score: {score:.4f}")

    # Show top edges
    edge_flow_series = scorer.score_edges_flow(edge_flows)
    top_edges = edge_flow_series.nlargest(10)

    print("\n🔝 Top 10 Edges by Flow Score:")
    for i, ((src, tgt), score) in enumerate(top_edges.items(), 1):
        src_title = entities_df[entities_df['id'] == src].iloc[0]['title']
        tgt_title = entities_df[entities_df['id'] == tgt].iloc[0]['title']
        print(f"  {i}. {src_title[:20]:20s} -> {tgt_title[:20]:20s} | Score: {score:.4f}")


def example_4_batch_configurations():
    """Example 4: Run multiple PathRAG configurations."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch PathRAG Configurations")
    print("="*80)

    baseline_dir = Path("workspace/output")

    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return

    # Define configurations to test
    configs = [
        {
            'name': 'default',
            'alpha': 0.8,
            'top_k_paths': 15,
            'output': 'workspace/output/batch_default'
        },
        {
            'name': 'alpha_0.6',
            'alpha': 0.6,
            'top_k_paths': 15,
            'output': 'workspace/output/batch_alpha_06'
        },
        {
            'name': 'k_20',
            'alpha': 0.8,
            'top_k_paths': 20,
            'output': 'workspace/output/batch_k20'
        }
    ]

    results = []

    for config in configs:
        logger.info(f"\nRunning configuration: {config['name']}")

        pruner = GraphPruner(baseline_dir, Path(config['output']))

        artifacts = pruner.apply_pathrag_pipeline(
            alpha=config['alpha'],
            theta=0.05,
            top_n_nodes=40,
            top_k_paths=config['top_k_paths'],
            max_path_length=5,
            seed_method='degree_centrality',
            path_scoring_method='min_edge_flow'
        )

        # Collect results
        baseline_stats = artifacts['metadata']['baseline_stats']
        pruned_stats = artifacts['metadata']['pruned_stats']
        edge_reduction = 100 * (1 - pruned_stats['num_relationships'] /
                               baseline_stats['num_relationships'])

        results.append({
            'name': config['name'],
            'alpha': config['alpha'],
            'k': config['top_k_paths'],
            'reduction_pct': edge_reduction
        })

    # Print summary
    print("\n📊 Batch Results Summary:")
    print(f"{'Configuration':<15} {'Alpha':<8} {'K':<6} {'Reduction':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<15} {r['alpha']:<8.2f} {r['k']:<6} {r['reduction_pct']:<12.1f}%")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("PathRAG Quick Start Examples")
    print("="*80)

    examples = [
        ("Basic PathRAG", example_1_basic_pathrag),
        ("Aggressive Pruning", example_2_custom_parameters),
        ("Inspect Scores", example_3_inspect_scores),
        ("Batch Configurations", example_4_batch_configurations),
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
    print("  1. Review pruned artifacts in workspace/output/pruned_pathrag_example/")
    print("  2. Run evaluation: python -m eval.generate_answers --workspace <pruned_dir>")
    print("  3. Compare with baseline using eval/run_eval.py")
    print("\nFor more details, see: PATHRAG_IMPLEMENTATION.md")


if __name__ == "__main__":
    main()
