#!/usr/bin/env python3
"""
POG (Path Over Graph) Quick Start Example

This script demonstrates how to use the POG implementation to prune
a GraphRAG knowledge graph.

POG uses LLM-based path evaluation and SBERT semantic filtering to identify
and keep important paths in the graph.

Usage:
    python examples/pog_quickstart.py
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


def example_1_basic_pog():
    """Example 1: Basic POG with default parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic POG Pruning")
    print("="*80)

    # Define paths
    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_pog")

    # Check baseline exists
    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        logger.error("Please run Stage 1 (ingest/build_index.py) first!")
        return

    # Initialize pruner
    logger.info("Initializing GraphPruner...")
    pruner = GraphPruner(baseline_dir, output_dir)

    # Run POG with default parameters
    logger.info("Running POG with default parameters...")
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

    # Print results
    baseline_stats = artifacts['metadata']['baseline_stats']
    pruned_stats = artifacts['metadata']['pruned_stats']

    print("\n📊 Results:")
    print(f"  Baseline:")
    print(f"    - Entities: {baseline_stats['num_entities']:,}")
    print(f"    - Relationships: {baseline_stats['num_relationships']:,}")
    print(f"  Pruned:")
    print(f"    - Entities: {pruned_stats['num_entities']:,}")
    print(f"    - Relationships: {pruned_stats['num_relationships']:,}")

    entity_reduction = 100 * (1 - pruned_stats['num_entities'] / baseline_stats['num_entities'])
    edge_reduction = 100 * (1 - pruned_stats['num_relationships'] / baseline_stats['num_relationships'])

    print(f"  Reduction:")
    print(f"    - Entities: {entity_reduction:.1f}%")
    print(f"    - Relationships: {edge_reduction:.1f}%")

    print(f"\n✅ Pruned artifacts saved to: {output_dir}")


def example_2_aggressive_pruning():
    """Example 2: POG with aggressive pruning (fewer paths)."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Aggressive POG Pruning")
    print("="*80)

    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_pog_aggressive")

    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return

    pruner = GraphPruner(baseline_dir, output_dir)

    # More aggressive pruning parameters
    logger.info("Running POG with aggressive parameters...")
    artifacts = pruner.apply_pog_pipeline(
        seed_method='degree_centrality',
        num_seeds=30,  # Fewer seeds
        max_path_length=4,  # Shorter paths
        top_k_paths=50,  # Fewer paths
        llm_provider='openai',
        llm_model='gpt-4o-mini',
        sbert_model='sentence-transformers/all-MiniLM-L6-v2',
        semantic_threshold=0.8  # Higher threshold
    )

    # Print results
    baseline_stats = artifacts['metadata']['baseline_stats']
    pruned_stats = artifacts['metadata']['pruned_stats']
    edge_reduction = 100 * (1 - pruned_stats['num_relationships'] / baseline_stats['num_relationships'])

    print(f"\n📊 Results:")
    print(f"  Edge reduction: {edge_reduction:.1f}%")
    print(f"  Output: {output_dir}")


def example_3_with_ollama():
    """Example 3: POG with Ollama LLM (local)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: POG with Ollama LLM")
    print("="*80)

    baseline_dir = Path("workspace/output")
    output_dir = Path("workspace/output/pruned_pog_ollama")

    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return

    pruner = GraphPruner(baseline_dir, output_dir)

    # Use Ollama for LLM scoring
    logger.info("Running POG with Ollama LLM...")
    artifacts = pruner.apply_pog_pipeline(
        seed_method='degree_centrality',
        num_seeds=50,
        max_path_length=5,
        top_k_paths=100,
        llm_provider='ollama',
        llm_model='llama3.2',
        llm_api_base_url='http://localhost:11434/v1',
        llm_api_key='ollama',
        sbert_model='sentence-transformers/all-MiniLM-L6-v2',
        semantic_threshold=0.7
    )

    # Print results
    baseline_stats = artifacts['metadata']['baseline_stats']
    pruned_stats = artifacts['metadata']['pruned_stats']
    edge_reduction = 100 * (1 - pruned_stats['num_relationships'] / baseline_stats['num_relationships'])

    print(f"\n📊 Results:")
    print(f"  Edge reduction: {edge_reduction:.1f}%")
    print(f"  Output: {output_dir}")


def example_4_compare_configurations():
    """Example 4: Run multiple POG configurations and compare."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch POG Configurations")
    print("="*80)

    baseline_dir = Path("workspace/output")

    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return

    # Define configurations to test
    configs = [
        {
            'name': 'default',
            'num_seeds': 50,
            'top_k_paths': 100,
            'output': 'workspace/output/batch_pog_default'
        },
        {
            'name': 'fewer_paths',
            'num_seeds': 50,
            'top_k_paths': 50,
            'output': 'workspace/output/batch_pog_fewer'
        },
        {
            'name': 'more_paths',
            'num_seeds': 50,
            'top_k_paths': 200,
            'output': 'workspace/output/batch_pog_more'
        }
    ]

    results = []

    for config in configs:
        logger.info(f"\nRunning configuration: {config['name']}")

        pruner = GraphPruner(baseline_dir, Path(config['output']))

        artifacts = pruner.apply_pog_pipeline(
            seed_method='degree_centrality',
            num_seeds=config['num_seeds'],
            max_path_length=5,
            top_k_paths=config['top_k_paths'],
            llm_provider='openai',
            llm_model='gpt-4o-mini',
            sbert_model='sentence-transformers/all-MiniLM-L6-v2',
            semantic_threshold=0.7
        )

        # Collect results
        baseline_stats = artifacts['metadata']['baseline_stats']
        pruned_stats = artifacts['metadata']['pruned_stats']
        edge_reduction = 100 * (1 - pruned_stats['num_relationships'] /
                               baseline_stats['num_relationships'])

        results.append({
            'name': config['name'],
            'num_seeds': config['num_seeds'],
            'top_k_paths': config['top_k_paths'],
            'reduction_pct': edge_reduction
        })

    # Print summary
    print("\n📊 Batch Results Summary:")
    print(f"{'Configuration':<15} {'Seeds':<8} {'Paths':<8} {'Reduction':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<15} {r['num_seeds']:<8} {r['top_k_paths']:<8} {r['reduction_pct']:<12.1f}%")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("POG (Path Over Graph) Quick Start Examples")
    print("="*80)

    examples = [
        ("Basic POG", example_1_basic_pog),
        ("Aggressive Pruning", example_2_aggressive_pruning),
        ("With Ollama LLM", example_3_with_ollama),
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
    print("  1. Review pruned artifacts in workspace/output/pruned_pog/")
    print("  2. Compare with other pruning methods")
    print("  3. Run evaluation to measure quality metrics")


if __name__ == "__main__":
    main()

