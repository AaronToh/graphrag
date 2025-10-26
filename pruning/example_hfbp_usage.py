#!/usr/bin/env python3
"""
HFBP Usage Examples

This script demonstrates various ways to use the Hierarchical Flow-Based Pruning
algorithm with GraphRAG data.
"""

import sys
from pathlib import Path
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from hfbp_pruning import HFBPPruning, HFBPConfig
from scoring_utils import load_graphrag_artifacts


def example_1_basic_usage():
    """Example 1: Basic HFBP usage with default configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage with Default Configuration")
    print("="*80)

    # Load GraphRAG artifacts
    workspace_dir = Path(__file__).parent.parent / "workspace" / "output"
    entities_df, relationships_df, _ = load_graphrag_artifacts(workspace_dir)

    # Sample for demo
    if len(entities_df) > 100:
        entities_df = entities_df.head(100)
        entity_titles = set(entities_df['title'].tolist())
        relationships_df = relationships_df[
            relationships_df['source'].isin(entity_titles) &
            relationships_df['target'].isin(entity_titles)
        ]

    # Default configuration
    config = HFBPConfig()
    hfbp = HFBPPruning(entities_df, relationships_df, config=config)

    # Execute with query
    query = "What are the relationships between diseases and treatments?"
    retrieved_nodes = entities_df['title'].head(5).tolist()

    result = hfbp.execute(query, retrieved_nodes)

    print(f"\n✅ Found {result['num_paths_found']} paths in {result['execution_time_seconds']:.2f}s")
    print(f"\nTop path:")
    if result['top_k_paths']:
        path = result['top_k_paths'][0]
        print(f"  Score: {path.score:.4f}")
        print(f"  Path: {' -> '.join(path.nodes)}")


def example_2_custom_config():
    """Example 2: Custom configuration for specific use case."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Configuration")
    print("="*80)

    workspace_dir = Path(__file__).parent.parent / "workspace" / "output"
    entities_df, relationships_df, _ = load_graphrag_artifacts(workspace_dir)

    # Sample
    entities_df = entities_df.head(100)
    entity_titles = set(entities_df['title'].tolist())
    relationships_df = relationships_df[
        relationships_df['source'].isin(entity_titles) &
        relationships_df['target'].isin(entity_titles)
    ]

    # Custom config: more paths, wider beam, stricter pruning
    config = HFBPConfig(
        K=20,                      # Return more paths
        beam_k=10,                 # Wider beam search
        theta=0.1,                 # Stricter pruning
        alpha=0.9,                 # Higher propagation
        max_propagation_steps=2    # Fewer iterations for speed
    )

    hfbp = HFBPPruning(entities_df, relationships_df, config=config)

    query = "Find comprehensive biomedical entity connections"
    retrieved_nodes = entities_df['title'].head(8).tolist()

    result = hfbp.execute(query, retrieved_nodes)

    print(f"\n✅ Configuration:")
    print(f"   K={config.K}, beam_k={config.beam_k}, theta={config.theta}")
    print(f"\n✅ Results:")
    print(f"   Found {result['num_paths_found']} paths")
    print(f"   Execution time: {result['execution_time_seconds']:.2f}s")


def example_3_step_by_step():
    """Example 3: Step-by-step execution with inspection."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Step-by-Step Execution")
    print("="*80)

    workspace_dir = Path(__file__).parent.parent / "workspace" / "output"
    entities_df, relationships_df, _ = load_graphrag_artifacts(workspace_dir)

    # Sample
    entities_df = entities_df.head(100)
    entity_titles = set(entities_df['title'].tolist())
    relationships_df = relationships_df[
        relationships_df['source'].isin(entity_titles) &
        relationships_df['target'].isin(entity_titles)
    ]

    config = HFBPConfig(K=10)
    hfbp = HFBPPruning(entities_df, relationships_df, config=config)

    # Step 1: Precompute supernodes
    print("\nStep 1: Precomputing supernodes...")
    supernodes = hfbp.precompute_supernodes()
    print(f"  ✅ Created {len(supernodes)} supernodes")

    # Inspect a supernode
    first_sn = list(supernodes.values())[0]
    print(f"  Sample supernode:")
    print(f"    - Context node: {first_sn.context_node}")
    print(f"    - Members: {len(first_sn.member_nodes)}")
    print(f"    - Summary: {first_sn.summary[:100]}...")

    # Step 2: Query enhancement
    print("\nStep 2: Query enhancement...")
    query = "Disease and treatment relationships"
    vq = entities_df['title'].head(3).tolist()
    enhanced = hfbp.enhance_retrieved_nodes(query, vq)
    print(f"  ✅ Enhanced {len(vq)} nodes to {len(enhanced)} nodes")

    # Step 3: Build supergraph
    print("\nStep 3: Building supergraph...")
    supergraph = hfbp.build_supergraph()
    print(f"  ✅ Supergraph: {supergraph.number_of_nodes()} nodes, {supergraph.number_of_edges()} edges")

    # Step 4-5: Find paths
    print("\nStep 4-5: Finding and ranking paths...")
    paths = hfbp.find_paths(query, vq)
    print(f"  ✅ Found {len(paths)} paths")

    if paths:
        print(f"\n  Top path details:")
        top = paths[0]
        print(f"    Score: {top.score:.4f}")
        print(f"    Nodes: {len(top.nodes)}")
        print(f"    Edges: {len(top.edges)}")
        print(f"    Context nodes: {len(top.context_nodes)}")


def example_4_batch_processing():
    """Example 4: Batch processing multiple queries."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Processing")
    print("="*80)

    workspace_dir = Path(__file__).parent.parent / "workspace" / "output"
    entities_df, relationships_df, _ = load_graphrag_artifacts(workspace_dir)

    # Sample
    entities_df = entities_df.head(100)
    entity_titles = set(entities_df['title'].tolist())
    relationships_df = relationships_df[
        relationships_df['source'].isin(entity_titles) &
        relationships_df['target'].isin(entity_titles)
    ]

    config = HFBPConfig(K=5)
    hfbp = HFBPPruning(entities_df, relationships_df, config=config)

    # Precompute once
    hfbp.precompute_supernodes()

    # Batch queries
    queries = [
        ("Disease mechanisms", entities_df['title'].head(4).tolist()),
        ("Treatment outcomes", entities_df['title'].tail(4).tolist()),
        ("Biomedical processes", entities_df['title'].sample(4).tolist()),
    ]

    print(f"\nProcessing {len(queries)} queries...")
    results = []

    for i, (query, nodes) in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        result = hfbp.execute(query, nodes)
        results.append(result)
        print(f"  ✅ Found {result['num_paths_found']} paths in {result['execution_time_seconds']:.2f}s")

    print(f"\n✅ Batch processing complete!")
    print(f"   Total queries: {len(results)}")
    print(f"   Total paths found: {sum(r['num_paths_found'] for r in results)}")
    print(f"   Average time per query: {sum(r['execution_time_seconds'] for r in results) / len(results):.2f}s")


def example_5_save_and_analyze():
    """Example 5: Save results and perform analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Save Results and Analyze")
    print("="*80)

    workspace_dir = Path(__file__).parent.parent / "workspace" / "output"
    entities_df, relationships_df, _ = load_graphrag_artifacts(workspace_dir)

    # Sample
    entities_df = entities_df.head(100)
    entity_titles = set(entities_df['title'].tolist())
    relationships_df = relationships_df[
        relationships_df['source'].isin(entity_titles) &
        relationships_df['target'].isin(entity_titles)
    ]

    config = HFBPConfig(K=10)
    hfbp = HFBPPruning(entities_df, relationships_df, config=config)

    query = "Comprehensive biomedical analysis"
    retrieved_nodes = entities_df['title'].head(6).tolist()

    result = hfbp.execute(query, retrieved_nodes)

    # Save results
    output_path = Path(__file__).parent / "example_hfbp_output.json"
    hfbp.save_results(result, output_path)
    print(f"\n✅ Results saved to: {output_path}")

    # Analyze paths
    print(f"\n📊 Path Analysis:")
    paths = result['top_k_paths']
    if paths:
        scores = [p.score for p in paths]
        lengths = [len(p.nodes) for p in paths]
        context_counts = [len(p.context_nodes) for p in paths]

        print(f"   Paths found: {len(paths)}")
        print(f"   Score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"   Avg score: {sum(scores)/len(scores):.4f}")
        print(f"   Path lengths: {min(lengths)} - {max(lengths)} nodes")
        print(f"   Avg length: {sum(lengths)/len(lengths):.1f} nodes")
        print(f"   Total context nodes: {sum(context_counts)}")

        # Show textual paths
        print(f"\n📝 Sample Path Text:")
        print(paths[0].text_representation[:300] + "...")


def main():
    """Run all examples."""
    print("="*80)
    print("🧪 HFBP USAGE EXAMPLES")
    print("="*80)

    try:
        example_1_basic_usage()
        example_2_custom_config()
        example_3_step_by_step()
        example_4_batch_processing()
        example_5_save_and_analyze()

        print("\n" + "="*80)
        print("✅ All examples completed successfully!")
        print("="*80)

    except FileNotFoundError:
        print("\n⚠️  GraphRAG workspace data not found.")
        print("   Please run the ingestion pipeline first:")
        print("   python ingest/build_index.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
