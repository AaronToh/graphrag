#!/usr/bin/env python3
"""
SuperHFBP Graph Pruning Pipeline

This script runs SuperHFBP on GraphRAG artifacts to generate a pruned graph.
The pruned graph can then be used for evaluation or downstream RAG tasks.

Usage:
    # Prune graph using query-based approach
    python run_superhfbp_pruning.py --workspace ../workspace --output ../workspace/pruned_superhfbp --queries queries.txt

    # Prune graph using automatic representative queries
    python run_superhfbp_pruning.py --workspace ../workspace --output ../workspace/pruned_superhfbp --mode auto --num-queries 10

    # Prune with custom config
    python run_superhfbp_pruning.py --workspace ../workspace --output ../workspace/pruned_superhfbp --config superhfbp_config.json

Author: GraphRAG Pruning Lab
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pruning.super_hfbp_pruning import SuperHFBP, SuperHFBPConfig
from pruning.hfbp_pruning import HFBPPruning, HFBPConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_graphrag_artifacts(workspace: Path) -> tuple:
    """
    Load GraphRAG artifacts from workspace.

    Args:
        workspace: Path to GraphRAG workspace

    Returns:
        Tuple of (entities_df, relationships_df, text_units_df)
    """
    output_dir = workspace / "output"

    if not output_dir.exists():
        raise FileNotFoundError(f"GraphRAG output directory not found: {output_dir}")

    logger.info(f"📂 Loading GraphRAG artifacts from {output_dir}...")

    entities = pd.read_parquet(output_dir / "entities.parquet")
    relationships = pd.read_parquet(output_dir / "relationships.parquet")

    # Text units are optional
    text_units_path = output_dir / "text_units.parquet"
    text_units = pd.read_parquet(text_units_path) if text_units_path.exists() else None

    logger.info(f"✅ Loaded: {len(entities)} entities, {len(relationships)} relationships")

    return entities, relationships, text_units


def load_queries_from_file(queries_file: Path) -> List[str]:
    """
    Load queries from a text file (one query per line).

    Args:
        queries_file: Path to queries file

    Returns:
        List of query strings
    """
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]

    logger.info(f"📄 Loaded {len(queries)} queries from {queries_file}")
    return queries


def generate_representative_queries(entities_df: pd.DataFrame, num_queries: int = 10) -> List[str]:
    """
    Generate representative queries from entity types and descriptions.

    This creates diverse queries covering different entity types and topics.

    Args:
        entities_df: Entities DataFrame
        num_queries: Number of queries to generate

    Returns:
        List of representative query strings
    """
    logger.info(f"🔍 Generating {num_queries} representative queries...")

    queries = []

    # Strategy 1: Sample diverse entity types
    if 'type' in entities_df.columns:
        entity_types = entities_df['type'].value_counts().head(5).index.tolist()
        for entity_type in entity_types[:min(3, len(entity_types))]:
            queries.append(f"What is {entity_type}?")

    # Strategy 2: Sample high-degree entities (by counting relationships)
    if 'title' in entities_df.columns:
        # Sample random entities
        sample_entities = entities_df.sample(min(num_queries, len(entities_df)))
        for _, entity in sample_entities.iterrows():
            title = entity.get('title', '')
            if title:
                queries.append(f"Tell me about {title}")

    # Strategy 3: Add broad queries
    queries.extend([
        "What are the main topics discussed?",
        "Summarize the key concepts",
        "What are the most important entities?"
    ])

    # Deduplicate and limit
    queries = list(set(queries))[:num_queries]

    logger.info(f"✅ Generated {len(queries)} representative queries")
    return queries


def get_initial_retrieved_nodes(entities_df: pd.DataFrame, query: str, top_n: int = 40) -> List[str]:
    """
    Simple entity retrieval based on text matching.

    Args:
        entities_df: Entities DataFrame
        query: Query string
        top_n: Number of top entities to retrieve

    Returns:
        List of entity IDs (titles)
    """
    query_lower = query.lower()
    query_terms = set(query_lower.split())

    scored_entities = []
    for _, entity in entities_df.iterrows():
        entity_title = str(entity.get('title', '')).lower()
        entity_desc = str(entity.get('description', '')).lower()

        # Simple relevance scoring
        title_matches = sum(1 for term in query_terms if term in entity_title)
        desc_matches = sum(1 for term in query_terms if term in entity_desc)
        score = title_matches * 2 + desc_matches

        if score > 0:
            scored_entities.append((str(entity['title']), score))

    scored_entities.sort(key=lambda x: x[1], reverse=True)
    return [entity_id for entity_id, _ in scored_entities[:top_n]]


def merge_pruned_graphs(all_paths: List[List], entities_df: pd.DataFrame,
                       relationships_df: pd.DataFrame,
                       text_units_df: Optional[pd.DataFrame]) -> tuple:
    """
    Merge pruned graphs from multiple queries into a single unified graph.

    Args:
        all_paths: List of path lists from multiple queries
        entities_df: Original entities DataFrame
        relationships_df: Original relationships DataFrame
        text_units_df: Original text units DataFrame

    Returns:
        Tuple of (merged_entities, merged_relationships, merged_text_units)
    """
    logger.info(f"🔀 Merging pruned graphs from {len(all_paths)} query results...")

    # Collect all unique nodes and edges
    all_node_ids = set()
    all_edges = set()

    for paths in all_paths:
        for path in paths:
            all_node_ids.update(path.nodes)
            all_edges.update(path.edges)

    logger.info(f"📊 Merged graph: {len(all_node_ids)} nodes, {len(all_edges)} edges")

    # Filter dataframes
    merged_entities = entities_df[
        entities_df['title'].astype(str).isin(all_node_ids)
    ].copy()
    # Deduplicate on stable key to avoid ndarray columns causing errors
    merged_entities = merged_entities.drop_duplicates(subset=['title'])

    def is_edge_in_merged(row):
        edge_tuple = (str(row['source']), str(row['target']))
        edge_tuple_rev = (str(row['target']), str(row['source']))
        return edge_tuple in all_edges or edge_tuple_rev in all_edges

    merged_relationships = relationships_df[
        relationships_df.apply(is_edge_in_merged, axis=1)
    ].copy()
    # Deduplicate on edge endpoints only
    merged_relationships = merged_relationships.drop_duplicates(subset=['source', 'target'])

    # Keep all edges between merged nodes for connectivity
    def connects_merged_nodes(row):
        return (str(row['source']) in all_node_ids and
               str(row['target']) in all_node_ids)

    additional_edges = relationships_df[
        relationships_df.apply(connects_merged_nodes, axis=1)
    ].copy()

    merged_relationships = pd.concat([merged_relationships, additional_edges]).drop_duplicates(subset=['source', 'target'])

    merged_text_units = text_units_df.copy() if text_units_df is not None else None

    return merged_entities, merged_relationships, merged_text_units


def run_pruning_pipeline(workspace: Path, output_dir: Path,
                        queries: List[str], config: Dict[str, Any],
                        strategy: str = "superhfbp") -> Dict[str, Any]:
    """
    Run complete pruning pipeline.

    Args:
        workspace: Path to GraphRAG workspace
        output_dir: Directory to save pruned artifacts
        queries: List of queries to use for pruning
        config: Pruning configuration
        strategy: Pruning strategy ("superhfbp" or "hfbp")

    Returns:
        Dictionary with pruning statistics and metadata
    """
    logger.info("="*80)
    logger.info(f"🚀 Starting {strategy.upper()} Pruning Pipeline")
    logger.info("="*80)

    start_time = datetime.now()

    # Load artifacts
    entities, relationships, text_units = load_graphrag_artifacts(workspace)

    # Initialize pruner
    if strategy == "superhfbp":
        pruner_config = SuperHFBPConfig(**config)
        pruner = SuperHFBP(entities, relationships, text_units, pruner_config)
    else:  # hfbp
        pruner_config = HFBPConfig(**config)
        pruner = HFBPPruning(entities, relationships, text_units, pruner_config)

    # Run pruning for each query and collect paths
    all_paths = []
    query_results = []

    for i, query in enumerate(queries, 1):
        logger.info(f"\n[Query {i}/{len(queries)}] {query[:80]}...")

        # Get initial retrieved nodes
        retrieved_nodes = get_initial_retrieved_nodes(entities, query, config.get('N', 40))

        if not retrieved_nodes:
            logger.warning(f"No entities retrieved for query: {query}")
            continue

        # Execute pruning
        result = pruner.execute(query, retrieved_nodes)
        all_paths.append(result['top_k_paths'])

        query_results.append({
            'query': query,
            'num_paths': result['num_paths_found'],
            'num_context_nodes': result.get('num_context_nodes', 0),
            'is_global_query': result.get('is_global_query', False)
        })

    # Merge all paths into unified pruned graph
    merged_entities, merged_relationships, merged_text_units = merge_pruned_graphs(
        all_paths, entities, relationships, text_units
    )

    # Save pruned artifacts
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n💾 Saving pruned artifacts to {output_dir}...")

    merged_entities.to_parquet(output_dir / "entities.parquet", index=False)
    merged_relationships.to_parquet(output_dir / "relationships.parquet", index=False)

    if merged_text_units is not None:
        merged_text_units.to_parquet(output_dir / "text_units.parquet", index=False)

    # Save metadata
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    metadata = {
        'timestamp': end_time.isoformat(),
        'pruning_method': strategy.upper(),
        'duration_seconds': duration,
        'num_queries': len(queries),
        'queries': queries[:10],  # Save first 10 queries
        'original_stats': {
            'num_entities': len(entities),
            'num_relationships': len(relationships),
            'num_text_units': len(text_units) if text_units is not None else 0
        },
        'pruned_stats': {
            'num_entities': len(merged_entities),
            'num_relationships': len(merged_relationships),
            'num_text_units': len(merged_text_units) if merged_text_units is not None else 0
        },
        'reduction_rate': {
            'entities': 1.0 - (len(merged_entities) / len(entities)) if len(entities) > 0 else 0,
            'relationships': 1.0 - (len(merged_relationships) / len(relationships)) if len(relationships) > 0 else 0
        },
        'config': config,
        'query_results': query_results
    }

    with open(output_dir / "pruning_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"\n✅ Pruning pipeline completed in {duration:.2f}s")
    logger.info(f"📊 Results:")
    logger.info(f"   Entities: {len(entities)} → {len(merged_entities)} "
               f"({100 * metadata['reduction_rate']['entities']:.1f}% reduction)")
    logger.info(f"   Relationships: {len(relationships)} → {len(merged_relationships)} "
               f"({100 * metadata['reduction_rate']['relationships']:.1f}% reduction)")
    logger.info(f"   Processed {len(queries)} queries")
    logger.info(f"   Artifacts saved to: {output_dir}")
    logger.info("="*80)

    return metadata


def load_config_from_file(config_file: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        config = json.load(f)

    logger.info(f"📄 Loaded config from {config_file}")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run SuperHFBP graph pruning to generate pruned GraphRAG artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prune using queries from file
  python run_superhfbp_pruning.py --workspace ../workspace --output ../workspace/pruned_superhfbp --queries queries.txt

  # Prune using auto-generated queries
  python run_superhfbp_pruning.py --workspace ../workspace --output ../workspace/pruned_superhfbp --mode auto --num-queries 10

  # Prune with custom config
  python run_superhfbp_pruning.py --workspace ../workspace --output ../workspace/pruned_superhfbp --config config.json
        """
    )

    parser.add_argument("--workspace", type=Path, required=True,
                       help="Path to GraphRAG workspace containing output/")
    parser.add_argument("--output", type=Path, required=True,
                       help="Directory to save pruned artifacts")
    parser.add_argument("--queries", type=Path,
                       help="File containing queries (one per line)")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "file"],
                       help="Query mode: 'auto' for auto-generated, 'file' to load from file")
    parser.add_argument("--num-queries", type=int, default=10,
                       help="Number of queries to generate in auto mode")
    parser.add_argument("--config", type=Path,
                       help="Path to SuperHFBP config JSON file")
    parser.add_argument("--strategy", type=str, default="superhfbp", choices=["superhfbp", "hfbp"],
                       help="Pruning strategy")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    # Config parameters (if not using --config file)
    parser.add_argument("--N", type=int, default=40,
                       help="Number of top retrieved nodes (default: 40)")
    parser.add_argument("--K", type=int, default=15,
                       help="Number of top paths to return (default: 15)")
    parser.add_argument("--alpha", type=float, default=0.8,
                       help="Resource propagation damping (default: 0.8)")
    parser.add_argument("--theta", type=float, default=0.05,
                       help="Pruning threshold (default: 0.05)")
    parser.add_argument("--beta", type=float, default=0.2,
                       help="Context bonus weight (SuperHFBP only, default: 0.2)")
    parser.add_argument("--context-top-pct", type=float, default=0.2,
                       help="Top percentage for context nodes (SuperHFBP only, default: 0.2)")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate workspace
    if not args.workspace.exists():
        logger.error(f"❌ Workspace not found: {args.workspace}")
        sys.exit(1)

    # Load or generate queries
    if args.mode == "file":
        if not args.queries:
            logger.error("❌ --queries file required when using --mode file")
            sys.exit(1)
        if not args.queries.exists():
            logger.error(f"❌ Queries file not found: {args.queries}")
            sys.exit(1)
        queries = load_queries_from_file(args.queries)
    else:  # auto mode
        # Load entities to generate queries
        entities, _, _ = load_graphrag_artifacts(args.workspace)
        queries = generate_representative_queries(entities, args.num_queries)

    if not queries:
        logger.error("❌ No queries available for pruning")
        sys.exit(1)

    # Load or build configuration
    if args.config:
        if not args.config.exists():
            logger.error(f"❌ Config file not found: {args.config}")
            sys.exit(1)
        config = load_config_from_file(args.config)
    else:
        # Build config from command-line args
        config = {
            'N': args.N,
            'K': args.K,
            'alpha': args.alpha,
            'theta': args.theta,
            'beam_k': 5
        }

        if args.strategy == "superhfbp":
            config.update({
                'beta': args.beta,
                'context_top_pct': args.context_top_pct,
                'context_penalty': 0.8,
                'inter_community_density_threshold': 0.05,
                'use_llm_global_detection': False  # Disable LLM for batch pruning
            })

    # Run pruning pipeline
    try:
        metadata = run_pruning_pipeline(
            workspace=args.workspace,
            output_dir=args.output,
            queries=queries,
            config=config,
            strategy=args.strategy
        )

        logger.info(f"\n🎉 Pruning completed successfully!")
        logger.info(f"📁 Pruned artifacts saved to: {args.output}")
        logger.info(f"📄 Metadata saved to: {args.output / 'pruning_metadata.json'}")

        sys.exit(0)

    except Exception as e:
        logger.error(f"\n❌ Pruning failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
