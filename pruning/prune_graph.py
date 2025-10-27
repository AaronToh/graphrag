#!/usr/bin/env python3
"""
GraphRAG Pruning Lab - Graph Pruning Framework

This script provides the framework for pruning GraphRAG artifacts based on scoring.
You implement the actual pruning logic here.

Framework Structure:
1. Load baseline GraphRAG artifacts
2. Apply scoring algorithms
3. Apply pruning strategies
4. Save pruned artifacts
5. Compare with baseline
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import logging
import json
from datetime import datetime
import networkx as nx

from .scoring_utils import GraphScorer, load_graphrag_artifacts, save_scores

logger = logging.getLogger(__name__)

class GraphPruner:
    """
    High-level framework for graph pruning operations.

    This class coordinates the pruning process:
    1. Load and score graph components
    2. Apply pruning strategies
    3. Generate pruned artifacts
    4. Save results
    """

    def __init__(self, baseline_dir: Path, output_dir: Path):
        """
        Initialize pruner with baseline artifacts.

        Args:
            baseline_dir: Directory containing baseline GraphRAG artifacts
            output_dir: Directory to save pruned artifacts
        """
        self.baseline_dir = baseline_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load baseline data
        self.entities_df, self.relationships_df, self.communities_df = load_graphrag_artifacts(baseline_dir)

        # Initialize scorer
        self.scorer = GraphScorer(self.entities_df, self.relationships_df, self.communities_df)

        # Storage for scores and pruning results
        self.node_scores = None
        self.edge_scores = None
        self.community_scores = None
        self.pruning_config = {}

        logger.info(f"Initialized GraphPruner with baseline from {baseline_dir}")

    def score_components(self, node_weights: Dict = None, edge_weights: Dict = None,
                        community_weights: Dict = None):
        """
        Score all graph components using configured weights.

        Args:
            node_weights: Weights for node scoring methods
            edge_weights: Weights for edge scoring methods
            community_weights: Weights for community scoring methods
        """
        logger.info("🔍 Scoring graph components...")

        # TODO: Implement your scoring logic here
        # self.node_scores = self.scorer.get_combined_node_scores(node_weights)
        # self.edge_scores = self.scorer.get_combined_edge_scores(edge_weights)
        # self.community_scores = self.scorer.get_combined_community_scores(community_weights)

        # Placeholder - replace with your actual scoring calls
        self.node_scores = pd.DataFrame()  # TODO: Implement
        self.edge_scores = pd.DataFrame()  # TODO: Implement
        self.community_scores = pd.DataFrame()  # TODO: Implement

        # Save scores
        save_scores(self.node_scores, self.output_dir, "node_scores")
        save_scores(self.edge_scores, self.output_dir, "edge_scores")
        save_scores(self.community_scores, self.output_dir, "community_scores")

        logger.info("✅ Component scoring completed")

    def prune_nodes(self, strategy: str = "top_k", **kwargs) -> pd.DataFrame:
        """
        Prune nodes based on scoring and strategy.

        Args:
            strategy: Pruning strategy ('top_k', 'threshold', 'percentile')
            **kwargs: Strategy-specific parameters

        Returns:
            Pruned entities DataFrame
        """
        logger.info(f"🪓 Pruning nodes using strategy: {strategy}")

        if self.node_scores is None or self.node_scores.empty:
            logger.warning("No node scores available - skipping node pruning")
            return self.entities_df

        # TODO: Implement your node pruning logic here
        # Example strategies:
        # - 'top_k': Keep top k nodes by score
        # - 'threshold': Keep nodes above score threshold
        # - 'percentile': Keep top percentile of nodes

        pruned_entities = self.entities_df.copy()  # TODO: Implement actual pruning

        logger.info(f"✅ Node pruning completed: {len(pruned_entities)}/{len(self.entities_df)} nodes kept")
        return pruned_entities

    def prune_edges(self, strategy: str = "top_k", **kwargs) -> pd.DataFrame:
        """
        Prune edges based on scoring and strategy.

        Args:
            strategy: Pruning strategy ('top_k', 'threshold', 'percentile')
            **kwargs: Strategy-specific parameters

        Returns:
            Pruned relationships DataFrame
        """
        logger.info(f"🪓 Pruning edges using strategy: {strategy}")

        if self.edge_scores is None or self.edge_scores.empty:
            logger.warning("No edge scores available - skipping edge pruning")
            return self.relationships_df

        # TODO: Implement your edge pruning logic here
        # Example strategies:
        # - 'top_k': Keep top k edges per node
        # - 'threshold': Keep edges above score threshold
        # - 'percentile': Keep top percentile of edges

        pruned_relationships = self.relationships_df.copy()  # TODO: Implement actual pruning

        logger.info(f"✅ Edge pruning completed: {len(pruned_relationships)}/{len(self.relationships_df)} edges kept")
        return pruned_relationships

    def prune_communities(self, strategy: str = "top_k", **kwargs) -> pd.DataFrame:
        """
        Prune communities based on scoring and strategy.

        Args:
            strategy: Pruning strategy ('top_k', 'threshold', 'percentile')
            **kwargs: Strategy-specific parameters

        Returns:
            Pruned communities DataFrame
        """
        logger.info(f"🪓 Pruning communities using strategy: {strategy}")

        if self.community_scores is None or self.community_scores.empty:
            logger.warning("No community scores available - skipping community pruning")
            return self.communities_df

        # TODO: Implement your community pruning logic here
        # Example strategies:
        # - 'top_k': Keep top k communities
        # - 'threshold': Keep communities above score threshold
        # - 'recluster': Re-cluster after node/edge pruning

        pruned_communities = self.communities_df.copy()  # TODO: Implement actual pruning

        logger.info(f"✅ Community pruning completed")
        return pruned_communities

    def apply_pruning_pipeline(self, config: Dict) -> Dict[str, pd.DataFrame]:
        """
        Apply complete pruning pipeline based on configuration.

        Args:
            config: Pruning configuration dictionary

        Returns:
            Dictionary with pruned artifacts
        """
        logger.info("🚀 Starting pruning pipeline...")

        # Store config for reproducibility
        self.pruning_config = config
        timestamp = datetime.now().isoformat()

        # Score components
        self.score_components(
            node_weights=config.get('node_weights'),
            edge_weights=config.get('edge_weights'),
            community_weights=config.get('community_weights')
        )

        # Apply pruning strategies
        pruned_entities = self.prune_nodes(
            strategy=config.get('node_strategy', 'top_k'),
            **config.get('node_params', {})
        )

        pruned_relationships = self.prune_edges(
            strategy=config.get('edge_strategy', 'top_k'),
            **config.get('edge_params', {})
        )

        pruned_communities = self.prune_communities(
            strategy=config.get('community_strategy', 'top_k'),
            **config.get('community_params', {})
        )

        # Save pruned artifacts
        pruned_artifacts = {
            'entities': pruned_entities,
            'relationships': pruned_relationships,
            'communities': pruned_communities,
            'metadata': {
                'timestamp': timestamp,
                'config': config,
                'baseline_stats': self._get_baseline_stats(),
                'pruned_stats': self._get_pruned_stats(pruned_entities, pruned_relationships, pruned_communities)
            }
        }

        self._save_pruned_artifacts(pruned_artifacts)
        logger.info("✅ Pruning pipeline completed")

        return pruned_artifacts

    def _get_baseline_stats(self) -> Dict:
        """Get statistics about baseline artifacts."""
        return {
            'num_entities': len(self.entities_df),
            'num_relationships': len(self.relationships_df),
            'num_communities': len(self.communities_df) if self.communities_df is not None else 0,
        }

    def _get_pruned_stats(self, entities: pd.DataFrame, relationships: pd.DataFrame,
                          communities: pd.DataFrame) -> Dict:
        """Get statistics about pruned artifacts."""
        return {
            'num_entities': len(entities),
            'num_relationships': len(relationships),
            'num_communities': len(communities) if communities is not None else 0,
        }

    def _save_pruned_artifacts(self, artifacts: Dict):
        """Save pruned artifacts to disk."""
        # Save DataFrames
        artifacts['entities'].to_parquet(self.output_dir / "pruned_entities.parquet")
        artifacts['relationships'].to_parquet(self.output_dir / "pruned_relationships.parquet")
        if artifacts['communities'] is not None:
            artifacts['communities'].to_parquet(self.output_dir / "pruned_communities.parquet")

        # Save metadata
        with open(self.output_dir / "pruning_metadata.json", 'w') as f:
            json.dump(artifacts['metadata'], f, indent=2, default=str)

        logger.info(f"💾 Pruned artifacts saved to {self.output_dir}")

    def apply_crumbtrail_pipeline(self,
                                  root_entity: str = None,
                                  protected_fraction: float = 0.2,
                                  protected_selection: str = 'degree_centrality',
                                  max_iterations: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Apply CrumbTrail pruning pipeline.

        Args:
            root_entity: Root node ID (if None, create virtual root)
            protected_fraction: Fraction of nodes to protect (0.0-1.0)
            protected_selection: Method to select protected nodes
                                ('degree_centrality', 'random', 'community_based')
            max_iterations: Maximum number of layers in CrumbTrail

        Returns:
            Dictionary with pruned artifacts and metadata
        """
        from pruning.crumbtrail import crumbtrail_prune

        logger.info("🚀 Starting CrumbTrail pruning pipeline...")
        timestamp = datetime.now().isoformat()

        # Build graph from entities and relationships
        logger.info("Building graph...")
        G = self.scorer.graph  # Use the graph from scorer

        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")

        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )

        # Select protected nodes
        logger.info(f"Selecting protected nodes ({protected_selection} method)...")
        protected_nodes = self._select_protected_nodes(
            G, protected_fraction, protected_selection
        )

        # Select or create root
        if root_entity is None:
            # Create a virtual root node connected to high-degree nodes
            root_entity = "__VIRTUAL_ROOT__"
            G.add_node(root_entity, title="Virtual Root", type="VIRTUAL")
            # Connect root to top 10 highest-degree nodes
            top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
            for node, _ in top_nodes:
                G.add_edge(root_entity, node, weight=1.0, description="Virtual root connection")
            logger.info(f"Created virtual root '{root_entity}' connected to {len(top_nodes)} top nodes")
        elif root_entity not in G:
            raise ValueError(f"Root entity '{root_entity}' not found in graph")

        # Add root to protected nodes
        protected_nodes.add(root_entity)

        logger.info(f"Root: {root_entity}")
        logger.info(f"Protected nodes: {len(protected_nodes)} ({100*len(protected_nodes)/len(G.nodes()):.1f}%)")

        # Run CrumbTrail
        logger.info("Running CrumbTrail algorithm...")
        pruned_graph = crumbtrail_prune(
            G, protected_nodes, root_entity, max_iterations
        )

        # Extract pruned entities and relationships
        logger.info("Extracting pruned artifacts...")
        pruned_node_ids = set(pruned_graph.nodes())

        # Filter entities to only those in pruned graph
        # Note: graph uses 'title' as node ID
        pruned_entities = self.entities_df[
            self.entities_df['title'].isin(pruned_node_ids)
        ].copy()

        # Filter relationships to only those with both endpoints in pruned graph
        pruned_relationships = self.relationships_df[
            self.relationships_df['source'].isin(pruned_node_ids) &
            self.relationships_df['target'].isin(pruned_node_ids)
        ].copy()

        # Remove virtual root from output if it was created
        if root_entity == "__VIRTUAL_ROOT__":
            pruned_entities = pruned_entities[pruned_entities['title'] != root_entity]
            pruned_graph.remove_node(root_entity)

        # Compute pruned stats
        pruned_stats = self._compute_detailed_stats(
            pruned_entities, pruned_relationships, pruned_graph
        )

        # Log reduction statistics
        self._log_reduction_stats(baseline_stats, pruned_stats)

        # Prepare artifacts
        pruned_artifacts = {
            'entities': pruned_entities,
            'relationships': pruned_relationships,
            'communities': None,  # CrumbTrail doesn't preserve communities
            'metadata': {
                'timestamp': timestamp,
                'algorithm': 'CrumbTrail',
                'parameters': {
                    'root_entity': root_entity,
                    'protected_fraction': protected_fraction,
                    'protected_selection': protected_selection,
                    'max_iterations': max_iterations,
                    'num_protected': len(protected_nodes)
                },
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats
            }
        }

        # Save artifacts
        self._save_pruned_artifacts(pruned_artifacts)

        logger.info("✅ CrumbTrail pipeline completed")
        return pruned_artifacts

    def _select_protected_nodes(self, G: nx.DiGraph, fraction: float,
                                method: str) -> Set[str]:
        """
        Select nodes to protect during pruning.

        Args:
            G: Graph
            fraction: Fraction of nodes to protect
            method: Selection method

        Returns:
            Set of protected node IDs
        """
        n_protect = max(1, int(len(G.nodes()) * fraction))

        if method == 'degree_centrality':
            # Select highest-degree nodes
            centrality = nx.degree_centrality(G)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            protected = {node for node, _ in top_nodes[:n_protect]}

        elif method == 'random':
            # Random selection
            import random
            protected = set(random.sample(list(G.nodes()), n_protect))

        elif method == 'community_based':
            # Select nodes from each community (if communities available)
            if self.communities_df is not None and len(self.communities_df) > 0:
                # Get nodes from top communities
                protected = set()
                # This is a simplified version - could be improved
                centrality = nx.degree_centrality(G)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                protected = {node for node, _ in top_nodes[:n_protect]}
            else:
                # Fall back to degree centrality
                logger.warning("No communities available, using degree_centrality")
                return self._select_protected_nodes(G, fraction, 'degree_centrality')

        else:
            raise ValueError(f"Unknown protected selection method: {method}")

        logger.info(f"Selected {len(protected)} protected nodes using {method}")
        return protected

    def _compute_detailed_stats(self, entities_df: pd.DataFrame,
                                relationships_df: pd.DataFrame,
                                graph: nx.DiGraph) -> Dict:
        """
        Compute detailed statistics about the graph.

        Args:
            entities_df: Entities DataFrame
            relationships_df: Relationships DataFrame
            graph: NetworkX graph

        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_entities': len(entities_df),
            'num_relationships': len(relationships_df),
            'num_communities': len(self.communities_df) if self.communities_df is not None else 0,
            'num_nodes_in_graph': len(graph.nodes()),
            'num_edges_in_graph': len(graph.edges()),
        }

        # Graph structure stats
        if len(graph.nodes()) > 0:
            # Isolated nodes
            isolated = list(nx.isolates(graph))
            stats['num_isolated_entities'] = len(isolated)

            # Connected components
            if graph.is_directed():
                wcc = list(nx.weakly_connected_components(graph))
                stats['num_weakly_connected_components'] = len(wcc)
                if wcc:
                    largest = max(wcc, key=len)
                    stats['largest_component_size'] = len(largest)
                    stats['largest_component_pct'] = 100 * len(largest) / len(graph.nodes())
            else:
                cc = list(nx.connected_components(graph))
                stats['num_connected_components'] = len(cc)
                if cc:
                    largest = max(cc, key=len)
                    stats['largest_component_size'] = len(largest)
                    stats['largest_component_pct'] = 100 * len(largest) / len(graph.nodes())

            # Degree statistics
            degrees = [d for n, d in graph.degree()]
            if degrees:
                stats['avg_degree'] = sum(degrees) / len(degrees)
                stats['max_degree'] = max(degrees)
                stats['min_degree'] = min(degrees)

            # Entity type distribution
            if 'type' in entities_df.columns:
                type_counts = entities_df['type'].value_counts().to_dict()
                stats['entity_types'] = type_counts
                stats['num_entity_types'] = len(type_counts)

        return stats

    def _log_reduction_stats(self, baseline_stats: Dict, pruned_stats: Dict):
        """Log reduction statistics."""
        logger.info("\n" + "="*60)
        logger.info("📊 PRUNING STATISTICS")
        logger.info("="*60)

        logger.info("\n🔵 Baseline:")
        logger.info(f"  Entities:      {baseline_stats['num_entities']:,}")
        logger.info(f"  Relationships: {baseline_stats['num_relationships']:,}")
        logger.info(f"  Avg Degree:    {baseline_stats.get('avg_degree', 0):.2f}")
        logger.info(f"  Components:    {baseline_stats.get('num_weakly_connected_components', 0)}")

        logger.info("\n🟢 Pruned:")
        logger.info(f"  Entities:      {pruned_stats['num_entities']:,}")
        logger.info(f"  Relationships: {pruned_stats['num_relationships']:,}")
        logger.info(f"  Avg Degree:    {pruned_stats.get('avg_degree', 0):.2f}")
        logger.info(f"  Components:    {pruned_stats.get('num_weakly_connected_components', 0)}")

        # Calculate reductions
        entity_reduction = 100 * (1 - pruned_stats['num_entities'] / baseline_stats['num_entities'])
        rel_reduction = 100 * (1 - pruned_stats['num_relationships'] / baseline_stats['num_relationships'])

        logger.info("\n📉 Reduction:")
        logger.info(f"  Entities:      {entity_reduction:.1f}%")
        logger.info(f"  Relationships: {rel_reduction:.1f}%")

        logger.info("\n" + "="*60)

    def compare_with_baseline(self) -> Dict:
        """
        Compare pruned artifacts with baseline.

        Returns:
            Dictionary with comparison statistics
        """
        # TODO: Implement comparison logic
        # Compare graph structure, density, component sizes, etc.
        comparison = {
            'reduction_stats': {},
            'quality_metrics': {},
            'structural_changes': {}
        }

        return comparison


def load_pruning_config(config_path: Path) -> Dict:
    """
    Load pruning configuration from file.

    Args:
        config_path: Path to pruning configuration file

    Returns:
        Configuration dictionary
    """
    # TODO: Implement config loading (YAML/JSON)
    # For now, return default config
    return {
        'node_weights': {'degree': 0.4, 'frequency': 0.3, 'semantic': 0.3},
        'edge_weights': {'weight': 0.6, 'plausibility': 0.4},
        'community_weights': {'size': 0.5, 'density': 0.5},
        'node_strategy': 'top_k',
        'node_params': {'k': 1000},
        'edge_strategy': 'top_k',
        'edge_params': {'k_per_node': 10},
        'community_strategy': 'top_k',
        'community_params': {'k': 50}
    }


def main():
    """Main pruning execution."""
    parser = argparse.ArgumentParser(description="Prune GraphRAG artifacts")
    parser.add_argument(
        "--baseline",
        type=str,
        default="../workspace/output",
        help="Directory with baseline GraphRAG artifacts"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../workspace/output/pruned",
        help="Directory to save pruned artifacts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pruning_config.yaml",
        help="Pruning configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("🎯 GraphRAG Pruning Lab - Stage 2: Graph Pruning")
    logger.info(f"📁 Baseline: {args.baseline}")
    logger.info(f"📤 Output: {args.output}")

    # Initialize pruner
    baseline_dir = Path(args.baseline)
    output_dir = Path(args.output)

    if not baseline_dir.exists():
        logger.error(f"❌ Baseline directory not found: {baseline_dir}")
        return 1

    pruner = GraphPruner(baseline_dir, output_dir)

    # Load pruning configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_pruning_config(config_path)
    else:
        logger.warning(f"⚠️ Config file not found: {config_path}, using defaults")
        config = load_pruning_config(None)

    # Apply pruning pipeline
    try:
        pruned_artifacts = pruner.apply_pruning_pipeline(config)

        # Compare with baseline
        comparison = pruner.compare_with_baseline()

        logger.info("🎉 Pruning completed successfully!")
        logger.info(f"📊 Results saved to {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"❌ Pruning failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
