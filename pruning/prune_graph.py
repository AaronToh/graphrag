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

        # Score nodes - default to degree centrality if no weights specified
        if node_weights is None:
            node_weights = {'degree_centrality': 1.0}
        
        node_score_list = []
        for method, weight in node_weights.items():
            if method == 'degree_centrality':
                scores = self.scorer.score_nodes_degree_centrality()
            elif method == 'frequency':
                scores = self.scorer.score_nodes_frequency()
            else:
                logger.warning(f"Unknown node scoring method: {method}, skipping")
                continue
            node_score_list.append(scores * weight)
        
        if node_score_list:
            self.node_scores = pd.DataFrame(pd.concat(node_score_list, axis=1).sum(axis=1), columns=['score'])
        else:
            # Fallback to degree centrality
            scores = self.scorer.score_nodes_degree_centrality()
            self.node_scores = pd.DataFrame(scores, columns=['score'])

        # Score edges - default to weight if available
        if edge_weights is None:
            edge_weights = {'weight': 1.0}
        
        edge_score_list = []
        for method, weight in edge_weights.items():
            if method == 'weight':
                scores = self.scorer.score_edges_weight()
                if isinstance(scores, pd.Series):
                    edge_score_list.append(scores * weight)
                else:
                    logger.warning("Edge scoring returned unexpected type")
            else:
                logger.warning(f"Unknown edge scoring method: {method}, skipping")
                continue
        
        if edge_score_list:
            # Convert Series to dict for easier handling
            combined_scores = pd.concat(edge_score_list, axis=1).sum(axis=1) if len(edge_score_list) > 1 else edge_score_list[0]
            self.edge_scores = pd.DataFrame(combined_scores, columns=['score'])
        else:
            # Fallback to uniform scores
            edge_scores_dict = {}
            for _, rel in self.relationships_df.iterrows():
                edge_key = (rel['source'], rel['target'])
                edge_scores_dict[edge_key] = 1.0
            self.edge_scores = pd.DataFrame(list(edge_scores_dict.items()), columns=['edge', 'score'])
            self.edge_scores.set_index('edge', inplace=True)

        # Communities - keep as is for now
        self.community_scores = pd.DataFrame()

        # Save scores
        save_scores(self.node_scores, self.output_dir, "node_scores")
        save_scores(self.edge_scores, self.output_dir, "edge_scores")
        if self.community_scores is not None and not self.community_scores.empty:
            save_scores(self.community_scores, self.output_dir, "community_scores")

        logger.info("✅ Component scoring completed")

    def prune_nodes(self, strategy: str = "top_k", **kwargs) -> pd.DataFrame:
        """
        Prune nodes based on scoring and strategy.

        Args:
            strategy: Pruning strategy ('top_k', 'threshold', 'percentile')
            **kwargs: Strategy-specific parameters
                - For 'top_k': k (float, 0-100, percentage to keep)
                - For 'threshold': threshold (float, minimum score)
                - For 'percentile': percentile (float, 0-100, percentile to keep)

        Returns:
            Pruned entities DataFrame
        """
        logger.info(f"🪓 Pruning nodes using strategy: {strategy}")

        if self.node_scores is None or self.node_scores.empty:
            logger.warning("No node scores available - computing default scores")
            self.score_components()

        # Get node scores as Series indexed by node title
        if isinstance(self.node_scores, pd.DataFrame):
            if 'score' in self.node_scores.columns:
                node_scores_series = self.node_scores['score']
            else:
                # Use first column
                node_scores_series = self.node_scores.iloc[:, 0]
        else:
            node_scores_series = self.node_scores

        # Map entity titles to scores
        entity_scores = self.entities_df['title'].map(node_scores_series).fillna(0.0)

        if strategy == "top_k":
            # Keep top k% of nodes
            k = kwargs.get('k', 50.0)  # Default 50%
            if k > 1.0:
                k = k / 100.0  # Convert percentage to fraction
            num_keep = int(len(self.entities_df) * k)
            threshold_score = entity_scores.nlargest(num_keep).min()
            keep_mask = entity_scores >= threshold_score
            logger.info(f"  Keeping top {k*100:.1f}% ({num_keep} nodes, threshold={threshold_score:.4f})")
            
        elif strategy == "threshold":
            # Keep nodes above threshold
            threshold = kwargs.get('threshold', 0.5)
            keep_mask = entity_scores >= threshold
            logger.info(f"  Keeping nodes with score >= {threshold} ({keep_mask.sum()} nodes)")
            
        elif strategy == "percentile":
            # Keep top percentile
            percentile = kwargs.get('percentile', 50.0)
            threshold_score = entity_scores.quantile(1.0 - percentile / 100.0)
            keep_mask = entity_scores >= threshold_score
            logger.info(f"  Keeping top {percentile}th percentile (threshold={threshold_score:.4f}, {keep_mask.sum()} nodes)")
        else:
            logger.warning(f"Unknown strategy '{strategy}', returning all nodes")
            return self.entities_df

        pruned_entities = self.entities_df[keep_mask].copy()
        logger.info(f"✅ Node pruning completed: {len(pruned_entities)}/{len(self.entities_df)} nodes kept")
        return pruned_entities

    def prune_edges(self, strategy: str = "top_k", **kwargs) -> pd.DataFrame:
        """
        Prune edges based on scoring and strategy.

        Args:
            strategy: Pruning strategy ('top_k', 'threshold', 'top_k_per_node')
            **kwargs: Strategy-specific parameters
                - For 'top_k': k (float, 0-100, percentage to keep)
                - For 'threshold': threshold (float, minimum score)
                - For 'top_k_per_node': k (int, number of top edges per node)

        Returns:
            Pruned relationships DataFrame
        """
        logger.info(f"🪓 Pruning edges using strategy: {strategy}")

        if self.edge_scores is None or self.edge_scores.empty:
            logger.warning("No edge scores available - computing default scores")
            self.score_components()

        # Get edge scores - handle both DataFrame and Series
        edge_scores_dict = {}
        if isinstance(self.edge_scores, pd.DataFrame) and len(self.edge_scores) > 0:
            if 'score' in self.edge_scores.columns:
                if len(self.edge_scores.index) > 0 and isinstance(self.edge_scores.index[0], tuple):
                    # Index is already (source, target) tuples
                    edge_scores_dict = self.edge_scores['score'].to_dict()
                else:
                    # Need to convert
                    for idx, score in self.edge_scores['score'].items():
                        edge_scores_dict[idx] = score
            elif len(self.edge_scores.columns) > 0:
                # Use first column
                if len(self.edge_scores.index) > 0 and isinstance(self.edge_scores.index[0], tuple):
                    edge_scores_dict = self.edge_scores.iloc[:, 0].to_dict()
                else:
                    for idx, score in self.edge_scores.iloc[:, 0].items():
                        edge_scores_dict[idx] = score
        elif isinstance(self.edge_scores, pd.Series) and len(self.edge_scores) > 0:
            edge_scores_dict = self.edge_scores.to_dict()
        else:
            # Fallback: create scores from relationships
            edge_scores_dict = {}
            for _, rel in self.relationships_df.iterrows():
                edge_key = (rel['source'], rel['target'])
                edge_scores_dict[edge_key] = 1.0

        # Map relationships to scores
        def get_edge_score(row):
            edge_key = (row['source'], row['target'])
            return edge_scores_dict.get(edge_key, 0.0)
        
        relationship_scores = self.relationships_df.apply(get_edge_score, axis=1)

        if strategy == "top_k":
            # Keep top k% of edges
            k = kwargs.get('k', 50.0)  # Default 50%
            if k > 1.0:
                k = k / 100.0  # Convert percentage to fraction
            num_keep = int(len(self.relationships_df) * k)
            threshold_score = relationship_scores.nlargest(num_keep).min()
            keep_mask = relationship_scores >= threshold_score
            logger.info(f"  Keeping top {k*100:.1f}% ({num_keep} edges, threshold={threshold_score:.4f})")
            
        elif strategy == "threshold":
            # Keep edges above threshold
            threshold = kwargs.get('threshold', 0.5)
            keep_mask = relationship_scores >= threshold
            logger.info(f"  Keeping edges with score >= {threshold} ({keep_mask.sum()} edges)")
            
        elif strategy == "top_k_per_node":
            # Keep top k edges per node
            k = kwargs.get('k', 5)
            logger.info(f"  Keeping top {k} edges per node")
            
            # Group by source node and keep top k
            keep_indices = set()
            for source_node in self.relationships_df['source'].unique():
                node_edges = self.relationships_df[self.relationships_df['source'] == source_node]
                node_scores = relationship_scores[node_edges.index]
                top_k_indices = node_scores.nlargest(k).index
                keep_indices.update(top_k_indices)
            
            keep_mask = pd.Series(False, index=self.relationships_df.index)
            keep_mask.loc[list(keep_indices)] = True
            logger.info(f"  Kept {keep_mask.sum()} edges")
        else:
            logger.warning(f"Unknown strategy '{strategy}', returning all edges")
            return self.relationships_df

        pruned_relationships = self.relationships_df[keep_mask].copy()
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
        import shutil

        # Save DataFrames
        artifacts['entities'].to_parquet(self.output_dir / "pruned_entities.parquet")
        artifacts['relationships'].to_parquet(self.output_dir / "pruned_relationships.parquet")
        if artifacts['communities'] is not None:
            artifacts['communities'].to_parquet(self.output_dir / "pruned_communities.parquet")

        # Save metadata
        with open(self.output_dir / "pruning_metadata.json", 'w') as f:
            json.dump(artifacts['metadata'], f, indent=2, default=str)

        # Copy corpus files needed for evaluation
        corpus_files = [
            'text_units.parquet',
            'documents.parquet',
            'community_reports.parquet',
        ]

        for filename in corpus_files:
            source = self.baseline_dir / filename
            if source.exists():
                dest = self.output_dir / filename
                shutil.copy2(source, dest)
                logger.info(f"  Copied {filename} for evaluation")

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

    def apply_aggressive_crumbtrail_pipeline(self,
                                           root_entity: str = None,
                                           protected_fraction: float = 0.03,  # Much more aggressive: 3%
                                           protected_selection: str = 'top_centrality_strict',
                                           max_iterations: int = 1000,
                                           connectivity_threshold: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Apply highly aggressive CrumbTrail pruning pipeline for maximum reduction.

        This method uses multiple aggressive strategies:
        1. Very low protection fraction (default 3%)
        2. Strict centrality-based selection
        3. Connectivity filtering
        4. Multi-stage pruning

        Args:
            root_entity: Root node ID (if None, create virtual root)
            protected_fraction: Fraction of nodes to protect (much lower than standard)
            protected_selection: Method to select protected nodes
            max_iterations: Maximum number of layers in CrumbTrail
            connectivity_threshold: Minimum connectivity to main component

        Returns:
            Dictionary with aggressively pruned artifacts and metadata
        """
        from pruning.crumbtrail import crumbtrail_prune

        logger.info("🔥 Starting AGGRESSIVE CrumbTrail pruning pipeline...")
        timestamp = datetime.now().isoformat()

        # Build graph from entities and relationships
        logger.info("Building graph...")
        G = self.scorer.graph.copy()

        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")

        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )

        # STAGE 1: Remove weakly connected components (aggressive connectivity filtering)
        logger.info("🔥 STAGE 1: Aggressive connectivity filtering...")
        weak_components = list(nx.weakly_connected_components(G))
        largest_component = max(weak_components, key=len)
        
        # Keep only nodes in components that are at least connectivity_threshold of largest
        min_component_size = max(10, int(len(largest_component) * connectivity_threshold))
        nodes_to_keep = set()
        for component in weak_components:
            if len(component) >= min_component_size:
                nodes_to_keep.update(component)
        
        # Remove small components
        nodes_to_remove = set(G.nodes()) - nodes_to_keep
        G.remove_nodes_from(nodes_to_remove)
        logger.info(f"  Removed {len(nodes_to_remove)} nodes from small components")
        logger.info(f"  Remaining nodes: {len(G.nodes())} ({100*len(G.nodes())/len(self.entities_df):.1f}% of original)")

        # STAGE 2: Aggressive protected node selection
        logger.info("🔥 STAGE 2: Ultra-strict protected node selection...")
        protected_nodes = self._select_protected_nodes_aggressive(
            G, protected_fraction, protected_selection
        )

        # Select or create root
        if root_entity is None:
            # Create a virtual root node connected to top centrality nodes only
            root_entity = "__VIRTUAL_ROOT__"
            G.add_node(root_entity, title="Virtual Root", type="VIRTUAL")
            # Connect root to only top 3 highest-degree nodes (more aggressive)
            centrality = nx.degree_centrality(G)
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            for node, _ in top_nodes:
                G.add_edge(root_entity, node, weight=1.0, description="Virtual root connection")
            logger.info(f"Created virtual root '{root_entity}' connected to only {len(top_nodes)} top nodes")
        elif root_entity not in G:
            # Try to find the root in the remaining graph, or pick highest centrality
            centrality = nx.degree_centrality(G)
            root_entity = max(centrality.items(), key=lambda x: x[1])[0]
            logger.warning(f"Original root not found, using highest centrality node: {root_entity}")

        # Add root to protected nodes
        protected_nodes.add(root_entity)

        logger.info(f"Root: {root_entity}")
        logger.info(f"Protected nodes: {len(protected_nodes)} ({100*len(protected_nodes)/len(G.nodes()):.1f}%)")

        # STAGE 3: Run CrumbTrail with aggressive settings
        logger.info("🔥 STAGE 3: Running CrumbTrail with aggressive settings...")
        pruned_graph = crumbtrail_prune(
            G, protected_nodes, root_entity, max_iterations
        )

        # STAGE 4: Post-processing - remove isolated nodes and small components again
        logger.info("🔥 STAGE 4: Post-processing cleanup...")
        # Remove isolated nodes
        isolated = list(nx.isolates(pruned_graph))
        pruned_graph.remove_nodes_from(isolated)
        logger.info(f"  Removed {len(isolated)} isolated nodes after pruning")

        # Extract pruned entities and relationships
        logger.info("Extracting aggressively pruned artifacts...")
        pruned_node_ids = set(pruned_graph.nodes())

        # Filter entities to only those in pruned graph
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
            if root_entity in pruned_graph:
                pruned_graph.remove_node(root_entity)

        # Compute pruned stats
        pruned_stats = self._compute_detailed_stats(
            pruned_entities, pruned_relationships, pruned_graph
        )

        # Log reduction statistics
        self._log_reduction_stats(baseline_stats, pruned_stats)

        # Calculate reduction percentages
        entity_reduction = (1 - len(pruned_entities) / len(self.entities_df)) * 100
        relationship_reduction = (1 - len(pruned_relationships) / len(self.relationships_df)) * 100

        logger.info(f"🔥 AGGRESSIVE PRUNING COMPLETE:")
        logger.info(f"  Entity reduction: {entity_reduction:.1f}% ({len(self.entities_df)} → {len(pruned_entities)})")
        logger.info(f"  Relationship reduction: {relationship_reduction:.1f}% ({len(self.relationships_df)} → {len(pruned_relationships)})")

        # Prepare artifacts
        pruned_artifacts = {
            'entities': pruned_entities,
            'relationships': pruned_relationships,
            'communities': None,  # Aggressive pruning doesn't preserve communities
            'metadata': {
                'timestamp': timestamp,
                'algorithm': 'AggressiveCrumbTrail',
                'parameters': {
                    'root_entity': root_entity,
                    'protected_fraction': protected_fraction,
                    'protected_selection': protected_selection,
                    'max_iterations': max_iterations,
                    'connectivity_threshold': connectivity_threshold,
                    'num_protected': len(protected_nodes)
                },
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats,
                'reduction_percentages': {
                    'entities': entity_reduction,
                    'relationships': relationship_reduction
                }
            }
        }

        # Save artifacts
        self._save_pruned_artifacts(pruned_artifacts)

        return pruned_artifacts

    def apply_ultra_aggressive_crumbtrail_pipeline(self,
                                                 root_entity: str = None,
                                                 protected_fraction: float = 0.005,  # Ultra-aggressive: 0.5%
                                                 protected_selection: str = 'ultra_strict_core',
                                                 max_iterations: int = 1000,
                                                 connectivity_threshold: float = 0.02,
                                                 degree_threshold_percentile: float = 90.0,
                                                 multi_stage_pruning: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Apply ultra-aggressive CrumbTrail pruning pipeline targeting 40-50% reduction.

        This method uses extreme aggressive strategies:
        1. Ultra-low protection fraction (default 0.5%)
        2. Multi-stage filtering (degree, connectivity, centrality)
        3. Strict core-based selection
        4. Iterative pruning rounds

        Args:
            root_entity: Root node ID (if None, create virtual root)
            protected_fraction: Fraction of nodes to protect (ultra-low)
            protected_selection: Method to select protected nodes
            max_iterations: Maximum number of layers in CrumbTrail
            connectivity_threshold: Minimum connectivity to main component
            degree_threshold_percentile: Percentile threshold for degree filtering
            multi_stage_pruning: Whether to apply multiple pruning stages

        Returns:
            Dictionary with ultra-aggressively pruned artifacts and metadata
        """
        from pruning.crumbtrail import crumbtrail_prune

        logger.info("🔥🔥 Starting ULTRA-AGGRESSIVE CrumbTrail pruning pipeline...")
        logger.info(f"🎯 Target: 40-50% entity reduction")
        timestamp = datetime.now().isoformat()

        # Build graph from entities and relationships
        logger.info("Building graph...")
        G = self.scorer.graph.copy()

        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")

        original_node_count = len(G.nodes())
        logger.info(f"Starting with {original_node_count} nodes")

        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )

        # STAGE 1: Aggressive degree-based filtering
        logger.info("🔥🔥 STAGE 1: Ultra-aggressive degree filtering...")
        degrees = dict(G.degree())
        degree_threshold = np.percentile(list(degrees.values()), degree_threshold_percentile)
        
        # Keep only high-degree nodes
        high_degree_nodes = {node for node, degree in degrees.items() if degree >= degree_threshold}
        nodes_to_remove_stage1 = set(G.nodes()) - high_degree_nodes
        G.remove_nodes_from(nodes_to_remove_stage1)
        
        logger.info(f"  Removed {len(nodes_to_remove_stage1)} low-degree nodes (< {degree_threshold:.1f} degree)")
        logger.info(f"  Remaining: {len(G.nodes())} ({100*len(G.nodes())/original_node_count:.1f}% of original)")

        # STAGE 2: Ultra-aggressive connectivity filtering
        logger.info("🔥🔥 STAGE 2: Ultra-aggressive connectivity filtering...")
        weak_components = list(nx.weakly_connected_components(G))
        if weak_components:
            largest_component = max(weak_components, key=len)
            
            # Keep only the largest component and components >= connectivity_threshold of largest
            min_component_size = max(3, int(len(largest_component) * connectivity_threshold))
            nodes_to_keep = set()
            for component in weak_components:
                if len(component) >= min_component_size:
                    nodes_to_keep.update(component)
            
            # Remove small components
            nodes_to_remove_stage2 = set(G.nodes()) - nodes_to_keep
            G.remove_nodes_from(nodes_to_remove_stage2)
            logger.info(f"  Removed {len(nodes_to_remove_stage2)} nodes from small components")
            logger.info(f"  Remaining: {len(G.nodes())} ({100*len(G.nodes())/original_node_count:.1f}% of original)")

        # STAGE 3: Core decomposition filtering
        logger.info("🔥🔥 STAGE 3: Core decomposition filtering...")
        try:
            core_numbers = nx.core_number(G.to_undirected())
            if core_numbers:
                # Keep only nodes in top 20% of core numbers
                core_values = list(core_numbers.values())
                core_threshold = np.percentile(core_values, 80)  # Top 20%
                
                high_core_nodes = {node for node, core_num in core_numbers.items() if core_num >= core_threshold}
                nodes_to_remove_stage3 = set(G.nodes()) - high_core_nodes
                G.remove_nodes_from(nodes_to_remove_stage3)
                
                logger.info(f"  Removed {len(nodes_to_remove_stage3)} low-core nodes (< {core_threshold:.1f} core)")
                logger.info(f"  Remaining: {len(G.nodes())} ({100*len(G.nodes())/original_node_count:.1f}% of original)")
        except Exception as e:
            logger.warning(f"Core decomposition failed: {e}, skipping stage 3")

        # STAGE 4: Ultra-strict protected node selection
        logger.info("🔥🔥 STAGE 4: Ultra-strict protected node selection...")
        protected_nodes = self._select_protected_nodes_ultra_aggressive(
            G, protected_fraction, protected_selection
        )

        # Select or create root
        if root_entity is None:
            # Create a virtual root node connected to only THE top centrality node
            root_entity = "__VIRTUAL_ROOT__"
            G.add_node(root_entity, title="Virtual Root", type="VIRTUAL")
            # Connect root to only the single highest-degree node (ultra-aggressive)
            centrality = nx.degree_centrality(G)
            if centrality:
                top_node = max(centrality.items(), key=lambda x: x[1])[0]
                G.add_edge(root_entity, top_node, weight=1.0, description="Virtual root connection")
                logger.info(f"Created virtual root '{root_entity}' connected to single top node: {top_node}")
        elif root_entity not in G:
            # Try to find the root in the remaining graph, or pick highest centrality
            centrality = nx.degree_centrality(G)
            if centrality:
                root_entity = max(centrality.items(), key=lambda x: x[1])[0]
                logger.warning(f"Original root not found, using highest centrality node: {root_entity}")

        # Add root to protected nodes
        protected_nodes.add(root_entity)

        logger.info(f"Root: {root_entity}")
        logger.info(f"Protected nodes: {len(protected_nodes)} ({100*len(protected_nodes)/len(G.nodes()):.2f}%)")

        # STAGE 5: Run CrumbTrail with ultra-aggressive settings
        logger.info("🔥🔥 STAGE 5: Running CrumbTrail with ultra-aggressive settings...")
        pruned_graph = crumbtrail_prune(
            G, protected_nodes, root_entity, max_iterations
        )

        # STAGE 6: Multi-stage post-processing if enabled
        if multi_stage_pruning:
            logger.info("🔥🔥 STAGE 6: Multi-stage post-processing...")
            
            # Remove isolated nodes
            isolated = list(nx.isolates(pruned_graph))
            pruned_graph.remove_nodes_from(isolated)
            logger.info(f"  Removed {len(isolated)} isolated nodes")
            
            # Remove nodes with degree 1 (leaf nodes) iteratively
            for iteration in range(3):  # Up to 3 iterations
                degree_1_nodes = [node for node, degree in pruned_graph.degree() if degree == 1 and node != root_entity]
                if not degree_1_nodes:
                    break
                pruned_graph.remove_nodes_from(degree_1_nodes)
                logger.info(f"  Iteration {iteration+1}: Removed {len(degree_1_nodes)} leaf nodes")
            
            # Final connectivity check - keep only largest component
            if len(pruned_graph.nodes()) > 0:
                weak_components = list(nx.weakly_connected_components(pruned_graph))
                if len(weak_components) > 1:
                    largest_component = max(weak_components, key=len)
                    nodes_to_remove_final = set(pruned_graph.nodes()) - largest_component
                    pruned_graph.remove_nodes_from(nodes_to_remove_final)
                    logger.info(f"  Final cleanup: Removed {len(nodes_to_remove_final)} nodes from small components")

        # Extract pruned entities and relationships
        logger.info("Extracting ultra-aggressively pruned artifacts...")
        pruned_node_ids = set(pruned_graph.nodes())

        # Filter entities to only those in pruned graph
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
            if root_entity in pruned_graph:
                pruned_graph.remove_node(root_entity)

        # Compute pruned stats
        pruned_stats = self._compute_detailed_stats(
            pruned_entities, pruned_relationships, pruned_graph
        )

        # Log reduction statistics
        self._log_reduction_stats(baseline_stats, pruned_stats)

        # Calculate reduction percentages
        entity_reduction = (1 - len(pruned_entities) / len(self.entities_df)) * 100
        relationship_reduction = (1 - len(pruned_relationships) / len(self.relationships_df)) * 100

        logger.info(f"🔥🔥 ULTRA-AGGRESSIVE PRUNING COMPLETE:")
        logger.info(f"  Entity reduction: {entity_reduction:.1f}% ({len(self.entities_df)} → {len(pruned_entities)})")
        logger.info(f"  Relationship reduction: {relationship_reduction:.1f}% ({len(self.relationships_df)} → {len(pruned_relationships)})")
        
        # Check if we hit our target
        if entity_reduction >= 40:
            logger.info(f"🎯 SUCCESS: Achieved target reduction of {entity_reduction:.1f}% (≥40%)")
        else:
            logger.warning(f"⚠️  Below target: {entity_reduction:.1f}% < 40%")

        # Prepare artifacts
        pruned_artifacts = {
            'entities': pruned_entities,
            'relationships': pruned_relationships,
            'communities': None,  # Ultra-aggressive pruning doesn't preserve communities
            'metadata': {
                'timestamp': timestamp,
                'algorithm': 'UltraAggressiveCrumbTrail',
                'parameters': {
                    'root_entity': root_entity,
                    'protected_fraction': protected_fraction,
                    'protected_selection': protected_selection,
                    'max_iterations': max_iterations,
                    'connectivity_threshold': connectivity_threshold,
                    'degree_threshold_percentile': degree_threshold_percentile,
                    'multi_stage_pruning': multi_stage_pruning,
                    'num_protected': len(protected_nodes)
                },
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats,
                'reduction_percentages': {
                    'entities': entity_reduction,
                    'relationships': relationship_reduction
                }
            }
        }

        # Save artifacts
        self._save_pruned_artifacts(pruned_artifacts)

        return pruned_artifacts

    def _select_protected_nodes_ultra_aggressive(self, G: nx.DiGraph, fraction: float,
                                               method: str) -> Set[str]:
        """
        Ultra-aggressively select nodes to protect during pruning with extremely strict criteria.

        Args:
            G: Graph
            fraction: Fraction of nodes to protect (should be ultra-low, e.g., 0.001-0.01)
            method: Selection method

        Returns:
            Set of protected node IDs
        """
        n_protect = max(2, int(len(G.nodes()) * fraction))  # Minimum 2 nodes

        if method == 'ultra_strict_core':
            # Select only nodes from the absolute highest k-core
            try:
                core_numbers = nx.core_number(G.to_undirected())
                if core_numbers:
                    max_core = max(core_numbers.values())
                    max_core_nodes = [node for node, core_num in core_numbers.items() if core_num == max_core]
                    
                    if len(max_core_nodes) <= n_protect:
                        protected = set(max_core_nodes)
                    else:
                        # From max core nodes, select top by combined centrality
                        centrality = nx.degree_centrality(G)
                        betweenness = nx.betweenness_centrality(G, k=min(500, len(G.nodes())))
                        
                        combined_scores = {}
                        for node in max_core_nodes:
                            combined_scores[node] = (
                                0.8 * centrality.get(node, 0) + 
                                0.2 * betweenness.get(node, 0)
                            )
                        
                        top_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                        protected = {node for node, _ in top_nodes[:n_protect]}
                else:
                    # Fall back to degree centrality
                    return self._select_protected_nodes_ultra_aggressive(G, fraction, 'ultra_strict_centrality')
            except:
                # Fall back to degree centrality
                return self._select_protected_nodes_ultra_aggressive(G, fraction, 'ultra_strict_centrality')

        elif method == 'ultra_strict_centrality':
            # Select only the absolute top centrality nodes with multiple measures
            centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G, k=min(500, len(G.nodes())))
            try:
                closeness = nx.closeness_centrality(G)
            except:
                closeness = {node: 0 for node in G.nodes()}
            
            # Combine multiple centrality measures
            combined_scores = {}
            for node in G.nodes():
                combined_scores[node] = (
                    0.5 * centrality.get(node, 0) + 
                    0.3 * betweenness.get(node, 0) +
                    0.2 * closeness.get(node, 0)
                )
            
            top_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            protected = {node for node, _ in top_nodes[:n_protect]}

        elif method == 'ultra_strict_hubs':
            # Select only the absolute top hub nodes (top 1% by degree)
            degrees = dict(G.degree())
            degree_threshold = np.percentile(list(degrees.values()), 99)  # Top 1%
            hub_candidates = [node for node, degree in degrees.items() if degree >= degree_threshold]
            
            if len(hub_candidates) <= n_protect:
                protected = set(hub_candidates)
            else:
                # From hub candidates, select top by centrality
                centrality = nx.degree_centrality(G)
                hub_scores = [(node, centrality[node]) for node in hub_candidates]
                hub_scores.sort(key=lambda x: x[1], reverse=True)
                protected = {node for node, _ in hub_scores[:n_protect]}

        else:
            # Fall back to ultra strict centrality
            return self._select_protected_nodes_ultra_aggressive(G, fraction, 'ultra_strict_centrality')

        logger.info(f"Ultra-aggressively selected {len(protected)} protected nodes using {method}")
        return protected

    def _select_protected_nodes_aggressive(self, G: nx.DiGraph, fraction: float,
                                         method: str) -> Set[str]:
        """
        Aggressively select nodes to protect during pruning with stricter criteria.

        Args:
            G: Graph
            fraction: Fraction of nodes to protect (should be very low, e.g., 0.02-0.05)
            method: Selection method

        Returns:
            Set of protected node IDs
        """
        n_protect = max(5, int(len(G.nodes()) * fraction))  # Minimum 5 nodes

        if method == 'top_centrality_strict':
            # Select only the absolute highest centrality nodes
            centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G, k=min(1000, len(G.nodes())))  # Sample for large graphs
            
            # Combine centrality measures with heavy weight on degree
            combined_scores = {}
            for node in G.nodes():
                combined_scores[node] = (
                    0.7 * centrality.get(node, 0) + 
                    0.3 * betweenness.get(node, 0)
                )
            
            top_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            protected = {node for node, _ in top_nodes[:n_protect]}

        elif method == 'hub_nodes_only':
            # Select only nodes that are true hubs (very high degree)
            degrees = dict(G.degree())
            degree_threshold = np.percentile(list(degrees.values()), 95)  # Top 5% by degree
            hub_candidates = [node for node, degree in degrees.items() if degree >= degree_threshold]
            
            # From hub candidates, select top by centrality
            centrality = nx.degree_centrality(G)
            hub_scores = [(node, centrality[node]) for node in hub_candidates]
            hub_scores.sort(key=lambda x: x[1], reverse=True)
            
            protected = {node for node, _ in hub_scores[:n_protect]}

        elif method == 'core_nodes_only':
            # Select nodes from the k-core with highest k
            try:
                core_numbers = nx.core_number(G.to_undirected())
                max_core = max(core_numbers.values()) if core_numbers else 1
                
                # Start from highest core and work down until we have enough nodes
                protected = set()
                for k in range(max_core, 0, -1):
                    core_nodes = [node for node, core_num in core_numbers.items() if core_num >= k]
                    if len(core_nodes) >= n_protect:
                        # Select top centrality nodes from this core
                        centrality = nx.degree_centrality(G)
                        core_scores = [(node, centrality[node]) for node in core_nodes]
                        core_scores.sort(key=lambda x: x[1], reverse=True)
                        protected = {node for node, _ in core_scores[:n_protect]}
                        break
                    else:
                        protected.update(core_nodes)
                        
                if len(protected) < n_protect:
                    # Fall back to degree centrality
                    return self._select_protected_nodes_aggressive(G, fraction, 'top_centrality_strict')
                        
            except:
                # Fall back to degree centrality if core computation fails
                return self._select_protected_nodes_aggressive(G, fraction, 'top_centrality_strict')

        else:
            # Fall back to strict centrality
            return self._select_protected_nodes_aggressive(G, fraction, 'top_centrality_strict')

        logger.info(f"Aggressively selected {len(protected)} protected nodes using {method}")
        return protected

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

    def _analyze_graph_regions(self, G: nx.DiGraph) -> Dict[str, str]:
        """
        Analyze graph structure and classify nodes into different regions.
        
        Classifies nodes into:
        - 'dense_core': High clustering coefficient, high degree
        - 'sparse_periphery': Low degree, isolated components
        - 'high_degree_hub': Degree > 75th percentile
        - 'low_degree_leaf': Degree = 1 (leaf nodes)
        - 'community_bridge': Nodes connecting multiple communities
        
        Args:
            G: NetworkX graph to analyze
            
        Returns:
            Dictionary mapping node IDs to region types
        """
        logger.info("🔍 Analyzing graph regions...")
        node_regions = {}
        
        if len(G.nodes()) == 0:
            return node_regions
        
        # Compute degree statistics
        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        if not degree_values:
            return node_regions
        
        # Calculate quartiles
        degree_75th = np.percentile(degree_values, 75)
        degree_25th = np.percentile(degree_values, 25)
        degree_median = np.median(degree_values)
        
        logger.info(f"  Degree statistics: min={min(degree_values)}, "
                   f"median={degree_median:.1f}, 75th={degree_75th:.1f}, max={max(degree_values)}")
        
        # Identify high-degree hubs (top 25%)
        hubs = {node for node, deg in degrees.items() if deg >= degree_75th}
        logger.info(f"  High-degree hubs: {len(hubs)} nodes (degree >= {degree_75th:.1f})")
        
        # Identify low-degree leaves (degree = 1)
        leaves = {node for node, deg in degrees.items() if deg == 1}
        logger.info(f"  Low-degree leaves: {len(leaves)} nodes")
        
        # Compute clustering coefficient (for dense regions)
        # Use local clustering for undirected, or convert to undirected for approximation
        try:
            if G.is_directed():
                G_undirected = G.to_undirected()
            else:
                G_undirected = G
            
            clustering = nx.clustering(G_undirected)
            clustering_values = [c for c in clustering.values() if c > 0]
            if clustering_values:
                clustering_median = np.median(clustering_values)
                clustering_75th = np.percentile(clustering_values, 75)
            else:
                clustering_median = 0.0
                clustering_75th = 0.0
        except Exception as e:
            logger.warning(f"  Could not compute clustering: {e}, using degree-based classification")
            clustering = {}
            clustering_median = 0.0
            clustering_75th = 0.0
        
        # Identify community bridges (nodes connecting multiple communities)
        bridges = set()
        if self.communities_df is not None and len(self.communities_df) > 0:
            # Map nodes to communities
            node_to_communities = {}
            for _, comm in self.communities_df.iterrows():
                comm_id = comm.get('id', comm.get('community_id', None))
                if comm_id is None:
                    continue
                
                # Get entities in this community
                if 'entity_ids' in comm:
                    entity_ids = comm['entity_ids']
                    if isinstance(entity_ids, list):
                        for entity_id in entity_ids:
                            entity = self.entities_df[self.entities_df['id'] == entity_id]
                            if len(entity) > 0:
                                node_title = entity.iloc[0]['title']
                                if node_title not in node_to_communities:
                                    node_to_communities[node_title] = set()
                                node_to_communities[node_title].add(comm_id)
            
            # Nodes connecting multiple communities (have neighbors in different communities)
            for node in G.nodes():
                if node not in node_to_communities:
                    continue
                node_comms = node_to_communities[node]
                if len(node_comms) > 1:
                    bridges.add(node)
                else:
                    # Check if neighbors are in different communities
                    neighbor_comms = set()
                    for neighbor in G.neighbors(node):
                        if neighbor in node_to_communities:
                            neighbor_comms.update(node_to_communities[neighbor])
                    if len(neighbor_comms) > 1:
                        bridges.add(node)
        
        logger.info(f"  Community bridges: {len(bridges)} nodes")
        
        # Classify each node
        for node in G.nodes():
            node_degree = degrees[node]
            node_clustering = clustering.get(node, 0.0)
            
            # Priority classification (most specific first)
            if node in bridges:
                node_regions[node] = 'community_bridge'
            elif node in hubs:
                node_regions[node] = 'high_degree_hub'
            elif node in leaves:
                node_regions[node] = 'low_degree_leaf'
            elif node_clustering >= clustering_75th and node_degree >= degree_median:
                node_regions[node] = 'dense_core'
            elif node_degree <= degree_25th:
                node_regions[node] = 'sparse_periphery'
            else:
                # Default: medium connectivity
                node_regions[node] = 'sparse_periphery'
        
        # Log region distribution
        region_counts = {}
        for region in node_regions.values():
            region_counts[region] = region_counts.get(region, 0) + 1
        
        logger.info("  Region distribution:")
        for region, count in sorted(region_counts.items()):
            pct = 100 * count / len(node_regions) if node_regions else 0
            logger.info(f"    {region}: {count} nodes ({pct:.1f}%)")
        
        logger.info(f"✅ Graph region analysis complete: {len(node_regions)} nodes classified")
        return node_regions

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

    def apply_kgtrimmer_pipeline(
        self,
        collective_weight: float = 0.5,
        holistic_weight: float = 0.5,
        min_importance_percentile: float = 0.2,
        preserve_connectivity: bool = True,
        max_iterations: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply KGTrimmer pruning pipeline.

        Args:
            collective_weight: Weight for community consensus (default: 0.5)
            holistic_weight: Weight for global importance (default: 0.5)
            min_importance_percentile: Keep top N% of nodes (default: 0.2)
            preserve_connectivity: Whether to maintain graph connectivity (default: True)
            max_iterations: Maximum pruning iterations (default: 10)

        Returns:
            Dictionary with pruned artifacts and metadata
        """
        from pruning.kgtrimmer import kgtrimmer_prune

        logger.info("🚀 Starting KGTrimmer pruning pipeline...")
        timestamp = datetime.now().isoformat()

        # Build graph from entities and relationships
        logger.info("Building graph...")
        G = self.scorer.graph

        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")

        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )

        # Run KGTrimmer
        logger.info("Running KGTrimmer algorithm...")
        pruned_graph = kgtrimmer_prune(
            G,
            self.entities_df,
            self.communities_df,
            collective_weight=collective_weight,
            holistic_weight=holistic_weight,
            min_importance_percentile=min_importance_percentile,
            preserve_connectivity=preserve_connectivity,
            max_iterations=max_iterations
        )

        # Extract pruned entities and relationships
        logger.info("Extracting pruned artifacts...")
        pruned_node_ids = set(pruned_graph.nodes())

        # Filter entities to only those in pruned graph
        pruned_entities = self.entities_df[
            self.entities_df['title'].isin(pruned_node_ids)
        ].copy()

        # Filter relationships to only those with both endpoints in pruned graph
        pruned_relationships = self.relationships_df[
            self.relationships_df['source'].isin(pruned_node_ids) &
            self.relationships_df['target'].isin(pruned_node_ids)
        ].copy()

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
            'communities': self.communities_df.copy() if self.communities_df is not None else None,
            'metadata': {
                'timestamp': timestamp,
                'algorithm': 'KGTrimmer',
                'parameters': {
                    'collective_weight': collective_weight,
                    'holistic_weight': holistic_weight,
                    'min_importance_percentile': min_importance_percentile,
                    'preserve_connectivity': preserve_connectivity,
                    'max_iterations': max_iterations
                },
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats
            }
        }

        # Save artifacts
        self._save_pruned_artifacts(pruned_artifacts)

        logger.info("✅ KGTrimmer pipeline completed")
        return pruned_artifacts

    def apply_pog_pipeline(
        self,
        seed_method: str = 'degree_centrality',
        num_seeds: int = 50,
        max_path_length: int = 5,
        top_k_paths: int = 100,
        llm_provider: str = 'openai',
        llm_model: str = 'gpt-4o-mini',
        llm_api_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        sbert_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        semantic_threshold: float = 0.7
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply POG (Path Over Graph) pruning pipeline.

        Args:
            seed_method: Method to select seed nodes (default: 'degree_centrality')
            num_seeds: Number of seed nodes (default: 50)
            max_path_length: Maximum path length to explore (default: 5)
            top_k_paths: Number of top paths to keep (default: 100)
            llm_provider: LLM provider for path scoring (default: 'openai')
            llm_model: Model name for path evaluation (default: 'gpt-4o-mini')
            llm_api_base_url: API base URL (for Ollama/OpenRouter)
            llm_api_key: API key for authentication
            sbert_model: Sentence transformer model (default: 'sentence-transformers/all-MiniLM-L6-v2')
            semantic_threshold: Minimum semantic similarity for paths (default: 0.7)

        Returns:
            Dictionary with pruned artifacts and metadata
        """
        from pruning.pog import pog_prune

        logger.info("🚀 Starting POG pruning pipeline...")
        timestamp = datetime.now().isoformat()

        # Build graph from entities and relationships
        logger.info("Building graph...")
        G = self.scorer.graph

        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")

        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )

        # Run POG
        logger.info("Running POG algorithm...")
        pruned_graph = pog_prune(
            G,
            self.entities_df,
            seed_method=seed_method,
            num_seeds=num_seeds,
            max_path_length=max_path_length,
            top_k_paths=top_k_paths,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_base_url=llm_api_base_url,
            llm_api_key=llm_api_key,
            sbert_model=sbert_model,
            semantic_threshold=semantic_threshold
        )

        # Extract pruned entities and relationships
        logger.info("Extracting pruned artifacts...")
        pruned_node_ids = set(pruned_graph.nodes())

        # Filter entities to only those in pruned graph
        pruned_entities = self.entities_df[
            self.entities_df['title'].isin(pruned_node_ids)
        ].copy()

        # Filter relationships to only those with both endpoints in pruned graph
        pruned_relationships = self.relationships_df[
            self.relationships_df['source'].isin(pruned_node_ids) &
            self.relationships_df['target'].isin(pruned_node_ids)
        ].copy()

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
            'communities': None,  # POG doesn't preserve communities
            'metadata': {
                'timestamp': timestamp,
                'algorithm': 'POG',
                'parameters': {
                    'seed_method': seed_method,
                    'num_seeds': num_seeds,
                    'max_path_length': max_path_length,
                    'top_k_paths': top_k_paths,
                    'llm_provider': llm_provider,
                    'llm_model': llm_model,
                    'sbert_model': sbert_model,
                    'semantic_threshold': semantic_threshold
                },
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats
            }
        }

        # Save artifacts
        self._save_pruned_artifacts(pruned_artifacts)

        logger.info("✅ POG pipeline completed")
        return pruned_artifacts

    def apply_pathrag_pipeline(
        self,
        alpha: float = 0.8,
        theta: float = 0.05,
        top_n_nodes: int = 40,
        top_k_paths: int = 15,
        max_path_length: int = 5,
        seed_method: str = 'degree_centrality',
        path_scoring_method: str = 'avg_edge_flow'
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply PathRAG pruning pipeline.

        Args:
            alpha: Flow decay factor (default: 0.8)
            theta: Early stopping threshold (default: 0.05)
            top_n_nodes: Number of seed nodes (default: 40)
            top_k_paths: Number of paths to keep (default: 15)
            max_path_length: Maximum path length (default: 5)
            seed_method: Method to select seeds (default: 'degree_centrality')
            path_scoring_method: How to score paths (default: 'avg_edge_flow')

        Returns:
            Dictionary with pruned artifacts and metadata
        """
        from pruning.pathrag import pathrag_prune

        logger.info("🚀 Starting PathRAG pruning pipeline...")
        timestamp = datetime.now().isoformat()

        # Build graph from entities and relationships
        logger.info("Building graph...")
        G = self.scorer.graph

        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")

        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )

        # Run PathRAG
        logger.info("Running PathRAG algorithm...")
        pruned_graph = pathrag_prune(
            G,
            self.entities_df,
            alpha=alpha,
            theta=theta,
            top_n_nodes=top_n_nodes,
            top_k_paths=top_k_paths,
            max_path_length=max_path_length,
            seed_method=seed_method,
            path_scoring_method=path_scoring_method
        )

        # Extract pruned entities and relationships
        logger.info("Extracting pruned artifacts...")
        pruned_node_ids = set(pruned_graph.nodes())

        # Filter entities to only those in pruned graph
        pruned_entities = self.entities_df[
            self.entities_df['title'].isin(pruned_node_ids)
        ].copy()

        # Filter relationships to only those with both endpoints in pruned graph
        pruned_relationships = self.relationships_df[
            self.relationships_df['source'].isin(pruned_node_ids) &
            self.relationships_df['target'].isin(pruned_node_ids)
        ].copy()

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
            'communities': None,  # PathRAG doesn't preserve communities
            'metadata': {
                'timestamp': timestamp,
                'algorithm': 'PathRAG',
                'parameters': {
                    'alpha': alpha,
                    'theta': theta,
                    'top_n_nodes': top_n_nodes,
                    'top_k_paths': top_k_paths,
                    'max_path_length': max_path_length,
                    'seed_method': seed_method,
                    'path_scoring_method': path_scoring_method
                },
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats
            }
        }

        # Save artifacts
        self._save_pruned_artifacts(pruned_artifacts)

        logger.info("✅ PathRAG pipeline completed")
        return pruned_artifacts

    def apply_pathrag_hybrid_pipeline(
        self,
        top_n_nodes: int = 500,
        top_k_paths: int = 5000,
        max_path_length: int = 6,
        node_retention_pct: float = 0.3,
        node_scoring_method: str = 'degree_centrality',
        alpha: float = 0.8,
        theta: float = 0.05,
        seed_method: str = 'degree_centrality',
        path_scoring_method: str = 'avg_edge_flow'
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply PathRAG hybrid pruning: combines path-based pruning with node-based retention.
        
        This method:
        1. Runs PathRAG to identify important paths
        2. Keeps all nodes in those paths
        3. Additionally keeps top N% of nodes by centrality/importance
        
        Args:
            top_n_nodes: Number of seed nodes for PathRAG
            top_k_paths: Number of paths to keep
            max_path_length: Maximum path length
            node_retention_pct: Percentage of additional nodes to keep (0-1)
            node_scoring_method: Method to score nodes ('degree_centrality', 'pagerank', 'frequency')
            alpha: Flow decay factor for PathRAG
            theta: Early stopping threshold for PathRAG
            seed_method: Method to select seeds
            path_scoring_method: How to score paths
            
        Returns:
            Dictionary with pruned artifacts and metadata
        """
        from pruning.pathrag import pathrag_prune
        
        logger.info("🚀 Starting PathRAG Hybrid pruning pipeline...")
        logger.info(f"  Path-based: {top_k_paths} paths from {top_n_nodes} seeds")
        logger.info(f"  Node-based: Top {node_retention_pct*100:.1f}% nodes by {node_scoring_method}")
        timestamp = datetime.now().isoformat()
        
        # Build graph
        G = self.scorer.graph
        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")
        
        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )
        
        # Step 1: Run PathRAG to get path-based nodes
        logger.info("\nStep 1: Running PathRAG to identify important paths...")
        path_graph = pathrag_prune(
            G,
            self.entities_df,
            alpha=alpha,
            theta=theta,
            top_n_nodes=top_n_nodes,
            top_k_paths=top_k_paths,
            max_path_length=max_path_length,
            seed_method=seed_method,
            path_scoring_method=path_scoring_method
        )
        path_nodes = set(path_graph.nodes())
        logger.info(f"  ✓ PathRAG identified {len(path_nodes)} nodes in paths")
        
        # Step 2: Score all nodes for node-based retention
        logger.info(f"\nStep 2: Scoring nodes using {node_scoring_method}...")
        if node_scoring_method == 'degree_centrality':
            node_scores = self.scorer.score_nodes_degree_centrality()
        elif node_scoring_method == 'pagerank':
            import networkx as nx
            pagerank = nx.pagerank(G, max_iter=100)
            node_scores = pd.Series(pagerank)
        elif node_scoring_method == 'frequency':
            node_scores = self.scorer.score_nodes_frequency()
        else:
            logger.warning(f"Unknown scoring method {node_scoring_method}, using degree_centrality")
            node_scores = self.scorer.score_nodes_degree_centrality()
        
        # Step 3: Select top N% of nodes (excluding those already in paths)
        logger.info(f"\nStep 3: Selecting top {node_retention_pct*100:.1f}% nodes...")
        num_additional = int(len(self.entities_df) * node_retention_pct)
        
        # Get scores for entities
        entity_scores = self.entities_df['title'].map(node_scores).fillna(0.0)
        
        # Sort and get top nodes, excluding path nodes
        non_path_entities = self.entities_df[~self.entities_df['title'].isin(path_nodes)]
        if len(non_path_entities) > 0:
            non_path_scores = non_path_entities['title'].map(node_scores).fillna(0.0)
            top_additional = non_path_scores.nlargest(min(num_additional, len(non_path_entities)))
            additional_nodes = list(top_additional.index)
            additional_entity_titles = set(non_path_entities.loc[additional_nodes, 'title'])
        else:
            additional_entity_titles = set()
        
        logger.info(f"  ✓ Selected {len(additional_entity_titles)} additional nodes by {node_scoring_method}")
        
        # Step 4: Combine path nodes and additional nodes
        final_nodes = path_nodes | additional_entity_titles
        logger.info(f"\nStep 4: Combined {len(path_nodes)} path nodes + {len(additional_entity_titles)} additional nodes = {len(final_nodes)} total nodes")
        
        # Step 5: Build final pruned graph
        logger.info("\nStep 5: Building final pruned graph...")
        pruned_graph = G.subgraph(final_nodes).copy()
        
        # Extract pruned entities and relationships
        logger.info("Extracting pruned artifacts...")
        pruned_entities = self.entities_df[
            self.entities_df['title'].isin(final_nodes)
        ].copy()
        
        pruned_relationships = self.relationships_df[
            self.relationships_df['source'].isin(final_nodes) &
            self.relationships_df['target'].isin(final_nodes)
        ].copy()
        
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
            'communities': self.communities_df.copy() if self.communities_df is not None else None,
            'metadata': {
                'timestamp': timestamp,
                'algorithm': 'PathRAG_Hybrid',
                'parameters': {
                    'top_n_nodes': top_n_nodes,
                    'top_k_paths': top_k_paths,
                    'max_path_length': max_path_length,
                    'node_retention_pct': node_retention_pct,
                    'node_scoring_method': node_scoring_method,
                    'alpha': alpha,
                    'theta': theta,
                    'path_nodes': len(path_nodes),
                    'additional_nodes': len(additional_entity_titles),
                    'total_nodes': len(final_nodes)
                },
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats
            }
        }
        
        # Save artifacts
        self._save_pruned_artifacts(pruned_artifacts)
        
        logger.info("✅ PathRAG Hybrid pipeline completed")
        return pruned_artifacts

    def apply_pog_hybrid_pipeline(
        self,
        num_seeds: int = 300,
        top_k_paths: int = 5000,
        max_path_length: int = 7,
        node_retention_pct: float = 0.3,
        node_scoring_method: str = 'degree_centrality',
        seed_method: str = 'degree_centrality',
        llm_provider: str = 'openai',
        llm_model: str = 'gpt-4o-mini',
        llm_api_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        sbert_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        semantic_threshold: float = 0.7
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply POG hybrid pruning: combines path-based pruning with node-based retention.
        
        This method:
        1. Runs POG to identify important paths
        2. Keeps all nodes in those paths
        3. Additionally keeps top N% of nodes by centrality/importance
        
        Args:
            num_seeds: Number of seed nodes for POG
            top_k_paths: Number of paths to keep
            max_path_length: Maximum path length
            node_retention_pct: Percentage of additional nodes to keep (0-1)
            node_scoring_method: Method to score nodes ('degree_centrality', 'pagerank', 'frequency')
            seed_method: Method to select seeds
            llm_provider: LLM provider for path scoring
            llm_model: LLM model name
            llm_api_base_url: LLM API base URL
            llm_api_key: LLM API key
            sbert_model: SBERT model name
            semantic_threshold: Semantic similarity threshold
            
        Returns:
            Dictionary with pruned artifacts and metadata
        """
        from pruning.pog import pog_prune
        
        logger.info("🚀 Starting POG Hybrid pruning pipeline...")
        logger.info(f"  Path-based: {top_k_paths} paths from {num_seeds} seeds")
        logger.info(f"  Node-based: Top {node_retention_pct*100:.1f}% nodes by {node_scoring_method}")
        timestamp = datetime.now().isoformat()
        
        # Build graph
        G = self.scorer.graph
        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")
        
        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )
        
        # Step 1: Run POG to get path-based nodes
        logger.info("\nStep 1: Running POG to identify important paths...")
        path_graph = pog_prune(
            G,
            self.entities_df,
            seed_method=seed_method,
            num_seeds=num_seeds,
            max_path_length=max_path_length,
            top_k_paths=top_k_paths,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_base_url=llm_api_base_url,
            llm_api_key=llm_api_key,
            sbert_model=sbert_model,
            semantic_threshold=semantic_threshold
        )
        path_nodes = set(path_graph.nodes())
        logger.info(f"  ✓ POG identified {len(path_nodes)} nodes in paths")
        
        # Step 2: Score all nodes for node-based retention
        logger.info(f"\nStep 2: Scoring nodes using {node_scoring_method}...")
        if node_scoring_method == 'degree_centrality':
            node_scores = self.scorer.score_nodes_degree_centrality()
        elif node_scoring_method == 'pagerank':
            import networkx as nx
            pagerank = nx.pagerank(G, max_iter=100)
            node_scores = pd.Series(pagerank)
        elif node_scoring_method == 'frequency':
            node_scores = self.scorer.score_nodes_frequency()
        else:
            logger.warning(f"Unknown scoring method {node_scoring_method}, using degree_centrality")
            node_scores = self.scorer.score_nodes_degree_centrality()
        
        # Step 3: Select top N% of nodes (excluding those already in paths)
        logger.info(f"\nStep 3: Selecting top {node_retention_pct*100:.1f}% nodes...")
        num_additional = int(len(self.entities_df) * node_retention_pct)
        
        # Get scores for entities
        entity_scores = self.entities_df['title'].map(node_scores).fillna(0.0)
        
        # Sort and get top nodes, excluding path nodes
        non_path_entities = self.entities_df[~self.entities_df['title'].isin(path_nodes)]
        if len(non_path_entities) > 0:
            non_path_scores = non_path_entities['title'].map(node_scores).fillna(0.0)
            top_additional = non_path_scores.nlargest(min(num_additional, len(non_path_entities)))
            additional_nodes = list(top_additional.index)
            additional_entity_titles = set(non_path_entities.loc[additional_nodes, 'title'])
        else:
            additional_entity_titles = set()
        
        logger.info(f"  ✓ Selected {len(additional_entity_titles)} additional nodes by {node_scoring_method}")
        
        # Step 4: Combine path nodes and additional nodes
        final_nodes = path_nodes | additional_entity_titles
        logger.info(f"\nStep 4: Combined {len(path_nodes)} path nodes + {len(additional_entity_titles)} additional nodes = {len(final_nodes)} total nodes")
        
        # Step 5: Build final pruned graph
        logger.info("\nStep 5: Building final pruned graph...")
        pruned_graph = G.subgraph(final_nodes).copy()
        
        # Extract pruned entities and relationships
        logger.info("Extracting pruned artifacts...")
        pruned_entities = self.entities_df[
            self.entities_df['title'].isin(final_nodes)
        ].copy()
        
        pruned_relationships = self.relationships_df[
            self.relationships_df['source'].isin(final_nodes) &
            self.relationships_df['target'].isin(final_nodes)
        ].copy()
        
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
            'communities': self.communities_df.copy() if self.communities_df is not None else None,
            'metadata': {
                'timestamp': timestamp,
                'algorithm': 'POG_Hybrid',
                'parameters': {
                    'num_seeds': num_seeds,
                    'top_k_paths': top_k_paths,
                    'max_path_length': max_path_length,
                    'node_retention_pct': node_retention_pct,
                    'node_scoring_method': node_scoring_method,
                    'path_nodes': len(path_nodes),
                    'additional_nodes': len(additional_entity_titles),
                    'total_nodes': len(final_nodes)
                },
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats
            }
        }
        
        # Save artifacts
        self._save_pruned_artifacts(pruned_artifacts)
        
        logger.info("✅ POG Hybrid pipeline completed")
        return pruned_artifacts

    def apply_adaptive_multi_strategy_pipeline(
        self,
        target_reduction: float = 0.55,
        min_connectivity_pct: float = 0.90,
        protected_fraction: float = 0.15,
        hub_degree_percentile: float = 0.80
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply Adaptive Multi-Strategy pruning pipeline.
        
        This method combines signals from CrumbTrail, KGTrimmer, PathRAG/POG,
        and community structure to apply region-specific pruning strategies.
        
        Args:
            target_reduction: Target reduction percentage (0-1), default 0.55 (50-60% range)
            min_connectivity_pct: Minimum percentage of nodes in largest component
            protected_fraction: Fraction of top-scored nodes to always protect
            hub_degree_percentile: Degree percentile for hub classification
            
        Returns:
            Dictionary with pruned artifacts and metadata
        """
        from pruning.adaptive_multi_strategy import adaptive_multi_strategy_prune
        
        logger.info("🚀 Starting Adaptive Multi-Strategy pruning pipeline...")
        logger.info(f"  Target reduction: {target_reduction*100:.1f}%")
        logger.info(f"  Min connectivity: {min_connectivity_pct*100:.1f}%")
        logger.info(f"  Protected fraction: {protected_fraction*100:.1f}%")
        timestamp = datetime.now().isoformat()
        
        # Build graph
        G = self.scorer.graph
        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")
        
        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )
        
        # Run adaptive multi-strategy pruning
        logger.info("Running Adaptive Multi-Strategy algorithm...")
        pruned_graph = adaptive_multi_strategy_prune(
            G,
            self.entities_df,
            self.relationships_df,
            self.communities_df,
            target_reduction=target_reduction,
            min_connectivity_pct=min_connectivity_pct,
            protected_fraction=protected_fraction,
            hub_degree_percentile=hub_degree_percentile
        )
        
        # Extract pruned entities and relationships
        logger.info("Extracting pruned artifacts...")
        pruned_node_ids = set(pruned_graph.nodes())
        
        pruned_entities = self.entities_df[
            self.entities_df['title'].isin(pruned_node_ids)
        ].copy()
        
        pruned_relationships = self.relationships_df[
            self.relationships_df['source'].isin(pruned_node_ids) &
            self.relationships_df['target'].isin(pruned_node_ids)
        ].copy()
        
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
            'communities': self.communities_df.copy() if self.communities_df is not None else None,
            'metadata': {
                'timestamp': timestamp,
                'algorithm': 'AdaptiveMultiStrategy',
                'parameters': {
                    'target_reduction': target_reduction,
                    'min_connectivity_pct': min_connectivity_pct,
                    'protected_fraction': protected_fraction,
                    'hub_degree_percentile': hub_degree_percentile
                },
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats
            }
        }
        
        # Save artifacts
        self._save_pruned_artifacts(pruned_artifacts)
        
        logger.info("✅ Adaptive Multi-Strategy pipeline completed")
        return pruned_artifacts

    def apply_top_k_pipeline(
        self,
        k: float,
        target: str = 'nodes'
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply top-k pruning pipeline.

        Args:
            k: Percentage to keep (0-100)
            target: What to prune ('nodes' or 'edges')

        Returns:
            Dictionary with pruned artifacts and metadata
        """
        logger.info(f"🚀 Starting top-k pruning pipeline (k={k}%, target={target})...")
        timestamp = datetime.now().isoformat()

        # Build graph
        G = self.scorer.graph
        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")

        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )

        # Score components
        self.score_components()

        # Prune based on target
        if target == 'nodes':
            pruned_entities = self.prune_nodes(strategy='top_k', k=k)
            # Filter relationships to only those with both endpoints in pruned entities
            pruned_node_ids = set(pruned_entities['title'])
            pruned_relationships = self.relationships_df[
                self.relationships_df['source'].isin(pruned_node_ids) &
                self.relationships_df['target'].isin(pruned_node_ids)
            ].copy()
        else:  # edges
            pruned_relationships = self.prune_edges(strategy='top_k', k=k)
            # Filter entities to only those that appear in pruned relationships
            pruned_node_ids = set(pruned_relationships['source']) | set(pruned_relationships['target'])
            pruned_entities = self.entities_df[
                self.entities_df['title'].isin(pruned_node_ids)
            ].copy()

        # Rebuild graph for stats
        from pruning.scoring_utils import GraphScorer
        temp_scorer = GraphScorer(pruned_entities, pruned_relationships, None)
        pruned_G = temp_scorer.graph
        pruned_stats = self._compute_detailed_stats(
            pruned_entities, pruned_relationships, pruned_G
        )

        self._log_reduction_stats(baseline_stats, pruned_stats)

        pruned_artifacts = {
            'entities': pruned_entities,
            'relationships': pruned_relationships,
            'communities': self.communities_df.copy() if self.communities_df is not None else None,
            'metadata': {
                'timestamp': timestamp,
                'algorithm': f'top_k_{target}',
                'parameters': {'k': k, 'target': target},
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats
            }
        }

        self._save_pruned_artifacts(pruned_artifacts)
        logger.info("✅ Top-k pipeline completed")
        return pruned_artifacts

    def apply_threshold_pipeline(
        self,
        threshold: float,
        target: str = 'nodes'
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply threshold pruning pipeline.

        Args:
            threshold: Minimum score to keep
            target: What to prune ('nodes' or 'edges')

        Returns:
            Dictionary with pruned artifacts and metadata
        """
        logger.info(f"🚀 Starting threshold pruning pipeline (threshold={threshold}, target={target})...")
        timestamp = datetime.now().isoformat()

        # Build graph
        G = self.scorer.graph
        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")

        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )

        # Score components
        self.score_components()

        # Prune based on target
        if target == 'nodes':
            pruned_entities = self.prune_nodes(strategy='threshold', threshold=threshold)
            pruned_node_ids = set(pruned_entities['title'])
            pruned_relationships = self.relationships_df[
                self.relationships_df['source'].isin(pruned_node_ids) &
                self.relationships_df['target'].isin(pruned_node_ids)
            ].copy()
        else:  # edges
            pruned_relationships = self.prune_edges(strategy='threshold', threshold=threshold)
            pruned_node_ids = set(pruned_relationships['source']) | set(pruned_relationships['target'])
            pruned_entities = self.entities_df[
                self.entities_df['title'].isin(pruned_node_ids)
            ].copy()

        # Rebuild graph for stats
        from pruning.scoring_utils import GraphScorer
        temp_scorer = GraphScorer(pruned_entities, pruned_relationships, None)
        pruned_G = temp_scorer.graph
        pruned_stats = self._compute_detailed_stats(
            pruned_entities, pruned_relationships, pruned_G
        )

        self._log_reduction_stats(baseline_stats, pruned_stats)

        pruned_artifacts = {
            'entities': pruned_entities,
            'relationships': pruned_relationships,
            'communities': self.communities_df.copy() if self.communities_df is not None else None,
            'metadata': {
                'timestamp': timestamp,
                'algorithm': f'threshold_{target}',
                'parameters': {'threshold': threshold, 'target': target},
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats
            }
        }

        self._save_pruned_artifacts(pruned_artifacts)
        logger.info("✅ Threshold pipeline completed")
        return pruned_artifacts

    def apply_edges_top_k_pipeline(
        self,
        k: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply top-k edges per node pruning pipeline.

        Args:
            k: Number of top edges to keep per node

        Returns:
            Dictionary with pruned artifacts and metadata
        """
        logger.info(f"🚀 Starting top-k edges per node pruning pipeline (k={k})...")
        timestamp = datetime.now().isoformat()

        # Build graph
        G = self.scorer.graph
        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")

        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )

        # Score components
        self.score_components()

        # Prune edges
        pruned_relationships = self.prune_edges(strategy='top_k_per_node', k=k)
        
        # Filter entities to only those that appear in pruned relationships
        pruned_node_ids = set(pruned_relationships['source']) | set(pruned_relationships['target'])
        pruned_entities = self.entities_df[
            self.entities_df['title'].isin(pruned_node_ids)
        ].copy()

        # Rebuild graph for stats
        from pruning.scoring_utils import GraphScorer
        temp_scorer = GraphScorer(pruned_entities, pruned_relationships, None)
        pruned_G = temp_scorer.graph
        pruned_stats = self._compute_detailed_stats(
            pruned_entities, pruned_relationships, pruned_G
        )

        self._log_reduction_stats(baseline_stats, pruned_stats)

        pruned_artifacts = {
            'entities': pruned_entities,
            'relationships': pruned_relationships,
            'communities': self.communities_df.copy() if self.communities_df is not None else None,
            'metadata': {
                'timestamp': timestamp,
                'algorithm': 'edges_top_k',
                'parameters': {'k': k},
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats
            }
        }

        self._save_pruned_artifacts(pruned_artifacts)
        logger.info("✅ Top-k edges per node pipeline completed")
        return pruned_artifacts

    def apply_combined_pipeline(
        self,
        node_k: float,
        edge_k: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply combined node and edge pruning pipeline.

        Args:
            node_k: Percentage of nodes to keep (0-100)
            edge_k: Number of top edges to keep per node

        Returns:
            Dictionary with pruned artifacts and metadata
        """
        logger.info(f"🚀 Starting combined pruning pipeline (node_k={node_k}%, edge_k={edge_k})...")
        timestamp = datetime.now().isoformat()

        # Build graph
        G = self.scorer.graph
        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")

        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )

        # Score components
        self.score_components()

        # First prune nodes
        pruned_entities = self.prune_nodes(strategy='top_k', k=node_k)
        pruned_node_ids = set(pruned_entities['title'])
        
        # Filter relationships to pruned nodes
        filtered_relationships = self.relationships_df[
            self.relationships_df['source'].isin(pruned_node_ids) &
            self.relationships_df['target'].isin(pruned_node_ids)
        ].copy()

        # Temporarily update relationships for edge pruning
        original_relationships = self.relationships_df
        self.relationships_df = filtered_relationships
        self.score_components()  # Re-score with filtered relationships

        # Then prune edges
        pruned_relationships = self.prune_edges(strategy='top_k_per_node', k=edge_k)
        
        # Restore original relationships
        self.relationships_df = original_relationships

        # Final entity filtering
        final_node_ids = set(pruned_relationships['source']) | set(pruned_relationships['target'])
        final_entities = pruned_entities[pruned_entities['title'].isin(final_node_ids)].copy()

        # Rebuild graph for stats
        from pruning.scoring_utils import GraphScorer
        temp_scorer = GraphScorer(final_entities, pruned_relationships, None)
        pruned_G = temp_scorer.graph
        pruned_stats = self._compute_detailed_stats(
            final_entities, pruned_relationships, pruned_G
        )

        self._log_reduction_stats(baseline_stats, pruned_stats)

        pruned_artifacts = {
            'entities': final_entities,
            'relationships': pruned_relationships,
            'communities': self.communities_df.copy() if self.communities_df is not None else None,
            'metadata': {
                'timestamp': timestamp,
                'algorithm': 'combined',
                'parameters': {'node_k': node_k, 'edge_k': edge_k},
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats
            }
        }

        self._save_pruned_artifacts(pruned_artifacts)
        logger.info("✅ Combined pipeline completed")
        return pruned_artifacts

    def apply_kgtrimmer_aggressive_pipeline(self) -> Dict[str, pd.DataFrame]:
        """Apply KGTrimmer with aggressive settings (10% keep, holistic_weight=0.7)."""
        return self.apply_kgtrimmer_pipeline(
            collective_weight=0.3,
            holistic_weight=0.7,
            min_importance_percentile=0.1,
            preserve_connectivity=False
        )

    def apply_kgtrimmer_conservative_pipeline(self) -> Dict[str, pd.DataFrame]:
        """Apply KGTrimmer with conservative settings (30% keep, collective_weight=0.6)."""
        return self.apply_kgtrimmer_pipeline(
            collective_weight=0.6,
            holistic_weight=0.4,
            min_importance_percentile=0.3,
            preserve_connectivity=True
        )

    def apply_pog_aggressive_pipeline(self) -> Dict[str, pd.DataFrame]:
        """Apply POG with aggressive settings (30 seeds, 50 paths, max_length=4)."""
        return self.apply_pog_pipeline(
            num_seeds=30,
            top_k_paths=50,
            max_path_length=4,
            semantic_threshold=0.7
        )

    def apply_pathrag_aggressive_pipeline(self) -> Dict[str, pd.DataFrame]:
        """Apply PathRAG with aggressive settings (alpha=0.7, theta=0.1, top_n=30, top_k=10)."""
        return self.apply_pathrag_pipeline(
            alpha=0.7,
            theta=0.1,
            top_n_nodes=30,
            top_k_paths=10,
            max_path_length=4
        )

    def apply_crumbtrail_conservative_pipeline(self) -> Dict[str, pd.DataFrame]:
        """Apply CrumbTrail with conservative settings (30% protected)."""
        return self.apply_crumbtrail_pipeline(
            protected_fraction=0.3,
            protected_selection='degree_centrality',
            max_iterations=1000
        )

    def apply_adaptive_hybrid_pipeline(
        self,
        target_reduction: float,
        min_accuracy: float = 0.7
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply adaptive hybrid pruning pipeline.

        This method analyzes graph characteristics and selects the optimal
        combination of pruning methods to achieve target reduction while
        preserving accuracy.

        Args:
            target_reduction: Desired graph size reduction (0.0-1.0)
            min_accuracy: Minimum acceptable accuracy threshold (0.0-1.0)

        Returns:
            Dictionary with pruned artifacts and metadata
        """
        from pruning.adaptive_hybrid import AdaptiveHybridPruner
        
        logger.info("🚀 Starting Adaptive Hybrid pruning pipeline...")
        logger.info(f"  Target reduction: {target_reduction*100:.1f}%")
        logger.info(f"  Min accuracy: {min_accuracy*100:.1f}%")
        
        timestamp = datetime.now().isoformat()
        
        # Build graph
        G = self.scorer.graph
        if G is None or len(G.nodes()) == 0:
            raise ValueError("Graph is empty. Check that entities and relationships are loaded.")
        
        # Compute baseline stats
        baseline_stats = self._compute_detailed_stats(
            self.entities_df, self.relationships_df, G
        )
        
        # Create adaptive pruner
        adaptive_pruner = AdaptiveHybridPruner(
            G,
            self.entities_df,
            self.relationships_df,
            self.communities_df
        )
        
        # Select strategy
        strategy = adaptive_pruner.select_strategy(target_reduction, min_accuracy)
        
        # Apply strategy
        pruned_graph = adaptive_pruner.apply_strategy(strategy, self)
        
        # Extract pruned artifacts
        logger.info("Extracting pruned artifacts...")
        pruned_node_ids = set(pruned_graph.nodes())
        pruned_entities = self.entities_df[
            self.entities_df['title'].isin(pruned_node_ids)
        ].copy()
        pruned_relationships = self.relationships_df[
            self.relationships_df['source'].isin(pruned_node_ids) &
            self.relationships_df['target'].isin(pruned_node_ids)
        ].copy()
        
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
            'communities': self.communities_df.copy() if self.communities_df is not None else None,
            'metadata': {
                'timestamp': timestamp,
                'algorithm': 'adaptive_hybrid',
                'parameters': {
                    'target_reduction': target_reduction,
                    'min_accuracy': min_accuracy,
                    'strategy': strategy
                },
                'baseline_stats': baseline_stats,
                'pruned_stats': pruned_stats
            }
        }
        
        self._save_pruned_artifacts(pruned_artifacts)
        logger.info("✅ Adaptive Hybrid pipeline completed")
        return pruned_artifacts

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
