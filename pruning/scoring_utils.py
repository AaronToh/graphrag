#!/usr/bin/env python3
"""
GraphRAG Pruning Lab - Scoring Utilities (High-Level Framework)

This module provides the framework for scoring nodes, edges, and communities.
You can implement your own scoring algorithms here.

Framework Structure:
- GraphScorer: Main class for coordinating scoring
- Individual scoring methods: degree, frequency, semantic, etc.
- Combined scoring: Weighted combination of multiple metrics
- Data loading: Load GraphRAG artifacts from parquet files
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import networkx as nx
from tqdm import tqdm

logger = logging.getLogger(__name__)

class GraphScorer:
    """
    High-level framework for scoring graph components.

    This class provides the structure for implementing your own scoring algorithms.
    """

    def __init__(self, entities_df: pd.DataFrame, relationships_df: pd.DataFrame,
                 communities_df: pd.DataFrame = None):
        """Initialize with GraphRAG data."""
        self.entities_df = entities_df.copy()
        self.relationships_df = relationships_df.copy()
        self.communities_df = communities_df.copy() if communities_df is not None else None

        # Build graph structure
        self.graph = self._build_graph()

        logger.info(f"Loaded {len(self.entities_df)} entities, {len(self.relationships_df)} relationships")

    def _build_graph(self) -> nx.DiGraph:
        """Build NetworkX DiGraph from GraphRAG entities and relationships."""
        logger.info("Building directed graph from entities and relationships...")

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes from entities
        # Use 'title' as node ID since relationships use entity titles
        for _, entity in self.entities_df.iterrows():
            node_id = entity['title']  # Use title, not UUID
            # Store all entity attributes including the UUID
            G.add_node(node_id, **entity.to_dict())

        # Add edges from relationships
        # Relationships use entity titles as source/target
        for _, rel in self.relationships_df.iterrows():
            source = rel['source']
            target = rel['target']
            # Add edge with all relationship attributes
            G.add_edge(source, target, **rel.to_dict())

        logger.info(f"Built graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        return G

    # === NODE SCORING METHODS ===

    def score_nodes_degree_centrality(self) -> pd.Series:
        """Score nodes by degree centrality."""
        centrality = nx.degree_centrality(self.graph)
        # Convert to Series indexed by node title
        scores = pd.Series(centrality, name='degree_centrality')
        return scores

    def score_nodes_frequency(self) -> pd.Series:
        """Score nodes by frequency/mention count."""
        if 'frequency' in self.entities_df.columns:
            # Use existing frequency column
            scores = pd.Series(
                self.entities_df.set_index('title')['frequency'].to_dict(),
                name='frequency'
            )
        elif 'count' in self.entities_df.columns:
            # Use count column
            scores = pd.Series(
                self.entities_df.set_index('title')['count'].to_dict(),
                name='frequency'
            )
        else:
            # Calculate frequency from relationships
            node_counts = {}
            for _, rel in self.relationships_df.iterrows():
                node_counts[rel['source']] = node_counts.get(rel['source'], 0) + 1
                node_counts[rel['target']] = node_counts.get(rel['target'], 0) + 1
            scores = pd.Series(node_counts, name='frequency')
        
        # Normalize to 0-1 range
        if len(scores) > 0 and scores.max() > 0:
            scores = scores / scores.max()
        return scores

    def score_nodes_semantic_relevance(self, query: str = None) -> pd.Series:
        """Score nodes by semantic relevance."""
        # TODO: Implement semantic scoring
        # Hint: Use embeddings and cosine similarity
        pass

    def score_nodes_custom_method(self) -> pd.Series:
        """Your custom node scoring method."""
        # TODO: Implement your own scoring logic
        pass

    # === EDGE SCORING METHODS ===

    def score_edges_weight(self) -> pd.Series:
        """Score edges by weight."""
        edge_weights = {}
        for _, rel in self.relationships_df.iterrows():
            source = rel['source']
            target = rel['target']
            edge_key = (source, target)
            
            # Try different weight column names
            if 'weight' in rel:
                edge_weights[edge_key] = float(rel['weight'])
            elif 'score' in rel:
                edge_weights[edge_key] = float(rel['score'])
            else:
                # Default weight of 1.0 if no weight column
                edge_weights[edge_key] = 1.0
        
        scores = pd.Series(edge_weights, name='weight')
        # Normalize to 0-1 range
        if len(scores) > 0 and scores.max() > 0:
            scores = scores / scores.max()
        return scores

    def score_edges_plausibility(self) -> pd.Series:
        """Score edges by relationship plausibility."""
        # TODO: Implement plausibility scoring
        # Hint: Use KGE models or domain knowledge
        pass

    def score_edges_custom_method(self) -> pd.Series:
        """Your custom edge scoring method."""
        # TODO: Implement your own edge scoring logic
        pass

    # === COMMUNITY SCORING METHODS ===

    def score_communities_size(self) -> pd.Series:
        """Score communities by size."""
        # TODO: Implement community size scoring
        pass

    def score_communities_density(self) -> pd.Series:
        """Score communities by density."""
        # TODO: Implement community density scoring
        pass

    def score_communities_custom_method(self) -> pd.Series:
        """Your custom community scoring method."""
        # TODO: Implement your own community scoring logic
        pass

    # === COMBINED SCORING ===

    def get_combined_node_scores(self, weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Combine multiple node scoring methods.

        Args:
            weights: Dictionary mapping scoring method names to weights
                    e.g., {'degree': 0.4, 'frequency': 0.3, 'semantic': 0.3}

        Returns:
            DataFrame with individual scores, combined score, and ranking
        """
        # TODO: Implement weighted combination of node scores
        pass

    def get_combined_edge_scores(self, weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Combine multiple edge scoring methods.

        Args:
            weights: Dictionary mapping scoring method names to weights

        Returns:
            DataFrame with individual scores, combined score, and ranking
        """
        # TODO: Implement weighted combination of edge scores
        pass

    def get_combined_community_scores(self, weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Combine multiple community scoring methods.

        Args:
            weights: Dictionary mapping scoring method names to weights

        Returns:
            DataFrame with individual scores, combined score, and ranking
        """
        # TODO: Implement weighted combination of community scores
        pass

    # === FLOW PROPAGATION FOR PATHRAG ===

    def score_nodes_flow_propagation(
        self,
        seed_nodes: Optional[List[str]] = None,
        alpha: float = 0.8,
        theta: float = 0.05,
        top_n_seeds: int = 40,
        seed_method: str = 'degree_centrality',
        max_iterations: int = 100
    ) -> Tuple[pd.Series, Dict[Tuple[str, str], float]]:
        """
        Propagate flow from seed nodes through the graph.
        
        Args:
            seed_nodes: List of seed node IDs (if None, auto-select)
            alpha: Flow decay factor (0-1)
            theta: Convergence threshold
            top_n_seeds: Number of seed nodes if auto-selecting
            seed_method: Method to select seeds ('degree_centrality', 'betweenness', 'pagerank')
            max_iterations: Maximum iterations for convergence
            
        Returns:
            Tuple of (node_flows Series, edge_flows dict)
        """
        G = self.graph
        
        # Select seed nodes if not provided
        if seed_nodes is None:
            if seed_method == 'degree_centrality':
                centrality = nx.degree_centrality(G)
            elif seed_method == 'betweenness':
                centrality = nx.betweenness_centrality(G, k=min(500, len(G.nodes())))
            elif seed_method == 'pagerank':
                centrality = nx.pagerank(G)
            else:
                centrality = nx.degree_centrality(G)
            
            top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n_seeds]
            seed_nodes = [node for node, _ in top_nodes]
        
        # Initialize flow values
        node_flows = {node: 0.0 for node in G.nodes()}
        for seed in seed_nodes:
            if seed in G:
                node_flows[seed] = 1.0
        
        # Iterative flow propagation
        logger.info(f"  Propagating flow from {len(seed_nodes)} seeds (max {max_iterations} iterations)...")
        for iteration in tqdm(range(max_iterations), desc="  Flow propagation", leave=False):
            new_flows = {node: 0.0 for node in G.nodes()}
            
            # Initialize seeds
            for seed in seed_nodes:
                if seed in G:
                    new_flows[seed] = 1.0
            
            # Propagate flow
            for node in G.nodes():
                if node in seed_nodes:
                    continue
                
                # Collect flow from predecessors
                total_inflow = 0.0
                for pred in G.predecessors(node):
                    if pred in node_flows:
                        # Flow decays by alpha
                        total_inflow += node_flows[pred] * alpha / G.out_degree(pred) if G.out_degree(pred) > 0 else 0
                
                new_flows[node] = total_inflow
            
            # Check convergence
            max_change = max(abs(new_flows[node] - node_flows[node]) for node in G.nodes())
            node_flows = new_flows
            
            if max_change < theta:
                logger.info(f"  ✓ Flow propagation converged after {iteration + 1} iterations")
                break
        
        # Compute edge flows
        logger.info("  Computing edge flows...")
        edge_flows = {}
        for u, v in tqdm(G.edges(), desc="  Computing edges", leave=False):
            if u in node_flows and G.out_degree(u) > 0:
                edge_flows[(u, v)] = node_flows[u] * alpha / G.out_degree(u)
            else:
                edge_flows[(u, v)] = 0.0
        
        nodes_with_flow = len([n for n, f in node_flows.items() if f > 0])
        logger.info(f"  ✓ Flow computation complete: {nodes_with_flow} nodes, {len(edge_flows)} edges")
        node_flows_series = pd.Series(node_flows, name='flow')
        return node_flows_series, edge_flows

    def score_edges_flow(self, edge_flows: Dict[Tuple[str, str], float]) -> pd.Series:
        """
        Convert edge flows dictionary to Series.
        
        Args:
            edge_flows: Dictionary mapping (source, target) to flow value
            
        Returns:
            Series of edge flows
        """
        return pd.Series(edge_flows, name='edge_flow')


def load_graphrag_artifacts(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load GraphRAG artifacts from parquet files.

    Args:
        output_dir: Directory containing GraphRAG output files

    Returns:
        Tuple of (entities_df, relationships_df, communities_df)
    """
    entities_path = output_dir / "entities.parquet"
    relationships_path = output_dir / "relationships.parquet"
    communities_path = output_dir / "communities.parquet"

    entities_df = pd.read_parquet(entities_path) if entities_path.exists() else pd.DataFrame()
    relationships_df = pd.read_parquet(relationships_path) if relationships_path.exists() else pd.DataFrame()
    communities_df = pd.read_parquet(communities_path) if communities_path.exists() else pd.DataFrame()

    return entities_df, relationships_df, communities_df


def save_scores(scores_df: pd.DataFrame, output_path: Path, name: str):
    """
    Save scoring results to file.

    Args:
        scores_df: DataFrame with scoring results
        output_path: Directory to save results
        name: Name prefix for output files
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as both CSV and Parquet
    csv_path = output_path / f"{name}.csv"
    parquet_path = output_path / f"{name}.parquet"
    
    if isinstance(scores_df, pd.Series):
        scores_df.to_csv(csv_path)
        scores_df.to_frame().to_parquet(parquet_path)
    else:
        scores_df.to_csv(csv_path, index=False)
        scores_df.to_parquet(parquet_path, index=False)
    
    logger.info(f"Saved scores to {csv_path} and {parquet_path}")


if __name__ == "__main__":
    # Example usage framework
    import sys

    # TODO: Load your data
    output_dir = Path("../../workspace/output")
    # entities_df, relationships_df, communities_df = load_graphrag_artifacts(output_dir)

    # TODO: Initialize scorer
    # scorer = GraphScorer(entities_df, relationships_df, communities_df)

    # TODO: Calculate your scores
    # node_scores = scorer.get_combined_node_scores()
    # edge_scores = scorer.get_combined_edge_scores()

    # TODO: Save results
    # save_scores(node_scores, output_dir, "node_scores")
    # save_scores(edge_scores, output_dir, "edge_scores")

    print("Scoring framework ready - implement your algorithms above!")
