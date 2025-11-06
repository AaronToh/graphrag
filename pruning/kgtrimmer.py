#!/usr/bin/env python3
"""
KGTrimmer Algorithm Implementation

KGTrimmer prunes knowledge graphs by evaluating node importance from both
collective (community-based) and holistic (global) perspectives.

Algorithm Overview:
1. Collective View: Score nodes based on community consensus
2. Holistic View: Score nodes using global importance metrics
3. Combined Scoring: Weighted combination of both views
4. Iterative Pruning: Remove low-scoring nodes while preserving connectivity
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Set, Optional, List
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class KGTrimmerPruner:
    """
    Implementation of KGTrimmer algorithm for knowledge graph pruning.
    """

    def __init__(self, graph: nx.DiGraph, entities_df: pd.DataFrame,
                 communities_df: Optional[pd.DataFrame] = None):
        """
        Initialize KGTrimmer pruner.

        Args:
            graph: Input directed graph (NetworkX DiGraph)
            entities_df: Entities DataFrame with node information
            communities_df: Optional communities DataFrame
        """
        self.graph = graph.copy()  # Work on a copy
        self.entities_df = entities_df
        self.communities_df = communities_df

        logger.info(f"Initialized KGTrimmer: {len(graph.nodes())} nodes, "
                   f"{len(graph.edges())} edges")

    def _score_collective_importance(self) -> Dict[str, float]:
        """
        Score nodes based on community consensus (collective view).

        Returns:
            Dictionary mapping node IDs to collective importance scores
        """
        logger.info("  Computing collective importance scores...")
        collective_scores = {}
        
        if self.communities_df is not None and len(self.communities_df) > 0:
            # Use community information if available
            logger.info("    Using community information for collective scoring...")
            community_sizes = {}
            node_communities = {}
            
            # Map nodes to communities
            logger.info(f"    Mapping {len(self.communities_df)} communities to nodes...")
            for _, comm in tqdm(self.communities_df.iterrows(), total=len(self.communities_df), desc="    Processing communities", leave=False):
                comm_id = comm.get('id', comm.get('community_id', None))
                if comm_id is None:
                    continue
                
                # Get entities in this community
                if 'entity_ids' in comm:
                    entity_ids = comm['entity_ids']
                    if isinstance(entity_ids, list):
                        for entity_id in entity_ids:
                            # Find entity title
                            entity = self.entities_df[self.entities_df['id'] == entity_id]
                            if len(entity) > 0:
                                node_title = entity.iloc[0]['title']
                                node_communities[node_title] = comm_id
                                community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
            
            logger.info(f"    Scoring {len(self.graph.nodes())} nodes based on communities...")
            max_degree = max(self.graph.degree(), key=lambda x: x[1])[1] if self.graph.nodes() else 1
            max_comm_size = max(community_sizes.values()) if community_sizes else 1
            
            for node in tqdm(self.graph.nodes(), desc="    Scoring nodes", leave=False):
                if node in node_communities:
                    comm_id = node_communities[node]
                    comm_size = community_sizes.get(comm_id, 1)
                    node_degree = self.graph.degree(node)
                    collective_scores[node] = (comm_size / max_comm_size) * (node_degree / max_degree)
                else:
                    collective_scores[node] = 0.0
        else:
            # Fallback: use local neighborhood consensus
            logger.info("    Using neighborhood consensus (no communities available)...")
            max_degree = max(self.graph.degree(), key=lambda x: x[1])[1] if self.graph.nodes() else 1
            
            for node in tqdm(self.graph.nodes(), desc="    Scoring nodes", leave=False):
                neighbors = list(self.graph.neighbors(node))
                if len(neighbors) == 0:
                    collective_scores[node] = 0.0
                else:
                    # Average degree of neighbors (consensus signal)
                    neighbor_degrees = [self.graph.degree(n) for n in neighbors]
                    avg_neighbor_degree = np.mean(neighbor_degrees) if neighbor_degrees else 0
                    collective_scores[node] = min(1.0, avg_neighbor_degree / max_degree)
        
        # Normalize scores
        if collective_scores:
            max_score = max(collective_scores.values())
            if max_score > 0:
                collective_scores = {k: v / max_score for k, v in collective_scores.items()}
        
        logger.info(f"  ✓ Collective scoring complete: {len(collective_scores)} nodes scored")
        return collective_scores

    def _score_holistic_importance(self) -> Dict[str, float]:
        """
        Score nodes based on global importance metrics (holistic view).

        Returns:
            Dictionary mapping node IDs to holistic importance scores
        """
        logger.info("  Computing holistic importance scores...")
        holistic_scores = {}
        
        # Compute multiple global metrics
        logger.info("    Computing degree centrality...")
        degree_centrality = nx.degree_centrality(self.graph)
        logger.info("    ✓ Degree centrality complete")
        
        # Try to compute betweenness (may be expensive for large graphs)
        # Skip for very large graphs to save time
        graph_size = len(self.graph.nodes())
        if graph_size > 10000:
            logger.info("    Skipping betweenness centrality (graph too large, using zeros)")
            betweenness = {node: 0.0 for node in self.graph.nodes()}
        else:
            logger.info("    Computing betweenness centrality (this may take a while)...")
            try:
                betweenness = nx.betweenness_centrality(self.graph, k=min(500, graph_size))
                logger.info("    ✓ Betweenness centrality complete")
            except Exception as e:
                logger.warning(f"    ⚠ Betweenness centrality failed: {e}, using zeros")
                betweenness = {node: 0.0 for node in self.graph.nodes()}
        
        # PageRank as another global importance measure
        logger.info("    Computing PageRank...")
        try:
            # Use fewer iterations for large graphs
            max_iter = 50 if graph_size > 10000 else 100
            pagerank = nx.pagerank(self.graph, max_iter=max_iter)
            logger.info("    ✓ PageRank complete")
        except Exception as e:
            logger.warning(f"    ⚠ PageRank failed: {e}, using zeros")
            pagerank = {node: 0.0 for node in self.graph.nodes()}
        
        # Frequency from entities if available
        logger.info("    Extracting frequency scores from entities...")
        frequency_scores = {}
        if 'frequency' in self.entities_df.columns:
            for _, entity in tqdm(self.entities_df.iterrows(), total=len(self.entities_df), desc="    Processing entities", leave=False):
                node_title = entity['title']
                frequency_scores[node_title] = float(entity.get('frequency', 0))
        elif 'count' in self.entities_df.columns:
            for _, entity in tqdm(self.entities_df.iterrows(), total=len(self.entities_df), desc="    Processing entities", leave=False):
                node_title = entity['title']
                frequency_scores[node_title] = float(entity.get('count', 0))
        
        # Normalize frequency scores
        if frequency_scores:
            max_freq = max(frequency_scores.values())
            if max_freq > 0:
                frequency_scores = {k: v / max_freq for k, v in frequency_scores.items()}
        
        # Combine metrics
        logger.info("    Combining metrics...")
        for node in tqdm(self.graph.nodes(), desc="    Combining scores", leave=False):
            dc = degree_centrality.get(node, 0.0)
            bc = betweenness.get(node, 0.0)
            pr = pagerank.get(node, 0.0)
            freq = frequency_scores.get(node, 0.0)
            
            # Weighted combination
            holistic_scores[node] = 0.4 * dc + 0.3 * bc + 0.2 * pr + 0.1 * freq
        
        # Normalize
        if holistic_scores:
            max_score = max(holistic_scores.values())
            if max_score > 0:
                holistic_scores = {k: v / max_score for k, v in holistic_scores.items()}
        
        logger.info(f"  ✓ Holistic scoring complete: {len(holistic_scores)} nodes scored")
        return holistic_scores

    def _compute_combined_scores(
        self,
        collective_weight: float = 0.5,
        holistic_weight: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute combined scores from collective and holistic views.

        Args:
            collective_weight: Weight for collective importance
            holistic_weight: Weight for holistic importance

        Returns:
            Dictionary mapping node IDs to combined scores
        """
        collective_scores = self._score_collective_importance()
        holistic_scores = self._score_holistic_importance()
        
        # Normalize weights
        total_weight = collective_weight + holistic_weight
        if total_weight > 0:
            collective_weight /= total_weight
            holistic_weight /= total_weight
        
        combined_scores = {}
        all_nodes = set(collective_scores.keys()) | set(holistic_scores.keys())
        
        for node in all_nodes:
            coll = collective_scores.get(node, 0.0)
            hol = holistic_scores.get(node, 0.0)
            combined_scores[node] = collective_weight * coll + holistic_weight * hol
        
        return combined_scores

    def prune(
        self,
        collective_weight: float = 0.5,
        holistic_weight: float = 0.5,
        min_importance_percentile: float = 0.2,
        preserve_connectivity: bool = True,
        max_iterations: int = 10
    ) -> nx.DiGraph:
        """
        Prune graph by removing low-importance nodes.

        Args:
            collective_weight: Weight for collective importance
            holistic_weight: Weight for holistic importance
            min_importance_percentile: Keep top N% of nodes (0.0-1.0)
            preserve_connectivity: Whether to maintain graph connectivity
            max_iterations: Maximum pruning iterations

        Returns:
            Pruned directed graph
        """
        logger.info("Starting KGTrimmer pruning...")
        logger.info(f"  Collective weight: {collective_weight}")
        logger.info(f"  Holistic weight: {holistic_weight}")
        logger.info(f"  Min importance percentile: {min_importance_percentile}")
        logger.info(f"  Preserve connectivity: {preserve_connectivity}")
        
        pruned_graph = self.graph.copy()
        original_size = len(pruned_graph.nodes())
        target_size = int(original_size * min_importance_percentile)
        
        logger.info(f"  Target: Keep {target_size} nodes ({min_importance_percentile*100:.0f}% of {original_size})")
        
        for iteration in tqdm(range(max_iterations), desc="  Pruning iterations"):
            logger.info(f"\n  Iteration {iteration + 1}/{max_iterations}:")
            logger.info(f"    Current graph size: {len(pruned_graph.nodes())} nodes, {len(pruned_graph.edges())} edges")
            
            # Compute combined scores
            logger.info("    Computing combined scores...")
            combined_scores = self._compute_combined_scores(
                collective_weight, holistic_weight
            )
            
            if not combined_scores:
                logger.warning("    No scores computed, stopping")
                break
            
            # Determine threshold
            scores_list = list(combined_scores.values())
            threshold = np.percentile(scores_list, (1 - min_importance_percentile) * 100)
            logger.info(f"    Score threshold: {threshold:.4f}")
            
            # Identify nodes to remove
            logger.info("    Identifying nodes to remove...")
            nodes_to_remove = [
                node for node, score in combined_scores.items()
                if score < threshold and node in pruned_graph
            ]
            logger.info(f"    Found {len(nodes_to_remove)} candidate nodes to remove")
            
            if not nodes_to_remove:
                logger.info(f"    No more nodes to remove after iteration {iteration + 1}")
                break
            
            # If preserving connectivity, only remove nodes that don't disconnect the graph
            if preserve_connectivity:
                logger.info("    Checking connectivity preservation...")
                # Optimized approach: Remove nodes in batches and check connectivity
                # This is much faster than checking each node individually
                
                # Get largest component first
                if pruned_graph.is_directed():
                    components = list(nx.weakly_connected_components(pruned_graph))
                else:
                    components = list(nx.connected_components(pruned_graph))
                
                if components:
                    largest_component = max(components, key=len)
                    largest_size = len(largest_component)
                    logger.info(f"    Largest component: {largest_size} nodes")
                    
                    # Filter: only consider nodes that are not critical bridges
                    # Simple heuristic: nodes with degree 1 are safe (leaf nodes)
                    # For others, we'll do a batch check
                    safe_to_remove = []
                    nodes_to_check = []
                    
                    for node in nodes_to_remove:
                        if pruned_graph.degree(node) <= 1:
                            # Leaf nodes are generally safe to remove
                            safe_to_remove.append(node)
                        else:
                            nodes_to_check.append(node)
                    
                    logger.info(f"    Quick filter: {len(safe_to_remove)} leaf nodes safe to remove")
                    logger.info(f"    Need to check: {len(nodes_to_check)} nodes")
                    
                    # For remaining nodes, check in batches (more efficient)
                    if nodes_to_check:
                        # Remove all at once and check if largest component still exists
                        test_graph = pruned_graph.copy()
                        test_graph.remove_nodes_from(nodes_to_check)
                        
                        if test_graph.is_directed():
                            test_components = list(nx.weakly_connected_components(test_graph))
                        else:
                            test_components = list(nx.connected_components(test_graph))
                        
                        if test_components:
                            new_largest = max(test_components, key=len)
                            # If largest component is still large enough, all are safe
                            if len(new_largest) >= largest_size * 0.9:
                                safe_to_remove.extend(nodes_to_check)
                                logger.info(f"    Batch check: All {len(nodes_to_check)} nodes safe to remove")
                            else:
                                # Need individual check (slower but more accurate)
                                logger.info(f"    Batch removal would break connectivity, checking individually...")
                                for node in tqdm(nodes_to_check, desc="      Checking nodes", leave=False):
                                    test_graph = pruned_graph.copy()
                                    test_graph.remove_node(node)
                                    if test_graph.is_directed():
                                        test_components = list(nx.weakly_connected_components(test_graph))
                                    else:
                                        test_components = list(nx.connected_components(test_graph))
                                    
                                    if test_components:
                                        new_largest = max(test_components, key=len)
                                        if len(new_largest) >= largest_size * 0.9:
                                            safe_to_remove.append(node)
                    
                    nodes_to_remove = safe_to_remove
                    logger.info(f"    Safe to remove: {len(nodes_to_remove)} nodes")
            
            # Remove nodes
            logger.info(f"    Removing {len(nodes_to_remove)} nodes...")
            pruned_graph.remove_nodes_from(nodes_to_remove)
            
            logger.info(f"    ✓ Iteration {iteration + 1} complete: {len(pruned_graph.nodes())} nodes remaining "
                       f"({100*len(pruned_graph.nodes())/original_size:.1f}% of original)")
            
            # Check if we've reached target size
            if len(pruned_graph.nodes()) <= target_size:
                logger.info(f"    ✓ Reached target size ({target_size} nodes)")
                break
        
        # Final cleanup: remove isolated nodes
        isolated = list(nx.isolates(pruned_graph))
        if isolated:
            pruned_graph.remove_nodes_from(isolated)
            logger.info(f"  Removed {len(isolated)} isolated nodes")
        
        reduction = 100 * (1 - len(pruned_graph.nodes()) / original_size)
        logger.info(f"✅ KGTrimmer complete: {len(pruned_graph.nodes())} nodes, "
                   f"{len(pruned_graph.edges())} edges ({reduction:.1f}% reduction)")
        
        return pruned_graph


def kgtrimmer_prune(
    graph: nx.DiGraph,
    entities_df: pd.DataFrame,
    communities_df: Optional[pd.DataFrame] = None,
    collective_weight: float = 0.5,
    holistic_weight: float = 0.5,
    min_importance_percentile: float = 0.2,
    preserve_connectivity: bool = True,
    max_iterations: int = 10
) -> nx.DiGraph:
    """
    Convenience function to run KGTrimmer pruning.

    Args:
        graph: Input directed graph
        entities_df: Entities DataFrame
        communities_df: Optional communities DataFrame
        collective_weight: Weight for collective importance
        holistic_weight: Weight for holistic importance
        min_importance_percentile: Keep top N% of nodes
        preserve_connectivity: Whether to maintain connectivity
        max_iterations: Maximum pruning iterations

    Returns:
        Pruned directed graph
    """
    pruner = KGTrimmerPruner(graph, entities_df, communities_df)
    return pruner.prune(
        collective_weight=collective_weight,
        holistic_weight=holistic_weight,
        min_importance_percentile=min_importance_percentile,
        preserve_connectivity=preserve_connectivity,
        max_iterations=max_iterations
    )

