#!/usr/bin/env python3
"""
Adaptive Multi-Strategy Pruning Algorithm

This module implements an adaptive pruning algorithm that analyzes graph regions
and applies different pruning strategies based on local graph characteristics.

The algorithm combines signals from:
- CrumbTrail (connectivity preservation)
- KGTrimmer (importance-based scoring)
- PathRAG/POG (path-based relevance)
- Community structure (bridge nodes)
- Semantic relevance

Algorithm Overview:
1. Analyze graph regions (dense core, sparse periphery, hubs, leaves, bridges)
2. Compute unified scores combining all signals
3. Select protected nodes (hubs, bridges, top-scored, path nodes)
4. Apply region-specific pruning strategies
5. Validate connectivity and adjust if needed
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Set, Optional, List, Tuple
from collections import defaultdict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AdaptiveMultiStrategyPruner:
    """
    Adaptive multi-strategy pruner that applies different pruning methods
    based on graph region characteristics.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        entities_df: pd.DataFrame,
        relationships_df: pd.DataFrame,
        communities_df: Optional[pd.DataFrame] = None
    ):
        """
        Initialize adaptive multi-strategy pruner.

        Args:
            graph: Input directed graph
            entities_df: Entities DataFrame
            relationships_df: Relationships DataFrame
            communities_df: Optional communities DataFrame
        """
        self.graph = graph.copy()
        self.entities_df = entities_df
        self.relationships_df = relationships_df
        self.communities_df = communities_df

        # Analysis results
        self.node_regions = {}
        self.unified_scores = {}
        self.protected_nodes = set()
        self.path_nodes = set()

        logger.info(f"Initialized AdaptiveMultiStrategyPruner: {len(graph.nodes())} nodes, "
                   f"{len(graph.edges())} edges")

    def analyze_graph_regions(self) -> Dict[str, str]:
        """
        Analyze graph structure and classify nodes into regions.

        Returns:
            Dictionary mapping node IDs to region types
        """
        logger.info("🔍 Analyzing graph regions...")
        node_regions = {}

        if len(self.graph.nodes()) == 0:
            return node_regions

        # Compute degree statistics
        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())
        if not degree_values:
            return node_regions

        # Calculate quartiles
        degree_75th = np.percentile(degree_values, 75)
        degree_25th = np.percentile(degree_values, 25)
        degree_median = np.median(degree_values)

        # Identify high-degree hubs
        hubs = {node for node, deg in degrees.items() if deg >= degree_75th}
        logger.info(f"  High-degree hubs: {len(hubs)} nodes (degree >= {degree_75th:.1f})")

        # Identify low-degree leaves
        leaves = {node for node, deg in degrees.items() if deg == 1}
        logger.info(f"  Low-degree leaves: {len(leaves)} nodes")

        # Compute clustering coefficient
        try:
            if self.graph.is_directed():
                G_undirected = self.graph.to_undirected()
            else:
                G_undirected = self.graph

            clustering = nx.clustering(G_undirected)
            clustering_values = [c for c in clustering.values() if c > 0]
            if clustering_values:
                clustering_median = np.median(clustering_values)
                clustering_75th = np.percentile(clustering_values, 75)
            else:
                clustering_median = 0.0
                clustering_75th = 0.0
        except Exception as e:
            logger.warning(f"  Could not compute clustering: {e}")
            clustering = {}
            clustering_median = 0.0
            clustering_75th = 0.0

        # Identify community bridges
        bridges = set()
        if self.communities_df is not None and len(self.communities_df) > 0:
            # Map nodes to communities
            node_to_communities = {}
            for _, comm in self.communities_df.iterrows():
                comm_id = comm.get('id', comm.get('community_id', None))
                if comm_id is None:
                    continue

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

            # Find bridge nodes
            for node in self.graph.nodes():
                if node not in node_to_communities:
                    continue
                node_comms = node_to_communities[node]
                if len(node_comms) > 1:
                    bridges.add(node)
                else:
                    neighbor_comms = set()
                    for neighbor in self.graph.neighbors(node):
                        if neighbor in node_to_communities:
                            neighbor_comms.update(node_to_communities[neighbor])
                    if len(neighbor_comms) > 1:
                        bridges.add(node)

        logger.info(f"  Community bridges: {len(bridges)} nodes")

        # Classify each node
        for node in self.graph.nodes():
            node_degree = degrees[node]
            node_clustering = clustering.get(node, 0.0)

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
                node_regions[node] = 'sparse_periphery'

        # Log distribution
        region_counts = {}
        for region in node_regions.values():
            region_counts[region] = region_counts.get(region, 0) + 1

        logger.info("  Region distribution:")
        for region, count in sorted(region_counts.items()):
            pct = 100 * count / len(node_regions) if node_regions else 0
            logger.info(f"    {region}: {count} nodes ({pct:.1f}%)")

        self.node_regions = node_regions
        logger.info(f"✅ Graph region analysis complete")
        return node_regions

    def _compute_connectivity_score(self) -> Dict[str, float]:
        """Compute connectivity score (CrumbTrail signal)."""
        logger.info("  Computing connectivity scores...")
        scores = {}
        
        # Use betweenness centrality as proxy for connectivity importance
        # For large graphs, use degree centrality as approximation
        if len(self.graph.nodes()) > 10000:
            centrality = nx.degree_centrality(self.graph)
        else:
            try:
                centrality = nx.betweenness_centrality(self.graph, k=min(500, len(self.graph.nodes())))
            except:
                centrality = nx.degree_centrality(self.graph)
        
        # Normalize
        max_score = max(centrality.values()) if centrality.values() else 1.0
        if max_score > 0:
            scores = {node: score / max_score for node, score in centrality.items()}
        else:
            scores = {node: 0.0 for node in self.graph.nodes()}
        
        return scores

    def _compute_importance_score(self) -> Dict[str, float]:
        """Compute importance score (KGTrimmer signal)."""
        logger.info("  Computing importance scores...")
        scores = {}
        
        # Combine degree centrality, PageRank, and frequency
        degree_cent = nx.degree_centrality(self.graph)
        
        try:
            pagerank = nx.pagerank(self.graph, max_iter=100)
        except:
            pagerank = {node: 0.0 for node in self.graph.nodes()}
        
        # Frequency from entities
        if 'frequency' in self.entities_df.columns:
            freq_map = dict(zip(self.entities_df['title'], self.entities_df['frequency']))
            max_freq = self.entities_df['frequency'].max() if len(self.entities_df) > 0 else 1.0
            freq_scores = {node: (freq_map.get(node, 0) / max_freq) if max_freq > 0 else 0.0 
                          for node in self.graph.nodes()}
        else:
            freq_scores = {node: 0.0 for node in self.graph.nodes()}
        
        # Combine (weighted average)
        for node in self.graph.nodes():
            scores[node] = (
                0.4 * degree_cent.get(node, 0.0) +
                0.4 * pagerank.get(node, 0.0) +
                0.2 * freq_scores.get(node, 0.0)
            )
        
        # Normalize
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {node: score / max_score for node, score in scores.items()}
        
        return scores

    def _compute_path_relevance_score(self) -> Dict[str, float]:
        """Compute path relevance score (PathRAG/POG signal)."""
        logger.info("  Computing path relevance scores...")
        scores = {node: 0.0 for node in self.graph.nodes()}
        
        # Simplified: use flow propagation from high-degree nodes
        # In full implementation, would use actual PathRAG flow
        try:
            # Select top 10% nodes by degree as seeds
            degrees = dict(self.graph.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            num_seeds = max(10, int(len(self.graph.nodes()) * 0.1))
            seed_nodes = [node for node, _ in sorted_nodes[:num_seeds]]
            
            # Simple flow: nodes reachable from seeds get higher scores
            for seed in seed_nodes:
                scores[seed] = 1.0
                # BFS to propagate score
                visited = {seed}
                queue = [(seed, 1.0)]
                
                for current, flow in queue:
                    for neighbor in self.graph.successors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_flow = flow * 0.8  # Decay
                            scores[neighbor] = max(scores[neighbor], new_flow)
                            if new_flow > 0.1:  # Threshold
                                queue.append((neighbor, new_flow))
        except Exception as e:
            logger.warning(f"  Path relevance computation failed: {e}, using degree")
            degrees = dict(self.graph.degree())
            max_deg = max(degrees.values()) if degrees.values() else 1.0
            scores = {node: deg / max_deg for node, deg in degrees.items()}
        
        return scores

    def _compute_community_bridge_score(self) -> Dict[str, float]:
        """Compute community bridge score."""
        logger.info("  Computing community bridge scores...")
        scores = {node: 0.0 for node in self.graph.nodes()}
        
        if self.communities_df is None or len(self.communities_df) == 0:
            return scores
        
        # Map nodes to communities
        node_to_communities = {}
        for _, comm in self.communities_df.iterrows():
            comm_id = comm.get('id', comm.get('community_id', None))
            if comm_id is None:
                continue

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
        
        # Score by number of communities connected
        for node in self.graph.nodes():
            if node in node_to_communities:
                num_comms = len(node_to_communities[node])
                # Also check neighbors
                neighbor_comms = set()
                for neighbor in self.graph.neighbors(node):
                    if neighbor in node_to_communities:
                        neighbor_comms.update(node_to_communities[neighbor])
                total_comms = len(node_to_communities[node] | neighbor_comms)
                scores[node] = min(1.0, total_comms / 5.0)  # Normalize to max 5 communities
        
        return scores

    def _compute_semantic_relevance_score(self) -> Dict[str, float]:
        """Compute semantic relevance score."""
        logger.info("  Computing semantic relevance scores...")
        scores = {node: 0.0 for node in self.graph.nodes()}
        
        # Use entity frequency and description length as proxy
        if 'frequency' in self.entities_df.columns:
            freq_map = dict(zip(self.entities_df['title'], self.entities_df['frequency']))
            max_freq = self.entities_df['frequency'].max() if len(self.entities_df) > 0 else 1.0
            
            for node in self.graph.nodes():
                freq = freq_map.get(node, 0)
                # Also check description length if available
                desc_len = 0
                entity = self.entities_df[self.entities_df['title'] == node]
                if len(entity) > 0 and 'description' in entity.columns:
                    desc = entity.iloc[0].get('description', '')
                    if isinstance(desc, str):
                        desc_len = len(desc)
                
                # Combine frequency and description length
                freq_score = (freq / max_freq) if max_freq > 0 else 0.0
                desc_score = min(1.0, desc_len / 500.0)  # Normalize to 500 chars
                scores[node] = 0.7 * freq_score + 0.3 * desc_score
        
        return scores

    def compute_unified_scores(self) -> Dict[str, float]:
        """
        Compute unified scores combining all signals.

        Returns:
            Dictionary mapping node IDs to unified scores
        """
        logger.info("📊 Computing unified scores...")

        # Compute all component scores
        connectivity_scores = self._compute_connectivity_score()
        importance_scores = self._compute_importance_score()
        path_scores = self._compute_path_relevance_score()
        bridge_scores = self._compute_community_bridge_score()
        semantic_scores = self._compute_semantic_relevance_score()

        # Adaptive weights based on region type
        unified_scores = {}
        
        for node in self.graph.nodes():
            region = self.node_regions.get(node, 'sparse_periphery')
            
            # Set weights based on region
            if region == 'dense_core':
                # Emphasize connectivity and bridges
                w1, w2, w3, w4, w5 = 0.3, 0.2, 0.1, 0.3, 0.1
            elif region == 'sparse_periphery':
                # Emphasize importance
                w1, w2, w3, w4, w5 = 0.1, 0.4, 0.2, 0.1, 0.2
            elif region == 'high_degree_hub':
                # Emphasize connectivity and importance
                w1, w2, w3, w4, w5 = 0.3, 0.3, 0.1, 0.2, 0.1
            elif region == 'low_degree_leaf':
                # Emphasize path relevance and semantic
                w1, w2, w3, w4, w5 = 0.1, 0.1, 0.4, 0.1, 0.3
            elif region == 'community_bridge':
                # Emphasize bridges and connectivity
                w1, w2, w3, w4, w5 = 0.2, 0.2, 0.1, 0.4, 0.1
            else:
                # Default balanced weights
                w1, w2, w3, w4, w5 = 0.2, 0.2, 0.2, 0.2, 0.2

            unified_scores[node] = (
                w1 * connectivity_scores.get(node, 0.0) +
                w2 * importance_scores.get(node, 0.0) +
                w3 * path_scores.get(node, 0.0) +
                w4 * bridge_scores.get(node, 0.0) +
                w5 * semantic_scores.get(node, 0.0)
            )

        self.unified_scores = unified_scores
        logger.info(f"✅ Unified scores computed for {len(unified_scores)} nodes")
        return unified_scores

    def select_protected_nodes(
        self,
        protected_fraction: float = 0.20,
        hub_degree_percentile: float = 0.75
    ) -> Set[str]:
        """
        Select nodes to always protect during pruning.

        Args:
            protected_fraction: Fraction of top-scored nodes to protect
            hub_degree_percentile: Degree percentile for hub classification

        Returns:
            Set of protected node IDs
        """
        logger.info("🛡️  Selecting protected nodes...")
        protected = set()

        # 1. Top N% by unified score (but exclude nodes that will be handled by region-specific pruning)
        sorted_nodes = sorted(
            self.unified_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        # Only protect top-scored nodes that are NOT in sparse_periphery or low_degree_leaf
        # (those will be handled by region-specific aggressive pruning)
        top_scored = set()
        for node, score in sorted_nodes:
            region = self.node_regions.get(node, 'sparse_periphery')
            if region not in ['sparse_periphery', 'low_degree_leaf']:
                top_scored.add(node)
            if len(top_scored) >= int(len(sorted_nodes) * protected_fraction):
                break
        protected.update(top_scored)
        logger.info(f"  Top {protected_fraction*100:.0f}% by unified score (excluding sparse/leaves): {len(top_scored)} nodes")

        # 2. All community bridges
        bridges = {node for node, region in self.node_regions.items() 
                  if region == 'community_bridge'}
        protected.update(bridges)
        logger.info(f"  Community bridges: {len(bridges)} nodes")

        # 3. All high-degree hubs
        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())
        if degree_values:
            degree_threshold = np.percentile(degree_values, hub_degree_percentile * 100)
            hubs = {node for node, deg in degrees.items() if deg >= degree_threshold}
            protected.update(hubs)
            logger.info(f"  High-degree hubs: {len(hubs)} nodes (degree >= {degree_threshold:.1f})")

        # 4. Nodes in important paths (only if not in sparse/leaf regions)
        path_threshold = np.percentile(list(self.unified_scores.values()), 85)
        path_nodes = set()
        for node, score in self.unified_scores.items():
            if score >= path_threshold:
                region = self.node_regions.get(node, 'sparse_periphery')
                # Only protect path nodes if they're not in aggressive pruning regions
                if region not in ['sparse_periphery', 'low_degree_leaf']:
                    path_nodes.add(node)
        protected.update(path_nodes)
        self.path_nodes = path_nodes
        logger.info(f"  Path-relevant nodes (excluding sparse/leaves): {len(path_nodes)} nodes")

        self.protected_nodes = protected
        logger.info(f"✅ Total protected nodes: {len(protected)}")
        return protected

    def prune_by_region(
        self,
        target_reduction: float = 0.55,
        protected_nodes: Optional[Set[str]] = None
    ) -> nx.DiGraph:
        """
        Apply region-specific pruning strategies.

        Args:
            target_reduction: Target reduction percentage (0-1)
            protected_nodes: Set of nodes to always keep

        Returns:
            Pruned graph
        """
        if protected_nodes is None:
            protected_nodes = self.protected_nodes

        logger.info(f"✂️  Pruning by region (target reduction: {target_reduction*100:.1f}%)...")
        
        nodes_to_keep = set(protected_nodes)
        
        # Region-specific pruning (adjusted to achieve target reduction)
        # Calculate dynamic keep fractions based on target reduction
        total_nodes = len(self.graph.nodes())
        target_nodes = int(total_nodes * (1 - target_reduction))
        current_kept = len(protected_nodes)
        remaining_slots = max(0, target_nodes - current_kept)
        
        # Count nodes per region (excluding protected)
        region_counts = {}
        for region_type in ['dense_core', 'sparse_periphery', 'low_degree_leaf']:
            region_nodes = [node for node, region in self.node_regions.items() 
                          if region == region_type and node not in protected_nodes]
            region_counts[region_type] = len(region_nodes)
        
        total_region_nodes = sum(region_counts.values())
        
        if total_region_nodes > 0 and remaining_slots > 0:
            # Calculate keep fractions to achieve target reduction
            # More aggressive pruning for sparse and leaves
            dense_keep_fraction = min(0.55, remaining_slots * 0.4 / max(1, region_counts.get('dense_core', 1)))
            sparse_keep_fraction = min(0.20, remaining_slots * 0.35 / max(1, region_counts.get('sparse_periphery', 1)))
            leaf_keep_fraction = min(0.10, remaining_slots * 0.25 / max(1, region_counts.get('low_degree_leaf', 1)))
        else:
            # Fallback to more aggressive default fractions
            dense_keep_fraction = 0.55
            sparse_keep_fraction = 0.20
            leaf_keep_fraction = 0.10
        
        region_keep_fractions = {
            'dense_core': dense_keep_fraction,  # Keep 60% (adjusted)
            'sparse_periphery': sparse_keep_fraction,  # Keep 25% (more aggressive)
            'high_degree_hub': 1.0,  # Always keep
            'low_degree_leaf': leaf_keep_fraction,  # Keep 15% (more aggressive)
            'community_bridge': 1.0  # Always keep
        }

        for region_type, keep_fraction in region_keep_fractions.items():
            region_nodes = [node for node, region in self.node_regions.items() 
                          if region == region_type and node not in protected_nodes]
            
            if not region_nodes:
                continue

            # Sort by unified score
            region_scores = [(node, self.unified_scores.get(node, 0.0)) 
                           for node in region_nodes]
            region_scores.sort(key=lambda x: x[1], reverse=True)
            
            num_keep = max(1, int(len(region_nodes) * keep_fraction))
            kept_nodes = {node for node, _ in region_scores[:num_keep]}
            nodes_to_keep.update(kept_nodes)
            
            logger.info(f"  {region_type}: kept {len(kept_nodes)}/{len(region_nodes)} "
                       f"({len(kept_nodes)/len(region_nodes)*100:.1f}%)")

        # Build pruned graph
        pruned_graph = self.graph.subgraph(nodes_to_keep).copy()
        
        # Edge pruning for hubs: keep top-k edges
        degrees = dict(self.graph.degree())
        hub_threshold = np.percentile(list(degrees.values()), 75)
        hubs = {node for node, deg in degrees.items() if deg >= hub_threshold}
        
        edges_to_remove = []
        for hub in hubs:
            if hub not in pruned_graph:
                continue
            out_edges = list(pruned_graph.out_edges(hub, data=True))
            if len(out_edges) > 10:  # Keep top 10 edges
                # Score edges by target node importance
                edge_scores = []
                for u, v, data in out_edges:
                    score = self.unified_scores.get(v, 0.0)
                    edge_scores.append((u, v, score))
                edge_scores.sort(key=lambda x: x[2], reverse=True)
                # Remove edges beyond top 10
                for u, v, _ in edge_scores[10:]:
                    edges_to_remove.append((u, v))
        
        pruned_graph.remove_edges_from(edges_to_remove)
        if edges_to_remove:
            logger.info(f"  Pruned {len(edges_to_remove)} edges from hubs")

        reduction = 100 * (1 - len(pruned_graph.nodes()) / len(self.graph.nodes()))
        logger.info(f"✅ Region-specific pruning complete: {len(pruned_graph.nodes())} nodes "
                   f"({reduction:.1f}% reduction)")
        
        return pruned_graph

    def validate_connectivity(
        self,
        pruned_graph: nx.DiGraph,
        min_connectivity_pct: float = 0.90
    ) -> Tuple[nx.DiGraph, bool]:
        """
        Validate and fix connectivity issues.

        Args:
            pruned_graph: Pruned graph to validate
            min_connectivity_pct: Minimum percentage of nodes in largest component

        Returns:
            Tuple of (validated graph, is_valid)
        """
        logger.info("🔗 Validating connectivity...")
        
        if pruned_graph.is_directed():
            components = list(nx.weakly_connected_components(pruned_graph))
        else:
            components = list(nx.connected_components(pruned_graph))
        
        if not components:
            logger.warning("  No components found!")
            return pruned_graph, False
        
        largest = max(components, key=len)
        largest_pct = len(largest) / len(pruned_graph.nodes()) if pruned_graph.nodes() else 0
        
        logger.info(f"  Largest component: {len(largest)}/{len(pruned_graph.nodes())} "
                   f"({largest_pct*100:.1f}%)")
        
        if largest_pct >= min_connectivity_pct:
            logger.info("✅ Connectivity validation passed")
            return pruned_graph, True
        
        # Re-add critical nodes to improve connectivity
        logger.info("  Re-adding critical nodes to improve connectivity...")
        nodes_to_add = set()
        
        # Find isolated important nodes
        isolated_components = [comp for comp in components if comp != largest]
        for comp in isolated_components:
            # Find highest-scored node in this component
            comp_scores = [(node, self.unified_scores.get(node, 0.0)) for node in comp]
            if comp_scores:
                best_node = max(comp_scores, key=lambda x: x[1])[0]
                # Add if it's a bridge or has high score
                if (self.node_regions.get(best_node) == 'community_bridge' or
                    self.unified_scores.get(best_node, 0.0) > 0.5):
                    nodes_to_add.add(best_node)
        
        if nodes_to_add:
            # Add nodes and their connections
            for node in nodes_to_add:
                if node in self.graph:
                    # Add node and edges to/from it that connect to existing graph
                    for neighbor in self.graph.neighbors(node):
                        if neighbor in pruned_graph:
                            pruned_graph.add_edge(node, neighbor)
                    for predecessor in self.graph.predecessors(node):
                        if predecessor in pruned_graph:
                            pruned_graph.add_edge(predecessor, node)
            
            logger.info(f"  Added {len(nodes_to_add)} critical nodes")
            
            # Re-check connectivity
            if pruned_graph.is_directed():
                components = list(nx.weakly_connected_components(pruned_graph))
            else:
                components = list(nx.connected_components(pruned_graph))
            
            if components:
                largest = max(components, key=len)
                largest_pct = len(largest) / len(pruned_graph.nodes())
                logger.info(f"  Updated largest component: {len(largest)}/{len(pruned_graph.nodes())} "
                           f"({largest_pct*100:.1f}%)")
        
        is_valid = largest_pct >= min_connectivity_pct
        if is_valid:
            logger.info("✅ Connectivity validation passed after fixes")
        else:
            logger.warning(f"⚠️  Connectivity still below threshold ({largest_pct*100:.1f}% < {min_connectivity_pct*100:.1f}%)")
        
        return pruned_graph, is_valid

    def prune(
        self,
        target_reduction: float = 0.55,
        min_connectivity_pct: float = 0.90,
        protected_fraction: float = 0.20,
        hub_degree_percentile: float = 0.75
    ) -> nx.DiGraph:
        """
        Run complete adaptive multi-strategy pruning pipeline.

        Args:
            target_reduction: Target reduction percentage (0-1)
            min_connectivity_pct: Minimum percentage in largest component
            protected_fraction: Fraction of top nodes to protect
            hub_degree_percentile: Degree percentile for hub classification

        Returns:
            Pruned graph
        """
        logger.info("🚀 Starting Adaptive Multi-Strategy Pruning Pipeline...")
        logger.info(f"  Target reduction: {target_reduction*100:.1f}%")
        logger.info(f"  Min connectivity: {min_connectivity_pct*100:.1f}%")

        # Stage 1: Analysis
        self.analyze_graph_regions()

        # Stage 2: Compute unified scores
        self.compute_unified_scores()

        # Stage 3: Select protected nodes
        self.select_protected_nodes(
            protected_fraction=protected_fraction,
            hub_degree_percentile=hub_degree_percentile
        )

        # Stage 4: Region-specific pruning
        pruned_graph = self.prune_by_region(
            target_reduction=target_reduction,
            protected_nodes=self.protected_nodes
        )

        # Stage 5: Connectivity validation
        pruned_graph, is_valid = self.validate_connectivity(
            pruned_graph,
            min_connectivity_pct=min_connectivity_pct
        )

        reduction = 100 * (1 - len(pruned_graph.nodes()) / len(self.graph.nodes()))
        logger.info(f"✅ Adaptive Multi-Strategy Pruning complete: "
                   f"{len(pruned_graph.nodes())} nodes ({reduction:.1f}% reduction), "
                   f"connectivity: {'✓' if is_valid else '⚠'}")

        return pruned_graph


def adaptive_multi_strategy_prune(
    graph: nx.DiGraph,
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    communities_df: Optional[pd.DataFrame] = None,
    target_reduction: float = 0.55,
    min_connectivity_pct: float = 0.90,
    protected_fraction: float = 0.20,
    hub_degree_percentile: float = 0.75
) -> nx.DiGraph:
    """
    Convenience function for adaptive multi-strategy pruning.

    Args:
        graph: Input graph
        entities_df: Entities DataFrame
        relationships_df: Relationships DataFrame
        communities_df: Optional communities DataFrame
        target_reduction: Target reduction percentage
        min_connectivity_pct: Minimum connectivity percentage
        protected_fraction: Protected node fraction
        hub_degree_percentile: Hub degree percentile

    Returns:
        Pruned graph
    """
    pruner = AdaptiveMultiStrategyPruner(
        graph, entities_df, relationships_df, communities_df
    )
    return pruner.prune(
        target_reduction=target_reduction,
        min_connectivity_pct=min_connectivity_pct,
        protected_fraction=protected_fraction,
        hub_degree_percentile=hub_degree_percentile
    )

