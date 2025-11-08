#!/usr/bin/env python3
"""
PathRAG Pruning Implementation

Flow propagation from seed nodes, path extraction, and top-k path retention.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class PathRAGPruner:
    """
    PathRAG pruning algorithm.
    
    Propagates flow from seed nodes and extracts high-flow paths.
    """
    
    def __init__(self, graph: nx.DiGraph, entities_df: pd.DataFrame):
        """
        Initialize PathRAG pruner.
        
        Args:
            graph: Graph to prune
            entities_df: Entities DataFrame
        """
        self.graph = graph.copy()
        self.entities_df = entities_df
        self.edge_flows = {}  # Store edge flows for path scoring
    
    def _select_seed_nodes(
        self,
        top_n_nodes: int = 40,
        seed_method: str = 'degree_centrality'
    ) -> List[str]:
        """
        Select seed nodes for flow propagation.
        
        Args:
            top_n_nodes: Number of top nodes to select
            seed_method: Method for selection
            
        Returns:
            List of seed node IDs
        """
        if seed_method == 'degree_centrality':
            centrality = nx.degree_centrality(self.graph)
        elif seed_method == 'combined':
            degree_cent = nx.degree_centrality(self.graph)
            try:
                graph_size = len(self.graph.nodes())
                betweenness = nx.betweenness_centrality(
                    self.graph, k=min(500, graph_size)
                )
            except Exception as e:
                logger.warning(f"Betweenness failed: {e}, using degree only")
                betweenness = {node: 0.0 for node in self.graph.nodes()}
            
            # Combine: 0.7 * degree + 0.3 * betweenness
            centrality = {}
            for node in self.graph.nodes():
                centrality[node] = 0.7 * degree_cent.get(node, 0) + 0.3 * betweenness.get(node, 0)
        else:
            centrality = nx.degree_centrality(self.graph)
        
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n_nodes]
        seed_nodes = [node for node, _ in top_nodes]
        
        return seed_nodes
    
    def _propagate_flow(
        self,
        seed_nodes: List[str],
        alpha: float = 0.8,  # Flow decay factor
        theta: float = 0.05,  # Convergence threshold
        max_iterations: int = 100
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        """
        Propagate flow from seed nodes iteratively.
        
        Args:
            seed_nodes: List of seed node IDs
            alpha: Flow decay factor
            theta: Convergence threshold
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (node_flows, edge_flows)
        """
        # Initialize flow values
        node_flows = {node: 0.0 for node in self.graph.nodes()}
        for seed in seed_nodes:
            if seed in self.graph:
                node_flows[seed] = 1.0
        
        # Iterative flow propagation
        for iteration in range(max_iterations):
            new_flows = {node: 0.0 for node in self.graph.nodes()}
            
            # Initialize seeds
            for seed in seed_nodes:
                if seed in self.graph:
                    new_flows[seed] = 1.0
            
            # Propagate flow
            for node in self.graph.nodes():
                if node in seed_nodes:
                    continue
                
                # Collect flow from predecessors
                total_inflow = 0.0
                for pred in self.graph.predecessors(node):
                    if pred in node_flows and self.graph.out_degree(pred) > 0:
                        # Flow decays by alpha and splits among outgoing edges
                        total_inflow += node_flows[pred] * alpha / self.graph.out_degree(pred)
                
                new_flows[node] = total_inflow
            
            # Check convergence
            max_change = max(
                abs(new_flows[node] - node_flows[node])
                for node in self.graph.nodes()
            )
            node_flows = new_flows
            
            if max_change < theta:
                break
        
        # Compute edge flows
        edge_flows = {}
        for u, v in self.graph.edges():
            if self.graph.out_degree(u) > 0:
                edge_flows[(u, v)] = node_flows.get(u, 0.0) / self.graph.out_degree(u)
            else:
                edge_flows[(u, v)] = 0.0
        
        return node_flows, edge_flows
    
    def _extract_paths_from_seeds(
        self,
        seed_nodes: List[str],
        edge_flows: Dict[Tuple[str, str], float],
        max_path_length: int = 5,
        max_paths_per_seed: int = 50
    ) -> List[Tuple[List[str], float]]:
        """
        Extract paths from seed nodes following high-flow edges.
        
        Args:
            seed_nodes: List of seed node IDs
            edge_flows: Dictionary of edge flows
            max_path_length: Maximum path length
            max_paths_per_seed: Maximum paths per seed
            
        Returns:
            List of (path, path_flow) tuples
        """
        all_paths = []
        
        for seed in seed_nodes:
            if seed not in self.graph:
                continue
            
            # BFS following high-flow edges
            queue = deque([(seed, [seed], 1.0)])  # (current, path, path_flow)
            paths_from_seed = []
            
            while queue and len(paths_from_seed) < max_paths_per_seed:
                current, path, path_flow = queue.popleft()
                
                if len(path) >= max_path_length:
                    continue
                
                # Explore neighbors via high-flow edges
                neighbors = list(self.graph.successors(current))
                neighbor_flows = [
                    (n, edge_flows.get((current, n), 0.0))
                    for n in neighbors
                ]
                neighbor_flows.sort(key=lambda x: x[1], reverse=True)
                
                # Take top neighbors by flow
                for neighbor, edge_flow in neighbor_flows[:5]:  # Top 5 neighbors
                    if neighbor not in path:  # Avoid cycles
                        new_path = path + [neighbor]
                        new_path_flow = path_flow * edge_flow  # Decay
                        paths_from_seed.append((new_path, new_path_flow))
                        if len(new_path) < max_path_length:
                            queue.append((neighbor, new_path, new_path_flow))
            
            all_paths.extend(paths_from_seed)
        
        return all_paths
    
    def _score_paths(
        self,
        paths: List[Tuple[List[str], float]],
        path_scoring_method: str = 'avg_edge_flow'  # 'min', 'avg', 'max'
    ) -> Dict[tuple, float]:
        """
        Score paths based on edge flows.
        
        Args:
            paths: List of (path, path_flow) tuples
            path_scoring_method: Method for scoring ('min', 'max', 'avg_edge_flow')
            
        Returns:
            Dictionary mapping path tuples to scores
        """
        path_scores = {}
        
        for path, path_flow in paths:
            if path_scoring_method == 'min':
                # Minimum edge flow in path
                edge_flows_in_path = [
                    self.edge_flows.get((path[i], path[i+1]), 0.0)
                    for i in range(len(path) - 1)
                ]
                score = min(edge_flows_in_path) if edge_flows_in_path else 0.0
            elif path_scoring_method == 'max':
                # Maximum edge flow in path
                edge_flows_in_path = [
                    self.edge_flows.get((path[i], path[i+1]), 0.0)
                    for i in range(len(path) - 1)
                ]
                score = max(edge_flows_in_path) if edge_flows_in_path else 0.0
            else:  # 'avg_edge_flow'
                # Average edge flow in path
                score = path_flow / len(path) if len(path) > 0 else 0.0
            
            path_scores[tuple(path)] = score
        
        return path_scores
    
    def prune(
        self,
        top_n_nodes: int = 40,
        top_k_paths: int = 15,
        max_path_length: int = 5,
        alpha: float = 0.8,
        theta: float = 0.05,
        seed_method: str = 'degree_centrality',
        path_scoring_method: str = 'avg_edge_flow'
    ) -> nx.DiGraph:
        """
        Prune graph by keeping top-k paths based on flow.
        
        Args:
            top_n_nodes: Number of seed nodes
            top_k_paths: Number of top paths to keep
            max_path_length: Maximum path length
            alpha: Flow decay factor
            theta: Convergence threshold
            seed_method: Seed selection method
            path_scoring_method: Path scoring method
            
        Returns:
            Pruned graph
        """
        # 1. Select seeds
        seed_nodes = self._select_seed_nodes(top_n_nodes, seed_method)
        
        # 2. Propagate flow
        node_flows, edge_flows = self._propagate_flow(
            seed_nodes, alpha, theta
        )
        self.edge_flows = edge_flows  # Store for path scoring
        
        # 3. Extract paths
        candidate_paths = self._extract_paths_from_seeds(
            seed_nodes, edge_flows, max_path_length
        )
        
        if not candidate_paths:
            logger.warning("No candidate paths found, returning original graph")
            return self.graph
        
        # 4. Score paths
        path_scores = self._score_paths(
            candidate_paths, path_scoring_method
        )
        
        # 5. Select top-k paths
        sorted_paths = sorted(
            path_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k_paths]
        
        # 6. Collect nodes and edges
        nodes_to_keep = set()
        edges_to_keep = set()
        
        for path, score in sorted_paths:
            nodes_to_keep.update(path)
            for i in range(len(path) - 1):
                edges_to_keep.add((path[i], path[i + 1]))
        
        # 7. Build pruned graph
        pruned_graph = self.graph.subgraph(nodes_to_keep).copy()
        edges_to_remove = [
            e for e in pruned_graph.edges()
            if e not in edges_to_keep
        ]
        pruned_graph.remove_edges_from(edges_to_remove)
        
        return pruned_graph


def pathrag_prune(
    graph: nx.DiGraph,
    entities_df: pd.DataFrame,
    top_n_nodes: int = 40,
    top_k_paths: int = 15,
    max_path_length: int = 5,
    alpha: float = 0.8,
    theta: float = 0.05,
    seed_method: str = 'degree_centrality',
    path_scoring_method: str = 'avg_edge_flow'
) -> nx.DiGraph:
    """
    Prune graph using PathRAG algorithm.
    
    Args:
        graph: Graph to prune
        entities_df: Entities DataFrame
        top_n_nodes: Number of seed nodes
        top_k_paths: Number of top paths to keep
        max_path_length: Maximum path length
        alpha: Flow decay factor
        theta: Convergence threshold
        seed_method: Seed selection method
        path_scoring_method: Path scoring method
        
    Returns:
        Pruned graph
    """
    pruner = PathRAGPruner(graph, entities_df)
    return pruner.prune(
        top_n_nodes=top_n_nodes,
        top_k_paths=top_k_paths,
        max_path_length=max_path_length,
        alpha=alpha,
        theta=theta,
        seed_method=seed_method,
        path_scoring_method=path_scoring_method
    )
