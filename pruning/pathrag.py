#!/usr/bin/env python3
"""
PathRAG Algorithm Implementation

PathRAG prunes knowledge graphs by:
1. Flow propagation from seed nodes
2. Path extraction and scoring based on edge flow
3. Keeping top-k paths connecting important nodes

Algorithm Overview:
1. Select seed nodes (top_n_nodes by centrality)
2. Initialize flow values at seed nodes
3. Propagate flow through graph with decay factor alpha
4. Extract paths from seeds following high-flow edges
5. Score paths using edge flow (min/avg/max)
6. Keep top-k paths and their nodes/edges
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Set, Optional, List, Tuple
from collections import deque
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PathRAGPruner:
    """
    Implementation of PathRAG algorithm for knowledge graph pruning.
    """

    def __init__(self, graph: nx.DiGraph, entities_df: pd.DataFrame):
        """
        Initialize PathRAG pruner.

        Args:
            graph: Input directed graph (NetworkX DiGraph)
            entities_df: Entities DataFrame with node information
        """
        self.graph = graph.copy()  # Work on a copy
        self.entities_df = entities_df

        logger.info(f"Initialized PathRAG: {len(graph.nodes())} nodes, "
                   f"{len(graph.edges())} edges")

    def _select_seed_nodes(
        self,
        top_n_nodes: int = 40,
        seed_method: str = 'degree_centrality'
    ) -> List[str]:
        """
        Select seed nodes for flow propagation.

        Args:
            top_n_nodes: Number of seed nodes to select
            seed_method: Method to select seeds ('degree_centrality', 'combined')

        Returns:
            List of seed node IDs
        """
        if seed_method == 'degree_centrality':
            centrality = nx.degree_centrality(self.graph)
        elif seed_method == 'combined':
            # Combine multiple centrality measures
            degree_cent = nx.degree_centrality(self.graph)
            try:
                betweenness = nx.betweenness_centrality(
                    self.graph, k=min(500, len(self.graph.nodes()))
                )
            except:
                betweenness = {node: 0.0 for node in self.graph.nodes()}
            
            # Combine scores
            centrality = {}
            for node in self.graph.nodes():
                centrality[node] = 0.7 * degree_cent.get(node, 0) + 0.3 * betweenness.get(node, 0)
        else:
            centrality = nx.degree_centrality(self.graph)

        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n_nodes]
        seed_nodes = [node for node, _ in top_nodes]

        logger.info(f"Selected {len(seed_nodes)} seed nodes using {seed_method}")
        return seed_nodes

    def _propagate_flow(
        self,
        seed_nodes: List[str],
        alpha: float = 0.8,
        theta: float = 0.05,
        max_iterations: int = 100
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        """
        Propagate flow from seed nodes through the graph.

        Args:
            seed_nodes: List of seed node IDs
            alpha: Flow decay factor (0-1)
            theta: Convergence threshold
            max_iterations: Maximum iterations

        Returns:
            Tuple of (node_flows dict, edge_flows dict)
        """
        # Initialize flow values
        node_flows = {node: 0.0 for node in self.graph.nodes()}
        for seed in seed_nodes:
            if seed in self.graph:
                node_flows[seed] = 1.0

        # Iterative flow propagation
        logger.info("    Propagating flow (this may take a while)...")
        for iteration in tqdm(range(max_iterations), desc="    Flow iterations", leave=False):
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
                logger.info(f"    ✓ Flow propagation converged after {iteration + 1} iterations")
                break

        # Compute edge flows
        edge_flows = {}
        for u, v in self.graph.edges():
            if u in node_flows and self.graph.out_degree(u) > 0:
                edge_flows[(u, v)] = node_flows[u] * alpha / self.graph.out_degree(u)
            else:
                edge_flows[(u, v)] = 0.0

        nodes_with_flow = len([n for n, f in node_flows.items() if f > 0])
        logger.info(f"    ✓ Flow propagation complete: {nodes_with_flow} nodes with flow")
        return node_flows, edge_flows

    def _extract_paths(
        self,
        seed_nodes: List[str],
        edge_flows: Dict[Tuple[str, str], float],
        max_path_length: int = 5,
        min_edge_flow: float = 0.01
    ) -> List[List[str]]:
        """
        Extract paths from seed nodes following high-flow edges.

        Args:
            seed_nodes: List of seed node IDs
            edge_flows: Dictionary of edge flow values
            max_path_length: Maximum path length
            min_edge_flow: Minimum edge flow to follow

        Returns:
            List of paths (each path is a list of node IDs)
        """
        logger.info(f"  Extracting paths from {len(seed_nodes)} seeds...")
        all_paths = []

        for seed in tqdm(seed_nodes, desc="  Processing seeds"):
            if seed not in self.graph:
                continue

            # DFS to extract paths following high-flow edges
            stack = [(seed, [seed])]
            paths_from_seed = []

            while stack and len(paths_from_seed) < 50:  # Limit paths per seed
                current, path = stack.pop()

                if len(path) >= max_path_length:
                    continue

                # Explore neighbors with high flow
                neighbors = list(self.graph.successors(current))
                for neighbor in neighbors:
                    edge_flow = edge_flows.get((current, neighbor), 0.0)
                    if edge_flow >= min_edge_flow and neighbor not in path:  # Avoid cycles
                        new_path = path + [neighbor]
                        paths_from_seed.append(new_path)
                        if len(new_path) < max_path_length:
                            stack.append((neighbor, new_path))

            all_paths.extend(paths_from_seed)

        logger.info(f"  ✓ Extracted {len(all_paths)} paths from {len(seed_nodes)} seeds")
        return all_paths

    def _score_paths(
        self,
        paths: List[List[str]],
        edge_flows: Dict[Tuple[str, str], float],
        path_scoring_method: str = 'avg_edge_flow'
    ) -> Dict[Tuple[str, ...], float]:
        """
        Score paths using edge flow values.

        Args:
            paths: List of paths to score
            edge_flows: Dictionary of edge flow values
            path_scoring_method: How to score paths ('min_edge_flow', 'avg_edge_flow', 'max_edge_flow')

        Returns:
            Dictionary mapping path tuples to scores
        """
        path_scores = {}

        for path in paths:
            if len(path) < 2:
                continue

            path_tuple = tuple(path)

            # Collect edge flows for this path
            edge_flow_values = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                flow = edge_flows.get((u, v), 0.0)
                edge_flow_values.append(flow)

            if not edge_flow_values:
                path_scores[path_tuple] = 0.0
                continue

            # Score based on method
            if path_scoring_method == 'min_edge_flow':
                path_scores[path_tuple] = min(edge_flow_values)
            elif path_scoring_method == 'max_edge_flow':
                path_scores[path_tuple] = max(edge_flow_values)
            else:  # avg_edge_flow (default)
                path_scores[path_tuple] = np.mean(edge_flow_values)

        logger.info(f"Scored {len(path_scores)} paths using {path_scoring_method}")
        return path_scores

    def _prune_by_paths(
        self,
        top_paths: List[List[str]]
    ) -> nx.DiGraph:
        """
        Prune graph to keep only nodes/edges in top paths.

        Args:
            top_paths: List of top paths to preserve

        Returns:
            Pruned graph
        """
        # Collect all nodes and edges in top paths
        nodes_to_keep = set()
        edges_to_keep = set()

        for path in top_paths:
            for i, node in enumerate(path):
                nodes_to_keep.add(node)
                if i < len(path) - 1:
                    edges_to_keep.add((path[i], path[i + 1]))

        # Create subgraph with only these nodes
        pruned_graph = self.graph.subgraph(nodes_to_keep).copy()

        # Remove edges not in top paths
        edges_to_remove = [
            (u, v) for u, v in pruned_graph.edges()
            if (u, v) not in edges_to_keep
        ]
        pruned_graph.remove_edges_from(edges_to_remove)

        logger.info(f"Pruned graph: {len(pruned_graph.nodes())} nodes, "
                   f"{len(pruned_graph.edges())} edges")

        return pruned_graph

    def prune(
        self,
        alpha: float = 0.8,
        theta: float = 0.05,
        top_n_nodes: int = 40,
        top_k_paths: int = 15,
        max_path_length: int = 5,
        seed_method: str = 'degree_centrality',
        path_scoring_method: str = 'avg_edge_flow'
    ) -> nx.DiGraph:
        """
        Run complete PathRAG pruning pipeline.

        Args:
            alpha: Flow decay factor (default: 0.8)
            theta: Early stopping threshold (default: 0.05)
            top_n_nodes: Number of seed nodes (default: 40)
            top_k_paths: Number of paths to keep (default: 15)
            max_path_length: Maximum path length (default: 5)
            seed_method: Method to select seeds ('degree_centrality', 'combined')
            path_scoring_method: How to score paths ('min_edge_flow', 'avg_edge_flow', 'max_edge_flow')

        Returns:
            Pruned directed graph
        """
        logger.info("Starting PathRAG pruning pipeline...")
        logger.info(f"  Alpha: {alpha}, Theta: {theta}")
        logger.info(f"  Top N nodes: {top_n_nodes}, Top K paths: {top_k_paths}")

        # Step 1: Select seed nodes
        logger.info("\nStep 1: Selecting seed nodes...")
        seed_nodes = self._select_seed_nodes(top_n_nodes, seed_method)
        logger.info(f"  ✓ Selected {len(seed_nodes)} seed nodes")

        # Step 2: Propagate flow
        logger.info("\nStep 2: Propagating flow...")
        node_flows, edge_flows = self._propagate_flow(
            seed_nodes, alpha=alpha, theta=theta
        )

        # Step 3: Extract paths
        logger.info("\nStep 3: Extracting paths...")
        candidate_paths = self._extract_paths(
            seed_nodes, edge_flows, max_path_length=max_path_length
        )

        if not candidate_paths:
            logger.warning("  ⚠ No candidate paths extracted, returning original graph")
            return self.graph

        # Step 4: Score paths
        logger.info("\nStep 4: Scoring paths...")
        path_scores = self._score_paths(
            candidate_paths, edge_flows, path_scoring_method=path_scoring_method
        )

        # Step 5: Select top-k paths
        logger.info("\nStep 5: Selecting top-k paths...")
        sorted_paths = sorted(
            path_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k_paths]
        top_paths = [list(path) for path, _ in sorted_paths]

        logger.info(f"  ✓ Selected top {len(top_paths)} paths")

        # Step 6: Prune graph by paths
        logger.info("\nStep 6: Pruning graph by paths...")
        pruned_graph = self._prune_by_paths(top_paths)

        reduction = 100 * (1 - len(pruned_graph.nodes()) / len(self.graph.nodes()))
        logger.info(f"✅ PathRAG complete: {len(pruned_graph.nodes())} nodes, "
                   f"{len(pruned_graph.edges())} edges ({reduction:.1f}% reduction)")

        return pruned_graph


def pathrag_prune(
    graph: nx.DiGraph,
    entities_df: pd.DataFrame,
    alpha: float = 0.8,
    theta: float = 0.05,
    top_n_nodes: int = 40,
    top_k_paths: int = 15,
    max_path_length: int = 5,
    seed_method: str = 'degree_centrality',
    path_scoring_method: str = 'avg_edge_flow'
) -> nx.DiGraph:
    """
    Convenience function to run PathRAG pruning.

    Args:
        graph: Input directed graph
        entities_df: Entities DataFrame
        alpha: Flow decay factor
        theta: Early stopping threshold
        top_n_nodes: Number of seed nodes
        top_k_paths: Number of paths to keep
        max_path_length: Maximum path length
        seed_method: Method to select seeds
        path_scoring_method: How to score paths

    Returns:
        Pruned directed graph
    """
    pruner = PathRAGPruner(graph, entities_df)
    return pruner.prune(
        alpha=alpha,
        theta=theta,
        top_n_nodes=top_n_nodes,
        top_k_paths=top_k_paths,
        max_path_length=max_path_length,
        seed_method=seed_method,
        path_scoring_method=path_scoring_method
    )

