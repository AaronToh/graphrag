#!/usr/bin/env python3
"""
CrumbTrail Algorithm Implementation

This module implements the CrumbTrail algorithm from:
"Efficient Pruning of Large Knowledge Graphs" (Faralli et al., 2018)

CrumbTrail is a bottom-up, iterative layering algorithm that prunes noisy/cyclic
graphs while preserving connectivity from a root to a set of protected nodes (P).

Algorithm Overview:
1. Preprocessing: Remove self-loops, edges to root, break cycles through P
2. Layering: Build layers bottom-up from leaf (Ground) nodes to root
3. Cycle Breaking: Remove cycle-inducing edges at each layer
4. Postponing: Defer nodes that depend on later layers
5. Pruning: Remove unessential nodes that don't connect P to leaf nodes

The algorithm is efficient: O(|V| + |E|) per iteration, scalable to large graphs.
"""

import networkx as nx
from collections import defaultdict, deque
from typing import Set, Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CrumbTrailPruner:
    """
    Implementation of the CrumbTrail algorithm for graph pruning.
    """

    def __init__(self, graph: nx.DiGraph, protected_nodes: Set[str], root: str):
        """
        Initialize CrumbTrail pruner.

        Args:
            graph: Input directed graph (NetworkX DiGraph)
            protected_nodes: Set of protected node IDs (must include root)
            root: Root node ID (must be in graph and protected_nodes)
        """
        if root not in protected_nodes:
            raise ValueError(f"Root '{root}' must be in protected_nodes")
        if root not in graph:
            raise ValueError(f"Root '{root}' not found in graph")

        self.original_graph = graph
        self.graph = graph.copy()  # Work on a copy
        self.protected_nodes = protected_nodes.copy()
        self.root = root

        # Algorithm data structures
        self.ground = set()  # Leaf nodes in P
        self.intermediate = set()  # Non-leaf nodes in P
        self.postponed = defaultdict(set)  # layer -> nodes
        self.layers = {}  # layer_num -> set of nodes

        logger.info(f"Initialized CrumbTrail: {len(graph.nodes())} nodes, "
                   f"{len(graph.edges())} edges, {len(protected_nodes)} protected, "
                   f"root='{root}'")

    def preprocess(self):
        """
        Preprocessing step (Algorithm 1, Section 4).

        Steps:
        1. Remove self-loops
        2. Remove edges to root
        3. Remove isolated nodes not in P
        4. Break cycles through outgoing edges of P nodes
        """
        logger.info("🔧 Preprocessing graph...")

        # 1. Remove self-loops
        self_loops = list(nx.selfloop_edges(self.graph))
        self.graph.remove_edges_from(self_loops)
        if self_loops:
            logger.debug(f"  Removed {len(self_loops)} self-loops")

        # 2. Remove edges to root
        edges_to_root = [(u, self.root) for u in self.graph.predecessors(self.root)]
        self.graph.remove_edges_from(edges_to_root)
        if edges_to_root:
            logger.debug(f"  Removed {len(edges_to_root)} edges to root")

        # 3. Remove isolated nodes not in P
        isolates = [n for n in nx.isolates(self.graph) if n not in self.protected_nodes]
        self.graph.remove_nodes_from(isolates)
        if isolates:
            logger.debug(f"  Removed {len(isolates)} isolated nodes")

        # 4. Break cycles through outgoing edges of P nodes
        cycles_broken = 0
        for p in self.protected_nodes:
            if p not in self.graph:
                continue
            for successor in list(self.graph.successors(p)):
                if nx.has_path(self.graph, successor, p):  # Cycle detected
                    self.graph.remove_edge(p, successor)
                    cycles_broken += 1
                    logger.debug(f"  Broke cycle: {p} -> {successor}")

        if cycles_broken:
            logger.info(f"  Broke {cycles_broken} cycles through P nodes")

        logger.info(f"✅ Preprocessing complete: {len(self.graph.nodes())} nodes, "
                   f"{len(self.graph.edges())} edges remaining")

    def initialize_ground_and_intermediate(self):
        """
        Initialize Ground (P leaves) and Intermediate (P non-leaves) sets.
        Also estimate initial postponement layers for Intermediate nodes.
        """
        logger.info("🌱 Initializing Ground and Intermediate sets...")

        # Ground: protected nodes with no outgoing edges (leaves)
        self.ground = {n for n in self.protected_nodes
                      if n in self.graph and self.graph.out_degree(n) == 0}

        # Intermediate: protected nodes that are not leaves
        self.intermediate = {n for n in self.protected_nodes
                            if n in self.graph and n not in self.ground}

        logger.info(f"  Ground: {len(self.ground)} nodes")
        logger.info(f"  Intermediate: {len(self.intermediate)} nodes")

        # Simplified postponement: all intermediate nodes start at layer 1
        # This avoids expensive shortest path computations
        # The algorithm will naturally postpone them as needed during processing
        if self.intermediate:
            self.postponed[1] = self.intermediate.copy()
            logger.info(f"  Postponed {len(self.intermediate)} intermediate nodes to layer 1")

        # Initialize layer 0 with Ground
        self.layers[0] = self.ground.copy()
        logger.info(f"  Layer 0 initialized with {len(self.ground)} nodes")

    def create_new_layer(self, layer_num: int) -> Set[str]:
        """
        Create a new layer (createNewLayer in Algorithm 1).

        Args:
            layer_num: Current layer number

        Returns:
            Set of nodes for this layer
        """
        logger.debug(f"  Creating layer {layer_num}...")

        V_ell = set()
        prev_layer = self.layers[layer_num - 1]

        # Add nodes with edges to previous layer
        # that don't have paths to postponed nodes
        candidates = set()
        for v in prev_layer:
            candidates.update(self.graph.predecessors(v))

        # Filter candidates: only nodes not leading to postponed nodes
        postponed_nodes = set()
        for nodes in self.postponed.values():
            postponed_nodes.update(nodes)

        for u in candidates:
            # Check if u has path to any postponed node
            has_path_to_postponed = False
            for p in postponed_nodes:
                if u != p and p in self.graph and nx.has_path(self.graph, u, p):
                    has_path_to_postponed = True
                    break

            if not has_path_to_postponed:
                V_ell.add(u)

        # Add postponed nodes for this layer
        if layer_num in self.postponed:
            V_ell.update(self.postponed[layer_num])
            del self.postponed[layer_num]

        logger.debug(f"    Layer {layer_num}: {len(V_ell)} nodes")
        return V_ell

    def break_cycles_in_layer(self, layer_nodes: Set[str]):
        """
        Break cycles through outgoing edges in the current layer.
        Uses BFS tree from root to identify non-tree edges to remove.

        Args:
            layer_nodes: Set of nodes in current layer
        """
        if not layer_nodes:
            return

        # Build BFS tree from root
        try:
            Tr = nx.bfs_tree(self.graph, source=self.root, reverse=False)
        except nx.NetworkXError:
            logger.warning(f"  Cannot build BFS tree from root '{self.root}'")
            return

        cycles_broken = 0
        for x in layer_nodes:
            if x not in self.graph:
                continue

            for y in list(self.graph.successors(x)):
                if nx.has_path(self.graph, y, x):  # Cycle detected
                    # Find non-tree edge in cycle path to remove
                    try:
                        path = nx.shortest_path(self.graph, y, x)
                        for i in range(len(path) - 1):
                            edge = (path[i], path[i + 1])
                            if edge not in Tr.edges():  # Non-tree edge
                                self.graph.remove_edge(*edge)
                                cycles_broken += 1
                                logger.debug(f"    Broke cycle: removed edge {edge}")
                                break
                    except nx.NetworkXNoPath:
                        pass

        if cycles_broken:
            logger.debug(f"    Broke {cycles_broken} cycles in layer")

    def postpone_nodes(self, layer_nodes: Set[str], layer_num: int) -> Set[str]:
        """
        Postpone nodes that have dependencies on other nodes in the same layer.

        Args:
            layer_nodes: Current layer nodes
            layer_num: Current layer number

        Returns:
            Updated layer_nodes with postponed nodes removed
        """
        to_postpone = {}

        for u in list(layer_nodes):
            for v in layer_nodes:
                if u != v and u in self.graph and v in self.graph:
                    if nx.has_path(self.graph, u, v):
                        # u depends on v, so postpone u
                        try:
                            path_len = nx.shortest_path_length(self.graph, u, v)
                            # Postpone by path length
                            new_layer = layer_num + path_len
                            to_postpone[u] = max(to_postpone.get(u, 0), new_layer)
                        except nx.NetworkXNoPath:
                            pass

        # Apply postponements
        for u, new_layer in to_postpone.items():
            self.postponed[new_layer].add(u)
            layer_nodes.remove(u)
            logger.debug(f"    Postponed {u} to layer {new_layer}")

        return layer_nodes

    def prune_unessential(self, layer_nodes: Set[str]) -> Set[str]:
        """
        Prune unessential nodes from the layer (pruneUnessential in Algorithm 1).

        A node v is unessential if:
        - I(v) is empty (no intermediate nodes reach it), OR
        - G(v) is empty (it doesn't reach any ground nodes)

        Optimized version using BFS for reachability.

        Args:
            layer_nodes: Current layer nodes

        Returns:
            Filtered layer_nodes with unessential nodes removed
        """
        to_remove = []

        # Build reverse graph for backward reachability (I(v))
        reverse_graph = self.graph.reverse()

        for v in list(layer_nodes):
            if v not in self.graph:
                continue

            # Compute I(v): intermediates that can reach v (using reverse graph)
            I_v = set()
            if v in reverse_graph:
                reachable_from_v = nx.single_source_shortest_path_length(
                    reverse_graph, v, cutoff=None
                )
                I_v = self.intermediate & set(reachable_from_v.keys())

            # Compute G(v): ground nodes reachable from v (using forward graph)
            G_v = set()
            if v in self.graph:
                reachable_to = nx.single_source_shortest_path_length(
                    self.graph, v, cutoff=None
                )
                G_v = self.ground & set(reachable_to.keys())

            # If either set is empty, v is unessential
            if not I_v or not G_v:
                self.graph.remove_node(v)
                to_remove.append(v)
                logger.debug(f"    Pruned unessential node: {v} "
                           f"(I(v)={len(I_v)}, G(v)={len(G_v)})")

        # Remove from layer
        for v in to_remove:
            layer_nodes.discard(v)

        if to_remove:
            logger.debug(f"    Removed {len(to_remove)} unessential nodes")

        return layer_nodes

    def run(self, max_iterations: int = 1000) -> nx.DiGraph:
        """
        Run the complete CrumbTrail algorithm.

        Args:
            max_iterations: Maximum number of layers to create (safety limit)

        Returns:
            Pruned directed graph
        """
        logger.info("🚀 Running CrumbTrail algorithm...")

        # Step 1: Preprocessing
        self.preprocess()

        # Step 2: Initialize Ground and Intermediate
        self.initialize_ground_and_intermediate()

        # Step 3: Iterative layering
        layer_num = 0

        while self.root not in self.ground and layer_num < max_iterations:
            layer_num += 1
            logger.debug(f"\n🔄 Processing layer {layer_num}...")

            # Create new layer
            V_ell = self.create_new_layer(layer_num)

            if not V_ell and not self.postponed:
                # No more nodes to process
                logger.info("  No more nodes to process, stopping")
                break

            # Break cycles in layer
            self.break_cycles_in_layer(V_ell)

            # Postpone dependent nodes
            V_ell = self.postpone_nodes(V_ell, layer_num)

            # Prune unessential nodes
            V_ell = self.prune_unessential(V_ell)

            # Update Ground and Intermediate
            P_ell = V_ell & self.protected_nodes
            self.ground.update(P_ell)
            self.intermediate -= P_ell

            # Store layer
            self.layers[layer_num] = V_ell

            logger.debug(f"  Layer {layer_num} complete: {len(V_ell)} nodes")
            logger.debug(f"  Ground now has {len(self.ground)} nodes")
            logger.debug(f"  Intermediate now has {len(self.intermediate)} nodes")

        if layer_num >= max_iterations:
            logger.warning(f"⚠️  Reached maximum iterations ({max_iterations})")

        # Build final pruned graph
        remaining_nodes = set(self.graph.nodes())
        pruned_graph = self.graph.subgraph(remaining_nodes).copy()

        logger.info(f"✅ CrumbTrail complete: {len(pruned_graph.nodes())} nodes, "
                   f"{len(pruned_graph.edges())} edges in {layer_num} layers")

        return pruned_graph


def crumbtrail_prune(graph: nx.DiGraph, protected_nodes: Set[str], root: str,
                     max_iterations: int = 1000) -> nx.DiGraph:
    """
    Convenience function to run CrumbTrail pruning.

    Args:
        graph: Input directed graph
        protected_nodes: Set of protected node IDs (must include root)
        root: Root node ID
        max_iterations: Maximum number of layers (default: 1000)

    Returns:
        Pruned directed graph
    """
    pruner = CrumbTrailPruner(graph, protected_nodes, root)
    return pruner.run(max_iterations=max_iterations)
