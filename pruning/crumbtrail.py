#!/usr/bin/env python3
"""
CrumbTrail Pruning Implementation

Bottom-up iterative layering that preserves connectivity from root to protected nodes.
"""

import networkx as nx
import numpy as np
from typing import Set, Dict, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CrumbTrailPruner:
    """
    CrumbTrail pruning algorithm.
    
    Preserves connectivity from root to protected nodes through iterative layering.
    """
    
    def __init__(self, graph: nx.DiGraph, protected_nodes: Set[str], root: str):
        """
        Initialize CrumbTrail pruner.
        
        Args:
            graph: Directed graph to prune
            protected_nodes: Set of nodes to protect
            root: Root node for the algorithm
        """
        self.graph = graph.copy()
        self.protected_nodes = protected_nodes.copy()
        self.root = root
        self.ground = set()  # Leaf nodes in P
        self.intermediate = set()  # Non-leaf nodes in P
        self.postponed = defaultdict(set)  # layer -> nodes
        self.layers = {}  # layer_num -> set of nodes
        
    def preprocess(self):
        """Preprocess graph: remove self-loops, edges to root, isolated nodes, break cycles."""
        # 1. Remove self-loops
        self_loops = list(nx.selfloop_edges(self.graph))
        self.graph.remove_edges_from(self_loops)
        
        # 2. Remove edges to root
        edges_to_root = [(u, self.root) for u in self.graph.predecessors(self.root)]
        self.graph.remove_edges_from(edges_to_root)
        
        # 3. Remove isolated nodes not in P
        isolates = [n for n in nx.isolates(self.graph) if n not in self.protected_nodes]
        self.graph.remove_nodes_from(isolates)
        
        # 4. Break cycles through outgoing edges of P nodes
        for p in self.protected_nodes:
            if p not in self.graph:
                continue
            for successor in list(self.graph.successors(p)):
                if successor in self.graph and nx.has_path(self.graph, successor, p):
                    self.graph.remove_edge(p, successor)
    
    def initialize_ground_and_intermediate(self):
        """Initialize ground (leaf protected nodes) and intermediate (non-leaf protected nodes)."""
        # Ground: protected nodes with no outgoing edges (leaves)
        self.ground = {n for n in self.protected_nodes
                      if n in self.graph and self.graph.out_degree(n) == 0}
        
        # Intermediate: protected nodes that are not leaves
        self.intermediate = {n for n in self.protected_nodes
                            if n in self.graph and n not in self.ground}
        
        # Initialize layer 0 with Ground
        self.layers[0] = self.ground.copy()
        
        # Postpone intermediate nodes to layer 1
        if self.intermediate:
            self.postponed[1] = self.intermediate.copy()
    
    def create_new_layer(self, layer_num: int) -> Set[str]:
        """
        Create new layer from previous layer.
        
        Args:
            layer_num: Current layer number
            
        Returns:
            Set of nodes in new layer
        """
        V_ell = set()
        prev_layer = self.layers.get(layer_num - 1, set())
        
        # Get predecessors of previous layer nodes
        for node in prev_layer:
            if node in self.graph:
                for pred in self.graph.predecessors(node):
                    if pred not in self.ground:  # Don't add ground nodes
                        V_ell.add(pred)
        
        return V_ell
    
    def break_cycles_in_layer(self, layer_nodes: Set[str]):
        """Break cycles within a layer."""
        # Find cycles involving layer nodes
        cycles_to_break = []
        for node in layer_nodes:
            if node not in self.graph:
                continue
            for successor in list(self.graph.successors(node)):
                if successor in layer_nodes and nx.has_path(self.graph, successor, node):
                    cycles_to_break.append((node, successor))
        
        # Remove cycle-inducing edges
        for u, v in cycles_to_break:
            if self.graph.has_edge(u, v):
                self.graph.remove_edge(u, v)
    
    def postpone_nodes(self, layer_nodes: Set[str], layer_num: int) -> Set[str]:
        """
        Postpone nodes that depend on nodes in later layers.
        
        Args:
            layer_nodes: Nodes in current layer
            layer_num: Current layer number
            
        Returns:
            Nodes remaining in layer after postponement
        """
        to_postpone = set()
        
        for node in layer_nodes:
            if node not in self.graph:
                continue
            
            # Check if node has successors in later layers
            for successor in self.graph.successors(node):
                # If successor is in a later layer or postponed, postpone this node
                in_later_layer = any(successor in self.layers.get(l, set()) 
                                    for l in range(layer_num + 1, max(self.layers.keys(), default=0) + 1))
                in_postponed = any(successor in self.postponed.get(l, set()) 
                                  for l in range(layer_num + 1, max(self.postponed.keys(), default=0) + 1))
                
                if in_later_layer or in_postponed:
                    to_postpone.add(node)
                    break
        
        # Postpone nodes
        for node in to_postpone:
            self.postponed[layer_num + 1].add(node)
        
        return layer_nodes - to_postpone
    
    def prune_unessential(self, layer_nodes: Set[str]) -> Set[str]:
        """
        Remove nodes that don't connect protected nodes to ground.
        
        Args:
            layer_nodes: Nodes in current layer
            
        Returns:
            Essential nodes remaining
        """
        essential = set()
        
        for node in layer_nodes:
            if node in self.protected_nodes:
                essential.add(node)
                continue
            
            # Check if node connects to ground
            if node not in self.graph:
                continue
            
            # Check if there's a path from this node to ground
            has_path_to_ground = False
            for ground_node in self.ground:
                if nx.has_path(self.graph, node, ground_node):
                    has_path_to_ground = True
                    break
            
            # Check if there's a path from protected nodes to this node
            has_path_from_protected = False
            for prot_node in self.protected_nodes:
                if prot_node in self.graph and nx.has_path(self.graph, prot_node, node):
                    has_path_from_protected = True
                    break
            
            if has_path_to_ground and has_path_from_protected:
                essential.add(node)
        
        return essential
    
    def run(self, max_iterations: int = 1000) -> nx.DiGraph:
        """
        Run CrumbTrail pruning algorithm.
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            Pruned graph
        """
        # Preprocess
        self.preprocess()
        
        # Initialize
        self.initialize_ground_and_intermediate()
        
        layer_num = 0
        
        logger.info("[CrumbTrail] Starting iterative layering (max_iterations=%s, protected=%s, initial_ground=%s)" % (max_iterations, len(self.protected_nodes), len(self.ground)))

        while self.root not in self.ground and layer_num < max_iterations:
            layer_num += 1
            
            # Create new layer
            V_ell = self.create_new_layer(layer_num)
            
            if not V_ell:
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

            postponed_total = sum(len(nodes) for nodes in self.postponed.values())
            logger.info(
                "[CrumbTrail] Layer %d complete: %d nodes (ground=%d, intermediate=%d, postponed=%d)"
                % (
                    layer_num,
                    len(V_ell),
                    len(self.ground),
                    len(self.intermediate),
                    postponed_total,
                )
            )

            # Add postponed nodes for this layer
            if layer_num in self.postponed:
                V_ell.update(self.postponed[layer_num])
                del self.postponed[layer_num]
        
        logger.info("[CrumbTrail] Finished after %d layers. Ground=%d, postponed_remaining=%d" % (len(self.layers), len(self.ground), sum(len(nodes) for nodes in self.postponed.values())))

        # Collect all nodes to keep
        nodes_to_keep = set()
        for layer_nodes in self.layers.values():
            nodes_to_keep.update(layer_nodes)
        nodes_to_keep.add(self.root)
        
        # Build pruned graph
        pruned_graph = self.graph.subgraph(nodes_to_keep).copy()
        
        return pruned_graph


def crumbtrail_prune(
    graph: nx.DiGraph,
    protected_nodes: Set[str],
    root: str,
    max_iterations: int = 1000
) -> nx.DiGraph:
    """
    Prune graph using CrumbTrail algorithm.
    
    Args:
        graph: Directed graph to prune
        protected_nodes: Set of nodes to protect
        root: Root node
        max_iterations: Maximum iterations
        
    Returns:
        Pruned graph
    """
    pruner = CrumbTrailPruner(graph, protected_nodes, root)
    return pruner.run(max_iterations)

