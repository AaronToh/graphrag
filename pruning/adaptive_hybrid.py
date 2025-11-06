"""
Adaptive Hybrid Pruning Method

This module implements an adaptive pruning strategy that selects and combines
pruning methods based on graph characteristics to optimize for both compute
reduction and accuracy preservation.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class GraphAnalyzer:
    """Analyze graph characteristics to inform pruning strategy selection."""
    
    def __init__(self, graph: nx.DiGraph, entities_df: pd.DataFrame):
        self.graph = graph
        self.entities_df = entities_df
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze graph characteristics.

        Returns:
            Dictionary with graph characteristics
        """
        num_nodes = len(self.graph.nodes())
        num_edges = len(self.graph.edges())
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
        
        # Component analysis
        if self.graph.is_directed():
            components = list(nx.weakly_connected_components(self.graph))
        else:
            components = list(nx.connected_components(self.graph))
        
        largest_component_size = max(len(c) for c in components) if components else 0
        connectivity_ratio = largest_component_size / num_nodes if num_nodes > 0 else 0
        
        # Graph size category
        if num_nodes < 5000:
            size_category = 'small'
        elif num_nodes < 15000:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        # Density category
        if avg_degree < 2:
            density_category = 'sparse'
        elif avg_degree < 5:
            density_category = 'moderate'
        else:
            density_category = 'dense'
        
        # Connectivity category
        if connectivity_ratio > 0.8:
            connectivity_category = 'well_connected'
        elif connectivity_ratio > 0.5:
            connectivity_category = 'moderately_connected'
        else:
            connectivity_category = 'fragmented'
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'num_components': len(components),
            'largest_component_size': largest_component_size,
            'connectivity_ratio': connectivity_ratio,
            'size_category': size_category,
            'density_category': density_category,
            'connectivity_category': connectivity_category
        }


class AdaptiveHybridPruner:
    """
    Adaptive hybrid pruning that selects methods based on graph characteristics.
    """
    
    def __init__(self, graph: nx.DiGraph, entities_df: pd.DataFrame,
                 relationships_df: pd.DataFrame, communities_df: Optional[pd.DataFrame] = None):
        self.graph = graph.copy()
        self.entities_df = entities_df
        self.relationships_df = relationships_df
        self.communities_df = communities_df
        self.analyzer = GraphAnalyzer(self.graph, self.entities_df)
    
    def select_strategy(self, target_reduction: float, min_accuracy: float = 0.7) -> Dict[str, Any]:
        """
        Select pruning strategy based on graph characteristics.

        Args:
            target_reduction: Desired graph size reduction (0.0-1.0)
            min_accuracy: Minimum acceptable accuracy threshold

        Returns:
            Dictionary with selected strategy and parameters
        """
        characteristics = self.analyzer.analyze()
        
        logger.info("Graph characteristics:")
        logger.info(f"  Size: {characteristics['size_category']} ({characteristics['num_nodes']} nodes)")
        logger.info(f"  Density: {characteristics['density_category']} (avg degree: {characteristics['avg_degree']:.2f})")
        logger.info(f"  Connectivity: {characteristics['connectivity_category']} ({characteristics['connectivity_ratio']:.2%})")
        
        strategy = {
            'stages': [],
            'target_reduction': target_reduction,
            'min_accuracy': min_accuracy
        }
        
        # Stage 1: Initial reduction based on size
        if characteristics['size_category'] == 'small':
            # Small graphs: Use path-based methods
            if target_reduction > 0.8:
                strategy['stages'].append({
                    'method': 'pathrag',
                    'params': {
                        'alpha': 0.7,
                        'theta': 0.1,
                        'top_n_nodes': max(20, int(characteristics['num_nodes'] * 0.1)),
                        'top_k_paths': max(10, int(characteristics['num_nodes'] * 0.05))
                    }
                })
            else:
                strategy['stages'].append({
                    'method': 'pog',
                    'params': {
                        'num_seeds': max(30, int(characteristics['num_nodes'] * 0.1)),
                        'top_k_paths': max(50, int(characteristics['num_nodes'] * 0.1))
                    }
                })
        
        elif characteristics['size_category'] == 'medium':
            # Medium graphs: Use KGTrimmer for balanced reduction
            keep_percentile = 1.0 - target_reduction
            strategy['stages'].append({
                'method': 'kgtrimmer',
                'params': {
                    'collective_weight': 0.5,
                    'holistic_weight': 0.5,
                    'min_importance_percentile': keep_percentile,
                    'preserve_connectivity': True
                }
            })
        
        else:  # large
            # Large graphs: Multi-stage approach
            # Stage 1: Initial reduction with KGTrimmer
            stage1_reduction = min(0.6, target_reduction * 0.7)
            stage1_keep = 1.0 - stage1_reduction
            strategy['stages'].append({
                'method': 'kgtrimmer',
                'params': {
                    'collective_weight': 0.4,
                    'holistic_weight': 0.6,
                    'min_importance_percentile': stage1_keep,
                    'preserve_connectivity': False
                }
            })
            
            # Stage 2: Path refinement if more reduction needed
            if target_reduction > 0.7:
                strategy['stages'].append({
                    'method': 'pathrag',
                    'params': {
                        'alpha': 0.8,
                        'theta': 0.05,
                        'top_n_nodes': 40,
                        'top_k_paths': 15
                    }
                })
        
        # Adjust for density
        if characteristics['density_category'] == 'dense':
            # Dense graphs: Add edge pruning stage
            strategy['stages'].insert(-1, {
                'method': 'edges_top_k',
                'params': {'k': 5}
            })
        
        # Adjust for connectivity
        if characteristics['connectivity_category'] == 'fragmented':
            # Fragmented graphs: Use CrumbTrail to preserve connectivity
            if 'kgtrimmer' in [s['method'] for s in strategy['stages']]:
                # Replace KGTrimmer with CrumbTrail
                strategy['stages'] = [s for s in strategy['stages'] if s['method'] != 'kgtrimmer']
                strategy['stages'].insert(0, {
                    'method': 'crumbtrail',
                    'params': {
                        'protected_fraction': 1.0 - target_reduction,
                        'protected_selection': 'degree_centrality'
                    }
                })
        
        logger.info(f"Selected strategy with {len(strategy['stages'])} stages:")
        for i, stage in enumerate(strategy['stages'], 1):
            logger.info(f"  Stage {i}: {stage['method']} with params {stage['params']}")
        
        return strategy
    
    def apply_strategy(self, strategy: Dict[str, Any], pruner_instance) -> nx.DiGraph:
        """
        Apply the selected strategy using the pruner instance.

        Args:
            strategy: Strategy dictionary from select_strategy()
            pruner_instance: GraphPruner instance to use for pruning

        Returns:
            Pruned graph
        """
        current_graph = self.graph.copy()
        current_entities = self.entities_df.copy()
        current_relationships = self.relationships_df.copy()
        
        for i, stage in enumerate(strategy['stages'], 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Stage {i}/{len(strategy['stages'])}: {stage['method']}")
            logger.info(f"{'='*60}")
            
            method = stage['method']
            params = stage['params']
            
            # Update pruner with current state
            pruner_instance.entities_df = current_entities
            pruner_instance.relationships_df = current_relationships
            pruner_instance.scorer.graph = current_graph
            
            # Apply method
            if method == 'kgtrimmer':
                from pruning.kgtrimmer import kgtrimmer_prune
                pruned_graph = kgtrimmer_prune(
                    current_graph,
                    current_entities,
                    self.communities_df,
                    **params
                )
            
            elif method == 'pog':
                from pruning.pog import pog_prune
                pruned_graph = pog_prune(
                    current_graph,
                    current_entities,
                    **params
                )
            
            elif method == 'pathrag':
                from pruning.pathrag import pathrag_prune
                pruned_graph = pathrag_prune(
                    current_graph,
                    current_entities,
                    **params
                )
            
            elif method == 'crumbtrail':
                from pruning.crumbtrail import crumbtrail_prune
                # Select protected nodes
                protected_nodes = pruner_instance._select_protected_nodes(
                    current_graph,
                    params['protected_fraction'],
                    params['protected_selection']
                )
                root_entity = "__VIRTUAL_ROOT__"
                pruned_graph = crumbtrail_prune(
                    current_graph,
                    protected_nodes,
                    root_entity,
                    max_iterations=1000
                )
            
            elif method == 'edges_top_k':
                # Simple edge pruning
                pruned_graph = current_graph.copy()
                for node in list(pruned_graph.nodes()):
                    out_edges = list(pruned_graph.out_edges(node, data=True))
                    if len(out_edges) > params['k']:
                        # Sort by weight if available, otherwise keep first k
                        sorted_edges = sorted(
                            out_edges,
                            key=lambda e: e[2].get('weight', 1.0),
                            reverse=True
                        )[:params['k']]
                        # Remove other edges
                        for u, v, _ in out_edges:
                            if (u, v) not in [(e[0], e[1]) for e in sorted_edges]:
                                pruned_graph.remove_edge(u, v)
            
            else:
                logger.warning(f"Unknown method: {method}, skipping stage")
                continue
            
            # Update current state
            pruned_node_ids = set(pruned_graph.nodes())
            current_entities = current_entities[
                current_entities['title'].isin(pruned_node_ids)
            ].copy()
            current_relationships = current_relationships[
                current_relationships['source'].isin(pruned_node_ids) &
                current_relationships['target'].isin(pruned_node_ids)
            ].copy()
            current_graph = pruned_graph
            
            logger.info(f"Stage {i} complete: {len(current_graph.nodes())} nodes, {len(current_graph.edges())} edges")
        
        return current_graph

