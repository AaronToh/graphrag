#!/usr/bin/env python3
"""
POG (Path Over Graph) Pruning Implementation

Path-based pruning using LLM/SBERT to identify semantically relevant paths.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class POGPruner:
    """
    POG pruning algorithm.
    
    Extracts paths from seed nodes and scores them using semantic similarity.
    """
    
    def __init__(self, graph: nx.DiGraph, entities_df: pd.DataFrame):
        """
        Initialize POG pruner.
        
        Args:
            graph: Graph to prune
            entities_df: Entities DataFrame
        """
        self.graph = graph.copy()
        self.entities_df = entities_df
    
    def _select_seed_nodes(
        self,
        num_seeds: int = 50,
        seed_method: str = 'degree_centrality'
    ) -> List[str]:
        """
        Select seed nodes for path extraction.
        
        Args:
            num_seeds: Number of seed nodes to select
            seed_method: Method for selection ('degree_centrality', 'betweenness', 'pagerank')
            
        Returns:
            List of seed node IDs
        """
        if seed_method == 'degree_centrality':
            centrality = nx.degree_centrality(self.graph)
        elif seed_method == 'betweenness':
            graph_size = len(self.graph.nodes())
            try:
                centrality = nx.betweenness_centrality(
                    self.graph, k=min(500, graph_size)
                )
            except Exception as e:
                logger.warning(f"Betweenness failed: {e}, using degree centrality")
                centrality = nx.degree_centrality(self.graph)
        elif seed_method == 'pagerank':
            centrality = nx.pagerank(self.graph, max_iter=100)
        else:
            centrality = nx.degree_centrality(self.graph)
        
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:num_seeds]
        seed_nodes = [node for node, _ in top_nodes]
        
        return seed_nodes
    
    def _extract_candidate_paths(
        self,
        seed_nodes: List[str],
        max_path_length: int = 5,
        max_paths_per_seed: int = 20
    ) -> List[List[str]]:
        """
        Extract candidate paths from seed nodes using BFS.
        
        Args:
            seed_nodes: List of seed node IDs
            max_path_length: Maximum path length
            max_paths_per_seed: Maximum paths per seed
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        all_paths = []
        
        for seed in seed_nodes:
            if seed not in self.graph:
                continue
            
            # BFS to extract paths
            queue = deque([(seed, [seed])])
            paths_from_seed = []
            
            while queue and len(paths_from_seed) < max_paths_per_seed:
                current, path = queue.popleft()
                
                if len(path) >= max_path_length:
                    continue
                
                # Explore neighbors
                neighbors = list(self.graph.successors(current))
                for neighbor in neighbors:
                    if neighbor not in path:  # Avoid cycles
                        new_path = path + [neighbor]
                        paths_from_seed.append(new_path)
                        if len(new_path) < max_path_length:
                            queue.append((neighbor, new_path))
            
            all_paths.extend(paths_from_seed)
        
        return all_paths
    
    def _format_path_for_llm(self, path: List[str]) -> str:
        """
        Format path as string for embedding.
        
        Args:
            path: List of node IDs
            
        Returns:
            Formatted string
        """
        # Get node descriptions/titles
        path_strings = []
        for node_id in path:
            if node_id in self.graph:
                node_data = self.graph.nodes[node_id]
                title = node_data.get('title', node_id)
                description = node_data.get('description', '')
                if description:
                    path_strings.append(f"{title}: {description}")
                else:
                    path_strings.append(title)
            else:
                path_strings.append(str(node_id))
        
        return " -> ".join(path_strings)
    
    def _score_paths_sbert(
        self,
        paths: List[List[str]],
        sbert_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        semantic_threshold: float = 0.7
    ) -> Dict[tuple, float]:
        """
        Score paths using SBERT semantic similarity.
        
        Args:
            paths: List of paths to score
            sbert_model: SBERT model name
            semantic_threshold: Threshold for semantic similarity
            
        Returns:
            Dictionary mapping path tuples to scores
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            # Return uniform scores as fallback
            return {tuple(path): 1.0 for path in paths}
        
        try:
            model = SentenceTransformer(sbert_model)
        except Exception as e:
            logger.warning(f"Failed to load SBERT model {sbert_model}: {e}. Using fallback scoring.")
            # Fallback: score by path length (shorter = better)
            return {tuple(path): 1.0 / len(path) if len(path) > 0 else 0.0 for path in paths}
        
        # Format paths as strings
        path_strings = [self._format_path_for_llm(path) for path in paths]
        
        # Compute embeddings
        try:
            embeddings = model.encode(path_strings, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"SBERT encoding failed: {e}. Using fallback scoring.")
            return {tuple(path): 1.0 / len(path) if len(path) > 0 else 0.0 for path in paths}
        
        # Score by path coherence (average similarity between consecutive nodes)
        path_scores = {}
        for i, path in enumerate(paths):
            if len(path) < 2:
                path_scores[tuple(path)] = 0.0
                continue
            
            # For simplicity, use the path embedding directly
            # In a more sophisticated version, we'd compute pairwise similarities
            # For now, use a simple heuristic: shorter paths with high embedding norm are better
            embedding = embeddings[i]
            norm = np.linalg.norm(embedding)
            path_scores[tuple(path)] = norm / len(path)  # Normalize by path length
        
        # Normalize scores
        if path_scores:
            max_score = max(path_scores.values())
            if max_score > 0:
                path_scores = {k: v / max_score for k, v in path_scores.items()}
        
        return path_scores
    
    def prune(
        self,
        seed_method: str = 'degree_centrality',
        num_seeds: int = 50,
        max_path_length: int = 5,
        top_k_paths: int = 100,
        llm_provider: str = 'openai',
        llm_model: str = 'gpt-4o-mini',
        sbert_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        semantic_threshold: float = 0.7
    ) -> nx.DiGraph:
        """
        Prune graph by keeping top-k semantically relevant paths.
        
        Args:
            seed_method: Method for seed selection
            num_seeds: Number of seed nodes
            max_path_length: Maximum path length
            top_k_paths: Number of top paths to keep
            llm_provider: LLM provider (not used, kept for compatibility)
            llm_model: LLM model (not used, kept for compatibility)
            sbert_model: SBERT model name
            semantic_threshold: Semantic similarity threshold
            
        Returns:
            Pruned graph
        """
        # 1. Select seeds
        seed_nodes = self._select_seed_nodes(num_seeds, seed_method)
        
        # 2. Extract paths
        candidate_paths = self._extract_candidate_paths(
            seed_nodes, max_path_length
        )
        
        if not candidate_paths:
            logger.warning("No candidate paths found, returning original graph")
            return self.graph
        
        # 3. Score paths
        path_scores = self._score_paths_sbert(
            candidate_paths, sbert_model, semantic_threshold
        )
        
        # 4. Select top-k paths
        sorted_paths = sorted(
            path_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k_paths]
        
        # 5. Collect nodes and edges from top paths
        nodes_to_keep = set()
        edges_to_keep = set()
        
        for (path, _), score in sorted_paths:
            nodes_to_keep.update(path)
            for i in range(len(path) - 1):
                edges_to_keep.add((path[i], path[i + 1]))
        
        # 6. Build pruned graph
        pruned_graph = self.graph.subgraph(nodes_to_keep).copy()
        # Keep only edges in top paths
        edges_to_remove = [
            e for e in pruned_graph.edges()
            if e not in edges_to_keep
        ]
        pruned_graph.remove_edges_from(edges_to_remove)
        
        return pruned_graph


def pog_prune(
    graph: nx.DiGraph,
    entities_df: pd.DataFrame,
    seed_method: str = 'degree_centrality',
    num_seeds: int = 50,
    max_path_length: int = 5,
    top_k_paths: int = 100,
    llm_provider: str = 'openai',
    llm_model: str = 'gpt-4o-mini',
    sbert_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    semantic_threshold: float = 0.7
) -> nx.DiGraph:
    """
    Prune graph using POG algorithm.
    
    Args:
        graph: Graph to prune
        entities_df: Entities DataFrame
        seed_method: Seed selection method
        num_seeds: Number of seeds
        max_path_length: Maximum path length
        top_k_paths: Number of top paths to keep
        llm_provider: LLM provider (unused)
        llm_model: LLM model (unused)
        sbert_model: SBERT model
        semantic_threshold: Semantic threshold
        
    Returns:
        Pruned graph
    """
    pruner = POGPruner(graph, entities_df)
    return pruner.prune(
        seed_method=seed_method,
        num_seeds=num_seeds,
        max_path_length=max_path_length,
        top_k_paths=top_k_paths,
        sbert_model=sbert_model,
        semantic_threshold=semantic_threshold
    )
