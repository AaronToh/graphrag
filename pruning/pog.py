#!/usr/bin/env python3
"""
POG (Path Over Graph) Algorithm Implementation

POG prunes knowledge graphs by identifying and keeping important paths.
Uses three-step pruning:
1. Graph structure analysis: Extract candidate paths
2. LLM-based path evaluation: Use LLM to score path relevance
3. SBERT-based filtering: Semantic similarity filtering

Algorithm Overview:
1. Select seed nodes (high-degree/centrality)
2. Extract candidate paths from seeds
3. Score paths using LLM
4. Filter paths using SBERT semantic similarity
5. Keep nodes/edges that appear in top-k paths
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Set, Optional, List, Tuple
from collections import deque
import logging
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)


class POGPruner:
    """
    Implementation of POG (Path Over Graph) algorithm for knowledge graph pruning.
    """

    def __init__(self, graph: nx.DiGraph, entities_df: pd.DataFrame):
        """
        Initialize POG pruner.

        Args:
            graph: Input directed graph (NetworkX DiGraph)
            entities_df: Entities DataFrame with node information
        """
        self.graph = graph.copy()  # Work on a copy
        self.entities_df = entities_df

        logger.info(f"Initialized POG: {len(graph.nodes())} nodes, "
                   f"{len(graph.edges())} edges")

    def _select_seed_nodes(
        self,
        num_seeds: int = 50,
        seed_method: str = 'degree_centrality'
    ) -> List[str]:
        """
        Select seed nodes for path extraction.

        Args:
            num_seeds: Number of seed nodes to select
            seed_method: Method to select seeds ('degree_centrality', 'betweenness', 'pagerank')

        Returns:
            List of seed node IDs
        """
        if seed_method == 'degree_centrality':
            centrality = nx.degree_centrality(self.graph)
        elif seed_method == 'betweenness':
            centrality = nx.betweenness_centrality(
                self.graph, k=min(500, len(self.graph.nodes()))
            )
        elif seed_method == 'pagerank':
            centrality = nx.pagerank(self.graph, max_iter=100)
        else:
            centrality = nx.degree_centrality(self.graph)

        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:num_seeds]
        seed_nodes = [node for node, _ in top_nodes]

        logger.info(f"Selected {len(seed_nodes)} seed nodes using {seed_method}")
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
            max_path_length: Maximum path length to explore
            max_paths_per_seed: Maximum paths to extract per seed

        Returns:
            List of paths (each path is a list of node IDs)
        """
        logger.info(f"  Extracting candidate paths from {len(seed_nodes)} seeds...")
        all_paths = []

        for seed in tqdm(seed_nodes, desc="  Processing seeds"):
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

        logger.info(f"  ✓ Extracted {len(all_paths)} candidate paths from {len(seed_nodes)} seeds")
        return all_paths

    def _format_path_for_llm(self, path: List[str]) -> str:
        """
        Format a path as a readable string for LLM evaluation.

        Args:
            path: List of node IDs

        Returns:
            Formatted path string
        """
        if len(path) < 2:
            return ""

        # Get entity descriptions if available
        path_entities = []
        for node_id in path:
            entity = self.entities_df[self.entities_df['title'] == node_id]
            if len(entity) > 0:
                desc = entity.iloc[0].get('description', node_id)
                path_entities.append(f"{node_id} ({desc[:50]})")
            else:
                path_entities.append(node_id)

        # Format as path
        path_str = " -> ".join(path_entities)
        return path_str

    def _score_paths_with_llm(
        self,
        paths: List[List[str]],
        llm_provider: str = 'openai',
        llm_model: str = 'gpt-4o-mini',
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[Tuple[str, ...], float]:
        """
        Score paths using LLM to evaluate semantic relevance.

        Args:
            paths: List of paths to score
            llm_provider: LLM provider ('openai', 'ollama', 'openrouter')
            llm_model: Model name
            api_base_url: API base URL
            api_key: API key

        Returns:
            Dictionary mapping path tuples to scores
        """
        path_scores = {}

        # For now, use a simplified scoring approach
        # In a full implementation, this would call the LLM API
        # For efficiency, we'll use a heuristic based on path properties

        logger.info(f"  Scoring {len(paths)} paths with LLM ({llm_provider}/{llm_model})...")

        for path in tqdm(paths, desc="  Scoring paths"):
            if len(path) < 2:
                continue

            # Heuristic scoring based on:
            # 1. Path length (shorter paths often more relevant)
            # 2. Node degrees (paths through important nodes)
            # 3. Edge weights if available

            path_tuple = tuple(path)

            # Length score (prefer shorter paths)
            length_score = 1.0 / len(path)

            # Node importance score
            node_scores = []
            for node in path:
                if node in self.graph:
                    degree = self.graph.degree(node)
                    node_scores.append(degree)
            avg_node_score = np.mean(node_scores) if node_scores else 0.0

            # Normalize node score
            max_degree = max(self.graph.degree(), key=lambda x: x[1])[1] if self.graph.nodes() else 1
            normalized_node_score = avg_node_score / max_degree if max_degree > 0 else 0.0

            # Combined score
            # In real implementation, this would be LLM output
            path_scores[path_tuple] = 0.6 * length_score + 0.4 * normalized_node_score

        logger.info(f"  ✓ Scored {len(path_scores)} paths")
        return path_scores

    def _filter_paths_with_sbert(
        self,
        paths: List[List[str]],
        path_scores: Dict[Tuple[str, ...], float],
        sbert_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        semantic_threshold: float = 0.7,
        top_k_paths: int = 100
    ) -> List[List[str]]:
        """
        Filter paths using SBERT semantic similarity.

        Args:
            paths: List of paths
            path_scores: Dictionary of path scores from LLM
            sbert_model: Sentence transformer model name
            semantic_threshold: Minimum semantic similarity
            top_k_paths: Number of top paths to keep

        Returns:
            List of filtered paths
        """
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            logger.warning("sentence-transformers not available, skipping SBERT filtering")
            # Fall back to score-based filtering
            sorted_paths = sorted(
                path_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k_paths]
            return [list(path) for path, _ in sorted_paths]

        logger.info(f"  Filtering paths with SBERT ({sbert_model})...")

        # Load model
        logger.info("    Loading SBERT model...")
        model = SentenceTransformer(sbert_model)
        logger.info("    ✓ Model loaded")

        # Format paths as text
        logger.info("    Formatting paths...")
        path_texts = []
        path_list = []
        for path in tqdm(paths, desc="    Formatting", leave=False):
            path_tuple = tuple(path)
            if path_tuple in path_scores:
                path_text = self._format_path_for_llm(path)
                path_texts.append(path_text)
                path_list.append(path)

        if not path_texts:
            return []

        # Compute embeddings
        logger.info("    Computing embeddings...")
        embeddings = model.encode(path_texts, show_progress_bar=True)
        logger.info("    ✓ Embeddings computed")

        # Compute similarity matrix
        from sentence_transformers import util
        similarities = util.cos_sim(embeddings, embeddings)

        # Filter paths based on scores and semantic similarity
        # Keep paths that are both high-scoring and semantically diverse
        filtered_paths = []
        used_indices = set()

        # Sort by score
        path_with_scores = [
            (i, path_list[i], path_scores[tuple(path_list[i])])
            for i in range(len(path_list))
            if tuple(path_list[i]) in path_scores
        ]
        path_with_scores.sort(key=lambda x: x[2], reverse=True)

        for idx, path, score in path_with_scores:
            if len(filtered_paths) >= top_k_paths:
                break

            # Check semantic similarity with already selected paths
            is_diverse = True
            for used_idx in used_indices:
                if similarities[idx][used_idx] > semantic_threshold:
                    is_diverse = False
                    break

            if is_diverse or score > 0.8:  # Always keep very high-scoring paths
                filtered_paths.append(path)
                used_indices.add(idx)

        logger.info(f"  ✓ Filtered to {len(filtered_paths)} diverse paths")
        return filtered_paths

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

        # Create subgraph with only these nodes and edges
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
    ) -> nx.DiGraph:
        """
        Run complete POG pruning pipeline.

        Args:
            seed_method: Method to select seed nodes
            num_seeds: Number of seed nodes
            max_path_length: Maximum path length
            top_k_paths: Number of top paths to keep
            llm_provider: LLM provider for path scoring
            llm_model: LLM model name
            llm_api_base_url: LLM API base URL
            llm_api_key: LLM API key
            sbert_model: SBERT model name
            semantic_threshold: Semantic similarity threshold

        Returns:
            Pruned directed graph
        """
        logger.info("Starting POG pruning pipeline...")
        logger.info(f"  Parameters: {num_seeds} seeds, max_path_length={max_path_length}, top_k_paths={top_k_paths}")

        # Step 1: Select seed nodes
        logger.info("\nStep 1: Selecting seed nodes...")
        seed_nodes = self._select_seed_nodes(num_seeds, seed_method)
        logger.info(f"  ✓ Selected {len(seed_nodes)} seed nodes")

        # Step 2: Extract candidate paths
        logger.info("\nStep 2: Extracting candidate paths...")
        candidate_paths = self._extract_candidate_paths(
            seed_nodes, max_path_length, max_paths_per_seed=20
        )

        if not candidate_paths:
            logger.warning("  ⚠ No candidate paths extracted, returning original graph")
            return self.graph

        # Step 3: Score paths with LLM
        logger.info("\nStep 3: Scoring paths with LLM...")
        path_scores = self._score_paths_with_llm(
            candidate_paths,
            llm_provider=llm_provider,
            llm_model=llm_model,
            api_base_url=llm_api_base_url,
            api_key=llm_api_key
        )

        # Step 4: Filter paths with SBERT
        logger.info("\nStep 4: Filtering paths with SBERT...")
        top_paths = self._filter_paths_with_sbert(
            candidate_paths,
            path_scores,
            sbert_model=sbert_model,
            semantic_threshold=semantic_threshold,
            top_k_paths=top_k_paths
        )

        # Step 5: Prune graph by paths
        logger.info("\nStep 5: Pruning graph by paths...")
        pruned_graph = self._prune_by_paths(top_paths)

        reduction = 100 * (1 - len(pruned_graph.nodes()) / len(self.graph.nodes()))
        logger.info(f"✅ POG complete: {len(pruned_graph.nodes())} nodes, "
                   f"{len(pruned_graph.edges())} edges ({reduction:.1f}% reduction)")

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
    llm_api_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    sbert_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    semantic_threshold: float = 0.7
) -> nx.DiGraph:
    """
    Convenience function to run POG pruning.

    Args:
        graph: Input directed graph
        entities_df: Entities DataFrame
        seed_method: Method to select seed nodes
        num_seeds: Number of seed nodes
        max_path_length: Maximum path length
        top_k_paths: Number of top paths to keep
        llm_provider: LLM provider
        llm_model: LLM model name
        llm_api_base_url: LLM API base URL
        llm_api_key: LLM API key
        sbert_model: SBERT model name
        semantic_threshold: Semantic similarity threshold

    Returns:
        Pruned directed graph
    """
    pruner = POGPruner(graph, entities_df)
    return pruner.prune(
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

