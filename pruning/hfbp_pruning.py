#!/usr/bin/env python3
"""
Hierarchical Flow-Based Pruning (HFBP) Algorithm Implementation

This module implements the HFBP algorithm as an improvement to PathRAG's path retrieval.
It uses hierarchical supergraph construction with resource propagation and beam search
for efficient path enumeration in large knowledge graphs.

Key Features:
- Leiden community detection for supernode creation
- Query-aware supernode enhancement
- Hierarchical resource propagation
- Beam search path enumeration
- Efficient for graphs with |V| = 50k+ nodes

Author: GraphRAG Pruning Lab
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from pathlib import Path as FilePath
import json
from datetime import datetime
import heapq

# Community detection
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    logging.warning("igraph not available - will use fallback community detection")

# Embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class Supernode:
    """Represents a supernode (community) in the hierarchical graph."""
    id: int
    member_nodes: List[str]  # Node IDs in this community
    context_node: str  # Highest PageRank node
    summary: str  # Generated summary text
    pagerank_scores: Dict[str, float] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Supernode):
            return self.id == other.id
        return False


@dataclass
class Path:
    """Represents a path through the graph with scoring information."""
    nodes: List[str]
    edges: List[Tuple[str, str]]
    score: float
    context_nodes: Set[str] = field(default_factory=set)
    resource_scores: Dict[str, float] = field(default_factory=dict)
    text_representation: str = ""

    def __lt__(self, other):
        return self.score < other.score


@dataclass
class HFBPConfig:
    """Configuration for HFBP algorithm."""
    N: int = 40  # Number of top retrieved nodes to consider
    K: int = 15  # Number of top paths to return
    alpha: float = 0.8  # Resource propagation damping factor
    theta: float = 0.05  # Pruning threshold (resource/outdegree)
    beam_k: int = 5  # Beam width for path enumeration
    max_propagation_steps: int = 3  # Maximum propagation iterations
    target_supernode_size_min: int = 20  # Min nodes per supernode
    target_supernode_size_max: int = 50  # Max nodes per supernode
    top_supernode_matches: int = 5  # Top matching supernodes to add to Vq
    context_bonus: float = 0.1  # Bonus per context node in path
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence transformer model
    max_path_length: int = 10  # Maximum path length to consider


class HFBPPruning:
    """
    Hierarchical Flow-Based Pruning implementation for GraphRAG.

    This class implements the complete HFBP algorithm including:
    1. Supernode precomputation via Leiden clustering
    2. Query-aware enhancement of retrieved nodes
    3. Supergraph construction
    4. Resource propagation with pruning
    5. Beam search path enumeration
    6. Path scoring and ranking
    """

    def __init__(self,
                 entities_df: pd.DataFrame,
                 relationships_df: pd.DataFrame,
                 text_units_df: pd.DataFrame = None,
                 config: HFBPConfig = None):
        """
        Initialize HFBP pruning.

        Args:
            entities_df: GraphRAG entities DataFrame
            relationships_df: GraphRAG relationships DataFrame
            text_units_df: GraphRAG text units DataFrame (optional)
            config: HFBP configuration
        """
        self.entities_df = entities_df.copy()
        self.relationships_df = relationships_df.copy()
        self.text_units_df = text_units_df.copy() if text_units_df is not None else None
        self.config = config or HFBPConfig()

        # Build main graph
        self.graph = self._build_graph()

        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)

        # Supernode structures
        self.supernodes: Dict[int, Supernode] = {}
        self.node_to_supernode: Dict[str, int] = {}
        self.supergraph: nx.Graph = None

        # Cache for embeddings
        self.node_embeddings: Dict[str, np.ndarray] = {}

        logger.info(f"Initialized HFBP with {len(self.graph.nodes)} nodes, "
                   f"{len(self.graph.edges)} edges")

    def _build_graph(self) -> nx.Graph:
        """Build NetworkX graph from GraphRAG artifacts."""
        G = nx.Graph()

        # Add nodes with attributes
        # Try multiple possible ID fields
        for _, entity in self.entities_df.iterrows():
            # Priority: title > id > name
            node_id = str(entity.get('title') or entity.get('id') or entity.get('name', ''))
            if not node_id:
                continue

            G.add_node(
                node_id,
                title=entity.get('title', node_id),
                type=entity.get('type', ''),
                description=entity.get('description', ''),
                text=entity.get('description', ''),  # For text retrieval
                original_id=entity.get('id', node_id)  # Keep track of original ID
            )

        # Add edges with weights
        for _, rel in self.relationships_df.iterrows():
            source = str(rel.get('source', ''))
            target = str(rel.get('target', ''))

            if not source or not target:
                continue

            weight = float(rel.get('weight', 1.0))
            description = rel.get('description', '')

            # Check if nodes exist
            if source in G.nodes and target in G.nodes:
                G.add_edge(
                    source,
                    target,
                    weight=weight,
                    description=description
                )

        logger.info(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Debug: if no edges, log a warning
        if G.number_of_edges() == 0 and len(self.relationships_df) > 0:
            logger.warning("No edges added! Checking node ID mismatches...")
            sample_nodes = list(G.nodes())[:5]
            sample_rels = [(r['source'], r['target']) for _, r in self.relationships_df.head().iterrows()]
            logger.warning(f"Sample nodes: {sample_nodes}")
            logger.warning(f"Sample relationship sources/targets: {sample_rels}")

        return G

    def precompute_supernodes(self, force_recompute: bool = False) -> Dict[int, Supernode]:
        """
        Step 1: Precompute supernodes using Leiden community detection.

        Args:
            force_recompute: Force recomputation even if supernodes exist

        Returns:
            Dictionary mapping supernode ID to Supernode object
        """
        if self.supernodes and not force_recompute:
            logger.info("Using cached supernodes")
            return self.supernodes

        logger.info("🔍 Step 1: Precomputing supernodes via Leiden clustering...")

        # Run Leiden community detection
        communities = self._leiden_clustering()

        # Compute PageRank for context node selection
        pagerank = nx.pagerank(self.graph, alpha=0.85)

        # Build supernodes
        self.supernodes = {}
        self.node_to_supernode = {}

        for comm_id, members in communities.items():
            # Find context node (highest PageRank)
            context_node = max(members, key=lambda n: pagerank.get(n, 0))

            # Compute PageRank scores for all members
            member_pageranks = {node: pagerank.get(node, 0) for node in members}

            # Generate summary
            summary = self._generate_supernode_summary(members, context_node)

            # Create supernode
            supernode = Supernode(
                id=comm_id,
                member_nodes=list(members),
                context_node=context_node,
                summary=summary,
                pagerank_scores=member_pageranks
            )

            # Generate embedding for summary
            supernode.embedding = self.embedding_model.encode(summary, convert_to_numpy=True)

            self.supernodes[comm_id] = supernode

            # Map nodes to supernodes
            for node in members:
                self.node_to_supernode[node] = comm_id

        logger.info(f"✅ Created {len(self.supernodes)} supernodes "
                   f"(avg size: {np.mean([len(s.member_nodes) for s in self.supernodes.values()]):.1f})")

        return self.supernodes

    def _leiden_clustering(self) -> Dict[int, Set[str]]:
        """
        Perform Leiden community detection.

        Returns:
            Dictionary mapping community ID to set of node IDs
        """
        if IGRAPH_AVAILABLE:
            return self._leiden_igraph()
        else:
            return self._leiden_fallback()

    def _leiden_igraph(self) -> Dict[int, Set[str]]:
        """Use igraph's Leiden implementation."""
        logger.info("Using igraph Leiden implementation")

        # Convert to igraph
        node_list = list(self.graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        g = ig.Graph()
        g.add_vertices(len(node_list))

        edges = []
        weights = []
        for u, v, data in self.graph.edges(data=True):
            edges.append((node_to_idx[u], node_to_idx[v]))
            weights.append(data.get('weight', 1.0))

        g.add_edges(edges)

        # Run Leiden
        partition = g.community_leiden(
            objective_function='modularity',
            weights=weights,
            resolution=1.0,
            n_iterations=2
        )

        # Convert back to node IDs
        communities = defaultdict(set)
        for idx, comm_id in enumerate(partition.membership):
            communities[comm_id].add(node_list[idx])

        return dict(communities)

    def _leiden_fallback(self) -> Dict[int, Set[str]]:
        """Fallback to Louvain community detection if igraph not available."""
        logger.info("Using Louvain fallback (igraph not available)")

        from networkx.algorithms import community

        # Use Louvain method as fallback
        communities_generator = community.louvain_communities(
            self.graph,
            weight='weight',
            resolution=1.0,
            seed=42
        )

        communities = {}
        for comm_id, members in enumerate(communities_generator):
            communities[comm_id] = members

        return communities

    def _generate_supernode_summary(self, members: List[str], context_node: str) -> str:
        """
        Generate textual summary for a supernode.

        Args:
            members: List of node IDs in the community
            context_node: The context node (highest PageRank)

        Returns:
            Summary text
        """
        # Collect entity information
        entity_texts = []
        relation_texts = []

        # Sample up to 10 entities for summary
        sample_size = min(10, len(members))
        sampled_members = [context_node] + [m for m in members if m != context_node][:sample_size-1]

        for node_id in sampled_members:
            if node_id in self.graph.nodes:
                node_data = self.graph.nodes[node_id]
                node_type = node_data.get('type', 'entity')
                node_desc = node_data.get('description', node_data.get('text', ''))
                entity_texts.append(f"{node_id} ({node_type}): {node_desc[:100]}")

        # Sample some edges
        subgraph = self.graph.subgraph(members)
        edge_sample = list(subgraph.edges(data=True))[:5]

        for u, v, data in edge_sample:
            desc = data.get('description', 'related to')
            relation_texts.append(f"{u} - {desc[:50]} - {v}")

        # Build summary
        summary = f"Community centered on {context_node}. "
        summary += f"Entities: {'; '.join(entity_texts[:5])}. "
        if relation_texts:
            summary += f"Relations: {'; '.join(relation_texts[:3])}."

        return summary

    def enhance_retrieved_nodes(self, query: str, vq: List[str]) -> List[str]:
        """
        Step 2: Enhance retrieved nodes by matching query to supernode summaries.

        Args:
            query: The search query
            vq: List of initially retrieved node IDs

        Returns:
            Enhanced list of node IDs (includes original + nodes from top matching supernodes)
        """
        logger.info(f"🔍 Step 2: Enhancing {len(vq)} retrieved nodes with query matching...")

        # Ensure supernodes exist
        if not self.supernodes:
            self.precompute_supernodes()

        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        # Match query to supernode summaries
        supernode_scores = []
        for sn_id, supernode in self.supernodes.items():
            similarity = self._cosine_similarity(query_embedding, supernode.embedding)
            supernode_scores.append((sn_id, similarity))

        # Get top matching supernodes
        supernode_scores.sort(key=lambda x: x[1], reverse=True)
        top_supernodes = [sn_id for sn_id, _ in supernode_scores[:self.config.top_supernode_matches]]

        # Add nodes from top supernodes to Vq
        enhanced_vq = set(vq)
        for sn_id in top_supernodes:
            supernode = self.supernodes[sn_id]
            enhanced_vq.update(supernode.member_nodes)

        logger.info(f"✅ Enhanced to {len(enhanced_vq)} nodes "
                   f"(added {len(enhanced_vq) - len(vq)} from {len(top_supernodes)} supernodes)")

        return list(enhanced_vq)

    def build_supergraph(self) -> nx.Graph:
        """
        Step 3: Build supergraph from supernodes.

        Returns:
            Supergraph with supernodes as vertices
        """
        logger.info("🔍 Step 3: Building supergraph...")

        # Ensure supernodes exist
        if not self.supernodes:
            self.precompute_supernodes()

        G_prime = nx.Graph()

        # Add supernode vertices
        for sn_id, supernode in self.supernodes.items():
            G_prime.add_node(sn_id, supernode=supernode)

        # Add edges between supernodes if inter-cluster connections exist
        edge_weights = defaultdict(list)

        for u, v, data in self.graph.edges(data=True):
            sn_u = self.node_to_supernode.get(u)
            sn_v = self.node_to_supernode.get(v)

            # Inter-cluster edge
            if sn_u is not None and sn_v is not None and sn_u != sn_v:
                # Get embeddings for edge weight
                emb_u = self._get_node_embedding(u)
                emb_v = self._get_node_embedding(v)
                similarity = self._cosine_similarity(emb_u, emb_v)

                edge_weights[(sn_u, sn_v)].append(similarity)

        # Add edges with average similarity as weight
        # Convert similarity (-1 to 1) to distance (0 to 2) for shortest path algorithms
        for (sn_u, sn_v), sims in edge_weights.items():
            avg_similarity = np.mean(sims)
            # Convert to distance: higher similarity = lower distance
            # distance = 1 - similarity maps [-1,1] to [0,2]
            distance = 1.0 - avg_similarity
            G_prime.add_edge(sn_u, sn_v, weight=distance)

        self.supergraph = G_prime

        logger.info(f"✅ Built supergraph: {G_prime.number_of_nodes()} supernodes, "
                   f"{G_prime.number_of_edges()} super-edges")

        return G_prime

    def find_paths(self, query: str, vq: List[str]) -> List[Path]:
        """
        Step 4-5: Main path finding algorithm.

        For each unique pair (start, end) in Vq:
        - Determine if same or different supernodes
        - Run resource propagation
        - Enumerate paths via beam search
        - Score and rank paths

        Args:
            query: The search query
            vq: List of retrieved node IDs

        Returns:
            Top K paths ranked by score
        """
        logger.info(f"🔍 Step 4-5: Finding paths for {len(vq)} retrieved nodes...")

        # Ensure supernodes and supergraph exist
        if not self.supernodes:
            self.precompute_supernodes()
        if self.supergraph is None:
            self.build_supergraph()

        # Enhance Vq with query matching
        vq_enhanced = self.enhance_retrieved_nodes(query, vq)

        # Generate unique pairs
        pairs = []
        for i, start in enumerate(vq_enhanced):
            for end in vq_enhanced[i+1:]:
                if start != end and start in self.graph.nodes and end in self.graph.nodes:
                    pairs.append((start, end))

        logger.info(f"Processing {len(pairs)} node pairs...")

        # Collect all paths from all pairs
        all_paths = []

        for start, end in pairs[:100]:  # Limit pairs for efficiency
            paths = self._find_paths_for_pair(start, end)
            all_paths.extend(paths)

        # Global ranking: select top K paths
        all_paths.sort(key=lambda p: p.score, reverse=True)
        top_paths = all_paths[:self.config.K]

        # Convert to textual representation
        for path in top_paths:
            path.text_representation = self._path_to_text(path)

        logger.info(f"✅ Found {len(top_paths)} top paths (from {len(all_paths)} total)")

        return top_paths

    def _find_paths_for_pair(self, start: str, end: str) -> List[Path]:
        """
        Find paths for a specific node pair.

        Args:
            start: Start node ID
            end: End node ID

        Returns:
            List of Path objects
        """
        # Check if same supernode
        sn_start = self.node_to_supernode.get(start)
        sn_end = self.node_to_supernode.get(end)

        if sn_start == sn_end:
            # Same supernode: propagate on local subgraph
            return self._propagate_local(start, end, sn_start)
        else:
            # Different supernodes: propagate on supergraph then refine
            return self._propagate_hierarchical(start, end, sn_start, sn_end)

    def _propagate_local(self, start: str, end: str, supernode_id: int) -> List[Path]:
        """
        Propagate resources on local subgraph (same supernode).

        Args:
            start: Start node
            end: End node
            supernode_id: Supernode ID

        Returns:
            List of paths
        """
        # Get local subgraph
        supernode = self.supernodes[supernode_id]
        subgraph = self.graph.subgraph(supernode.member_nodes)

        # Resource propagation
        resource_scores = self._resource_propagation(subgraph, start, end)

        # Beam search for paths
        paths = self._beam_search_paths(subgraph, start, end, resource_scores)

        return paths

    def _propagate_hierarchical(self, start: str, end: str,
                                sn_start: int, sn_end: int) -> List[Path]:
        """
        Propagate on supergraph then refine to node-level paths.

        Args:
            start: Start node
            end: End node
            sn_start: Start supernode ID
            sn_end: End supernode ID

        Returns:
            List of paths
        """
        # Find superpath on supergraph
        try:
            superpath = nx.shortest_path(self.supergraph, sn_start, sn_end, weight='weight')
        except nx.NetworkXNoPath:
            return []

        # Expand superpath to node-level path
        # For each supernode in path, find best node-level connections
        expanded_paths = self._expand_superpath(start, end, superpath)

        return expanded_paths

    def _resource_propagation(self, graph: nx.Graph, start: str, end: str) -> Dict[str, float]:
        """
        Perform resource propagation on graph.

        Uses equation: S(v_i) = sum over predecessors: alpha * S(v_j) / outdegree(v_j)
        Prunes nodes where S(v_i) / outdegree(v_i) < theta

        Args:
            graph: NetworkX graph
            start: Start node
            end: End node

        Returns:
            Dictionary mapping node IDs to resource scores
        """
        # Initialize resources
        S = {node: 0.0 for node in graph.nodes()}
        S[start] = 1.0

        # Iterative propagation (2-3 iterations)
        for iteration in range(self.config.max_propagation_steps):
            S_new = {node: 0.0 for node in graph.nodes()}
            S_new[start] = 1.0  # Keep source at 1.0

            for node in graph.nodes():
                if node == start:
                    continue

                # Sum contributions from predecessors
                for pred in graph.predecessors(node) if graph.is_directed() else graph.neighbors(node):
                    outdegree = graph.out_degree(pred) if graph.is_directed() else graph.degree(pred)
                    if outdegree > 0:
                        S_new[node] += self.config.alpha * S[pred] / outdegree

            S = S_new

            # Pruning: remove low-resource nodes
            nodes_to_remove = []
            for node in graph.nodes():
                if node not in [start, end]:
                    outdegree = graph.out_degree(node) if graph.is_directed() else graph.degree(node)
                    if outdegree > 0 and S[node] / outdegree < self.config.theta:
                        nodes_to_remove.append(node)

            # Don't actually remove from graph (needed for structure), just mark with 0 score
            for node in nodes_to_remove:
                S[node] = 0.0

        return S

    def _beam_search_paths(self, graph: nx.Graph, start: str, end: str,
                          resource_scores: Dict[str, float]) -> List[Path]:
        """
        Enumerate paths using beam search with greedy backtracking.

        Args:
            graph: NetworkX graph
            start: Start node
            end: End node
            resource_scores: Resource scores for nodes

        Returns:
            List of top beam_k paths
        """
        # Beam search from end node, backtracking to start
        # Track paths by (current_node, path_so_far, score)

        beam = [(end, [end], 0.0)]  # (node, path, cumulative_score)
        complete_paths = []

        for step in range(self.config.max_path_length):
            if not beam:
                break

            new_beam = []

            for current_node, path, cum_score in beam:
                # Check if reached start
                if current_node == start:
                    complete_paths.append((path[::-1], cum_score))  # Reverse path
                    continue

                # Expand to predecessors
                neighbors = list(graph.neighbors(current_node))

                for pred in neighbors:
                    if pred not in path:  # Avoid cycles
                        pred_score = resource_scores.get(pred, 0.0)
                        new_score = cum_score + pred_score
                        new_path = path + [pred]
                        new_beam.append((pred, new_path, new_score))

            # Keep top beam_k candidates
            new_beam.sort(key=lambda x: x[2], reverse=True)
            beam = new_beam[:self.config.beam_k]

        # Convert to Path objects and score
        paths = []
        for path_nodes, raw_score in complete_paths[:self.config.beam_k]:
            path_obj = self._create_path_object(path_nodes, resource_scores)
            paths.append(path_obj)

        return paths

    def _expand_superpath(self, start: str, end: str,
                         superpath: List[int]) -> List[Path]:
        """
        Expand a supernode path to node-level paths.

        Args:
            start: Start node
            end: End node
            superpath: List of supernode IDs

        Returns:
            List of expanded paths
        """
        # For each adjacent pair of supernodes, find best inter-supernode edges
        node_path = [start]
        current = start

        for i in range(len(superpath) - 1):
            sn_current = superpath[i]
            sn_next = superpath[i + 1]

            # Find best edge from current supernode to next
            best_next = None
            best_score = -1

            supernode_next = self.supernodes[sn_next]

            # Try context node first
            context = supernode_next.context_node
            if self.graph.has_edge(current, context):
                best_next = context
                best_score = 1.0
            else:
                # Find any connection
                for candidate in supernode_next.member_nodes:
                    if self.graph.has_edge(current, candidate):
                        # Score by PageRank
                        score = supernode_next.pagerank_scores.get(candidate, 0)
                        if score > best_score:
                            best_score = score
                            best_next = candidate

            if best_next:
                node_path.append(best_next)
                current = best_next

        # Ensure end node is included
        if node_path[-1] != end and self.graph.has_edge(node_path[-1], end):
            node_path.append(end)

        # Compute resource scores for this path
        subgraph = self.graph.subgraph(node_path)
        resource_scores = self._resource_propagation(subgraph, start, end)

        # Create path object
        path = self._create_path_object(node_path, resource_scores)

        return [path]

    def _create_path_object(self, nodes: List[str],
                           resource_scores: Dict[str, float]) -> Path:
        """
        Create a Path object from node list and scores.

        Args:
            nodes: List of node IDs in path
            resource_scores: Resource scores

        Returns:
            Path object with computed score
        """
        # Build edges
        edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes) - 1)]

        # Compute score: (1/|E|) * sum S(v_i) + context_bonus
        if len(edges) == 0:
            avg_score = resource_scores.get(nodes[0], 0)
        else:
            avg_score = sum(resource_scores.get(n, 0) for n in nodes) / len(edges)

        # Context bonus
        context_nodes = set()
        for node in nodes:
            sn_id = self.node_to_supernode.get(node)
            if sn_id is not None:
                context = self.supernodes[sn_id].context_node
                if node == context:
                    context_nodes.add(node)

        context_bonus = len(context_nodes) * self.config.context_bonus
        final_score = avg_score + context_bonus

        return Path(
            nodes=nodes,
            edges=edges,
            score=final_score,
            context_nodes=context_nodes,
            resource_scores={n: resource_scores.get(n, 0) for n in nodes}
        )

    def _path_to_text(self, path: Path) -> str:
        """
        Convert path to textual representation for prompting.

        Args:
            path: Path object

        Returns:
            Textual representation
        """
        text_parts = []

        for i, node in enumerate(path.nodes):
            # Get node information
            node_data = self.graph.nodes.get(node, {})
            node_desc = node_data.get('description', node_data.get('text', ''))
            node_type = node_data.get('type', 'entity')

            # Check if context node
            is_context = node in path.context_nodes
            context_marker = " [CONTEXT]" if is_context else ""

            text_parts.append(f"{node} ({node_type}){context_marker}: {node_desc[:100]}")

            # Add edge description
            if i < len(path.edges):
                edge = path.edges[i]
                edge_data = self.graph.get_edge_data(edge[0], edge[1], {})
                edge_desc = edge_data.get('description', 'related to')
                text_parts.append(f"  → {edge_desc[:80]}")

        # Add supernode summaries for context nodes
        summary_texts = []
        for node in path.context_nodes:
            sn_id = self.node_to_supernode.get(node)
            if sn_id is not None:
                summary = self.supernodes[sn_id].summary
                summary_texts.append(f"Context: {summary[:150]}")

        full_text = "\n".join(text_parts)
        if summary_texts:
            full_text += "\n\n" + "\n".join(summary_texts)

        return full_text

    def _get_node_embedding(self, node_id: str) -> np.ndarray:
        """Get or compute embedding for a node."""
        if node_id in self.node_embeddings:
            return self.node_embeddings[node_id]

        node_data = self.graph.nodes.get(node_id, {})
        text = node_data.get('description', node_data.get('text', node_id))
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)

        self.node_embeddings[node_id] = embedding
        return embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def execute(self, query: str, retrieved_nodes: List[str]) -> Dict:
        """
        Execute complete HFBP algorithm.

        Args:
            query: Search query
            retrieved_nodes: Initially retrieved node IDs

        Returns:
            Dictionary with paths, scores, and debug information
        """
        logger.info("="*80)
        logger.info("🚀 Executing Hierarchical Flow-Based Pruning (HFBP)")
        logger.info("="*80)

        start_time = datetime.now()

        # Step 1: Precompute supernodes
        self.precompute_supernodes()

        # Step 2-5: Find paths
        paths = self.find_paths(query, retrieved_nodes)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Prepare output
        result = {
            'query': query,
            'num_retrieved_nodes': len(retrieved_nodes),
            'num_supernodes': len(self.supernodes),
            'num_paths_found': len(paths),
            'top_k_paths': paths,
            'execution_time_seconds': duration,
            'config': {
                'N': self.config.N,
                'K': self.config.K,
                'alpha': self.config.alpha,
                'theta': self.config.theta,
                'beam_k': self.config.beam_k
            },
            'debug': {
                'graph_nodes': len(self.graph.nodes),
                'graph_edges': len(self.graph.edges),
                'supergraph_nodes': self.supergraph.number_of_nodes() if self.supergraph else 0,
                'supergraph_edges': self.supergraph.number_of_edges() if self.supergraph else 0
            }
        }

        logger.info(f"✅ HFBP execution completed in {duration:.2f}s")
        logger.info(f"📊 Found {len(paths)} paths from {len(retrieved_nodes)} initial nodes")
        logger.info("="*80)

        return result

    def save_results(self, result: Dict, output_path: Union[str, FilePath]):
        """
        Save HFBP results to file.

        Args:
            result: Result dictionary from execute()
            output_path: Path to save results
        """
        output_path = FilePath(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert paths to serializable format
        serializable_result = result.copy()
        serializable_result['top_k_paths'] = [
            {
                'nodes': p.nodes,
                'edges': p.edges,
                'score': p.score,
                'text': p.text_representation,
                'context_nodes': list(p.context_nodes),
                'resource_scores': p.resource_scores
            }
            for p in result['top_k_paths']
        ]

        with open(output_path, 'w') as f:
            json.dump(serializable_result, f, indent=2, default=str)

        logger.info(f"💾 Results saved to {output_path}")


if __name__ == "__main__":
    print("HFBP Pruning module loaded successfully!")
    print("Use HFBPPruning class to implement hierarchical flow-based path retrieval.")
