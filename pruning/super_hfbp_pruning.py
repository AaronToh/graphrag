#!/usr/bin/env python3
"""
SuperHFBP: Enhanced Hierarchical Flow-Based Pruning with Context Nodes

Extends HFBP with:
- Context node identification and forcing in paths
- Enhanced inter-community edge modeling (density threshold)
- Global query detection for supernode-level retrieval
- Improved reliability scoring with context bonus

Based on SuperPathRAG logic for improved relevance and token efficiency.

Author: GraphRAG Pruning Lab
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from pathlib import Path as FilePath
import json

# Handle both package import and direct script execution
try:
    from .hfbp_pruning import (
        HFBPPruning,
        HFBPConfig,
        Supernode,
        Path as HFBPPath
    )
except ImportError:
    from hfbp_pruning import (
        HFBPPruning,
        HFBPConfig,
        Supernode,
        Path as HFBPPath
    )

# Optional LLM for global query detection
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available - global query detection will use heuristics")

logger = logging.getLogger(__name__)


@dataclass
class SuperHFBPConfig(HFBPConfig):
    """Enhanced configuration for SuperHFBP algorithm."""
    # Context node parameters
    beta: float = 0.2  # Context bonus weight in reliability scoring
    context_top_pct: float = 0.2  # Top percentage of nodes to consider as context
    context_penalty: float = 0.8  # Penalty multiplier for paths without context nodes

    # Inter-community edge threshold
    inter_community_density_threshold: float = 0.05  # 5% minimum density

    # Global query detection
    use_llm_global_detection: bool = True  # Use LLM for global query detection
    global_query_keywords: List[str] = field(default_factory=lambda: [
        'summarize', 'overview', 'trends', 'patterns', 'all', 'general',
        'comprehensive', 'broad', 'survey', 'review'
    ])

    # OpenAI settings for global query detection
    openai_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None


class SuperHFBP(HFBPPruning):
    """
    SuperHFBP: Enhanced HFBP with context nodes and refined hierarchical pruning.

    Enhancements:
    1. Context node identification: Top-K% nodes by query similarity
    2. Global query detection: Determines if query needs supernode-level context
    3. Enhanced supergraph: Inter-community edges with density threshold
    4. Context forcing: Paths must include context nodes or receive penalty
    5. Improved reliability scoring: S(P) includes context node bonus
    """

    def __init__(self,
                 entities_df: pd.DataFrame,
                 relationships_df: pd.DataFrame,
                 text_units_df: pd.DataFrame = None,
                 config: SuperHFBPConfig = None):
        """
        Initialize SuperHFBP.

        Args:
            entities_df: GraphRAG entities DataFrame
            relationships_df: GraphRAG relationships DataFrame
            text_units_df: GraphRAG text units DataFrame (optional)
            config: SuperHFBP configuration
        """
        # Use SuperHFBPConfig if not provided
        if config is None:
            config = SuperHFBPConfig()

        # Initialize parent HFBP
        super().__init__(entities_df, relationships_df, text_units_df, config)

        # SuperHFBP-specific structures
        self.context_nodes: Set[str] = set()
        self.is_global_query: bool = False
        self.node_query_similarities: Dict[str, float] = {}

        # Configure OpenAI if available
        if OPENAI_AVAILABLE and config.openai_api_key:
            openai.api_key = config.openai_api_key

        logger.info("Initialized SuperHFBP with context node enhancement")

    def identify_context_nodes(self, query: str, vq: List[str]) -> Set[str]:
        """
        Identify context nodes from retrieved set Vq.

        Context nodes are:
        1. Top-K% nodes by query similarity (using embeddings)
        2. Supernode representatives if query is global

        Args:
            query: Search query
            vq: Retrieved node IDs

        Returns:
            Set of context node IDs
        """
        logger.info(f"🔍 Identifying context nodes from {len(vq)} retrieved nodes...")

        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        # Compute similarity for all nodes in Vq
        similarities = []
        for node_id in vq:
            if node_id not in self.graph.nodes:
                continue

            node_emb = self._get_node_embedding(node_id)
            sim = self._cosine_similarity(query_embedding, node_emb)
            similarities.append((node_id, sim))
            self.node_query_similarities[node_id] = sim

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Select top-K%
        k_count = max(1, int(len(similarities) * self.config.context_top_pct))
        context_nodes = {node_id for node_id, _ in similarities[:k_count]}

        # Check if global query
        self.is_global_query = self._detect_global_query(query)

        # If global query, add supernode context nodes
        if self.is_global_query:
            logger.info("🌐 Global query detected - adding supernode representatives")
            # Ensure supernodes exist
            if not self.supernodes:
                self.precompute_supernodes()

            # Add context node from each supernode that has members in Vq
            vq_supernodes = set()
            for node in vq:
                if node in self.node_to_supernode:
                    vq_supernodes.add(self.node_to_supernode[node])

            for sn_id in vq_supernodes:
                context_nodes.add(self.supernodes[sn_id].context_node)

        self.context_nodes = context_nodes

        logger.info(f"✅ Identified {len(context_nodes)} context nodes "
                   f"({'global' if self.is_global_query else 'local'} query)")

        return context_nodes

    def _detect_global_query(self, query: str) -> bool:
        """
        Detect if query is global (requires broad/summary context).

        Uses LLM if available, otherwise heuristic keyword matching.

        Args:
            query: Search query

        Returns:
            True if global query
        """
        query_lower = query.lower()

        # Heuristic: keyword matching
        keyword_match = any(kw in query_lower for kw in self.config.global_query_keywords)

        # If LLM available and configured, use it
        if (OPENAI_AVAILABLE and
            self.config.use_llm_global_detection and
            self.config.openai_api_key):
            try:
                return self._llm_detect_global_query(query)
            except Exception as e:
                logger.warning(f"LLM global detection failed: {e}, using heuristic")
                return keyword_match

        return keyword_match

    def _llm_detect_global_query(self, query: str) -> bool:
        """
        Use LLM to detect global query intent.

        Args:
            query: Search query

        Returns:
            True if global query
        """
        prompt = f"""Classify if the following query requires a broad, comprehensive, or summary-level answer (global) versus a specific, focused answer (local).

Query: "{query}"

Answer with only "global" or "local"."""

        response = openai.ChatCompletion.create(
            model=self.config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )

        answer = response.choices[0].message.content.strip().lower()
        return 'global' in answer

    def build_supergraph(self) -> nx.Graph:
        """
        Enhanced supergraph construction with inter-community density threshold.

        Only adds inter-supernode edges if inter-community density > threshold.

        Returns:
            Enhanced supergraph
        """
        logger.info("🔍 Step 3: Building enhanced supergraph with density threshold...")

        # Ensure supernodes exist
        if not self.supernodes:
            self.precompute_supernodes()

        G_prime = nx.Graph()

        # Add supernode vertices
        for sn_id, supernode in self.supernodes.items():
            G_prime.add_node(sn_id, supernode=supernode)

        # Compute inter-community connections with density check
        edge_data = defaultdict(lambda: {'weights': [], 'edge_count': 0})

        for u, v, data in self.graph.edges(data=True):
            sn_u = self.node_to_supernode.get(u)
            sn_v = self.node_to_supernode.get(v)

            # Inter-cluster edge
            if sn_u is not None and sn_v is not None and sn_u != sn_v:
                key = tuple(sorted([sn_u, sn_v]))

                # Get embeddings for edge weight
                emb_u = self._get_node_embedding(u)
                emb_v = self._get_node_embedding(v)
                similarity = self._cosine_similarity(emb_u, emb_v)

                edge_data[key]['weights'].append(similarity)
                edge_data[key]['edge_count'] += 1

        # Add edges only if density threshold met
        edges_added = 0
        edges_filtered = 0

        for (sn_u, sn_v), data in edge_data.items():
            # Calculate density: edge_count / (|members_u| * |members_v|)
            size_u = len(self.supernodes[sn_u].member_nodes)
            size_v = len(self.supernodes[sn_v].member_nodes)
            max_possible_edges = size_u * size_v
            density = data['edge_count'] / max_possible_edges if max_possible_edges > 0 else 0

            if density >= self.config.inter_community_density_threshold:
                avg_similarity = np.mean(data['weights'])
                distance = 1.0 - avg_similarity
                G_prime.add_edge(sn_u, sn_v,
                               weight=distance,
                               density=density,
                               inter_edge_count=data['edge_count'])
                edges_added += 1
            else:
                edges_filtered += 1

        self.supergraph = G_prime

        logger.info(f"✅ Built enhanced supergraph: {G_prime.number_of_nodes()} supernodes, "
                   f"{edges_added} super-edges (filtered {edges_filtered} low-density edges)")

        return G_prime

    def find_paths(self, query: str, vq: List[str]) -> List[HFBPPath]:
        """
        Enhanced path finding with context node forcing.

        Args:
            query: Search query
            vq: Retrieved node IDs

        Returns:
            Top K paths with context node consideration
        """
        logger.info(f"🔍 Step 4-5: Finding paths with context node forcing...")

        # Ensure supernodes and supergraph exist
        if not self.supernodes:
            self.precompute_supernodes()
        if self.supergraph is None:
            self.build_supergraph()

        # Enhance Vq with query matching
        vq_enhanced = self.enhance_retrieved_nodes(query, vq)

        # Identify context nodes
        self.identify_context_nodes(query, vq_enhanced)

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

        # Re-score paths with context node bonus/penalty
        for path in all_paths:
            path.score = self._compute_enhanced_score(path)

        # Global ranking: select top K paths
        all_paths.sort(key=lambda p: p.score, reverse=True)
        top_paths = all_paths[:self.config.K]

        # Convert to textual representation with supernode summaries
        for path in top_paths:
            path.text_representation = self._enhanced_path_to_text(path)

        logger.info(f"✅ Found {len(top_paths)} top paths (from {len(all_paths)} total)")

        return top_paths

    def _compute_enhanced_score(self, path: HFBPPath) -> float:
        """
        Compute enhanced reliability score with context node bonus.

        S(P) = (1/|E_P|) * sum_{v in V_P} S(v) + beta * (|C_q ∩ V_P| / |C_q|)

        If path contains no context nodes, apply penalty.

        Args:
            path: Path object

        Returns:
            Enhanced score
        """
        # Base score: average resource score
        if len(path.edges) == 0:
            base_score = path.resource_scores.get(path.nodes[0], 0) if path.nodes else 0
        else:
            base_score = sum(path.resource_scores.get(n, 0) for n in path.nodes) / len(path.edges)

        # Context node bonus
        if len(self.context_nodes) > 0:
            path_context_nodes = set(path.nodes) & self.context_nodes
            context_ratio = len(path_context_nodes) / len(self.context_nodes)
            context_bonus = self.config.beta * context_ratio

            # Apply penalty if no context nodes
            if len(path_context_nodes) == 0:
                base_score *= self.config.context_penalty
        else:
            context_bonus = 0

        return base_score + context_bonus

    def _enhanced_path_to_text(self, path: HFBPPath) -> str:
        """
        Enhanced textual representation with supernode summaries.

        Format: t_P = concat([t_s for supernodes; ...; t_v; t_e; ...])

        Args:
            path: Path object

        Returns:
            Enhanced textual representation
        """
        text_parts = []

        # Add supernode summaries for supernodes in path
        supernodes_in_path = set()
        for node in path.nodes:
            sn_id = self.node_to_supernode.get(node)
            if sn_id is not None:
                supernodes_in_path.add(sn_id)

        # Supernode context
        if supernodes_in_path:
            text_parts.append("=== Supernode Context ===")
            for sn_id in sorted(supernodes_in_path):
                summary = self.supernodes[sn_id].summary
                text_parts.append(f"Supernode {sn_id}: {summary}")
            text_parts.append("")

        # Path details
        text_parts.append("=== Path Details ===")
        for i, node in enumerate(path.nodes):
            # Get node information
            node_data = self.graph.nodes.get(node, {})
            node_desc = node_data.get('description', node_data.get('text', ''))
            node_type = node_data.get('type', 'entity')

            # Check if context node
            is_context = node in self.context_nodes
            context_marker = " [CONTEXT]" if is_context else ""

            # Check if supernode representative
            sn_id = self.node_to_supernode.get(node)
            is_sn_rep = (sn_id is not None and
                        self.supernodes[sn_id].context_node == node)
            rep_marker = " [SUPERNODE_REP]" if is_sn_rep else ""

            text_parts.append(
                f"{node} ({node_type}){context_marker}{rep_marker}: {node_desc[:100]}"
            )

            # Add edge description
            if i < len(path.edges):
                edge = path.edges[i]
                edge_data = self.graph.get_edge_data(edge[0], edge[1], {})
                edge_desc = edge_data.get('description', 'related to')
                text_parts.append(f"  → {edge_desc[:80]}")

        return "\n".join(text_parts)

    def export_pruned_graph(self, paths: List[HFBPPath], output_dir: Union[str, FilePath]):
        """
        Export pruned graph from top-K paths as GraphRAG artifacts.

        Extracts all nodes and edges that appear in the top-K paths and saves them
        as pruned GraphRAG artifacts (entities.parquet, relationships.parquet, etc.).

        Args:
            paths: List of top-K paths from which to extract pruned graph
            output_dir: Directory to save pruned artifacts

        Returns:
            Dictionary with pruning statistics
        """
        output_dir = FilePath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔍 Extracting pruned graph from {len(paths)} paths...")

        # Collect all nodes and edges from paths
        pruned_node_ids = set()
        pruned_edges = set()

        for path in paths:
            pruned_node_ids.update(path.nodes)
            pruned_edges.update(path.edges)

        # Also include context nodes (important for SuperHFBP)
        pruned_node_ids.update(self.context_nodes)

        logger.info(f"📊 Pruned graph contains {len(pruned_node_ids)} nodes, {len(pruned_edges)} edges")

        # Filter entities DataFrame
        # Match by 'title' field (which we use as node_id in graph construction)
        pruned_entities = self.entities_df[
            self.entities_df['title'].astype(str).isin(pruned_node_ids)
        ].copy()
        # Ensure duplicates removed on identifier only to avoid ndarray hashing issues
        pruned_entities = pruned_entities.drop_duplicates(subset=['title'])

        # Filter relationships DataFrame
        # Match by source and target
        def is_edge_in_pruned(row):
            edge_tuple = (str(row['source']), str(row['target']))
            edge_tuple_rev = (str(row['target']), str(row['source']))
            return edge_tuple in pruned_edges or edge_tuple_rev in pruned_edges

        pruned_relationships = self.relationships_df[
            self.relationships_df.apply(is_edge_in_pruned, axis=1)
        ].copy()
        # Deduplicate on edge endpoints only
        pruned_relationships = pruned_relationships.drop_duplicates(subset=['source', 'target'])

        # Also keep edges between pruned nodes (even if not in paths)
        # This maintains graph connectivity
        def connects_pruned_nodes(row):
            return (str(row['source']) in pruned_node_ids and
                   str(row['target']) in pruned_node_ids)

        additional_edges = self.relationships_df[
            self.relationships_df.apply(connects_pruned_nodes, axis=1)
        ].copy()

        # Combine path edges and connecting edges
        pruned_relationships = pd.concat([pruned_relationships, additional_edges]).drop_duplicates(subset=['source', 'target'])

        # Filter text_units if available
        if self.text_units_df is not None:
            # Keep text units that reference pruned entities
            # This depends on your schema - adjust field names as needed
            pruned_text_units = self.text_units_df.copy()
        else:
            pruned_text_units = None

        # Save pruned artifacts
        logger.info(f"💾 Saving pruned artifacts to {output_dir}...")

        pruned_entities.to_parquet(output_dir / "entities.parquet", index=False)
        pruned_relationships.to_parquet(output_dir / "relationships.parquet", index=False)

        if pruned_text_units is not None:
            pruned_text_units.to_parquet(output_dir / "text_units.parquet", index=False)

        # Save pruning metadata
        metadata = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'pruning_method': 'SuperHFBP',
            'original_stats': {
                'num_entities': len(self.entities_df),
                'num_relationships': len(self.relationships_df),
                'num_text_units': len(self.text_units_df) if self.text_units_df is not None else 0
            },
            'pruned_stats': {
                'num_entities': len(pruned_entities),
                'num_relationships': len(pruned_relationships),
                'num_text_units': len(pruned_text_units) if pruned_text_units is not None else 0,
                'num_paths': len(paths),
                'num_context_nodes': len(self.context_nodes)
            },
            'reduction_rate': {
                'entities': 1.0 - (len(pruned_entities) / len(self.entities_df)) if len(self.entities_df) > 0 else 0,
                'relationships': 1.0 - (len(pruned_relationships) / len(self.relationships_df)) if len(self.relationships_df) > 0 else 0
            },
            'config': {
                'N': self.config.N,
                'K': self.config.K,
                'alpha': self.config.alpha,
                'theta': self.config.theta,
                'beta': self.config.beta,
                'context_top_pct': self.config.context_top_pct,
                'context_penalty': self.config.context_penalty,
                'inter_community_density_threshold': self.config.inter_community_density_threshold
            }
        }

        with open(output_dir / "pruning_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"✅ Pruned graph exported successfully!")
        logger.info(f"   Entities: {len(self.entities_df)} → {len(pruned_entities)} "
                   f"({100 * metadata['reduction_rate']['entities']:.1f}% reduction)")
        logger.info(f"   Relationships: {len(self.relationships_df)} → {len(pruned_relationships)} "
                   f"({100 * metadata['reduction_rate']['relationships']:.1f}% reduction)")

        return metadata

    def execute(self, query: str, retrieved_nodes: List[str]) -> Dict:
        """
        Execute complete SuperHFBP algorithm.

        Args:
            query: Search query
            retrieved_nodes: Initially retrieved node IDs

        Returns:
            Dictionary with paths, scores, and debug information
        """
        logger.info("="*80)
        logger.info("🚀 Executing SuperHFBP (Enhanced Context-Aware Pruning)")
        logger.info("="*80)

        from datetime import datetime
        start_time = datetime.now()

        # Step 1: Precompute supernodes
        self.precompute_supernodes()

        # Step 2-5: Find paths with context forcing
        paths = self.find_paths(query, retrieved_nodes)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Prepare output
        result = {
            'query': query,
            'is_global_query': self.is_global_query,
            'num_retrieved_nodes': len(retrieved_nodes),
            'num_context_nodes': len(self.context_nodes),
            'context_nodes': list(self.context_nodes),
            'num_supernodes': len(self.supernodes),
            'num_paths_found': len(paths),
            'top_k_paths': paths,
            'execution_time_seconds': duration,
            'config': {
                'N': self.config.N,
                'K': self.config.K,
                'alpha': self.config.alpha,
                'theta': self.config.theta,
                'beta': self.config.beta,
                'context_top_pct': self.config.context_top_pct,
                'context_penalty': self.config.context_penalty,
                'inter_community_density_threshold': self.config.inter_community_density_threshold,
                'beam_k': self.config.beam_k
            },
            'debug': {
                'graph_nodes': len(self.graph.nodes),
                'graph_edges': len(self.graph.edges),
                'supergraph_nodes': self.supergraph.number_of_nodes() if self.supergraph else 0,
                'supergraph_edges': self.supergraph.number_of_edges() if self.supergraph else 0
            }
        }

        logger.info(f"✅ SuperHFBP execution completed in {duration:.2f}s")
        logger.info(f"📊 Found {len(paths)} paths from {len(retrieved_nodes)} initial nodes")
        logger.info(f"🎯 Context nodes: {len(self.context_nodes)} ({'global' if self.is_global_query else 'local'})")
        logger.info("="*80)

        return result


if __name__ == "__main__":
    print("SuperHFBP module loaded successfully!")
    print("Use SuperHFBP class for enhanced context-aware hierarchical path retrieval.")
