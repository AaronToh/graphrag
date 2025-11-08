#!/usr/bin/env python3
"""
KGTrimmer Pruning Implementation (optimized)

Evaluates node importance from collective (community) and holistic (global) perspectives
with vectorised scoring and single-pass pruning for faster execution.
"""

import json
import logging
from typing import Dict, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class KGTrimmerPruner:
    """Fast KGTrimmer implementation with cached feature matrix."""

    def __init__(
        self,
        graph: nx.DiGraph,
        entities_df: pd.DataFrame,
        communities_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.graph = graph.copy()
        self.entities_df = entities_df
        self.communities_df = communities_df
        self._features = self._build_feature_frame()

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------
    def _build_feature_frame(self) -> pd.DataFrame:
        nodes = list(self.graph.nodes())
        df = pd.DataFrame(index=nodes, dtype=float)

        # Degree based metrics
        degree_series = pd.Series(dict(self.graph.degree()), dtype=float)
        df["degree"] = degree_series.reindex(nodes, fill_value=0.0)
        max_degree = max(df["degree"].max(), 1.0)
        df["degree_norm"] = df["degree"] / max_degree

        avg_neighbor_degree = pd.Series(nx.average_neighbor_degree(self.graph))
        df["avg_neighbor_degree"] = avg_neighbor_degree.reindex(nodes, fill_value=0.0)
        max_neighbor = max(df["avg_neighbor_degree"].max(), 1.0)
        df["avg_neighbor_norm"] = df["avg_neighbor_degree"] / max_neighbor

        # Frequency / semantic proxy
        freq_map: Dict[str, float] = {}
        if "title" in self.entities_df.columns:
            freq_series = (
                self.entities_df.assign(
                    title=self.entities_df["title"].astype(str),
                    frequency=self.entities_df.get("frequency", 0.0).astype(float),
                )
                .groupby("title")["frequency"]
                .max()
            )
            freq_map = freq_series.to_dict()
        df["frequency"] = [float(freq_map.get(str(node), 0.0)) for node in nodes]
        max_freq = max(df["frequency"].max(), 1.0)
        df["frequency_norm"] = df["frequency"] / max_freq

        # Pagerank (single run)
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85, max_iter=50, tol=1e-6)
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("PageRank failed (%s); using zeros", exc)
            pagerank = {node: 0.0 for node in nodes}
        df["pagerank"] = pd.Series(pagerank).reindex(nodes, fill_value=0.0)
        max_pr = max(df["pagerank"].max(), 1.0)
        df["pagerank_norm"] = df["pagerank"] / max_pr

        # Community metrics (size + bridge flag)
        df["community_size"] = 0.0
        df["community_bridge_flag"] = 0.0
        if self.communities_df is not None and len(self.communities_df) > 0:
            id_to_title = (
                self.entities_df.assign(
                    id=self.entities_df["id"].astype(str),
                    title=self.entities_df.get("title", "").astype(str),
                )
                .set_index("id")["title"]
                .to_dict()
            )
            node_to_comms: Dict[str, Set[str]] = {}
            comm_sizes: Dict[str, int] = {}

            for _, row in self.communities_df.iterrows():
                comm_id = row.get("id")
                entity_ids = row.get("entity_ids", [])
                if isinstance(entity_ids, str):
                    try:
                        entity_ids = json.loads(entity_ids)
                    except json.JSONDecodeError:
                        entity_ids = []
                if not isinstance(entity_ids, (list, tuple)):
                    continue
                for ent_id in entity_ids:
                    node_title = id_to_title.get(str(ent_id))
                    if not node_title:
                        continue
                    node_to_comms.setdefault(node_title, set()).add(comm_id)
                    comm_sizes[comm_id] = comm_sizes.get(comm_id, 0) + 1

            for node, comm_ids in node_to_comms.items():
                size = max((comm_sizes.get(cid, 0) for cid in comm_ids), default=0)
                if node in df.index:
                    df.at[node, "community_size"] = float(size)
                    if len(comm_ids) > 1:
                        df.at[node, "community_bridge_flag"] = 1.0

        max_comm = max(df["community_size"].max(), 1.0)
        df["community_size_norm"] = df["community_size"] / max_comm

        # Collective and holistic proxies
        df["collective_score"] = 0.6 * df["community_size_norm"] + 0.4 * df["avg_neighbor_norm"]
        max_coll = max(df["collective_score"].max(), 1.0)
        df["collective_score"] = df["collective_score"] / max_coll

        df["holistic_score"] = (
            0.6 * df["degree_norm"]
            + 0.25 * df["pagerank_norm"]
            + 0.15 * df["frequency_norm"]
        )
        max_hol = max(df["holistic_score"].max(), 1.0)
        df["holistic_score"] = df["holistic_score"] / max_hol

        df.fillna(0.0, inplace=True)
        return df

    # ------------------------------------------------------------------
    # Scoring utilities
    # ------------------------------------------------------------------
    def _score_collective_importance(self) -> Dict[str, float]:
        return self._features["collective_score"].to_dict()

    def _score_holistic_importance(self) -> Dict[str, float]:
        return self._features["holistic_score"].to_dict()

    def _compute_combined_scores(
        self,
        collective_weight: float = 0.5,
        holistic_weight: float = 0.5,
    ) -> pd.Series:
        total = collective_weight + holistic_weight
        if total <= 0:
            collective_weight = holistic_weight = 0.5
            total = 1.0
        collective_weight /= total
        holistic_weight /= total
        combined = (
            collective_weight * self._features["collective_score"]
            + holistic_weight * self._features["holistic_score"]
        )
        return combined.sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Connectivity helpers
    # ------------------------------------------------------------------
    def _component_ratio(self, nodes: Set[str]) -> float:
        if not nodes:
            return 0.0
        subgraph = self.graph.subgraph(nodes)
        if subgraph.number_of_nodes() == 0:
            return 0.0
        components = (
            nx.weakly_connected_components(subgraph)
            if subgraph.is_directed()
            else nx.connected_components(subgraph)
        )
        components = list(components)
        if not components:
            return 0.0
        largest = max(components, key=len)
        return len(largest) / subgraph.number_of_nodes()

    def _ensure_connectivity(
        self,
        keep_nodes: Set[str],
        ordered_scores: pd.Series,
        threshold: float = 0.82,
        max_extra_ratio: float = 0.08,
    ) -> Set[str]:
        keep_nodes = set(keep_nodes)
        threshold = max(threshold, 0.5)
        ratio = self._component_ratio(keep_nodes)
        if ratio >= threshold:
            return keep_nodes

        candidates = [node for node in ordered_scores.index if node not in keep_nodes]
        max_extra = max(1, int(len(self.graph) * max_extra_ratio))
        added = 0
        for node in candidates:
            keep_nodes.add(node)
            added += 1
            if added % 5 == 0 or added >= max_extra:
                ratio = self._component_ratio(keep_nodes)
                if ratio >= threshold or added >= max_extra:
                    break
        return keep_nodes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prune(
        self,
        collective_weight: float = 0.5,
        holistic_weight: float = 0.5,
        min_importance_percentile: float = 0.45,
        preserve_connectivity: bool = True,
        max_iterations: int = 1,  # retained for API compatibility
    ) -> nx.DiGraph:
        if self.graph.number_of_nodes() == 0:
            return self.graph.copy()

        ordered_scores = self._compute_combined_scores(
            collective_weight, holistic_weight
        )
        if ordered_scores.empty:
            return self.graph.copy()

        keep_count = max(1, int(len(ordered_scores) * min_importance_percentile))
        base_nodes = set(ordered_scores.head(keep_count).index)

        # Always keep bridge nodes and top hubs
        bridge_nodes = set(
            self._features[self._features["community_bridge_flag"] > 0.0].index
        )
        hub_nodes = set(
            self._features[self._features["degree_norm"] >= 0.95].index
        )
        keep_nodes = base_nodes | bridge_nodes | hub_nodes

        if preserve_connectivity:
            keep_nodes = self._ensure_connectivity(keep_nodes, ordered_scores)

        pruned_graph = self.graph.subgraph(keep_nodes).copy()
        isolates = list(nx.isolates(pruned_graph))
        if isolates:
            pruned_graph.remove_nodes_from(isolates)
        return pruned_graph


def kgtrimmer_prune(
    graph: nx.DiGraph,
    entities_df: pd.DataFrame,
    communities_df: Optional[pd.DataFrame] = None,
    collective_weight: float = 0.5,
    holistic_weight: float = 0.5,
    min_importance_percentile: float = 0.45,
    preserve_connectivity: bool = True,
    max_iterations: int = 1,
) -> nx.DiGraph:
    """Convenience wrapper for the KGTrimmerPruner."""
    pruner = KGTrimmerPruner(graph, entities_df, communities_df)
    return pruner.prune(
        collective_weight=collective_weight,
        holistic_weight=holistic_weight,
        min_importance_percentile=min_importance_percentile,
        preserve_connectivity=preserve_connectivity,
        max_iterations=max_iterations,
    )
