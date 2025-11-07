#!/usr/bin/env python3
"""
Adaptive Multi-Strategy Pruning (revamped)

Blends KGTrimmer scores, structural heuristics, and hybrid path signals to
produce aggressive-yet-safe graph pruning.
"""

import json
import logging
from typing import Dict, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd

try:  # Local package execution
    from .kgtrimmer import KGTrimmerPruner
except ImportError:  # pragma: no cover - direct script execution
    from kgtrimmer import KGTrimmerPruner

logger = logging.getLogger(__name__)


class AdaptiveMultiStrategyPruner:
    """Aggressive adaptive pruning that retains critical hubs and bridges."""

    def __init__(
        self,
        graph: nx.DiGraph,
        entities_df: pd.DataFrame,
        communities_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.graph = graph.copy()
        self.entities_df = entities_df
        self.communities_df = communities_df
        self.kg_helper = KGTrimmerPruner(self.graph, self.entities_df, self.communities_df)
        self.features = self._build_feature_frame()

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _build_feature_frame(self) -> pd.DataFrame:
        base = self.kg_helper._features.copy()
        nodes = base.index.tolist()

        # Map KG scores
        kg_scores = self.kg_helper._compute_combined_scores(0.5, 0.5)
        if kg_scores.max() > 0:
            kg_scores = kg_scores / kg_scores.max()
        base['kg_score'] = kg_scores.reindex(nodes, fill_value=0.0)

        # Extra structural signals
        try:
            undirected = self.graph.to_undirected()
            triangle_count = nx.triangles(undirected)
            base['triangle_norm'] = pd.Series(triangle_count).reindex(nodes, fill_value=0.0)
            tri_max = max(base['triangle_norm'].max(), 1.0)
            base['triangle_norm'] /= tri_max
        except Exception:  # pragma: no cover - optional optimisation
            base['triangle_norm'] = 0.0

        base['bridge_flag'] = base.get('community_bridge_flag', 0.0)

        # Normalise raw metrics to avoid division by zero
        def _norm(series_name: str) -> pd.Series:
            values = base.get(series_name, pd.Series(dtype=float))
            if values.empty:
                return pd.Series(0.0, index=base.index)
            max_val = max(values.max(), 1.0)
            return values / max_val

        degree_norm = _norm('degree')
        neighbor_norm = _norm('avg_neighbor_degree')
        pagerank_norm = _norm('pagerank')
        frequency_norm = _norm('frequency')
        triangle_norm = base['triangle_norm']

        base['adaptive_score'] = (
            0.55 * base['kg_score']
            + 0.15 * degree_norm
            + 0.10 * neighbor_norm
            + 0.08 * pagerank_norm
            + 0.07 * frequency_norm
            + 0.05 * triangle_norm
            + 0.10 * base['bridge_flag']
        )
        max_adaptive = max(base['adaptive_score'].max(), 1.0)
        base['adaptive_score'] = base['adaptive_score'] / max_adaptive

        return base

    # ------------------------------------------------------------------
    # Anchor + selection helpers
    # ------------------------------------------------------------------
    def _select_anchors(
        self,
        protected_fraction: float,
        hub_degree_percentile: float,
    ) -> Set[str]:
        features = self.features
        sorted_nodes = features['adaptive_score'].sort_values(ascending=False)
        top_k = max(1, int(len(sorted_nodes) * protected_fraction))
        anchors = set(sorted_nodes.head(top_k).index)

        degree_series = features.get('degree', pd.Series(dtype=float))
        if not degree_series.empty:
            threshold = np.percentile(degree_series, hub_degree_percentile * 100)
            anchors.update(degree_series[degree_series >= threshold].index)

        anchors.update(features[features['bridge_flag'] > 0].index)
        return anchors

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
        min_connectivity_pct: float,
        max_extra_ratio: float = 0.1,
    ) -> Set[str]:
        keep_nodes = set(keep_nodes)
        ratio = self._component_ratio(keep_nodes)
        if ratio >= min_connectivity_pct:
            return keep_nodes

        candidates = [node for node in ordered_scores.index if node not in keep_nodes]
        max_extra = max(1, int(len(self.graph) * max_extra_ratio))
        added = 0
        for node in candidates:
            keep_nodes.add(node)
            added += 1
            ratio = self._component_ratio(keep_nodes)
            if ratio >= min_connectivity_pct or added >= max_extra:
                break
        return keep_nodes

    def _refine_selection(
        self,
        keep_nodes: Set[str],
        ordered_scores: pd.Series,
        anchors: Set[str],
        min_keep: int,
        min_connectivity_pct: float,
        min_keep_fraction: float,
    ) -> Set[str]:
        keep_nodes = set(keep_nodes)
        floor_keep = max(len(anchors), int(len(self.graph) * min_keep_fraction))
        removable = [
            node
            for node in reversed(ordered_scores.index)
            if node in keep_nodes and node not in anchors
        ]
        for node in removable:
            if len(keep_nodes) <= floor_keep or len(keep_nodes) <= min_keep:
                break
            keep_nodes.remove(node)
            if self._component_ratio(keep_nodes) < min_connectivity_pct:
                keep_nodes.add(node)
        return keep_nodes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prune(
        self,
        target_reduction: float = 0.60,
        min_connectivity_pct: float = 0.85,
        protected_fraction: float = 0.20,
        hub_degree_percentile: float = 0.75,
    ) -> nx.DiGraph:
        features = self.features
        ordered_scores = features['adaptive_score'].sort_values(ascending=False)
        anchors = self._select_anchors(protected_fraction, hub_degree_percentile)
        keep_nodes: Set[str] = set(anchors)

        total_nodes = len(features)
        target_keep = max(len(anchors), int(total_nodes * (1 - target_reduction)))
        for node in ordered_scores.index:
            if node in keep_nodes:
                continue
            keep_nodes.add(node)
            if len(keep_nodes) >= target_keep:
                break

        keep_nodes = self._ensure_connectivity(keep_nodes, ordered_scores, min_connectivity_pct)

        # Allow up to +5% additional reduction if connectivity remains healthy
        min_keep_fraction = max(len(anchors) / max(total_nodes, 1), 1 - target_reduction)
        keep_nodes = self._refine_selection(
            keep_nodes,
            ordered_scores,
            anchors,
            min_keep=target_keep,
            min_connectivity_pct=min_connectivity_pct,
            min_keep_fraction=min_keep_fraction,
        )

        pruned_graph = self.graph.subgraph(keep_nodes).copy()
        isolates = list(nx.isolates(pruned_graph))
        if isolates:
            pruned_graph.remove_nodes_from(isolates)
        return pruned_graph


def adaptive_multi_strategy_prune(
    graph: nx.DiGraph,
    entities_df: pd.DataFrame,
    communities_df: Optional[pd.DataFrame] = None,
    target_reduction: float = 0.60,
    min_connectivity_pct: float = 0.85,
    protected_fraction: float = 0.20,
    hub_degree_percentile: float = 0.75,
) -> nx.DiGraph:
    pruner = AdaptiveMultiStrategyPruner(graph, entities_df, communities_df)
    return pruner.prune(
        target_reduction=target_reduction,
        min_connectivity_pct=min_connectivity_pct,
        protected_fraction=protected_fraction,
        hub_degree_percentile=hub_degree_percentile,
    )
