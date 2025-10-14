"""
GraphRAG Pruning Lab

A research framework for exploring graph pruning techniques in GraphRAG systems.

This package provides:
- Document ingestion and GraphRAG indexing (Stage 1)
- Graph scoring and pruning algorithms (Stage 2)
- Comprehensive evaluation metrics (Stage 3)

Quick Start:
    from graphrag_pruning import GraphScorer, GraphPruner, RAGEvaluator

    # Load your GraphRAG artifacts
    # Implement your scoring and pruning logic
    # Evaluate performance improvements
"""

__version__ = "0.1.0"
__author__ = "GraphRAG Pruning Lab"

# Import main classes for easy access
from .pruning import GraphScorer, GraphPruner
from .eval import RAGEvaluator

__all__ = ['GraphScorer', 'GraphPruner', 'RAGEvaluator']
