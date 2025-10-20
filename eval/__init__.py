"""
RAG Pipeline Evaluation Module

This module provides tools for evaluating RAG (Retrieval-Augmented Generation) pipelines
using multiple metrics including Document MRR, Faithfulness, and Semantic Answer Similarity.
"""

from eval.eval import evaluate_rag_pipeline, evaluate_with_defaults

__all__ = ["evaluate_rag_pipeline", "evaluate_with_defaults"]
