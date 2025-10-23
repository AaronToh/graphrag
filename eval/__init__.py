"""
RAG Pipeline Evaluation Module

This module provides tools for evaluating RAG (Retrieval-Augmented Generation) pipelines
using multiple metrics including Document MRR, Faithfulness, and Semantic Answer Similarity.

Scripts:
- generate_answers.py: Generate answers using GraphRAG for evaluation questions
- run_eval.py: End-to-end evaluation runner comparing baseline vs pruned systems
- eval.py: Core evaluation logic and metrics
"""

from eval.eval import evaluate_rag_pipeline, evaluate_with_defaults

__all__ = ["evaluate_rag_pipeline", "evaluate_with_defaults"]
