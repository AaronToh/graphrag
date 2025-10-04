#!/usr/bin/env python3
"""
Quick Start Example: Using the End-to-End Evaluation Runner

This script demonstrates how to use the evaluation runner to compare
baseline vs pruned GraphRAG systems.

Run with PubMedQA:
    python quickstart_example.py --use-pubmedqa

Run with custom test questions:
    python quickstart_example.py --custom-questions
"""

import argparse
from pathlib import Path
from typing import Tuple, List
from haystack import Document

# Import the evaluation runner
from run_eval import (
    EvaluationRunner,
    TestQuestion,
    RAGSystemInterface,
    load_test_questions_from_pubmedqa,
)


# Step 1: Implement RAGSystemInterface for your GraphRAG system
class MyGraphRAGSystem(RAGSystemInterface):
    """
    Example implementation of RAGSystemInterface.
    Replace this with your actual GraphRAG query logic.
    """

    def query(self, question: str) -> Tuple[str, List[Document]]:
        """
        Query the GraphRAG system.

        In a real implementation, this would:
        1. Load the GraphRAG index from self.system_path
        2. Query the system with the question
        3. Return the answer and retrieved documents
        """
        # TODO: Replace with actual GraphRAG query logic
        # Example:
        # from graphrag import GraphRAG
        # graphrag = GraphRAG.load(self.system_path)
        # result = graphrag.query(question)
        # return result.answer, result.documents

        # Mock implementation for demonstration
        docs = [
            Document(
                content=f"Retrieved context for: {question}",
                meta={"source": "doc1.txt", "score": 0.95},
            )
        ]
        answer = f"Answer to: {question}"

        return answer, docs


# Step 2: Prepare test questions
def create_test_questions() -> List[TestQuestion]:
    """Create test questions for evaluation."""
    return [
        TestQuestion(
            question="What is machine learning?",
            ground_truth_answer="Machine learning is a subset of AI that learns from data.",
            metadata={"category": "ML", "difficulty": "easy"},
        ),
        TestQuestion(
            question="How does deep learning work?",
            ground_truth_answer="Deep learning uses multi-layer neural networks.",
            metadata={"category": "DL", "difficulty": "medium"},
        ),
        TestQuestion(
            question="What are transformers in NLP?",
            ground_truth_answer="Transformers are attention-based neural network architectures.",
            metadata={"category": "NLP", "difficulty": "hard"},
        ),
    ]


# Step 3: Run comparison
def main():
    """Run baseline vs pruned system comparison."""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Quick start evaluation example")
    parser.add_argument(
        "--use-pubmedqa",
        action="store_true",
        help="Use PubMedQA dataset instead of custom questions",
    )
    parser.add_argument(
        "--pubmedqa-samples",
        type=int,
        default=5,
        help="Number of PubMedQA samples to use (default: 5)",
    )
    parser.add_argument(
        "--custom-questions",
        action="store_true",
        help="Use custom test questions (default)",
    )
    args = parser.parse_args()

    # Load test questions
    if args.use_pubmedqa:
        print(f"Loading {args.pubmedqa_samples} samples from PubMedQA...\n")
        test_questions = load_test_questions_from_pubmedqa(
            split="train", max_samples=args.pubmedqa_samples
        )
    else:
        print("Using custom test questions...\n")
        test_questions = create_test_questions()

    print(f"Loaded {len(test_questions)} test questions\n")

    # Create evaluation runner
    runner = EvaluationRunner(
        test_questions=test_questions,
        faithfulness_llm_provider="openai",  # or "ollama" for local
        faithfulness_llm_model="gpt-4o-mini",  # or "llama3.2" for Ollama
    )

    # Create baseline and pruned systems
    baseline_system = MyGraphRAGSystem(
        system_path=Path("workspace/output"), system_name="Baseline GraphRAG"
    )

    pruned_system = MyGraphRAGSystem(
        system_path=Path("workspace/pruned_output"),
        system_name="Pruned GraphRAG (70% nodes)",
    )

    # Compare systems
    print("Starting evaluation...\n")
    results = runner.compare_systems(
        baseline_system=baseline_system,
        pruned_system=pruned_system,
        output_dir=Path("eval/results"),
    )

    # Access results
    baseline_metrics = results["baseline"]["metrics"]
    pruned_metrics = results["pruned"]["metrics"]

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\nBaseline Faithfulness: {baseline_metrics.faithfulness_score:.4f}")
    print(f"Pruned Faithfulness:   {pruned_metrics.faithfulness_score:.4f}")

    if baseline_metrics.sas_score and pruned_metrics.sas_score:
        print(f"\nBaseline SAS Score: {baseline_metrics.sas_score:.4f}")
        print(f"Pruned SAS Score:   {pruned_metrics.sas_score:.4f}")

    print(f"\nBaseline Avg Time: {baseline_metrics.avg_response_time:.2f}s")
    print(f"Pruned Avg Time:   {pruned_metrics.avg_response_time:.2f}s")

    # Determine if pruning was successful
    quality_preserved = (
        pruned_metrics.faithfulness_score >= 0.95 * baseline_metrics.faithfulness_score
    )
    faster = pruned_metrics.avg_response_time < baseline_metrics.avg_response_time

    if quality_preserved and faster:
        print("\n✅ SUCCESS: Pruning maintained quality and improved speed!")
    elif quality_preserved:
        print("\n⚠️  PARTIAL: Quality preserved but no speed improvement")
    else:
        print("\n❌ WARNING: Quality degraded significantly")

    print(f"\nDetailed results saved to: eval/results/")


if __name__ == "__main__":
    main()
