#!/usr/bin/env python3
"""
End-to-End GraphRAG Evaluation Runner

This module orchestrates comprehensive evaluation of baseline vs pruned GraphRAG systems.
It handles:
- Loading test questions and ground truth data
- Running queries against baseline and pruned systems
- Collecting performance metrics
- Generating comparison reports
- Running ablation studies

Usage:
    # Basic comparison (uses default paths for baseline and CrumbTrail pruned)
    python eval/run_eval.py --use-pubmedqa --pubmedqa-samples 20

    # Compare with custom paths
    python eval/run_eval.py --baseline workspace/output --pruned workspace/output/pruned_crumbtrail

    # Ablation study
    python eval/run_eval.py --ablation --ablation-config eval/ablation_config.json

    # Custom test data
    python eval/run_eval.py --test-data data/gold/test_questions.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

from haystack import Document
from eval import evaluate_rag_pipeline, evaluate_with_defaults
from eval import calculate_mrr, calculate_sas

# Load environment variables from .env file
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestQuestion:
    """Structure for a test question with ground truth."""

    question: str
    ground_truth_answer: Optional[str] = None
    ground_truth_doc_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """Performance metrics for a RAG system."""

    system_name: str
    faithfulness_score: float
    sas_score: Optional[float] = None
    mrr_score: Optional[float] = None
    avg_response_time: float = 0.0
    total_queries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGSystemInterface:
    """
    Abstract interface for querying a RAG system.
    Subclass this to implement specific GraphRAG query logic.
    """

    def __init__(self, system_path: Path, system_name: str = "RAG System"):
        """
        Initialize RAG system interface.

        Args:
            system_path: Path to the system artifacts (e.g., workspace/output)
            system_name: Human-readable name for this system
        """
        self.system_path = system_path
        self.system_name = system_name

    def query(self, question: str) -> Tuple[str, List[Document]]:
        """
        Query the RAG system.

        Args:
            question: The question to answer

        Returns:
            Tuple of (answer, retrieved_documents)
        """
        raise NotImplementedError("Subclasses must implement query()")


class MockGraphRAGSystem(RAGSystemInterface):
    """
    Mock GraphRAG system for testing the evaluation framework.
    Replace this with actual GraphRAG query implementation.
    """

    def query(self, question: str) -> Tuple[str, List[Document]]:
        """Mock query implementation."""
        # Simulate retrieved documents
        docs = [
            Document(
                content=f"Mock document 1 for question: {question}",
                meta={"source": "mock_doc_1.txt", "score": 0.95},
            ),
            Document(
                content=f"Mock document 2 with additional context.",
                meta={"source": "mock_doc_2.txt", "score": 0.85},
            ),
        ]

        # Simulate answer generation
        answer = f"Mock answer to: {question}"

        return answer, docs


class FileBackedGraphRAGSystem(RAGSystemInterface):
    """
    Simple file-backed GraphRAG system that loads corpus from workspace artifacts
    and performs keyword-overlap retrieval over text units or documents.
    """

    def __init__(self, system_path: Path, system_name: str = "GraphRAG System", top_k: int = 5):
        super().__init__(system_path, system_name)
        self.top_k = top_k
        self.corpus_texts: List[str] = []
        self.corpus_ids: List[str] = []
        self._doc_tokens: List[set] = []
        self._load_corpus()

    def _load_corpus(self) -> None:
        import re
        import pandas as pd

        # Prefer text_units.parquet, fall back to documents.parquet
        candidates = [
            self.system_path / "text_units.parquet",
            self.system_path / "documents.parquet",
        ]
        file_path = None
        for p in candidates:
            if p.exists():
                file_path = p
                break
        if file_path is None:
            raise FileNotFoundError(
                f"No corpus parquet found in {self.system_path}. Expected one of: text_units.parquet, documents.parquet"
            )

        df = pd.read_parquet(file_path)
        
        # Check if this is a pruned system by looking for pruned artifacts
        pruned_entities_path = self.system_path / "pruned_entities.parquet"
        pruned_relationships_path = self.system_path / "pruned_relationships.parquet"
        
        if pruned_entities_path.exists() and pruned_relationships_path.exists():
            logger.info(f"Found pruned artifacts, filtering corpus for {self.system_name}")
            
            # Load pruned entities and relationships to get valid IDs
            pruned_entities = pd.read_parquet(pruned_entities_path)
            pruned_relationships = pd.read_parquet(pruned_relationships_path)
            
            # Get sets of valid entity and relationship IDs
            valid_entity_ids = set(pruned_entities['id'].astype(str)) if 'id' in pruned_entities.columns else set()
            valid_relationship_ids = set(pruned_relationships['id'].astype(str)) if 'id' in pruned_relationships.columns else set()
            
            # Filter text units to only include those with entities/relationships in the pruned set
            if 'entity_ids' in df.columns and 'relationship_ids' in df.columns:
                def has_valid_entities_or_relationships(row):
                    # Check if any entity_ids or relationship_ids are in the pruned sets
                    entity_ids = row['entity_ids']
                    relationship_ids = row['relationship_ids']
                    
                    # Convert to sets for intersection, handling None and empty arrays
                    text_entity_ids = set()
                    if entity_ids is not None and len(entity_ids) > 0:
                        text_entity_ids = set(str(eid) for eid in entity_ids)
                    
                    text_relationship_ids = set()
                    if relationship_ids is not None and len(relationship_ids) > 0:
                        text_relationship_ids = set(str(rid) for rid in relationship_ids)
                    
                    # Keep text unit if it has any entities or relationships that are in the pruned graph
                    has_valid_entities = bool(text_entity_ids.intersection(valid_entity_ids))
                    has_valid_relationships = bool(text_relationship_ids.intersection(valid_relationship_ids))
                    
                    return has_valid_entities or has_valid_relationships
                
                # Apply filter
                original_count = len(df)
                df = df[df.apply(has_valid_entities_or_relationships, axis=1)]
                filtered_count = len(df)
                
                logger.info(f"Filtered corpus from {original_count} to {filtered_count} text units based on pruned entities/relationships")
            else:
                logger.warning("Text units don't have entity_ids/relationship_ids columns, cannot filter based on pruned graph")

        # Identify content and id columns with robust fallbacks
        content_col = None
        for col in ["text", "content", "body", "document", "chunk_text", "chunk"]:
            if col in df.columns:
                content_col = col
                break
        if content_col is None:
            # Heuristic: first string-like column
            for col in df.columns:
                if df[col].dtype == object:
                    content_col = col
                    break
        id_col = None
        for col in [
            "id",
            "document_id",
            "text_unit_id",
            "doc_id",
            "source_id",
            "node_id",
        ]:
            if col in df.columns:
                id_col = col
                break
        if id_col is None:
            id_col = None  # will generate sequential ids

        texts = df[content_col].astype(str).tolist()
        if id_col:
            ids = df[id_col].astype(str).tolist()
        else:
            ids = [f"doc_{i}" for i in range(len(texts))]

        # Minimal cleaning and tokenization cache
        def tokenize(s: str) -> set:
            return set(re.findall(r"\w+", s.lower()))

        self.corpus_texts = texts
        self.corpus_ids = ids
        self._doc_tokens = [tokenize(t) for t in texts]

        logger.info(
            f"Loaded corpus for {self.system_name} from {file_path} with {len(self.corpus_texts)} items"
        )

    def _score(self, query: str, doc_tokens: set) -> int:
        import re

        q_tokens = set(re.findall(r"\w+", query.lower()))
        # Simple overlap count
        return sum(1 for t in q_tokens if t in doc_tokens)

    def query(self, question: str) -> Tuple[str, List[Document]]:
        # Score all docs (naive token overlap); for larger corpora, consider sampling
        scores = [self._score(question, tok) for tok in self._doc_tokens]
        # Select top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]

        docs: List[Document] = []
        for i in top_indices:
            docs.append(
                Document(
                    content=self.corpus_texts[i],
                    meta={
                        "doc_id": self.corpus_ids[i],
                        "source": str(self.system_path),
                        "score": float(scores[i]),
                    },
                )
            )

        # Generate a simple extractive answer by concatenating snippets
        def truncate(text: str, limit: int = 400) -> str:
            return text if len(text) <= limit else text[: limit] + "..."

        answer_context = " \n".join([truncate(self.corpus_texts[i], 400) for i in top_indices])
        answer = f"Answer (extractive):\n{answer_context}"

        return answer, docs


class EvaluationRunner:
    """
    Orchestrates end-to-end evaluation of RAG systems.
    """

    def __init__(
        self,
        test_questions: List[TestQuestion],
        faithfulness_llm_provider: str = "openai",
        faithfulness_llm_model: str = "gpt-4o-mini",
        faithfulness_api_base_url: Optional[str] = None,
        faithfulness_api_key: Optional[str] = None,
        sas_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize evaluation runner.

        Args:
            test_questions: List of test questions to evaluate
            faithfulness_llm_provider: LLM provider for faithfulness eval
            faithfulness_llm_model: Model name for faithfulness eval
            faithfulness_api_base_url: API base URL (for Ollama/OpenRouter)
            faithfulness_api_key: API key for authentication
            sas_model: Model for semantic answer similarity
        """
        self.test_questions = test_questions
        self.faithfulness_llm_provider = faithfulness_llm_provider
        self.faithfulness_llm_model = faithfulness_llm_model
        self.faithfulness_api_base_url = faithfulness_api_base_url
        self.faithfulness_api_key = faithfulness_api_key
        self.sas_model = sas_model

    def run_queries(
        self, rag_system: RAGSystemInterface
    ) -> Tuple[List[str], List[List[Document]], List[str], List[float]]:
        """
        Run all test questions against a RAG system.

        Args:
            rag_system: The RAG system to query

        Returns:
            Tuple of (questions, retrieved_docs, answers, response_times)
        """
        logger.info(
            f"Running {len(self.test_questions)} queries against {rag_system.system_name}..."
        )

        questions = []
        retrieved_docs = []
        answers = []
        response_times = []

        for i, test_q in enumerate(self.test_questions, 1):
            logger.info(
                f"  Query {i}/{len(self.test_questions)}: {test_q.question[:50]}..."
            )

            start_time = time.time()
            try:
                answer, docs = rag_system.query(test_q.question)
                elapsed = time.time() - start_time

                questions.append(test_q.question)
                retrieved_docs.append(docs)
                answers.append(answer)
                response_times.append(elapsed)

                logger.info(f"    ✓ Response time: {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"    ✗ Query failed: {e}")
                # Add placeholder data for failed queries
                questions.append(test_q.question)
                retrieved_docs.append([])
                answers.append(f"ERROR: {str(e)}")
                response_times.append(0.0)

        return questions, retrieved_docs, answers, response_times

    def evaluate_system(
        self, rag_system: RAGSystemInterface, run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG system.

        Args:
            rag_system: The RAG system to evaluate
            run_name: Name for this evaluation run

        Returns:
            Dictionary with evaluation results and metrics
        """
        if run_name is None:
            run_name = (
                f"{rag_system.system_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {rag_system.system_name}")
        logger.info(f"{'='*60}\n")

        # Run queries
        questions, retrieved_docs, answers, response_times = self.run_queries(
            rag_system
        )

        # Prepare ground truth data
        ground_truth_answers = []
        ground_truth_documents = []
        has_gt_answers = False
        has_gt_docs = False

        for test_q in self.test_questions:
            if test_q.ground_truth_answer:
                ground_truth_answers.append(test_q.ground_truth_answer)
                has_gt_answers = True
            else:
                ground_truth_answers.append(None)

            # Handle PubMedQA context as ground truth documents
            if test_q.metadata and test_q.metadata.get("source") == "PubMedQA":
                context = test_q.metadata.get("context", "")
                if isinstance(context, str) and context.strip():
                    # Split context into sentences/paragraphs as individual documents
                    sentences = [s.strip() for s in context.split('.') if s.strip() and len(s.strip()) > 20]
                    if sentences:
                        gt_docs = [Document(content=sentence) for sentence in sentences]
                    else:
                        # Fallback: use entire context as single document
                        gt_docs = [Document(content=context.strip())]
                else:
                    gt_docs = []
                ground_truth_documents.append(gt_docs)
                if gt_docs:
                    has_gt_docs = True
            # TODO: Load ground truth documents from IDs
            elif test_q.ground_truth_doc_ids:
                # Placeholder: would need to load actual documents
                ground_truth_documents.append([])
                has_gt_docs = True
            else:
                ground_truth_documents.append([])

        # Run evaluation for faithfulness only
        logger.info("\nRunning evaluation metrics...")
        eval_results = evaluate_with_defaults(
            questions=questions,
            retrieved_documents=retrieved_docs,
            predicted_answers=answers,
            ground_truth_answers=None,
            ground_truth_documents=None,
            run_name=run_name,
            faithfulness_llm_provider=self.faithfulness_llm_provider,
            faithfulness_llm_model=self.faithfulness_llm_model,
            faithfulness_api_base_url=self.faithfulness_api_base_url,
            faithfulness_api_key=self.faithfulness_api_key,
            model=self.sas_model,
        )

        # Compute MRR and SAS using modular functions
        mrr_score = calculate_mrr(ground_truth_documents, retrieved_docs) if has_gt_docs else None
        sas_score = calculate_sas(answers, ground_truth_answers, self.sas_model) if has_gt_answers else None

        # Extract faithfulness
        aggregated = eval_results["aggregated_report"]
        faithfulness_mean = aggregated.get("faithfulness", {}).get("mean", None)
        if faithfulness_mean is None:
            df = eval_results.get("detailed_results", None)
            if df is not None:
                if "faithfulness" in df.columns:
                    try:
                        faithfulness_mean = float(df["faithfulness"].mean())
                    except Exception:
                        faithfulness_mean = None
                elif "faithful" in df.columns:
                    try:
                        faithfulness_mean = float(df["faithful"].mean())
                    except Exception:
                        faithfulness_mean = None
        if faithfulness_mean is None:
            faithfulness_mean = 0.0

        metrics = SystemMetrics(
            system_name=rag_system.system_name,
            faithfulness_score=faithfulness_mean,
            sas_score=sas_score,
            mrr_score=mrr_score,
            avg_response_time=(
                sum(response_times) / len(response_times) if response_times else 0.0
            ),
            total_queries=len(questions),
        )

        logger.info("\n" + "=" * 60)
        logger.info(f"Results for {rag_system.system_name}:")
        logger.info("=" * 60)
        logger.info(f"Faithfulness Score: {metrics.faithfulness_score:.4f}")
        if metrics.sas_score is not None:
            logger.info(f"SAS Score: {metrics.sas_score:.4f}")
        if metrics.mrr_score is not None:
            logger.info(f"MRR Score: {metrics.mrr_score:.4f}")
        logger.info(f"Avg Response Time: {metrics.avg_response_time:.2f}s")
        logger.info("=" * 60 + "\n")

        return {
            "metrics": metrics,
            "eval_results": eval_results,
            "response_times": response_times,
            "run_name": run_name,
        }

    def compare_systems(
        self,
        baseline_system: RAGSystemInterface,
        pruned_system: RAGSystemInterface,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Compare baseline vs pruned systems.

        Args:
            baseline_system: The baseline RAG system
            pruned_system: The pruned RAG system
            output_dir: Directory to save comparison reports

        Returns:
            Dictionary with comparison results
        """
        logger.info("\n" + "#" * 60)
        logger.info("# BASELINE vs PRUNED SYSTEM COMPARISON")
        logger.info("#" * 60 + "\n")

        # Evaluate both systems
        baseline_results = self.evaluate_system(baseline_system, run_name="baseline")
        pruned_results = self.evaluate_system(pruned_system, run_name="pruned")

        # Extract metrics
        baseline_metrics = baseline_results["metrics"]
        pruned_metrics = pruned_results["metrics"]

        # Calculate improvements
        faithfulness_change = (
            (
                (
                    pruned_metrics.faithfulness_score
                    - baseline_metrics.faithfulness_score
                )
                / baseline_metrics.faithfulness_score
                * 100
            )
            if baseline_metrics.faithfulness_score > 0
            else 0
        )

        response_time_change = (
            (
                (pruned_metrics.avg_response_time - baseline_metrics.avg_response_time)
                / baseline_metrics.avg_response_time
                * 100
            )
            if baseline_metrics.avg_response_time > 0
            else 0
        )

        # Generate comparison report
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 60)
        logger.info(f"\nFaithfulness:")
        logger.info(f"  Baseline: {baseline_metrics.faithfulness_score:.4f}")
        logger.info(f"  Pruned:   {pruned_metrics.faithfulness_score:.4f}")
        logger.info(f"  Change:   {faithfulness_change:+.2f}%")

        if (
            baseline_metrics.sas_score is not None
            and pruned_metrics.sas_score is not None
        ):
            sas_change = (
                (
                    (pruned_metrics.sas_score - baseline_metrics.sas_score)
                    / baseline_metrics.sas_score
                    * 100
                )
                if baseline_metrics.sas_score > 0
                else 0
            )
            logger.info(f"\nSemantic Answer Similarity (SAS):")
            logger.info(f"  Baseline: {baseline_metrics.sas_score:.4f}")
            logger.info(f"  Pruned:   {pruned_metrics.sas_score:.4f}")
            logger.info(f"  Change:   {sas_change:+.2f}%")

        if (
            baseline_metrics.mrr_score is not None
            and pruned_metrics.mrr_score is not None
        ):
            mrr_change = (
                (
                    (pruned_metrics.mrr_score - baseline_metrics.mrr_score)
                    / baseline_metrics.mrr_score
                    * 100
                )
                if baseline_metrics.mrr_score > 0
                else 0
            )
            logger.info(f"\nMean Reciprocal Rank (MRR):")
            logger.info(f"  Baseline: {baseline_metrics.mrr_score:.4f}")
            logger.info(f"  Pruned:   {pruned_metrics.mrr_score:.4f}")
            logger.info(f"  Change:   {mrr_change:+.2f}%")

        logger.info(f"\nResponse Time:")
        logger.info(f"  Baseline: {baseline_metrics.avg_response_time:.2f}s")
        logger.info(f"  Pruned:   {pruned_metrics.avg_response_time:.2f}s")
        logger.info(f"  Change:   {response_time_change:+.2f}%")
        logger.info("=" * 60 + "\n")

        comparison = {
            "baseline": baseline_results,
            "pruned": pruned_results,
            "comparison": {
                "faithfulness_change_pct": faithfulness_change,
                "response_time_change_pct": response_time_change,
            },
        }

        # Save results if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save metrics as JSON
            metrics_file = output_dir / f"comparison_metrics_{timestamp}.json"
            with open(metrics_file, "w") as f:
                json.dump(
                    {
                        "baseline": baseline_metrics.to_dict(),
                        "pruned": pruned_metrics.to_dict(),
                        "comparison": comparison["comparison"],
                    },
                    f,
                    indent=2,
                )
            logger.info(f"✓ Saved metrics to {metrics_file}")

            # Save detailed results as CSV
            baseline_df = baseline_results["eval_results"]["detailed_results"]
            pruned_df = pruned_results["eval_results"]["detailed_results"]

            baseline_df.to_csv(
                output_dir / f"baseline_details_{timestamp}.csv", index=False
            )
            pruned_df.to_csv(
                output_dir / f"pruned_details_{timestamp}.csv", index=False
            )
            logger.info(f"✓ Saved detailed results to {output_dir}")

        return comparison

    def run_ablation_study(
        self,
        system_configs: List[Dict[str, Any]],
        system_factory: Callable[[Dict[str, Any]], RAGSystemInterface],
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Run ablation study with multiple system configurations.

        Args:
            system_configs: List of configuration dictionaries
            system_factory: Function that creates RAGSystemInterface from config
            output_dir: Directory to save ablation results

        Returns:
            DataFrame with results for all configurations
        """
        logger.info("\n" + "#" * 60)
        logger.info("# ABLATION STUDY")
        logger.info("#" * 60 + "\n")
        logger.info(f"Testing {len(system_configs)} configurations...\n")

        results = []

        for i, config in enumerate(system_configs, 1):
            logger.info(f"\n--- Configuration {i}/{len(system_configs)} ---")
            logger.info(f"Config: {config}")

            # Create system with this configuration
            system = system_factory(config)

            # Evaluate
            eval_result = self.evaluate_system(
                system, run_name=f"ablation_{i}_{config.get('name', 'unnamed')}"
            )

            # Collect results
            metrics = eval_result["metrics"]
            result_row = {
                "config_id": i,
                "config_name": config.get("name", f"config_{i}"),
                **config,
                **metrics.to_dict(),
            }
            results.append(result_row)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Sort by faithfulness score
        results_df = results_df.sort_values("faithfulness_score", ascending=False)

        logger.info("\n" + "=" * 60)
        logger.info("ABLATION STUDY RESULTS (sorted by faithfulness)")
        logger.info("=" * 60)
        logger.info(f"\n{results_df.to_string()}\n")

        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"ablation_results_{timestamp}.csv"
            results_df.to_csv(output_file, index=False)
            logger.info(f"✓ Saved ablation results to {output_file}")

        return results_df


def load_test_questions_from_json(test_data_path: Path) -> List[TestQuestion]:
    """
    Load test questions from JSON file.

    Expected format:
    [
        {
            "question": "What is machine learning?",
            "ground_truth_answer": "ML is...",
            "ground_truth_doc_ids": ["doc1", "doc2"],
            "metadata": {"category": "ML"}
        },
        ...
    ]
    """
    with open(test_data_path, "r") as f:
        data = json.load(f)

    return [TestQuestion(**item) for item in data]


def load_test_questions_from_pubmedqa(
    split: str = "train",
    max_samples: Optional[int] = None,
    subset: Optional[str] = None,
) -> List[TestQuestion]:
    """
    Load test questions from PubMedQA dataset on HuggingFace.

    Dataset: vblagoje/PubMedQA_instruction

    Args:
        split: Dataset split to use ('train', 'test', 'validation')
        max_samples: Maximum number of samples to load (None = all)
        subset: Optional subset name if dataset has multiple configurations

    Returns:
        List of TestQuestion objects

    Raises:
        ImportError: If datasets library is not installed

    Example:
        >>> questions = load_test_questions_from_pubmedqa(split='train', max_samples=10)
        >>> print(f"Loaded {len(questions)} questions")
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to load PubMedQA. "
            "Install it with: pip install datasets"
        )

    logger.info(f"Loading PubMedQA dataset (split={split})...")

    # Load dataset
    if subset:
        ds = load_dataset("vblagoje/PubMedQA_instruction", subset, split=split)
    else:
        ds = load_dataset("vblagoje/PubMedQA_instruction", split=split)

    # Limit samples if specified
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    logger.info(f"Loaded {len(ds)} samples from PubMedQA")

    # Convert to TestQuestion format
    test_questions = []
    for i, item in enumerate(ds):
        # PubMedQA fields: 'instruction' (question), 'context' (ground truth docs), 'response' (answer)
        test_q = TestQuestion(
            question=item.get("instruction", ""),
            ground_truth_answer=item.get("response", None),
            ground_truth_doc_ids=None,  # Will use context directly
            metadata={
                "source": "PubMedQA",
                "index": i,
                "context": item.get("context", ""),  # Store context for potential use
            },
        )
        test_questions.append(test_q)

    return test_questions


def main():
    """Main entry point for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="End-to-end GraphRAG evaluation runner"
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("workspace/output"),
        help="Path to baseline system artifacts (default: workspace/output)"
    )
    parser.add_argument(
        "--pruned",
        type=Path,
        default=Path("workspace/output/pruned_crumbtrail_aggressive"),
        help="Path to pruned system artifacts (default: workspace/output/pruned_crumbtrail_aggressive)"
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        help="Path to test questions JSON file (if not using --use-pubmedqa)",
    )
    parser.add_argument(
        "--use-pubmedqa",
        action="store_true",
        help="Load test questions from PubMedQA dataset instead of JSON file",
    )
    parser.add_argument(
        "--pubmedqa-split",
        default="train",
        choices=["train", "test", "validation"],
        help="PubMedQA dataset split to use (default: train)",
    )
    parser.add_argument(
        "--pubmedqa-samples",
        type=int,
        help="Maximum number of PubMedQA samples to load (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study instead of single comparison",
    )
    parser.add_argument(
        "--ablation-config", type=Path, help="Path to ablation study configuration JSON"
    )
    parser.add_argument(
        "--faithfulness-provider",
        default="openai",
        choices=["openai", "ollama", "openrouter"],
        help="LLM provider for faithfulness evaluation",
    )
    parser.add_argument(
        "--faithfulness-model",
        default="gpt-4o-mini",
        help="Model for faithfulness evaluation",
    )
    # Add API configuration flags for faithfulness evaluator
    parser.add_argument(
        "--faithfulness-api-base-url",
        type=str,
        help="API base URL for faithfulness evaluator (e.g., OpenRouter/Ollama)",
    )
    parser.add_argument(
        "--faithfulness-api-key",
        type=str,
        help="API key/token for faithfulness evaluator",
    )

    args = parser.parse_args()

    # Load test questions
    if args.use_pubmedqa:
        # Load from PubMedQA dataset
        logger.info("Loading test questions from PubMedQA dataset...")
        test_questions = load_test_questions_from_pubmedqa(
            split=args.pubmedqa_split,
            max_samples=args.pubmedqa_samples,
        )
    else:
        # Load from JSON file
        if not args.test_data:
            logger.error("--test-data required when not using --use-pubmedqa")
            logger.info("Either provide --test-data or use --use-pubmedqa")
            return

        if not args.test_data.exists():
            logger.error(f"Test data file not found: {args.test_data}")
            logger.info(
                "Create a test questions file or use --use-pubmedqa for PubMedQA dataset"
            )
            return

        test_questions = load_test_questions_from_json(args.test_data)

    logger.info(f"Loaded {len(test_questions)} test questions")

    # Create evaluation runner
    runner = EvaluationRunner(
        test_questions=test_questions,
        faithfulness_llm_provider=args.faithfulness_provider,
        faithfulness_llm_model=args.faithfulness_model,
        faithfulness_api_base_url=args.faithfulness_api_base_url,
        faithfulness_api_key=args.faithfulness_api_key,
    )

    if args.ablation:
        # Run ablation study
        if not args.ablation_config:
            logger.error("--ablation-config required for ablation study")
            return

        with open(args.ablation_config, "r") as f:
            configs = json.load(f)

        def system_factory(config):
            # TODO: Replace with actual system creation logic
            return MockGraphRAGSystem(
                Path(config.get("artifacts_path", "workspace/output")),
                system_name=config.get("name", "unnamed"),
            )

        runner.run_ablation_study(
            system_configs=configs,
            system_factory=system_factory,
            output_dir=args.output_dir,
        )
    else:
        # Run single comparison
        # Check if default or custom paths exist
        if not args.baseline.exists():
            logger.error(f"Baseline directory not found: {args.baseline}")
            logger.info("Make sure you've run Stage 1 (ingest/build_index.py) first")
            return

        if not args.pruned.exists():
            logger.error(f"Pruned directory not found: {args.pruned}")
            logger.info("Run CrumbTrail pruning first: python examples/crumbtrail_quickstart.py")
            return

        logger.info(f"Baseline system: {args.baseline}")
        logger.info(f"Pruned system: {args.pruned}")

        baseline_system = FileBackedGraphRAGSystem(args.baseline, "Baseline")
        pruned_system = FileBackedGraphRAGSystem(args.pruned, "Pruned")

        runner.compare_systems(
            baseline_system=baseline_system,
            pruned_system=pruned_system,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
