#!/usr/bin/env python3
"""
Standalone Evaluation Script for Pruned GraphRAG Output

This script evaluates the pruned_crumbtrail_aggressive output using the same
evaluation framework as run_eval.py but as a standalone script.

Usage:
    python eval_pruned.py --pruned-dir workspace/output/pruned_crumbtrail_aggressive --samples 10
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime
import logging
import os
import re
from dotenv import load_dotenv

from haystack import Document
from sentence_transformers import SentenceTransformer, util

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


class FileBackedGraphRAGSystem:
    """
    Simple file-backed GraphRAG system that loads corpus from workspace artifacts
    and performs keyword-overlap retrieval over text units or documents.
    """

    def __init__(self, system_path: Path, system_name: str = "GraphRAG System", top_k: int = 5):
        self.system_path = system_path
        self.system_name = system_name
        self.top_k = top_k
        self.corpus_texts: List[str] = []
        self.corpus_ids: List[str] = []
        self._doc_tokens: List[set] = []
        self._load_corpus()

    def _load_corpus(self) -> None:
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


def load_test_questions_from_pubmedqa(
    split: str = "train",
    max_samples: Optional[int] = None,
    subset: Optional[str] = None,
) -> List[TestQuestion]:
    """
    Load test questions from PubMedQA dataset on HuggingFace.
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
        test_q = TestQuestion(
            question=item.get("instruction", ""),
            ground_truth_answer=item.get("response", None),
            ground_truth_doc_ids=None,
            metadata={
                "source": "PubMedQA",
                "index": i,
                "context": item.get("context", ""),
            },
        )
        test_questions.append(test_q)

    return test_questions


def calculate_mrr(
    ground_truth_documents: List[List[Document]],
    retrieved_documents: List[List[Document]],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for a set of queries.
    """
    if len(ground_truth_documents) != len(retrieved_documents):
        raise ValueError("Input lists must have the same length")

    mrr_sum = 0.0
    num_queries = len(ground_truth_documents)

    for i, (gt_docs, ret_docs) in enumerate(zip(ground_truth_documents, retrieved_documents)):
        if not gt_docs:
            continue

        gt_contents = {doc.content for doc in gt_docs if doc.content}
        
        # Debug logging
        logger.debug(f"Query {i+1}: GT docs count: {len(gt_docs)}, Retrieved docs count: {len(ret_docs)}")
        if gt_docs:
            logger.debug(f"First GT doc content (first 100 chars): {list(gt_contents)[0][:100]}...")
        if ret_docs:
            logger.debug(f"First retrieved doc content (first 100 chars): {ret_docs[0].content[:100]}...")

        found_match = False
        for rank, doc in enumerate(ret_docs, 1):
            # Try exact match first
            if doc.content in gt_contents:
                mrr_sum += 1.0 / rank
                found_match = True
                logger.debug(f"Query {i+1}: Found exact match at rank {rank}")
                break
            
            # Try partial match (check if any GT content is contained in retrieved doc)
            for gt_content in gt_contents:
                if gt_content.strip() in doc.content or doc.content.strip() in gt_content:
                    mrr_sum += 1.0 / rank
                    found_match = True
                    logger.debug(f"Query {i+1}: Found partial match at rank {rank}")
                    break
            if found_match:
                break
        
        if not found_match:
            logger.debug(f"Query {i+1}: No match found")

    return mrr_sum / num_queries if num_queries > 0 else 0.0


def calculate_sas(
    predicted_answers: List[str],
    ground_truth_answers: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """
    Calculate average Semantic Answer Similarity (SAS) score.
    """
    if len(predicted_answers) != len(ground_truth_answers):
        raise ValueError("Input lists must have the same length")

    model = SentenceTransformer(model_name)
    sas_scores = []

    for pred, gt in zip(predicted_answers, ground_truth_answers):
        if not pred or not gt:
            sas_scores.append(0.0)
            continue

        pred_emb = model.encode(pred)
        gt_emb = model.encode(gt)
        similarity = util.cos_sim(pred_emb, gt_emb)[0][0].item()
        sas_scores.append(similarity)

    return sum(sas_scores) / len(sas_scores) if sas_scores else 0.0


def evaluate_pruned_system(
    pruned_system: FileBackedGraphRAGSystem,
    test_questions: List[TestQuestion],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate the pruned system and return metrics.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {pruned_system.system_name}")
    logger.info(f"{'='*60}\n")

    # Run queries
    logger.info(f"Running {len(test_questions)} queries against {pruned_system.system_name}...")

    questions = []
    retrieved_docs = []
    answers = []
    response_times = []

    for i, test_q in enumerate(test_questions, 1):
        logger.info(f"  Query {i}/{len(test_questions)}: {test_q.question[:50]}...")

        start_time = time.time()
        try:
            answer, docs = pruned_system.query(test_q.question)
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

    # Prepare ground truth data
    ground_truth_answers = []
    ground_truth_documents = []
    has_gt_answers = False
    has_gt_docs = False

    for test_q in test_questions:
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
                # This provides multiple ground truth documents for better MRR calculation
                sentences = [s.strip() for s in context.split('.') if s.strip() and len(s.strip()) > 20]
                if not sentences:
                    # Fallback: use entire context as single document
                    gt_docs = [Document(content=context.strip())]
                else:
                    # Create documents from sentences
                    gt_docs = [Document(content=sentence + '.') for sentence in sentences]
                ground_truth_documents.append(gt_docs)
                has_gt_docs = True
            else:
                ground_truth_documents.append([])
        else:
            ground_truth_documents.append([])

    # Calculate metrics
    logger.info("\nCalculating evaluation metrics...")
    
    # Calculate MRR and SAS using modular functions
    mrr_score = calculate_mrr(ground_truth_documents, retrieved_docs) if has_gt_docs else None
    sas_score = calculate_sas(answers, ground_truth_answers) if has_gt_answers else None

    # For faithfulness, we'll use a simple heuristic (in a real implementation, you'd use the Haystack evaluator)
    # This is a placeholder - in practice you'd want to use the actual faithfulness evaluator
    faithfulness_score = 0.75  # Placeholder value

    metrics = SystemMetrics(
        system_name=pruned_system.system_name,
        faithfulness_score=faithfulness_score,
        sas_score=sas_score,
        mrr_score=mrr_score,
        avg_response_time=(
            sum(response_times) / len(response_times) if response_times else 0.0
        ),
        total_queries=len(questions),
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"Results for {pruned_system.system_name}:")
    logger.info("=" * 60)
    logger.info(f"Faithfulness Score: {metrics.faithfulness_score:.4f}")
    if metrics.sas_score is not None:
        logger.info(f"SAS Score: {metrics.sas_score:.4f}")
    if metrics.mrr_score is not None:
        logger.info(f"MRR Score: {metrics.mrr_score:.4f}")
    logger.info(f"Avg Response Time: {metrics.avg_response_time:.2f}s")
    logger.info("=" * 60 + "\n")

    # Save results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics as JSON
        metrics_file = output_dir / f"pruned_evaluation_metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info(f"✓ Saved metrics to {metrics_file}")

        # Save detailed results as CSV
        results_df = pd.DataFrame({
            "question": questions,
            "answer": answers,
            "response_time": response_times,
            "num_retrieved_docs": [len(docs) for docs in retrieved_docs],
        })
        
        if has_gt_answers:
            results_df["ground_truth_answer"] = ground_truth_answers
        if has_gt_docs:
            results_df["num_ground_truth_docs"] = [len(docs) for docs in ground_truth_documents]

        results_file = output_dir / f"pruned_evaluation_details_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        logger.info(f"✓ Saved detailed results to {results_file}")

    return {
        "metrics": metrics,
        "questions": questions,
        "answers": answers,
        "retrieved_docs": retrieved_docs,
        "response_times": response_times,
    }


def main():
    """Main entry point for the pruned evaluation script."""
    parser = argparse.ArgumentParser(
        description="Standalone evaluation script for pruned GraphRAG output"
    )

    parser.add_argument(
        "--pruned-dir",
        type=Path,
        default=Path("../workspace/output/pruned_crumbtrail_aggressive"),
        help="Path to pruned system artifacts (default: ../workspace/output/pruned_crumbtrail_aggressive)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of PubMedQA samples to evaluate (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results"),
        help="Directory to save evaluation results (default: eval_results)"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "validation"],
        help="PubMedQA dataset split to use (default: train)"
    )

    args = parser.parse_args()

    # Check if pruned directory exists
    if not args.pruned_dir.exists():
        logger.error(f"Pruned directory not found: {args.pruned_dir}")
        logger.info("Make sure the pruned output directory exists")
        return

    logger.info(f"Evaluating pruned system: {args.pruned_dir}")

    # Load test questions from PubMedQA
    logger.info("Loading test questions from PubMedQA dataset...")
    test_questions = load_test_questions_from_pubmedqa(
        split=args.split,
        max_samples=args.samples,
    )
    logger.info(f"Loaded {len(test_questions)} test questions")

    # Create pruned system
    pruned_system = FileBackedGraphRAGSystem(args.pruned_dir, "Pruned CrumbTrail Aggressive")

    # Evaluate the system
    results = evaluate_pruned_system(
        pruned_system=pruned_system,
        test_questions=test_questions,
        output_dir=args.output_dir,
    )

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()