#!/usr/bin/env python3
"""
Evaluate generated answers from JSON files using RAG evaluation metrics.

This script:
1. Loads all generated answers JSON files from a directory
2. Evaluates each one using Faithfulness, Document MRR, and Semantic Answer Similarity
3. Reports aggregated and detailed evaluation metrics for each file
4. Saves all evaluation results to a single JSON file in the evals directory

Usage:
    cd eval
    python evaluate_answers_multi.py --answers ../data/generated/local
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

from haystack import Document
from eval import evaluate_rag_pipeline

load_dotenv()

# --------------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Load generated answers from JSON
# --------------------------------------------------------------------------------
def load_generated_answers(json_file: Path) -> Dict[str, Any]:
    """Load generated answers from JSON file created by generate_answers.py.

    Args:
        json_file: Path to JSON file with generated answers

    Returns:
        Dictionary containing metadata and results
    """
    logger.info(f"Loading generated answers from {json_file}...")

    if not json_file.exists():
        raise FileNotFoundError(f"Answers file not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "results" not in data:
        raise ValueError(f"Invalid JSON format: 'results' key not found in {json_file}")

    num_results = len(data["results"])
    logger.info(f"Found {num_results} results in the JSON file")

    # Print metadata if available
    if "metadata" in data:
        metadata = data["metadata"]
        logger.info(
            f"Metadata: Total={metadata.get('total')}, Successful={metadata.get('successful')}, "
            f"Failed={metadata.get('failed')}, Success Rate={metadata.get('success_rate', 0)*100:.1f}%"
        )

    return data


# --------------------------------------------------------------------------------
# Convert JSON results to test cases for evaluation
# --------------------------------------------------------------------------------
def convert_to_test_cases(data: Dict[str, Any]) -> tuple:
    """Convert loaded JSON data to format expected by evaluate_rag_pipeline.

    Args:
        data: Dictionary with 'results' key containing list of answer records

    Returns:
        Tuple of (test_cases, ground_truth_answers, ground_truth_documents)
    """
    test_cases = []
    ground_truth_answers = []
    ground_truth_documents = []

    for result in data["results"]:
        # Skip failed results
        if not result.get("success", False):
            logger.warning(
                f"Skipping failed question {result.get('question_id')}: {result.get('error')}"
            )
            continue

        question = result.get("question", "")
        generated_answer = result.get("generated_answer", "")
        ground_truth_answer = result.get("ground_truth_answer", "")

        # Create Document objects from retrieved doc IDs
        # Note: We don't have the actual content, so we use doc IDs as placeholders
        retrieved_doc_ids = result.get("retrieved_doc_ids", [])
        retrieved_docs = [
            Document(content=f"Document {doc_id}", meta={"doc_id": doc_id})
            for doc_id in retrieved_doc_ids
        ]

        # For MRR evaluation, create ground truth documents
        ground_truth_doc_ids = result.get("ground_truth_doc_ids", [])
        gt_docs = [
            Document(content=f"Document {doc_id}", meta={"doc_id": doc_id})
            for doc_id in ground_truth_doc_ids
        ]

        test_cases.append((question, retrieved_docs, generated_answer))
        ground_truth_answers.append(ground_truth_answer)
        ground_truth_documents.append(gt_docs)

    logger.info(f"Prepared {len(test_cases)} test cases for evaluation")
    return test_cases, ground_truth_answers, ground_truth_documents


# --------------------------------------------------------------------------------
# Main evaluation function for a single file
# --------------------------------------------------------------------------------
def evaluate_answers_from_json(
    json_file: Path,
    faithfulness_llm_provider: str = "openai",
    faithfulness_llm_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Evaluate generated answers from JSON file.

    Args:
        json_file: Path to JSON file with generated answers
        faithfulness_llm_provider: LLM provider for faithfulness evaluation
        faithfulness_llm_model: Model to use for faithfulness evaluation

    Returns:
        Dictionary with evaluation results
    """
    # Load data
    data = load_generated_answers(json_file)

    # Convert to test cases
    test_cases, ground_truth_answers, ground_truth_documents = convert_to_test_cases(
        data
    )

    if not test_cases:
        logger.error("No valid test cases to evaluate!")
        return {}

    # Run evaluation
    logger.info("Starting evaluation...")
    logger.info(
        f"Using {faithfulness_llm_provider}/{faithfulness_llm_model} for faithfulness evaluation"
    )

    results = evaluate_rag_pipeline(
        test_cases=test_cases,
        ground_truth_answers=ground_truth_answers,
        ground_truth_documents=ground_truth_documents,
        faithfulness_llm_provider=faithfulness_llm_provider,
        faithfulness_llm_model=faithfulness_llm_model,
    )

    return results


# --------------------------------------------------------------------------------
# Evaluate all JSON files in a directory
# --------------------------------------------------------------------------------
def evaluate_all_answers_in_directory(
    answers_dir: Path,
    faithfulness_llm_provider: str = "openai",
    faithfulness_llm_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Evaluate all generated answers JSON files in a directory.

    Args:
        answers_dir: Path to directory containing generated answer JSON files
        faithfulness_llm_provider: LLM provider for faithfulness evaluation
        faithfulness_llm_model: Model to use for faithfulness evaluation

    Returns:
        Dictionary with all evaluation results
    """
    # Find all JSON files in the directory
    json_files = list(answers_dir.glob("*.json"))

    if not json_files:
        logger.error(f"No JSON files found in directory: {answers_dir}")
        return {}

    logger.info(f"Found {len(json_files)} JSON files to evaluate")

    all_results = {}
    for json_file in json_files:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {json_file.name}")
        logger.info(f"{'='*80}")

        try:
            results = evaluate_answers_from_json(
                json_file=json_file,
                faithfulness_llm_provider=faithfulness_llm_provider,
                faithfulness_llm_model=faithfulness_llm_model,
            )

            if results:
                all_results[json_file.name] = results
                logger.info(f"✓ Successfully evaluated {json_file.name}")
            else:
                logger.warning(f"⚠ No results for {json_file.name}")

        except Exception as e:
            logger.error(f"✗ Failed to evaluate {json_file.name}: {e}", exc_info=True)
            all_results[json_file.name] = {
                "error": str(e),
                "aggregated_report": {},
            }

    logger.info(f"\n{'='*80}")
    logger.info(f"Completed evaluation of {len(all_results)}/{len(json_files)} files")
    logger.info(f"{'='*80}\n")

    return all_results


# --------------------------------------------------------------------------------
# Pretty print results for all files
# --------------------------------------------------------------------------------
def print_evaluation_results(all_results: Dict[str, Any]):
    """Pretty print evaluation results for all files.

    Args:
        all_results: Dictionary mapping filenames to evaluation results
    """
    if not all_results:
        logger.error("No results to display")
        return

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - ALL FILES")
    print("=" * 80)

    for filename, results in all_results.items():
        print("\n" + "=" * 80)
        print(f"FILE: {filename}")
        print("=" * 80)

        # Check if evaluation failed
        if "error" in results and not results.get("aggregated_report"):
            print(f"\n✗ Evaluation failed: {results['error']}")
            continue

        # Print aggregated metrics
        print("\nAggregated Metrics:")
        print("-" * 40)

        agg_report = results.get("aggregated_report", {})
        if (
            isinstance(agg_report, dict)
            and "metrics" in agg_report
            and "score" in agg_report
        ):
            print(f"\n{'Metric':<30} {'Score':>10}")
            print("-" * 42)
            for metric, score in zip(agg_report["metrics"], agg_report["score"]):
                print(f"{metric:<30} {score:>10.4f}")
        else:
            print(agg_report)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)


# --------------------------------------------------------------------------------
# Save all evaluation results to JSON
# --------------------------------------------------------------------------------
def save_all_evaluation_results(
    all_results: Dict[str, Any], source_dir: Path, output_dir: Path = Path("evals")
) -> Path:
    """Save all evaluation results to a single JSON file.

    Args:
        all_results: Dictionary mapping filenames to evaluation results
        source_dir: Path to the source answers directory
        output_dir: Directory to save evaluation results (default: evals)

    Returns:
        Path to saved JSON file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"eval_multi_all_files_{timestamp}.json"
    output_path = output_dir / output_filename

    # Prepare data to save
    save_data = {
        "metadata": {
            "source_directory": str(source_dir),
            "evaluation_timestamp": datetime.now().isoformat(),
            "num_files_evaluated": len(all_results),
            "files": list(all_results.keys()),
        },
        "results": {},
    }

    # Process each file's results
    for filename, results in all_results.items():
        save_data["results"][filename] = {
            "aggregated_report": results.get("aggregated_report", {}),
            "error": results.get("error", None),
        }

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    logger.info(f"All evaluation results saved to: {output_path}")
    return output_path


# --------------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate generated answers from JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""""",
    )
    parser.add_argument(
        "--answers",
        type=Path,
        required=True,
        help="Path to directory containing generated answer JSON files",
    )
    parser.add_argument(
        "--faithfulness-provider",
        type=str,
        default="openai",
        choices=["openai", "ollama", "openrouter"],
        help="LLM provider for faithfulness evaluation (default: openai)",
    )
    parser.add_argument(
        "--faithfulness-model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for faithfulness evaluation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evals"),
        help="Directory to save evaluation results (default: evals)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.answers.exists():
        logger.error(f"Answers directory not found: {args.answers}")
        return 1

    if not args.answers.is_dir():
        logger.error(f"Answers path is not a directory: {args.answers}")
        return 1

    try:
        # Evaluate all answer files in directory
        logger.info(f"Evaluating all JSON files in: {args.answers}")
        all_results = evaluate_all_answers_in_directory(
            answers_dir=args.answers,
            faithfulness_llm_provider=args.faithfulness_provider,
            faithfulness_llm_model=args.faithfulness_model,
        )

        if not all_results:
            logger.error("No results to save")
            return 1

        # Print results
        print_evaluation_results(all_results)

        # Save all results to a single JSON file
        output_path = save_all_evaluation_results(
            all_results, args.answers, args.output_dir
        )
        print(f"\n✓ All results saved to: {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
