#!/usr/bin/env python3
"""
Evaluate generated answers from JSON file using RAG evaluation metrics.

This script:
1. Loads generated answers from a JSON file (created by generate_answers.py)
2. Evaluates them using Faithfulness, Document MRR, and Semantic Answer Similarity
3. Reports aggregated and detailed evaluation metrics
4. Saves evaluation results to a JSON file in the evals directory

Usage:
    cd eval
    python evaluate_answers.py --answers ../data/generated/local/generated_answers_local_20251023_035626.json
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
        logger.info(f"Metadata: Total={metadata.get('total')}, Successful={metadata.get('successful')}, "
                   f"Failed={metadata.get('failed')}, Success Rate={metadata.get('success_rate', 0)*100:.1f}%")
    
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
            logger.warning(f"Skipping failed question {result.get('question_id')}: {result.get('error')}")
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
# Main evaluation function
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
    test_cases, ground_truth_answers, ground_truth_documents = convert_to_test_cases(data)
    
    if not test_cases:
        logger.error("No valid test cases to evaluate!")
        return {}
    
    # Run evaluation
    logger.info("Starting evaluation...")
    logger.info(f"Using {faithfulness_llm_provider}/{faithfulness_llm_model} for faithfulness evaluation")
    
    results = evaluate_rag_pipeline(
        test_cases=test_cases,
        ground_truth_answers=ground_truth_answers,
        ground_truth_documents=ground_truth_documents,
        faithfulness_llm_provider=faithfulness_llm_provider,
        faithfulness_llm_model=faithfulness_llm_model,
    )
    
    return results


# --------------------------------------------------------------------------------
# Pretty print results
# --------------------------------------------------------------------------------
def print_evaluation_results(results: Dict[str, Any]):
    """Pretty print evaluation results.
    
    Args:
        results: Dictionary with evaluation results from evaluate_rag_pipeline
    """
    if not results:
        logger.error("No results to display")
        return
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    # Print aggregated metrics
    print("\n" + "=" * 60)
    print("Aggregated Metrics")
    print("=" * 60)
    
    agg_report = results.get("aggregated_report", {})
    if isinstance(agg_report, dict) and "metrics" in agg_report and "score" in agg_report:
        print(f"\n{'Metric':<25} {'Score':>10}")
        print("-" * 37)
        for metric, score in zip(agg_report["metrics"], agg_report["score"]):
            print(f"{metric:<25} {score:>10.4f}")
    else:
        print(agg_report)
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("Detailed Results")
    print("=" * 60)
    print()
    
    detailed_results = results.get("detailed_results")
    if detailed_results is not None:
        print(detailed_results)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


# --------------------------------------------------------------------------------
# Save evaluation results to JSON
# --------------------------------------------------------------------------------
def save_evaluation_results(results: Dict[str, Any], source_file: Path, output_dir: Path = Path("evals")) -> Path:
    """Save evaluation results to JSON file.
    
    Args:
        results: Dictionary with evaluation results from evaluate_rag_pipeline
        source_file: Path to the source answers JSON file
        output_dir: Directory to save evaluation results (default: evals)
        
    Returns:
        Path to saved JSON file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename based on source file and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_basename = source_file.stem  # filename without extension
    output_filename = f"eval_{source_basename}_{timestamp}.json"
    output_path = output_dir / output_filename
    
    # Convert pandas DataFrame to dict for JSON serialization
    detailed_results = results.get("detailed_results")
    if detailed_results is not None:
        # Convert DataFrame to list of dicts
        detailed_results_dict = detailed_results.to_dict(orient="records")
    else:
        detailed_results_dict = None
    
    # Prepare data to save
    save_data = {
        "metadata": {
            "source_file": str(source_file),
            "evaluation_timestamp": datetime.now().isoformat(),
            "num_test_cases": len(detailed_results_dict) if detailed_results_dict else 0,
        },
        "aggregated_report": results.get("aggregated_report", {}),
        "detailed_results": detailed_results_dict,
    }
    
    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation results saved to: {output_path}")
    return output_path


# --------------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers from JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_answers.py --answers ../data/generated/local/generated_answers_local_20251023_035626.json
  python evaluate_answers.py --answers results.json --faithfulness-provider openai --faithfulness-model gpt-4o-mini
        """
    )
    parser.add_argument(
        "--answers",
        type=Path,
        required=True,
        help="Path to JSON file with generated answers (created by generate_answers.py)"
    )
    parser.add_argument(
        "--faithfulness-provider",
        type=str,
        default="openai",
        choices=["openai", "ollama", "openrouter"],
        help="LLM provider for faithfulness evaluation (default: openai)"
    )
    parser.add_argument(
        "--faithfulness-model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for faithfulness evaluation (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evals"),
        help="Directory to save evaluation results (default: evals)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.answers.exists():
        logger.error(f"Answers file not found: {args.answers}")
        return 1
    
    try:
        # Evaluate answers
        print("starting")
        results = evaluate_answers_from_json(
            json_file=args.answers,
            faithfulness_llm_provider=args.faithfulness_provider,
            faithfulness_llm_model=args.faithfulness_model,
        )
        
        # Print results
        print_evaluation_results(results)
        
        # Save results to JSON
        output_path = save_evaluation_results(results, args.answers, args.output_dir)
        print(f"\n✓ Results saved to: {output_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
