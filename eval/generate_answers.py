#!/usr/bin/env python3
"""
Generate answers using GraphRAG for local test questions.

This script:
1. Loads test questions from local passages.jsonl file
2. Queries your GraphRAG index to generate answers
3. Saves results in JSON format for later evaluation

Usage:
    cd eval
    python generate_answers.py --workspace ../workspace --questions 50 --search-method local
    python generate_answers.py --workspace ../workspace --questions 100 --search-method global
"""

import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from dataclasses import dataclass


@dataclass
class TestQuestion:
    """Structure for a test question with ground truth."""
    question: str
    ground_truth_answer: Optional[str] = None
    ground_truth_doc_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_questions_from_local_passages(
    passages_file: Path = Path("../data/gold/input/passages.jsonl"),
    max_samples: Optional[int] = None
) -> List[TestQuestion]:
    """
    Load test questions from local passages.jsonl file.
    
    Args:
        passages_file: Path to passages.jsonl file
        max_samples: Maximum number of samples to load (None = all)
        
    Returns:
        List of TestQuestion objects
    """
    logger.info(f"Loading test questions from {passages_file}...")
    
    if not passages_file.exists():
        raise FileNotFoundError(f"Passages file not found: {passages_file}")
    
    test_questions = []
    
    with open(passages_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
                
            try:
                data = json.loads(line.strip())
                attrs = data.get('attrs', {})
                
                # Extract question and answer from attrs
                question = attrs.get('question', '')
                ground_truth_answer = attrs.get('long_answer', '')
                final_decision = attrs.get('final_decision', '')
                
                if question:  # Only include entries with questions
                    test_q = TestQuestion(
                        question=question,
                        ground_truth_answer=ground_truth_answer,
                        ground_truth_doc_ids=[data.get('doc_id', '')],
                        metadata={
                            'source': 'local_passages',
                            'index': i,
                            'passage_id': data.get('id', ''),
                            'doc_id': data.get('doc_id', ''),
                            'final_decision': final_decision,
                            'dataset_split': attrs.get('dataset_split', ''),
                            'dataset_config': attrs.get('dataset_config', ''),
                            'section': attrs.get('section', ''),
                            'text_snippet': data.get('text', '')[:200] + '...' if data.get('text') else ''
                        }
                    )
                    test_questions.append(test_q)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line {i+1}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error processing line {i+1}: {e}")
                continue
    
    logger.info(f"Loaded {len(test_questions)} questions from local passages")
    return test_questions


def query_graphrag_cli(
    question: str, 
    workspace_path: Path, 
    method: str = "local"
) -> Dict[str, Any]:
    """
    Query GraphRAG using CLI interface.
    
    Args:
        question: Question to ask
        workspace_path: Path to GraphRAG workspace
        method: "local" or "global"
        
    Returns:
        Dict with answer and metadata
    """
    try:
        # Build GraphRAG command using pixi
        cmd = [
            "pixi", "run", "python", "-m", "graphrag", "query",
            "--root", str(workspace_path),
            "--method", method,
            "--query", question
        ]
        
        # Execute command
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=str(workspace_path.parent)  # Run from parent directory
        )
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            return {
                "answer": result.stdout.strip(),
                "response_time": elapsed_time,
                "success": True,
                "error": None
            }
        else:
            return {
                "answer": f"ERROR: {result.stderr.strip()}",
                "response_time": elapsed_time,
                "success": False,
                "error": result.stderr.strip()
            }
            
    except subprocess.TimeoutExpired:
        return {
            "answer": "ERROR: Query timeout",
            "response_time": 120.0,
            "success": False,
            "error": "Query timeout after 120 seconds"
        }
    except Exception as e:
        return {
            "answer": f"ERROR: {str(e)}",
            "response_time": 0.0,
            "success": False,
            "error": str(e)
        }


def generate_answers_cli(
    questions: List[TestQuestion],
    workspace_path: Path,
    search_method: str = "local",
    output_file: Path = None
) -> List[Dict[str, Any]]:
    """Generate answers using GraphRAG CLI."""
    results = []
    
    logger.info(f"Generating answers for {len(questions)} questions using {search_method} search...")
    
    for i, test_q in enumerate(questions, 1):
        logger.info(f"Question {i}/{len(questions)}: {test_q.question[:60]}...")
        
        # Query GraphRAG
        response = query_graphrag_cli(
            question=test_q.question,
            workspace_path=workspace_path,
            method=search_method
        )
        
        # Create result record
        result = {
            "question_id": i - 1,
            "question": test_q.question,
            "generated_answer": response["answer"],
            "ground_truth_answer": test_q.ground_truth_answer,
            "ground_truth_doc_ids": test_q.ground_truth_doc_ids,
            "search_method": search_method,
            "response_time": response["response_time"],
            "success": response["success"],
            "metadata": test_q.metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        if not response["success"]:
            result["error"] = response["error"]
            logger.error(f"  ✗ Failed: {response['error']}")
        else:
            logger.info(f"  ✓ Generated answer ({response['response_time']:.2f}s)")
        
        results.append(result)
    
    # Save results if output file specified
    if output_file:
        save_results(results, output_file)
    
    return results


def save_results(results: List[Dict[str, Any]], output_file: Path):
    """Save results to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create summary statistics
    successful = len([r for r in results if r["success"]])
    failed = len(results) - successful
    avg_time = sum(r["response_time"] for r in results) / len(results) if results else 0
    
    output_data = {
        "metadata": {
            "total_questions": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(results) if results else 0,
            "average_response_time": avg_time,
            "generation_timestamp": datetime.now().isoformat(),
            "search_method": results[0]["search_method"] if results else None
        },
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved {len(results)} results to {output_file}")


def preview_questions(questions: List[TestQuestion], count: int = 5):
    """Preview the first few questions for verification."""
    logger.info(f"\nPreviewing first {min(count, len(questions))} questions:")
    logger.info("=" * 80)
    
    for i, q in enumerate(questions[:count], 1):
        logger.info(f"\nQuestion {i}:")
        logger.info(f"  Q: {q.question}")
        logger.info(f"  A: {q.ground_truth_answer[:100]}..." if q.ground_truth_answer else "  A: [No answer]")
        logger.info(f"  Doc ID: {q.ground_truth_doc_ids}")
        logger.info(f"  Decision: {q.metadata.get('final_decision', 'N/A')}")
    
    logger.info("=" * 80)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate answers using GraphRAG from local questions")
    parser.add_argument(
        "--workspace", 
        type=Path, 
        default=Path("../workspace"),
        help="Path to GraphRAG workspace (default: ../workspace)"
    )
    parser.add_argument(
        "--passages-file",
        type=Path,
        default=Path("../data/gold/input/passages.jsonl"),
        help="Path to passages.jsonl file (default: ../data/gold/input/passages.jsonl)"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=None,  # Will be auto-generated based on search method
        help="Output file for generated answers (default: auto-generated in ../data/generated/)"
    )
    parser.add_argument(
        "--search-method",
        choices=["local", "global"],
        default="local",
        help="GraphRAG search method (default: local)"
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=20,
        help="Number of questions to process (default: 20)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview questions without generating answers"
    )
    
    args = parser.parse_args()
    
    # Validate workspace
    if not args.workspace.exists():
        logger.error(f"Workspace not found: {args.workspace}")
        return 1
    
    output_dir = args.workspace / "output"
    if not output_dir.exists():
        logger.error(f"GraphRAG output directory not found: {output_dir}")
        logger.info("Run indexing first: python ingest/build_index.py")
        return 1
    
    # Validate passages file
    if not args.passages_file.exists():
        logger.error(f"Passages file not found: {args.passages_file}")
        return 1
    
    try:
        # Load test questions from local passages
        logger.info(f"Loading {args.questions} questions from local passages...")
        test_questions = load_test_questions_from_local_passages(
            passages_file=args.passages_file,
            max_samples=args.questions
        )
        
        if not test_questions:
            logger.error("No test questions loaded")
            return 1
        
        logger.info(f"Loaded {len(test_questions)} questions")
        
        # Preview questions if requested
        if args.preview:
            preview_questions(test_questions, count=10)
            return 0
        
        # Generate timestamp for output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Auto-generate output path if not provided
        if args.output is None:
            output_dir = Path("../data/generated") / args.search_method
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"generated_answers_{args.search_method}_{timestamp}.json"
        else:
            output_file = args.output.parent / f"{args.output.stem}_{args.search_method}_{timestamp}.json"
        
        # Generate answers
        results = generate_answers_cli(
            questions=test_questions,
            workspace_path=args.workspace,
            search_method=args.search_method,
            output_file=output_file
        )
        
        # Print summary
        successful = len([r for r in results if r["success"]])
        failed = len(results) - successful
        avg_time = sum(r["response_time"] for r in results) / len(results) if results else 0
        
        logger.info("\n" + "="*60)
        logger.info("ANSWER GENERATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total questions: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/len(results)*100:.1f}%" if results else "0%")
        logger.info(f"Average response time: {avg_time:.2f}s")
        logger.info(f"Search method: {args.search_method}")
        logger.info(f"Results saved to: {output_file}")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())