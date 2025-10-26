#!/usr/bin/env python3
"""
Generate answers using GraphRAG API (no CLI).

This script:
1. Loads test questions from local passages.jsonl file
2. Loads GraphRAG artifacts directly from the workspace
3. Uses graphrag.api.query (local_search or global_search)
4. Saves results to JSON for later evaluation

Usage:
    cd eval
    python generate_answers.py --workspace ../workspace --questions 50 --search-method local
    python generate_answers.py --workspace ../workspace --questions 100 --search-method global
    python generate_answers.py --workspace ../workspace --questions 100 --search-method global --parallel 5
"""

import json
import time
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from graphrag.config.load_config import load_config
from graphrag.api.query import global_search
from graphrag.api.query import local_search


# --------------------------------------------------------------------------------
# Dataclass for Questions
# --------------------------------------------------------------------------------
@dataclass
class TestQuestion:
    question: str
    ground_truth_answer: Optional[str] = None
    ground_truth_doc_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Load test questions
# --------------------------------------------------------------------------------
def load_test_questions_from_local_passages(
    passages_file: Path, max_samples: Optional[int] = None
) -> List[TestQuestion]:
    logger.info(f"Loading test questions from {passages_file}...")

    if not passages_file.exists():
        raise FileNotFoundError(f"Passages file not found: {passages_file}")

    test_questions = []
    with open(passages_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                attrs = data.get("attrs", {})
                question = attrs.get("question", "")
                ground_truth_answer = attrs.get("long_answer", "")
                final_decision = attrs.get("final_decision", "")
                if question:
                    test_questions.append(
                        TestQuestion(
                            question=question,
                            ground_truth_answer=ground_truth_answer,
                            ground_truth_doc_ids=[data.get("doc_id", "")],
                            metadata={
                                "source": "local_passages",
                                "index": i,
                                "passage_id": data.get("id", ""),
                                "doc_id": data.get("doc_id", ""),
                                "final_decision": final_decision,
                                "dataset_split": attrs.get("dataset_split", ""),
                                "dataset_config": attrs.get("dataset_config", ""),
                                "section": attrs.get("section", ""),
                                "text_snippet": (
                                    data.get("text", "")[:200] + "..."
                                    if data.get("text")
                                    else ""
                                ),
                            },
                        )
                    )
            except Exception as e:
                logger.warning(f"Skipping line {i + 1}: {e}")
    logger.info(f"Loaded {len(test_questions)} questions from local passages")
    return test_questions


# --------------------------------------------------------------------------------
# Load GraphRAG Artifacts
# --------------------------------------------------------------------------------
def load_graphrag_artifacts(workspace: Path):
    """Load all necessary GraphRAG artifacts for API querying."""
    output_dir = workspace / "output"
    config = load_config(workspace)

    logger.info(f"Loading GraphRAG artifacts from {output_dir}...")
    entities = pd.read_parquet(output_dir / "entities.parquet")
    communities = pd.read_parquet(output_dir / "communities.parquet")
    community_reports = pd.read_parquet(output_dir / "community_reports.parquet")
    text_units = pd.read_parquet(output_dir / "text_units.parquet")
    relationships = pd.read_parquet(output_dir / "relationships.parquet")

    return config, entities, communities, community_reports, text_units, relationships


# --------------------------------------------------------------------------------
# Extract Source Document IDs from Context
# --------------------------------------------------------------------------------
def extract_source_doc_ids(context: Any) -> List[str]:
    """Extract source document IDs from the context returned by GraphRAG API.

    The context is a dictionary with a 'sources' key containing a DataFrame.
    Document IDs are in the first line of the 'text' column.
    """
    doc_ids = []

    if context is None:
        return doc_ids

    try:
        # Context should be a dict with 'sources' key
        if isinstance(context, dict) and "sources" in context:
            sources = context["sources"]

            # Sources should be a DataFrame with a 'text' column
            if isinstance(sources, pd.DataFrame) and "text" in sources.columns:
                for text_content in sources["text"]:
                    if text_content and isinstance(text_content, str):
                        # Extract document ID from the first line
                        first_line = text_content.split("\n")[0].strip()
                        # First line has the format "Document ID: <doc_id>"
                        if first_line.startswith("Document ID: "):
                            doc_id = first_line.split("Document ID: ")[1]
                            doc_ids.append(doc_id)
        else:
            logger.debug(
                f"Context structure: {type(context)}, keys: {context.keys() if isinstance(context, dict) else 'N/A'}"
            )
    except Exception as e:
        logger.warning(f"Error extracting source document IDs from context: {e}")

    return doc_ids


# --------------------------------------------------------------------------------
# Query GraphRAG via API
# --------------------------------------------------------------------------------
async def query_graphrag_api(question: str, workspace: Path, method: str = "local"):
    """Use graphrag.api.query directly to answer a question."""
    try:
        config, entities, communities, community_reports, text_units, relationships = (
            load_graphrag_artifacts(workspace)
        )
        start_time = time.time()

        if method == "local":
            answer, context = await local_search(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                covariates=None,
                community_level=2,
                response_type="simple",
                query=question,
                verbose=False,
            )
        else:
            answer, context = await global_search(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                community_level=None,
                dynamic_community_selection=False,
                response_type="simple",
                query=question,
                verbose=False,
            )

        elapsed = time.time() - start_time
        retrieved_doc_ids = extract_source_doc_ids(context)

        return {
            "answer": answer,
            "context": context,
            "response_time": elapsed,
            "success": True,
            "error": None,
            "retrieved_doc_ids": retrieved_doc_ids,
        }

    except Exception as e:
        return {
            "answer": f"ERROR: {str(e)}",
            "context": None,
            "response_time": 0.0,
            "success": False,
            "error": str(e),
            "retrieved_doc_ids": [],
        }


# --------------------------------------------------------------------------------
# Generate Answers
# --------------------------------------------------------------------------------
def save_results(results: List[Dict[str, Any]], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    successful = len([r for r in results if r["success"]])
    avg_time = sum(r["response_time"] for r in results) / len(results) if results else 0
    data = {
        "metadata": {
            "total": len(results),
            "successful": successful,
            "failed": len(results) - successful,
            "success_rate": successful / len(results) if results else 0,
            "average_response_time": avg_time,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved results to {output_file}")


async def process_single_question(
    question: TestQuestion,
    question_id: int,
    total: int,
    workspace_path: Path,
    search_method: str,
) -> Dict[str, Any]:
    """Process a single question and return the result."""
    logger.info(f"({question_id}/{total}) Q: {question.question[:80]}...")
    resp = await query_graphrag_api(question.question, workspace_path, search_method)
    result = {
        "question_id": question_id,
        "question": question.question,
        "generated_answer": resp["answer"],
        "ground_truth_answer": question.ground_truth_answer,
        "ground_truth_doc_ids": question.ground_truth_doc_ids,
        "retrieved_doc_ids": resp.get("retrieved_doc_ids", []),
        "search_method": search_method,
        "response_time": resp["response_time"],
        "success": resp["success"],
        "error": resp.get("error"),
        "metadata": question.metadata or {},
        "timestamp": datetime.now().isoformat(),
    }
    if resp["success"]:
        logger.info(
            f"  ✓ Success ({resp['response_time']:.2f}s) - Retrieved {len(resp.get('retrieved_doc_ids', []))} source documents"
        )
    else:
        logger.error(f"  ✗ Failed: {resp['error']}")
    return result


async def generate_answers_api(
    questions: List[TestQuestion],
    workspace_path: Path,
    search_method: str,
    output_file: Path,
    max_parallel: int = 1,
):
    results = []
    logger.info(
        f"Generating answers for {len(questions)} questions using {search_method} search (parallelism: {max_parallel})..."
    )

    if max_parallel <= 1:
        # Sequential processing
        for i, q in enumerate(questions, 1):
            result = await process_single_question(
                q, i, len(questions), workspace_path, search_method
            )
            results.append(result)
    else:
        # Parallel processing in batches
        for batch_start in range(0, len(questions), max_parallel):
            batch_end = min(batch_start + max_parallel, len(questions))
            batch = questions[batch_start:batch_end]
            logger.info(
                f"Processing batch {batch_start // max_parallel + 1} (questions {batch_start + 1}-{batch_end})..."
            )

            tasks = [
                process_single_question(
                    q,
                    batch_start + i + 1,
                    len(questions),
                    workspace_path,
                    search_method,
                )
                for i, q in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

    save_results(results, output_file)


# --------------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate answers using GraphRAG API")
    parser.add_argument("--workspace", type=Path, default=Path("../workspace"))
    parser.add_argument(
        "--passages-file", type=Path, default=Path("../data/gold/input/passages.jsonl")
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--search-method", choices=["local", "global"], default="local")
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel queries to run (default: 1 for sequential)",
    )
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    if not args.workspace.exists():
        logger.error(f"Workspace not found: {args.workspace}")
        return 1
    if not (args.workspace / "output").exists():
        logger.error(f"No output directory found in workspace: {args.workspace}/output")
        return 1
    if not args.passages_file.exists():
        logger.error(f"Passages file not found: {args.passages_file}")
        return 1

    # Load questions
    questions = load_test_questions_from_local_passages(
        args.passages_file, args.questions
    )
    if args.preview:
        for q in questions[:5]:
            print("Q:", q.question)
            print(
                "A:", q.ground_truth_answer[:80] if q.ground_truth_answer else "[none]"
            )
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        out_dir = Path("../data/generated") / args.search_method
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            out_dir / f"generated_answers_{args.search_method}_{timestamp}.json"
        )
    else:
        output_file = args.output

    asyncio.run(
        generate_answers_api(
            questions, args.workspace, args.search_method, output_file, args.parallel
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
