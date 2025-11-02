#!/usr/bin/env python3
"""
Generate answers using GraphRAG API for MedHop dataset.

This script:
1. Loads MedHop test questions from passages.jsonl file (query/answer/candidates format)
2. Loads GraphRAG artifacts directly from the workspace
3. Uses graphrag.api.query (local_search or global_search)
4. Constrains answers to candidate selection from the provided list
5. Saves results to JSON for later evaluation

Usage:
    cd eval
    python generate_answers_MH.py --workspace ../workspace --questions 50 --search-method local
    python generate_answers_MH.py --workspace ../workspace --questions 100 --search-method global
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Load test questions
# --------------------------------------------------------------------------------
def load_test_questions_from_local_passages(
    passages_file: Path, max_samples: Optional[int] = None
) -> List[TestQuestion]:
    """Load MedHop test questions from passages.jsonl file.
    
    Expected format:
    {
        "id": "MH_train_0::support_0",
        "doc_id": "MH_train_0",
        "text": "...",
        "attrs": {
            "query": "interacts_with DB00773?",
            "answer": "DB00072",
            "candidates": ["DB00072", "DB00294", ...],
            "support_index": 0,
            "total_supports": 47
        }
    }
    """
    logger.info(f"Loading MedHop test questions from {passages_file}...")

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
                
                # MedHop format only
                question = attrs.get("query", "")
                answer = attrs.get("answer", "")
                candidates = attrs.get("candidates", [])
                
                if not question:
                    logger.warning(f"Skipping line {i + 1}: missing query field")
                    continue

                test_questions.append(
                    TestQuestion(
                        question=question,
                        ground_truth_answer=answer,
                        ground_truth_doc_ids=[data.get("doc_id", "")],
                        metadata={
                            "source": "medhop",
                            "index": i,
                            "id": data.get("id", ""),
                            "doc_id": data.get("doc_id", ""),
                            "candidates": candidates,
                            "support_index": attrs.get("support_index", -1),
                            "total_supports": attrs.get("total_supports", 0),
                            "text_snippet": data.get("text", "")[:200] + "..." if data.get("text") else "",
                        },
                    )
                )
            except Exception as e:
                logger.warning(f"Skipping line {i + 1}: {e}")
    logger.info(f"Loaded {len(test_questions)} questions from MedHop passages")
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
    
    # Load documents for ID mapping
    documents = pd.read_parquet(output_dir / "documents.parquet")

    return config, entities, communities, community_reports, text_units, relationships, documents


# --------------------------------------------------------------------------------
# Extract Source Document IDs from Context
# --------------------------------------------------------------------------------
def extract_source_doc_ids(context: Any, text_units: pd.DataFrame, documents: pd.DataFrame) -> List[str]:
    """Extract source document IDs from the context returned by GraphRAG API.
    
    The context is a dictionary with a 'sources' key containing a DataFrame.
    Maps text unit IDs to original document titles (e.g., MH_train_0).
    """
    doc_ids = []
    
    if context is None:
        return doc_ids
    
    try:
        if isinstance(context, dict) and 'sources' in context:
            sources = context['sources']
            
            if isinstance(sources, pd.DataFrame) and 'id' in sources.columns:
                text_unit_ids = sources['id'].tolist()
                
                # Map text unit IDs to document IDs
                for tu_id in text_unit_ids:
                    if tu_id:
                        # Convert to int if it's a numeric string
                        try:
                            tu_id_int = int(tu_id) if isinstance(tu_id, str) else tu_id
                        except (ValueError, TypeError):
                            tu_id_int = tu_id
                        
                        # Find text unit by human_readable_id
                        tu_rows = text_units[text_units['human_readable_id'] == tu_id_int]
                        if not tu_rows.empty:
                            # Get document_ids array from text unit
                            doc_id_hashes = tu_rows.iloc[0]['document_ids']
                            if doc_id_hashes is not None and len(doc_id_hashes) > 0:
                                # Map document hash to original title
                                for doc_hash in doc_id_hashes:
                                    doc_rows = documents[documents['id'] == doc_hash]
                                    if not doc_rows.empty:
                                        title = doc_rows.iloc[0]['title']
                                        # Extract base doc_id from title (e.g., "MH_train_0" from "MH_train_0_support_5.txt")
                                        if title:
                                            # Remove .txt extension and support suffix
                                            base_id = title.replace('.txt', '')
                                            # Extract just MH_train_X part
                                            if '_support_' in base_id:
                                                base_id = base_id.split('_support_')[0]
                                            doc_ids.append(base_id)
          
    except Exception as e:
        logger.error(f"Error extracting source document IDs: {e}", exc_info=True)
      
    return list(dict.fromkeys(doc_ids))  # Remove duplicates while preserving order

# --------------------------------------------------------------------------------
# Query GraphRAG via API
# --------------------------------------------------------------------------------
async def query_graphrag_api(question: str, workspace: Path, method: str = "local"):
    """Use graphrag.api.query directly to answer a question."""
    try:
        config, entities, communities, community_reports, text_units, relationships, documents = load_graphrag_artifacts(workspace)
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
        retrieved_doc_ids = extract_source_doc_ids(context, text_units, documents)
        
        return {
            "answer": answer,
            "context": context,
            "response_time": elapsed,
            "success": True,
            "error": None,
            "retrieved_doc_ids": retrieved_doc_ids
        }

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return {
            "answer": f"ERROR: {str(e)}",
            "context": None,
            "response_time": 0.0,
            "success": False,
            "error": str(e),
            "retrieved_doc_ids": []
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


async def generate_answers_api(
    questions: List[TestQuestion], workspace_path: Path, search_method: str, output_file: Path
):
    results = []
    logger.info(f"Generating answers for {len(questions)} questions using {search_method} search...")

    for i, q in enumerate(questions, 1):
        logger.info(f"({i}/{len(questions)}) Q: {q.question[:80]}...")
        
        # Get candidates from metadata
        candidates = q.metadata.get("candidates", [])
        
        # Format query to constrain GraphRAG to choose from candidates
        if candidates:
            formatted_query = (
                f"{q.question}\n\n"
                f"You must select ONLY ONE answer from these DrugBank IDs:\n"
                f"{chr(10).join(f'  - {c}' for c in candidates)}\n\n"
                f"Return only the selected DrugBank ID with no additional text or explanation."
            )
        else:
            formatted_query = q.question

        resp = await query_graphrag_api(formatted_query, workspace_path, search_method)
        
        # Extract candidate ID from response
        generated_answer = resp["answer"]
        extracted_candidate = None
        
        # Try to find a candidate in the response
        if candidates:
            for candidate in candidates:
                if candidate in generated_answer:
                    extracted_candidate = candidate
                    break
            
            # If no candidate found, log warning
            if not extracted_candidate:
                logger.warning(f"  ⚠ Could not extract candidate from response: {generated_answer[:100]}")
                extracted_candidate = generated_answer
        else:
            extracted_candidate = generated_answer

        result = {
            "question_id": i,
            "question": q.question,
            "generated_answer": extracted_candidate or generated_answer,
            "raw_response": generated_answer,  # Keep full response for debugging
            "ground_truth_answer": q.ground_truth_answer,
            "ground_truth_doc_ids": q.ground_truth_doc_ids,
            "retrieved_doc_ids": resp.get("retrieved_doc_ids", []),
            "candidates": candidates,
            "is_valid_candidate": extracted_candidate in candidates if candidates else True,
            "is_correct": extracted_candidate == q.ground_truth_answer if extracted_candidate else False,
            "search_method": search_method,
            "response_time": resp["response_time"],
            "success": resp["success"],
            "error": resp.get("error"),
            "metadata": q.metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        
        if resp["success"]:
            correctness = "✓ CORRECT" if result["is_correct"] else "✗ WRONG"
            logger.info(f"  ✓ Success ({resp['response_time']:.2f}s) - {correctness} - Retrieved {len(resp.get('retrieved_doc_ids', []))} docs")
            logger.info(f"    Predicted: {extracted_candidate} | Truth: {q.ground_truth_answer}")
        else:
            logger.error(f"  ✗ Failed: {resp['error']}")
        
        results.append(result)

    save_results(results, output_file)


# --------------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate answers using GraphRAG API")
    parser.add_argument("--workspace", type=Path, default=Path("../workspace"))
    parser.add_argument("--passages-file", type=Path, default=Path("../data/gold/input/passages.jsonl"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--search-method", choices=["local", "global"], default="local")
    parser.add_argument("--questions", type=int, default=20)
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
    questions = load_test_questions_from_local_passages(args.passages_file, args.questions)
    if args.preview:
        for q in questions[:5]:
            print("Q:", q.question)
            print("A:", q.ground_truth_answer[:80] if q.ground_truth_answer else "[none]")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        out_dir = Path("data/generated") / args.search_method
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / f"generated_answers_{args.search_method}_{timestamp}.json"
    else:
        output_file = args.output

    asyncio.run(generate_answers_api(questions, args.workspace, args.search_method, output_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
