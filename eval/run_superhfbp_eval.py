#!/usr/bin/env python3
"""
Evaluation wrapper for SuperHFBP-pruned GraphRAG using existing eval modules.

This script does NOT modify existing eval code. It implements a RAGSystemInterface
that queries GraphRAG via graphrag.api.query while constraining entities/relationships
for the pruned system to the artifacts saved by SuperHFBP.

Usage:
    python eval/run_superhfbp_eval.py --baseline workspace/output \
        --pruned workspace/pruned_superhfbp --use-pubmedqa --pubmedqa-samples 10

Notes:
- Baseline uses full GraphRAG artifacts from `workspace/output`.
- Pruned system loads communities/community_reports/text_units from baseline workspace,
  but filters entities/relationships to those in `workspace/pruned_superhfbp`.
- This aligns with SuperHFBP export (entities/relationships/text_units) without
  changing any existing eval logic.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from haystack import Document

# Use existing evaluation framework without modification
from eval.run_eval import (
    EvaluationRunner,
    TestQuestion,
    RAGSystemInterface,
    load_test_questions_from_pubmedqa,
)

# GraphRAG API
from graphrag.config.load_config import load_config
from graphrag.api.query import local_search
import asyncio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_source_doc_ids(context) -> List[str]:
    """Extract source doc IDs from GraphRAG API context structure.
    Expects context['sources'] to be a DataFrame with 'text' containing
    a first line like 'Document ID: <doc_id>'.
    """
    doc_ids: List[str] = []
    try:
        if isinstance(context, dict) and "sources" in context:
            sources = context["sources"]
            if isinstance(sources, pd.DataFrame) and "text" in sources.columns:
                for text_content in sources["text"]:
                    if text_content and isinstance(text_content, str):
                        first_line = text_content.split("\n")[0].strip()
                        if first_line.startswith("Document ID: "):
                            doc_id = first_line.split("Document ID: ")[1]
                            doc_ids.append(doc_id)
    except Exception as e:
        logger.debug(f"Failed to extract doc IDs from context: {e}")
    return doc_ids


class GraphRAGSystem(RAGSystemInterface):
    """GraphRAG-backed RAGSystemInterface.

    - Loads full GraphRAG artifacts from `workspace/output`.
    - If `system_path` points to a pruned dir, filters entities/relationships
      to the pruned set before querying.
    """

    def __init__(self, system_path: Path, system_name: str = "GraphRAG System"):
        super().__init__(system_path=system_path, system_name=system_name)
        # Derive workspace path (parent of output/pruned dir)
        self.workspace_path = system_path.parent
        self.is_pruned = (system_path.name.lower().startswith("pruned")) or (
            system_path.name.lower() == "pruned_superhfbp"
        )

        # Load baseline artifacts once (communities/reports/text_units)
        self.output_dir = self.workspace_path / "output"
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Baseline output not found: {self.output_dir}")

        self.config = load_config(self.workspace_path)
        self.entities_full = pd.read_parquet(self.output_dir / "entities.parquet")
        self.communities = pd.read_parquet(self.output_dir / "communities.parquet")
        self.community_reports = pd.read_parquet(
            self.output_dir / "community_reports.parquet"
        )
        self.text_units = pd.read_parquet(self.output_dir / "text_units.parquet")
        self.relationships_full = pd.read_parquet(
            self.output_dir / "relationships.parquet"
        )

        # If pruned, load pruned artifacts for filtering
        if self.is_pruned:
            self.pruned_entities = pd.read_parquet(
                self.system_path / "entities.parquet"
            )
            self.pruned_relationships = pd.read_parquet(
                self.system_path / "relationships.parquet"
            )
            # Prefer pruned text_units if available; otherwise fall back to baseline
            pruned_text_units_path = self.system_path / "text_units.parquet"
            if pruned_text_units_path.exists():
                self.pruned_text_units = pd.read_parquet(pruned_text_units_path)
            else:
                self.pruned_text_units = None
        else:
            self.pruned_entities = None
            self.pruned_relationships = None
            self.pruned_text_units = None

    def _build_query_artifacts(self):
        """Return artifacts appropriate for baseline or pruned queries."""
        if not self.is_pruned:
            return (
                self.config,
                self.entities_full,
                self.communities,
                self.community_reports,
                self.text_units,
                self.relationships_full,
            )

        # Filter entities and relationships using pruned artifacts
        pruned_titles = set(self.pruned_entities["title"].astype(str))
        entities = self.entities_full[self.entities_full["title"].astype(str).isin(pruned_titles)].copy()

        # relationships: filter to source/target pairs in pruned set
        pruned_edges = set(
            (str(row["source"]), str(row["target"]))
            for _, row in self.pruned_relationships[["source", "target"]].iterrows()
        )
        pruned_edges |= set((t, s) for (s, t) in pruned_edges)  # consider undirected match

        def is_pruned_edge(row) -> bool:
            edge = (str(row["source"]), str(row["target"]))
            return edge in pruned_edges

        relationships = self.relationships_full[self.relationships_full.apply(is_pruned_edge, axis=1)].copy()
        # Use pruned text_units if available
        text_units = self.pruned_text_units if self.pruned_text_units is not None else self.text_units

        return (
            self.config,
            entities,
            self.communities,
            self.community_reports,
            text_units,
            relationships,
        )

    def query(self, question: str) -> Tuple[str, List[Document]]:
        artifacts = self._build_query_artifacts()
        (
            config,
            entities,
            communities,
            community_reports,
            text_units,
            relationships,
        ) = artifacts

        # Use local_search for consistency
        answer, context = asyncio.run(local_search(
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
        ))

        # Convert context to haystack.Documents (by doc_id)
        doc_ids = extract_source_doc_ids(context)
        retrieved_docs = [Document(content=f"Document {doc_id}", meta={"doc_id": doc_id}) for doc_id in doc_ids]

        return answer, retrieved_docs


def main():
    parser = argparse.ArgumentParser(description="Evaluate SuperHFBP-pruned GraphRAG using existing eval modules")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("workspace/output"),
        help="Path to baseline GraphRAG artifacts (default: workspace/output)",
    )
    parser.add_argument(
        "--pruned",
        type=Path,
        default=Path("workspace/pruned_superhfbp"),
        help="Path to pruned GraphRAG artifacts (default: workspace/pruned_superhfbp)",
    )
    parser.add_argument(
        "--use-pubmedqa",
        action="store_true",
        help="Load PubMedQA test questions",
    )
    parser.add_argument(
        "--pubmedqa-samples",
        type=int,
        default=10,
        help="Number of PubMedQA samples (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory to save comparison outputs",
    )
    # Faithfulness evaluator configuration
    parser.add_argument(
        "--faithfulness-provider",
        type=str,
        default="openai",
        help="LLM provider for faithfulness evaluation (default: openai)",
    )
    parser.add_argument(
        "--faithfulness-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for faithfulness evaluation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--faithfulness-api-base-url",
        type=str,
        default=None,
        help="API base URL for faithfulness evaluator (optional)",
    )
    parser.add_argument(
        "--faithfulness-api-key",
        type=str,
        default=None,
        help="API key for faithfulness evaluator (optional; falls back to OPENAI_API_KEY env)",
    )

    args = parser.parse_args()

    # Load test questions
    if args.use_pubmedqa:
        test_questions: List[TestQuestion] = load_test_questions_from_pubmedqa(
            split="train", max_samples=args.pubmedqa_samples
        )
    else:
        # Minimal fallback questions if PubMedQA not used
        test_questions = [
            TestQuestion(question="What is the relationship between aspirin and platelet aggregation?"),
            TestQuestion(question="Describe mechanisms of insulin resistance in type 2 diabetes."),
            TestQuestion(question="Summarize recent trends in CRISPR applications."),
        ]

    # Resolve API key
    faithfulness_api_key = args.faithfulness_api_key or os.getenv("OPENAI_API_KEY")

    runner = EvaluationRunner(
        test_questions=test_questions,
        faithfulness_llm_provider=args.faithfulness_provider,
        faithfulness_llm_model=args.faithfulness_model,
        faithfulness_api_base_url=args.faithfulness_api_base_url,
        faithfulness_api_key=faithfulness_api_key,
    )

    baseline_system = GraphRAGSystem(system_path=args.baseline, system_name="Baseline GraphRAG")
    pruned_system = GraphRAGSystem(system_path=args.pruned, system_name="Pruned GraphRAG (SuperHFBP)")

    results = runner.compare_systems(
        baseline_system=baseline_system,
        pruned_system=pruned_system,
        output_dir=args.output_dir,
    )

    logger.info("Evaluation complete. See outputs in: %s", args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())