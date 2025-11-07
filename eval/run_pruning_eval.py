#!/usr/bin/env python3
"""
Evaluation runner for all pruning strategies.

Runs all 7 pruning strategies and evaluates each on 5 PubMedQA samples.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from haystack import Document

# Use existing evaluation framework
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
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

# Pruning
from pruning.prune_graph import GraphPruner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_source_doc_ids(context) -> List[str]:
    """Extract document IDs from GraphRAG context."""
    doc_ids = []
    try:
        if isinstance(context, dict):
            if 'source_documents' in context:
                for doc in context['source_documents']:
                    if isinstance(doc, dict) and 'id' in doc:
                        doc_ids.append(str(doc['id']))
            elif 'documents' in context:
                for doc in context['documents']:
                    if isinstance(doc, dict) and 'id' in doc:
                        doc_ids.append(str(doc['id']))
        elif isinstance(context, list):
            for item in context:
                if isinstance(item, dict) and 'id' in item:
                    doc_ids.append(str(item['id']))
    except Exception as e:
        logger.warning(f"Failed to extract doc IDs from context: {e}")
    return doc_ids


class GraphRAGSystem(RAGSystemInterface):
    """
    GraphRAG system interface for evaluation.
    
    Loads pruned artifacts and queries GraphRAG.
    """
    
    def __init__(self, system_path: Path, system_name: str = "GraphRAG System"):
        self.system_path = system_path
        self.system_name = system_name
        
        # Derive workspace path (parent of output/pruned dir)
        self.workspace_path = system_path.parent
        self.is_pruned = (system_path.name.lower().startswith("pruned")) or (
            "pruned" in system_path.name.lower()
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
            if (self.system_path / "entities.parquet").exists():
                self.pruned_entities = pd.read_parquet(
                    self.system_path / "entities.parquet"
                )
            else:
                logger.warning(f"No pruned entities found at {self.system_path}, using baseline")
                self.pruned_entities = self.entities_full
            
            if (self.system_path / "relationships.parquet").exists():
                self.pruned_relationships = pd.read_parquet(
                    self.system_path / "relationships.parquet"
                )
            else:
                logger.warning(f"No pruned relationships found at {self.system_path}, using baseline")
                self.pruned_relationships = self.relationships_full
            
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


def get_graph_stats(entities_df: pd.DataFrame, relationships_df: pd.DataFrame) -> dict:
    """Get statistics about the graph."""
    return {
        'num_entities': len(entities_df),
        'num_relationships': len(relationships_df),
        'unique_sources': relationships_df['source'].nunique() if len(relationships_df) > 0 else 0,
        'unique_targets': relationships_df['target'].nunique() if len(relationships_df) > 0 else 0,
    }


def run_all_pruning_strategies(workspace_path: Path, baseline_dir: Path):
    """
    Run all pruning strategies and save results.
    
    Args:
        workspace_path: Path to workspace directory
        baseline_dir: Path to baseline artifacts
        
    Returns:
        Dictionary with pruning results for each strategy
    """
    strategies = [
        {
            'name': 'crumbtrail',
            'method': 'apply_crumbtrail_pipeline',
            'params': {}
        },
        {
            'name': 'kgtrimmer',
            'method': 'apply_kgtrimmer_pipeline',
            'params': {}
        },
        {
            'name': 'pathrag_hybrid',
            'method': 'apply_pathrag_hybrid_pipeline',
            'params': {}
        },
        {
            'name': 'pog_hybrid',
            'method': 'apply_pog_hybrid_pipeline',
            'params': {}
        },
        {
            'name': 'adaptive_multi_strategy',
            'method': 'apply_adaptive_multi_strategy_pipeline',
            'params': {}
        },
    ]
    
    # Load baseline stats
    from pruning.scoring_utils import load_graphrag_artifacts
    baseline_entities, baseline_relationships, baseline_communities = load_graphrag_artifacts(baseline_dir)
    baseline_stats = get_graph_stats(baseline_entities, baseline_relationships)
    
    logger.info("\n" + "="*80)
    logger.info("BASELINE GRAPH STATISTICS")
    logger.info("="*80)
    logger.info(f"  Entities:        {baseline_stats['num_entities']:,}")
    logger.info(f"  Relationships:   {baseline_stats['num_relationships']:,}")
    logger.info(f"  Unique Sources:  {baseline_stats['unique_sources']:,}")
    logger.info(f"  Unique Targets:  {baseline_stats['unique_targets']:,}")
    logger.info("="*80 + "\n")
    
    pruning_results = {}  # Initialize results dictionary
    total_strategies = len(strategies)
    for idx, strategy in enumerate(strategies, 1):
        logger.info("\n" + "="*80)
        logger.info(f"PRUNING STRATEGY {idx}/{total_strategies}: {strategy['name'].upper()}")
        logger.info("="*80)
        
        output_dir = workspace_path / f"pruned_{strategy['name']}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"🔧 Method: {strategy['method']}")
        logger.info(f"⚙️  Parameters: {strategy['params']}")
        
        # Create a new pruner instance for each strategy
        logger.info(f"\n🔨 Initializing GraphPruner...")
        strategy_pruner = GraphPruner(baseline_dir, output_dir)
        
        # Get the method and call it
        method = getattr(strategy_pruner, strategy['method'])
        try:
            logger.info(f"\n📊 BEFORE PRUNING:")
            logger.info(f"  Entities:        {baseline_stats['num_entities']:,}")
            logger.info(f"  Relationships:   {baseline_stats['num_relationships']:,}")
            
            logger.info(f"\n🔄 Running {strategy['name']} pruning algorithm...")
            logger.info(f"   This may take several minutes depending on graph size...")
            pruned_artifacts = method(**strategy['params'])
            
            # Get after stats
            pruned_entities = pruned_artifacts['entities']
            pruned_relationships = pruned_artifacts['relationships']
            after_stats = get_graph_stats(pruned_entities, pruned_relationships)
            
            # Calculate reductions
            entity_reduction = ((baseline_stats['num_entities'] - after_stats['num_entities']) / baseline_stats['num_entities'] * 100) if baseline_stats['num_entities'] > 0 else 0
            relationship_reduction = ((baseline_stats['num_relationships'] - after_stats['num_relationships']) / baseline_stats['num_relationships'] * 100) if baseline_stats['num_relationships'] > 0 else 0
            
            logger.info(f"\n📊 AFTER PRUNING:")
            logger.info(f"  Entities:        {after_stats['num_entities']:,} ({entity_reduction:.1f}% reduction)")
            logger.info(f"  Relationships:   {after_stats['num_relationships']:,} ({relationship_reduction:.1f}% reduction)")
            logger.info(f"  Unique Sources:  {after_stats['unique_sources']:,}")
            logger.info(f"  Unique Targets:  {after_stats['unique_targets']:,}")
            
            logger.info(f"\n✅ {strategy['name']} pruning completed successfully!")
            logger.info(f"   Results saved to: {output_dir}")
            logger.info("="*80)
            
            # Store results for summary
            pruning_results[strategy['name']] = {
                'num_entities': after_stats['num_entities'],
                'num_relationships': after_stats['num_relationships'],
                'entity_reduction': entity_reduction,
                'edge_reduction': relationship_reduction
            }
            
        except Exception as e:
            logger.error(f"\n❌ {strategy['name']} pruning failed: {e}")
            import traceback
            traceback.print_exc()
            logger.info("="*80)
    
    return pruning_results


def main():
    parser = argparse.ArgumentParser(description="Run all pruning strategies and evaluate")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace"),
        help="Path to workspace directory (default: workspace)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline artifacts (default: workspace/output)",
    )
    parser.add_argument(
        "--pubmedqa-samples",
        type=int,
        default=5,
        help="Number of PubMedQA samples to use (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--skip-pruning",
        action="store_true",
        help="Skip pruning step (use existing pruned artifacts)",
    )
    parser.add_argument(
        "--pruning-only",
        action="store_true",
        help="Only run pruning, skip evaluation",
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
    
    args = parser.parse_args()
    
    workspace_path = args.workspace
    baseline_dir = args.baseline or workspace_path / "output"
    
    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return 1
    
    # Load baseline stats for later use
    from pruning.scoring_utils import load_graphrag_artifacts
    baseline_entities, baseline_relationships, _ = load_graphrag_artifacts(baseline_dir)
    baseline_stats = get_graph_stats(baseline_entities, baseline_relationships)
    
    # Step 1: Run all pruning strategies
    pruning_results = {}
    if not args.skip_pruning:
        logger.info("="*80)
        logger.info("STEP 1: RUNNING ALL PRUNING STRATEGIES")
        logger.info("="*80)
        pruning_results = run_all_pruning_strategies(workspace_path, baseline_dir)
        
        # Show final summary
        logger.info("\n" + "="*80)
        logger.info("PRUNING SUMMARY - ALL STRATEGIES")
        logger.info("="*80)
        logger.info(f"\n{'Strategy':<30} {'Entities':<15} {'Relationships':<15} {'Entity %':<12} {'Edge %':<12}")
        logger.info("-" * 80)
        
        for strategy_name, stats in pruning_results.items():
            entity_reduction = stats.get('entity_reduction', 0)
            edge_reduction = stats.get('edge_reduction', 0)
            logger.info(f"{strategy_name:<30} "
                       f"{stats['num_entities']:<15,} "
                       f"{stats['num_relationships']:<15,} "
                       f"{entity_reduction:<12.1f}% "
                       f"{edge_reduction:<12.1f}%")
        
        logger.info("\n" + "="*80)
        logger.info("BASELINE:")
        logger.info(f"  Entities:      {baseline_stats['num_entities']:,}")
        logger.info(f"  Relationships: {baseline_stats['num_relationships']:,}")
        logger.info("="*80 + "\n")
        
        # If --pruning-only flag, exit here
        if args.pruning_only:
            logger.info("Pruning completed. Exiting (--pruning-only flag set).")
            return 0
    else:
        logger.info("Skipping pruning step (using existing artifacts)")
    
    # Step 2: Load test questions
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LOADING TEST QUESTIONS")
    logger.info("="*80)
    test_questions = load_test_questions_from_pubmedqa(
        split="train",
        max_samples=args.pubmedqa_samples,
    )
    logger.info(f"\n✓ Loaded {len(test_questions)} test questions from PubMedQA")
    logger.info("="*80)
    
    # Step 3: Evaluate baseline
    logger.info("\n" + "="*80)
    logger.info("STEP 3: EVALUATING BASELINE SYSTEM")
    logger.info("="*80)
    baseline_system = GraphRAGSystem(baseline_dir, "Baseline")
    runner = EvaluationRunner(
        test_questions=test_questions,
        faithfulness_llm_provider=args.faithfulness_provider,
        faithfulness_llm_model=args.faithfulness_model,
    )
    baseline_results = runner.evaluate_system(baseline_system, run_name="baseline")
    
    baseline_metrics = baseline_results['metrics']
    logger.info("\n" + "="*80)
    logger.info("BASELINE EVALUATION RESULTS")
    logger.info("="*80)
    logger.info(f"  Faithfulness Score: {baseline_metrics.faithfulness_score:.4f}")
    if baseline_metrics.sas_score is not None:
        logger.info(f"  SAS Score:          {baseline_metrics.sas_score:.4f}")
    if baseline_metrics.mrr_score is not None:
        logger.info(f"  MRR Score:          {baseline_metrics.mrr_score:.4f}")
    logger.info(f"  Avg Response Time:  {baseline_metrics.avg_response_time:.2f}s")
    logger.info(f"  Total Queries:      {baseline_metrics.total_queries}")
    logger.info("="*80)
    
    # Step 4: Evaluate all pruned strategies
    strategies = [
        'crumbtrail',
        'kgtrimmer',
        'pathrag_hybrid',
        'pog_hybrid',
        'adaptive_multi_strategy',
    ]
    
    all_results = {'baseline': baseline_results}
    
    for strategy_name in strategies:
        pruned_dir = workspace_path / f"pruned_{strategy_name}"
        if not pruned_dir.exists():
            logger.warning(f"\n⚠️  Pruned directory not found: {pruned_dir}, skipping {strategy_name}")
            continue
        
        logger.info("\n" + "="*80)
        logger.info(f"EVALUATING: {strategy_name.upper()}")
        logger.info("="*80)
        
        # Load and show pruned stats
        try:
            from pruning.scoring_utils import load_graphrag_artifacts
            pruned_entities, pruned_relationships, _ = load_graphrag_artifacts(pruned_dir)
            pruned_stats = get_graph_stats(pruned_entities, pruned_relationships)
            
            logger.info(f"\n📊 Pruned Graph Statistics:")
            logger.info(f"  Entities:        {pruned_stats['num_entities']:,}")
            logger.info(f"  Relationships:   {pruned_stats['num_relationships']:,}")
            
            entity_reduction = ((baseline_stats['num_entities'] - pruned_stats['num_entities']) / baseline_stats['num_entities'] * 100) if baseline_stats['num_entities'] > 0 else 0
            relationship_reduction = ((baseline_stats['num_relationships'] - pruned_stats['num_relationships']) / baseline_stats['num_relationships'] * 100) if baseline_stats['num_relationships'] > 0 else 0
            
            logger.info(f"  Entity Reduction: {entity_reduction:.1f}%")
            logger.info(f"  Edge Reduction:   {relationship_reduction:.1f}%")
        except Exception as e:
            logger.warning(f"Could not load pruned stats: {e}")
        
        pruned_system = GraphRAGSystem(pruned_dir, f"Pruned ({strategy_name})")
        try:
            logger.info(f"\n🔄 Running evaluation queries...")
            results = runner.evaluate_system(pruned_system, run_name=strategy_name)
            all_results[strategy_name] = results
            
            metrics = results['metrics']
            logger.info(f"\n📊 Evaluation Results:")
            logger.info(f"  Faithfulness Score: {metrics.faithfulness_score:.4f}")
            if metrics.sas_score is not None:
                logger.info(f"  SAS Score:          {metrics.sas_score:.4f}")
            if metrics.mrr_score is not None:
                logger.info(f"  MRR Score:          {metrics.mrr_score:.4f}")
            logger.info(f"  Avg Response Time:  {metrics.avg_response_time:.2f}s")
            
            # Compare to baseline
            faithfulness_change = ((metrics.faithfulness_score - baseline_metrics.faithfulness_score) / baseline_metrics.faithfulness_score * 100) if baseline_metrics.faithfulness_score > 0 else 0
            logger.info(f"\n📈 vs Baseline:")
            logger.info(f"  Faithfulness:      {faithfulness_change:+.2f}%")
            if metrics.sas_score is not None and baseline_metrics.sas_score is not None:
                sas_change = ((metrics.sas_score - baseline_metrics.sas_score) / baseline_metrics.sas_score * 100) if baseline_metrics.sas_score > 0 else 0
                logger.info(f"  SAS:                {sas_change:+.2f}%")
            
            logger.info("="*80)
        except Exception as e:
            logger.error(f"\n❌ Evaluation failed for {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
            logger.info("="*80)
    
    # Step 5: Generate comparison report
    logger.info("\n" + "="*80)
    logger.info("STEP 5: GENERATING COMPARISON REPORT")
    logger.info("="*80)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary DataFrame with before/after stats
    summary_data = []
    for name, results in all_results.items():
        metrics = results['metrics']
        
        # Get graph stats
        if name == 'baseline':
            graph_stats = baseline_stats
        else:
            pruned_dir = workspace_path / f"pruned_{name}"
            try:
                from pruning.scoring_utils import load_graphrag_artifacts
                pruned_entities, pruned_relationships, _ = load_graphrag_artifacts(pruned_dir)
                graph_stats = get_graph_stats(pruned_entities, pruned_relationships)
            except:
                graph_stats = {'num_entities': 0, 'num_relationships': 0}
        
        entity_reduction = ((baseline_stats['num_entities'] - graph_stats['num_entities']) / baseline_stats['num_entities'] * 100) if baseline_stats['num_entities'] > 0 else 0
        relationship_reduction = ((baseline_stats['num_relationships'] - graph_stats['num_relationships']) / baseline_stats['num_relationships'] * 100) if baseline_stats['num_relationships'] > 0 else 0
        
        faithfulness_change = ((metrics.faithfulness_score - baseline_metrics.faithfulness_score) / baseline_metrics.faithfulness_score * 100) if baseline_metrics.faithfulness_score > 0 else 0
        
        summary_data.append({
            'strategy': name,
            'num_entities': graph_stats['num_entities'],
            'num_relationships': graph_stats['num_relationships'],
            'entity_reduction_pct': entity_reduction,
            'relationship_reduction_pct': relationship_reduction,
            'faithfulness_score': metrics.faithfulness_score,
            'faithfulness_change_pct': faithfulness_change,
            'sas_score': metrics.sas_score,
            'mrr_score': metrics.mrr_score,
            'avg_response_time': metrics.avg_response_time,
            'total_queries': metrics.total_queries,
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('faithfulness_score', ascending=False)
    
    # Save summary
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = args.output_dir / f"pruning_comparison_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\n✓ Saved comparison summary to {summary_path}")
    
    # Print detailed summary
    logger.info("\n" + "="*80)
    logger.info("FINAL COMPARISON SUMMARY")
    logger.info("="*80)
    logger.info(f"\n{'Strategy':<25} {'Entities':<12} {'Edges':<12} {'Entity %':<10} {'Edge %':<10} {'Faith':<8} {'Δ Faith':<10}")
    logger.info("-" * 80)
    for _, row in summary_df.iterrows():
        logger.info(f"{row['strategy']:<25} "
                   f"{row['num_entities']:<12,} "
                   f"{row['num_relationships']:<12,} "
                   f"{row['entity_reduction_pct']:<10.1f}% "
                   f"{row['relationship_reduction_pct']:<10.1f}% "
                   f"{row['faithfulness_score']:<8.4f} "
                   f"{row['faithfulness_change_pct']:+.2f}%")
    
    logger.info("\n" + "="*80)
    logger.info("DETAILED METRICS")
    logger.info("="*80)
    logger.info(f"\n{summary_df.to_string()}\n")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

