#!/usr/bin/env python3
"""
Generate Comprehensive Pruning Report

This script generates a comprehensive report with:
- Executive summary
- All methods comparison table
- Pareto frontier analysis
- Best methods for different use cases
- Hybrid method results
- Recommendations
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_analysis_results() -> Dict[str, Any]:
    """Load analysis results."""
    results_dir = Path("eval/results")
    
    results = {}
    
    # Load metrics
    metrics_file = results_dir / "pruning_analysis_metrics.csv"
    if metrics_file.exists():
        results['metrics'] = pd.read_csv(metrics_file)
    
    # Load Pareto frontier
    pareto_file = results_dir / "pareto_frontier_methods.csv"
    if pareto_file.exists():
        results['pareto'] = pd.read_csv(pareto_file)
    
    # Load evaluation results
    eval_file = results_dir / "method_evaluations.json"
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            results['evaluations'] = json.load(f)
    
    # Load analysis report if exists
    report_file = results_dir / "pruning_analysis_report.md"
    if report_file.exists():
        with open(report_file, 'r') as f:
            results['analysis_report'] = f.read()
    
    return results


def generate_executive_summary(results: Dict[str, Any]) -> str:
    """Generate executive summary."""
    summary = []
    summary.append("# Executive Summary\n\n")
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    if 'metrics' in results and len(results['metrics']) > 0:
        df = results['metrics']
        
        summary.append("## Key Findings\n\n")
        
        # Best overall
        best_accuracy = df.loc[df['accuracy_score'].idxmax()]
        summary.append(f"- **Best Accuracy**: {best_accuracy['method']} achieves {best_accuracy['accuracy_score']:.2f}% accuracy score\n")
        
        # Best compute reduction
        best_compute = df.loc[df['compute_reduction'].idxmax()]
        summary.append(f"- **Best Compute Reduction**: {best_compute['method']} achieves {best_compute['compute_reduction']:.2f}% reduction\n")
        
        # Pareto optimal count
        if 'pareto' in results and len(results['pareto']) > 0:
            summary.append(f"- **Pareto Optimal Methods**: {len(results['pareto'])} methods represent optimal trade-offs\n")
        
        summary.append("\n")
    
    summary.append("## Methodology\n\n")
    summary.append("This report compares multiple graph pruning strategies:\n")
    summary.append("- **Basic Methods**: Top-k, threshold, edge pruning\n")
    summary.append("- **Advanced Methods**: CrumbTrail, KGTrimmer, POG, PathRAG\n")
    summary.append("- **Hybrid Method**: Adaptive hybrid that selects methods based on graph characteristics\n\n")
    
    summary.append("All methods were evaluated using:\n")
    summary.append("- 100+ PubMedQA samples\n")
    summary.append("- Metrics: Faithfulness, SAS, MRR, Response Time\n")
    summary.append("- Graph structure metrics: Size reduction, connectivity, components\n\n")
    
    return "\n".join(summary)


def generate_comparison_table(results: Dict[str, Any]) -> str:
    """Generate comparison table."""
    if 'metrics' not in results or len(results['metrics']) == 0:
        return "## Comparison Table\n\nNo data available.\n\n"
    
    df = results['metrics']
    
    # Select key columns for display
    display_cols = [
        'method', 'entity_reduction_pct', 'compute_reduction',
        'accuracy_score', 'faithfulness', 'sas_score', 'mrr_score'
    ]
    
    display_df = df[display_cols].copy()
    display_df.columns = [
        'Method', 'Entity Reduction %', 'Compute Reduction %',
        'Accuracy Score %', 'Faithfulness %', 'SAS Score %', 'MRR Score %'
    ]
    
    # Format numbers
    for col in display_df.columns:
        if '%' in col or 'Score' in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    
    summary = []
    summary.append("## Method Comparison Table\n\n")
    summary.append(display_df.to_markdown(index=False))
    summary.append("\n\n")
    
    return "\n".join(summary)


def generate_recommendations(results: Dict[str, Any]) -> str:
    """Generate recommendations section."""
    if 'analysis_report' in results:
        # Extract recommendations from analysis report
        report = results['analysis_report']
        if "## Use Case Recommendations" in report:
            idx = report.find("## Use Case Recommendations")
            return report[idx:] + "\n\n"
    
    # Fallback recommendations
    recommendations = []
    recommendations.append("## Recommendations\n\n")
    
    if 'metrics' in results and len(results['metrics']) > 0:
        df = results['metrics']
        
        # Latency-critical
        latency = df[df['response_time_reduction_pct'] > 50].sort_values('response_time_reduction_pct', ascending=False)
        if len(latency) > 0:
            recommendations.append("### For Latency-Critical Applications\n")
            recommendations.append(f"Use **{latency.iloc[0]['method']}** for maximum response time reduction.\n\n")
        
        # Quality-critical
        quality = df.sort_values('accuracy_score', ascending=False)
        if len(quality) > 0:
            recommendations.append("### For Quality-Critical Applications\n")
            recommendations.append(f"Use **{quality.iloc[0]['method']}** for maximum accuracy preservation.\n\n")
        
        # Balanced
        df['balance'] = (df['accuracy_score'] + df['compute_reduction']) / 2
        balanced = df.sort_values('balance', ascending=False)
        if len(balanced) > 0:
            recommendations.append("### For Balanced Trade-off\n")
            recommendations.append(f"Use **{balanced.iloc[0]['method']}** for optimal balance.\n\n")
    
    return "\n".join(recommendations)


def main():
    """Generate comprehensive report."""
    logger.info("="*80)
    logger.info("Generating Comprehensive Pruning Report")
    logger.info("="*80)
    
    # Load results
    logger.info("Loading analysis results...")
    results = load_analysis_results()
    
    if not results:
        logger.error("No analysis results found. Please run analyze_pruning_results.py first.")
        return 1
    
    # Generate report sections
    logger.info("Generating report sections...")
    
    report_sections = []
    
    # Executive summary
    report_sections.append(generate_executive_summary(results))
    
    # Comparison table
    report_sections.append(generate_comparison_table(results))
    
    # Pareto frontier analysis
    if 'pareto' in results and len(results['pareto']) > 0:
        report_sections.append("## Pareto Frontier Analysis\n\n")
        report_sections.append("The following methods represent Pareto optimal trade-offs:\n\n")
        pareto_df = results['pareto']
        report_sections.append(pareto_df.to_markdown(index=False))
        report_sections.append("\n\n")
    
    # Recommendations
    report_sections.append(generate_recommendations(results))
    
    # Hybrid method section
    report_sections.append("## Adaptive Hybrid Method\n\n")
    report_sections.append("The adaptive hybrid method analyzes graph characteristics and ")
    report_sections.append("selects the optimal combination of pruning methods.\n\n")
    report_sections.append("### Strategy Selection\n\n")
    report_sections.append("- **Small graphs (<5k nodes)**: PathRAG or POG for path-based pruning\n")
    report_sections.append("- **Medium graphs (5k-15k nodes)**: KGTrimmer for balanced reduction\n")
    report_sections.append("- **Large graphs (>15k nodes)**: Multi-stage (KGTrimmer → PathRAG)\n")
    report_sections.append("- **Dense graphs**: Edge pruning + node pruning\n")
    report_sections.append("- **Fragmented graphs**: CrumbTrail to preserve connectivity\n\n")
    
    # Combine and save
    report_content = "\n".join(report_sections)
    
    output_file = Path("eval/results/comprehensive_pruning_report.md")
    with open(output_file, 'w') as f:
        f.write(report_content)
    
    logger.info(f"✓ Saved comprehensive report to {output_file}")
    
    logger.info("\n" + "="*80)
    logger.info("Complete!")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

