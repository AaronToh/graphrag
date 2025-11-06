#!/usr/bin/env python3
"""
Analyze Pruning Results and Create Pareto Frontier

This script:
1. Loads all evaluation results
2. Computes compute vs accuracy metrics
3. Creates Pareto frontier visualization
4. Generates recommendations
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_evaluation_results() -> Dict[str, Any]:
    """Load all evaluation results."""
    results_file = Path("eval/results/method_evaluations.json")
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.error("Please run eval_all_pruning_methods.py first")
        return {}
    
    with open(results_file, 'r') as f:
        return json.load(f)


def load_pruning_metadata() -> Dict[str, Any]:
    """Load pruning metadata for all methods."""
    metadata = {}
    
    # Load from ablation config
    config_path = Path("eval/ablation_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            configs = json.load(f)
            for config in configs:
                method_name = config['name']
                artifacts_path = Path(config['artifacts_path'])
                metadata_file = artifacts_path / "pruning_metadata.json"
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata[method_name] = json.load(f)
    
    # Also check existing pruned directories
    baseline_dir = Path("workspace/output")
    for pruned_dir in baseline_dir.glob("pruned_*"):
        metadata_file = pruned_dir / "pruning_metadata.json"
        if metadata_file.exists():
            method_name = pruned_dir.name.replace("pruned_", "")
            with open(metadata_file, 'r') as f:
                metadata[method_name] = json.load(f)
    
    return metadata


def compute_metrics(eval_results: Dict, metadata: Dict) -> pd.DataFrame:
    """
    Compute compute vs accuracy metrics for all methods.

    Returns:
        DataFrame with columns: method, compute_reduction, accuracy_score, etc.
    """
    rows = []
    
    for method_name, eval_data in eval_results.items():
        if not eval_data or 'comparison' not in eval_data:
            continue
        
        # Get pruning stats
        method_metadata = metadata.get(method_name, {})
        pruned_stats = method_metadata.get('pruned_stats', {})
        baseline_stats = method_metadata.get('baseline_stats', {})
        
        # Compute metrics
        entity_reduction = 0.0
        if baseline_stats.get('num_entities') and pruned_stats.get('num_entities'):
            entity_reduction = 100 * (1 - pruned_stats['num_entities'] / baseline_stats['num_entities'])
        
        response_time_reduction = eval_data['comparison'].get('response_time_change_pct', 0.0)
        
        # Accuracy metrics
        pruned_metrics = eval_data.get('pruned', {})
        faithfulness = pruned_metrics.get('faithfulness_score', 0.0)
        sas = pruned_metrics.get('sas_score', 0.0)
        mrr = pruned_metrics.get('mrr_score', 0.0)
        
        # Combined accuracy score (weighted average)
        accuracy_score = (0.4 * faithfulness + 0.4 * sas + 0.2 * mrr) * 100
        
        # Combined compute reduction (weighted: 70% size, 30% time)
        compute_reduction = 0.7 * entity_reduction + 0.3 * abs(response_time_reduction)
        
        rows.append({
            'method': method_name,
            'entity_reduction_pct': entity_reduction,
            'response_time_reduction_pct': abs(response_time_reduction),
            'compute_reduction': compute_reduction,
            'faithfulness': faithfulness * 100,
            'sas_score': sas * 100,
            'mrr_score': mrr * 100,
            'accuracy_score': accuracy_score,
            'num_entities': pruned_stats.get('num_entities', 0),
            'num_relationships': pruned_stats.get('num_relationships', 0),
        })
    
    return pd.DataFrame(rows)


def find_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find Pareto optimal points (maximize accuracy, maximize compute reduction).

    Returns:
        DataFrame with Pareto optimal methods
    """
    if len(df) == 0:
        return df
    
    pareto_points = []
    
    for idx, row in df.iterrows():
        is_pareto = True
        
        # Check if this point is dominated by any other
        for other_idx, other_row in df.iterrows():
            if idx == other_idx:
                continue
            
            # A point is dominated if another has both higher accuracy AND higher compute reduction
            if (other_row['accuracy_score'] >= row['accuracy_score'] and 
                other_row['compute_reduction'] >= row['compute_reduction'] and
                (other_row['accuracy_score'] > row['accuracy_score'] or 
                 other_row['compute_reduction'] > row['compute_reduction'])):
                is_pareto = False
                break
        
        if is_pareto:
            pareto_points.append(row)
    
    return pd.DataFrame(pareto_points)


def create_visualizations(df: pd.DataFrame, pareto_df: pd.DataFrame):
    """Create visualization plots."""
    logger.info("Creating visualizations...")
    
    # Create output directory
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Pareto Frontier Plot
    fig = go.Figure()
    
    # All points
    fig.add_trace(go.Scatter(
        x=df['compute_reduction'],
        y=df['accuracy_score'],
        mode='markers+text',
        text=df['method'],
        textposition="top center",
        name='All Methods',
        marker=dict(size=10, color='lightblue', opacity=0.7)
    ))
    
    # Pareto frontier
    if len(pareto_df) > 0:
        pareto_sorted = pareto_df.sort_values('compute_reduction')
        fig.add_trace(go.Scatter(
            x=pareto_sorted['compute_reduction'],
            y=pareto_sorted['accuracy_score'],
            mode='lines+markers',
            name='Pareto Frontier',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=12, color='red')
        ))
    
    fig.update_layout(
        title='Pareto Frontier: Compute Reduction vs Accuracy',
        xaxis_title='Compute Reduction (%)',
        yaxis_title='Accuracy Score (%)',
        hovermode='closest',
        width=1000,
        height=700
    )
    
    fig.write_html(output_dir / "pareto_frontier.html")
    logger.info(f"✓ Saved Pareto frontier to {output_dir / 'pareto_frontier.html'}")
    
    # 2. Scatter plot: Entity Reduction vs Accuracy
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=df['entity_reduction_pct'],
        y=df['accuracy_score'],
        mode='markers+text',
        text=df['method'],
        textposition="top center",
        name='Methods',
        marker=dict(
            size=10,
            color=df['compute_reduction'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Compute Reduction")
        )
    ))
    
    fig2.update_layout(
        title='Entity Reduction vs Accuracy',
        xaxis_title='Entity Reduction (%)',
        yaxis_title='Accuracy Score (%)',
        width=1000,
        height=700
    )
    
    fig2.write_html(output_dir / "entity_reduction_vs_accuracy.html")
    logger.info(f"✓ Saved entity reduction plot to {output_dir / 'entity_reduction_vs_accuracy.html'}")
    
    # 3. Bar chart: Method comparison
    fig3 = go.Figure()
    
    methods = df['method'].tolist()
    x_pos = np.arange(len(methods))
    
    fig3.add_trace(go.Bar(
        x=methods,
        y=df['accuracy_score'],
        name='Accuracy Score',
        marker_color='lightblue'
    ))
    
    fig3.add_trace(go.Bar(
        x=methods,
        y=df['compute_reduction'],
        name='Compute Reduction',
        marker_color='lightcoral'
    ))
    
    fig3.update_layout(
        title='Method Comparison: Accuracy vs Compute Reduction',
        xaxis_title='Method',
        yaxis_title='Score (%)',
        barmode='group',
        xaxis_tickangle=-45,
        width=1200,
        height=600
    )
    
    fig3.write_html(output_dir / "method_comparison.html")
    logger.info(f"✓ Saved method comparison to {output_dir / 'method_comparison.html'}")


def generate_recommendations(df: pd.DataFrame, pareto_df: pd.DataFrame) -> str:
    """Generate recommendations based on analysis."""
    recommendations = []
    recommendations.append("# Pruning Method Recommendations\n")
    recommendations.append("Based on comprehensive evaluation and Pareto frontier analysis.\n\n")
    
    # Best overall (highest accuracy with good compute reduction)
    if len(df) > 0:
        best_overall = df.loc[df['accuracy_score'].idxmax()]
        recommendations.append(f"## Best Overall Accuracy\n")
        recommendations.append(f"- **Method**: {best_overall['method']}\n")
        recommendations.append(f"- **Accuracy Score**: {best_overall['accuracy_score']:.2f}%\n")
        recommendations.append(f"- **Compute Reduction**: {best_overall['compute_reduction']:.2f}%\n\n")
    
    # Best compute reduction (highest reduction with acceptable accuracy)
    if len(df) > 0:
        # Filter methods with accuracy > 50%
        high_accuracy = df[df['accuracy_score'] > 50]
        if len(high_accuracy) > 0:
            best_compute = high_accuracy.loc[high_accuracy['compute_reduction'].idxmax()]
            recommendations.append(f"## Best Compute Reduction (Accuracy > 50%)\n")
            recommendations.append(f"- **Method**: {best_compute['method']}\n")
            recommendations.append(f"- **Compute Reduction**: {best_compute['compute_reduction']:.2f}%\n")
            recommendations.append(f"- **Accuracy Score**: {best_compute['accuracy_score']:.2f}%\n\n")
    
    # Pareto optimal methods
    if len(pareto_df) > 0:
        recommendations.append("## Pareto Optimal Methods\n")
        recommendations.append("These methods represent optimal trade-offs:\n\n")
        for _, row in pareto_df.iterrows():
            recommendations.append(f"- **{row['method']}**: {row['accuracy_score']:.2f}% accuracy, {row['compute_reduction']:.2f}% compute reduction\n")
        recommendations.append("\n")
    
    # Use case recommendations
    recommendations.append("## Use Case Recommendations\n\n")
    
    if len(df) > 0:
        # Latency-critical
        latency_critical = df[df['response_time_reduction_pct'] > 50].sort_values('response_time_reduction_pct', ascending=False)
        if len(latency_critical) > 0:
            recommendations.append("### Latency-Critical Applications\n")
            recommendations.append(f"- **{latency_critical.iloc[0]['method']}**: {latency_critical.iloc[0]['response_time_reduction_pct']:.2f}% response time reduction\n\n")
        
        # Quality-critical
        quality_critical = df.sort_values('accuracy_score', ascending=False)
        if len(quality_critical) > 0:
            recommendations.append("### Quality-Critical Applications\n")
            recommendations.append(f"- **{quality_critical.iloc[0]['method']}**: {quality_critical.iloc[0]['accuracy_score']:.2f}% accuracy score\n\n")
        
        # Balanced
        df['balance_score'] = (df['accuracy_score'] + df['compute_reduction']) / 2
        balanced = df.sort_values('balance_score', ascending=False)
        if len(balanced) > 0:
            recommendations.append("### Balanced Trade-off\n")
            recommendations.append(f"- **{balanced.iloc[0]['method']}**: Best balance of accuracy and compute reduction\n\n")
    
    return "\n".join(recommendations)


def main():
    """Main analysis function."""
    logger.info("="*80)
    logger.info("Analyzing Pruning Results")
    logger.info("="*80)
    
    # Load data
    logger.info("Loading evaluation results...")
    eval_results = load_evaluation_results()
    if not eval_results:
        logger.error("No evaluation results found. Please run eval_all_pruning_methods.py first.")
        return 1
    
    logger.info("Loading pruning metadata...")
    metadata = load_pruning_metadata()
    
    # Compute metrics
    logger.info("Computing metrics...")
    df = compute_metrics(eval_results, metadata)
    
    if len(df) == 0:
        logger.error("No valid metrics computed. Check evaluation results.")
        return 1
    
    logger.info(f"Computed metrics for {len(df)} methods")
    
    # Find Pareto frontier
    logger.info("Finding Pareto frontier...")
    pareto_df = find_pareto_frontier(df)
    logger.info(f"Found {len(pareto_df)} Pareto optimal methods")
    
    # Save metrics
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "pruning_analysis_metrics.csv", index=False)
    pareto_df.to_csv(output_dir / "pareto_frontier_methods.csv", index=False)
    logger.info(f"✓ Saved metrics to {output_dir / 'pruning_analysis_metrics.csv'}")
    
    # Create visualizations
    create_visualizations(df, pareto_df)
    
    # Generate recommendations
    logger.info("Generating recommendations...")
    recommendations = generate_recommendations(df, pareto_df)
    
    # Save report
    report_file = output_dir / "pruning_analysis_report.md"
    with open(report_file, 'w') as f:
        f.write("# Pruning Method Analysis Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Pareto Optimal Methods\n\n")
        f.write(pareto_df.to_markdown(index=False))
        f.write("\n\n")
        f.write(recommendations)
    
    logger.info(f"✓ Saved analysis report to {report_file}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Analysis Summary")
    logger.info("="*80)
    logger.info(f"\nTotal methods analyzed: {len(df)}")
    logger.info(f"Pareto optimal methods: {len(pareto_df)}")
    logger.info(f"\nBest accuracy: {df.loc[df['accuracy_score'].idxmax(), 'method']} ({df['accuracy_score'].max():.2f}%)")
    logger.info(f"Best compute reduction: {df.loc[df['compute_reduction'].idxmax(), 'method']} ({df['compute_reduction'].max():.2f}%)")
    
    logger.info("\n" + "="*80)
    logger.info("Complete!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Visualizations: pareto_frontier.html, entity_reduction_vs_accuracy.html, method_comparison.html")
    logger.info("Report: pruning_analysis_report.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

