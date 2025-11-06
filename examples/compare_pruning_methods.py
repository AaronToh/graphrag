#!/usr/bin/env python3
"""
Compare Pruning Methods - Comprehensive Analysis Script

Compares CrumbTrail against other pruning strategies using the pruning metadata
from each output directory. Generates comparison tables and visualizations.

Usage:
    python examples/compare_pruning_methods.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys


def load_pruning_metadata(artifacts_path: Path) -> Dict:
    """Load pruning metadata from a configuration directory."""
    metadata_file = artifacts_path / "pruning_metadata.json"

    if not metadata_file.exists():
        return None

    with open(metadata_file, 'r') as f:
        return json.load(f)


def extract_stats(metadata: Dict, config_name: str) -> Dict:
    """Extract key statistics from metadata."""
    if metadata is None:
        return {
            'name': config_name,
            'status': 'NOT RUN',
            'entities': None,
            'relationships': None,
            'entity_reduction_pct': None,
            'edge_reduction_pct': None,
            'avg_degree': None,
            'largest_component_pct': None,
            'num_components': None
        }

    baseline = metadata.get('baseline_stats', {})
    pruned = metadata.get('pruned_stats', {})

    # Calculate reductions
    entity_reduction = None
    edge_reduction = None
    if baseline.get('num_entities') and pruned.get('num_entities'):
        entity_reduction = 100 * (1 - pruned['num_entities'] / baseline['num_entities'])
    if baseline.get('num_relationships') and pruned.get('num_relationships'):
        edge_reduction = 100 * (1 - pruned['num_relationships'] / baseline['num_relationships'])

    return {
        'name': config_name,
        'status': 'COMPLETE',
        'entities': pruned.get('num_entities'),
        'relationships': pruned.get('num_relationships'),
        'entity_reduction_pct': entity_reduction,
        'edge_reduction_pct': edge_reduction,
        'avg_degree': pruned.get('avg_degree'),
        'largest_component_pct': pruned.get('largest_component_pct'),
        'num_components': pruned.get('num_weakly_connected_components'),
        'layers': metadata.get('layers_created', 'N/A'),
        'cycles_broken': metadata.get('cycles_broken', 'N/A'),
        'runtime_seconds': metadata.get('runtime_seconds', 'N/A')
    }


def load_baseline_stats(baseline_path: Path) -> Dict:
    """Load baseline statistics."""
    metadata_file = baseline_path / "pruning_metadata.json"

    # Baseline might not have metadata, try to load from entities/relationships directly
    if not metadata_file.exists():
        try:
            import pyarrow.parquet as pq
            entities = pq.read_table(baseline_path / "entities.parquet")
            relationships = pq.read_table(baseline_path / "relationships.parquet")

            return {
                'name': 'Baseline (No Pruning)',
                'status': 'N/A',
                'entities': len(entities),
                'relationships': len(relationships),
                'entity_reduction_pct': 0.0,
                'edge_reduction_pct': 0.0,
                'avg_degree': None,
                'largest_component_pct': None,
                'num_components': None,
                'layers': 'N/A',
                'cycles_broken': 'N/A',
                'runtime_seconds': 'N/A'
            }
        except Exception as e:
            print(f"Warning: Could not load baseline stats: {e}")
            return None

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    baseline = metadata.get('baseline_stats', {})

    return {
        'name': 'Baseline (No Pruning)',
        'status': 'N/A',
        'entities': baseline.get('num_entities'),
        'relationships': baseline.get('num_relationships'),
        'entity_reduction_pct': 0.0,
        'edge_reduction_pct': 0.0,
        'avg_degree': baseline.get('avg_degree'),
        'largest_component_pct': baseline.get('largest_component_pct'),
        'num_components': baseline.get('num_weakly_connected_components'),
        'layers': 'N/A',
        'cycles_broken': 'N/A',
        'runtime_seconds': 'N/A'
    }


def format_number(value, decimal_places=1, suffix=''):
    """Format number for display."""
    if value is None or value == 'N/A':
        return 'N/A'
    try:
        if isinstance(value, str):
            return value
        return f"{value:.{decimal_places}f}{suffix}"
    except (ValueError, TypeError):
        return str(value)


def print_comparison_table(stats_list: List[Dict]):
    """Print formatted comparison table."""

    print("\n" + "="*120)
    print("PRUNING METHODS COMPARISON")
    print("="*120)

    # Graph Structure Metrics
    print("\n📊 GRAPH STRUCTURE METRICS")
    print("-"*120)

    headers = ["Method", "Entities", "Relationships", "Entity ↓%", "Edge ↓%", "Avg Degree", "Components"]
    col_widths = [30, 12, 15, 12, 12, 12, 12]

    # Print header
    header_row = ""
    for header, width in zip(headers, col_widths):
        header_row += f"{header:<{width}}"
    print(header_row)
    print("-"*120)

    # Print rows
    for stats in stats_list:
        if stats['status'] == 'NOT RUN':
            print(f"{stats['name']:<30} {'NOT RUN':<89}")
            continue

        row = ""
        row += f"{stats['name']:<30}"
        row += f"{format_number(stats['entities'], 0):<12}"
        row += f"{format_number(stats['relationships'], 0):<15}"
        row += f"{format_number(stats['entity_reduction_pct'], 1, '%'):<12}"
        row += f"{format_number(stats['edge_reduction_pct'], 1, '%'):<12}"
        row += f"{format_number(stats['avg_degree'], 2):<12}"
        row += f"{format_number(stats['num_components'], 0):<12}"
        print(row)

    # Connectivity Metrics
    print("\n🔗 CONNECTIVITY METRICS")
    print("-"*120)

    headers2 = ["Method", "Largest Component %", "Layers", "Cycles Broken", "Runtime (s)"]
    col_widths2 = [30, 25, 15, 20, 15]

    header_row = ""
    for header, width in zip(headers2, col_widths2):
        header_row += f"{header:<{width}}"
    print(header_row)
    print("-"*120)

    for stats in stats_list:
        if stats['status'] == 'NOT RUN':
            print(f"{stats['name']:<30} {'NOT RUN':<74}")
            continue

        row = ""
        row += f"{stats['name']:<30}"
        row += f"{format_number(stats['largest_component_pct'], 1, '%'):<25}"
        row += f"{format_number(stats['layers'], 0):<15}"
        row += f"{format_number(stats['cycles_broken'], 0):<20}"
        row += f"{format_number(stats['runtime_seconds'], 2):<15}"
        print(row)

    print("="*120)


def print_crumbtrail_comparison(stats_list: List[Dict]):
    """Print focused comparison of CrumbTrail variants."""

    crumbtrail_configs = [s for s in stats_list if 'crumbtrail' in s['name'].lower()]

    if not crumbtrail_configs:
        print("\n⚠️  No CrumbTrail configurations found.")
        return

    print("\n" + "="*100)
    print("CRUMBTRAIL CONFIGURATION COMPARISON")
    print("="*100)

    headers = ["Configuration", "Protected %", "Entities ↓%", "Edges ↓%", "Largest Comp %", "Runtime (s)"]
    col_widths = [30, 15, 15, 15, 18, 15]

    header_row = ""
    for header, width in zip(headers, col_widths):
        header_row += f"{header:<{width}}"
    print(header_row)
    print("-"*100)

    for stats in crumbtrail_configs:
        if stats['status'] == 'NOT RUN':
            continue

        # Extract protected percentage from name
        protected_pct = 'N/A'
        if 'aggressive' in stats['name'].lower():
            protected_pct = '10%'
        elif 'conservative' in stats['name'].lower():
            protected_pct = '30%'
        elif 'default' in stats['name'].lower():
            protected_pct = '20%'

        row = ""
        row += f"{stats['name']:<30}"
        row += f"{protected_pct:<15}"
        row += f"{format_number(stats['entity_reduction_pct'], 1, '%'):<15}"
        row += f"{format_number(stats['edge_reduction_pct'], 1, '%'):<15}"
        row += f"{format_number(stats['largest_component_pct'], 1, '%'):<18}"
        row += f"{format_number(stats['runtime_seconds'], 2):<15}"
        print(row)

    print("="*100)

    # Print trade-off insights
    print("\n💡 KEY INSIGHTS:")
    print("   • Lower protected % → Higher reduction, faster runtime")
    print("   • Higher protected % → Better connectivity preservation, lower fragmentation")
    print("   • Optimal protected % depends on downstream task requirements")


def print_recommendations(stats_list: List[Dict]):
    """Print recommendations based on the comparison."""

    print("\n" + "="*100)
    print("📋 RECOMMENDATIONS")
    print("="*100)

    completed = [s for s in stats_list if s['status'] == 'COMPLETE']

    if not completed:
        print("\n⚠️  No pruning configurations have been run yet.")
        print("\nTo get started:")
        print("  1. Run: python examples/crumbtrail_quickstart.py")
        print("  2. Run: python examples/kgtrimmer_quickstart.py")
        print("  3. Run: python examples/pog_quickstart.py")
        print("  4. Run: python examples/pathrag_quickstart.py")
        return

    # Find best reduction
    max_reduction = max([s['entity_reduction_pct'] for s in completed if s['entity_reduction_pct'] is not None])
    best_reduction = [s for s in completed if s.get('entity_reduction_pct') == max_reduction][0]

    # Find best connectivity preservation
    connectivity_scores = [(s, s.get('largest_component_pct', 0)) for s in completed if s.get('largest_component_pct') is not None]
    if connectivity_scores:
        best_connectivity = max(connectivity_scores, key=lambda x: x[1])[0]
    else:
        best_connectivity = None

    print(f"\n🏆 Maximum Reduction: {best_reduction['name']}")
    print(f"   Reduces {format_number(max_reduction, 1, '%')} entities, {format_number(best_reduction['edge_reduction_pct'], 1, '%')} edges")

    if best_connectivity:
        print(f"\n🔗 Best Connectivity: {best_connectivity['name']}")
        print(f"   Maintains {format_number(best_connectivity['largest_component_pct'], 1, '%')} in largest component")

    # Method-specific recommendations
    crumbtrail_configs = [s for s in completed if 'crumbtrail' in s['name'].lower()]
    kgtrimmer_configs = [s for s in completed if 'kgtrimmer' in s['name'].lower()]
    pog_configs = [s for s in completed if 'pog' in s['name'].lower()]
    pathrag_configs = [s for s in completed if 'pathrag' in s['name'].lower()]

    # Load evaluation metrics if available
    eval_results = {}
    eval_file = Path("eval/results/method_evaluations.json")
    if eval_file.exists():
        import json
        with open(eval_file, 'r') as f:
            eval_results = json.load(f)
    
    print("\n🎯 Use Case Recommendations:")
    print("   • Latency-Critical Applications → Use aggressive pruning (10-20% reduction)")
    print("   • Quality-Critical Applications → Use conservative pruning (5-10% reduction)")
    print("   • Balanced Trade-off → Start with CrumbTrail default (20% protected)")
    print("   • Hierarchical Knowledge → CrumbTrail preserves hierarchical structure better")
    print("   • Community-Aware Pruning → KGTrimmer uses collective + holistic importance")
    print("   • Path-Based Queries → PathRAG for high reduction with path preservation")
    print("   • LLM-Guided Pruning → POG uses semantic path evaluation")
    print("   • Adaptive Selection → Use Adaptive Hybrid for optimal method selection")
    
    # Add evaluation metrics if available
    if eval_results:
        print("\n📊 Evaluation Metrics Available:")
        print("   Run 'python analyze_pruning_results.py' for detailed analysis")
        print("   Run 'python generate_pruning_report.py' for comprehensive report")

    print("\n📊 Method Comparison:")
    if crumbtrail_configs:
        avg_reduction = np.mean([s['entity_reduction_pct'] for s in crumbtrail_configs if s['entity_reduction_pct'] is not None])
        print(f"   • CrumbTrail: Average {format_number(avg_reduction, 1, '%')} reduction")
    if kgtrimmer_configs:
        avg_reduction = np.mean([s['entity_reduction_pct'] for s in kgtrimmer_configs if s['entity_reduction_pct'] is not None])
        print(f"   • KGTrimmer: Average {format_number(avg_reduction, 1, '%')} reduction")
    if pog_configs:
        avg_reduction = np.mean([s['entity_reduction_pct'] for s in pog_configs if s['entity_reduction_pct'] is not None])
        print(f"   • POG: Average {format_number(avg_reduction, 1, '%')} reduction")
    if pathrag_configs:
        avg_reduction = np.mean([s['entity_reduction_pct'] for s in pathrag_configs if s['entity_reduction_pct'] is not None])
        print(f"   • PathRAG: Average {format_number(avg_reduction, 1, '%')} reduction")

    print("\n📊 Next Steps:")
    print("   1. Run evaluation: python eval/run_eval.py --ablation --use-pubmedqa --pubmedqa-samples 50")
    print("   2. Compare quality metrics (Faithfulness, SAS) across configurations")
    print("   3. Measure query latency improvements with pruned graphs")

    print("="*100 + "\n")


def main():
    """Main comparison script."""

    print("\n" + "="*100)
    print("🔍 GRAPHRAG PRUNING METHODS COMPARISON")
    print("="*100)

    # Load ablation config
    config_path = Path("eval/ablation_config.json")
    if not config_path.exists():
        print(f"❌ Error: {config_path} not found")
        sys.exit(1)

    with open(config_path, 'r') as f:
        configs = json.load(f)

    # Collect statistics from all configurations
    stats_list = []

    # Add baseline
    baseline_path = Path("workspace/output")
    baseline_stats = load_baseline_stats(baseline_path)
    if baseline_stats:
        stats_list.append(baseline_stats)

    # Add pruned configurations
    for config in configs:
        if config['name'] == 'baseline':
            continue

        artifacts_path = Path(config['artifacts_path'])
        metadata = load_pruning_metadata(artifacts_path)
        stats = extract_stats(metadata, config['name'])
        stats_list.append(stats)
    
    # Check for adaptive hybrid results
    hybrid_path = Path("workspace/output/pruned_adaptive_hybrid")
    if hybrid_path.exists():
        metadata = load_pruning_metadata(hybrid_path)
        if metadata:
            stats = extract_stats(metadata, "adaptive_hybrid")
            stats_list.append(stats)

    # Print comparisons
    print_comparison_table(stats_list)
    print_crumbtrail_comparison(stats_list)
    print_recommendations(stats_list)

    # Export to CSV for further analysis
    output_file = Path("eval/results/pruning_comparison.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(stats_list)
    df.to_csv(output_file, index=False)

    print(f"\n💾 Detailed results exported to: {output_file}")
    print(f"   Use this file for plotting and statistical analysis.\n")


if __name__ == "__main__":
    main()
