#!/usr/bin/env python3
"""
Show Pruning Results Summary

Displays a summary of all pruning methods with their graph statistics.
"""

import json
import pandas as pd
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
    }


def load_baseline_stats(baseline_path: Path) -> Dict:
    """Load baseline statistics."""
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
            'num_components': None
        }
    except Exception as e:
        print(f"Warning: Could not load baseline stats: {e}")
        return None


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


def print_results_table(stats_list: List[Dict]):
    """Print formatted results table."""
    print("\n" + "="*120)
    print("PRUNING METHODS RESULTS SUMMARY")
    print("="*120)
    
    print("\n📊 GRAPH STRUCTURE METRICS")
    print("-"*120)
    
    headers = ["Method", "Status", "Entities", "Relationships", "Entity ↓%", "Edge ↓%", "Avg Degree", "Components"]
    col_widths = [30, 12, 12, 15, 12, 12, 12, 12]
    
    # Print header
    header_row = ""
    for header, width in zip(headers, col_widths):
        header_row += f"{header:<{width}}"
    print(header_row)
    print("-"*120)
    
    for stats in stats_list:
        row = ""
        row += f"{stats['name']:<30}"
        row += f"{stats['status']:<12}"
        row += f"{format_number(stats['entities'], 0):<12}"
        row += f"{format_number(stats['relationships'], 0):<15}"
        row += f"{format_number(stats['entity_reduction_pct'], 1, '%'):<12}"
        row += f"{format_number(stats['edge_reduction_pct'], 1, '%'):<12}"
        row += f"{format_number(stats['avg_degree'], 2):<12}"
        row += f"{format_number(stats['num_components'], 0):<12}"
        print(row)
    
    print("="*120)


def print_summary(stats_list: List[Dict]):
    """Print summary statistics."""
    completed = [s for s in stats_list if s['status'] == 'COMPLETE' and s['entity_reduction_pct'] is not None]
    
    if not completed:
        print("\n⚠️  No pruning results found.")
        return
    
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    
    # Reduction ranges
    reductions = [s['entity_reduction_pct'] for s in completed]
    print(f"\n📉 Entity Reduction Range: {min(reductions):.1f}% - {max(reductions):.1f}%")
    print(f"   Average: {sum(reductions)/len(reductions):.1f}%")
    
    # Best reduction
    best_reduction = max(completed, key=lambda x: x['entity_reduction_pct'])
    print(f"\n🏆 Maximum Reduction: {best_reduction['name']}")
    print(f"   {best_reduction['entity_reduction_pct']:.1f}% entities, {best_reduction['edge_reduction_pct']:.1f}% edges")
    
    # Best connectivity
    connectivity_scores = [(s, s.get('largest_component_pct', 0)) for s in completed if s.get('largest_component_pct') is not None]
    if connectivity_scores:
        best_connectivity = max(connectivity_scores, key=lambda x: x[1])[0]
        print(f"\n🔗 Best Connectivity: {best_connectivity['name']}")
        print(f"   {best_connectivity['largest_component_pct']:.1f}% in largest component")
    
    # Method categories
    print("\n📊 Method Categories:")
    crumbtrail = [s for s in completed if 'crumbtrail' in s['name'].lower()]
    kgtrimmer = [s for s in completed if 'kgtrimmer' in s['name'].lower()]
    pog = [s for s in completed if 'pog' in s['name'].lower()]
    pathrag = [s for s in completed if 'pathrag' in s['name'].lower()]
    adaptive = [s for s in completed if 'adaptive' in s['name'].lower()]
    
    if crumbtrail:
        avg = sum(s['entity_reduction_pct'] for s in crumbtrail) / len(crumbtrail)
        print(f"   • CrumbTrail: {len(crumbtrail)} variants, avg {avg:.1f}% reduction")
    if kgtrimmer:
        avg = sum(s['entity_reduction_pct'] for s in kgtrimmer) / len(kgtrimmer)
        print(f"   • KGTrimmer: {len(kgtrimmer)} variants, avg {avg:.1f}% reduction")
    if pog:
        avg = sum(s['entity_reduction_pct'] for s in pog) / len(pog)
        print(f"   • POG: {len(pog)} variants, avg {avg:.1f}% reduction")
    if pathrag:
        avg = sum(s['entity_reduction_pct'] for s in pathrag) / len(pathrag)
        print(f"   • PathRAG: {len(pathrag)} variants, avg {avg:.1f}% reduction")
    if adaptive:
        avg = sum(s['entity_reduction_pct'] for s in adaptive) / len(adaptive)
        print(f"   • Adaptive Hybrid: {len(adaptive)} variants, avg {avg:.1f}% reduction")
    
    print("="*120)


def main():
    """Main function."""
    print("\n" + "="*120)
    print("🔍 PRUNING METHODS RESULTS")
    print("="*120)
    
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
    for hybrid_dir in Path("workspace/output").glob("pruned_adaptive_hybrid*"):
        metadata = load_pruning_metadata(hybrid_dir)
        if metadata:
            method_name = hybrid_dir.name.replace("pruned_", "")
            stats = extract_stats(metadata, method_name)
            stats_list.append(stats)
    
    # Print results
    print_results_table(stats_list)
    print_summary(stats_list)
    
    # Export to CSV
    output_file = Path("eval/results/pruning_results_summary.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(stats_list)
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Results exported to: {output_file}\n")


if __name__ == "__main__":
    main()

