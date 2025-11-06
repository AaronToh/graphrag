#!/usr/bin/env python3
"""
Show Evaluation Results Summary

Displays evaluation results for all pruning methods with metrics comparison.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys


def load_evaluation_results() -> Dict:
    """Load evaluation results from method_evaluations.json."""
    results_file = Path("eval/results/method_evaluations.json")
    
    if not results_file.exists():
        print(f"❌ Error: {results_file} not found")
        print("   Run eval_all_pruning_methods.py first")
        sys.exit(1)
    
    with open(results_file, 'r') as f:
        return json.load(f)


def format_number(value, decimal_places=4, suffix=''):
    """Format number for display."""
    if value is None or value == 'N/A':
        return 'N/A'
    try:
        if isinstance(value, str):
            return value
        return f"{value:.{decimal_places}f}{suffix}"
    except (ValueError, TypeError):
        return str(value)


def print_evaluation_table(results: Dict):
    """Print formatted evaluation results table."""
    print("\n" + "="*120)
    print("EVALUATION RESULTS SUMMARY (5 Samples)")
    print("="*120)
    
    headers = ["Method", "Faithfulness", "SAS", "MRR", "Response Time", "Faith Δ%", "Time Δ%"]
    col_widths = [30, 14, 10, 10, 15, 12, 12]
    
    # Print header
    header_row = ""
    for header, width in zip(headers, col_widths):
        header_row += f"{header:<{width}}"
    print(header_row)
    print("-"*120)
    
    # Get baseline metrics
    baseline_metrics = None
    for method_name, metrics in results.items():
        if 'baseline' in metrics:
            baseline_metrics = metrics['baseline']
            break
    
    if not baseline_metrics:
        print("⚠️  No baseline metrics found")
        return
    
    # Print baseline
    row = ""
    row += f"{'Baseline':<30}"
    row += f"{format_number(baseline_metrics.get('faithfulness_score'), 4):<14}"
    row += f"{format_number(baseline_metrics.get('sas_score'), 4):<10}"
    row += f"{format_number(baseline_metrics.get('mrr_score'), 4):<10}"
    row += f"{format_number(baseline_metrics.get('avg_response_time'), 4, 's'):<15}"
    row += f"{'0.00%':<12}"
    row += f"{'0.00%':<12}"
    print(row)
    print("-"*120)
    
    # Print pruned methods
    for method_name, metrics in sorted(results.items()):
        if 'pruned' not in metrics:
            continue
        
        pruned = metrics['pruned']
        comparison = metrics.get('comparison', {})
        
        row = ""
        row += f"{method_name:<30}"
        row += f"{format_number(pruned.get('faithfulness_score'), 4):<14}"
        row += f"{format_number(pruned.get('sas_score'), 4):<10}"
        row += f"{format_number(pruned.get('mrr_score'), 4):<10}"
        row += f"{format_number(pruned.get('avg_response_time'), 4, 's'):<15}"
        
        faith_change = comparison.get('faithfulness_change_pct', 0)
        time_change = comparison.get('response_time_change_pct', 0)
        
        # Color code changes
        faith_str = f"{faith_change:+.2f}%"
        time_str = f"{time_change:+.2f}%"
        
        row += f"{faith_str:<12}"
        row += f"{time_str:<12}"
        print(row)
    
    print("="*120)


def print_summary(results: Dict):
    """Print summary statistics."""
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    
    # Get baseline
    baseline_metrics = None
    for method_name, metrics in results.items():
        if 'baseline' in metrics:
            baseline_metrics = metrics['baseline']
            break
    
    if not baseline_metrics:
        return
    
    baseline_faith = baseline_metrics.get('faithfulness_score', 0)
    baseline_sas = baseline_metrics.get('sas_score', 0)
    baseline_mrr = baseline_metrics.get('mrr_score', 0)
    
    print(f"\n📊 Baseline Performance:")
    print(f"   Faithfulness: {baseline_faith:.4f}")
    print(f"   SAS: {baseline_sas:.4f}")
    print(f"   MRR: {baseline_mrr:.4f}")
    
    # Find best methods
    best_faith = None
    best_sas = None
    best_mrr = None
    best_faith_score = baseline_faith
    best_sas_score = baseline_sas
    best_mrr_score = baseline_mrr
    
    for method_name, metrics in results.items():
        if 'pruned' not in metrics:
            continue
        
        pruned = metrics['pruned']
        faith = pruned.get('faithfulness_score', 0)
        sas = pruned.get('sas_score', 0)
        mrr = pruned.get('mrr_score', 0)
        
        if faith >= best_faith_score:
            best_faith = method_name
            best_faith_score = faith
        
        if sas >= best_sas_score:
            best_sas = method_name
            best_sas_score = sas
        
        if mrr >= best_mrr_score:
            best_mrr = method_name
            best_mrr_score = mrr
    
    print(f"\n🏆 Best Methods:")
    if best_faith:
        print(f"   Faithfulness: {best_faith} ({best_faith_score:.4f})")
    if best_sas:
        print(f"   SAS: {best_sas} ({best_sas_score:.4f})")
    if best_mrr:
        print(f"   MRR: {best_mrr} ({best_mrr_score:.4f})")
    
    # Methods that maintain or improve all metrics
    print(f"\n✅ Methods Maintaining/Improving All Metrics:")
    for method_name, metrics in sorted(results.items()):
        if 'pruned' not in metrics:
            continue
        
        pruned = metrics['pruned']
        comparison = metrics.get('comparison', {})
        
        faith = pruned.get('faithfulness_score', 0)
        sas = pruned.get('sas_score', 0)
        mrr = pruned.get('mrr_score', 0)
        faith_change = comparison.get('faithfulness_change_pct', 0)
        
        if (faith >= baseline_faith * 0.95 and 
            sas >= baseline_sas * 0.95 and 
            mrr >= baseline_mrr * 0.95):
            print(f"   • {method_name}: Faith={faith:.4f}, SAS={sas:.4f}, MRR={mrr:.4f}")
    
    print("="*120)


def main():
    """Main function."""
    print("\n" + "="*120)
    print("🔍 EVALUATION RESULTS")
    print("="*120)
    
    results = load_evaluation_results()
    
    if not results:
        print("❌ No evaluation results found")
        sys.exit(1)
    
    print_evaluation_table(results)
    print_summary(results)
    
    print(f"\n💾 Full results saved to: eval/results/method_evaluations.json\n")


if __name__ == "__main__":
    main()

