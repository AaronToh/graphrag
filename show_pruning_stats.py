#!/usr/bin/env python3
"""
Show Pruning Statistics

Displays entity and relationship counts for all pruning methods.
"""

import json
from pathlib import Path
import sys


def load_pruning_stats(method_name: str) -> dict:
    """Load pruning statistics for a method."""
    meta_file = Path(f"workspace/output/pruned_{method_name}/pruning_metadata.json")
    
    if not meta_file.exists():
        return None
    
    with open(meta_file) as f:
        meta = json.load(f)
    
    baseline = meta.get('baseline_stats', {})
    pruned = meta.get('pruned_stats', {})
    
    return {
        'baseline_entities': baseline.get('num_entities', 0),
        'baseline_relationships': baseline.get('num_relationships', 0),
        'pruned_entities': pruned.get('num_entities', 0),
        'pruned_relationships': pruned.get('num_relationships', 0),
        'entity_reduction_pct': 100 * (1 - pruned.get('num_entities', 0) / max(1, baseline.get('num_entities', 1))),
        'relationship_reduction_pct': 100 * (1 - pruned.get('num_relationships', 0) / max(1, baseline.get('num_relationships', 1)))
    }


def main():
    """Main function."""
    methods = [
        'crumbtrail_conservative',
        'kgtrimmer_default',
        'pathrag_hybrid',
        'pog_hybrid',
        'adaptive_multi_strategy'
    ]
    
    print("\n" + "="*100)
    print("PRUNING STATISTICS SUMMARY")
    print("="*100)
    print(f"\n{'Method':<35} {'Entities':<30} {'Relationships':<30} {'Entity Red%':<15} {'Rel Red%':<15}")
    print("-"*100)
    
    for method in methods:
        stats = load_pruning_stats(method)
        if stats:
            print(f"{method:<35} "
                  f"{stats['baseline_entities']:>6} -> {stats['pruned_entities']:>6} "
                  f"{stats['baseline_relationships']:>6} -> {stats['pruned_relationships']:>6} "
                  f"{stats['entity_reduction_pct']:>6.1f}%        "
                  f"{stats['relationship_reduction_pct']:>6.1f}%")
        else:
            print(f"{method:<35} {'METADATA NOT FOUND':<30}")
    
    print("="*100)
    print("\nNote: These statistics show the actual graph reduction achieved by each method.")
    print("The evaluation scores are similar because the evaluation system uses simple")
    print("keyword-overlap retrieval rather than graph-aware retrieval.\n")


if __name__ == "__main__":
    main()

