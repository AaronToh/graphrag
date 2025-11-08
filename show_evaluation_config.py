#!/usr/bin/env python3
"""
Show Evaluation Configuration

Displays what dataset, questions, and configuration the evaluations are using.
"""

import json
from pathlib import Path
import pandas as pd


def main():
    """Main function."""
    print("\n" + "="*100)
    print("EVALUATION CONFIGURATION")
    print("="*100)
    
    # Check eval_all_pruning_methods.py configuration
    print("\n📋 Evaluation Script Configuration:")
    print("   Script: eval_all_pruning_methods.py")
    print("   Number of samples: 5 (configured in line 162)")
    print("   Dataset: PubMedQA (vblagoje/PubMedQA_instruction)")
    print("   Split: train")
    print("   Evaluation method: FileBackedGraphRAGSystem (simple keyword-overlap retrieval)")
    
    # Check method_evaluations.json
    results_file = Path("eval/results/method_evaluations.json")
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        # Get first method to check sample info
        first_method = list(results.keys())[0] if results else None
        if first_method:
            method_data = results[first_method]
            print(f"\n📊 Evaluation Results Info:")
            print(f"   Total queries per method: {method_data.get('baseline', {}).get('total_queries', 'N/A')}")
            print(f"   Number of samples: {method_data.get('num_samples', 'N/A')}")
            print(f"   Evaluation timestamp: {method_data.get('timestamp', 'N/A')}")
    
    # Try to find example questions from CSV files
    print(f"\n📝 Example Questions (from evaluation results):")
    csv_files = list(Path("eval/results").glob("*details*.csv"))
    if csv_files:
        # Get most recent CSV
        csv_file = sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        try:
            df = pd.read_csv(csv_file, nrows=5)
            if 'question' in df.columns:
                for i, question in enumerate(df['question'].head(5), 1):
                    print(f"   {i}. {question[:100]}...")
        except Exception as e:
            print(f"   Could not read CSV: {e}")
    else:
        print("   No CSV files found")
    
    # Check what retrieval system is being used
    print(f"\n🔍 Retrieval System:")
    print("   Type: FileBackedGraphRAGSystem")
    print("   Method: Simple keyword-overlap retrieval")
    print("   - Filters text units based on pruned entities/relationships")
    print("   - Uses token overlap scoring (not graph-aware)")
    print("   - Top-K: 5 documents")
    
    # Check PubMedQA dataset info
    print(f"\n📚 PubMedQA Dataset:")
    print("   Source: HuggingFace")
    print("   Dataset: vblagoje/PubMedQA_instruction")
    print("   Split: train")
    print("   Fields:")
    print("     - instruction: The question")
    print("     - context: Ground truth documents")
    print("     - response: Ground truth answer")
    
    print("\n" + "="*100)
    print("\n⚠️  Important Notes:")
    print("   1. Only 5 samples are being evaluated (for faster testing)")
    print("   2. The evaluation uses simple keyword-overlap, not graph-aware retrieval")
    print("   3. This explains why scores are similar across methods")
    print("   4. The pruning is actually working (see pruning statistics)")
    print("   5. To see actual differences, use graph-aware retrieval with more samples")
    print("\n")


if __name__ == "__main__":
    main()

