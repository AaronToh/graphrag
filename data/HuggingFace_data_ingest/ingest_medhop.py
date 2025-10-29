#!/usr/bin/env python3
"""
MedHop QA Dataset Ingestion Script

Converts the HuggingFace MedHop dataset into JSONL format
compatible with Microsoft GraphRAG ingestion pipeline.

Usage:
    python ingest_medhop.py [--output-dir OUTPUT_DIR] [--limit LIMIT | --percentage PERCENT] [--verbose]

Examples:
    # Process 10% of dataset for testing
    python ingest_medhop.py --percentage 0.1
    
    # Process first 100 records
    python ingest_medhop.py --limit 100
    
    # Process full dataset with verbose output
    python ingest_medhop.py --verbose

Output:
    - passages.jsonl: All passage-level records ready for GraphRAG ingestion
    - dataset_stats.json: Statistics about the processed dataset
"""

import json
import logging
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """
    Find the project root directory by looking for pixi.toml or other project markers.
    
    Returns:
        Path to the project root directory
    """
    current_path = Path(__file__).resolve()
    
    # Look for project markers (pixi.toml, .git, etc.)
    project_markers = ['pixi.toml', 'pyproject.toml', '.git', 'README.md']
    
    # Start from current file directory and go up
    for parent in [current_path.parent] + list(current_path.parents):
        for marker in project_markers:
            if (parent / marker).exists():
                # Additional check: ensure this looks like our GraphRAG project
                if (parent / 'data').exists() and (parent / 'ingest').exists():
                    logger.debug(f"Found project root: {parent}")
                    return parent
    
    # Fallback: assume we're in the project somewhere and go up until we find data/
    for parent in [current_path.parent] + list(current_path.parents):
        if (parent / 'data').exists():
            logger.debug(f"Found project root (fallback): {parent}")
            return parent
    
    # Last resort: use current working directory
    cwd = Path.cwd()
    logger.warning(f"Could not find project root, using current directory: {cwd}")
    return cwd


def resolve_output_path(output_dir: str) -> Path:
    """
    Resolve output directory path relative to project root.
    
    Args:
        output_dir: Output directory path (can be relative or absolute)
        
    Returns:
        Resolved absolute path
    """
    output_path = Path(output_dir)
    
    if output_path.is_absolute():
        return output_path
    
    # If relative, resolve relative to project root
    project_root = find_project_root()
    resolved_path = project_root / output_path
    
    logger.debug(f"Resolved output path: {output_dir} -> {resolved_path}")
    return resolved_path


class MedHopIngester:
    """Handles ingestion and conversion of MedHop dataset to GraphRAG format."""
    
    def __init__(self, output_dir: str = "data/input"):
        """
        Initialize the ingester.
        
        Args:
            output_dir: Directory to save the processed JSONL files (relative to project root or absolute)
        """
        self.output_dir = resolve_output_path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        
        # Statistics tracking
        self.stats = {
            "total_records": 0,
            "processed_records": 0,
            "skipped_records": 0,
            "total_passages": 0,
            "avg_passage_length": 0,
            "avg_supports_per_record": 0,
            "dataset_config": "medhop_source"
        }
    
    def load_dataset(self, config: str = "medhop_source", split: str = "train") -> Any:
        """
        Load MedHop dataset from HuggingFace.
        
        Args:
            config: Dataset configuration (medhop_source, medhop_bigbio_qa)
            split: Dataset split to load (train, validation)
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading MedHop dataset: config={config}, split={split}")
        
        try:
            dataset = load_dataset("bigbio/medhop", config)
            
            # Safe access to dataset keys
            available_splits = []
            try:
                if hasattr(dataset, 'keys'):
                    available_splits = list(dataset.keys())
                logger.info(f"Available splits: {available_splits}")
            except (AttributeError, TypeError):
                logger.warning("Could not determine available splits")
            
            # Check if split exists
            split_to_use = split
            try:
                if split not in dataset and available_splits:
                    logger.warning(f"Split '{split}' not found. Using first available split.")
                    split_to_use = str(available_splits[0])
                elif not available_splits:
                    split_to_use = "train"  # fallback
            except (TypeError, AttributeError):
                split_to_use = "train"
            
            records = dataset[split_to_use]
            
            # Handle different dataset types safely
            try:
                if hasattr(records, '__len__'):
                    record_count = len(records)
                    logger.info(f"Loaded {record_count} records from {config}/{split_to_use}")
                else:
                    logger.info(f"Loaded dataset from {config}/{split_to_use} (streaming/iterable)")
            except (TypeError, AttributeError):
                logger.info(f"Loaded dataset from {config}/{split_to_use} (unknown size)")
            
            # Log sample record structure safely
            try:
                if hasattr(records, '__getitem__'):
                    sample = records[0]
                else:
                    sample = next(iter(records))
                
                if hasattr(sample, 'keys') and callable(getattr(sample, 'keys')):
                    logger.info(f"Sample record fields: {list(sample.keys())}")
                elif isinstance(sample, dict):
                    logger.info(f"Sample record fields: {list(sample.keys())}")
            except (IndexError, StopIteration, TypeError, AttributeError):
                logger.warning("Could not access sample record for field inspection")
                
            return records
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def create_passage_records(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create GraphRAG-compatible passage records from a MedHop record.
        
        Args:
            record: Original MedHop record
            
        Returns:
            List of formatted passage records
        """
        record_id = str(record.get('id', 'unknown'))
        query = record.get('query', '')
        answer = record.get('answer', '')
        candidates = record.get('candidates', [])
        supports = record.get('supports', [])
        
        if not supports:
            logger.warning(f"No supports found for record: {record_id}")
            return []
        
        passage_records = []
        
        # Create a passage record for each support
        for idx, support_text in enumerate(supports):
            if not support_text or not support_text.strip():
                continue
                
            # Create the passage record following GraphRAG schema
            passage_record = {
                "id": f"{record_id}::support_{idx}",
                "doc_id": record_id,
                "text": support_text.strip(),
                "attrs": {
                    "section": "support_passage",
                    "source": "medhop",
                    "dataset_split": "train",
                    "dataset_config": self.stats["dataset_config"],
                    "support_index": idx,
                    "query": query,
                    "answer": answer,
                    "candidates": candidates,
                    "total_supports": len(supports)
                }
            }
            
            passage_records.append(passage_record)
        
        return passage_records
    
    def process_dataset(self, records: Any, limit: Optional[int] = None, percentage: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Process the entire dataset and convert to passage records.
        
        Args:
            records: Dataset records to process
            limit: Optional limit on number of records to process
            percentage: Optional percentage of dataset to process (0.1 = 10%, 1.0 = 100%)
            
        Returns:
            List of passage records
        """
        passages = []
        total_chars = 0
        total_supports = 0
        
        # Apply percentage or limit if specified
        if percentage is not None:
            # Validate percentage
            if not (0.0 < percentage <= 1.0):
                raise ValueError(f"Percentage must be between 0.0 and 1.0, got {percentage}")
            
            try:
                if hasattr(records, 'select') and hasattr(records, '__len__'):
                    # Standard dataset with select method
                    total_records = len(records)
                    sample_size = int(total_records * percentage)
                    records = records.select(range(sample_size))
                    logger.info(f"Processing {percentage*100:.1f}% of dataset: {sample_size}/{total_records} records")
                elif hasattr(records, '__iter__'):
                    # For iterable datasets, we need to estimate or count first
                    logger.warning("Percentage sampling on streaming dataset - converting to list first")
                    records_list = list(records)
                    total_records = len(records_list)
                    sample_size = int(total_records * percentage)
                    records = records_list[:sample_size]
                    logger.info(f"Processing {percentage*100:.1f}% of dataset: {sample_size}/{total_records} records")
                else:
                    logger.warning("Could not apply percentage to dataset, processing all records")
            except (TypeError, AttributeError) as e:
                logger.warning(f"Could not apply percentage to dataset: {e}, processing all records")
        elif limit:
            try:
                if hasattr(records, 'select') and hasattr(records, '__len__'):
                    # Standard dataset with select method
                    records = records.select(range(min(limit, len(records))))
                    logger.info(f"Processing limited dataset: {limit} records")
                elif hasattr(records, '__iter__'):
                    # Iterable dataset, convert to list with limit
                    records_list = []
                    for i, record in enumerate(records):
                        if i >= limit:
                            break
                        records_list.append(record)
                    records = records_list
                    logger.info(f"Processing limited dataset: {len(records_list)} records")
                else:
                    logger.warning("Could not apply limit to dataset, processing all records")
            except (TypeError, AttributeError) as e:
                logger.warning(f"Could not apply limit to dataset: {e}, processing all records")
        
        # Get total records count safely
        try:
            if hasattr(records, '__len__'):
                self.stats["total_records"] = len(records)
            else:
                # For streaming datasets, we'll count as we go
                self.stats["total_records"] = 0
        except (TypeError, AttributeError):
            # For streaming datasets, we'll count as we go
            self.stats["total_records"] = 0
        
        logger.info("Processing records...")
        for idx, record in enumerate(tqdm(records, desc="Converting records")):
            try:
                # Update total count for streaming datasets
                if self.stats["total_records"] == 0:
                    self.stats["total_records"] = idx + 1
                
                record_passages = self.create_passage_records(record)
                
                if record_passages:
                    passages.extend(record_passages)
                    # Track statistics
                    for passage in record_passages:
                        total_chars += len(passage["text"])
                    total_supports += len(record_passages)
                    self.stats["processed_records"] += 1
                else:
                    self.stats["skipped_records"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing record {idx}: {e}")
                self.stats["skipped_records"] += 1
                continue
        
        # Update statistics
        self.stats["total_passages"] = len(passages)
        if passages:
            self.stats["avg_passage_length"] = total_chars / len(passages)
        if self.stats["processed_records"] > 0:
            self.stats["avg_supports_per_record"] = total_supports / self.stats["processed_records"]
        
        logger.info(f"Processed {len(passages)} passages from {self.stats['processed_records']} records")
        logger.info(f"Average {self.stats['avg_supports_per_record']:.1f} supports per record")
        return passages
    
    def save_passages(self, passages: List[Dict[str, Any]], filename: str = "passages.jsonl") -> Path:
        """
        Save passages to JSONL file.
        
        Args:
            passages: List of passage records
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving {len(passages)} passages to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for passage in passages:
                f.write(json.dumps(passage, ensure_ascii=False) + '\n')
        
        logger.info(f"Successfully saved passages to {output_path}")
        return output_path
    
    def save_statistics(self, filename: str = "dataset_stats.json") -> Path:
        """
        Save processing statistics to JSON file.
        
        Args:
            filename: Output filename for statistics
            
        Returns:
            Path to saved statistics file
        """
        stats_path = self.output_dir / filename
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Statistics saved to {stats_path}")
        return stats_path
    
    def run_ingestion(self, config: str = "medhop_source", limit: Optional[int] = None, percentage: Optional[float] = None) -> Dict[str, Path]:
        """
        Run the complete ingestion pipeline.
        
        Args:
            config: Dataset configuration to use
            limit: Optional limit on records to process
            percentage: Optional percentage of dataset to process (0.1 = 10%, 1.0 = 100%)
            
        Returns:
            Dictionary with paths to output files
        """
        logger.info("Starting MedHop ingestion pipeline")
        
        # Update config in stats
        self.stats["dataset_config"] = config
        
        # Load dataset
        records = self.load_dataset(config=config)
        
        # Process records
        passages = self.process_dataset(records, limit=limit, percentage=percentage)
        
        if not passages:
            raise ValueError("No passages were successfully processed")
        
        # Save outputs
        passages_path = self.save_passages(passages)
        stats_path = self.save_statistics()
        
        # Log summary
        logger.info("Ingestion completed successfully!")
        logger.info(f"Total records: {self.stats['total_records']}")
        logger.info(f"Processed: {self.stats['processed_records']}")
        logger.info(f"Skipped: {self.stats['skipped_records']}")
        logger.info(f"Total passages: {self.stats['total_passages']}")
        logger.info(f"Average passage length: {self.stats['avg_passage_length']:.1f} characters")
        logger.info(f"Average supports per record: {self.stats['avg_supports_per_record']:.1f}")
        
        return {
            "passages": passages_path,
            "statistics": stats_path
        }


def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(description="Ingest MedHop dataset for GraphRAG")
    parser.add_argument(
        "--output-dir", 
        default="data/input",
        help="Output directory for processed files (default: data/input, relative to project root)"
    )
    parser.add_argument(
        "--config",
        default="medhop_source",
        choices=["medhop_source", "medhop_bigbio_qa"],
        help="MedHop dataset configuration (default: medhop_source - recommended for GraphRAG)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of records to process (for testing)"
    )
    parser.add_argument(
        "--percentage",
        type=float,
        help="Percentage of dataset to process (0.1 = 10%%, 1.0 = 100%%)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.limit is not None and args.percentage is not None:
        parser.error("Cannot specify both --limit and --percentage. Use one or the other.")
    
    if args.percentage is not None and not (0.0 < args.percentage <= 1.0):
        parser.error("Percentage must be between 0.0 and 1.0 (e.g., 0.1 for 10%)")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize ingester
        ingester = MedHopIngester(output_dir=args.output_dir)
        
        # Run ingestion
        output_files = ingester.run_ingestion(
            config=args.config,
            limit=args.limit,
            percentage=args.percentage
        )
        
        print("\n" + "="*60)
        print("MEDHOP INGESTION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Passages file: {output_files['passages']}")
        print(f"Statistics file: {output_files['statistics']}")
        print("\nFiles are ready for GraphRAG ingestion!")
        print("\nNext steps:")
        print("1. Run: python -m ingest.convert_jsonl")
        print("2. Run: python -m ingest.build_index --verbose")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
