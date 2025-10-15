#!/usr/bin/env python3
"""
PubMedQA Dataset Ingestion Script

Converts the HuggingFace PubMedQA dataset (pqa_artificial) into JSONL format
compatible with Microsoft GraphRAG ingestion pipeline.

Usage:
    python ingest_pubmedqa.py [--output-dir OUTPUT_DIR] [--limit LIMIT] [--verbose]

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


class PubMedQAIngester:
    """Handles ingestion and conversion of PubMedQA dataset to GraphRAG format."""
    
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
            "dataset_config": "pqa_artificial"
        }
    
    def load_dataset(self, config: str = "pqa_artificial", split: str = "train") -> Any:
        """
        Load PubMedQA dataset from HuggingFace.
        
        Args:
            config: Dataset configuration (pqa_artificial, pqa_labeled, pqa_unlabeled)
            split: Dataset split to load (train, test, validation)
            
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading PubMedQA dataset: config={config}, split={split}")
        
        try:
            dataset = load_dataset("pubmed_qa", config)
            
            # Safe access to dataset keys
            available_splits = []
            try:
                if hasattr(dataset, 'keys'):
                    available_splits = list(dataset.keys())  # type: ignore
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
                    record_count = len(records)  # type: ignore
                    logger.info(f"Loaded {record_count} records from {config}/{split_to_use}")
                else:
                    logger.info(f"Loaded dataset from {config}/{split_to_use} (streaming/iterable)")
            except (TypeError, AttributeError):
                logger.info(f"Loaded dataset from {config}/{split_to_use} (unknown size)")
            
            # Log sample record structure safely
            try:
                if hasattr(records, '__getitem__'):
                    sample = records[0]  # type: ignore
                else:
                    sample = next(iter(records))
                
                if hasattr(sample, 'keys') and callable(getattr(sample, 'keys')):
                    logger.info(f"Sample record fields: {list(sample.keys())}")  # type: ignore
                elif isinstance(sample, dict):
                    logger.info(f"Sample record fields: {list(sample.keys())}")
            except (IndexError, StopIteration, TypeError, AttributeError):
                logger.warning("Could not access sample record for field inspection")
                
            return records
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def extract_text_content(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Extract the main text content from a PubMedQA record.
        
        Args:
            record: Single record from the dataset
            
        Returns:
            Extracted text content or None if not available
        """
        # Handle PubMedQA specific structure where context is a dict with 'contexts' key
        if 'context' in record and record['context']:
            context = record['context']
            
            if isinstance(context, dict):
                # Extract contexts list from the context dict
                if 'contexts' in context and context['contexts']:
                    contexts_list = context['contexts']
                    if isinstance(contexts_list, list):
                        # Join all context passages with double newlines
                        text_content = "\n\n".join(str(item) for item in contexts_list if item)
                        if text_content.strip():
                            return text_content.strip()
            elif isinstance(context, list):
                # Direct list of contexts
                text_content = "\n\n".join(str(item) for item in context if item)
                if text_content.strip():
                    return text_content.strip()
            elif isinstance(context, str):
                # Direct string content
                if context.strip():
                    return context.strip()
        
        # Fallback: try other possible field names
        fallback_fields = ['context_text', 'CONTEXTS', 'text']
        for field in fallback_fields:
            if field in record and record[field]:
                content = record[field]
                if isinstance(content, str) and content.strip():
                    return content.strip()
                elif isinstance(content, list):
                    text_content = "\n\n".join(str(item) for item in content if item)
                    if text_content.strip():
                        return text_content.strip()
        
        logger.warning(f"No text content found for record: {record.get('pubid', 'unknown')}")
        return None
    
    def create_passage_record(self, record: Dict[str, Any], passage_idx: int = 0) -> Optional[Dict[str, Any]]:
        """
        Create a GraphRAG-compatible passage record.
        
        Args:
            record: Original PubMedQA record
            passage_idx: Index for this passage (for chunking support)
            
        Returns:
            Formatted passage record or None if text content is not available
        """
        pubid = str(record.get('pubid', f'unknown_{passage_idx}'))
        text_content = self.extract_text_content(record)
        
        if not text_content:
            return None
        
        # Create the passage record following the specified schema
        passage_record = {
            "id": f"{pubid}::p{passage_idx}",
            "doc_id": pubid,
            "text": text_content,
            "attrs": {
                "section": "abstract",
                "source": "pubmed_qa",
                "dataset_split": "train",
                "dataset_config": self.stats["dataset_config"]
            }
        }
        
        # Add optional metadata if available (for evaluation later)
        optional_fields = ['question', 'final_decision', 'long_answer']
        for field in optional_fields:
            if field in record and record[field]:
                passage_record["attrs"][field] = record[field]
        
        return passage_record
    
    def process_dataset(self, records: Any, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process the entire dataset and convert to passage records.
        
        Args:
            records: Dataset records to process
            limit: Optional limit on number of records to process
            
        Returns:
            List of passage records
        """
        passages = []
        total_chars = 0
        
        # Apply limit if specified
        if limit:
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
                
                passage = self.create_passage_record(record, passage_idx=0)
                
                if passage:
                    passages.append(passage)
                    total_chars += len(passage["text"])
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
        
        logger.info(f"Processed {len(passages)} passages from {len(records)} records")
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
    
    def run_ingestion(self, config: str = "pqa_artificial", limit: Optional[int] = None) -> Dict[str, Path]:
        """
        Run the complete ingestion pipeline.
        
        Args:
            config: Dataset configuration to use
            limit: Optional limit on records to process
            
        Returns:
            Dictionary with paths to output files
        """
        logger.info("Starting PubMedQA ingestion pipeline")
        
        # Update config in stats
        self.stats["dataset_config"] = config
        
        # Load dataset
        records = self.load_dataset(config=config)
        
        # Process records
        passages = self.process_dataset(records, limit=limit)
        
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
        
        return {
            "passages": passages_path,
            "statistics": stats_path
        }


def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(description="Ingest PubMedQA dataset for GraphRAG")
    parser.add_argument(
        "--output-dir", 
        default="data/input",
        help="Output directory for processed files (default: data/input, relative to project root)"
    )
    parser.add_argument(
        "--config",
        default="pqa_artificial",
        choices=["pqa_artificial", "pqa_labeled", "pqa_unlabeled"],
        help="PubMedQA dataset configuration (default: pqa_artificial)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of records to process (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize ingester
        ingester = PubMedQAIngester(output_dir=args.output_dir)
        
        # Run ingestion
        output_files = ingester.run_ingestion(
            config=args.config,
            limit=args.limit
        )
        
        print("\n" + "="*60)
        print("INGESTION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Passages file: {output_files['passages']}")
        print(f"Statistics file: {output_files['statistics']}")
        print("\nFiles are ready for GraphRAG ingestion!")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
