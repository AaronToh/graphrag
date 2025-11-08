#!/usr/bin/env python3
"""
Convert JSONL files to text format for GraphRAG indexing.
Each JSON record becomes a separate text file named by its doc_id.
"""

import json
import logging
from pathlib import Path
from typing import Optional

def setup_logging() -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def convert_jsonl_to_text(
    jsonl_path: str,
    output_dir: str,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Convert JSONL file to individual text files.
    
    Args:
        jsonl_path: Path to input JSONL file
        output_dir: Directory to write text files
        logger: Optional logger instance
    
    Returns:
        bool: True if conversion successful
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        jsonl_path = Path(jsonl_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting {jsonl_path} to text files in {output_dir}")
        
        # Track statistics
        processed = 0
        errors = 0
        
        with open(jsonl_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    
                    # Extract the passage ID (e.g., "MH_train_0::support_0")
                    passage_id = record['id']
                    text = record['text']
                    
                    # Create a safe filename from the passage ID
                    safe_filename = passage_id.replace("::", "_")
                    
                    # Write text file with only the text content
                    text_file = output_dir / f"{safe_filename}.txt"
                    with open(text_file, "w") as tf:
                        tf.write(text)
                    
                    processed += 1
                    if processed % 10000 == 0:
                        logger.info(f"Processed {processed} records...")
                        
                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing line {line_num}: {str(e)}")
                    continue
        
        logger.info(f"Conversion complete. Processed {processed} records with {errors} errors")
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert JSONL: {str(e)}")
        return False

def main():
    """Main entry point."""
    logger = setup_logging()
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Input/output paths
    jsonl_path = project_root / "data" / "input" / "passages.jsonl"
    output_dir = project_root / "data" / "text_input"
    
    # Run conversion
    success = convert_jsonl_to_text(jsonl_path, output_dir, logger)
    if not success:
        logger.error("Conversion failed")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
