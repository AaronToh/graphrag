"""
Logging configuration for GraphRAG ingestion module.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_detailed_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Set up clean logging configuration for ingestion processes.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger('graphrag_ingest')
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler (cleaner format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (detailed format)
    file_handler = logging.FileHandler(log_dir / 'indexing.log')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger
