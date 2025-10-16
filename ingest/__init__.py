"""
GraphRAG ingestion module for OpenAI API-based indexing.

This module provides tools for:
- Converting JSONL data to text format for GraphRAG
- Building knowledge graph indexes using OpenAI API
- Logging and monitoring the ingestion process
"""

from pathlib import Path

# Module level constants
INGEST_DIR = Path(__file__).parent
PROJECT_ROOT = INGEST_DIR.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
