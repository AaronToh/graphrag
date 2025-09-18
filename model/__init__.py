"""
GraphRAG Model Management Module

This module provides comprehensive Ollama model management including:
- Auto-starting Ollama server
- Auto-pulling required models (mistral, nomic-embed-text)
- Comprehensive logging
- Timeout protection
- Health checks
"""

from .ollama_manager import OllamaManager, ModelStatus
from .exceptions import OllamaError, TimeoutError, ModelNotFoundError

__all__ = ['OllamaManager', 'ModelStatus', 'OllamaError', 'TimeoutError', 'ModelNotFoundError']
