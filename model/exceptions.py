"""
Custom exceptions for Ollama model management.
"""

class OllamaError(Exception):
    """Base exception for Ollama-related errors."""
    pass

class TimeoutError(OllamaError):
    """Raised when an operation times out."""
    pass

class ModelNotFoundError(OllamaError):
    """Raised when a required model is not found."""
    pass

class ServerStartError(OllamaError):
    """Raised when Ollama server fails to start."""
    pass
