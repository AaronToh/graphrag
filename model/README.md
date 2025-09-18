# GraphRAG Ollama Model Manager

This directory contains the comprehensive Ollama model management system for GraphRAG.

## Features

- 🚀 **Auto-start Ollama server** when GraphRAG runs
- 📥 **Auto-pull required models** (mistral, nomic-embed-text)
- 📝 **Comprehensive logging** of all operations
- ⏰ **Timeout protection** to prevent hanging
- 🔒 **Black-boxed implementation** - everything contained within this directory
- 📁 **Centralized logging** - all Ollama logs stored in `model/` directory

## Files

- `__init__.py` - Package initialization and exports
- `ollama_manager.py` - Main OllamaManager class implementation
- `exceptions.py` - Custom exception classes
- `ollama.log` - Comprehensive log file for all operations
- `README.md` - This documentation

## Usage

The Ollama manager is automatically integrated into `build_index.py`. When you run:

```bash
python ingest/build_index.py
```

The system will:

1. Check if Ollama binary is available
2. Auto-start the Ollama server (if not already running)
3. Check if required models are installed
4. Auto-pull any missing models (mistral, nomic-embed-text)
5. Log all operations to `model/ollama.log`
6. Proceed with GraphRAG indexing

## Required Models

The system automatically manages these models:
- **mistral** - Used for text generation and entity extraction
- **nomic-embed-text** - Used for text embeddings

## Configuration

Models are configured in `workspace/settings.yaml`:
- Chat model: `mistral` via Ollama API (`http://localhost:11434/v1`)
- Embedding model: `nomic-embed-text` via Ollama API

## Logging

All operations are logged to:
- `model/ollama.log` - Detailed operation logs
- Console output - Summary information
- `ingest/indexing.log` - Integration with main build process

## Error Handling

The system includes comprehensive error handling:
- Timeout protection for all operations
- Graceful fallback if Ollama is not available
- Detailed error messages and recovery suggestions
- Automatic cleanup on failures

## Manual Testing

You can test the Ollama manager independently:

```python
from model import OllamaManager

with OllamaManager() as manager:
    if manager.initialize():
        print("✅ Ollama ready!")
        print(manager.get_model_status_report())
    else:
        print("❌ Initialization failed")
```

## Dependencies

- `requests` - For Ollama API communication
- Ollama binary must be installed on the system

## Troubleshooting

1. **Ollama not found**: Install Ollama from https://ollama.ai
2. **Models not pulling**: Check internet connection and disk space
3. **Server won't start**: Check if port 11434 is available
4. **Timeout errors**: Increase timeout values in `ollama_manager.py` if needed

## Architecture

The OllamaManager uses:
- Context manager pattern for automatic cleanup
- ThreadPoolExecutor for timeout protection
- Comprehensive logging with both file and console handlers
- REST API communication with Ollama server
- Graceful process management for server lifecycle
