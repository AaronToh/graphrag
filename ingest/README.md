# GraphRAG Ingestion Module

This module handles data ingestion and knowledge graph indexing for the GraphRAG Pruning Lab project using OpenAI API.

## Overview

The ingestion pipeline processes biomedical text data (PubMedQA) and creates knowledge graph artifacts using Microsoft's GraphRAG framework with OpenAI models.

## Components

### 1. Data Conversion (`convert_jsonl.py`)
Converts JSONL format data to individual text files for GraphRAG processing.

**Usage:**
```bash
python -m ingest.convert_jsonl
```

**Input:** `data/input/passages.jsonl`  
**Output:** `data/text_input/*.txt` files

### 2. Index Building (`build_index.py`)
Runs GraphRAG indexing pipeline using OpenAI API to create knowledge graph artifacts.

**Usage:**
```bash
python -m ingest.build_index [--config CONFIG] [--overwrite] [--verbose]
```

**Arguments:**
- `--config`: Path to GraphRAG settings file (default: `workspace/settings.yaml`)
- `--overwrite`: Overwrite existing output files
- `--verbose`: Enable detailed logging

**Input:** `data/text_input/*.txt` files  
**Output:** `workspace/output/` (entities, relationships, communities, embeddings)

### 3. Logging (`logging_config.py`)
Provides centralized logging configuration for the ingestion process.

## Configuration

The ingestion process uses OpenAI API through GraphRAG's configuration system:

1. **API Key**: Set in `workspace/.env`
   ```
   GRAPHRAG_API_KEY=your_openai_api_key_here
   ```

2. **Models**: Configured in `workspace/settings.yaml`
   - Chat model: `gpt-4-turbo-preview`
   - Embedding model: `text-embedding-3-small`

3. **Entity Types**: Configured for biomedical domain
   - organization, person, geo, event
   - technology, symptom, disease, treatment, medication, biological_process

## Workflow

1. **Data Preparation**
   ```bash
   # Convert JSONL to text files
   python -m ingest.convert_jsonl
   ```

2. **Index Building**
   ```bash
   # Build knowledge graph index
   python -m ingest.build_index --verbose
   ```

3. **Verification**
   The system automatically verifies that all expected output files are created:
   - `entities.parquet`
   - `relationships.parquet`
   - `communities.parquet`
   - `community_reports.parquet`
   - `text_units.parquet`
   - `lancedb/` directory

## Output Structure

```
workspace/
├── output/
│   ├── entities.parquet          # Extracted entities
│   ├── relationships.parquet     # Entity relationships
│   ├── communities.parquet       # Community clusters
│   ├── community_reports.parquet # Community summaries
│   ├── text_units.parquet        # Text chunks
│   └── lancedb/                  # Vector embeddings
├── cache/                        # Processing cache
├── logs/                         # GraphRAG logs
└── prompts/                      # LLM prompts
```

## Logging

All operations are logged to:
- Console output (INFO level)
- `ingest/logs/indexing.log` (detailed logs)
- `workspace/logs/` (GraphRAG internal logs)

## Error Handling

The system includes comprehensive error checking:
- Input data validation
- OpenAI API configuration verification
- Output artifact verification
- Detailed error logging and recovery suggestions

## Dependencies

- `graphrag` - Microsoft GraphRAG framework
- OpenAI API access
- Python 3.12+
- Pixi environment management
