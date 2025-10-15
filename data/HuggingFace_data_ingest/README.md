
---

# PubMedQA Dataset Preparation for GraphRAG Ingestion

This README defines how to **extract, clean, and structure data from the Hugging Face `pubmed_qa` dataset** into a format ready for ingestion by Microsoft’s GraphRAG pipeline.
It focuses purely on **dataset handling and formatting** — ingestion and graph construction happen downstream.

---

## 1. Objective

Convert the **PubMedQA** dataset from the Hugging Face Hub into a clean, structured corpus that GraphRAG can ingest.

Each record will yield one or more **text passages** and minimal metadata, serialized into **JSONL** files under `data/input`.

---

## 2. Source Dataset

* **Dataset name:** [`pubmed_qa`](https://huggingface.co/datasets/pubmed_qa)
* **Load method:**

  ```python
  from datasets import load_dataset
  dataset = load_dataset("pubmed_qa", "pqa_artificial")  # or "pqa_labeled", "pqa_unlabeled"
  ```
* **Available splits:** typically `train` (some configs also have `test` / `validation`)
* **Primary fields of interest:**

| Field                      | Description                              | Usage                                 |
| -------------------------- | ---------------------------------------- | ------------------------------------- |
| `pubid`                    | PubMed identifier (string)               | Used as document ID                   |
| `context` / `context_text` | Title and abstract combined              | Used as the main text body            |
| `question`                 | Yes/No/Maybe biomedical question         | Used for evaluation only, not indexed |
| `final_decision`           | Ground truth answer (`yes`/`no`/`maybe`) | Used for supervision/evaluation only  |
| `long_answer`              | Optional long explanation                | Optional metadata, not indexed        |

> Note: Only `pubid` and `context` (or `context_text`) are needed for the retrieval corpus.
> Other fields are **ignored** for ingestion, they are to be used as ground truth for evaluation later.

---

## 3. Output Schema (to feed into GraphRAG)

You will produce **JSONL** records that represent the raw textual corpus and its metadata.
Each record corresponds to **one passage** derived from a PubMed abstract.

### Example

```json
{
  "id": "25685786::p0",
  "doc_id": "25685786",
  "text": "The nephrotic syndrome is associated with an increased risk of thromboembolic events...",
  "attrs": {
    "section": "abstract",
    "source": "pubmed_qa",
    "dataset_split": "train"
  }
}
```

### Required Fields

| Key      | Type   | Description                                        |
| -------- | ------ | -------------------------------------------------- |
| `id`     | string | Unique passage ID (e.g., `"PMID::p{chunk_index}"`) |
| `doc_id` | string | Parent document identifier (`pubid`)               |
| `text`   | string | Passage text (plain UTF-8, normalized whitespace)  |
| `attrs`  | object | Optional metadata (source, section, split, etc.)   |

---

## 4. Data Extraction Steps

### Step 1 — Load from Hugging Face

```python
from datasets import load_dataset
dataset = load_dataset("pubmed_qa", "pqa_labeled")
records = dataset["train"]
```

This automatically handles caching and versioning under:

```
~/.cache/huggingface/datasets/pubmed_qa/
```

---

### Step 2 — Select Relevant Columns

This is the corpus that will go into data/input

Keep only:

```python
record["pubid"]
record.get("context") or record.get("context_text")
```

Drop:

```python
question, final_decision, long_answer
```

---

This ensures GraphRAG’s ingestion layer can treat each passage as a discrete node without needing to handle large documents.

---

### Step 5 — Build Output Records

For each chunk, construct a minimal dictionary:

```python
{
  "id": f"{pubid}::p{chunk_idx}",
  "doc_id": pubid,
  "text": chunk_text,
  "attrs": {"section": "abstract", "source": "pubmed_qa"}
}
```

Collect these into a list and write them to a `.jsonl` file:

```python
with open("graph_ingest_pubmedqa/passages.jsonl", "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec) + "\n")
```

---

## 5. File Output Summary

| File                       | Purpose                                                 | Format             |
| -------------------------- | ------------------------------------------------------- | ------------------ |
| `passages.jsonl`           | All passage-level records, ready for GraphRAG ingestion | JSON Lines (UTF-8) |

---

## 6. Usage

### 6.1 Quick Start

The ingestion scripts are designed to be run from anywhere in the project. All scripts automatically handle directory paths correctly.

**From the project root:**
```bash
# Full dataset ingestion (211K+ records)
pixi run python data/HuggingFace_data_ingest/ingest_pubmedqa.py

# Test with smaller dataset (recommended first)
pixi run python data/HuggingFace_data_ingest/ingest_pubmedqa.py --limit 1000

# Run from any directory (scripts handle paths automatically)
cd /any/directory
pixi run python /path/to/graphrag/data/HuggingFace_data_ingest/ingest_pubmedqa.py
```

### 6.2 Command Line Options

```bash
python data/HuggingFace_data_ingest/ingest_pubmedqa.py [OPTIONS]

Options:
  --output-dir PATH     Output directory for processed files (default: data/input)
  --config TEXT         Dataset configuration: pqa_artificial, pqa_labeled, pqa_unlabeled (default: pqa_artificial)
  --limit INTEGER       Limit number of records to process (for testing)
  --verbose            Enable verbose logging
  --help               Show help message
```

### 6.3 Example Usage Scenarios

**1. Initial Testing (Recommended)**
```bash
# Test with 100 records to verify everything works
pixi run python data/HuggingFace_data_ingest/test_ingestion.py

# Test with 1K records
pixi run python data/HuggingFace_data_ingest/ingest_pubmedqa.py --limit 1000 --verbose
```

**2. Full Dataset Ingestion**
```bash
# Process all 211K+ records (takes ~15 seconds)
pixi run python data/HuggingFace_data_ingest/ingest_pubmedqa.py --verbose

# Custom output directory
pixi run python data/HuggingFace_data_ingest/ingest_pubmedqa.py --output-dir /custom/path --verbose
```

**3. Different Dataset Configurations**
```bash
# Use labeled dataset (expert annotations)
pixi run python data/HuggingFace_data_ingest/ingest_pubmedqa.py --config pqa_labeled

# Use unlabeled dataset
pixi run python data/HuggingFace_data_ingest/ingest_pubmedqa.py --config pqa_unlabeled
```

**4. Monitoring Progress**
```bash
# In another terminal, monitor ingestion progress
pixi run python data/HuggingFace_data_ingest/monitor_progress.py
```

### 6.4 Output Files

After successful ingestion, you'll find:

```
data/input/
├── passages.jsonl          # Main corpus file (GraphRAG-ready)
└── dataset_stats.json      # Processing statistics
```

**passages.jsonl format:**
```json
{
  "id": "25429730::p0",
  "doc_id": "25429730", 
  "text": "Chronic rhinosinusitis (CRS) is a heterogeneous disease...",
  "attrs": {
    "section": "abstract",
    "source": "pubmed_qa", 
    "dataset_split": "train",
    "dataset_config": "pqa_artificial",
    "question": "Are group 2 innate lymphoid cells...",
    "final_decision": "yes",
    "long_answer": "As ILC2s are elevated in patients..."
  }
}
```

### 6.5 Troubleshooting

**Common Issues:**

1. **ModuleNotFoundError**: Ensure you're using pixi environment
   ```bash
   pixi install  # Install dependencies
   pixi run python ...  # Always use pixi run
   ```

2. **Network Issues**: HuggingFace dataset download may be slow
   ```bash
   # Check if dataset is cached
   ls ~/.cache/huggingface/datasets/pubmed_qa/
   ```

3. **Disk Space**: Full dataset creates ~300MB JSONL file
   ```bash
   # Check available space
   df -h data/input/
   ```

4. **Permission Issues**: Ensure write access to output directory
   ```bash
   # Create directory if needed
   mkdir -p data/input
   chmod 755 data/input
   ```

### 6.6 Integration with GraphRAG

Once ingestion completes, the files are ready for GraphRAG indexing:

```bash
# Next step: Build GraphRAG index
pixi run python ingest/build_index.py

# Verify the passages.jsonl is compatible
head -n 1 data/input/passages.jsonl | jq .
```

### 6.7 Development & Testing

**Available Scripts:**

- `ingest_pubmedqa.py` - Main ingestion script
- `test_ingestion.py` - Quick test with 100 records  
- `inspect_dataset.py` - Explore dataset structure
- `monitor_progress.py` - Monitor ingestion progress

**Running Tests:**
```bash
# Quick validation test
pixi run python data/HuggingFace_data_ingest/test_ingestion.py

# Inspect dataset structure
pixi run python data/HuggingFace_data_ingest/inspect_dataset.py
```

## 7. Summary

**Goal:** Transform `pubmed_qa` from Hugging Face into text records for GraphRAG ingestion (pre-chunking JSON).
#### **We use:**
* Only `pubid` and `context/context_text`
* Cleaned and chunked into `passages.jsonl`
* No GraphRAG parameters or graph edges — those are handled downstream

> This step is purely about **data preparation and formatting** — ensuring the biomedical text corpus is ready for ingestion by GraphRAG’s existing ingestion and graph-building pipeline.
