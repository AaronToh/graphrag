# Evaluation Runner

Framework for evaluating and comparing GraphRAG systems (baseline vs pruned configurations).

## Quick Start

### Compare Baseline vs Pruned

```bash
python eval/run_eval.py \
  --baseline workspace/output \
  --pruned workspace/pruned_output \
  --use-pubmedqa \
  --pubmedqa-samples 20 \
  --output-dir eval/results
```

### Run Ablation Study

```bash
python eval/run_eval.py \
  --ablation \
  --ablation-config eval/ablation_config.json \
  --use-pubmedqa \
  --pubmedqa-samples 50 \
  --output-dir eval/results
```

## Test Data

Uses the PubMedQA dataset for evaluation:

```bash
pip install datasets

python eval/run_eval.py \
  --use-pubmedqa \
  --pubmedqa-split train \
  --pubmedqa-samples 10 \
  --baseline workspace/output \
  --pruned workspace/pruned_output
```

## Evaluation Metrics

### Faithfulness Score

- **Range**: 0-1 (higher is better)
- Uses LLM to verify answers are grounded in retrieved documents
- No ground truth needed
- Default: OpenAI `gpt-4o-mini`

### Semantic Answer Similarity (SAS)

- **Range**: -1 to 1 (higher is better)
- Requires ground truth answers
- Uses sentence transformers

### Mean Reciprocal Rank (MRR)

- **Range**: 0-1 (higher is better)
- Requires ground truth document IDs
- Measures retrieval quality

### Response Time

- Average query response time in seconds
- Lower is better

## LLM Configuration

### OpenAI (Default)

```bash
python eval/run_eval.py \
  --baseline workspace/output \
  --pruned workspace/pruned_output \
  --faithfulness-provider openai \
  --faithfulness-model gpt-4o-mini
```

### Ollama (Local)

```bash
python eval/run_eval.py \
  --baseline workspace/output \
  --pruned workspace/pruned_output \
  --faithfulness-provider ollama \
  --faithfulness-model llama3.2
```

### OpenRouter

```bash
python eval/run_eval.py \
  --baseline workspace/output \
  --pruned workspace/pruned_output \
  --faithfulness-provider openrouter \
  --faithfulness-model meta-llama/llama-3.2-3b-instruct
```

## Output Files

### Comparison Mode

- `comparison_metrics_YYYYMMDD_HHMMSS.json`: Summary metrics
- `baseline_details_YYYYMMDD_HHMMSS.csv`: Per-question results for baseline
- `pruned_details_YYYYMMDD_HHMMSS.csv`: Per-question results for pruned

### Ablation Mode

- `ablation_results_YYYYMMDD_HHMMSS.csv`: Results for all configurations
