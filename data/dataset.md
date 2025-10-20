# Dataset Information

## PubMedQA Instruction Dataset

We will use the [vblagoje/PubMedQA_instruction](https://huggingface.co/datasets/vblagoje/PubMedQA_instruction) dataset for evaluation purposes.

### Why This Dataset?

This dataset enables comprehensive evaluation across multiple dimensions:

1. **Correctness of Retrieved Documents**: Evaluate whether the system retrieves relevant context documents
2. **Faithfulness of the Response**: Assess if the generated response accurately reflects the retrieved information without hallucination
3. **Accuracy**: Compare generated responses against ground truth answers to measure correctness

### Dataset Structure

The PubMedQA dataset contains medical questions paired with:

- Ground truth context documents (abstracts from PubMed)
- Ground truth answers
- Supporting evidence from the context

### Optimization Strategy

**Note**: We can potentially split each ground truth context document into multiple nodes for more efficient GraphRAG processing. This approach allows:

- Better granularity in retrieval
- More precise node-level relevance scoring
- Improved graph construction with finer-grained relationships
- Reduced noise in community detection and summarization

## Example Usage

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("vblagoje/PubMedQA_instruction")

# Explore the dataset structure
print(ds)
print("\nExample entry:")
print(ds['train'][0])

# Access specific fields
example = ds['train'][0]
print(f"\nQuestion: {example['instruction']}")
print(f"\nContext: {example['context']}")
print(f"\nAnswer: {example['response']}")
```
