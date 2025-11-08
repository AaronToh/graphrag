# Test Questions Summary

## Overview

After searching through the codebase, here's what I found regarding test questions:

## Question Types Found

### ❌ **No Traditional MCQ Questions Found**

The codebase does **NOT** contain traditional multiple-choice questions (MCQ) with options like:
- A) Option 1
- B) Option 2  
- C) Option 3
- D) Option 4

### ✅ **Yes/No/Maybe Questions (PubMedQA)**

Instead, the codebase uses **binary/ternary questions** from the **PubMedQA dataset**:

#### Question Format:
- **Question**: Yes/No/Maybe biomedical question (e.g., "Are group 2 innate lymphoid cells (ILC2s) increased in chronic rhinosinusitis?")
- **Final Decision**: Ground truth answer (`yes` / `no` / `maybe`)
- **Long Answer**: Optional detailed explanation

#### Example from `data/gold/input/passages.jsonl`:
```json
{
  "question": "Are group 2 innate lymphoid cells (ILC2s) increased in chronic rhinosinusitis with nasal polyps or eosinophilia?",
  "final_decision": "yes",
  "long_answer": "As ILC2s are elevated in patients with CRSwNP, they may drive nasal polyp formation in CRS..."
}
```

## Question Sources

### 1. **PubMedQA Dataset** (Primary Source)
- **Dataset**: `vblagoje/PubMedQA_instruction` (HuggingFace)
- **Format**: Yes/No/Maybe questions
- **Location**: Loaded dynamically via `load_test_questions_from_pubmedqa()`
- **Files**: 
  - `eval/run_eval.py` (line 747)
  - `eval/eval_pruned.py` (line 172)
  - `data/HuggingFace_data_ingest/ingest_pubmedqa.py`

### 2. **Local Passages** (Secondary Source)
- **File**: `data/gold/input/passages.jsonl`
- **Format**: JSONL with questions embedded in `attrs.question`
- **Location**: Loaded via `load_test_questions_from_local_passages()` in `eval/generate_answers.py`
- **Structure**: Same Yes/No/Maybe format

### 3. **Custom Test Questions** (Tertiary Source)
- **File**: `eval/run_eval_usage.py` (line 65)
- **Format**: Hardcoded example questions (not MCQ)
- **Example**:
  ```python
  TestQuestion(
      question="What is machine learning?",
      ground_truth_answer="Machine learning is a subset of AI...",
      metadata={"category": "ML", "difficulty": "easy"}
  )
  ```

## Question Statistics

From `data/gold/input/passages.jsonl`:
- **Total questions**: 1,056+ (based on file count in `data/gold/text_input/`)
- **Question type**: All Yes/No/Maybe biomedical questions
- **Source**: PubMedQA dataset (`pqa_artificial` config)
- **Format**: Binary/ternary (not multiple choice)

## Evaluation Usage

### Current Evaluation Setup:
1. **Dataset**: PubMedQA (`vblagoje/PubMedQA_instruction`)
2. **Split**: `train`
3. **Samples**: 5 (for testing, can be increased)
4. **Question Type**: Yes/No/Maybe (not MCQ)

### Evaluation Scripts:
- `eval/run_eval.py`: Main evaluation runner
- `eval_all_pruning_methods.py`: Batch evaluation
- `eval/eval_pruned.py`: Pruned system evaluation

## Why No MCQ Questions?

The PubMedQA dataset is designed for **binary/ternary classification** rather than multiple-choice:
- Questions are answerable with `yes`, `no`, or `maybe`
- This format is common in biomedical literature QA
- Simpler evaluation metrics (accuracy, not MRR for multiple options)

## If You Need MCQ Questions

If you want to add traditional MCQ questions, you would need to:

1. **Create a new dataset** with MCQ format:
   ```json
   {
     "question": "What is the primary function of mitochondria?",
     "options": {
       "A": "Protein synthesis",
       "B": "Energy production",
       "C": "DNA replication",
       "D": "Waste removal"
     },
     "correct_answer": "B",
     "explanation": "Mitochondria produce ATP through cellular respiration."
   }
   ```

2. **Update TestQuestion dataclass** to support options:
   ```python
   @dataclass
   class TestQuestion:
       question: str
       options: Optional[Dict[str, str]] = None  # {"A": "...", "B": "..."}
       correct_answer: Optional[str] = None  # "A", "B", "C", "D"
       ground_truth_answer: Optional[str] = None
       ...
   ```

3. **Update evaluation metrics** to handle MCQ format (accuracy, MRR for options)

## Summary

| Aspect | Details |
|--------|---------|
| **Question Type** | Yes/No/Maybe (binary/ternary) |
| **MCQ Questions** | ❌ None found |
| **Primary Source** | PubMedQA dataset (HuggingFace) |
| **Total Questions** | 1,056+ in local files, unlimited from HuggingFace |
| **Format** | `question` + `final_decision` (yes/no/maybe) |
| **Evaluation** | Currently uses 5 samples for testing |

## Files to Check

- `data/gold/input/passages.jsonl` - Local questions
- `eval/run_eval.py` - Evaluation runner with PubMedQA loader
- `data/HuggingFace_data_ingest/ingest_pubmedqa.py` - Dataset ingestion
- `eval/generate_answers.py` - Local passage loader

