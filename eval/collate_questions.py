"""
Collate questions from passages.jsonl

This script reads passages.jsonl and collates questions, answers, and options
for entries with the same doc_id.

Each output line includes:
- An array of original IDs (e.g., MH_train_1::support_0, MH_train_1::support_1, ...)
- A question
- The options
- The answer
"""

import json
from collections import defaultdict
from pathlib import Path


def collate_questions(input_file: str, output_file: str):
    """
    Collate questions from passages.jsonl file.

    Args:
        input_file: Path to the input passages.jsonl file
        output_file: Path to the output collated questions file
    """
    # Dictionary to group passages by doc_id
    doc_groups = defaultdict(list)

    # Read all passages and group by doc_id
    print(f"Reading passages from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line)
                    doc_id = entry["doc_id"]
                    question_id = int(doc_id.split("_")[-1])
                    doc_groups[question_id].append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                except KeyError as e:
                    print(f"Warning: Skipping line {line_num} due to missing key: {e}")

    print(f"Found {len(doc_groups)} unique doc_ids")

    # Collate questions and write output
    print(f"Writing collated questions to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for question_id in sorted(doc_groups.keys()):
            entries = doc_groups[question_id]

            # Sort entries by support_index to maintain order
            entries.sort(key=lambda x: x.get("attrs", {}).get("support_index", 0))

            # Extract common information (should be same for all entries with same question_id)
            first_entry = entries[0]
            attrs = first_entry.get("attrs", {})

            # Collect all IDs
            ids = [entry["id"] for entry in entries]

            # Extract question, answer, and candidates
            query = attrs.get("query", "")
            answer = attrs.get("answer", "")
            candidates = attrs.get("candidates", [])

            # Create output object
            output = {
                "question_id": question_id,
                "ids": ids,
                "question": query,
                "answer": answer,
                "options": candidates,
                "num_supports": len(entries),
            }

            # Write as JSON line
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"Successfully collated {len(doc_groups)} questions")


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    workspace_dir = script_dir.parent

    input_file = workspace_dir / "data" / "gold" / "input" / "passages.jsonl"
    output_file = workspace_dir / "data" / "gold" / "input" / "collated_questions.jsonl"

    # Run collation
    collate_questions(str(input_file), str(output_file))

    # Print sample output
    print("\nSample output:")
    with open(output_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 3:  # Show first 3 examples
                data = json.loads(line)
                print(f"\nQuestion ID: {data['question_id']}")
                print(f"Number of IDs: {len(data['ids'])}")
                print(f"First ID: {data['ids'][0]}")
                print(f"Last ID: {data['ids'][-1]}")
                print(f"Question: {data['question']}")
                print(f"Answer: {data['answer']}")
                print(f"Options: {data['options']}")
            else:
                break


if __name__ == "__main__":
    main()
