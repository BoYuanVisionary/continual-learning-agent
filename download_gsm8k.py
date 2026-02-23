#!/usr/bin/env python3
"""Download the GSM8K test set from HuggingFace and save locally.

GSM8K (Grade School Math 8K) is a math reasoning benchmark with
'question' and 'answer' fields. The answer field contains full
chain-of-thought reasoning followed by #### and the final numeric answer.

Usage:
    python download_gsm8k.py
"""
import json
import os
from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading GSM8K test set from HuggingFace...")
ds = load_dataset("openai/gsm8k", "main", split="test")
print(f"Total test examples: {len(ds)}")

records = []
for ex in ds:
    records.append({
        "question": ex["question"],
        "answer": ex["answer"],
    })

output_path = os.path.join(OUTPUT_DIR, "test.json")
with open(output_path, "w") as f:
    json.dump(records, f, indent=2)
print(f"Saved {len(records)} examples to {output_path}")

# Show a sample
print("\nSample entry:")
print(f"  Question: {records[0]['question'][:120]}...")
print(f"  Answer:   {records[0]['answer'][:120]}...")

# Extract and show the final numeric answer from the first example
final_answer = records[0]["answer"].split("####")[-1].strip()
print(f"  Final numeric answer: {final_answer}")
