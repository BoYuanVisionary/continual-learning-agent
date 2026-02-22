#!/usr/bin/env python3
"""Download NuminaMath-CoT and save locally for SFT training."""
import json
import os
from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "numinamath")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading NuminaMath-CoT from HuggingFace...")
ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
print(f"Total examples: {len(ds)}")

# Save as JSON with problem, solution, source fields
records = []
for ex in ds:
    records.append({
        "problem": ex["problem"],
        "solution": ex["solution"],
        "source": ex["source"],
    })

output_path = os.path.join(OUTPUT_DIR, "numinamath_cot.json")
with open(output_path, "w") as f:
    json.dump(records, f, indent=2)
print(f"Saved {len(records)} examples to {output_path}")

# Print source distribution
from collections import Counter
sources = Counter(r["source"] for r in records)
print("\nSource distribution:")
for src, cnt in sources.most_common():
    print(f"  {src}: {cnt}")
