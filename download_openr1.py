#!/usr/bin/env python3
"""Download OpenR1-Math-220k sample and save locally."""
import json
import os
from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "openr1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading open-r1/OpenR1-Math-220k (streaming)...")
ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", streaming=True)

# Collect up to 50000 examples that have boxed answers
records = []
count = 0
for ex in ds:
    count += 1
    if count % 10000 == 0:
        print(f"  Scanned {count} examples, collected {len(records)}...")

    # Extract messages
    msgs = ex.get("messages", [])
    if len(msgs) < 2:
        continue

    user_msg = None
    asst_msg = None
    for m in msgs:
        if m.get("role") == "user":
            user_msg = m.get("content", "")
        elif m.get("role") == "assistant":
            asst_msg = m.get("content", "")

    if user_msg and asst_msg and "\\boxed{" in asst_msg:
        records.append({
            "problem": user_msg,
            "solution": asst_msg,
            "source": ex.get("source", ""),
        })

    if len(records) >= 50000:
        break

output_path = os.path.join(OUTPUT_DIR, "openr1_math.json")
with open(output_path, "w") as f:
    json.dump(records, f)
print(f"\nSaved {len(records)} examples to {output_path}")
print(f"Scanned {count} total examples")
