#!/usr/bin/env python3
"""Download CodeAlpaca-20k dataset from HuggingFace.

Saves to data/codealpaca/codealpaca_20k.json

Usage:
    python download_codealpaca.py
"""

import json
import os
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "codealpaca")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "codealpaca_20k.json")

    if os.path.exists(output_path):
        print(f"Already exists: {output_path}")
        with open(output_path) as f:
            data = json.load(f)
        print(f"  {len(data)} examples")
        return

    print("Downloading CodeAlpaca-20k from HuggingFace...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

    data = []
    for ex in ds:
        data.append({
            "instruction": ex["instruction"],
            "input": ex.get("input", ""),
            "output": ex["output"],
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} examples to {output_path}")


if __name__ == "__main__":
    main()
