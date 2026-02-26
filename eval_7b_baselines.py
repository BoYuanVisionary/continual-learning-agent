#!/usr/bin/env python3
"""Evaluate Qwen2.5-7B-Instruct baseline (no adapter) on all benchmarks.

Usage:
    python eval_7b_baselines.py [--batch_size 32]

Runs ORZ (1024 samples), SciKnowEval, ToolAlpaca, and GSM8K.
Saves to results/baseline_7b_eval.json
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--orz_samples", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()

    # Import eval functions from eval_finetuned (they work with any model/tokenizer)
    from eval_finetuned import eval_orz, eval_sciknoweval, eval_toolalpaca
    from utils import load_model

    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"=== 7B Baseline Evaluation: {MODEL_NAME} ===")
    model, tokenizer = load_model(MODEL_NAME)

    results = {"model": MODEL_NAME, "type": "baseline"}

    # ORZ
    print("\n--- ORZ Math ---")
    results["orz"] = eval_orz(model, tokenizer, args.batch_size, args.orz_samples)

    # SciKnowEval
    print("\n--- SciKnowEval ---")
    results["sciknoweval"] = eval_sciknoweval(model, tokenizer, args.batch_size)

    # ToolAlpaca
    print("\n--- ToolAlpaca ---")
    results["toolalpaca"] = eval_toolalpaca(model, tokenizer, min(args.batch_size, 16))

    # GSM8K
    print("\n--- GSM8K ---")
    try:
        from eval_gsm8k import eval_gsm8k
        results["gsm8k_strict"] = eval_gsm8k(model, tokenizer, args.batch_size, tolerant=False)
        # Remove individual results to save space
        results["gsm8k_strict"].pop("results", None)
        results["gsm8k_tolerant"] = eval_gsm8k(model, tokenizer, args.batch_size, tolerant=True)
        results["gsm8k_tolerant"].pop("results", None)
    except FileNotFoundError as e:
        print(f"  GSM8K skipped: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("7B BASELINE RESULTS")
    print("=" * 60)
    if "orz" in results:
        print(f"  ORZ Math:        {results['orz']['accuracy']:.4f}")
    if "sciknoweval" in results:
        print(f"  SciKnowEval:     {results['sciknoweval']['accuracy']:.4f}")
    if "toolalpaca" in results:
        ta = results["toolalpaca"]
        if "simulated" in ta and ta["simulated"]:
            print(f"  TA Sim func:     {ta['simulated']['func_accuracy']:.4f}")
            print(f"  TA Sim pass:     {ta['simulated']['pass_rate']:.4f}")
        if "real" in ta and ta["real"]:
            print(f"  TA Real func:    {ta['real']['func_accuracy']:.4f}")
            print(f"  TA Real pass:    {ta['real']['pass_rate']:.4f}")
    if "gsm8k_strict" in results:
        print(f"  GSM8K (strict):  {results['gsm8k_strict']['accuracy']:.4f}")
    if "gsm8k_tolerant" in results:
        print(f"  GSM8K (tolerant):{results['gsm8k_tolerant']['accuracy']:.4f}")
    print("=" * 60)

    output_path = os.path.join(results_dir, "baseline_7b_eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
