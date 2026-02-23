#!/usr/bin/env python3
"""Analyze model outputs to disentangle format changes from knowledge loss.

For selected checkpoints, runs ORZ evaluation with detailed output logging:
- Count responses with valid \boxed{} answers
- Count responses where correct answer appears somewhere but not in \boxed{}
- Measure average response length
- Categorize errors: format error vs genuine reasoning error
- Compare with baseline outputs

This is the KEY analysis for the paper: does SFT degradation come from
the model forgetting math knowledge, or from changing its output format?

Requires GPU - submit via SLURM.
"""

import json
import os
import re
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from math_grader import math_equal
from utils import extract_boxed_answer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then put your final answer in \\boxed{}."
)


def load_model(adapter_path=None):
    """Load base model, optionally with LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True,
    )

    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
    else:
        model = base_model

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, messages_list, max_new_tokens=1024):
    """Generate responses for a batch."""
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.8,
        )

    padded_len = inputs["input_ids"].shape[1]
    responses = []
    for i in range(len(texts)):
        generated = output_ids[i][padded_len:]
        resp = tokenizer.decode(generated, skip_special_tokens=True).strip()
        responses.append(resp)
    return responses


def extract_last_number(text):
    """Extract the last number from text as fallback."""
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if matches:
        return matches[-1].replace(",", "")
    return None


def answer_appears_in_text(gold, text):
    """Check if the gold answer appears somewhere in the response text."""
    if gold is None or text is None:
        return False
    # Try direct string match
    if gold in text:
        return True
    # Try matching as a number
    try:
        gold_num = float(gold.replace(",", ""))
        # Find all numbers in text
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
        for n in numbers:
            try:
                if abs(float(n.replace(",", "")) - gold_num) < 1e-6:
                    return True
            except:
                pass
    except:
        pass
    # Try math_equal on extracted numbers
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    for n in numbers:
        if math_equal(n.replace(",", ""), gold):
            return True
    return False


def analyze_checkpoint(model, tokenizer, experiment_name, num_samples=1024, batch_size=64):
    """Run ORZ evaluation with detailed output analysis."""
    data_path = os.path.join(SCRIPT_DIR, "data", "orz", "train.json")
    with open(data_path) as f:
        data = json.load(f)
    data = data[:num_samples]

    results = []
    correct_boxed = 0
    correct_tolerant = 0
    has_boxed = 0
    answer_in_text = 0
    total = 0
    response_lengths = []

    for batch_start in tqdm(range(0, len(data), batch_size), desc=f"Eval {experiment_name}"):
        batch = data[batch_start:batch_start + batch_size]
        messages_list = [
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": ex["0"]["value"]}]
            for ex in batch
        ]
        responses = generate_batch(model, tokenizer, messages_list)

        for ex, resp in zip(batch, responses):
            gold = ex["1"]["ground_truth"]["value"]
            pred_boxed = extract_boxed_answer(resp)
            pred_last_num = extract_last_number(resp)

            is_correct_boxed = math_equal(pred_boxed, gold) if pred_boxed else False
            is_correct_tolerant = is_correct_boxed or (
                math_equal(pred_last_num, gold) if pred_last_num else False
            )
            gold_in_text = answer_appears_in_text(gold, resp)

            entry = {
                "index": total,
                "gold": gold,
                "pred_boxed": pred_boxed,
                "pred_last_num": pred_last_num,
                "has_boxed": pred_boxed is not None,
                "correct_boxed": is_correct_boxed,
                "correct_tolerant": is_correct_tolerant,
                "gold_in_text": gold_in_text,
                "response_length": len(resp),
                "response": resp,
            }
            results.append(entry)

            if pred_boxed is not None:
                has_boxed += 1
            if is_correct_boxed:
                correct_boxed += 1
            if is_correct_tolerant:
                correct_tolerant += 1
            if gold_in_text:
                answer_in_text += 1
            response_lengths.append(len(resp))
            total += 1

    # Compute statistics
    avg_len = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    median_len = sorted(response_lengths)[len(response_lengths) // 2] if response_lengths else 0

    # Error categorization
    format_errors = 0  # Has correct answer in text but not in \boxed{}
    reasoning_errors = 0  # Gold answer not anywhere in text
    boxed_wrong = 0  # Has \boxed{} but wrong answer
    no_boxed_no_answer = 0  # No \boxed{} and gold not in text

    for r in results:
        if r["correct_boxed"]:
            continue  # Correct, no error
        if r["has_boxed"] and not r["correct_boxed"]:
            boxed_wrong += 1
        if not r["has_boxed"] and r["gold_in_text"]:
            format_errors += 1
        if not r["gold_in_text"]:
            reasoning_errors += 1
        if not r["has_boxed"] and not r["gold_in_text"]:
            no_boxed_no_answer += 1

    summary = {
        "experiment_name": experiment_name,
        "total": total,
        "correct_boxed": correct_boxed,
        "correct_tolerant": correct_tolerant,
        "has_boxed_count": has_boxed,
        "gold_in_text_count": answer_in_text,
        "accuracy_boxed": correct_boxed / total if total else 0,
        "accuracy_tolerant": correct_tolerant / total if total else 0,
        "boxed_rate": has_boxed / total if total else 0,
        "gold_in_text_rate": answer_in_text / total if total else 0,
        "avg_response_length": avg_len,
        "median_response_length": median_len,
        "error_analysis": {
            "format_errors": format_errors,
            "reasoning_errors": reasoning_errors,
            "boxed_wrong": boxed_wrong,
            "no_boxed_no_answer": no_boxed_no_answer,
        },
        "results": results,
    }

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint names to analyze (or 'baseline')")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_summaries = []

    for ckpt_name in args.checkpoints:
        print(f"\n{'='*60}")
        print(f"Analyzing: {ckpt_name}")
        print(f"{'='*60}")

        if ckpt_name == "baseline":
            model, tokenizer = load_model(adapter_path=None)
        else:
            adapter_path = os.path.join(SCRIPT_DIR, "checkpoints", ckpt_name, "final_adapter")
            if not os.path.exists(adapter_path):
                print(f"Adapter not found: {adapter_path}, skipping")
                continue
            model, tokenizer = load_model(adapter_path=adapter_path)

        summary = analyze_checkpoint(
            model, tokenizer, ckpt_name,
            num_samples=args.num_samples, batch_size=args.batch_size
        )

        # Print summary
        print(f"\n--- Summary for {ckpt_name} ---")
        print(f"  Accuracy (boxed):    {summary['accuracy_boxed']:.4f}")
        print(f"  Accuracy (tolerant): {summary['accuracy_tolerant']:.4f}")
        print(f"  Boxed rate:          {summary['boxed_rate']:.4f}")
        print(f"  Gold in text rate:   {summary['gold_in_text_rate']:.4f}")
        print(f"  Avg response length: {summary['avg_response_length']:.0f}")
        print(f"  Error analysis:")
        ea = summary["error_analysis"]
        print(f"    Format errors (correct in text, not in box): {ea['format_errors']}")
        print(f"    Reasoning errors (gold not in text):         {ea['reasoning_errors']}")
        print(f"    Boxed but wrong:                             {ea['boxed_wrong']}")

        # Save individual results
        out_path = os.path.join(RESULTS_DIR, f"{ckpt_name}_output_analysis.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved to: {out_path}")

        # Save summary without individual results for cross-experiment comparison
        summary_lite = {k: v for k, v in summary.items() if k != "results"}
        all_summaries.append(summary_lite)

        # Free model memory
        del model
        torch.cuda.empty_cache()

    # Save combined summary
    combined_path = os.path.join(RESULTS_DIR, "output_analysis_summary.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nCombined summary saved to: {combined_path}")

    # Print comparison table
    print(f"\n{'='*100}")
    print(f"{'Experiment':<55} {'Acc(box)':>8} {'Acc(tol)':>8} {'Box%':>6} {'Gold%':>6} {'AvgLen':>7}")
    print(f"{'-'*100}")
    for s in all_summaries:
        print(f"  {s['experiment_name']:<53} "
              f"{s['accuracy_boxed']:8.4f} {s['accuracy_tolerant']:8.4f} "
              f"{s['boxed_rate']:6.2f} {s['gold_in_text_rate']:6.2f} "
              f"{s['avg_response_length']:7.0f}")


if __name__ == "__main__":
    main()
