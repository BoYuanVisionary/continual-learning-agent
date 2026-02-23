#!/usr/bin/env python3
"""Evaluate a LoRA-finetuned Qwen2.5-3B-Instruct on GSM8K.

Usage:
    # Evaluate a finetuned model:
    python eval_gsm8k.py --adapter_path checkpoints/sft_numinamath_n1000_r16/final_adapter \
        --experiment_name sft_numinamath_n1000_r16

    # Evaluate baseline (no adapter):
    python eval_gsm8k.py --baseline --experiment_name baseline

    # With tolerant answer extraction (fallback to last number):
    python eval_gsm8k.py --adapter_path checkpoints/... --experiment_name ... --tolerant

Saves results to results/<experiment_name>_gsm8k.json.
"""

import argparse
import json
import os
import re
import torch
from datetime import datetime
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from math_grader import math_equal
from utils import extract_boxed_answer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then put your final answer in \\boxed{}."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Experiment name for results file naming")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate base model without any adapter")
    parser.add_argument("--tolerant", action="store_true",
                        help="Fall back to extracting the last number in the response "
                             "when \\boxed{} is not found")
    return parser.parse_args()


def load_finetuned_model(adapter_path):
    """Load base model + LoRA adapter, merge, and move to GPU."""
    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # Merge for faster inference
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model, tokenizer


def load_base_model():
    """Load the base model without any adapter."""
    print(f"Loading base model (no adapter): {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"Base model loaded on {device}")
    return model, tokenizer


def generate_batch(model, tokenizer, messages_list, max_new_tokens=1024):
    """Generate responses for a batch of message lists."""
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )

    padded_len = inputs["input_ids"].shape[1]
    responses = []
    for i in range(len(texts)):
        generated = output_ids[i][padded_len:]
        resp = tokenizer.decode(generated, skip_special_tokens=True).strip()
        responses.append(resp)
    return responses


def extract_gold_answer(answer_text):
    """Extract the final numeric answer from GSM8K answer field.

    GSM8K answers contain reasoning followed by #### and the final answer.
    """
    parts = answer_text.split("####")
    if len(parts) >= 2:
        return parts[-1].strip()
    # Fallback: return the whole thing stripped
    return answer_text.strip()


def extract_last_number(text):
    """Extract the last number appearing in the text.

    Used as a fallback when \\boxed{} is not found and --tolerant is set.
    Handles integers, decimals, negatives, and comma-separated numbers.
    """
    # Match numbers like 123, 1,234, 12.5, -3.14, etc.
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if matches:
        # Return the last match, removing commas
        return matches[-1].replace(",", "")
    return None


def eval_gsm8k(model, tokenizer, batch_size=64, tolerant=False):
    """Evaluate on GSM8K test set."""
    data_path = os.path.join(SCRIPT_DIR, "data", "gsm8k", "test.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"GSM8K test data not found at {data_path}. "
            "Run 'python download_gsm8k.py' first."
        )

    with open(data_path) as f:
        data = json.load(f)

    print(f"Evaluating on {len(data)} GSM8K test examples (tolerant={tolerant})")

    correct = 0
    total = 0
    boxed_found = 0
    fallback_used = 0
    individual_results = []

    for batch_start in tqdm(range(0, len(data), batch_size), desc="GSM8K eval"):
        batch = data[batch_start:batch_start + batch_size]
        messages_list = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ]
            for ex in batch
        ]
        responses = generate_batch(model, tokenizer, messages_list)

        for ex, resp in zip(batch, responses):
            gold = extract_gold_answer(ex["answer"])
            pred = extract_boxed_answer(resp)

            used_fallback = False
            if pred is not None:
                boxed_found += 1
            elif tolerant:
                # Fallback: extract last number from response
                pred = extract_last_number(resp)
                if pred is not None:
                    used_fallback = True
                    fallback_used += 1

            is_correct = math_equal(pred, gold) if pred is not None else False
            if is_correct:
                correct += 1
            total += 1

            individual_results.append({
                "index": total - 1,
                "question": ex["question"],
                "gold_answer": gold,
                "pred_answer": pred,
                "correct": is_correct,
                "used_fallback": used_fallback,
                "response": resp,
            })

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nGSM8K Results:")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Boxed answers found: {boxed_found}/{total} ({boxed_found / total * 100:.1f}%)")
    if tolerant:
        print(f"  Fallback extractions used: {fallback_used}/{total} ({fallback_used / total * 100:.1f}%)")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "boxed_found": boxed_found,
        "fallback_used": fallback_used,
        "tolerant": tolerant,
        "results": individual_results,
    }


def main():
    args = parse_args()

    if not args.baseline and args.adapter_path is None:
        raise ValueError("Must specify either --adapter_path or --baseline")

    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    if args.baseline:
        model, tokenizer = load_base_model()
    else:
        model, tokenizer = load_finetuned_model(args.adapter_path)

    # Run evaluation
    gsm8k_results = eval_gsm8k(
        model, tokenizer,
        batch_size=args.batch_size,
        tolerant=args.tolerant,
    )

    # Build full results object
    results = {
        "experiment_name": args.experiment_name,
        "adapter_path": args.adapter_path if not args.baseline else None,
        "baseline": args.baseline,
        "tolerant": args.tolerant,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gsm8k": {
            "accuracy": gsm8k_results["accuracy"],
            "correct": gsm8k_results["correct"],
            "total": gsm8k_results["total"],
            "boxed_found": gsm8k_results["boxed_found"],
            "fallback_used": gsm8k_results["fallback_used"],
        },
        "results": gsm8k_results["results"],
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"GSM8K RESULTS: {args.experiment_name}")
    print("=" * 60)
    print(f"  Accuracy:      {gsm8k_results['accuracy']:.4f} ({gsm8k_results['accuracy'] * 100:.2f}%)")
    print(f"  Correct:       {gsm8k_results['correct']}/{gsm8k_results['total']}")
    print(f"  Boxed found:   {gsm8k_results['boxed_found']}/{gsm8k_results['total']}")
    if args.tolerant:
        print(f"  Fallback used: {gsm8k_results['fallback_used']}/{gsm8k_results['total']}")
    print(f"  Mode:          {'baseline' if args.baseline else 'finetuned'}")
    if not args.baseline:
        print(f"  Adapter:       {args.adapter_path}")
    print("=" * 60)

    # Save results
    output_path = os.path.join(results_dir, f"{args.experiment_name}_gsm8k.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
