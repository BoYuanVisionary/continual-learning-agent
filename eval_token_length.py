#!/usr/bin/env python3
"""Evaluate the effect of max_new_tokens on ORZ and GSM8K accuracy.

Tests whether the OpenR1 cliff is a generation truncation artifact by
re-evaluating with higher token budgets (1024, 2048, 4096).

Usage:
    python eval_token_length.py --adapter_path checkpoints/X/final_adapter \
        --experiment_name X --max_tokens 1024 2048 4096
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
from utils import extract_boxed_answer, DEFAULT_MODEL_NAME

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(adapter_path, model_name=DEFAULT_MODEL_NAME):
    """Load base model + LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    device = "cuda:0"
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, messages_list, max_new_tokens=1024):
    """Generate responses with specified token budget."""
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
    gen_lengths = []
    for i in range(len(texts)):
        generated = output_ids[i][padded_len:]
        gen_lengths.append(len(generated))
        resp = tokenizer.decode(generated, skip_special_tokens=True).strip()
        responses.append(resp)
    return responses, gen_lengths


def eval_orz(model, tokenizer, max_new_tokens, batch_size=32, num_samples=1024):
    """Evaluate ORZ with specified max_new_tokens."""
    data_path = os.path.join(SCRIPT_DIR, "data", "orz", "train.json")
    with open(data_path) as f:
        data = json.load(f)
    data = data[:num_samples]

    system_prompt = (
        "You are a helpful math assistant. Solve the problem step by step, "
        "then put your final answer in \\boxed{}."
    )

    correct = 0
    total = 0
    total_gen_tokens = 0
    hit_limit = 0
    boxed_found = 0
    char_lengths = []

    for batch_start in tqdm(range(0, len(data), batch_size),
                            desc=f"ORZ (max_tokens={max_new_tokens})"):
        batch = data[batch_start:batch_start + batch_size]
        messages_list = [
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": ex["0"]["value"]}]
            for ex in batch
        ]
        responses, gen_lengths = generate_batch(
            model, tokenizer, messages_list, max_new_tokens=max_new_tokens
        )

        for ex, resp, gl in zip(batch, responses, gen_lengths):
            gold = ex["1"]["ground_truth"]["value"]
            pred = extract_boxed_answer(resp)

            total_gen_tokens += gl
            if gl >= max_new_tokens - 1:  # Hit the limit
                hit_limit += 1
            if pred is not None:
                boxed_found += 1
            char_lengths.append(len(resp))

            if math_equal(pred, gold):
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    avg_gen_tokens = total_gen_tokens / total if total > 0 else 0
    avg_chars = sum(char_lengths) / len(char_lengths) if char_lengths else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "max_new_tokens": max_new_tokens,
        "avg_gen_tokens": avg_gen_tokens,
        "avg_char_length": avg_chars,
        "hit_token_limit": hit_limit,
        "hit_limit_pct": hit_limit / total if total > 0 else 0,
        "boxed_found": boxed_found,
        "boxed_rate": boxed_found / total if total > 0 else 0,
    }


def eval_gsm8k(model, tokenizer, max_new_tokens, batch_size=32, tolerant=False):
    """Evaluate GSM8K with specified max_new_tokens."""
    data_path = os.path.join(SCRIPT_DIR, "data", "gsm8k_test.json")
    if not os.path.exists(data_path):
        # Try downloading
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
            examples = [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]
            with open(data_path, "w") as f:
                json.dump(examples, f)
        except Exception:
            print(f"  GSM8K data not found at {data_path}, skipping")
            return None

    with open(data_path) as f:
        data = json.load(f)

    system_prompt = (
        "You are a helpful math assistant. Solve the problem step by step, "
        "then put your final answer in \\boxed{}."
    )

    correct = 0
    total = 0
    hit_limit = 0
    boxed_found = 0

    for batch_start in tqdm(range(0, len(data), batch_size),
                            desc=f"GSM8K {'tolerant' if tolerant else 'strict'} (max_tokens={max_new_tokens})"):
        batch = data[batch_start:batch_start + batch_size]
        messages_list = [
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": ex["question"]}]
            for ex in batch
        ]
        responses, gen_lengths = generate_batch(
            model, tokenizer, messages_list, max_new_tokens=max_new_tokens
        )

        for ex, resp, gl in zip(batch, responses, gen_lengths):
            # Extract gold answer from GSM8K format: "#### number"
            gold_match = re.search(r'####\s*(.+)', ex["answer"])
            gold = gold_match.group(1).strip() if gold_match else ""

            if gl >= max_new_tokens - 1:
                hit_limit += 1

            pred = extract_boxed_answer(resp)
            if pred is not None:
                boxed_found += 1
            elif tolerant:
                # Fallback: last number in response
                numbers = re.findall(r'[-+]?\d*\.?\d+', resp)
                pred = numbers[-1] if numbers else ""

            if math_equal(pred, gold):
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "max_new_tokens": max_new_tokens,
        "tolerant": tolerant,
        "hit_token_limit": hit_limit,
        "hit_limit_pct": hit_limit / total if total > 0 else 0,
        "boxed_found": boxed_found,
        "boxed_rate": boxed_found / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter (None for baseline)")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_gsm8k", action="store_true")
    args = parser.parse_args()

    print(f"=== Token Length Evaluation: {args.experiment_name} ===")
    print(f"Model: {args.model_name}")
    print(f"Adapter: {args.adapter_path or 'None (baseline)'}")
    print(f"Token budgets: {args.max_tokens}")

    model, tokenizer = load_model(args.adapter_path, args.model_name)

    results = {
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "orz_by_max_tokens": {},
        "gsm8k_strict_by_max_tokens": {},
        "gsm8k_tolerant_by_max_tokens": {},
    }

    for mt in args.max_tokens:
        print(f"\n--- max_new_tokens = {mt} ---")

        orz = eval_orz(model, tokenizer, mt, batch_size=args.batch_size)
        results["orz_by_max_tokens"][str(mt)] = orz
        print(f"  ORZ: {orz['accuracy']:.4f} ({orz['correct']}/{orz['total']})")
        print(f"    Avg gen tokens: {orz['avg_gen_tokens']:.0f}, "
              f"hit limit: {orz['hit_token_limit']}/{orz['total']} ({orz['hit_limit_pct']:.1%})")
        print(f"    Boxed rate: {orz['boxed_rate']:.1%}")

        if not args.skip_gsm8k:
            gsm_s = eval_gsm8k(model, tokenizer, mt, batch_size=args.batch_size, tolerant=False)
            if gsm_s:
                results["gsm8k_strict_by_max_tokens"][str(mt)] = gsm_s
                print(f"  GSM8K strict: {gsm_s['accuracy']:.4f}, "
                      f"hit limit: {gsm_s['hit_token_limit']}/{gsm_s['total']} ({gsm_s['hit_limit_pct']:.1%})")

            gsm_t = eval_gsm8k(model, tokenizer, mt, batch_size=args.batch_size, tolerant=True)
            if gsm_t:
                results["gsm8k_tolerant_by_max_tokens"][str(mt)] = gsm_t
                print(f"  GSM8K tolerant: {gsm_t['accuracy']:.4f}")

    # Save results
    output_path = os.path.join(SCRIPT_DIR, "results",
                               f"{args.experiment_name}_token_length_eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'max_tokens':>12} | {'ORZ':>8} | {'Boxed%':>7} | {'HitLim%':>8} | {'GSM8K-S':>8} | {'GSM8K-T':>8}")
    print("-" * 70)
    for mt in args.max_tokens:
        mt_str = str(mt)
        orz = results["orz_by_max_tokens"].get(mt_str, {})
        gs = results["gsm8k_strict_by_max_tokens"].get(mt_str, {})
        gt = results["gsm8k_tolerant_by_max_tokens"].get(mt_str, {})
        print(f"{mt:>12} | {orz.get('accuracy',0):>7.1%} | {orz.get('boxed_rate',0):>6.1%} | "
              f"{orz.get('hit_limit_pct',0):>7.1%} | {gs.get('accuracy',0):>7.1%} | {gt.get('accuracy',0):>7.1%}")


if __name__ == "__main__":
    main()
