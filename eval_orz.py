#!/usr/bin/env python3
"""Evaluate Qwen2.5-3B-Instruct on ORZ math train split (local data)."""

import argparse
import json
import os
from datetime import datetime
from tqdm import tqdm
from utils import (
    load_model, generate_responses_batch, extract_boxed_answer,
    load_checkpoint, save_checkpoint, add_shard_args, shard_data, merge_shard_results,
)
from math_grader import math_equal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "orz")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then put your final answer in \\boxed{}."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run on first 1024 samples only")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--no_resume", action="store_true", help="Start fresh, ignore checkpoint")
    parser.add_argument("--merge_only", action="store_true", help="Only merge shard results")
    add_shard_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load local train data
    data_path = os.path.join(DATA_DIR, "train.json")
    print(f"Loading ORZ train data from {data_path}...")
    with open(data_path) as f:
        all_samples = json.load(f)

    n = 1024 if args.test else len(all_samples)
    all_samples = all_samples[:n]
    print(f"Total samples: {len(all_samples)}")

    if args.merge_only:
        results = merge_shard_results(RESULTS_DIR, "orz_train_checkpoint", args.num_shards)
        print(f"Merged {len(results)} results from {args.num_shards} shards")
    else:
        samples, shard_offset = shard_data(list(range(len(all_samples))), args.num_shards, args.shard_id)
        shard_tag = f"[shard {args.shard_id}/{args.num_shards}] " if args.num_shards > 1 else ""
        print(f"{shard_tag}{'[TEST] ' if args.test else ''}Evaluating {len(samples)} problems, batch_size={args.batch_size}")

        ckpt_path = os.path.join(RESULTS_DIR, f"orz_train_checkpoint_shard{args.shard_id}.json")
        results = []
        if not args.no_resume:
            results = load_checkpoint(ckpt_path)
        start_idx = len(results)

        if start_idx >= len(samples):
            print("All samples already completed. Use --no_resume to re-run.")
        else:
            model, tokenizer = load_model()

            for batch_start in tqdm(range(start_idx, len(samples), args.batch_size),
                                    desc=f"ORZ-train {shard_tag}",
                                    initial=start_idx // args.batch_size,
                                    total=(len(samples) + args.batch_size - 1) // args.batch_size):
                batch_end = min(batch_start + args.batch_size, len(samples))
                batch_indices = samples[batch_start:batch_end]
                batch = [all_samples[j] for j in batch_indices]

                messages_list = [
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": ex["0"]["value"]}]
                    for ex in batch
                ]

                responses, token_counts = generate_responses_batch(model, tokenizer, messages_list)

                for j, (example, response, n_tokens) in enumerate(zip(batch, responses, token_counts)):
                    gold_answer = example["1"]["ground_truth"]["value"]
                    pred_answer = extract_boxed_answer(response)
                    is_correct = math_equal(pred_answer, gold_answer)

                    results.append({
                        "index": shard_offset + batch_start + j,
                        "question": example["0"]["value"],
                        "gold_answer": gold_answer,
                        "pred_answer": pred_answer,
                        "correct": is_correct,
                        "gen_tokens": n_tokens,
                        "response": response,
                    })

                save_checkpoint(ckpt_path, results)

                if batch_end % 256 < args.batch_size or batch_end == len(samples):
                    correct_so_far = sum(r["correct"] for r in results)
                    print(f"  Progress: {batch_end}/{len(samples)}, "
                          f"Accuracy: {correct_so_far}/{len(results)} = {correct_so_far/len(results):.4f}")

    # Final metrics
    correct = sum(r["correct"] for r in results)
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    token_counts = [r["gen_tokens"] for r in results if "gen_tokens" in r]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    max_tokens_used = max(token_counts) if token_counts else 0
    hit_max = sum(1 for t in token_counts if t >= 1024)

    print(f"\nFinal ORZ Train Accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"Token stats: avg={avg_tokens:.1f}, max={max_tokens_used}, hit_max_limit={hit_max}/{len(token_counts)}")

    output = {
        "metadata": {
            "task": "ORZ math evaluation on train split",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "dataset": "Open-Reasoner-Zero/orz_math_72k_collection_extended",
            "split": "train (90% of original)",
            "num_samples": total,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_file": data_path,
        },
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "token_stats": {
            "avg_gen_tokens": avg_tokens,
            "max_gen_tokens": max_tokens_used,
            "hit_max_limit": hit_max,
            "total_with_stats": len(token_counts),
        },
        "results": results,
    }

    output_path = os.path.join(RESULTS_DIR, "orz_train_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
