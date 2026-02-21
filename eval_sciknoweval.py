"""Evaluate Qwen2.5-3B-Instruct on SciKnowEval Chemistry L3 (MCQ only)."""

import argparse
import json
import os
import re
from tqdm import tqdm
from utils import (
    load_model, generate_responses_batch,
    load_checkpoint, save_checkpoint,
    add_shard_args, shard_data, merge_shard_results,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "sciknoweval", "train.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def build_prompt(example):
    system_msg = "You are a helpful scientific assistant. Think through problems carefully before answering."
    question = example["question"]

    choices = example["choices"]["text"]
    labels = ["A", "B", "C", "D"]
    options_str = "\n".join(
        f"{labels[i]}. {choices[i]}" for i in range(min(len(choices), 4))
    )
    user_msg = (
        f"{question}\n\nOptions:\n{options_str}\n\n"
        "Think step by step, then provide your final answer as \"Answer: X\" "
        "where X is A, B, C, or D."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def extract_mcq_answer(response):
    response = response.strip()
    # Look for "Answer: X" pattern (last occurrence, since CoT comes first)
    matches = re.findall(r"[Aa]nswer\s*:\s*([A-Da-d])\b", response)
    if matches:
        return matches[-1].upper()
    # Fallback: last standalone letter A-D in the response
    match = re.findall(r"\b([A-Da-d])\b", response)
    if match:
        return match[-1].upper()
    return response.strip()[:1].upper() if response else ""


def get_gold_answer(example):
    answer_key = example.get("answerKey", "")
    if answer_key:
        return str(answer_key).strip().upper()
    return str(example.get("answer", "")).strip().upper()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run on first 10 samples only")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--no_resume", action="store_true", help="Start fresh, ignore checkpoint")
    parser.add_argument("--merge_only", action="store_true", help="Only merge shard results, no inference")
    add_shard_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load local MCQ-only dataset
    print("Loading SciKnowEval MCQ dataset...")
    with open(DATA_PATH) as f:
        all_filtered = json.load(f)
    print(f"  MCQ examples: {len(all_filtered)}")

    if not all_filtered:
        print("ERROR: No examples found. Exiting.")
        return

    if args.test:
        all_filtered = all_filtered[:10]
        print(f"[TEST MODE] Using first {len(all_filtered)} examples")

    # Merge mode: combine shard results and produce final output
    if args.merge_only:
        results = merge_shard_results(RESULTS_DIR, "sciknoweval_checkpoint", args.num_shards)
        print(f"Merged {len(results)} results from {args.num_shards} shards")
    else:
        # Shard the data
        samples, shard_offset = shard_data(list(range(len(all_filtered))), args.num_shards, args.shard_id)
        shard_tag = f"[shard {args.shard_id}/{args.num_shards}] " if args.num_shards > 1 else ""
        print(f"{shard_tag}{'[TEST] ' if args.test else ''}Evaluating {len(samples)} problems, batch_size={args.batch_size}")

        # Checkpoint per shard
        ckpt_path = os.path.join(RESULTS_DIR, f"sciknoweval_checkpoint_shard{args.shard_id}.json")
        results = []
        if not args.no_resume:
            results = load_checkpoint(ckpt_path)
        start_idx = len(results)

        if start_idx >= len(samples):
            print("All samples already completed. Use --no_resume to re-run.")
        else:
            model, tokenizer = load_model()

            for batch_start in tqdm(range(start_idx, len(samples), args.batch_size),
                                    desc=f"SciKnowEval {shard_tag}",
                                    initial=start_idx // args.batch_size,
                                    total=(len(samples) + args.batch_size - 1) // args.batch_size):
                batch_end = min(batch_start + args.batch_size, len(samples))
                batch_indices = samples[batch_start:batch_end]
                batch = [all_filtered[j] for j in batch_indices]

                messages_list = [build_prompt(ex) for ex in batch]

                responses, token_counts = generate_responses_batch(model, tokenizer, messages_list)

                for j, (example, response, n_tokens) in enumerate(zip(batch, responses, token_counts)):
                    gold = get_gold_answer(example)
                    pred = extract_mcq_answer(response)

                    results.append({
                        "index": shard_offset + batch_start + j,
                        "question": example["question"],
                        "gold": gold,
                        "pred": pred,
                        "correct": pred == gold,
                        "gen_tokens": n_tokens,
                        "response": response,
                    })

                save_checkpoint(ckpt_path, results)

                if batch_end % 100 < args.batch_size or batch_end == len(samples):
                    correct_so_far = sum(r["correct"] for r in results)
                    print(f"  Progress: {batch_end}/{len(samples)}, "
                          f"Accuracy: {correct_so_far}/{len(results)} = {correct_so_far/len(results):.4f}")

    # Compute metrics
    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    token_counts = [r["gen_tokens"] for r in results if "gen_tokens" in r]
    max_tokens_used = max(token_counts) if token_counts else 0
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    hit_max = sum(1 for t in token_counts if t >= 1024)

    print(f"\nSciKnowEval MCQ Results:")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"  Token stats: avg={avg_tokens:.1f}, max={max_tokens_used}, hit_max_limit={hit_max}/{len(token_counts)}")

    output = {
        "dataset": "SciKnowEval Chemistry L3 (MCQ only)",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "metrics": {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        },
        "token_stats": {
            "avg_gen_tokens": avg_tokens,
            "max_gen_tokens": max_tokens_used,
            "hit_max_limit": hit_max,
            "total_with_stats": len(token_counts),
        },
        "results": results,
    }

    output_path = os.path.join(RESULTS_DIR, "sciknoweval_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
