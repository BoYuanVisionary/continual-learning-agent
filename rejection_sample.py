#!/usr/bin/env python3
"""Generate in-distribution SFT data by rejection sampling from the base model.

For each ORZ train problem, generate K candidate solutions from Qwen2.5-3B-Instruct,
check if the extracted \boxed{} answer matches the gold answer using math_grader.math_equal,
and keep correct solutions as training data.

Usage:
    python rejection_sample.py --max_problems 5000 --num_attempts 4 --batch_size 32

Output:
    data/orz_self/train_rejection.json
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import torch

# Project root (directory containing this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils import load_model, generate_responses_batch, extract_boxed_answer
from math_grader import math_equal

MATH_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then put your final answer in \\boxed{}."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rejection sampling from base model on ORZ problems"
    )
    parser.add_argument(
        "--max_problems", type=int, default=5000,
        help="Maximum number of ORZ problems to attempt (default: 5000)"
    )
    parser.add_argument(
        "--num_attempts", type=int, default=4,
        help="Number of solution attempts per problem (K, default: 4)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for generation (default: 32)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048,
        help="Max new tokens per generation (default: 2048)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.8,
        help="Top-p (nucleus) sampling threshold (default: 0.8)"
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=100,
        help="Save checkpoint every N problems (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling problems (default: 42)"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output file path (default: data/orz_self/train_rejection.json)"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Checkpoint file path (default: data/orz_self/rejection_checkpoint.json)"
    )
    parser.add_argument(
        "--keep_all_correct", action="store_true",
        help="Keep all correct solutions per problem (default: keep only first correct)"
    )
    return parser.parse_args()


def load_orz_problems(max_problems, seed=42):
    """Load and shuffle ORZ train problems."""
    data_path = os.path.join(SCRIPT_DIR, "data", "orz", "train.json")
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} ORZ problems from {data_path}")

    rng = random.Random(seed)
    rng.shuffle(data)
    selected = data[:max_problems]
    print(f"Selected {len(selected)} problems (seed={seed})")
    return selected


def build_messages(question):
    """Build chat messages for a single problem."""
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def format_training_example(question, solution):
    """Format a correct solution as a training example (same format as train_sft.py)."""
    return {
        "messages": [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution},
        ]
    }


def load_checkpoint(checkpoint_path):
    """Load checkpoint with processed problem indices and collected results."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        print(f"Resumed from checkpoint: {ckpt['num_processed']} problems processed, "
              f"{len(ckpt['results'])} correct solutions collected")
        return ckpt
    return {
        "num_processed": 0,
        "results": [],
        "stats": {
            "total_attempted": 0,
            "total_generations": 0,
            "problems_with_correct": 0,
            "problems_with_boxed": 0,
        }
    }


def save_checkpoint(checkpoint_path, ckpt):
    """Save checkpoint atomically."""
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp_path, checkpoint_path)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set up paths
    output_dir = os.path.join(SCRIPT_DIR, "data", "orz_self")
    os.makedirs(output_dir, exist_ok=True)

    if args.output_path is None:
        args.output_path = os.path.join(output_dir, "train_rejection.json")
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(output_dir, "rejection_checkpoint.json")

    print("=" * 70)
    print("Rejection Sampling from Qwen2.5-3B-Instruct on ORZ Problems")
    print("=" * 70)
    print(f"Max problems:    {args.max_problems}")
    print(f"Attempts per problem (K): {args.num_attempts}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Temperature:     {args.temperature}")
    print(f"Top-p:           {args.top_p}")
    print(f"Max new tokens:  {args.max_new_tokens}")
    print(f"Keep all correct: {args.keep_all_correct}")
    print(f"Output:          {args.output_path}")
    print(f"Checkpoint:      {args.checkpoint_path}")
    print("=" * 70)

    # Load problems
    problems = load_orz_problems(args.max_problems, args.seed)

    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint_path)
    start_idx = ckpt["num_processed"]

    if start_idx >= len(problems):
        print("All problems already processed. Nothing to do.")
        # Save final output
        with open(args.output_path, "w") as f:
            json.dump(ckpt["results"], f, indent=2)
        print(f"Saved {len(ckpt['results'])} examples to {args.output_path}")
        return

    # Load model
    model, tokenizer = load_model()

    # Process problems in batches
    # Strategy: for each batch of problems, generate K attempts for all of them.
    # We process `batch_size` problems at a time. For each problem in the batch,
    # we generate K attempts sequentially (one attempt per round), but each round
    # processes all problems in the batch simultaneously.
    #
    # This means: for a batch of B problems with K attempts each, we do K rounds
    # of batch generation, each with B items.

    total_problems = len(problems)
    problems_remaining = problems[start_idx:]

    t_start = time.time()
    batch_count = 0

    for batch_start in range(0, len(problems_remaining), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(problems_remaining))
        batch_problems = problems_remaining[batch_start:batch_end]
        batch_size_actual = len(batch_problems)

        # Extract questions and gold answers
        questions = [p["0"]["value"] for p in batch_problems]
        gold_answers = [p["1"]["ground_truth"]["value"] for p in batch_problems]

        # Track which problems in this batch already have a correct solution
        # (relevant if not keep_all_correct: stop once we find one)
        found_correct = [False] * batch_size_actual
        batch_results = [[] for _ in range(batch_size_actual)]
        batch_had_boxed = [False] * batch_size_actual

        # Generate K attempts
        for attempt in range(args.num_attempts):
            # Determine which problems still need attempts
            if args.keep_all_correct:
                # Always attempt all problems for all K rounds
                active_indices = list(range(batch_size_actual))
            else:
                # Skip problems that already have a correct solution
                active_indices = [i for i in range(batch_size_actual) if not found_correct[i]]

            if not active_indices:
                break  # All problems in batch have correct solutions

            # Build messages for active problems
            messages_list = [build_messages(questions[i]) for i in active_indices]

            # Generate responses
            responses, _ = generate_responses_batch(
                model, tokenizer, messages_list,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            # Check each response
            for j, idx in enumerate(active_indices):
                response = responses[j]
                pred_answer = extract_boxed_answer(response)

                if pred_answer is not None:
                    batch_had_boxed[idx] = True
                    # Check correctness using math_grader.math_equal
                    try:
                        correct = math_equal(pred_answer, gold_answers[idx])
                    except Exception:
                        correct = False

                    if correct:
                        found_correct[idx] = True
                        example = format_training_example(questions[idx], response)
                        batch_results[idx].append(example)

        # Collect results from this batch
        for idx in range(batch_size_actual):
            ckpt["stats"]["total_attempted"] += 1
            ckpt["stats"]["total_generations"] += args.num_attempts
            if batch_had_boxed[idx]:
                ckpt["stats"]["problems_with_boxed"] += 1
            if found_correct[idx]:
                ckpt["stats"]["problems_with_correct"] += 1
                if args.keep_all_correct:
                    ckpt["results"].extend(batch_results[idx])
                else:
                    # Keep only the first correct solution
                    ckpt["results"].append(batch_results[idx][0])

        # Update processed count
        ckpt["num_processed"] = start_idx + batch_start + batch_size_actual
        batch_count += 1

        # Print progress
        elapsed = time.time() - t_start
        processed_so_far = batch_start + batch_size_actual
        total_to_process = len(problems_remaining)
        pct = 100 * processed_so_far / total_to_process
        success_rate = (
            100 * ckpt["stats"]["problems_with_correct"] / ckpt["stats"]["total_attempted"]
            if ckpt["stats"]["total_attempted"] > 0 else 0
        )
        rate = processed_so_far / elapsed if elapsed > 0 else 0
        eta = (total_to_process - processed_so_far) / rate if rate > 0 else 0

        print(
            f"[{processed_so_far}/{total_to_process}] ({pct:.1f}%) | "
            f"Correct: {ckpt['stats']['problems_with_correct']}/{ckpt['stats']['total_attempted']} "
            f"({success_rate:.1f}%) | "
            f"Total examples: {len(ckpt['results'])} | "
            f"Rate: {rate:.1f} prob/s | "
            f"ETA: {eta/60:.1f} min"
        )

        # Save checkpoint periodically
        if batch_count % max(1, args.checkpoint_every // args.batch_size) == 0:
            save_checkpoint(args.checkpoint_path, ckpt)
            print(f"  [Checkpoint saved: {ckpt['num_processed']} processed]")

    # Final checkpoint save
    save_checkpoint(args.checkpoint_path, ckpt)

    # Save final output
    with open(args.output_path, "w") as f:
        json.dump(ckpt["results"], f, indent=2)

    # Print summary
    elapsed_total = time.time() - t_start
    stats = ckpt["stats"]
    print("\n" + "=" * 70)
    print("REJECTION SAMPLING COMPLETE")
    print("=" * 70)
    print(f"Time elapsed:        {elapsed_total/60:.1f} minutes")
    print(f"Problems attempted:  {stats['total_attempted']}")
    print(f"Total generations:   {stats['total_generations']}")
    print(f"Problems with \\boxed: {stats['problems_with_boxed']} "
          f"({100*stats['problems_with_boxed']/max(1,stats['total_attempted']):.1f}%)")
    print(f"Problems solved:     {stats['problems_with_correct']} "
          f"({100*stats['problems_with_correct']/max(1,stats['total_attempted']):.1f}%)")
    print(f"Training examples:   {len(ckpt['results'])}")
    print(f"Output saved to:     {args.output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
