#!/usr/bin/env python3
"""SFT training script for Qwen2.5-3B-Instruct with LoRA.

Usage:
    python train_sft.py --num_samples 1000 --data_source numinamath \
        --lora_rank 16 --lr 2e-4 --epochs 3

Produces checkpoints under checkpoints/<experiment_name>/
"""

import argparse
import json
import os
import random
import torch
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

MATH_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then put your final answer in \\boxed{}."
)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training with LoRA")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="Number of training samples")
    parser.add_argument("--data_source", type=str, default="numinamath",
                        choices=["numinamath", "numinamath_comp", "numinamath_hard", "openr1", "orz_self", "orz_reject", "openr1_truncated", "mixed"],
                        help="Data source for training")
    parser.add_argument("--mix_ratio", type=float, default=0.5,
                        help="For 'mixed' source: fraction from NuminaMath (0.0=all OpenR1, 1.0=all NM)")
    parser.add_argument("--filter_sources", type=str, default=None,
                        help="Comma-separated NuminaMath sources to keep (e.g. olympiads,amc_aime,aops_forum)")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Custom experiment name (auto-generated if not set)")
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated LoRA target modules")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Warmup ratio")
    return parser.parse_args()


def load_numinamath_data(num_samples, seed=42, filter_sources=None):
    """Load NuminaMath-CoT data and format as chat messages."""
    data_path = os.path.join(SCRIPT_DIR, "data", "numinamath", "numinamath_cot.json")
    with open(data_path) as f:
        data = json.load(f)

    # Filter by source if specified
    if filter_sources:
        sources = set(s.strip() for s in filter_sources.split(","))
        data = [ex for ex in data if ex.get("source", "") in sources]
        print(f"Filtered to sources {sources}: {len(data)} examples")

    # Filter to only examples with \boxed{} in solution
    data = [ex for ex in data if "\\boxed{" in ex.get("solution", "")]
    print(f"After \\boxed filter: {len(data)} examples")

    # Shuffle and select
    rng = random.Random(seed)
    rng.shuffle(data)
    selected = data[:num_samples]

    # Format as chat messages for SFT
    formatted = []
    for ex in selected:
        solution = ex["solution"]
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": ex["problem"]},
            {"role": "assistant", "content": solution},
        ]
        formatted.append({"messages": messages})

    print(f"Loaded {len(formatted)} NuminaMath-CoT examples")
    return formatted


def load_numinamath_hard_data(num_samples, seed=42):
    """Load NuminaMath-CoT data, selecting problems with longest solutions (hardest)."""
    data_path = os.path.join(SCRIPT_DIR, "data", "numinamath", "numinamath_cot.json")
    with open(data_path) as f:
        data = json.load(f)

    # Filter to only competition sources + examples with boxed answers
    comp_sources = {"olympiads", "amc_aime", "aops_forum", "math"}
    data = [ex for ex in data if ex.get("source", "") in comp_sources and "\\boxed{" in ex.get("solution", "")]
    print(f"Competition math with \\boxed: {len(data)} examples")

    # Sort by solution length (longest = hardest) and take top
    data.sort(key=lambda x: len(x["solution"]), reverse=True)
    # Take top 3x samples to have variety, then sample from those
    pool = data[:max(num_samples * 3, len(data))]
    rng = random.Random(seed)
    rng.shuffle(pool)
    selected = pool[:num_samples]

    formatted = []
    for ex in selected:
        solution = ex["solution"]
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": ex["problem"]},
            {"role": "assistant", "content": solution},
        ]
        formatted.append({"messages": messages})

    avg_len = sum(len(ex["solution"]) for ex in selected) / len(selected) if selected else 0
    print(f"Loaded {len(formatted)} hard NuminaMath-CoT examples (avg solution length: {avg_len:.0f} chars)")
    return formatted


def load_openr1_data(num_samples, seed=42):
    """Load OpenR1-Math data and format as chat messages."""
    data_path = os.path.join(SCRIPT_DIR, "data", "openr1", "openr1_math.json")
    with open(data_path) as f:
        data = json.load(f)

    # OpenR1 has 'problem' and 'solution' or 'messages' fields
    # Check the format
    if data and "messages" in data[0]:
        # Has messages format - extract
        valid = []
        for ex in data:
            msgs = ex.get("messages", [])
            if len(msgs) >= 2:
                user_msg = None
                asst_msg = None
                for m in msgs:
                    if m.get("role") == "user":
                        user_msg = m.get("content", "")
                    elif m.get("role") == "assistant":
                        asst_msg = m.get("content", "")
                if user_msg and asst_msg and "\\boxed{" in asst_msg:
                    valid.append({"problem": user_msg, "solution": asst_msg})
        data = valid
        print(f"OpenR1 with boxed answers: {len(data)} examples")

    rng = random.Random(seed)
    rng.shuffle(data)
    selected = data[:num_samples]

    formatted = []
    for ex in selected:
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": ex["problem"]},
            {"role": "assistant", "content": ex["solution"]},
        ]
        formatted.append({"messages": messages})

    print(f"Loaded {len(formatted)} OpenR1 examples")
    return formatted


def load_orz_self_data(num_samples, seed=42):
    """Load ORZ problems with self-generated solutions (if available)."""
    data_path = os.path.join(SCRIPT_DIR, "data", "orz", "train.json")
    with open(data_path) as f:
        data = json.load(f)

    # For now, use ORZ problems with a simple answer format
    # In practice, we'd want to generate CoT solutions first
    rng = random.Random(seed)
    rng.shuffle(data)
    selected = data[:num_samples]

    formatted = []
    for ex in selected:
        question = ex["0"]["value"]
        answer = ex["1"]["ground_truth"]["value"]
        # Create a minimal solution with boxed answer
        solution = f"The answer is \\boxed{{{answer}}}."
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution},
        ]
        formatted.append({"messages": messages})

    print(f"Loaded {len(formatted)} ORZ self examples")
    return formatted


def load_orz_reject_data(num_samples, seed=42):
    """Load rejection-sampled ORZ data (generated by rejection_sample.py)."""
    data_path = os.path.join(SCRIPT_DIR, "data", "orz_self", "train_rejection.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Rejection-sampled data not found at {data_path}. "
            "Run 'python rejection_sample.py' first."
        )
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} rejection-sampled examples from {data_path}")

    rng = random.Random(seed)
    rng.shuffle(data)
    selected = data[:num_samples]
    print(f"Selected {len(selected)} rejection-sampled examples")
    return selected


def load_openr1_truncated_data(num_samples, seed=42, max_chars=1500):
    """Load OpenR1-Math data with solutions truncated to NuminaMath-like length.

    Truncates reasoning at max_chars, then splices the extracted boxed answer
    back onto the end. Produces ~1600 char solutions (vs 16,469 original).
    """
    import re as _re

    def _extract_boxed(text):
        """Extract answer from \\boxed{...} handling nested braces."""
        match = _re.search(r"\\boxed\{", text)
        if match:
            start = match.end()
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            if depth == 0:
                return text[start:i - 1].strip()
        return None

    data_path = os.path.join(SCRIPT_DIR, "data", "openr1", "openr1_math.json")
    with open(data_path) as f:
        data = json.load(f)

    # Extract valid examples with boxed answers
    # OpenR1 data has 'problem', 'solution', 'source' keys
    valid = []
    if data and "messages" in data[0]:
        # Messages format
        for ex in data:
            msgs = ex.get("messages", [])
            user_msg = None
            asst_msg = None
            for m in msgs:
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
                elif m.get("role") == "assistant":
                    asst_msg = m.get("content", "")
            if user_msg and asst_msg and "\\boxed{" in asst_msg:
                answer = _extract_boxed(asst_msg)
                if answer is not None:
                    valid.append({"problem": user_msg, "solution": asst_msg, "answer": answer})
    else:
        # Direct problem/solution format
        for ex in data:
            problem = ex.get("problem", "")
            solution = ex.get("solution", "")
            if problem and solution and "\\boxed{" in solution:
                answer = _extract_boxed(solution)
                if answer is not None:
                    valid.append({"problem": problem, "solution": solution, "answer": answer})
    print(f"OpenR1 truncated: {len(valid)} valid examples with extractable answers")

    rng = random.Random(seed)
    rng.shuffle(valid)
    selected = valid[:num_samples]

    # Truncate and splice
    formatted = []
    total_len = 0
    for ex in selected:
        solution = ex["solution"]
        answer = ex["answer"]
        if len(solution) > max_chars:
            truncated = solution[:max_chars]
            # Splice answer back
            truncated = truncated.rstrip() + f"\n\nTherefore, the answer is \\boxed{{{answer}}}."
        else:
            truncated = solution
        total_len += len(truncated)
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": ex["problem"]},
            {"role": "assistant", "content": truncated},
        ]
        formatted.append({"messages": messages})

    avg_len = total_len / len(formatted) if formatted else 0
    print(f"Loaded {len(formatted)} OpenR1-truncated examples (avg solution length: {avg_len:.0f} chars)")
    return formatted


def load_mixed_data(num_samples, mix_ratio=0.5, seed=42):
    """Load mixed NuminaMath + OpenR1 data.

    Args:
        num_samples: Total number of samples
        mix_ratio: Fraction from NuminaMath (0.0 = all OpenR1, 1.0 = all NM)
        seed: Random seed
    """
    n_nm = int(num_samples * mix_ratio)
    n_or = num_samples - n_nm

    nm_data = load_numinamath_data(n_nm, seed) if n_nm > 0 else []
    or_data = load_openr1_data(n_or, seed) if n_or > 0 else []

    combined = nm_data + or_data
    rng = random.Random(seed)
    rng.shuffle(combined)

    print(f"Mixed data: {n_nm} NuminaMath + {n_or} OpenR1 = {len(combined)} total (ratio={mix_ratio})")
    return combined


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate experiment name
    if args.experiment_name is None:
        args.experiment_name = (
            f"sft_{args.data_source}_n{args.num_samples}_r{args.lora_rank}_"
            f"lr{args.lr}_ep{args.epochs}"
        )

    checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints", args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"=== SFT Training ===")
    print(f"Experiment: {args.experiment_name}")
    print(f"Samples: {args.num_samples}, Source: {args.data_source}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"LR: {args.lr}, Epochs: {args.epochs}")
    print(f"Batch: {args.batch_size} x {args.grad_accum} grad_accum")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # Load data
    if args.data_source == "numinamath":
        train_data = load_numinamath_data(args.num_samples, args.seed,
                                          filter_sources=args.filter_sources)
    elif args.data_source == "numinamath_comp":
        # Competition-level math only
        train_data = load_numinamath_data(
            args.num_samples, args.seed,
            filter_sources="olympiads,amc_aime,aops_forum,math"
        )
    elif args.data_source == "numinamath_hard":
        train_data = load_numinamath_hard_data(args.num_samples, args.seed)
    elif args.data_source == "openr1":
        train_data = load_openr1_data(args.num_samples, args.seed)
    elif args.data_source == "orz_self":
        train_data = load_orz_self_data(args.num_samples, args.seed)
    elif args.data_source == "orz_reject":
        train_data = load_orz_reject_data(args.num_samples, args.seed)
    elif args.data_source == "openr1_truncated":
        train_data = load_openr1_truncated_data(args.num_samples, args.seed)
    elif args.data_source == "mixed":
        train_data = load_mixed_data(args.num_samples, args.mix_ratio, args.seed)
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    # Save training data info
    data_info = {
        "source": args.data_source,
        "num_samples": len(train_data),
        "seed": args.seed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(checkpoint_dir, "data_info.json"), "w") as f:
        json.dump(data_info, f, indent=2)

    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Right padding for training

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # Apply LoRA
    target_modules = args.target_modules.split(",")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dataset from messages
    from datasets import Dataset
    dataset = Dataset.from_list(train_data)

    # Format messages using chat template
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_chat, remove_columns=["messages"])

    # Effective batch size
    effective_batch = args.batch_size * args.grad_accum
    total_steps = (len(dataset) * args.epochs) // effective_batch

    print(f"\nDataset size: {len(dataset)}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Estimated total steps: {total_steps}")

    # Training config
    training_args = SFTConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=max(1, total_steps // 20),
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting training...")
    train_result = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Train runtime: {train_result.metrics['train_runtime']:.1f}s")

    # Save final model (LoRA adapter only)
    final_path = os.path.join(checkpoint_dir, "final_adapter")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    # Save training results
    train_info = {
        "experiment_name": args.experiment_name,
        "model": MODEL_NAME,
        "data_source": args.data_source,
        "num_samples": args.num_samples,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_length": args.max_seq_length,
        "target_modules": target_modules,
        "seed": args.seed,
        "train_loss": train_result.training_loss,
        "train_runtime_s": train_result.metrics["train_runtime"],
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "adapter_path": final_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(checkpoint_dir, "train_results.json"), "w") as f:
        json.dump(train_info, f, indent=2)

    print(f"\nAdapter saved to: {final_path}")
    print(f"Training info saved to: {checkpoint_dir}/train_results.json")


if __name__ == "__main__":
    main()
