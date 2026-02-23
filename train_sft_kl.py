#!/usr/bin/env python3
"""KL-regularized SFT training script for Qwen2.5-3B-Instruct with LoRA.

This is a modification of train_sft.py that adds a KL divergence penalty between
the fine-tuned model's output distribution and a frozen reference (base) model's
output distribution. This encourages the fine-tuned model to not deviate too far
from the base model, mitigating catastrophic forgetting.

Loss = SFT_loss + kl_weight * KL(fine_tuned || base)

Usage:
    python train_sft_kl.py --num_samples 1000 --data_source numinamath \
        --lora_rank 16 --lr 2e-4 --epochs 3 --kl_weight 0.1

Produces checkpoints under checkpoints/<experiment_name>/
"""

import argparse
import json
import os
import random
import torch
import torch.nn.functional as F
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Import data loading functions from train_sft.py
from train_sft import (
    load_numinamath_data,
    load_numinamath_hard_data,
    load_openr1_data,
    load_orz_self_data,
    MATH_SYSTEM_PROMPT,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="KL-regularized SFT training with LoRA")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="Number of training samples")
    parser.add_argument("--data_source", type=str, default="numinamath",
                        choices=["numinamath", "numinamath_comp", "numinamath_hard", "openr1", "orz_self"],
                        help="Data source for training")
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
    # KL-specific arguments
    parser.add_argument("--kl_weight", type=float, default=0.1,
                        help="Weight for KL divergence penalty (default: 0.1)")
    return parser.parse_args()


class KLRegularizedSFTTrainer(SFTTrainer):
    """SFTTrainer with KL divergence regularization against a frozen reference model.

    The loss is: L = SFT_loss + kl_weight * KL(policy || reference)

    The KL divergence is computed token-by-token on the logits, averaged over
    all non-padding tokens in the batch.
    """

    def __init__(self, ref_model, kl_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.kl_weight = kl_weight
        # Move reference model to same device as training model, freeze it
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute SFT loss + KL divergence penalty.

        The KL divergence KL(policy || ref) is computed as:
            KL = sum_t [ policy(t) * (log policy(t) - log ref(t)) ]
        where the sum is over vocabulary at each token position, then averaged
        over non-padding token positions.
        """
        # Forward pass through the trainable (policy) model
        outputs = model(**inputs)
        sft_loss = outputs.loss

        # Forward pass through the frozen reference model (no gradients)
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)

        # Get logits from both models
        # Shape: (batch_size, seq_len, vocab_size)
        policy_logits = outputs.logits
        ref_logits = ref_outputs.logits

        # Compute log probabilities
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # Compute KL divergence token-by-token: KL(policy || ref)
        # KL(P || Q) = sum_x P(x) * (log P(x) - log Q(x))
        policy_probs = F.softmax(policy_logits, dim=-1)
        kl_div = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
        # kl_div shape: (batch_size, seq_len)

        # Mask out padding tokens if attention_mask is available
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]
            # Also mask out the positions where labels == -100 (not part of loss)
            if "labels" in inputs:
                label_mask = (inputs["labels"] != -100).float()
                mask = attention_mask.float() * label_mask
            else:
                mask = attention_mask.float()
            # Average KL over non-masked positions
            kl_loss = (kl_div * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            kl_loss = kl_div.mean()

        # Combined loss
        total_loss = sft_loss + self.kl_weight * kl_loss

        # Log the individual loss components (will appear in training logs)
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            if hasattr(self, '_wandb_log') or True:
                # Store for logging - these will be printed in the training loop
                self._last_sft_loss = sft_loss.detach().item()
                self._last_kl_loss = kl_loss.detach().item()
                self._last_total_loss = total_loss.detach().item()

        if return_outputs:
            return total_loss, outputs
        return total_loss


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate experiment name
    if args.experiment_name is None:
        args.experiment_name = (
            f"sft_kl_{args.data_source}_n{args.num_samples}_r{args.lora_rank}_"
            f"lr{args.lr}_ep{args.epochs}_kl{args.kl_weight}"
        )

    checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints", args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"=== KL-Regularized SFT Training ===")
    print(f"Experiment: {args.experiment_name}")
    print(f"Samples: {args.num_samples}, Source: {args.data_source}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"LR: {args.lr}, Epochs: {args.epochs}")
    print(f"KL weight: {args.kl_weight}")
    print(f"Batch: {args.batch_size} x {args.grad_accum} grad_accum")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # Load data
    if args.data_source == "numinamath":
        train_data = load_numinamath_data(args.num_samples, args.seed,
                                          filter_sources=args.filter_sources)
    elif args.data_source == "numinamath_comp":
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
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

    # Save training data info
    data_info = {
        "source": args.data_source,
        "num_samples": len(train_data),
        "seed": args.seed,
        "kl_weight": args.kl_weight,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(checkpoint_dir, "data_info.json"), "w") as f:
        json.dump(data_info, f, indent=2)

    # ====================================================================
    # Load TWO copies of the model:
    #   1. Policy model (with LoRA, trainable)
    #   2. Reference model (frozen base, no LoRA)
    # Both fit on a single H200 (80GB): ~6GB each in bf16 = ~12GB total
    # ====================================================================

    print(f"\nLoading tokenizer from: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Right padding for training

    # Load the policy model (will get LoRA applied)
    print(f"Loading policy model: {MODEL_NAME}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # Load the reference model (frozen, no LoRA)
    print(f"Loading reference model: {MODEL_NAME}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    # Move reference model to GPU and freeze
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_model = ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print(f"Reference model loaded and frozen on {device}")

    # Apply LoRA to policy model
    target_modules = args.target_modules.split(",")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.print_trainable_parameters()

    # Report GPU memory usage
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU memory after loading both models: {mem_allocated:.1f}GB allocated, {mem_reserved:.1f}GB reserved")

    # Create dataset from messages
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

    # Use the custom KL-regularized trainer
    trainer = KLRegularizedSFTTrainer(
        ref_model=ref_model,
        kl_weight=args.kl_weight,
        model=policy_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting KL-regularized training (kl_weight={args.kl_weight})...")
    train_result = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Train runtime: {train_result.metrics['train_runtime']:.1f}s")

    # Log final KL component values if available
    if hasattr(trainer, '_last_sft_loss'):
        print(f"  Last SFT loss: {trainer._last_sft_loss:.4f}")
        print(f"  Last KL loss: {trainer._last_kl_loss:.4f}")
        print(f"  Last total loss: {trainer._last_total_loss:.4f}")

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
        "kl_weight": args.kl_weight,
        "train_loss": train_result.training_loss,
        "train_runtime_s": train_result.metrics["train_runtime"],
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        "adapter_path": final_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Include last KL component values if available
    if hasattr(trainer, '_last_sft_loss'):
        train_info["last_sft_loss"] = trainer._last_sft_loss
        train_info["last_kl_loss"] = trainer._last_kl_loss
        train_info["last_total_loss"] = trainer._last_total_loss

    with open(os.path.join(checkpoint_dir, "train_results.json"), "w") as f:
        json.dump(train_info, f, indent=2)

    # Free reference model memory
    del ref_model
    del trainer
    torch.cuda.empty_cache()

    print(f"\nAdapter saved to: {final_path}")
    print(f"Training info saved to: {checkpoint_dir}/train_results.json")


if __name__ == "__main__":
    main()
