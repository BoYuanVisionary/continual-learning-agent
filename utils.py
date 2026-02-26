"""Shared utilities for Qwen2.5-3B-Instruct evaluation."""

import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME = DEFAULT_MODEL_NAME  # backward compat


def load_model(model_name=MODEL_NAME):
    """Load model and tokenizer onto a single GPU.

    Use CUDA_VISIBLE_DEVICES to control which physical GPU is used.
    The model is small enough (3B ~ 6GB FP16) to fit entirely on one H200.
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for batch generation
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"Model loaded on {device} (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
    return model, tokenizer


def generate_response(model, tokenizer, messages, max_new_tokens=1024):
    """Generate a response using the chat template."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def generate_responses_batch(model, tokenizer, messages_list, max_new_tokens=1024,
                             temperature=0.7, top_p=0.8):
    """Generate responses for a batch of message lists.

    Args:
        messages_list: list of message lists, each being [{"role":..., "content":...}, ...]
        temperature: sampling temperature (default 0.7)
        top_p: nucleus sampling threshold (default 0.8)

    Returns:
        (responses, gen_token_counts) tuple.
    """
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]
    # Tokenize each prompt individually to get true per-item lengths (before padding)
    prompt_lengths = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        prompt_lengths.append(ids.shape[1])
    # Now tokenize as a batch with left-padding
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    # With left-padding, each prompt ends at the same position (end of padded input).
    # The generated tokens start right after the padded input length.
    padded_len = inputs["input_ids"].shape[1]
    responses = []
    gen_token_counts = []
    for i in range(len(texts)):
        generated = output_ids[i][padded_len:]
        non_pad = (generated != tokenizer.pad_token_id).sum().item()
        gen_token_counts.append(non_pad)
        resp = tokenizer.decode(generated, skip_special_tokens=True).strip()
        responses.append(resp)
    return responses, gen_token_counts


# --- Checkpoint / Resume helpers ---

def load_checkpoint(checkpoint_path):
    """Load checkpoint results. Returns list of saved results or empty list."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            data = json.load(f)
        print(f"Resumed from checkpoint: {len(data)} results loaded from {checkpoint_path}")
        return data
    return []


def save_checkpoint(checkpoint_path, results):
    """Save results to checkpoint file."""
    with open(checkpoint_path, "w") as f:
        json.dump(results, f, indent=2)


# --- Sharding helpers ---

def shard_data(data, num_shards, shard_id):
    """Split data into shards. Returns the slice for shard_id."""
    total = len(data)
    per_shard = (total + num_shards - 1) // num_shards
    start = shard_id * per_shard
    end = min(start + per_shard, total)
    return data[start:end], start


def merge_shard_results(result_dir, prefix, num_shards):
    """Merge results from multiple shard checkpoint files."""
    all_results = []
    for sid in range(num_shards):
        path = os.path.join(result_dir, f"{prefix}_shard{sid}.json")
        if os.path.exists(path):
            with open(path) as f:
                all_results.extend(json.load(f))
        else:
            print(f"WARNING: shard file missing: {path}")
    return all_results


def add_shard_args(parser):
    """Add --num_shards and --shard_id arguments to an argparse parser."""
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="This shard's ID (0-indexed)")


def extract_boxed_answer(text):
    """Extract answer from \\boxed{...} or <answer>...</answer> tags."""
    # Try \boxed{...} first (handle nested braces)
    match = re.search(r"\\boxed\{", text)
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
            return text[start : i - 1].strip()

    # Try <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def normalize_math_answer(answer):
    """Normalize a math answer string for comparison."""
    if answer is None:
        return None
    s = str(answer).strip()
    s = s.strip("$")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.rstrip(".")
    s = s.replace(" ", "")
    s = s.replace("\\frac", "FRAC")
    s = s.replace("\\dfrac", "FRAC")
    s = s.replace("\\tfrac", "FRAC")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    s = s.replace("\\div", "/")
    s = s.replace("\\%", "%")
    s = s.replace("\\$", "$")
    s = s.replace("\\infty", "inf")
    s = s.replace("\\pi", "pi")
    s = s.lower()
    return s


def _try_parse_number(s):
    """Try to parse a string as a number."""
    try:
        frac_match = re.match(r"^-?FRAC\{([^}]+)\}\{([^}]+)\}$", s, re.IGNORECASE)
        if frac_match:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                return num / den
        if s.endswith("%"):
            return float(s[:-1]) / 100
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def math_equal(pred, gold):
    """Compare two math answers after normalization."""
    if pred is None or gold is None:
        return False
    pred_norm = normalize_math_answer(pred)
    gold_norm = normalize_math_answer(gold)
    if pred_norm is None or gold_norm is None:
        return False
    if pred_norm == gold_norm:
        return True
    pred_num = _try_parse_number(pred_norm)
    gold_num = _try_parse_number(gold_norm)
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < 1e-6
    return False
