#!/usr/bin/env python3
"""Evaluate a LoRA-finetuned Qwen2.5-3B-Instruct on all three benchmarks.

Usage:
    python eval_finetuned.py --adapter_path checkpoints/sft_numinamath_n1000_r16/final_adapter \
        --experiment_name sft_numinamath_n1000_r16

Runs ORZ (1024 test samples), SciKnowEval (full), and ToolAlpaca (both splits).
Saves results to results/<experiment_name>_eval.json and appends to experiment_log.json.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Import grading utilities
from math_grader import math_equal
from utils import extract_boxed_answer

# --- Baselines (from CLAUDE.md) ---
BASELINES = {
    "orz_accuracy": 0.2891,
    "sciknoweval_accuracy": 0.3434,
    "toolalpaca_sim_func_acc": 0.7889,
    "toolalpaca_sim_pass_rate": 0.7222,
    "toolalpaca_real_func_acc": 0.8922,
    "toolalpaca_real_pass_rate": 0.8725,
}
BASELINES_7B = {
    "orz_accuracy": 0.3916,
    "sciknoweval_accuracy": 0.3582,
    "toolalpaca_sim_func_acc": 0.80,
    "toolalpaca_sim_pass_rate": 0.77,
    "toolalpaca_real_func_acc": 0.8860,
    "toolalpaca_real_pass_rate": 0.8860,
}
FORGETTING_THRESHOLD = 0.03  # 3% degradation threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Experiment name for results")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Base model name (default: Qwen2.5-3B-Instruct)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--orz_samples", type=int, default=1024,
                        help="Number of ORZ samples to evaluate")
    parser.add_argument("--skip_orz", action="store_true")
    parser.add_argument("--skip_sciknoweval", action="store_true")
    parser.add_argument("--skip_toolalpaca", action="store_true")
    return parser.parse_args()


def load_finetuned_model(adapter_path, model_name=DEFAULT_MODEL_NAME):
    """Load base model + LoRA adapter."""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
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


# --- ORZ Math Evaluation ---
def eval_orz(model, tokenizer, batch_size=64, num_samples=1024):
    """Evaluate on ORZ math."""
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
    for batch_start in tqdm(range(0, len(data), batch_size), desc="ORZ eval"):
        batch = data[batch_start:batch_start + batch_size]
        messages_list = [
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": ex["0"]["value"]}]
            for ex in batch
        ]
        responses = generate_batch(model, tokenizer, messages_list)

        for ex, resp in zip(batch, responses):
            gold = ex["1"]["ground_truth"]["value"]
            pred = extract_boxed_answer(resp)
            if math_equal(pred, gold):
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"ORZ: {correct}/{total} = {accuracy:.4f}")
    return {"accuracy": accuracy, "correct": correct, "total": total}


# --- SciKnowEval Evaluation ---
def eval_sciknoweval(model, tokenizer, batch_size=64):
    """Evaluate on SciKnowEval Chemistry MCQ."""
    data_path = os.path.join(SCRIPT_DIR, "data", "sciknoweval", "train.json")
    with open(data_path) as f:
        data = json.load(f)

    system_prompt = "You are a helpful scientific assistant. Think through problems carefully before answering."

    def build_prompt(example):
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def extract_mcq(response):
        matches = re.findall(r"[Aa]nswer\s*:\s*([A-Da-d])\b", response)
        if matches:
            return matches[-1].upper()
        match = re.findall(r"\b([A-Da-d])\b", response)
        if match:
            return match[-1].upper()
        return response.strip()[:1].upper() if response else ""

    def get_gold(example):
        answer_key = example.get("answerKey", "")
        if answer_key:
            return str(answer_key).strip().upper()
        return str(example.get("answer", "")).strip().upper()

    correct = 0
    total = 0
    for batch_start in tqdm(range(0, len(data), batch_size), desc="SciKnowEval"):
        batch = data[batch_start:batch_start + batch_size]
        messages_list = [build_prompt(ex) for ex in batch]
        responses = generate_batch(model, tokenizer, messages_list)

        for ex, resp in zip(batch, responses):
            gold = get_gold(ex)
            pred = extract_mcq(resp)
            if pred == gold:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"SciKnowEval: {correct}/{total} = {accuracy:.4f}")
    return {"accuracy": accuracy, "correct": correct, "total": total}


# --- ToolAlpaca Evaluation ---
def eval_toolalpaca(model, tokenizer, batch_size=32):
    """Evaluate on ToolAlpaca (both splits)."""
    toolalpaca_dir = os.path.join(SCRIPT_DIR, "ToolAlpaca", "data")

    ta_system = (
        "You are a helpful assistant that can use tools/APIs. "
        "Given the tool documentation and user request, determine which function to call "
        "and with what parameters. Respond in this format:\n"
        "Action: <function_name>\n"
        "Action Input: <parameters as JSON>"
    )

    def load_and_flatten(filename):
        filepath = os.path.join(toolalpaca_dir, filename)
        if not os.path.exists(filepath):
            return []
        with open(filepath) as f:
            data = json.load(f)
        examples = []
        for tool in data:
            nl_doc = tool.get("NLDocumentation", "")
            tool_name = tool.get("Name", "")
            instructions = tool.get("Instructions", [])
            golden = tool.get("Golden_Answers", [])
            for idx, instr in enumerate(instructions):
                gold_actions = golden[idx] if idx < len(golden) else []
                examples.append({
                    "instruction": instr,
                    "gold_actions": gold_actions,
                    "nl_doc": nl_doc,
                    "tool_name": tool_name,
                })
        return examples

    def parse_action(response):
        func_name = None
        params = None
        m = re.search(r"Action:\s*(.+?)(?:\n|$)", response)
        if m:
            func_name = m.group(1).strip()
        m = re.search(r"Action Input:\s*(.+?)(?:\n\n|\n(?=Action:)|$)", response, re.DOTALL)
        if m:
            ps = m.group(1).strip()
            try:
                params = json.loads(ps)
            except json.JSONDecodeError:
                params = ps
        return func_name, params

    def normalize_func(name):
        if name is None:
            return ""
        return name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")

    def evaluate_split(examples, split_name):
        func_correct = 0
        pass_count = 0
        total = 0
        for batch_start in tqdm(range(0, len(examples), batch_size), desc=f"ToolAlpaca {split_name}"):
            batch = examples[batch_start:batch_start + batch_size]
            messages_list = [
                [{"role": "system", "content": ta_system},
                 {"role": "user", "content": f"## Available Tools:\n{ex['nl_doc']}\n\n## User Request:\n{ex['instruction']}"}]
                for ex in batch
            ]
            responses = generate_batch(model, tokenizer, messages_list)

            for ex, resp in zip(batch, responses):
                pred_func, pred_params = parse_action(resp)
                gold_actions = ex["gold_actions"]
                gold_func = None
                gold_params = None
                if gold_actions and isinstance(gold_actions, list) and len(gold_actions) > 0:
                    first = gold_actions[0]
                    gold_func = first.get("Action")
                    gp = first.get("Action_Input", "{}")
                    try:
                        gold_params = json.loads(gp) if isinstance(gp, str) else gp
                    except json.JSONDecodeError:
                        gold_params = gp

                fm = normalize_func(pred_func) == normalize_func(gold_func) if gold_func else False
                if fm:
                    func_correct += 1

                op = False
                if fm and pred_params is not None:
                    if isinstance(gold_params, dict) and isinstance(pred_params, dict):
                        gk = set(k.lower() for k in gold_params.keys())
                        pk = set(k.lower() for k in pred_params.keys())
                        op = gk.issubset(pk) or pk == gk
                    else:
                        op = fm
                elif fm:
                    op = True
                if op:
                    pass_count += 1
                total += 1

        func_acc = func_correct / total if total > 0 else 0.0
        pass_rate = pass_count / total if total > 0 else 0.0
        print(f"ToolAlpaca {split_name}: func_acc={func_acc:.4f}, pass_rate={pass_rate:.4f} ({total} examples)")
        return {
            "func_accuracy": func_acc,
            "func_correct": func_correct,
            "pass_rate": pass_rate,
            "pass_count": pass_count,
            "total": total,
        }

    sim = load_and_flatten("eval_simulated.json")
    real = load_and_flatten("eval_real.json")

    sim_results = evaluate_split(sim, "simulated") if sim else {}
    real_results = evaluate_split(real, "real") if real else {}

    return {"simulated": sim_results, "real": real_results}


def check_validity(results, model_name=DEFAULT_MODEL_NAME):
    """Check if results stay within 3% of baseline."""
    baselines = BASELINES_7B if "7B" in model_name else BASELINES
    issues = []

    if "sciknoweval" in results:
        acc = results["sciknoweval"]["accuracy"]
        baseline_val = baselines.get("sciknoweval_accuracy")
        if baseline_val is None:
            issues.append(f"SciKnowEval: baseline not yet available for {model_name}")
            return len(issues) == 0, issues
        threshold = baseline_val - FORGETTING_THRESHOLD
        if acc < threshold:
            issues.append(f"SciKnowEval: {acc:.4f} < {threshold:.4f} (baseline {baseline_val:.4f})")

    if "toolalpaca" in results:
        ta = results["toolalpaca"]
        if "simulated" in ta and ta["simulated"]:
            fa = ta["simulated"]["func_accuracy"]
            bl = baselines.get("toolalpaca_sim_func_acc")
            if bl is not None:
                thr = bl - FORGETTING_THRESHOLD
                if fa < thr:
                    issues.append(f"ToolAlpaca Sim func_acc: {fa:.4f} < {thr:.4f}")
        if "real" in ta and ta["real"]:
            fa = ta["real"]["func_accuracy"]
            bl = baselines.get("toolalpaca_real_func_acc")
            if bl is not None:
                thr = bl - FORGETTING_THRESHOLD
                if fa < thr:
                    issues.append(f"ToolAlpaca Real func_acc: {fa:.4f} < {thr:.4f}")

    return len(issues) == 0, issues


def main():
    args = parse_args()
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_finetuned_model(args.adapter_path, args.model_name)

    results = {
        "experiment_name": args.experiment_name,
        "adapter_path": args.adapter_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Run evaluations
    if not args.skip_orz:
        results["orz"] = eval_orz(model, tokenizer, args.batch_size, args.orz_samples)

    if not args.skip_sciknoweval:
        results["sciknoweval"] = eval_sciknoweval(model, tokenizer, args.batch_size)

    if not args.skip_toolalpaca:
        results["toolalpaca"] = eval_toolalpaca(model, tokenizer, min(args.batch_size, 32))

    # Check validity
    valid, issues = check_validity(results, args.model_name)
    results["valid"] = valid
    results["forgetting_issues"] = issues

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS SUMMARY: {args.experiment_name}")
    print("=" * 60)
    if "orz" in results:
        print(f"  ORZ Math:       {results['orz']['accuracy']:.4f} (baseline: {BASELINES['orz_accuracy']:.4f})")
    if "sciknoweval" in results:
        print(f"  SciKnowEval:    {results['sciknoweval']['accuracy']:.4f} (baseline: {BASELINES['sciknoweval_accuracy']:.4f})")
    if "toolalpaca" in results:
        ta = results["toolalpaca"]
        if "simulated" in ta and ta["simulated"]:
            print(f"  TA Sim func:    {ta['simulated']['func_accuracy']:.4f} (baseline: {BASELINES['toolalpaca_sim_func_acc']:.4f})")
            print(f"  TA Sim pass:    {ta['simulated']['pass_rate']:.4f} (baseline: {BASELINES['toolalpaca_sim_pass_rate']:.4f})")
        if "real" in ta and ta["real"]:
            print(f"  TA Real func:   {ta['real']['func_accuracy']:.4f} (baseline: {BASELINES['toolalpaca_real_func_acc']:.4f})")
            print(f"  TA Real pass:   {ta['real']['pass_rate']:.4f} (baseline: {BASELINES['toolalpaca_real_pass_rate']:.4f})")
    print(f"\n  Valid (≤3% forgetting): {'YES' if valid else 'NO'}")
    if issues:
        for issue in issues:
            print(f"    - {issue}")
    print("=" * 60)

    # Save results
    output_path = os.path.join(results_dir, f"{args.experiment_name}_eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Append to experiment log
    log_path = os.path.join(results_dir, "experiment_log.json")
    log_entry = {
        "experiment_name": args.experiment_name,
        "adapter_path": args.adapter_path,
        "timestamp": results["timestamp"],
        "valid": valid,
    }
    if "orz" in results:
        log_entry["orz_accuracy"] = results["orz"]["accuracy"]
    if "sciknoweval" in results:
        log_entry["sciknoweval_accuracy"] = results["sciknoweval"]["accuracy"]
    if "toolalpaca" in results:
        ta = results["toolalpaca"]
        if "simulated" in ta and ta["simulated"]:
            log_entry["toolalpaca_sim_func_acc"] = ta["simulated"]["func_accuracy"]
            log_entry["toolalpaca_sim_pass_rate"] = ta["simulated"]["pass_rate"]
        if "real" in ta and ta["real"]:
            log_entry["toolalpaca_real_func_acc"] = ta["real"]["func_accuracy"]
            log_entry["toolalpaca_real_pass_rate"] = ta["real"]["pass_rate"]

    # Load existing log or create new
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
    else:
        log = []
    log.append(log_entry)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Experiment log updated: {log_path}")


if __name__ == "__main__":
    main()
