"""Evaluate Qwen2.5-3B-Instruct on ToolAlpaca benchmark."""

import argparse
import json
import os
import re
import subprocess
from tqdm import tqdm
from utils import (
    load_model, generate_responses_batch,
    load_checkpoint, save_checkpoint,
    add_shard_args, shard_data, merge_shard_results,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TOOLALPACA_DIR = os.path.join(SCRIPT_DIR, "ToolAlpaca")

SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools/APIs. "
    "Given the tool documentation and user request, determine which function to call "
    "and with what parameters. Respond in this format:\n"
    "Action: <function_name>\n"
    "Action Input: <parameters as JSON>"
)


def ensure_repo():
    if not os.path.exists(TOOLALPACA_DIR):
        print("Cloning ToolAlpaca repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/tangqiaoyu/ToolAlpaca.git", TOOLALPACA_DIR],
            check=True,
        )


def load_eval_data(filename):
    filepath = os.path.join(TOOLALPACA_DIR, "data", filename)
    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found")
        return []
    with open(filepath) as f:
        return json.load(f)


def flatten_examples(data):
    """Flatten: each tool has multiple Instructions/Golden_Answers."""
    examples = []
    for tool in data:
        nl_doc = tool.get("NLDocumentation", "")
        tool_name = tool.get("Name", "")
        instructions = tool.get("Instructions", [])
        golden_answers = tool.get("Golden_Answers", [])
        for idx, instruction in enumerate(instructions):
            gold_actions = golden_answers[idx] if idx < len(golden_answers) else []
            examples.append({
                "instruction": instruction,
                "gold_actions": gold_actions,
                "nl_doc": nl_doc,
                "tool_name": tool_name,
            })
    return examples


def build_prompt(nl_doc, instruction):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"## Available Tools:\n{nl_doc}\n\n## User Request:\n{instruction}"},
    ]


def parse_action(response):
    func_name = None
    params = None
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", response)
    if action_match:
        func_name = action_match.group(1).strip()
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n\n|\n(?=Action:)|$)", response, re.DOTALL)
    if input_match:
        params_str = input_match.group(1).strip()
        try:
            params = json.loads(params_str)
        except json.JSONDecodeError:
            params = params_str
    return func_name, params


def normalize_func_name(name):
    if name is None:
        return ""
    return name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def evaluate_single(example, response):
    """Evaluate a single example given model response. Returns result dict."""
    pred_func, pred_params = parse_action(response)

    gold_actions = example["gold_actions"]
    gold_func, gold_params = None, None
    if gold_actions and isinstance(gold_actions, list) and len(gold_actions) > 0:
        first_gold = gold_actions[0]
        gold_func = first_gold.get("Action")
        gp = first_gold.get("Action_Input", "{}")
        try:
            gold_params = json.loads(gp) if isinstance(gp, str) else gp
        except json.JSONDecodeError:
            gold_params = gp

    func_match = (
        normalize_func_name(pred_func) == normalize_func_name(gold_func)
        if gold_func else False
    )

    overall_pass = False
    if func_match and pred_params is not None:
        if isinstance(gold_params, dict) and isinstance(pred_params, dict):
            gold_keys = set(k.lower() for k in gold_params.keys())
            pred_keys = set(k.lower() for k in pred_params.keys())
            overall_pass = gold_keys.issubset(pred_keys) or pred_keys == gold_keys
        else:
            overall_pass = func_match
    elif func_match:
        overall_pass = True

    return {
        "tool": example["tool_name"],
        "instruction": example["instruction"],
        "gold_function": gold_func,
        "pred_function": pred_func,
        "func_match": func_match,
        "overall_pass": overall_pass,
        "response": response,
    }


def evaluate_split(examples, model, tokenizer, split_name, batch_size, checkpoint_path,
                    no_resume, num_shards=1, shard_id=0):
    """Evaluate one split with batching, checkpointing, resume, and sharding."""
    # Shard the examples
    indices, shard_offset = shard_data(list(range(len(examples))), num_shards, shard_id)
    shard_tag = f"[shard {shard_id}/{num_shards}] " if num_shards > 1 else ""

    results = []
    if not no_resume:
        results = load_checkpoint(checkpoint_path)
    start_idx = len(results)

    if start_idx >= len(indices):
        print(f"  {shard_tag}{split_name}: all {len(indices)} examples already done.")
    else:
        for batch_start in tqdm(range(start_idx, len(indices), batch_size),
                                desc=f"ToolAlpaca {split_name} {shard_tag}",
                                initial=start_idx // batch_size,
                                total=(len(indices) + batch_size - 1) // batch_size):
            batch_end = min(batch_start + batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            batch = [examples[j] for j in batch_indices]

            messages_list = [build_prompt(ex["nl_doc"], ex["instruction"]) for ex in batch]
            responses, token_counts = generate_responses_batch(model, tokenizer, messages_list)

            for j, (ex, resp, n_tokens) in enumerate(zip(batch, responses, token_counts)):
                entry = evaluate_single(ex, resp)
                entry["index"] = shard_offset + batch_start + j
                entry["gen_tokens"] = n_tokens
                results.append(entry)

            save_checkpoint(checkpoint_path, results)

    func_correct = sum(1 for r in results if r["func_match"])
    pass_count = sum(1 for r in results if r["overall_pass"])
    total = len(results)
    func_acc = func_correct / total if total > 0 else 0.0
    pass_rate = pass_count / total if total > 0 else 0.0

    token_counts = [r["gen_tokens"] for r in results if "gen_tokens" in r]
    max_tokens_used = max(token_counts) if token_counts else 0
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    hit_max = sum(1 for t in token_counts if t >= 1024)

    return {
        "split": split_name,
        "num_queries": total,
        "function_accuracy": func_acc,
        "function_correct": func_correct,
        "pass_rate": pass_rate,
        "pass_count": pass_count,
        "token_stats": {
            "avg_gen_tokens": avg_tokens,
            "max_gen_tokens": max_tokens_used,
            "hit_max_limit": hit_max,
            "total_with_stats": len(token_counts),
        },
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run on first 10 examples per split")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--no_resume", action="store_true", help="Start fresh, ignore checkpoint")
    parser.add_argument("--merge_only", action="store_true", help="Only merge shard results, no inference")
    add_shard_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ensure_repo()

    simulated_data = load_eval_data("eval_simulated.json")
    real_data = load_eval_data("eval_real.json")

    sim_examples = flatten_examples(simulated_data)
    real_examples = flatten_examples(real_data)
    print(f"Simulated: {len(sim_examples)} queries from {len(simulated_data)} tools")
    print(f"Real: {len(real_examples)} queries from {len(real_data)} tools")

    if args.test:
        sim_examples = sim_examples[:10]
        real_examples = real_examples[:10]
        print(f"[TEST MODE] Using first 10 examples per split")

    if not sim_examples and not real_examples:
        print("ERROR: No eval data found. Exiting.")
        return

    output = {
        "dataset": "ToolAlpaca",
        "model": "Qwen/Qwen2.5-3B-Instruct",
    }

    # Merge mode: combine shard results and produce final output
    if args.merge_only:
        if sim_examples:
            sim_results_list = merge_shard_results(RESULTS_DIR, "toolalpaca_sim_checkpoint", args.num_shards)
            print(f"Merged {len(sim_results_list)} simulated results from {args.num_shards} shards")
            total = len(sim_results_list)
            func_correct = sum(1 for r in sim_results_list if r["func_match"])
            pass_count = sum(1 for r in sim_results_list if r["overall_pass"])
            output["simulated"] = {
                "split": "simulated", "num_queries": total,
                "function_accuracy": func_correct / total if total else 0,
                "function_correct": func_correct,
                "pass_rate": pass_count / total if total else 0,
                "pass_count": pass_count,
                "results": sim_results_list,
            }
        if real_examples:
            real_results_list = merge_shard_results(RESULTS_DIR, "toolalpaca_real_checkpoint", args.num_shards)
            print(f"Merged {len(real_results_list)} real results from {args.num_shards} shards")
            total = len(real_results_list)
            func_correct = sum(1 for r in real_results_list if r["func_match"])
            pass_count = sum(1 for r in real_results_list if r["overall_pass"])
            output["real"] = {
                "split": "real", "num_queries": total,
                "function_accuracy": func_correct / total if total else 0,
                "function_correct": func_correct,
                "pass_rate": pass_count / total if total else 0,
                "pass_count": pass_count,
                "results": real_results_list,
            }
    else:
        model, tokenizer = load_model()

        if sim_examples:
            cp = os.path.join(RESULTS_DIR, f"toolalpaca_sim_checkpoint_shard{args.shard_id}.json")
            sim_results = evaluate_split(sim_examples, model, tokenizer, "simulated",
                                         args.batch_size, cp, args.no_resume,
                                         args.num_shards, args.shard_id)
            output["simulated"] = sim_results
            print(f"\nSimulated - Func Acc: {sim_results['function_accuracy']:.4f}, "
                  f"Pass Rate: {sim_results['pass_rate']:.4f}")

        if real_examples:
            cp = os.path.join(RESULTS_DIR, f"toolalpaca_real_checkpoint_shard{args.shard_id}.json")
            real_results = evaluate_split(real_examples, model, tokenizer, "real",
                                          args.batch_size, cp, args.no_resume,
                                          args.num_shards, args.shard_id)
            output["real"] = real_results
            print(f"Real - Func Acc: {real_results['function_accuracy']:.4f}, "
                  f"Pass Rate: {real_results['pass_rate']:.4f}")

    output_path = os.path.join(RESULTS_DIR, "toolalpaca_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
