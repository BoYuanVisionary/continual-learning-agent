#!/usr/bin/env python3
"""Analyze distributional distance between SFT data sources and ORZ evaluation data.

Computes multiple distance metrics between training data (NuminaMath, NuminaMath-Hard,
NuminaMath-Comp, OpenR1) and the ORZ evaluation problems. This quantifies the
distribution mismatch that drives the inverse SFT scaling curve.

Metrics computed:
1. Problem text statistics (length, vocabulary size)
2. Character/word n-gram overlap between problem sets
3. Solution length distributions + KL divergence
4. Topic/category analysis (NuminaMath source breakdown)
5. Answer format analysis (numeric vs symbolic vs expression)
6. TF-IDF cosine similarity between problem corpora

Usage:
    python analyze_distributions.py

Output:
    results/distribution_analysis.json
"""

import json
import os
import re
import math
import random
from collections import Counter, defaultdict
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def load_orz_problems(max_n=None):
    """Load ORZ problems (questions only)."""
    path = os.path.join(SCRIPT_DIR, "data", "orz", "train.json")
    with open(path) as f:
        data = json.load(f)
    if max_n:
        data = data[:max_n]
    problems = [d["0"]["value"] for d in data]
    answers = [d["1"]["ground_truth"]["value"] for d in data]
    return problems, answers


def load_numinamath_problems(max_n=None, filter_sources=None, hard_mode=False):
    """Load NuminaMath problems."""
    path = os.path.join(SCRIPT_DIR, "data", "numinamath", "numinamath_cot.json")
    print(f"Loading NuminaMath from {path}...")
    with open(path) as f:
        data = json.load(f)

    if filter_sources:
        data = [d for d in data if d.get("source", "") in filter_sources]

    # Filter to boxed answers
    data = [d for d in data if "\\boxed{" in d.get("solution", "")]

    if hard_mode:
        comp_sources = {"olympiads", "amc_aime", "aops_forum", "math"}
        data = [d for d in data if d.get("source", "") in comp_sources]
        data.sort(key=lambda x: len(x["solution"]), reverse=True)

    if max_n:
        rng = random.Random(42)
        rng.shuffle(data)
        data = data[:max_n]

    problems = [d["problem"] for d in data]
    solutions = [d["solution"] for d in data]
    sources = [d.get("source", "unknown") for d in data]

    # Extract answers from \boxed{} in solutions
    answers = []
    for sol in solutions:
        match = re.search(r"\\boxed\{([^}]+)\}", sol)
        answers.append(match.group(1) if match else "")

    return problems, solutions, answers, sources


def load_openr1_problems(max_n=None):
    """Load OpenR1 problems."""
    path = os.path.join(SCRIPT_DIR, "data", "openr1", "openr1_math.json")
    print(f"Loading OpenR1 from {path}...")
    with open(path) as f:
        data = json.load(f)

    problems = []
    solutions = []
    for d in data:
        # OpenR1 has "problem" and "solution" fields directly
        if "problem" in d and "solution" in d:
            if "\\boxed{" in d["solution"]:
                problems.append(d["problem"])
                solutions.append(d["solution"])
        elif "messages" in d:
            msgs = d["messages"]
            user_msg = ""
            asst_msg = ""
            for m in msgs:
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
                elif m.get("role") == "assistant":
                    asst_msg = m.get("content", "")
            if user_msg and asst_msg and "\\boxed{" in asst_msg:
                problems.append(user_msg)
                solutions.append(asst_msg)

    print(f"  OpenR1 with boxed answers: {len(problems)}")
    if max_n and len(problems) > max_n:
        rng = random.Random(42)
        indices = list(range(len(problems)))
        rng.shuffle(indices)
        indices = indices[:max_n]
        problems = [problems[i] for i in indices]
        solutions = [solutions[i] for i in indices]

    return problems, solutions


def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    # Remove LaTeX commands but keep content
    text = text.lower()
    # Split on whitespace and punctuation
    tokens = re.findall(r'[a-z]+|[0-9]+|\\[a-z]+|[^\s]', text)
    return tokens


def compute_ngrams(tokens, n):
    """Compute n-gram counter from token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def jaccard_similarity(set1, set2):
    """Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def kl_divergence(p, q, epsilon=1e-10):
    """KL divergence D(P || Q) with smoothing."""
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p, q):
    """Jensen-Shannon divergence (symmetric)."""
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def compute_length_distribution(texts, num_bins=50, max_len=5000):
    """Compute length histogram."""
    lengths = [min(len(t), max_len) for t in texts]
    bins = np.linspace(0, max_len, num_bins + 1)
    hist, _ = np.histogram(lengths, bins=bins)
    return hist, lengths


def compute_vocabulary_stats(problems):
    """Compute vocabulary statistics for a set of problems."""
    all_tokens = []
    for p in problems:
        all_tokens.extend(tokenize(p))
    vocab = Counter(all_tokens)
    return {
        "total_tokens": len(all_tokens),
        "unique_tokens": len(vocab),
        "avg_tokens_per_problem": len(all_tokens) / len(problems) if problems else 0,
        "top_50_tokens": vocab.most_common(50),
    }


def compute_tfidf_similarity(corpus1, corpus2, max_features=5000):
    """Compute corpus-level TF-IDF cosine similarity between two problem sets."""
    # Build vocabulary from both corpora
    vocab = Counter()
    for text in corpus1 + corpus2:
        tokens = set(tokenize(text))
        for t in tokens:
            vocab[t] += 1

    # Keep top features
    top_features = [w for w, _ in vocab.most_common(max_features)]
    word2idx = {w: i for i, w in enumerate(top_features)}

    def corpus_to_tfidf(corpus):
        """Convert corpus to TF-IDF vector (corpus-level, not document-level)."""
        tf = np.zeros(len(top_features))
        df = np.zeros(len(top_features))
        for text in corpus:
            tokens = tokenize(text)
            token_counts = Counter(tokens)
            doc_tokens = set(tokens)
            for t in doc_tokens:
                if t in word2idx:
                    df[word2idx[t]] += 1
            for t, c in token_counts.items():
                if t in word2idx:
                    tf[word2idx[t]] += c

        # Average TF
        tf = tf / len(corpus) if corpus else tf
        # IDF
        n_docs = len(corpus)
        idf = np.log(n_docs / (df + 1)) + 1
        return tf * idf

    vec1 = corpus_to_tfidf(corpus1)
    vec2 = corpus_to_tfidf(corpus2)

    # Cosine similarity
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def classify_answer(answer):
    """Classify answer type: numeric, fraction, expression, symbolic, other."""
    if not answer:
        return "empty"
    # Pure integer
    if re.match(r'^-?\d+$', answer.strip()):
        return "integer"
    # Decimal
    if re.match(r'^-?\d+\.\d+$', answer.strip()):
        return "decimal"
    # Simple fraction (a/b or \frac{a}{b})
    if re.match(r'^-?\d+/\d+$', answer.strip()) or '\\frac' in answer:
        return "fraction"
    # Contains variables (x, y, n, etc.)
    if re.search(r'[a-z]', answer.lower()) and not answer.startswith('\\'):
        return "symbolic"
    # LaTeX expression
    if '\\' in answer:
        return "latex_expression"
    return "other"


def analyze_source(name, problems, solutions=None, answers=None):
    """Compute comprehensive statistics for a data source."""
    print(f"  Analyzing {name}: {len(problems)} problems")
    result = {"name": name, "num_problems": len(problems)}

    if len(problems) == 0:
        return result

    # Problem length statistics
    lengths = [len(p) for p in problems]
    result["problem_length"] = {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
    }

    # Token statistics
    token_lengths = [len(tokenize(p)) for p in problems]
    result["token_length"] = {
        "mean": float(np.mean(token_lengths)),
        "median": float(np.median(token_lengths)),
        "std": float(np.std(token_lengths)),
    }

    # Vocabulary
    all_tokens = []
    for p in problems:
        all_tokens.extend(tokenize(p))
    vocab = set(all_tokens)
    result["vocabulary"] = {
        "total_tokens": len(all_tokens),
        "unique_tokens": len(vocab),
        "type_token_ratio": len(vocab) / len(all_tokens) if all_tokens else 0,
    }

    # N-gram vocabularies (for overlap computation)
    unigrams = set()
    bigrams = set()
    trigrams = set()
    for p in problems:
        tokens = tokenize(p)
        unigrams.update(tokens)
        bigrams.update(tuple(tokens[i:i+2]) for i in range(len(tokens)-1))
        trigrams.update(tuple(tokens[i:i+3]) for i in range(len(tokens)-2))
    result["_unigrams"] = unigrams
    result["_bigrams"] = bigrams
    result["_trigrams"] = trigrams

    # Solution length (if available)
    if solutions:
        sol_lengths = [len(s) for s in solutions]
        result["solution_length"] = {
            "mean": float(np.mean(sol_lengths)),
            "median": float(np.median(sol_lengths)),
            "std": float(np.std(sol_lengths)),
            "min": int(min(sol_lengths)),
            "max": int(max(sol_lengths)),
        }

    # Answer type distribution
    if answers:
        answer_types = Counter(classify_answer(a) for a in answers)
        result["answer_types"] = dict(answer_types)

    # LaTeX command usage (indicator of mathematical content type)
    latex_cmds = Counter()
    for p in problems:
        cmds = re.findall(r'\\([a-zA-Z]+)', p)
        latex_cmds.update(cmds)
    result["top_latex_commands"] = latex_cmds.most_common(20)

    # Presence of key math topic indicators
    topic_indicators = {
        "algebra": ["equation", "solve", "variable", "polynomial", "root", "factor"],
        "geometry": ["triangle", "circle", "area", "perimeter", "angle", "polygon"],
        "number_theory": ["prime", "divisor", "modulo", "gcd", "integer", "divisible"],
        "combinatorics": ["choose", "permutation", "combination", "ways", "arrange", "count"],
        "calculus": ["integral", "derivative", "limit", "continuous", "differentiate"],
        "probability": ["probability", "expected", "random", "dice", "coin", "chance"],
    }
    topic_counts = {}
    for topic, keywords in topic_indicators.items():
        count = sum(1 for p in problems if any(k in p.lower() for k in keywords))
        topic_counts[topic] = count / len(problems) if problems else 0
    result["topic_distribution"] = topic_counts

    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print("Distribution Analysis: Training Data vs ORZ Evaluation")
    print("=" * 70)

    # Load all data sources
    print("\nLoading data sources...")

    # ORZ (evaluation target)
    orz_problems, orz_answers = load_orz_problems()
    print(f"  ORZ: {len(orz_problems)} problems")

    # NuminaMath (random sample matching experiment sizes)
    numi_problems, numi_solutions, numi_answers, numi_sources = load_numinamath_problems(max_n=10000)
    print(f"  NuminaMath: {len(numi_problems)} problems")

    # NuminaMath-Hard
    numi_hard_problems, numi_hard_solutions, numi_hard_answers, _ = load_numinamath_problems(
        max_n=10000, hard_mode=True
    )
    print(f"  NuminaMath-Hard: {len(numi_hard_problems)} problems")

    # NuminaMath-Comp
    comp_sources = {"olympiads", "amc_aime", "aops_forum", "math"}
    numi_comp_problems, numi_comp_solutions, numi_comp_answers, _ = load_numinamath_problems(
        max_n=10000, filter_sources=comp_sources
    )
    print(f"  NuminaMath-Comp: {len(numi_comp_problems)} problems")

    # OpenR1
    openr1_problems, openr1_solutions = load_openr1_problems(max_n=10000)
    print(f"  OpenR1: {len(openr1_problems)} problems")

    # NuminaMath source breakdown
    print("\n--- NuminaMath Source Distribution ---")
    source_counts = Counter(numi_sources)
    print(f"  Total sources found: {len(source_counts)}")
    for src, cnt in source_counts.most_common(15):
        print(f"    {src}: {cnt} ({100*cnt/len(numi_sources):.1f}%)")

    # Analyze each source
    print("\n--- Computing Source Statistics ---")
    sources = {}
    sources["orz"] = analyze_source("orz", orz_problems, answers=orz_answers)
    sources["numinamath"] = analyze_source("numinamath", numi_problems, numi_solutions, numi_answers)
    sources["numinamath_hard"] = analyze_source("numinamath_hard", numi_hard_problems, numi_hard_solutions, numi_hard_answers)
    sources["numinamath_comp"] = analyze_source("numinamath_comp", numi_comp_problems, numi_comp_solutions, numi_comp_answers)
    sources["openr1"] = analyze_source("openr1", openr1_problems, openr1_solutions)

    # Compute pairwise distances to ORZ
    print("\n--- Computing Distances to ORZ ---")
    orz_stats = sources["orz"]

    distances = {}
    for name, stats in sources.items():
        if name == "orz":
            continue
        if stats.get("num_problems", 0) == 0:
            print(f"\n  {name}: skipped (0 problems)")
            continue

        print(f"\n  {name} vs ORZ:")
        d = {}

        # N-gram overlap (Jaccard)
        d["unigram_jaccard"] = jaccard_similarity(orz_stats["_unigrams"], stats["_unigrams"])
        d["bigram_jaccard"] = jaccard_similarity(orz_stats["_bigrams"], stats["_bigrams"])
        d["trigram_jaccard"] = jaccard_similarity(orz_stats["_trigrams"], stats["_trigrams"])
        print(f"    Unigram Jaccard: {d['unigram_jaccard']:.4f}")
        print(f"    Bigram Jaccard:  {d['bigram_jaccard']:.4f}")
        print(f"    Trigram Jaccard: {d['trigram_jaccard']:.4f}")

        # Problem length distribution distance (JS divergence)
        orz_hist, _ = compute_length_distribution(orz_problems)
        train_hist, _ = compute_length_distribution(
            numi_problems if "numi" in name else openr1_problems
        )
        d["problem_length_js"] = js_divergence(orz_hist + 1, train_hist + 1)
        print(f"    Problem length JS div: {d['problem_length_js']:.4f}")

        # Solution length distribution distance (if available)
        if "solution_length" in stats:
            train_probs = numi_problems if "numi" in name else openr1_problems
            train_sols = numi_solutions if "numi" in name else openr1_solutions
            orz_lens = [len(p) for p in orz_problems]
            sol_lens = [len(s) for s in train_sols]
            # Compare problem lengths as a proxy
            orz_hist_p, _ = compute_length_distribution(orz_problems)
            sol_hist, _ = compute_length_distribution(train_sols, max_len=10000)
            d["solution_length_mean"] = stats["solution_length"]["mean"]

        # TF-IDF cosine similarity
        # Sample for efficiency
        sample_size = min(5000, len(orz_problems))
        orz_sample = random.Random(42).sample(orz_problems, sample_size)
        if "numi" in name:
            if name == "numinamath":
                train_sample = random.Random(42).sample(numi_problems, min(sample_size, len(numi_problems)))
            elif name == "numinamath_hard":
                train_sample = random.Random(42).sample(numi_hard_problems, min(sample_size, len(numi_hard_problems)))
            else:
                train_sample = random.Random(42).sample(numi_comp_problems, min(sample_size, len(numi_comp_problems)))
        else:
            train_sample = random.Random(42).sample(openr1_problems, min(sample_size, len(openr1_problems)))

        d["tfidf_cosine"] = compute_tfidf_similarity(orz_sample, train_sample)
        print(f"    TF-IDF cosine similarity: {d['tfidf_cosine']:.4f}")

        # Topic distribution distance
        orz_topics = np.array([orz_stats["topic_distribution"][t] for t in sorted(orz_stats["topic_distribution"])])
        train_topics = np.array([stats["topic_distribution"][t] for t in sorted(stats["topic_distribution"])])
        d["topic_js"] = js_divergence(orz_topics + 1e-6, train_topics + 1e-6)
        print(f"    Topic JS divergence: {d['topic_js']:.4f}")

        # Answer type distribution distance
        if "answer_types" in stats and "answer_types" in orz_stats:
            all_types = set(orz_stats["answer_types"].keys()) | set(stats["answer_types"].keys())
            orz_type_vec = np.array([orz_stats["answer_types"].get(t, 0) for t in sorted(all_types)], dtype=float)
            train_type_vec = np.array([stats["answer_types"].get(t, 0) for t in sorted(all_types)], dtype=float)
            if orz_type_vec.sum() > 0 and train_type_vec.sum() > 0:
                d["answer_type_js"] = js_divergence(orz_type_vec + 1, train_type_vec + 1)
                print(f"    Answer type JS div: {d['answer_type_js']:.4f}")

        # Composite distance (weighted average of all metrics)
        # Higher = more distant from ORZ
        metrics_for_composite = []
        if "unigram_jaccard" in d:
            metrics_for_composite.append(1 - d["unigram_jaccard"])
        if "bigram_jaccard" in d:
            metrics_for_composite.append(1 - d["bigram_jaccard"])
        if "tfidf_cosine" in d:
            metrics_for_composite.append(1 - d["tfidf_cosine"])
        if "topic_js" in d:
            metrics_for_composite.append(d["topic_js"])
        d["composite_distance"] = float(np.mean(metrics_for_composite)) if metrics_for_composite else 0
        print(f"    Composite distance: {d['composite_distance']:.4f}")

        distances[name] = d

    # Correlate with degradation rates from experiment log
    print("\n--- Correlating Distance with Degradation ---")
    exp_log_path = os.path.join(RESULTS_DIR, "experiment_log.json")
    if os.path.exists(exp_log_path):
        with open(exp_log_path) as f:
            experiments = json.load(f)

        # Get best ORZ accuracy per (data_source, N) at standard config (r=8, lr=5e-5, ep=1)
        source_curves = defaultdict(list)
        for exp in experiments:
            src = exp.get("data_source", "")
            n = exp.get("num_samples", exp.get("num_samples", 0))
            lr = exp.get("lr", "")
            rank = exp.get("lora_rank", 0)
            epochs = exp.get("epochs", 0)

            # Filter to standard config
            if str(lr) == "5e-5" and rank == 8 and epochs == 1:
                orz_acc = exp.get("orz_accuracy", 0)
                source_curves[src].append((n, orz_acc))

        print("\n  Degradation rates at standard config (r=8, lr=5e-5, ep=1):")
        degradation_rates = {}
        baseline_acc = 0.2891
        for src, points in sorted(source_curves.items()):
            points.sort()
            if len(points) >= 2:
                # Compute degradation: slope of accuracy vs log(N)
                ns = [p[0] for p in points]
                accs = [p[1] for p in points]
                log_ns = [math.log(n) for n in ns]
                # Simple linear regression of acc vs log(N)
                n_pts = len(log_ns)
                mean_x = sum(log_ns) / n_pts
                mean_y = sum(accs) / n_pts
                ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ns, accs))
                ss_xx = sum((x - mean_x) ** 2 for x in log_ns)
                slope = ss_xy / ss_xx if ss_xx > 0 else 0
                degradation_rates[src] = {
                    "slope_per_logN": slope,
                    "points": [(n, acc) for n, acc in points],
                    "max_acc": max(accs),
                    "min_acc": min(accs),
                    "range": max(accs) - min(accs),
                }
                print(f"    {src}: slope={slope:.4f}/logN, range={max(accs)-min(accs):.4f}")

        # Map source names to distance keys
        source_to_dist_key = {
            "numinamath": "numinamath",
            "numinamath_hard": "numinamath_hard",
            "openr1": "openr1",
        }

        correlation_data = []
        for src, rate in degradation_rates.items():
            dist_key = source_to_dist_key.get(src)
            if dist_key and dist_key in distances:
                correlation_data.append({
                    "source": src,
                    "composite_distance": distances[dist_key]["composite_distance"],
                    "degradation_slope": rate["slope_per_logN"],
                    "degradation_range": rate["range"],
                })

        if correlation_data:
            print("\n  Distance vs Degradation correlation:")
            for c in correlation_data:
                print(f"    {c['source']}: dist={c['composite_distance']:.4f}, "
                      f"slope={c['degradation_slope']:.4f}, range={c['degradation_range']:.4f}")

    # Clean up internal fields before saving
    output = {
        "description": "Distribution analysis of SFT data sources vs ORZ evaluation",
        "sources": {},
        "distances_to_orz": distances,
        "degradation_correlation": correlation_data if 'correlation_data' in dir() else [],
        "numinamath_source_breakdown": dict(source_counts) if 'source_counts' in dir() else {},
    }

    for name, stats in sources.items():
        clean_stats = {k: v for k, v in stats.items() if not k.startswith("_")}
        # Convert tuples in top_latex_commands to lists for JSON
        if "top_latex_commands" in clean_stats:
            clean_stats["top_latex_commands"] = [[cmd, cnt] for cmd, cnt in clean_stats["top_latex_commands"]]
        output["sources"][name] = clean_stats

    out_path = os.path.join(RESULTS_DIR, "distribution_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Source':<20} {'Prob Len':>10} {'Sol Len':>10} {'Vocab':>8} "
          f"{'Uni Jacc':>10} {'TF-IDF':>8} {'Comp Dist':>10}")
    print("-" * 90)
    for name in ["numinamath", "numinamath_hard", "numinamath_comp", "openr1"]:
        s = output["sources"][name]
        d = distances.get(name, {})
        sol_len = s.get("solution_length", {}).get("mean", 0)
        print(f"  {name:<18} {s['problem_length']['mean']:10.1f} {sol_len:10.1f} "
              f"{s['vocabulary']['unique_tokens']:8d} "
              f"{d.get('unigram_jaccard', 0):10.4f} {d.get('tfidf_cosine', 0):8.4f} "
              f"{d.get('composite_distance', 0):10.4f}")
    print(f"\n  ORZ reference:")
    s = output["sources"]["orz"]
    print(f"  {'orz':<18} {s['problem_length']['mean']:10.1f} {'N/A':>10} "
          f"{s['vocabulary']['unique_tokens']:8d}")


if __name__ == "__main__":
    main()
