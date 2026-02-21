"""
Math grading utility adapted from lm-evaluation-harness.
https://github.com/EleutherAI/lm-evaluation-harness/blob/760639fc54214af4a6c70bcc6d8e92801748215f/lm_eval/tasks/score/math/math_grader.py

Copyright (c) 2024, NVIDIA CORPORATION / Microsoft Corporation / OpenAI / Dan Hendrycks.
Licensed under Apache 2.0 / MIT.
"""

import contextlib
import re
import signal
from math import isclose
from typing import Union


def _fix_fracs(string):
    while "\\frac " in string:
        string = string.replace("\\frac ", "\\frac")
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
        string = new_str
    return string


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    if "_" in x:
        x = x.split("_")[0]
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    p1 = re.compile(r"([0-9]) +([0-9])")
    step = p1.sub(r"\1+\2", step)
    return step


def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _remove_right_units(expr):
    if "\\text" in expr:
        try:
            splits = re.split(r"\\text\s*{\s*", expr)
            assert len(splits) == 2 and splits[0] not in ("", "(")
            return splits[0]
        except AssertionError:
            pass

    if "\\text{" in expr:
        return re.sub(r"\\text{([^}]+)}", r"\1", expr)
    elif "\\mbox{" in expr:
        splits = expr.split("\\mbox{")
        assert len(splits) == 2
        return splits[0]
    else:
        return expr


def _process_and_or_inside_text(string):
    string = re.sub(r"\s*\\text{\s*(or|and)\s*}\s*", ",", string)
    string = re.sub(r",\s*,", ",", string)
    return string


def _remove_left_and_right(expr):
    expr = re.sub(r"\\left", "", expr)
    expr = re.sub(r"\\right", "", expr)
    return expr


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\s*\w+)", r"\\sqrt{\1}", string)
    return _string


def _fix_interval(expr):
    if "\\in " in expr:
        return expr.split("\\in ")[1].strip()
    return expr


def _inject_implicit_mixed_fraction(step: str):
    p1 = re.compile(r"(\d+) *\\frac{(\d+)}{(\d+)}")

    def replacer(match):
        whole_part = match.group(1)
        numerator = match.group(2)
        denominator = match.group(3)
        if whole_part:
            return f"{whole_part} + {numerator}/{denominator}"
        else:
            return f"{numerator}/{denominator}"

    step = p1.sub(replacer, step)
    return step


def normalize_answer_string(expr: str) -> str:
    if expr is None:
        return None

    expr = _remove_left_and_right(expr)
    expr = _process_and_or_inside_text(expr)
    expr = _remove_right_units(expr)
    expr = _fix_interval(expr)
    for surround_str in [
        "\\\\text",
        "\\\\mathrm",
        "\\\\mathcal",
        "\\\\textbf",
        "\\\\textit",
    ]:
        expr = expr.replace(surround_str, "")
        pattern = f"^{surround_str}" + r"\{(?P<text>.+?)\}$"
        m = re.search(pattern, expr)
        if m is not None:
            expr = m.group("text")

    expr = expr.replace(r"\!", "")
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace("^{\\circ}", "")

    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second",
        "minute", "hour", "week", "month", "year", "foot", "feet",
        "inch", "yard", "p.m.", "PM",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)

    if "day" in expr:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_expressed = any(day in expr for day in days)
        if not weekday_expressed:
            expr = re.sub(r"day(s)?", "", expr)

    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = _fix_sqrt(expr)
    expr = _fix_fracs(expr)

    expr = re.sub(r"- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = _inject_implicit_mixed_fraction(expr)
    expr = expr.replace(" ", "")

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def is_digit(s):
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num
        num = float(str(s).replace(",", ""))
        return True, num
    except ValueError:
        return False, None


def normalize(answer) -> str:
    if isinstance(answer, str) and bool(re.match(r"\$\d+(\.\d+)?", answer)):
        return answer[1:]
    if isinstance(answer, str) and (
        bool(re.match(r"^\d+(\.\d+)?%$", answer))
        or bool(re.match(r"^\d+(\.\d+)?\%$", answer))
    ):
        return answer.replace("\\%", "").replace("%", "")
    return answer


def format_intervals(prediction):
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }
    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner_content = match.group(1)
            if key == "Interval(":
                return f"[{inner_content}]"
            elif key == "Interval.Ropen(":
                return f"[{inner_content})"
            elif key == "Interval.Lopen(":
                return f"({inner_content}]"
            elif key == "Interval.open(":
                return f"({inner_content})"
    return prediction


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def symbolic_equal(a, b, tolerance, timeout=10.0):
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr

    def _parse(s):
        for f in [parse_expr, parse_latex]:
            try:
                with time_limit(timeout):
                    return f(s)
            except Exception:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        with time_limit(timeout):
            if sympy.simplify(a - b) == 0:
                return True
    except Exception:
        pass

    try:
        with time_limit(timeout):
            if isclose(sympy.N(a), sympy.N(b), rel_tol=tolerance):
                return True
    except Exception:
        pass
    return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    from sympy.parsing.sympy_parser import parse_expr

    prediction = normalize(prediction)
    reference = normalize(reference)

    prediction = normalize_answer_string(prediction)
    reference = normalize_answer_string(reference)

    if isinstance(prediction, str) and len(prediction) > 1000:
        prediction = prediction[:1000]

    # 0. string comparison
    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True

    try:  # 1. numerical equal
        if is_digit(prediction)[0] and is_digit(reference)[0]:
            prediction = is_digit(prediction)[1]
            reference = is_digit(reference)[1]
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if isclose(item, prediction, rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    prediction = format_intervals(prediction)

    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")
    ) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()").strip()
        ref_str = ref_str.strip("[]()").strip()
        for s in ["{", "}", "(", ")"]:
            ref_str = ref_str.replace(s, "")
            pred_str = pred_str.replace(s, "")
        if pred_str == ref_str:
            return True

    if (
        prediction and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts)
            ):
                return True

    if "," in prediction and "," in reference:
        pred_parts = [item.strip() for item in prediction.split(",")]
        ref_parts = [item.strip() for item in reference.split(",")]
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance)
                for i in range(len(pred_parts))
            ):
                return True
        else:
            return False

    if prediction.startswith("Point") and reference[0] == "(" and reference[-1] == ")":
        pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts)
            ):
                return True

    if reference.startswith("\\begin{pmatrix}") and prediction.startswith("Matrix"):
        try:
            pred_matrix = parse_expr(prediction)
            ref_matrix_items = reference.split()[1:-1:2]
            if len(pred_matrix) == len(ref_matrix_items):
                if all(
                    math_equal(ref, pred, include_percentage, tolerance)
                    for ref, pred in zip(ref_matrix_items, pred_matrix)
                ):
                    return True
        except Exception:
            pass

    return symbolic_equal(prediction, reference, tolerance, timeout)


if __name__ == "__main__":
    import json
    import os
    import sys

    results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "orz_results.json")
    with open(results_file) as f:
        data = json.load(f)

    # If a specific index is passed as argument, test just that one
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        for r in data["results"]:
            if r["index"] == idx:
                pred = r["pred_answer"]
                gold = r["gold_answer"]
                result = math_equal(pred, gold) if pred is not None else False
                print(f"Index {idx}:")
                print(f"  gold_answer: {gold}")
                print(f"  pred_answer: {pred}")
                print(f"  original correct: {r['correct']}")
                print(f"  math_equal result: {result}")
                break
        sys.exit(0)

    # Otherwise, scan all entries for format mismatches
    print("Scanning for format-mismatch failures (correct=false but math_equal=true)...\n")
    mismatches = []
    for r in data["results"]:
        if r["correct"] is False and r["pred_answer"] is not None:
            try:
                if math_equal(r["pred_answer"], r["gold_answer"]):
                    mismatches.append(r)
            except Exception as e:
                pass

    if mismatches:
        print(f"Found {len(mismatches)} format-mismatch failures:\n")
        for r in mismatches:
            print(f"  Index {r['index']}:")
            print(f"    gold_answer: {r['gold_answer']}")
            print(f"    pred_answer: {r['pred_answer']}")
            print()
    else:
        print("No format-mismatch failures found.")
