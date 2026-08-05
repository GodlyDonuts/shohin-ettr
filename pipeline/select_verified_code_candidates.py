#!/usr/bin/env python3
"""Select sampled code using only tests visible in the task prompt."""

from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_product_reasoning_eval import (
    _bounded_program_result,
    _humaneval_candidate_source,
    _row_identity,
)


SCHEMA = "shohin-visible-code-candidate-selection-v1"


class CodeCandidateSelectionError(RuntimeError):
    """Candidate rows cannot support visible-test selection."""


def visible_humaneval_result(
    row: dict[str, Any], completion: str, timeout_seconds: float
) -> dict[str, Any]:
    """Run only doctest examples embedded in the public function prompt."""

    prompt = str(row["prompt"])
    try:
        module = ast.parse(prompt)
        function = next(
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        public_examples = ast.get_docstring(function, clean=False) or ""
    except (SyntaxError, StopIteration, TypeError):
        public_examples = ""
    source = _humaneval_candidate_source(row, completion)
    attempted = public_examples.count(">>>")
    if attempted:
        source += (
            "\n\nimport doctest as _shohin_doctest\n"
            f"_shohin_prompt = {public_examples!r}\n"
            "_shohin_test = _shohin_doctest.DocTestParser().get_doctest(\n"
            "    _shohin_prompt, globals(), 'visible_prompt_examples', None, 0\n"
            ")\n"
            "_shohin_result = _shohin_doctest.DocTestRunner().run(_shohin_test)\n"
            "assert _shohin_result.attempted > 0\n"
            "assert _shohin_result.failed == 0\n"
        )
    execution = _bounded_program_result(source, timeout_seconds)
    return {
        "attempted_examples": attempted,
        "syntax_clean": bool(execution["passed"]) if not attempted else None,
        "visible_tests_passed": bool(execution["passed"]) if attempted else False,
        "execution": execution,
    }


def _ordered_groups(
    candidates: list[dict[str, Any]],
) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for candidate in candidates:
        identity = str(candidate.get("identity_sha256") or "")
        if not identity:
            raise CodeCandidateSelectionError("candidate identity is missing")
        grouped.setdefault(identity, []).append(candidate)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["sample_index"]))
        indices = [int(row["sample_index"]) for row in group]
        if indices != list(range(len(group))):
            raise CodeCandidateSelectionError("candidate sample indices differ")
        if any(str(row["task"]) != str(group[0]["task"]) for row in group):
            raise CodeCandidateSelectionError("candidate tasks differ")
    return grouped


def _shortest(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        candidates,
        key=lambda row: (
            int(row.get("generated_tokens") or 0),
            len(str(row.get("completion") or "")),
            int(row["sample_index"]),
        ),
    )


def _visible_choice(
    group: list[dict[str, Any]],
    visible: list[dict[str, Any]],
    policy: str,
) -> dict[str, Any]:
    if policy == "anchor-first" and group[0] in visible:
        return group[0]
    return _shortest(visible)


def select(
    candidates: list[dict[str, Any]],
    bank_rows: list[dict[str, Any]],
    timeout_seconds: float,
    policy: str = "shortest-visible",
) -> dict[str, Any]:
    if policy not in {"shortest-visible", "anchor-first"}:
        raise CodeCandidateSelectionError(f"unsupported selection policy: {policy}")
    bank = {
        str(row.get("identity_sha256") or _row_identity(str(row["task"]), row)): row
        for row in bank_rows
    }
    grouped = _ordered_groups(candidates)
    if set(grouped) != set(bank):
        raise CodeCandidateSelectionError("candidate and bank identities differ")
    first_correct = oracle_correct = selected_correct = 0
    visible_pass_groups = syntax_fallback_groups = 0
    results: list[dict[str, Any]] = []
    for identity, group in grouped.items():
        row = bank[identity]
        task = str(group[0]["task"])
        if task == "mbpp":
            visible = [
                candidate
                for candidate in group
                if bool((candidate.get("execution") or {}).get("passed"))
            ]
            selected = _visible_choice(group, visible, policy) if visible else group[0]
            visible_pass_groups += int(bool(visible))
            selection = "visible_task_tests" if visible else "first_fallback"
            diagnostics = None
        elif task == "humaneval":
            diagnostics = [
                visible_humaneval_result(
                    row, str(candidate["completion"]), timeout_seconds
                )
                for candidate in group
            ]
            visible = [
                candidate
                for candidate, diagnostic in zip(group, diagnostics, strict=True)
                if diagnostic["visible_tests_passed"]
            ]
            syntax_clean = [
                candidate
                for candidate, diagnostic in zip(group, diagnostics, strict=True)
                if diagnostic["attempted_examples"] == 0 and diagnostic["syntax_clean"]
            ]
            if visible:
                selected = _visible_choice(group, visible, policy)
                visible_pass_groups += 1
                selection = "visible_docstring_tests"
            elif syntax_clean:
                selected = _visible_choice(group, syntax_clean, policy)
                syntax_fallback_groups += 1
                selection = "syntax_clean_fallback"
            else:
                selected = group[0]
                selection = "first_fallback"
        else:
            raise CodeCandidateSelectionError(f"unsupported code task: {task}")
        first = bool(group[0]["correct"])
        oracle = any(bool(candidate["correct"]) for candidate in group)
        picked = bool(selected["correct"])
        first_correct += int(first)
        oracle_correct += int(oracle)
        selected_correct += int(picked)
        results.append(
            {
                "identity_sha256": identity,
                "task": task,
                "first_correct": first,
                "oracle_correct": oracle,
                "selected_correct": picked,
                "selected_sample_index": int(selected["sample_index"]),
                "selection": selection,
                "visible_diagnostics": diagnostics,
            }
        )
    total = len(results)
    if not total:
        raise CodeCandidateSelectionError("candidate source is empty")
    return {
        "schema": SCHEMA,
        "total": total,
        "samples": len(candidates) // total,
        "first_correct": first_correct,
        "first_accuracy": first_correct / total,
        "oracle_correct": oracle_correct,
        "oracle_accuracy": oracle_correct / total,
        "selected_correct": selected_correct,
        "selected_accuracy": selected_correct / total,
        "visible_pass_groups": visible_pass_groups,
        "syntax_fallback_groups": syntax_fallback_groups,
        "selector_reads_hidden_tests": False,
        "selection_policy": policy,
        "results": results,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise CodeCandidateSelectionError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument(
        "--policy",
        choices=("shortest-visible", "anchor-first"),
        default="shortest-visible",
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    candidate_bytes = args.candidates.read_bytes()
    bank_bytes = args.bank.read_bytes()
    candidates = [json.loads(line) for line in candidate_bytes.splitlines() if line]
    rows = [json.loads(line) for line in bank_bytes.splitlines() if line]
    report = select(candidates, rows, args.timeout_seconds, args.policy)
    report["candidates"] = str(args.candidates.resolve())
    report["candidates_sha256"] = hashlib.sha256(candidate_bytes).hexdigest()
    report["bank"] = str(args.bank.resolve())
    report["bank_sha256"] = hashlib.sha256(bank_bytes).hexdigest()
    _atomic_json(args.output, report)
    print(
        f"[code-selection] selected={report['selected_correct']}/{report['total']} "
        f"first={report['first_correct']} oracle={report['oracle_correct']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
