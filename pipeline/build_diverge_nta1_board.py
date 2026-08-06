#!/usr/bin/env python3
"""Build the frozen corpus-derived DIVERGE-NTA1 arithmetic board."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "shohin-diverge-nta1-board-v1"
THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
EQUATION = re.compile(
    r"(?P<lhs>-?\d+)\s*(?P<operator>[+\-*])\s*"
    r"(?P<argument>-?\d+)\s*=\s*(?P<rhs>-?\d+)"
)
CRT_SAFE_ABS = 3_000_000


class NTA1BuildError(RuntimeError):
    """The natural arithmetic board cannot satisfy its frozen contract."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _apply(left: int, operator: str, right: int) -> int:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    raise NTA1BuildError("unsupported natural arithmetic operator")


def _render(template: str, left: int, argument: int, rhs: int) -> str:
    match = EQUATION.fullmatch(template)
    if match is None:
        raise NTA1BuildError("natural equation template differs")
    output = template
    for group, value in reversed(
        (("lhs", left), ("argument", argument), ("rhs", rhs))
    ):
        start, end = match.span(group)
        output = output[:start] + str(value) + output[end:]
    return output


def derive_row(source: dict[str, Any], source_sha256: str) -> dict[str, Any] | None:
    if (
        source.get("source") != "reasoning_gym_trace"
        or source.get("training_group") != "procedural"
    ):
        return None
    response = str(source.get("response") or "")
    think = THINK.search(response)
    if think is None:
        return None
    matches = list(EQUATION.finditer(think.group(1)))
    equations = [
        (
            int(match.group("lhs")),
            match.group("operator"),
            int(match.group("argument")),
            int(match.group("rhs")),
            match.group(0),
        )
        for match in matches
    ]
    if len(equations) < 2 or len(equations) > 5:
        return None
    if any(argument < 0 for _, _, argument, _, _ in equations):
        return None
    if any(_apply(left, operator, argument) != rhs for left, operator, argument, rhs, _ in equations):
        return None
    if any(equations[index][0] != equations[index - 1][3] for index in range(1, len(equations))):
        return None
    try:
        answer = int(str(source["answer"]))
    except (KeyError, ValueError):
        return None
    if equations[-1][3] != answer:
        return None
    if max(abs(value) for row in equations for value in (row[0], row[2], row[3])) >= CRT_SAFE_ABS:
        return None
    identity = hashlib.sha256(
        (source_sha256 + "\0" + str(source["question"]) + "\0" + response).encode()
    ).hexdigest()
    depth = len(equations)
    error_index = 1 + int(identity[:8], 16) % depth
    magnitude = 1 + int(identity[8:10], 16) % 3
    delta = magnitude if int(identity[10:12], 16) % 2 else -magnitude
    wrong_steps: list[str] = []
    wrong_state = equations[0][0]
    for index, (_, operator, argument, correct_rhs, template) in enumerate(equations):
        computed = _apply(wrong_state, operator, argument)
        if index + 1 == error_index:
            computed = correct_rhs + delta
        wrong_steps.append(_render(template, wrong_state, argument, computed))
        wrong_state = computed
    if wrong_state == answer or max(
        abs(int(value))
        for step in wrong_steps
        for value in re.findall(r"-?\d+", step)
    ) >= CRT_SAFE_ABS:
        return None
    program = [
        [
            {"+": "add", "-": "subtract", "*": "multiply"}[operator],
            argument,
        ]
        for _, operator, argument, _, _ in equations
    ]
    return {
        "schema": SCHEMA,
        "split": "evaluation",
        "identity_sha256": identity,
        "source_corpus_sha256": source_sha256,
        "source": "reasoning_gym_trace",
        "question": str(source["question"]),
        "family": "scalar",
        "depth": depth,
        "error_index": error_index,
        "initial_state": equations[0][0],
        "program": program,
        "correct_steps": [row[4] for row in equations],
        "wrong_steps": wrong_steps,
        "answer": str(answer),
        "wrong_answer": str(wrong_state),
        "heldout": {
            "source": "independent_answer_verified_corpus",
            "renderer": "natural_equation_without_step_prefix",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=279)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing existing NTA1 board or report")
    if sha256_path(args.input) != args.input_sha256:
        raise SystemExit("NTA1 source corpus hash differs")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            row = derive_row(json.loads(line), args.input_sha256)
            if row is None:
                continue
            identity = str(row["identity_sha256"])
            if identity in identities:
                raise NTA1BuildError("duplicate natural trace identity")
            identities.add(identity)
            rows.append(row)
    rows.sort(key=lambda row: row["identity_sha256"])
    if len(rows) != args.expected_rows:
        raise NTA1BuildError(
            f"natural board row count differs: {len(rows)} != {args.expected_rows}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, args.output)
    operation_counts: Counter[str] = Counter()
    for row in rows:
        operation_counts[row["program"][int(row["error_index"]) - 1][0]] += 1
    report = {
        "schema": "shohin-diverge-nta1-board-report-v1",
        "input": str(args.input),
        "input_sha256": args.input_sha256,
        "output": str(args.output),
        "output_sha256": sha256_path(args.output),
        "rows": len(rows),
        "transitions": sum(int(row["depth"]) for row in rows),
        "depth_counts": dict(sorted(Counter(str(row["depth"]) for row in rows).items())),
        "error_operation_counts": dict(sorted(operation_counts.items())),
        "zero_answer_collisions": all(row["answer"] != row["wrong_answer"] for row in rows),
    }
    report_tmp = args.report.with_suffix(args.report.suffix + ".tmp")
    report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(report_tmp, args.report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
