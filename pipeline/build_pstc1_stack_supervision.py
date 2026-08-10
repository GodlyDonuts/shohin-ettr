#!/usr/bin/env python3
"""Build exact postorder pushdown supervision from admitted arithmetic ledgers."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA = "shohin-pstc1-stack-supervision-v1"
REPORT_SCHEMA = "shohin-pstc1-stack-supervision-report-v1"
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])(?:\d+\.\d+|\d+|\.\d+)")
OP_NAMES = {ast.Add: "ADD", ast.Sub: "SUB", ast.Mult: "MUL", ast.Div: "DIV"}


class PSTC1SupervisionError(ValueError):
    """Raised when a source expression cannot be admitted exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise PSTC1SupervisionError(f"invalid JSON at line {line_number}") from error
            if not isinstance(row, dict):
                raise PSTC1SupervisionError(f"non-object row at line {line_number}")
            yield row


def extract_expression(question: str, family: str) -> tuple[str, int]:
    if family in {"chain_sum", "decimal_chain_sum"}:
        marker = "arithmetic problem:"
        start = question.find(marker)
        if start < 0:
            raise PSTC1SupervisionError("chain expression marker differs")
        start += len(marker)
        end = question.rfind("=")
    elif family == "basic_arithmetic":
        marker = "Calculate "
        start = question.find(marker)
        if start < 0:
            raise PSTC1SupervisionError("basic expression marker differs")
        start += len(marker)
        end = question.rfind(".")
    elif family == "products":
        marker = "multiplication:"
        start = question.find(marker)
        if start < 0:
            raise PSTC1SupervisionError("product expression marker differs")
        start += len(marker)
        end = question.find(". Give", start)
    elif family == "decimal_arithmetic":
        line_start = question.rfind("\n") + 1
        start = line_start
        end = question.find("= ?", start)
    else:
        raise PSTC1SupervisionError("unsupported family")
    if end < start:
        raise PSTC1SupervisionError("expression boundary differs")
    raw = question[start:end]
    left_trim = len(raw) - len(raw.lstrip())
    expression = raw.strip()
    start += left_trim
    if not expression or question[start : start + len(expression)] != expression:
        raise PSTC1SupervisionError("expression extraction differs")
    return expression, start


def _fraction(value: dict[str, Any]) -> Fraction:
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise PSTC1SupervisionError("ledger fraction differs")
    return Fraction(numerator, denominator)


def compile_actions(question: str, family: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    expression, expression_start = extract_expression(question, family)
    if "^" in expression or "**" in expression:
        raise PSTC1SupervisionError("power expression is outside PSTC1")
    try:
        tree = ast.parse(expression, mode="eval").body
    except SyntaxError as error:
        raise PSTC1SupervisionError("source expression syntax differs") from error
    spans = [
        {
            "start": match.start(),
            "end": match.end(),
            "surface": match.group(0),
            "magnitude": str(Fraction(match.group(0))),
        }
        for match in NUMBER_RE.finditer(question)
    ]
    span_lookup = {
        (span["start"], span["end"]): index for index, span in enumerate(spans)
    }
    actions: list[dict[str, Any]] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            start = expression_start + int(node.col_offset)
            end = expression_start + int(node.end_col_offset)
            source_index = span_lookup.get((start, end))
            if source_index is None:
                raise PSTC1SupervisionError("number AST span has no source owner")
            actions.append({"action": "PUSH", "source_index": source_index})
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            visit(node.operand)
            if isinstance(node.op, ast.USub):
                actions.append({"action": "NEGATE"})
            return
        if isinstance(node, ast.BinOp) and type(node.op) in OP_NAMES:
            visit(node.left)
            visit(node.right)
            actions.append({"action": f"APPLY_{OP_NAMES[type(node.op)]}"})
            return
        raise PSTC1SupervisionError("expression AST contains an unsupported node")

    visit(tree)
    actions.append({"action": "STOP"})
    return actions, spans, expression_start


def execute_actions(actions: list[dict[str, Any]], spans: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Fraction, int]:
    stack: list[Fraction] = []
    records = []
    maximum_stack = 0
    stopped = False
    for action in actions:
        name = action["action"]
        if name == "PUSH":
            index = action.get("source_index")
            if type(index) is not int or not 0 <= index < len(spans):
                raise PSTC1SupervisionError("PUSH pointer differs")
            stack.append(Fraction(spans[index]["magnitude"]))
        elif name == "NEGATE":
            if not stack:
                raise PSTC1SupervisionError("NEGATE underflow")
            stack[-1] = -stack[-1]
        elif name.startswith("APPLY_"):
            if len(stack) < 2:
                raise PSTC1SupervisionError("APPLY underflow")
            right = stack.pop()
            left = stack.pop()
            operation = name.removeprefix("APPLY_")
            if operation == "ADD":
                result = left + right
            elif operation == "SUB":
                result = left - right
            elif operation == "MUL":
                result = left * right
            elif operation == "DIV" and right:
                result = left / right
            else:
                raise PSTC1SupervisionError("APPLY operation differs")
            records.append({"operation": operation, "result": result})
            stack.append(result)
        elif name == "STOP":
            if stopped or len(stack) != 1:
                raise PSTC1SupervisionError("STOP state differs")
            stopped = True
        else:
            raise PSTC1SupervisionError("action differs")
        maximum_stack = max(maximum_stack, len(stack))
    if not stopped or len(stack) != 1:
        raise PSTC1SupervisionError("action sequence did not commit")
    return records, stack[0], maximum_stack


def compile_row(row: dict[str, Any]) -> dict[str, Any]:
    identity = row.get("identity_sha256")
    family = row.get("family")
    question = row.get("question")
    if not isinstance(identity, str) or not isinstance(family, str) or not isinstance(question, str):
        raise PSTC1SupervisionError("row identity differs")
    actions, spans, expression_start = compile_actions(question, family)
    records, terminal, maximum_stack = execute_actions(actions, spans)
    gold_records = row.get("records")
    if not isinstance(gold_records, list) or len(records) != len(gold_records):
        raise PSTC1SupervisionError("binary record count differs")
    for predicted, gold in zip(records, gold_records, strict=True):
        if predicted["operation"] != gold.get("operation") or predicted["result"] != _fraction(gold.get("result", {})):
            raise PSTC1SupervisionError("binary operation/result parity differs")
    if terminal != _fraction(row.get("terminal_value", {})):
        raise PSTC1SupervisionError("terminal parity differs")
    return {
        "schema": SCHEMA,
        "identity_sha256": identity,
        "source_question_sha256": row.get("source_question_sha256"),
        "split": row.get("split"),
        "family": family,
        "question": question,
        "expression_start": expression_start,
        "number_spans": spans,
        "actions": actions,
        "action_count": len(actions),
        "maximum_stack": maximum_stack,
        "binary_record_count": len(records),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = canonical_json_bytes(row)
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise PSTC1SupervisionError("refusing existing output root")
    args.output_root.mkdir(parents=True)
    outputs = {}
    for split, path, expected, expected_rows in (
        ("train", args.train, args.expected_train_sha256, 75935),
        ("development", args.development, args.expected_development_sha256, 3917),
    ):
        if sha256_file(path) != expected:
            raise PSTC1SupervisionError(f"{split} source SHA-256 differs")
        counters: Counter[str] = Counter()
        admitted = []
        exclusions = []
        for row in _iter_jsonl(path):
            counters["source_rows"] += 1
            try:
                compiled = compile_row(row)
            except PSTC1SupervisionError as error:
                counters[f"excluded:{error}"] += 1
                exclusions.append(
                    {"identity_sha256": row.get("identity_sha256"), "reason": str(error)}
                )
                continue
            admitted.append(compiled)
            counters["admitted_rows"] += 1
            counters[f"family:{compiled['family']}"] += 1
            counters[f"actions:{compiled['action_count']}"] += 1
            counters[f"stack:{compiled['maximum_stack']}"] += 1
        if counters["source_rows"] != expected_rows:
            raise PSTC1SupervisionError(f"{split} source population differs")
        data_path = args.output_root / f"{split}.jsonl"
        exclusion_path = args.output_root / f"{split}.exclusions.jsonl"
        outputs[split] = {
            "path": str(data_path.resolve()),
            "sha256": _write_jsonl(data_path, admitted),
            "rows": len(admitted),
            "exclusion_path": str(exclusion_path.resolve()),
            "exclusion_sha256": _write_jsonl(exclusion_path, exclusions),
            "exclusions": len(exclusions),
            "counts": dict(sorted(counters.items())),
            "maximum_action_count": max((row["action_count"] for row in admitted), default=0),
            "maximum_stack": max((row["maximum_stack"] for row in admitted), default=0),
        }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "sources": {
            "train": {"path": str(args.train.resolve()), "sha256": args.expected_train_sha256},
            "development": {"path": str(args.development.resolve()), "sha256": args.expected_development_sha256},
        },
        "outputs": outputs,
        "development_admission_rate": outputs["development"]["rows"] / 3917,
        "admitted": outputs["development"]["rows"] / 3917 >= 0.99,
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-development-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    return 0 if result["admitted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
