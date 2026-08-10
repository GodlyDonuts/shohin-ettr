#!/usr/bin/env python3
"""Build exact, addressable operation ledgers from verified RG trajectories."""

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


SCHEMA = "shohin-structured-reasoning-ledger-v1"
REPORT_SCHEMA = "shohin-structured-reasoning-ledger-report-v1"
SUPPORTED_FAMILIES = frozenset(
    {
        "basic_arithmetic",
        "chain_sum",
        "decimal_arithmetic",
        "decimal_chain_sum",
        "products",
    }
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
FINAL_RE = re.compile(r"The answer is\s*(.*?)\s*\.?\s*$", re.DOTALL)


class StructuredLedgerError(ValueError):
    """Raised when source custody or exact ledger mechanics differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _number(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _number(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    raise StructuredLedgerError("operand is not one exact numeric literal")


def _atomic_expression(text: str) -> tuple[str, Fraction, Fraction, Fraction]:
    normalized = text.strip().replace(",", "").replace("^", "**")
    try:
        node = ast.parse(normalized, mode="eval").body
    except SyntaxError as error:
        raise StructuredLedgerError("expression is not valid arithmetic") from error
    if not isinstance(node, ast.BinOp):
        raise StructuredLedgerError("expression is not one binary operation")
    left, right = _number(node.left), _number(node.right)
    if isinstance(node.op, ast.Add):
        op, result = "ADD", left + right
    elif isinstance(node.op, ast.Sub):
        op, result = "SUB", left - right
    elif isinstance(node.op, ast.Mult):
        op, result = "MUL", left * right
    elif isinstance(node.op, ast.Div):
        if right == 0:
            raise StructuredLedgerError("division by zero")
        op, result = "DIV", left / right
    elif isinstance(node.op, ast.Pow):
        if right.denominator != 1 or abs(right.numerator) > 20:
            raise StructuredLedgerError("unsupported exponent")
        op, result = "POW", left ** right.numerator
    else:
        raise StructuredLedgerError("unsupported arithmetic operation")
    return op, left, right, result


def _fraction(text: str) -> Fraction:
    normalized = text.strip().replace(",", "")
    if normalized.startswith("+"):
        normalized = normalized[1:]
    try:
        return Fraction(normalized)
    except (ValueError, ZeroDivisionError) as error:
        raise StructuredLedgerError("result is not an exact numeric literal") from error


def _value(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _answer_fraction(answer: object) -> Fraction:
    if not isinstance(answer, str):
        raise StructuredLedgerError("answer is not a string")
    return _fraction(answer.rstrip("."))


def compile_response(response: str, answer: object) -> list[dict[str, Any]]:
    match = THINK_RE.search(response)
    final_match = FINAL_RE.search(response)
    if match is None or final_match is None:
        raise StructuredLedgerError("response lacks exact think/final boundaries")
    if _answer_fraction(final_match.group(1)) != _answer_fraction(answer):
        raise StructuredLedgerError("surface final answer differs from answer field")

    think = match.group(1).strip()
    clauses = [clause.strip() for clause in think.split(";") if clause.strip()]
    if not clauses:
        raise StructuredLedgerError("empty reasoning trace")

    records: list[dict[str, Any]] = []
    latest_value_owner: dict[Fraction, int] = {}
    search_cursor = 0
    for index, clause in enumerate(clauses):
        if clause.count("=") != 1:
            raise StructuredLedgerError("clause does not contain one equality")
        expression, rendered_result = [part.strip() for part in clause.split("=", 1)]
        op, left, right, computed = _atomic_expression(expression)
        stated = _fraction(rendered_result)
        if computed != stated:
            raise StructuredLedgerError("clause arithmetic is false")
        start = think.find(clause, search_cursor)
        if start < 0:
            raise StructuredLedgerError("clause provenance is ambiguous")
        search_cursor = start + len(clause)
        dependencies = []
        for role, operand in (("left", left), ("right", right)):
            if operand in latest_value_owner:
                dependencies.append(
                    {"operand_role": role, "record_index": latest_value_owner[operand]}
                )
        record = {
            "address": index,
            "operation": op,
            "operands": [_value(left), _value(right)],
            "result": _value(stated),
            "dependencies": dependencies,
            "provenance": {"think_start": start, "think_end": start + len(clause)},
        }
        records.append(record)
        latest_value_owner[stated] = index

    if _answer_fraction(answer) != _fraction(
        f"{records[-1]['result']['numerator']}/{records[-1]['result']['denominator']}"
    ):
        raise StructuredLedgerError("terminal ledger value differs from answer")
    return records


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise StructuredLedgerError(f"invalid JSON at line {line_number}") from error
            if not isinstance(row, dict):
                raise StructuredLedgerError(f"non-object row at line {line_number}")
            yield row


def _split(question_sha256: str, development_modulus: int) -> str:
    return (
        "development"
        if int(question_sha256[:16], 16) % development_modulus == 0
        else "train"
    )


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise StructuredLedgerError(f"refusing existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            payload = canonical_json_bytes(row)
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def build(
    source: Path,
    output_root: Path,
    *,
    expected_source_sha256: str | None = None,
    development_modulus: int = 20,
) -> dict[str, Any]:
    if output_root.exists():
        raise StructuredLedgerError(f"refusing existing output root: {output_root}")
    if development_modulus < 2:
        raise StructuredLedgerError("development modulus must be at least two")
    source_sha256 = sha256_file(source)
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise StructuredLedgerError("source SHA-256 differs")

    counters: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "development": Counter(),
    }
    operation_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "development": Counter(),
    }
    seen_questions: set[str] = set()
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "development": []}

    for source_index, source_row in enumerate(_iter_jsonl(source)):
        counters["source_rows"] += 1
        family = str(source_row.get("family", ""))
        if family not in SUPPORTED_FAMILIES:
            counters["unsupported_family"] += 1
            continue
        question = source_row.get("question")
        response = source_row.get("response")
        if not isinstance(question, str) or not isinstance(response, str):
            counters["malformed_surface"] += 1
            continue
        question_sha256 = hashlib.sha256(question.strip().encode()).hexdigest()
        if question_sha256 in seen_questions:
            raise StructuredLedgerError("duplicate normalized question")
        seen_questions.add(question_sha256)
        try:
            records = compile_response(response, source_row.get("answer"))
        except StructuredLedgerError as error:
            counters[f"rejected:{error}"] += 1
            continue
        split = _split(question_sha256, development_modulus)
        identity = hashlib.sha256(
            f"{SCHEMA}\0{question_sha256}\0{source_index}".encode()
        ).hexdigest()
        row = {
            "schema": SCHEMA,
            "identity_sha256": identity,
            "source_question_sha256": question_sha256,
            "split": split,
            "family": family,
            "question": question,
            "records": records,
            "terminal_value": records[-1]["result"],
            "source_response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        }
        outputs[split].append(row)
        counters["admitted_rows"] += 1
        counters[f"admitted_{split}"] += 1
        family_counts[split][family] += 1
        operation_counts[split].update(record["operation"] for record in records)

    if not outputs["train"] or not outputs["development"]:
        raise StructuredLedgerError("one output split is empty")
    output_root.mkdir(parents=True)
    paths = {split: output_root / f"{split}.jsonl" for split in outputs}
    hashes = {split: _atomic_jsonl(paths[split], outputs[split]) for split in outputs}
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "supported_families": sorted(SUPPORTED_FAMILIES),
        "development_modulus": development_modulus,
        "counters": dict(sorted(counters.items())),
        "family_counts": {
            split: dict(sorted(counts.items())) for split, counts in family_counts.items()
        },
        "operation_counts": {
            split: dict(sorted(counts.items())) for split, counts in operation_counts.items()
        },
        "outputs": {
            split: {
                "path": str(paths[split].resolve()),
                "rows": len(outputs[split]),
                "sha256": hashes[split],
            }
            for split in outputs
        },
        "exact_arithmetic_verified": True,
        "holdout_used": False,
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--development-modulus", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.source,
                args.output,
                expected_source_sha256=args.expected_source_sha256,
                development_modulus=args.development_modulus,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
