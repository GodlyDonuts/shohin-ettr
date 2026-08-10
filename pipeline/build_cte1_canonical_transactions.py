#!/usr/bin/env python3
"""Build exact canonical transaction targets from immutable NMC1 programs."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from draft_transaction_compiler import compile_draft_transactions
from natural_microcode_program import (
    RegisterProgram,
    canonical_fraction,
    execute_fraction,
    parse_fraction,
    parse_program,
)
from typed_microcode_graph import execute_fraction as execute_typed_fraction


SCHEMA = "shohin-cte1-canonical-transaction-data-v1"
REPORT_SCHEMA = "shohin-cte1-canonical-transaction-data-report-v1"
PROMPT_PREFIX = (
    "Emit a concise arithmetic transaction trace for the word problem. Use "
    "<<expression=result>> for every step, then write #### followed by the "
    "final result. Emit no other text.\n\nPROBLEM:\n"
)
OPERATOR = {
    "APPLY_ADD": "+",
    "APPLY_SUB": "-",
    "APPLY_MUL": "*",
    "APPLY_DIV": "/",
}


class CTE1DataError(ValueError):
    """Canonical transaction data violates the frozen CTE1 contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise CTE1DataError(f"invalid JSON at line {line_number}") from error
            if not isinstance(row, dict):
                raise CTE1DataError(f"non-object row at line {line_number}")
            yield row


def render_transaction_trace(program: RegisterProgram) -> tuple[str, Fraction, int]:
    if program.commit != len(program.records) - 1:
        raise CTE1DataError("commit is not the final causal record")
    register_values: list[Fraction] = []
    lines: list[str] = []
    load_count = 0
    for record in program.records:
        stack: list[tuple[str, Fraction]] = []
        for action in record:
            name = action.get("action")
            if name == "PUSH":
                value = parse_fraction(str(action.get("surface")))
                stack.append((canonical_fraction(value), value))
            elif name == "LOAD":
                register = action.get("register")
                if type(register) is not int or not 0 <= register < len(register_values):
                    raise CTE1DataError("LOAD is not causal")
                value = register_values[register]
                stack.append((canonical_fraction(value), value))
                load_count += 1
            elif name == "NEGATE":
                if not stack:
                    raise CTE1DataError("NEGATE underflow")
                expression, value = stack.pop()
                stack.append((f"(-{expression})", -value))
            elif name in OPERATOR:
                if len(stack) < 2:
                    raise CTE1DataError("binary operation underflow")
                right_expression, right = stack.pop()
                left_expression, left = stack.pop()
                if name == "APPLY_ADD":
                    result = left + right
                elif name == "APPLY_SUB":
                    result = left - right
                elif name == "APPLY_MUL":
                    result = left * right
                elif right:
                    result = left / right
                else:
                    raise CTE1DataError("division by zero")
                stack.append(
                    (
                        f"({left_expression}{OPERATOR[str(name)]}{right_expression})",
                        result,
                    )
                )
            else:
                raise CTE1DataError("program action differs")
        if len(stack) != 1:
            raise CTE1DataError("record stack differs")
        expression, result = stack[0]
        register_values.append(result)
        lines.append(f"<<{expression}={canonical_fraction(result)}>>")
    final = register_values[program.commit]
    lines.append(f"#### {canonical_fraction(final)}")
    return "\n".join(lines), final, load_count


def convert_row(row: dict[str, Any], split: str) -> tuple[dict[str, Any], Counter[str]]:
    required = ("identity_sha256", "original_question", "gold_program", "gold_answer")
    if any(not isinstance(row.get(key), str) for key in required):
        raise CTE1DataError("source row fields differ")
    program = parse_program(row["gold_program"])
    response, final, load_count = render_transaction_trace(program)
    expected = Fraction(row["gold_answer"])
    if execute_fraction(program) != expected or final != expected:
        raise CTE1DataError("source program terminal differs")
    graph, receipt = compile_draft_transactions(row["original_question"], response)
    if (
        execute_typed_fraction(graph) != expected
        or receipt.accepted != len(program.records)
        or receipt.rejected
        # The typed graph also uses STATE edges inside a fully parenthesized
        # record, so cross-record LOAD ownership is a lower bound rather than
        # the complete state-read count.
        or receipt.state_reads < load_count
    ):
        raise CTE1DataError("canonical transaction round trip differs")
    converted = {
        "schema": SCHEMA,
        "identity_sha256": row["identity_sha256"],
        "split": split,
        "source": row.get("source"),
        "original_question": row["original_question"],
        "gold_answer": row["gold_answer"],
        "register_depth": row.get("register_depth"),
        "question": PROMPT_PREFIX + row["original_question"],
        "response": response,
        "arm": "cte1",
    }
    counts: Counter[str] = Counter(
        {
            "rows": 1,
            "transactions": receipt.accepted,
            "state_reads": receipt.state_reads,
            "source_reads": receipt.source_reads,
            "literal_reads": receipt.literal_reads,
            "response_characters": len(response),
        }
    )
    return converted, counts


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
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
    if args.output.exists():
        raise CTE1DataError("refusing existing output root")
    inputs = {
        "train": (args.train, args.expected_train_sha256, 6333),
        "development": (args.development, args.expected_development_sha256, 666),
    }
    converted: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, Counter[str]] = {}
    identities: dict[str, set[str]] = {}
    for split, (path, expected_sha256, expected_rows) in inputs.items():
        if sha256_file(path) != expected_sha256:
            raise CTE1DataError(f"{split} input SHA-256 differs")
        converted[split] = []
        metrics[split] = Counter()
        identities[split] = set()
        for row in iter_jsonl(path):
            item, counts = convert_row(row, split)
            identity = str(item["identity_sha256"])
            if identity in identities[split]:
                raise CTE1DataError(f"duplicate {split} identity")
            identities[split].add(identity)
            converted[split].append(item)
            metrics[split].update(counts)
        if len(converted[split]) != expected_rows:
            raise CTE1DataError(f"{split} population differs")
    if identities["train"] & identities["development"]:
        raise CTE1DataError("train/development identity overlap")
    args.output.mkdir(parents=True)
    outputs = {}
    for split, rows in converted.items():
        path = args.output / f"{split}.jsonl"
        outputs[split] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "sha256": write_jsonl(path, rows),
            "counts": dict(sorted(metrics[split].items())),
            "maximum_response_characters": max(len(row["response"]) for row in rows),
        }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "pass",
        "holdout_used": False,
        "public_test_opened": False,
        "inputs": {
            "train_sha256": args.expected_train_sha256,
            "development_sha256": args.expected_development_sha256,
        },
        "outputs": outputs,
    }
    report_path = args.output / "report.json"
    temporary = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--expected-development-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
