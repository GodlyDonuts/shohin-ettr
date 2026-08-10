#!/usr/bin/env python3
"""Aggregate the frozen TMC1 development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "shohin-tmc1-development-aggregate-v1"
DIRECT_CORRECT = 267
ROWS = 666


class TMC1AggregateError(ValueError):
    """TMC1 evaluation reports differ from the frozen gate."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, control: str) -> dict[str, object]:
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "shohin-tmc1-development-evaluation-v1"
        or report.get("control") != control
        or report.get("holdout_used") is not False
        or report.get("public_test_opened") is not False
    ):
        raise TMC1AggregateError("input report differs")
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise TMC1AggregateError("refusing existing aggregate")
    normal = load(args.normal, "normal")
    shuffled = load(args.shuffled, "source_shuffled")
    identities = [
        {str(row["identity_sha256"]) for row in report["details"]}
        for report in (normal, shuffled)
    ]
    if identities[0] != identities[1] or len(identities[0]) != ROWS:
        raise TMC1AggregateError("evaluation identities differ")
    nc = normal["counts"]
    sc = shuffled["counts"]
    answer = nc.get("normal:correct", 0) / ROWS
    shuffled_answer = sc.get("normal:correct", 0) / ROWS
    operation = nc.get("operation_correct", 0) / nc["operation_fields"]
    operand = nc.get("operand_correct", 0) / nc["operand_fields"]
    normal_correct = nc.get("normal_correct_multi_digit_rows", 0)
    carry_retained = nc.get("carry_reset:normal_correct_multi_digit_correct", 0) / max(
        normal_correct, 1
    )
    opcode = nc.get("opcode_permuted:correct", 0) / ROWS
    direct = DIRECT_CORRECT / ROWS
    metrics = {
        "answer_rate": answer,
        "answer_correct": nc.get("normal:correct", 0),
        "direct_answer_rate": direct,
        "direct_answer_correct": DIRECT_CORRECT,
        "source_shuffled_answer_rate": shuffled_answer,
        "operation_accuracy": operation,
        "operand_owner_accuracy": operand,
        "graph_exact_rate": nc.get("graph_exact", 0) / ROWS,
        "instruction_count_accuracy": nc.get("instruction_count_exact", 0) / ROWS,
        "carry_reset_retention_on_normal_correct": carry_retained,
        "normal_correct_multi_digit_rows": normal_correct,
        "opcode_permuted_answer_rate": opcode,
    }
    gates = {
        "answer_at_least_0p45": answer >= 0.45,
        "beats_direct_by_0p05": answer - direct >= 0.05,
        "operation_at_least_0p80": operation >= 0.80,
        "operand_owner_at_least_0p80": operand >= 0.80,
        "source_shuffled_at_most_0p10": shuffled_answer <= 0.10,
        "carry_reset_loses_0p10": (normal_correct > 0 and 1.0 - carry_retained >= 0.10),
        "opcode_permutation_loses_0p30": answer - opcode >= 0.30,
        "zero_invalid": nc.get("normal:invalid", 0) == 0,
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "rows": ROWS,
        "inputs": {
            "normal": {
                "path": str(args.normal.resolve()),
                "sha256": sha256_file(args.normal),
            },
            "shuffled": {
                "path": str(args.shuffled.resolve()),
                "sha256": sha256_file(args.shuffled),
            },
        },
        "metrics": metrics,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    return 0 if result["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
