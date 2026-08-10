#!/usr/bin/env python3
"""Reduce the three frozen DTMC1 development controls conjunctively."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "shohin-dtmc1-development-aggregate-v1"


class DTMC1AggregateError(ValueError):
    """Frozen DTMC1 evaluation reports differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, control: str) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "shohin-dtmc1-development-evaluation-v1"
        or report.get("status") != "complete"
        or report.get("control") != control
        or report.get("holdout_used") is not False
        or report.get("public_test_opened") is not False
        or report.get("counts", {}).get("rows") != 666
    ):
        raise DTMC1AggregateError(f"{control} report differs")
    return report


def run(args: argparse.Namespace) -> dict[str, object]:
    reports = {
        "normal": load(args.normal, "normal"),
        "draft_shuffled": load(args.draft_shuffled, "draft_shuffled"),
        "source_draft_shuffled": load(
            args.source_draft_shuffled, "source_draft_shuffled"
        ),
    }
    custody = {
        key: reports["normal"].get(key)
        for key in (
            "model_revision",
            "owner_checkpoint_sha256",
            "compiler_checkpoint_sha256",
            "compiler_updates",
            "compiler_training_data_sha256",
            "development_data_sha256",
            "draft_report_sha256",
            "lam_checkpoint_sha256",
        )
    }
    for control, report in reports.items():
        if any(report.get(key) != value for key, value in custody.items()):
            raise DTMC1AggregateError(f"{control} custody differs")
    normal = reports["normal"]["counts"]
    draft_shuffled = reports["draft_shuffled"]["counts"]
    source_shuffled = reports["source_draft_shuffled"]["counts"]
    rows = 666
    normal_correct = int(normal.get("normal:correct", 0))
    draft_shuffled_correct = int(draft_shuffled.get("normal:correct", 0))
    source_shuffled_correct = int(source_shuffled.get("normal:correct", 0))
    operation_fields = int(normal.get("operation_fields", 0))
    operand_fields = int(normal.get("operand_fields", 0))
    operation_accuracy = int(normal.get("operation_correct", 0)) / operation_fields
    operand_accuracy = int(normal.get("operand_correct", 0)) / operand_fields
    multi_digit = int(normal.get("normal_correct_multi_digit_rows", 0))
    carry_retained = int(
        normal.get("carry_reset:normal_correct_multi_digit_correct", 0)
    )
    carry_loss = 1.0 - carry_retained / multi_digit if multi_digit else 0.0
    opcode_loss = (
        normal_correct - int(normal.get("opcode_permuted:correct", 0))
    ) / rows
    gates = {
        "aligned_answers_at_least_301": normal_correct >= 301,
        "aligned_beats_direct_by_5_points": normal_correct - 267 >= 34,
        "operation_accuracy_at_least_80pct": operation_accuracy >= 0.80,
        "operand_accuracy_at_least_80pct": operand_accuracy >= 0.80,
        "draft_margin_at_least_10_points": (normal_correct - draft_shuffled_correct)
        / rows
        >= 0.10,
        "source_draft_shuffle_at_most_10pct": source_shuffled_correct / rows <= 0.10,
        "carry_loss_at_least_10_points": carry_loss >= 0.10,
        "opcode_loss_at_least_30_points": opcode_loss >= 0.30,
        "zero_normal_invalid": int(normal.get("normal:invalid", 0)) == 0,
    }
    result = {
        "schema": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "holdout_used": False,
        "public_test_opened": False,
        "custody": custody,
        "report_sha256": {
            "normal": sha256_file(args.normal),
            "draft_shuffled": sha256_file(args.draft_shuffled),
            "source_draft_shuffled": sha256_file(args.source_draft_shuffled),
        },
        "scores": {
            "aligned_correct": normal_correct,
            "aligned_accuracy": normal_correct / rows,
            "draft_shuffled_correct": draft_shuffled_correct,
            "draft_shuffled_accuracy": draft_shuffled_correct / rows,
            "source_draft_shuffled_correct": source_shuffled_correct,
            "source_draft_shuffled_accuracy": source_shuffled_correct / rows,
            "direct_reference_correct": 267,
            "tmc1_reference_correct": 44,
            "operation_accuracy": operation_accuracy,
            "operand_accuracy": operand_accuracy,
            "multi_digit_normal_correct": multi_digit,
            "carry_reset_retained": carry_retained,
            "carry_loss": carry_loss,
            "opcode_loss": opcode_loss,
            "normal_invalid": int(normal.get("normal:invalid", 0)),
        },
        "gates": gates,
    }
    if args.output.exists():
        raise DTMC1AggregateError("refusing existing aggregate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--draft-shuffled", type=Path, required=True)
    parser.add_argument("--source-draft-shuffled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
