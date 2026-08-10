#!/usr/bin/env python3
"""Aggregate the frozen PSTC1 stack compiler and causal controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-pstc1-stack-comparison-v1"
EVAL_SCHEMA = "shohin-pstc1-stack-evaluation-v1"


class PSTC1AggregateError(ValueError):
    """Raised when PSTC1 evaluation receipts differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_report(path: Path, control: str) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if (
        report.get("schema") != EVAL_SCHEMA
        or report.get("status") != "complete"
        or report.get("holdout_used") is not False
        or report.get("control") != control
        or int(report.get("counts", {}).get("rows", -1)) != 3917
    ):
        raise PSTC1AggregateError(f"{control} report differs")
    return report


def _group_rate(report: dict[str, Any], key: str) -> float:
    group = report["groups"][key]
    return int(group["exact_skeleton"]) / int(group["rows"])


def _intervention_loss(
    normal: dict[str, Any], control: dict[str, Any], predicate: Any
) -> float:
    normal_by_id = {detail["identity_sha256"]: detail for detail in normal["details"]}
    control_by_id = {detail["identity_sha256"]: detail for detail in control["details"]}
    selected = [identity for identity, detail in normal_by_id.items() if predicate(detail)]
    if not selected or set(normal_by_id) != set(control_by_id):
        raise PSTC1AggregateError("intervention identity coverage differs")
    return sum(
        int(normal_by_id[identity]["exact_skeleton"])
        - int(control_by_id[identity]["exact_skeleton"])
        for identity in selected
    ) / len(selected)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise PSTC1AggregateError("refusing existing output")
    reports = {
        "normal": load_report(args.normal, "normal"),
        "source_shuffled": load_report(args.source_shuffled, "source_shuffled"),
        "stack_reset": load_report(args.stack_reset, "stack_reset"),
        "stack_top_permuted": load_report(args.stack_top_permuted, "stack_top_permuted"),
    }
    custody = {
        (report["checkpoint_sha256"], report["data_sha256"])
        for report in reports.values()
    }
    if len(custody) != 1:
        raise PSTC1AggregateError("evaluation custody differs")
    normal = reports["normal"]
    shuffled = reports["source_shuffled"]
    reset = reports["stack_reset"]
    permuted = reports["stack_top_permuted"]
    normal_exact = float(normal["rates"]["exact_skeleton"])
    shuffled_exact = float(shuffled["rates"]["exact_skeleton"])
    family_minimum = min(
        _group_rate(normal, key)
        for key in normal["groups"]
        if key.startswith("family:")
    )
    reset_loss = _intervention_loss(
        normal,
        reset,
        lambda detail: detail["mixed_precedence"]
        or detail["unary_group"]
        or detail["parenthesis_count"] > 0,
    )
    permutation_loss = _intervention_loss(
        normal, permuted, lambda detail: int(detail["binary_depth"]) >= 3
    )
    gates = {
        "action_length_at_least_0p99": float(normal["rates"]["action_length_exact"]) >= 0.99,
        "action_sequence_at_least_0p97": float(normal["rates"]["action_sequence_exact"]) >= 0.97,
        "pointer_value_at_least_0p97": float(normal["rates"]["pointer_value_exact"]) >= 0.97,
        "valid_program_at_least_0p99": float(normal["rates"]["valid_program"]) >= 0.99,
        "exact_skeleton_at_least_0p92": normal_exact >= 0.92,
        "every_family_at_least_0p88": family_minimum >= 0.88,
        "mixed_precedence_at_least_0p85": _group_rate(normal, "mixed:true") >= 0.85,
        "unary_group_at_least_0p80": _group_rate(normal, "unary:true") >= 0.80,
        "parentheses_three_plus_at_least_0p80": _group_rate(normal, "parentheses:3+") >= 0.80,
        "source_margin_at_least_0p65": normal_exact - shuffled_exact >= 0.65,
        "source_shuffled_at_most_0p25": shuffled_exact <= 0.25,
        "stack_reset_loss_at_least_0p30": reset_loss >= 0.30,
        "stack_top_permutation_loss_at_least_0p20": permutation_loss >= 0.20,
        "zero_invalid_transitions": int(normal["counts"]["invalid_transitions"]) == 0,
    }
    comparison = {
        "schema": SCHEMA,
        "status": "complete",
        "overall_pass": all(gates.values()),
        "holdout_used": False,
        "gates": gates,
        "normal_exact": normal_exact,
        "source_shuffled_exact": shuffled_exact,
        "source_causal_margin": normal_exact - shuffled_exact,
        "every_family_minimum": family_minimum,
        "mixed_precedence_exact": _group_rate(normal, "mixed:true"),
        "unary_group_exact": _group_rate(normal, "unary:true"),
        "parentheses_three_plus_exact": _group_rate(normal, "parentheses:3+"),
        "stack_reset_hierarchical_loss": reset_loss,
        "stack_top_permutation_deep_loss": permutation_loss,
        "checkpoint_sha256": normal["checkpoint_sha256"],
        "data_sha256": normal["data_sha256"],
        "reports": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in (
                ("normal", args.normal),
                ("source_shuffled", args.source_shuffled),
                ("stack_reset", args.stack_reset),
                ("stack_top_permuted", args.stack_top_permuted),
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--source-shuffled", type=Path, required=True)
    parser.add_argument("--stack-reset", type=Path, required=True)
    parser.add_argument("--stack-top-permuted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    return 0 if result["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
