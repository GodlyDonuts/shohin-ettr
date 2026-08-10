#!/usr/bin/env python3
"""Aggregate the frozen FSTC1 skeleton treatment and causal controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-fstc1-skeleton-comparison-v1"
EVAL_SCHEMA = "shohin-fstc1-skeleton-evaluation-v1"


class FSTC1AggregateError(ValueError):
    """Raised when an FSTC1 evaluation receipt differs."""


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
        or int(report.get("shard_count", -1)) != 1
        or int(report.get("shard_index", -1)) != 0
    ):
        raise FSTC1AggregateError(f"{control} report differs")
    return report


def _rate(report: dict[str, Any], metric: str) -> float:
    return float(report["rates"][metric])


def _group_rate(counts: dict[str, Any], metric: str) -> float:
    return int(counts[metric]) / int(counts["rows"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FSTC1AggregateError("refusing existing output")
    reports = {
        "normal": load_report(args.normal, "normal"),
        "source_shuffled": load_report(args.source_shuffled, "source_shuffled"),
        "recurrence_reset": load_report(args.recurrence_reset, "recurrence_reset"),
    }
    custody = {
        (report["checkpoint_sha256"], report["data_sha256"])
        for report in reports.values()
    }
    if len(custody) != 1:
        raise FSTC1AggregateError("evaluation custody differs")
    identities = [
        {detail["identity_sha256"] for detail in report["details"]}
        for report in reports.values()
    ]
    if any(len(values) != 3917 for values in identities) or not all(
        values == identities[0] for values in identities[1:]
    ):
        raise FSTC1AggregateError("evaluation identity coverage differs")

    normal = reports["normal"]
    shuffled = reports["source_shuffled"]
    reset = reports["recurrence_reset"]
    normal_complete = _rate(normal, "complete_skeleton_exact")
    shuffled_complete = _rate(shuffled, "complete_skeleton_exact")
    every_family = min(
        _group_rate(counts, "complete_skeleton_exact")
        for counts in normal["by_family"].values()
    )
    depth_five = _group_rate(normal["by_depth"]["5"], "complete_skeleton_exact")
    normal_deep_rows = sum(int(normal["by_depth"][str(depth)]["rows"]) for depth in (3, 4, 5))
    normal_deep_exact = sum(
        int(normal["by_depth"][str(depth)]["complete_skeleton_exact"])
        for depth in (3, 4, 5)
    )
    reset_deep_exact = sum(
        int(reset["by_depth"][str(depth)]["complete_skeleton_exact"])
        for depth in (3, 4, 5)
    )
    reset_deep_loss = (normal_deep_exact - reset_deep_exact) / normal_deep_rows
    gates = {
        "depth_at_least_0p99": _rate(normal, "depth_exact") >= 0.99,
        "operation_sequence_at_least_0p97": _rate(normal, "operation_sequence_exact") >= 0.97,
        "operand_kind_at_least_0p97": _rate(normal, "reference_kind_exact") >= 0.97,
        "operand_value_at_least_0p97": _rate(normal, "operand_value_exact") >= 0.97,
        "polarity_at_least_0p97": _rate(normal, "polarity_exact") >= 0.97,
        "complete_at_least_0p90": normal_complete >= 0.90,
        "every_family_complete_at_least_0p85": every_family >= 0.85,
        "depth_five_complete_at_least_0p80": depth_five >= 0.80,
        "source_causal_margin_at_least_0p65": normal_complete - shuffled_complete >= 0.65,
        "source_shuffled_complete_at_most_0p25": shuffled_complete <= 0.25,
        "recurrence_reset_deep_loss_at_least_0p20": reset_deep_loss >= 0.20,
        "zero_invalid_references": int(normal["counts"]["invalid_reference"]) == 0,
    }
    comparison = {
        "schema": SCHEMA,
        "status": "complete",
        "overall_pass": all(gates.values()),
        "holdout_used": False,
        "gates": gates,
        "normal_complete": normal_complete,
        "source_shuffled_complete": shuffled_complete,
        "source_causal_margin": normal_complete - shuffled_complete,
        "every_family_complete_minimum": every_family,
        "depth_five_complete": depth_five,
        "recurrence_reset_deep_loss": reset_deep_loss,
        "reports": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in (
                ("normal", args.normal),
                ("source_shuffled", args.source_shuffled),
                ("recurrence_reset", args.recurrence_reset),
            )
        },
        "checkpoint_sha256": normal["checkpoint_sha256"],
        "data_sha256": normal["data_sha256"],
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
    parser.add_argument("--recurrence-reset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    return 0 if result["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
