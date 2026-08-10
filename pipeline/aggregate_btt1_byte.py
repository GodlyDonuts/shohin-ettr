#!/usr/bin/env python3
"""Aggregate the frozen BTT1 development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class BTT1AggregateError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, control: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "shohin-btt1-evaluation-v1" or payload.get("status") != "complete" or payload.get("control") != control or payload.get("holdout_used"):
        raise BTT1AggregateError(f"{control} report differs")
    return payload


def group(report: dict[str, Any], key: str) -> float:
    value = report["groups"].get(key)
    return 0.0 if not value else float(value["exact_rate"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise BTT1AggregateError("refusing existing output")
    reports = {
        "normal": load(args.normal, "normal"), "source_shuffled": load(args.source_shuffled, "source_shuffled"),
        "zero_bytes": load(args.zero_bytes, "zero_bytes"), "flat_executor": load(args.flat_executor, "flat_executor"),
    }
    reference = reports["normal"]
    for report in reports.values():
        if report["checkpoint_sha256"] != reference["checkpoint_sha256"] or report["data_sha256"] != reference["data_sha256"] or report["counts"]["rows"] != 3917:
            raise BTT1AggregateError("evaluation custody differs")
    normal = reference["rates"]
    family_min = min(value["exact_rate"] for key, value in reference["groups"].items() if key.startswith("family:"))
    shuffled = reports["source_shuffled"]["rates"]["exact_skeleton"]
    zeroed = reports["zero_bytes"]["rates"]["exact_skeleton"]
    flat_loss = group(reference, "hierarchical:true") - group(reports["flat_executor"], "hierarchical:true")
    gates = {
        "byte_role_sequence_at_least_0p99": normal["byte_role_sequence_exact"] >= 0.99,
        "selected_byte_sequence_at_least_0p99": normal["selected_byte_sequence_exact"] >= 0.99,
        "valid_program_at_least_0p995": normal["valid_program"] >= 0.995,
        "exact_skeleton_at_least_0p97": normal["exact_skeleton"] >= 0.97,
        "every_family_at_least_0p95": family_min >= 0.95,
        "mixed_precedence_at_least_0p90": group(reference, "mixed:true") >= 0.90,
        "unary_group_at_least_0p90": group(reference, "unary:true") >= 0.90,
        "parentheses_three_plus_at_least_0p90": group(reference, "parentheses:3+") >= 0.90,
        "source_shuffled_at_most_0p25": shuffled <= 0.25,
        "source_margin_at_least_0p70": normal["exact_skeleton"] - shuffled >= 0.70,
        "zero_bytes_at_most_0p25": zeroed <= 0.25,
        "zero_bytes_margin_at_least_0p70": normal["exact_skeleton"] - zeroed >= 0.70,
        "flat_hierarchical_loss_at_least_0p35": flat_loss >= 0.35,
    }
    result = {
        "schema": "shohin-btt1-comparison-v1", "status": "complete", "holdout_used": False,
        "checkpoint_sha256": reference["checkpoint_sha256"], "data_sha256": reference["data_sha256"],
        "normal_rates": normal, "family_minimum": family_min,
        "mixed_precedence_exact": group(reference, "mixed:true"),
        "unary_group_exact": group(reference, "unary:true"),
        "parentheses_three_plus_exact": group(reference, "parentheses:3+"),
        "source_shuffled_exact": shuffled, "source_causal_margin": normal["exact_skeleton"] - shuffled,
        "zero_bytes_exact": zeroed, "zero_bytes_causal_margin": normal["exact_skeleton"] - zeroed,
        "flat_hierarchical_loss": flat_loss, "gates": gates, "overall_pass": all(gates.values()),
        "reports": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in {
            "normal": args.normal, "source_shuffled": args.source_shuffled,
            "zero_bytes": args.zero_bytes, "flat_executor": args.flat_executor,
        }.items()},
    }
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--source-shuffled", type=Path, required=True)
    parser.add_argument("--zero-bytes", type=Path, required=True)
    parser.add_argument("--flat-executor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    return 0 if result["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
