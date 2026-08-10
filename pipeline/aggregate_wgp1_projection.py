#!/usr/bin/env python3
"""Aggregate the frozen WGP1 weighted grammar projection gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class WGP1AggregateError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, control: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "shohin-btt1-evaluation-v1" or payload.get("status") != "complete" or payload.get("control") != control or payload.get("projection") != "grammar-v1" or payload.get("holdout_used"):
        raise WGP1AggregateError(f"{control} report differs")
    return payload


def group(report: dict[str, Any], key: str) -> float:
    value = report["groups"].get(key)
    return 0.0 if not value else float(value["exact_rate"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise WGP1AggregateError("refusing existing output")
    reports = {
        "normal": load(args.normal, "normal"), "source_shuffled": load(args.source_shuffled, "source_shuffled"),
        "zero_bytes": load(args.zero_bytes, "zero_bytes"), "flat_executor": load(args.flat_executor, "flat_executor"),
    }
    reference = reports["normal"]
    for report in reports.values():
        if report["checkpoint_sha256"] != reference["checkpoint_sha256"] or report["data_sha256"] != reference["data_sha256"] or report["counts"]["rows"] != 3917:
            raise WGP1AggregateError("evaluation custody differs")
    normal = reference["rates"]
    family_min = min(value["exact_rate"] for key, value in reference["groups"].items() if key.startswith("family:"))
    shuffled = reports["source_shuffled"]["rates"]["exact_skeleton"]
    zeroed = reports["zero_bytes"]["rates"]["exact_skeleton"]
    flat_loss = group(reference, "hierarchical:true") - group(reports["flat_executor"], "hierarchical:true")
    gates = {
        "byte_role_sequence_at_least_0p995": normal["byte_role_sequence_exact"] >= 0.995,
        "selected_byte_sequence_at_least_0p995": normal["selected_byte_sequence_exact"] >= 0.995,
        "valid_program_exact": normal["valid_program"] == 1.0,
        "exact_skeleton_at_least_0p995": normal["exact_skeleton"] >= 0.995,
        "every_family_at_least_0p99": family_min >= 0.99,
        "mixed_precedence_at_least_0p95": group(reference, "mixed:true") >= 0.95,
        "unary_group_at_least_0p95": group(reference, "unary:true") >= 0.95,
        "parentheses_three_plus_at_least_0p95": group(reference, "parentheses:3+") >= 0.95,
        "repairs_at_least_30": reference["counts"]["repairs"] >= 30,
        "breaks_at_most_2": reference["counts"]["breaks"] <= 2,
        "zero_search_exhaustion": reference["counts"]["search_exhausted"] == 0,
        "source_shuffled_at_most_0p25": shuffled <= 0.25,
        "source_margin_at_least_0p74": normal["exact_skeleton"] - shuffled >= 0.74,
        "zero_bytes_at_most_0p25": zeroed <= 0.25,
        "zero_bytes_margin_at_least_0p74": normal["exact_skeleton"] - zeroed >= 0.74,
        "flat_hierarchical_loss_at_least_0p35": flat_loss >= 0.35,
    }
    result = {
        "schema": "shohin-wgp1-comparison-v1", "status": "complete", "holdout_used": False,
        "checkpoint_sha256": reference["checkpoint_sha256"], "data_sha256": reference["data_sha256"],
        "normal_rates": normal, "normal_counts": reference["counts"], "family_minimum": family_min,
        "mixed_precedence_exact": group(reference, "mixed:true"), "unary_group_exact": group(reference, "unary:true"),
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
