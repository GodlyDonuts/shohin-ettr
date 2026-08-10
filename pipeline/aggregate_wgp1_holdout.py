#!/usr/bin/env python3
"""Aggregate the single WGP1 held-out-seed confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class WGP1HoldoutAggregateError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, control: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != "shohin-btt1-evaluation-v1" or value.get("status") != "complete" or value.get("control") != control or value.get("projection") != "grammar-v1" or not value.get("holdout_used"):
        raise WGP1HoldoutAggregateError(f"{control} confirmation report differs")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise WGP1HoldoutAggregateError("refusing existing output")
    reports = {"normal": load(args.normal, "normal"), "source_shuffled": load(args.source_shuffled, "source_shuffled"), "zero_bytes": load(args.zero_bytes, "zero_bytes")}
    normal = reports["normal"]
    rows = normal["counts"]["rows"]
    for report in reports.values():
        if report["checkpoint_sha256"] != normal["checkpoint_sha256"] or report["data_sha256"] != normal["data_sha256"] or report["counts"]["rows"] != rows:
            raise WGP1HoldoutAggregateError("confirmation custody differs")
    family_min = min(value["exact_rate"] for key, value in normal["groups"].items() if key.startswith("family:"))
    exact = normal["rates"]["exact_skeleton"]
    shuffled = reports["source_shuffled"]["rates"]["exact_skeleton"]
    zeroed = reports["zero_bytes"]["rates"]["exact_skeleton"]
    gates = {
        "exact_skeleton_at_least_0p99": exact >= 0.99,
        "every_family_at_least_0p98": family_min >= 0.98,
        "valid_program_exact": normal["rates"]["valid_program"] == 1.0,
        "source_shuffled_at_most_0p25": shuffled <= 0.25,
        "zero_bytes_at_most_0p25": zeroed <= 0.25,
        "zero_search_exhaustion": normal["counts"]["search_exhausted"] == 0,
    }
    result = {
        "schema": "shohin-wgp1-holdout-comparison-v1", "status": "complete", "holdout_used": True,
        "rows": rows, "checkpoint_sha256": normal["checkpoint_sha256"], "data_sha256": normal["data_sha256"],
        "normal_rates": normal["rates"], "normal_counts": normal["counts"], "family_minimum": family_min,
        "source_shuffled_exact": shuffled, "zero_bytes_exact": zeroed,
        "gates": gates, "overall_pass": all(gates.values()),
        "reports": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in {"normal": args.normal, "source_shuffled": args.source_shuffled, "zero_bytes": args.zero_bytes}.items()},
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
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    return 0 if result["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
