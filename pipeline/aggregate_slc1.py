#!/usr/bin/env python3
"""Aggregate complete SLC1 shards and apply the prospectively frozen gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-slc1-comparison-v1"
METRICS = (
    "syntax_valid",
    "canonical_exact",
    "record_count_exact",
    "operation_sequence_exact",
    "all_records_exact",
    "terminal_exact",
)


class SLC1AggregateError(ValueError):
    """Raised when SLC1 evaluation coverage or custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_arm(paths: list[Path], expected_control: str) -> tuple[dict[str, Any], set[str]]:
    totals: Counter[str] = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_depth: dict[str, Counter[str]] = defaultdict(Counter)
    identities: set[str] = set()
    checkpoint_hash = None
    data_hash = None
    generated_tokens = 0
    exhausted = 0
    receipts = []
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("status") != "complete" or report.get("control") != expected_control:
            raise SLC1AggregateError("shard status or control differs")
        if report.get("holdout_used") is not False:
            raise SLC1AggregateError("a shard used holdout")
        checkpoint_hash = checkpoint_hash or report.get("checkpoint_sha256")
        data_hash = data_hash or report.get("data_sha256")
        if report.get("checkpoint_sha256") != checkpoint_hash or report.get("data_sha256") != data_hash:
            raise SLC1AggregateError("shard checkpoint or data differs")
        for detail in report.get("details", []):
            identity = str(detail.get("identity_sha256"))
            if identity in identities:
                raise SLC1AggregateError("duplicate identity across shards")
            identities.add(identity)
        totals.update({key: int(value) for key, value in report["counts"].items()})
        for family, values in report["by_family"].items():
            by_family[family].update({key: int(value) for key, value in values.items()})
        for depth, values in report["by_depth"].items():
            by_depth[depth].update({key: int(value) for key, value in values.items()})
        generated_tokens += int(report["generated_tokens"])
        exhausted += int(report["exhausted"])
        receipts.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    rows = totals["rows"]
    if rows != 3917 or len(identities) != rows:
        raise SLC1AggregateError("arm does not cover all 3,917 identities")
    return (
        {
            "counts": dict(sorted(totals.items())),
            "rates": {metric: totals[metric] / rows for metric in METRICS},
            "by_family": {
                family: {
                    "counts": dict(sorted(values.items())),
                    "terminal_rate": values["terminal_exact"] / values["rows"],
                }
                for family, values in sorted(by_family.items())
            },
            "by_depth": {
                depth: {
                    "counts": dict(sorted(values.items())),
                    "terminal_rate": values["terminal_exact"] / values["rows"],
                }
                for depth, values in sorted(by_depth.items())
            },
            "checkpoint_sha256": checkpoint_hash,
            "data_sha256": data_hash,
            "generated_tokens": generated_tokens,
            "exhausted": exhausted,
            "receipts": receipts,
        },
        identities,
    )


def aggregate(normal_paths: list[Path], shuffled_paths: list[Path]) -> dict[str, Any]:
    normal, normal_ids = _load_arm(normal_paths, "normal")
    shuffled, shuffled_ids = _load_arm(shuffled_paths, "source_shuffled")
    if normal_ids != shuffled_ids:
        raise SLC1AggregateError("aligned and shuffled identity coverage differs")
    if normal["checkpoint_sha256"] != shuffled["checkpoint_sha256"]:
        raise SLC1AggregateError("aligned and shuffled checkpoints differ")
    aligned_terminal = normal["rates"]["terminal_exact"]
    shuffled_terminal = shuffled["rates"]["terminal_exact"]
    gates = {
        "syntax_at_least_0p99": normal["rates"]["syntax_valid"] >= 0.99,
        "record_count_at_least_0p99": normal["rates"]["record_count_exact"] >= 0.99,
        "operation_sequence_at_least_0p95": normal["rates"]["operation_sequence_exact"] >= 0.95,
        "all_records_at_least_0p90": normal["rates"]["all_records_exact"] >= 0.90,
        "terminal_at_least_0p95": aligned_terminal >= 0.95,
        "every_family_terminal_at_least_0p90": all(
            value["terminal_rate"] >= 0.90 for value in normal["by_family"].values()
        ),
        "depth_five_terminal_at_least_0p85": normal["by_depth"]["5"]["terminal_rate"] >= 0.85,
        "source_causal_margin_at_least_0p65": aligned_terminal - shuffled_terminal >= 0.65,
        "shuffled_terminal_at_most_0p25": shuffled_terminal <= 0.25,
        "zero_exhaustion": normal["exhausted"] == 0 and shuffled["exhausted"] == 0,
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "normal": normal,
        "source_shuffled": shuffled,
        "terminal_causal_margin": aligned_terminal - shuffled_terminal,
        "gates": gates,
        "overall_pass": all(gates.values()),
        "holdout_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, nargs="+", required=True)
    parser.add_argument("--source-shuffled", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SLC1AggregateError(f"refusing existing output: {args.output}")
    report = aggregate(args.normal, args.source_shuffled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"overall_pass": report["overall_pass"], "gates": report["gates"]}, sort_keys=True))
    return 0 if report["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
