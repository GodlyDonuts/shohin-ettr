#!/usr/bin/env python3
"""Aggregate the frozen AQC1 treatment and matched control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "shohin-aqc1-commit-report-v1"
AGGREGATE_SCHEMA = "shohin-aqc1-aggregate-v1"


class AQC1AggregateError(RuntimeError):
    """AQC1 arm custody or comparison differs."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path, arm: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete":
        raise AQC1AggregateError(f"incomplete AQC1 report: {path}")
    if report.get("arm") != arm or report.get("protected_adapter_unchanged") is not True:
        raise AQC1AggregateError(f"AQC1 arm or protected host differs: {path}")
    return report


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise AQC1AggregateError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    treatment = load(args.treatment, "antisymmetric")
    control = load(args.control, "independent")
    shared = (
        "adapter_checkpoint_sha256",
        "pairs_sha256",
        "model_revision",
        "updates",
        "max_sequence_length",
    )
    if any(treatment.get(key) != control.get(key) for key in shared):
        raise AQC1AggregateError("AQC1 matched-arm settings differ")
    treatment_score = treatment["holdout"]["overall"]["selected_correct"]
    control_score = control["holdout"]["overall"]["selected_correct"]
    mechanism_gate = {
        "treatment_capability_gate_pass": treatment["holdout_gate_pass"] is True,
        "treatment_beats_control_by_5": treatment_score >= control_score + 5,
    }
    practical = max(
        ((treatment_score, "antisymmetric"), (control_score, "independent")),
        key=lambda item: item[0],
    )
    report = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete",
        "treatment": {
            "path": str(args.treatment.resolve()),
            "sha256": sha256_file(args.treatment),
            "selected_correct": treatment_score,
            "capability_gate_pass": treatment["holdout_gate_pass"],
        },
        "control": {
            "path": str(args.control.resolve()),
            "sha256": sha256_file(args.control),
            "selected_correct": control_score,
            "capability_gate_pass": control["holdout_gate_pass"],
        },
        "mechanism_gate": mechanism_gate,
        "mechanism_gate_pass": all(mechanism_gate.values()),
        "practical_winner": practical[1],
        "practical_winner_score": practical[0],
        "metadata_control_score": 645,
        "always_idr1_score": 625,
        "oracle_score": 671,
        "product_remains_sealed": True,
    }
    atomic_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
