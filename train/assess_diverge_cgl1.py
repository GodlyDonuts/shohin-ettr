#!/usr/bin/env python3
"""Assess the three frozen DIVERGE-CGL1 development arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "shohin-diverge-cgl1-development-assessment-v1"


class CGL1AssessmentError(RuntimeError):
    """A CGL1 development arm or receipt differs."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise CGL1AssessmentError(f"CGL1 evaluation hash differs: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "shohin-diverge-cgl1-evaluation-v1":
        raise CGL1AssessmentError("CGL1 evaluation schema differs")
    return report


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("shohin", "smollm2", "flipped"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing CGL1 assessment: {args.output}")
    reports = {
        name: _load(getattr(args, name), getattr(args, f"{name}_sha256"))
        for name in ("shohin", "smollm2", "flipped")
    }
    expected = {
        "shohin": ("shohin", False),
        "smollm2": ("smollm2", False),
        "flipped": ("smollm2", True),
    }
    for name, (backbone, flipped) in expected.items():
        report = reports[name]
        if (
            report.get("backbone_name") != backbone
            or bool(report.get("flip_outcomes")) != flipped
            or report.get("board_type") != "development"
        ):
            raise CGL1AssessmentError(f"CGL1 {name} arm identity differs")

    treatment_passes = {
        name: bool(reports[name]["promotion_gate"]["passed"])
        for name in ("shohin", "smollm2")
    }
    flipped_exact = int(reports["flipped"]["normal"]["overall"]["exact"])
    control_pass = flipped_exact <= 430
    candidates = [name for name, passed in treatment_passes.items() if passed]
    selected = None
    if candidates and control_pass:
        selected = max(
            candidates,
            key=lambda name: (
                int(reports[name]["normal"]["overall"]["exact"]),
                float(reports[name]["normal"]["mean_signed_margin"]),
                name == "smollm2",
            ),
        )
    passed = selected is not None
    report = {
        "schema": SCHEMA,
        "status": "pass" if passed else "fail",
        "conditions": {
            "at_least_one_treatment_passes": bool(candidates),
            "flipped_control_at_most_430": control_pass,
        },
        "treatment_passes": treatment_passes,
        "flipped_control_exact": flipped_exact,
        "selected_treatment": selected,
        "arm_receipts": {
            name: {
                "path": str(getattr(args, name)),
                "sha256": getattr(args, f"{name}_sha256"),
                "normal": reports[name]["normal"],
                "promotion_gate": reports[name]["promotion_gate"],
            }
            for name in reports
        },
        "decision": (
            "admit_one_fresh_balanced_confirmation_board"
            if passed
            else "close_cgl1_without_local_variants"
        ),
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "status": report["status"],
                "selected_treatment": selected,
                "treatment_passes": treatment_passes,
                "flipped_control_exact": flipped_exact,
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit("CGL1 development gate failed")


if __name__ == "__main__":
    main()
