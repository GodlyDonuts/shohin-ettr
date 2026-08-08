#!/usr/bin/env python3
"""Apply the frozen TTR1 development gate to complete matched reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-ttr1-development-comparison-v1"
DOMAINS = ("math500", "bbh_logic", "mbpp")
CONTROLS = (
    "unchanged_second_pass",
    "self_refinement",
    "long_single_generation",
    "best_of_two",
    "independent_commitment",
)


class TTR1ComparisonError(RuntimeError):
    """A TTR1 report set is incomplete or not causally matched."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise TTR1ComparisonError(f"incomplete TTR1 report: {path}")
    return value


def accuracy(report: dict[str, Any], domain: str = "overall") -> float:
    metrics = report.get("metrics", {}).get(domain)
    if not isinstance(metrics, dict) or metrics.get("total", 0) <= 0:
        raise TTR1ComparisonError(f"missing TTR1 metrics: {domain}")
    return float(metrics["generated_correct"]) / int(metrics["total"])


def correct(report: dict[str, Any], domain: str) -> int:
    metrics = report.get("metrics", {}).get(domain)
    if not isinstance(metrics, dict):
        raise TTR1ComparisonError(f"missing TTR1 domain: {domain}")
    return int(metrics["generated_correct"])


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing comparison: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    treatment = load_report(args.treatment)
    control_paths = dict(zip(CONTROLS, args.control_report, strict=True))
    controls = {name: load_report(path) for name, path in control_paths.items()}
    if treatment.get("schema") != "shohin-idr1-revision-evaluation-v1":
        raise TTR1ComparisonError("treatment report schema differs")
    shared = ("split", "model_revision", "data_sha256", "data_report_sha256")
    if treatment.get("split") != "development":
        raise TTR1ComparisonError("TTR1 development comparison opened another split")
    for name, report in controls.items():
        if (
            report.get("schema") != "shohin-ttr1-control-evaluation-v1"
            or report.get("control") != name
            or any(report.get(key) != treatment.get(key) for key in shared)
            or report.get("shard_count") != 1
            or report.get("full_row_count") != 1289
        ):
            raise TTR1ComparisonError(f"TTR1 control is not matched: {name}")
    treatment_accuracy = accuracy(treatment)
    control_accuracy = {name: accuracy(report) for name, report in controls.items()}
    strongest_name = max(control_accuracy, key=control_accuracy.get)
    unchanged = controls["unchanged_second_pass"]
    domain_deltas = {
        domain: correct(treatment, domain) - correct(unchanged, domain)
        for domain in DOMAINS
    }
    gates = {
        "treatment_beats_unchanged_by_5_points": treatment_accuracy
        >= control_accuracy["unchanged_second_pass"] + 0.05,
        "all_domain_deltas_nonnegative": all(
            delta >= 0 for delta in domain_deltas.values()
        ),
        "treatment_beats_strongest_control_by_3_points": treatment_accuracy
        >= control_accuracy[strongest_name] + 0.03,
        "complete_identity_coverage": all(
            report.get("full_row_count") == 1289
            for report in (treatment, *controls.values())
        ),
    }
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "split": "development",
        "model_revision": treatment["model_revision"],
        "data_sha256": treatment["data_sha256"],
        "data_report_sha256": treatment["data_report_sha256"],
        "treatment": {
            "path": str(args.treatment.resolve()),
            "sha256": sha256_file(args.treatment),
            "accuracy": treatment_accuracy,
        },
        "controls": {
            name: {
                "path": str(control_paths[name].resolve()),
                "sha256": sha256_file(control_paths[name]),
                "accuracy": control_accuracy[name],
            }
            for name in CONTROLS
        },
        "strongest_control": strongest_name,
        "treatment_minus_unchanged_points": 100
        * (treatment_accuracy - control_accuracy["unchanged_second_pass"]),
        "treatment_minus_strongest_points": 100
        * (treatment_accuracy - control_accuracy[strongest_name]),
        "domain_correct_count_deltas_vs_unchanged": domain_deltas,
        "gates": gates,
        "gate_pass": all(gates.values()),
        "holdout_authorized": all(gates.values()),
    }
    atomic_json(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument(
        "--control-report", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.control_report) != len(CONTROLS):
        parser.error(f"exactly {len(CONTROLS)} ordered control reports are required")
    result = compare(args)
    print(json.dumps({"gate_pass": result["gate_pass"], "gates": result["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
