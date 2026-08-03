#!/usr/bin/env python3
"""Aggregate matched product-reasoning reports into one promotion decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-product-reasoning-campaign-v1"
TASKS_BY_DOMAIN = {
    "grade_school_math": ("gsm8k",),
    "competition_math": ("math500",),
    "code": ("humaneval", "mbpp"),
    "science": ("gpqa",),
    "logic": ("bbh_logic",),
}
TASKS = tuple(task for tasks in TASKS_BY_DOMAIN.values() for task in tasks)
COMPARABILITY_FIELDS = (
    "task",
    "data_sha256",
    "selection_sha256",
    "generation_mode",
    "generation_seed",
    "max_new_tokens",
    "subset_seed",
    "effective_enable_thinking",
    "total",
)


class CampaignAggregationError(RuntimeError):
    """Raised when reports are incomplete or not a matched comparison."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(path: Path, expected_task: str) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignAggregationError(f"missing report: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "complete":
        raise CampaignAggregationError(f"report is not complete: {path}")
    if report.get("task") != expected_task:
        raise CampaignAggregationError(
            f"task mismatch for {path}: {report.get('task')!r} != {expected_task!r}"
        )
    correct = report.get("correct")
    total = report.get("total")
    if not isinstance(correct, int) or not isinstance(total, int):
        raise CampaignAggregationError(f"non-integer score in {path}")
    if total <= 0 or correct < 0 or correct > total:
        raise CampaignAggregationError(f"invalid score {correct}/{total} in {path}")
    return report


def _arm_report_paths(prefix: Path, suffix: str) -> dict[str, Path]:
    return {task: Path(f"{prefix}_{task}{suffix}") for task in TASKS}


def _validate_comparability(reports: dict[str, dict[str, dict[str, Any]]]) -> None:
    arm_names = tuple(reports)
    baseline = reports[arm_names[0]]
    for task in TASKS:
        reference = baseline[task]
        for arm_name in arm_names[1:]:
            candidate = reports[arm_name][task]
            for field in COMPARABILITY_FIELDS:
                if candidate.get(field) != reference.get(field):
                    raise CampaignAggregationError(
                        f"unmatched {task} field {field}: "
                        f"{arm_names[0]}={reference.get(field)!r}, "
                        f"{arm_name}={candidate.get(field)!r}"
                    )


def _summarize_arm(task_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    domains: dict[str, dict[str, Any]] = {}
    for domain, tasks in TASKS_BY_DOMAIN.items():
        correct = sum(task_reports[task]["correct"] for task in tasks)
        total = sum(task_reports[task]["total"] for task in tasks)
        domains[domain] = {
            "accuracy": correct / total,
            "correct": correct,
            "tasks": list(tasks),
            "total": total,
        }
    macro = sum(entry["accuracy"] for entry in domains.values()) / len(domains)
    return {
        "domains": domains,
        "macro_accuracy": macro,
        "solved": sum(report["correct"] for report in task_reports.values()),
        "total": sum(report["total"] for report in task_reports.values()),
    }


def _relative_solved_gate(baseline_solved: int, treatment_solved: int) -> bool:
    if baseline_solved == 0:
        return treatment_solved > 0
    return (treatment_solved - baseline_solved) / baseline_solved >= 0.10


def aggregate_campaign(
    *,
    baseline_name: str,
    baseline_prefix: Path,
    treatment_name: str,
    treatment_prefix: Path,
    control_name: str,
    control_prefix: Path,
    suffix: str = "_dev_v2.json",
) -> dict[str, Any]:
    arm_prefixes = {
        baseline_name: baseline_prefix,
        treatment_name: treatment_prefix,
        control_name: control_prefix,
    }
    if len(arm_prefixes) != 3:
        raise CampaignAggregationError("arm names must be distinct")

    reports: dict[str, dict[str, dict[str, Any]]] = {}
    report_files: dict[str, dict[str, dict[str, str]]] = {}
    for arm_name, prefix in arm_prefixes.items():
        reports[arm_name] = {}
        report_files[arm_name] = {}
        for task, path in _arm_report_paths(prefix, suffix).items():
            reports[arm_name][task] = _load_report(path, task)
            report_files[arm_name][task] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }

    _validate_comparability(reports)
    arms = {name: _summarize_arm(value) for name, value in reports.items()}
    baseline = arms[baseline_name]
    treatment = arms[treatment_name]
    control = arms[control_name]

    domain_deltas = {
        domain: {
            "accuracy_delta": treatment["domains"][domain]["accuracy"]
            - baseline["domains"][domain]["accuracy"],
            "solved_delta": treatment["domains"][domain]["correct"]
            - baseline["domains"][domain]["correct"],
        }
        for domain in TASKS_BY_DOMAIN
    }
    macro_delta = treatment["macro_accuracy"] - baseline["macro_accuracy"]
    macro_or_relative = macro_delta >= 0.03 or (
        baseline["macro_accuracy"] < 0.30
        and _relative_solved_gate(baseline["solved"], treatment["solved"])
    )
    improved_domains = sum(
        delta["solved_delta"] > 0 for delta in domain_deltas.values()
    )
    maximum_regression = max(
        (max(0.0, -delta["accuracy_delta"]) for delta in domain_deltas.values()),
        default=0.0,
    )
    treatment_beats_control = (
        treatment["macro_accuracy"] > control["macro_accuracy"]
        and treatment["solved"] >= control["solved"]
    )
    numeric_gates = {
        "macro_or_relative_solved": macro_or_relative,
        "improves_at_least_three_domains": improved_domains >= 3,
        "no_domain_regression_over_two_points": maximum_regression <= 0.02,
        "treatment_beats_dense_control": treatment_beats_control,
    }

    return {
        "arms": arms,
        "comparison": {
            "baseline": baseline_name,
            "control": control_name,
            "domain_deltas_treatment_vs_baseline": domain_deltas,
            "improved_domain_count": improved_domains,
            "macro_delta_treatment_vs_baseline": macro_delta,
            "maximum_domain_regression": maximum_regression,
            "numeric_gates": numeric_gates,
            "numeric_gate_pass": all(numeric_gates.values()),
            "treatment": treatment_name,
            "transcript_coherence_gate": "manual_review_required",
        },
        "report_files": report_files,
        "schema": SCHEMA,
        "status": "complete",
    }


def _parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must be NAME=REPORT_PREFIX")
    name, prefix = value.split("=", 1)
    if not name or not prefix:
        raise argparse.ArgumentTypeError("arm must be NAME=REPORT_PREFIX")
    return name, Path(prefix)


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=_parse_arm)
    parser.add_argument("--treatment", required=True, type=_parse_arm)
    parser.add_argument("--control", required=True, type=_parse_arm)
    parser.add_argument("--suffix", default="_dev_v2.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline_name, baseline_prefix = args.baseline
    treatment_name, treatment_prefix = args.treatment
    control_name, control_prefix = args.control
    report = aggregate_campaign(
        baseline_name=baseline_name,
        baseline_prefix=baseline_prefix,
        treatment_name=treatment_name,
        treatment_prefix=treatment_prefix,
        control_name=control_name,
        control_prefix=control_prefix,
        suffix=args.suffix,
    )
    _write_atomic(args.output, report)
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
