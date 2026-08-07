#!/usr/bin/env python3
"""Apply the frozen two-arm QPT1 product-reasoning promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-qpt1-product-gate-v1"
TASKS_BY_DOMAIN = {
    "grade_school_math": ("gsm8k",),
    "competition_math": ("math500",),
    "code": ("humaneval", "mbpp"),
    "science": ("gpqa",),
    "logic": ("bbh_logic",),
}
MAIN_TASKS = tuple(task for tasks in TASKS_BY_DOMAIN.values() for task in tasks)
TASKS = (*MAIN_TASKS, "aime")
COMPARABILITY_FIELDS = (
    "task",
    "data_sha256",
    "selection_sha256",
    "generation_mode",
    "generation_seed",
    "max_new_tokens",
    "generation_stop_token_ids",
    "subset_seed",
    "effective_enable_thinking",
    "total",
)
TRAINING_MATCH_FIELDS = (
    "data_sha256",
    "data_seed",
    "learning_rate",
    "lora_alpha",
    "lora_layers",
    "lora_rank",
    "max_sequence_length",
    "model_revision",
    "seed",
    "selected_rows",
    "updates",
)


class QPT1AggregationError(RuntimeError):
    """The QPT1 reports are incomplete or not a matched comparison."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise QPT1AggregationError(f"missing report: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise QPT1AggregationError(f"report is not an object: {path}")
    return payload


def _load_eval(path: Path, expected_task: str) -> dict[str, Any]:
    report = _load_json(path)
    if report.get("status") != "complete":
        raise QPT1AggregationError(f"evaluation is not complete: {path}")
    if report.get("task") != expected_task:
        raise QPT1AggregationError(f"task mismatch: {path}")
    correct, total = report.get("correct"), report.get("total")
    if (
        not isinstance(correct, int)
        or not isinstance(total, int)
        or total <= 0
        or not 0 <= correct <= total
    ):
        raise QPT1AggregationError(f"invalid score in {path}")
    return report


def _eval_paths(prefix: Path) -> dict[str, Path]:
    return {task: Path(f"{prefix}_{task}.json") for task in TASKS}


def _finite_training(report: dict[str, Any]) -> bool:
    trace = report.get("trace")
    if not isinstance(trace, list) or not trace:
        return False
    numeric_fields = ("loss", "language_loss", "gradient_norm")
    return all(
        isinstance(row, dict)
        and all(
            isinstance(row.get(field), (int, float))
            and math.isfinite(float(row[field]))
            for field in numeric_fields
        )
        for row in trace
    )


def _summarize(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    domains: dict[str, dict[str, Any]] = {}
    for domain, tasks in TASKS_BY_DOMAIN.items():
        correct = sum(reports[task]["correct"] for task in tasks)
        total = sum(reports[task]["total"] for task in tasks)
        domains[domain] = {
            "accuracy": correct / total,
            "correct": correct,
            "tasks": list(tasks),
            "total": total,
        }
    return {
        "aime": {
            "accuracy": reports["aime"]["correct"] / reports["aime"]["total"],
            "correct": reports["aime"]["correct"],
            "total": reports["aime"]["total"],
        },
        "domains": domains,
        "macro_accuracy": sum(row["accuracy"] for row in domains.values())
        / len(domains),
        "solved": sum(reports[task]["correct"] for task in MAIN_TASKS),
        "total": sum(reports[task]["total"] for task in MAIN_TASKS),
    }


def aggregate_qpt1(
    *,
    baseline_prefix: Path,
    treatment_prefix: Path,
    baseline_training: Path,
    treatment_training: Path,
) -> dict[str, Any]:
    eval_reports: dict[str, dict[str, dict[str, Any]]] = {}
    eval_files: dict[str, dict[str, dict[str, str]]] = {}
    for arm, prefix in (("B1", baseline_prefix), ("QPT1", treatment_prefix)):
        eval_reports[arm] = {}
        eval_files[arm] = {}
        for task, path in _eval_paths(prefix).items():
            eval_reports[arm][task] = _load_eval(path, task)
            eval_files[arm][task] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }

    for task in TASKS:
        baseline = eval_reports["B1"][task]
        treatment = eval_reports["QPT1"][task]
        for field in COMPARABILITY_FIELDS:
            if baseline.get(field) != treatment.get(field):
                raise QPT1AggregationError(
                    f"unmatched {task} field {field}: "
                    f"{baseline.get(field)!r} != {treatment.get(field)!r}"
                )

    training = {
        "B1": _load_json(baseline_training),
        "QPT1": _load_json(treatment_training),
    }
    for arm, report in training.items():
        if report.get("status") != "complete":
            raise QPT1AggregationError(f"{arm} training is not complete")
    for field in TRAINING_MATCH_FIELDS:
        if training["B1"].get(field) != training["QPT1"].get(field):
            raise QPT1AggregationError(f"unmatched training field: {field}")

    arms = {arm: _summarize(reports) for arm, reports in eval_reports.items()}
    baseline = arms["B1"]
    treatment = arms["QPT1"]
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
    solved_delta = treatment["solved"] - baseline["solved"]
    improved_domains = sum(
        delta["solved_delta"] > 0 for delta in domain_deltas.values()
    )
    maximum_regression = max(
        max(0.0, -delta["accuracy_delta"]) for delta in domain_deltas.values()
    )

    training_gates = {
        "both_finite": _finite_training(training["B1"])
        and _finite_training(training["QPT1"]),
        "both_256_updates": training["B1"].get("updates") == 256,
        "both_16_examples_per_update": (
            training["B1"].get("batch_size", 0)
            * training["B1"].get("gradient_accumulation", 0)
            == 16
            and training["QPT1"].get("batch_size", 0)
            * training["QPT1"].get("gradient_accumulation", 0)
            == 16
        ),
        "correct_arms": training["B1"].get("arm") == "baseline"
        and training["QPT1"].get("arm") == "diverge_qpt1",
        "qpt1_frozen_weights_unchanged": training["QPT1"].get(
            "frozen_parameters_unchanged"
        )
        is True,
    }
    numeric_gates = {
        "macro_delta_at_least_three_points": macro_delta >= 0.03,
        "solved_delta_at_least_fifteen": solved_delta >= 15,
        "improves_at_least_three_domains": improved_domains >= 3,
        "no_domain_regression_over_two_points": maximum_regression <= 0.02,
    }
    score_pass = all(numeric_gates.values()) and all(training_gates.values())

    return {
        "arms": arms,
        "comparison": {
            "domain_deltas_qpt1_vs_b1": domain_deltas,
            "improved_domain_count": improved_domains,
            "macro_delta_qpt1_vs_b1": macro_delta,
            "maximum_domain_regression": maximum_regression,
            "numeric_gates": numeric_gates,
            "score_and_training_gate_pass": score_pass,
            "solved_delta_qpt1_vs_b1": solved_delta,
            "training_gates": training_gates,
        },
        "eval_files": eval_files,
        "promotion_authorized": False,
        "required_next_step": (
            "packet_swap_state_reset_release_off_controls"
            if score_pass
            else "close_exact_qpt1"
        ),
        "schema": SCHEMA,
        "status": "complete",
        "training_files": {
            "B1": {
                "path": str(baseline_training.resolve()),
                "sha256": _sha256(baseline_training),
            },
            "QPT1": {
                "path": str(treatment_training.resolve()),
                "sha256": _sha256(treatment_training),
            },
        },
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-prefix", required=True, type=Path)
    parser.add_argument("--treatment-prefix", required=True, type=Path)
    parser.add_argument("--baseline-training", required=True, type=Path)
    parser.add_argument("--treatment-training", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = aggregate_qpt1(
        baseline_prefix=args.baseline_prefix,
        treatment_prefix=args.treatment_prefix,
        baseline_training=args.baseline_training,
        treatment_training=args.treatment_training,
    )
    _atomic_write(args.output, report)
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
