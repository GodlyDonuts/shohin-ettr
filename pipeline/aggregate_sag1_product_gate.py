#!/usr/bin/env python3
"""Apply the frozen three-arm SAG1 product-reasoning development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SCHEMA = "shohin-sag1-product-gate-v2"
BASE_CHECKPOINT_SHA256 = (
    "f7354e6a0c4311ad792b73358b4e62d9dbe0ae1bd2d41896cf55482d9ce81feb"
)
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATA_SHA256 = "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549"
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
    "batch_size",
    "data_seed",
    "data_sha256",
    "gradient_accumulation",
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
EXPECTED_TREATMENT_TRACE_UPDATES = (1, *range(8, 257, 8))
TRACE_FINITE_FIELDS = ("loss", "language_loss", "gradient_norm")


class SAG1AggregationError(RuntimeError):
    """The SAG1 reports are incomplete or not a matched comparison."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SAG1AggregationError(f"missing report: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SAG1AggregationError(f"report is not an object: {path}")
    return payload


def _eval_paths(prefix: Path) -> dict[str, Path]:
    return {task: Path(f"{prefix}_{task}.json") for task in TASKS}


def _load_eval(path: Path, expected_task: str) -> dict[str, Any]:
    report = _load_json(path)
    if report.get("status") != "complete" or report.get("task") != expected_task:
        raise SAG1AggregationError(f"invalid evaluation identity: {path}")
    correct, total = report.get("correct"), report.get("total")
    if (
        not isinstance(correct, int)
        or not isinstance(total, int)
        or total <= 0
        or not 0 <= correct <= total
    ):
        raise SAG1AggregationError(f"invalid score in {path}")
    return report


def _finite_training(report: dict[str, Any]) -> bool:
    trace = report.get("trace")
    if not isinstance(trace, list) or not trace:
        return False
    return all(
        isinstance(row, dict)
        and all(
            isinstance(row.get(field), (int, float))
            and math.isfinite(float(row[field]))
            for field in TRACE_FINITE_FIELDS
        )
        for row in trace
    )


def _trace_rows_by_update(
    rows: Any,
    *,
    source: str,
) -> dict[int, dict[str, Any]]:
    if not isinstance(rows, list):
        raise SAG1AggregationError(f"trace is not a list: {source}")
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SAG1AggregationError(f"trace row is not an object: {source}")
        update = row.get("update")
        if not isinstance(update, int) or update <= 0:
            raise SAG1AggregationError(f"trace update is invalid: {source}")
        if update in indexed:
            raise SAG1AggregationError(f"duplicate trace update {update}: {source}")
        required = (*TRACE_FINITE_FIELDS, "router_commit_rate")
        if not all(
            isinstance(row.get(field), (int, float))
            and math.isfinite(float(row[field]))
            for field in required
        ):
            raise SAG1AggregationError(
                f"trace update {update} has a missing/nonfinite metric: {source}"
            )
        indexed[update] = row
    return indexed


def _load_jsonl_trace(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise SAG1AggregationError(f"missing treatment prefix trace log: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise SAG1AggregationError(
                    f"malformed JSON trace row at {path}:{line_number}"
                ) from error
            if isinstance(payload, dict) and isinstance(payload.get("update"), int):
                rows.append(payload)
    return _trace_rows_by_update(rows, source=str(path))


def _complete_treatment_trace(
    report: dict[str, Any],
    prefix_trace_log: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_rows = _trace_rows_by_update(
        report.get("trace"), source="SAG1 treatment report"
    )
    resume_update = report.get("resume_update")
    receipts: dict[str, Any] = {
        "expected_updates": list(EXPECTED_TREATMENT_TRACE_UPDATES),
        "report_segment_updates": sorted(report_rows),
    }
    if resume_update is None:
        if prefix_trace_log is not None:
            raise SAG1AggregationError(
                "treatment prefix trace log supplied for a non-resumed SAG1 report"
            )
        merged = report_rows
        receipts["mode"] = "single_report"
    else:
        if not isinstance(resume_update, int) or resume_update <= 0:
            raise SAG1AggregationError("SAG1 resume_update is invalid")
        if prefix_trace_log is None:
            raise SAG1AggregationError(
                "resumed SAG1 report requires the immutable prefix trace log"
            )
        prefix_rows = _load_jsonl_trace(prefix_trace_log)
        prefix = {
            update: row
            for update, row in prefix_rows.items()
            if update <= resume_update
        }
        suffix = {
            update: row for update, row in report_rows.items() if update > resume_update
        }
        if any(update <= resume_update for update in report_rows):
            raise SAG1AggregationError(
                "resumed SAG1 report unexpectedly contains pre-resume trace rows"
            )
        merged = {**prefix, **suffix}
        receipts.update(
            {
                "mode": "checkpoint_prefix_plus_resumed_report",
                "prefix_log": {
                    "path": str(prefix_trace_log.resolve()),
                    "sha256": _sha256(prefix_trace_log),
                },
                "prefix_updates_used": sorted(prefix),
                "resume_update": resume_update,
                "suffix_updates_used": sorted(suffix),
            }
        )

    observed_updates = tuple(sorted(merged))
    if observed_updates != EXPECTED_TREATMENT_TRACE_UPDATES:
        raise SAG1AggregationError(
            "SAG1 treatment trace coverage differs: "
            f"expected={list(EXPECTED_TREATMENT_TRACE_UPDATES)} "
            f"observed={list(observed_updates)}"
        )
    logical_tokens = [row.get("logical_charged_tokens") for row in merged.values()]
    if all(isinstance(value, int) for value in logical_tokens):
        ordered_tokens = [
            merged[update]["logical_charged_tokens"] for update in observed_updates
        ]
        if any(
            right <= left for left, right in zip(ordered_tokens, ordered_tokens[1:])
        ):
            raise SAG1AggregationError(
                "SAG1 treatment trace logical token accounting is not increasing"
            )
    receipts["merged_updates"] = list(observed_updates)
    return [merged[update] for update in observed_updates], receipts


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


def _comparison(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    domain_deltas = {
        domain: {
            "accuracy_delta": candidate["domains"][domain]["accuracy"]
            - reference["domains"][domain]["accuracy"],
            "solved_delta": candidate["domains"][domain]["correct"]
            - reference["domains"][domain]["correct"],
        }
        for domain in TASKS_BY_DOMAIN
    }
    return {
        "domain_deltas": domain_deltas,
        "improved_domain_count": sum(
            row["solved_delta"] > 0 for row in domain_deltas.values()
        ),
        "macro_delta": candidate["macro_accuracy"] - reference["macro_accuracy"],
        "maximum_domain_regression": max(
            max(0.0, -row["accuracy_delta"]) for row in domain_deltas.values()
        ),
        "solved_delta": candidate["solved"] - reference["solved"],
    }


def aggregate_sag1(
    *,
    original_prefix: Path,
    continuation_prefix: Path,
    treatment_prefix: Path,
    continuation_training: Path,
    treatment_training: Path,
    treatment_prefix_trace_log: Path | None = None,
) -> dict[str, Any]:
    prefixes = {
        "B1_original": original_prefix,
        "B1_continuation": continuation_prefix,
        "SAG1": treatment_prefix,
    }
    eval_reports: dict[str, dict[str, dict[str, Any]]] = {}
    eval_files: dict[str, dict[str, dict[str, str]]] = {}
    for arm, prefix in prefixes.items():
        eval_reports[arm] = {}
        eval_files[arm] = {}
        for task, path in _eval_paths(prefix).items():
            eval_reports[arm][task] = _load_eval(path, task)
            eval_files[arm][task] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }

    for task in TASKS:
        reference = eval_reports["B1_original"][task]
        for arm in ("B1_continuation", "SAG1"):
            candidate = eval_reports[arm][task]
            for field in COMPARABILITY_FIELDS:
                if reference.get(field) != candidate.get(field):
                    raise SAG1AggregationError(
                        f"unmatched {arm} {task} field {field}: "
                        f"{reference.get(field)!r} != {candidate.get(field)!r}"
                    )

    training = {
        "B1_continuation": _load_json(continuation_training),
        "SAG1": _load_json(treatment_training),
    }
    for arm, report in training.items():
        if report.get("status") != "complete":
            raise SAG1AggregationError(f"{arm} training is not complete")
    for field in TRAINING_MATCH_FIELDS:
        if training["B1_continuation"].get(field) != training["SAG1"].get(field):
            raise SAG1AggregationError(f"unmatched training field: {field}")

    treatment_trace, treatment_trace_receipt = _complete_treatment_trace(
        training["SAG1"], treatment_prefix_trace_log
    )
    treatment_for_gates = {**training["SAG1"], "trace": treatment_trace}

    arms = {arm: _summarize(reports) for arm, reports in eval_reports.items()}
    versus_original = _comparison(arms["SAG1"], arms["B1_original"])
    versus_continuation = _comparison(arms["SAG1"], arms["B1_continuation"])
    router_commit_samples = [
        float(row["router_commit_rate"])
        for row in treatment_trace
        if isinstance(row, dict)
        and isinstance(row.get("router_commit_rate"), (int, float))
        and math.isfinite(float(row["router_commit_rate"]))
    ]
    mean_router_commit_rate = (
        sum(router_commit_samples) / len(router_commit_samples)
        if router_commit_samples
        else None
    )

    training_gates = {
        "both_finite": _finite_training(training["B1_continuation"])
        and _finite_training(treatment_for_gates),
        "both_256_updates": all(
            report.get("updates") == 256 for report in training.values()
        ),
        "both_16_examples_per_update": all(
            report.get("batch_size", 0) * report.get("gradient_accumulation", 0) == 16
            for report in training.values()
        ),
        "correct_arms": training["B1_continuation"].get("arm") == "baseline"
        and training["SAG1"].get("arm") == "diverge_sag1",
        "same_logical_target_exposure": training["B1_continuation"].get(
            "charged_tokens"
        )
        == training["SAG1"].get("logical_charged_tokens"),
        "shared_protected_base_checkpoint": training["B1_continuation"].get(
            "warm_start_sha256"
        )
        == BASE_CHECKPOINT_SHA256
        and training["SAG1"].get("base_checkpoint_sha256") == BASE_CHECKPOINT_SHA256,
        "continuation_starts_at_update_256": training["B1_continuation"].get(
            "warm_start_update"
        )
        == 256,
        "frozen_base_unchanged": training["SAG1"].get("frozen_parameters_unchanged")
        is True,
        "frozen_revision_and_data": training["SAG1"].get("model_revision")
        == MODEL_REVISION
        and training["SAG1"].get("data_sha256") == DATA_SHA256,
        "nontrivial_nonuniversal_router": isinstance(
            mean_router_commit_rate, (int, float)
        )
        and 0.05 <= float(mean_router_commit_rate) <= 0.95,
    }
    numeric_gates = {
        "retains_original_b1_code_30_of_40": arms["SAG1"]["domains"]["code"]["correct"]
        >= 30,
        "original_macro_delta_at_least_three_points": versus_original["macro_delta"]
        >= 0.03,
        "original_solved_delta_at_least_fifteen": versus_original["solved_delta"] >= 15,
        "original_improves_at_least_three_domains": versus_original[
            "improved_domain_count"
        ]
        >= 3,
        "continuation_macro_delta_at_least_three_points": versus_continuation[
            "macro_delta"
        ]
        >= 0.03,
        "continuation_solved_delta_at_least_fifteen": versus_continuation[
            "solved_delta"
        ]
        >= 15,
        "continuation_improves_at_least_three_domains": versus_continuation[
            "improved_domain_count"
        ]
        >= 3,
        "continuation_no_domain_regression_over_two_points": versus_continuation[
            "maximum_domain_regression"
        ]
        <= 0.02,
    }
    development_pass = all(training_gates.values()) and all(numeric_gates.values())

    return {
        "arms": arms,
        "comparison": {
            "development_gate_pass": development_pass,
            "mean_logged_router_commit_rate": mean_router_commit_rate,
            "numeric_gates": numeric_gates,
            "router_commit_sample_count": len(router_commit_samples),
            "training_gates": training_gates,
            "versus_b1_continuation": versus_continuation,
            "versus_b1_original": versus_original,
        },
        "eval_files": eval_files,
        "promotion_authorized": False,
        "required_next_step": (
            "source_disjoint_confirmation_and_qwen9b_transplant"
            if development_pass
            else "close_exact_sag1"
        ),
        "schema": SCHEMA,
        "status": "complete",
        "training_files": {
            arm: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for arm, path in (
                ("B1_continuation", continuation_training),
                ("SAG1", treatment_training),
            )
        },
        "treatment_trace_receipt": treatment_trace_receipt,
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
    parser.add_argument("--original-prefix", required=True, type=Path)
    parser.add_argument("--continuation-prefix", required=True, type=Path)
    parser.add_argument("--treatment-prefix", required=True, type=Path)
    parser.add_argument("--continuation-training", required=True, type=Path)
    parser.add_argument("--treatment-training", required=True, type=Path)
    parser.add_argument("--treatment-prefix-trace-log", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = aggregate_sag1(
        original_prefix=args.original_prefix,
        continuation_prefix=args.continuation_prefix,
        treatment_prefix=args.treatment_prefix,
        continuation_training=args.continuation_training,
        treatment_training=args.treatment_training,
        treatment_prefix_trace_log=args.treatment_prefix_trace_log,
    )
    _atomic_write(args.output, report)
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
