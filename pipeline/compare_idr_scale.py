#!/usr/bin/env python3
"""Compare trained IDR revision against its unchanged matched second pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "shohin-idr-scale-comparison-v1"
EVAL_SCHEMA = "shohin-idr1-revision-evaluation-v1"
SPLITS = ("development", "holdout")
TASKS = ("math500", "bbh_logic", "mbpp")
EXPECTED_ROWS = {"development": 1_289, "holdout": 1_279}


class IDRScaleComparisonError(RuntimeError):
    """The scale-transfer reports violate the matched decision contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_report(path: Path, split: str) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise IDRScaleComparisonError(f"invalid evaluation report: {path}") from error
    if not isinstance(report, dict):
        raise IDRScaleComparisonError("evaluation report is not an object")
    if report.get("schema") != EVAL_SCHEMA or report.get("status") != "complete":
        raise IDRScaleComparisonError("evaluation report schema/status differs")
    if report.get("split") != split:
        raise IDRScaleComparisonError("evaluation report split differs")
    if report.get("merged_from_shards") is not True:
        raise IDRScaleComparisonError("evaluation was not merged from complete shards")
    if report.get("full_row_count") != EXPECTED_ROWS[split]:
        raise IDRScaleComparisonError("evaluation row count differs")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or not {"overall", *TASKS} <= set(metrics):
        raise IDRScaleComparisonError("evaluation metrics are incomplete")
    for task in ("overall", *TASKS):
        metric = metrics[task]
        if (
            not isinstance(metric, dict)
            or not isinstance(metric.get("total"), int)
            or not isinstance(metric.get("generated_correct"), int)
            or metric["total"] <= 0
            or not 0 <= metric["generated_correct"] <= metric["total"]
        ):
            raise IDRScaleComparisonError(f"evaluation metric is invalid: {task}")
    return report


def compare_split(
    trained: dict[str, Any], control: dict[str, Any], split: str
) -> dict[str, Any]:
    shared = (
        "split",
        "model_root",
        "model_revision",
        "data_sha256",
        "data_report_sha256",
        "generation_mode",
        "max_new_tokens",
        "batch_size",
        "seed",
        "full_row_count",
    )
    if any(trained.get(key) != control.get(key) for key in shared):
        raise IDRScaleComparisonError(f"{split} matched settings differ")
    if trained.get("adapter_checkpoint_sha256") == control.get(
        "adapter_checkpoint_sha256"
    ):
        raise IDRScaleComparisonError(f"{split} treatment/control adapters are identical")

    deltas = {}
    for task in ("overall", *TASKS):
        trained_metric = trained["metrics"][task]
        control_metric = control["metrics"][task]
        if trained_metric["total"] != control_metric["total"]:
            raise IDRScaleComparisonError(f"{split} {task} denominator differs")
        correct = (
            trained_metric["generated_correct"]
            - control_metric["generated_correct"]
        )
        deltas[task] = {
            "correct": correct,
            "accuracy": correct / trained_metric["total"],
            "trained_correct": trained_metric["generated_correct"],
            "control_correct": control_metric["generated_correct"],
            "total": trained_metric["total"],
        }
    gates = {
        "overall_gain_at_least_0_05": deltas["overall"]["accuracy"] >= 0.05,
        "math_delta_nonnegative": deltas["math500"]["correct"] >= 0,
        "logic_science_delta_nonnegative": deltas["bbh_logic"]["correct"] >= 0,
        "code_delta_nonnegative": deltas["mbpp"]["correct"] >= 0,
    }
    return {"deltas": deltas, "gates": gates, "gate_pass": all(gates.values())}


def compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise IDRScaleComparisonError(f"refusing existing output: {args.output}")
    inputs = {
        "development": {
            "trained": args.trained_development,
            "control": args.control_development,
        },
        "holdout": {
            "trained": args.trained_holdout,
            "control": args.control_holdout,
        },
    }
    results = {}
    bindings = {}
    model_revisions = set()
    for split in SPLITS:
        trained_path = inputs[split]["trained"]
        control_path = inputs[split]["control"]
        trained = load_report(trained_path, split)
        control = load_report(control_path, split)
        results[split] = compare_split(trained, control, split)
        bindings[split] = {
            "trained_report": str(trained_path.resolve()),
            "trained_report_sha256": sha256_file(trained_path),
            "trained_adapter_sha256": trained["adapter_checkpoint_sha256"],
            "control_report": str(control_path.resolve()),
            "control_report_sha256": sha256_file(control_path),
            "control_adapter_sha256": control["adapter_checkpoint_sha256"],
            "data_sha256": trained["data_sha256"],
        }
        model_revisions.update((trained["model_revision"], control["model_revision"]))
    if len(model_revisions) != 1:
        raise IDRScaleComparisonError("development/holdout model revisions differ")

    gate_pass = all(results[split]["gate_pass"] for split in SPLITS)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_revision": next(iter(model_revisions)),
        "bindings": bindings,
        "splits": results,
        "gate_pass": gate_pass,
        "scratch_scale_decision": "shohin_390m" if gate_pass else "shohin_920m",
        "decision_rule": (
            "select shohin_390m only when trained revision gains at least five "
            "absolute points overall and has nonnegative math, logic/science, "
            "and code deltas on both development and holdout"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-development", type=Path, required=True)
    parser.add_argument("--control-development", type=Path, required=True)
    parser.add_argument("--trained-holdout", type=Path, required=True)
    parser.add_argument("--control-holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = compare(parser.parse_args())
    print(
        json.dumps(
            {
                "gate_pass": report["gate_pass"],
                "scratch_scale_decision": report["scratch_scale_decision"],
                "splits": report["splits"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
