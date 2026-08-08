#!/usr/bin/env python3
"""Score a trained internal-draft reviser against its matched product control."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

from build_aqc1_product import MERGED_SCHEMA, load_bound, sha256_file

REPORT_SCHEMA = "shohin-idr-product-comparison-v1"
TASKS = ("aime", "bbh_logic", "gpqa", "gsm8k", "humaneval", "math500", "mbpp")


class ProductComparisonError(RuntimeError):
    """The matched product-comparison contract was violated."""


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {task: Counter(total=0, correct=0) for task in TASKS}
    for row in rows:
        task = str(row.get("task"))
        if task not in counters:
            raise ProductComparisonError(f"unexpected product task: {task}")
        counters[task]["total"] += 1
        counters[task]["correct"] += int(bool(row.get("correct")))
    tasks = {
        task: {
            "correct": counter["correct"],
            "total": counter["total"],
            "accuracy": counter["correct"] / counter["total"],
        }
        for task, counter in counters.items()
    }
    domains = {
        "grade_school_math": dict(tasks["gsm8k"]),
        "competition_math": dict(tasks["math500"]),
        "science": dict(tasks["gpqa"]),
        "logic": dict(tasks["bbh_logic"]),
        "code": {
            "correct": tasks["humaneval"]["correct"] + tasks["mbpp"]["correct"],
            "total": tasks["humaneval"]["total"] + tasks["mbpp"]["total"],
        },
    }
    domains["code"]["accuracy"] = domains["code"]["correct"] / domains["code"]["total"]
    solved = sum(int(value["correct"]) for value in domains.values())
    total = sum(int(value["total"]) for value in domains.values())
    return {
        "tasks": tasks,
        "domains": domains,
        "aime": dict(tasks["aime"]),
        "solved": solved,
        "total": total,
        "accuracy": solved / total,
        "macro_accuracy": sum(float(value["accuracy"]) for value in domains.values())
        / len(domains),
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.report.exists():
        raise ProductComparisonError(f"refusing existing report: {args.report}")
    source = load_bound(args.source, args.source_report, "source")
    treatment = load_bound(args.treatment, args.treatment_report, args.treatment_stage)
    control = load_bound(args.control, args.control_report, args.control_stage)
    expected = [row["identity_sha256"] for row in source]
    if len(expected) != 568 or len(set(expected)) != 568:
        raise ProductComparisonError("protected product source coverage differs")
    for name, rows in (("treatment", treatment), ("control", control)):
        if len(rows) != len(expected):
            raise ProductComparisonError(f"{name} row count differs")
        if [row.get("identity_sha256") for row in rows] != expected:
            raise ProductComparisonError(f"{name} identity order differs")
        if any(row.get("schema") != MERGED_SCHEMA for row in rows):
            raise ProductComparisonError(f"{name} schema differs")
    treatment_summary = summarize(treatment)
    control_summary = summarize(control)
    domain_deltas = {
        domain: {
            "correct": treatment_summary["domains"][domain]["correct"]
            - control_summary["domains"][domain]["correct"],
            "accuracy": treatment_summary["domains"][domain]["accuracy"]
            - control_summary["domains"][domain]["accuracy"],
        }
        for domain in treatment_summary["domains"]
    }
    deltas = {
        "solved": treatment_summary["solved"] - control_summary["solved"],
        "accuracy": treatment_summary["accuracy"] - control_summary["accuracy"],
        "macro_accuracy": treatment_summary["macro_accuracy"]
        - control_summary["macro_accuracy"],
        "aime_correct": treatment_summary["aime"]["correct"]
        - control_summary["aime"]["correct"],
        "domains": domain_deltas,
    }
    gates = {
        "at_least_27_additional_main_answers": deltas["solved"] >= 27,
        "at_least_0_05_macro_gain": deltas["macro_accuracy"] >= 0.05,
        "all_five_domain_deltas_nonnegative": all(
            value["correct"] >= 0 for value in domain_deltas.values()
        ),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "source_report_sha256": sha256_file(args.source_report),
        "treatment": str(args.treatment.resolve()),
        "treatment_sha256": sha256_file(args.treatment),
        "treatment_report_sha256": sha256_file(args.treatment_report),
        "control": str(args.control.resolve()),
        "control_sha256": sha256_file(args.control),
        "control_report_sha256": sha256_file(args.control_report),
        "treatment_summary": treatment_summary,
        "control_summary": control_summary,
        "deltas": deltas,
        "gates": gates,
        "gate_pass": all(gates.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--treatment-report", type=Path, required=True)
    parser.add_argument("--treatment-stage", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--control-stage", required=True)
    parser.add_argument("--report", type=Path, required=True)
    report = compare(parser.parse_args())
    print(
        json.dumps(
            {"gate_pass": report["gate_pass"], "deltas": report["deltas"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
