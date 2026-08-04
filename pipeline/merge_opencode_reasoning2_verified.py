#!/usr/bin/env python3
"""Merge independently replayed OCR2 rows into one auditable code corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import Any

from verify_opencode_reasoning2_candidates import REPORT_SCHEMA, SCHEMA, sha256_file


MERGE_SCHEMA = "shohin-opencode-reasoning2-execution-merge-v1"
EXPECTED_DATASETS = frozenset({"apps", "taco", "code_contests"})
WORD = re.compile(r"\w+")


class OpenCodeReasoningMergeError(RuntimeError):
    """Verified shards cannot be combined without weakening their contract."""


def question_key(value: Any) -> str:
    return " ".join(WORD.findall(str(value).casefold()))


def row_order(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row.get("response_chars") or len(str(row.get("response") or ""))),
        int(row.get("solution_chars") or len(str(row.get("solution") or ""))),
        str(row["identity_sha256"]),
    )


def read_bound_inputs(
    inputs: list[Path], reports: list[Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(inputs) != len(reports):
        raise OpenCodeReasoningMergeError("inputs and reports must have equal length")
    selected: dict[str, dict[str, Any]] = {}
    by_question: dict[str, str] = {}
    datasets: set[str] = set()
    counters: Counter[str] = Counter()
    receipts: list[dict[str, Any]] = []
    for input_path, report_path in zip(inputs, reports):
        if not input_path.is_file() or not report_path.is_file():
            raise OpenCodeReasoningMergeError("verified input or report is missing")
        report = json.loads(report_path.read_text())
        if report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete":
            raise OpenCodeReasoningMergeError("verification report is not complete")
        dataset = str(report.get("dataset") or "")
        if dataset not in EXPECTED_DATASETS or dataset in datasets:
            raise OpenCodeReasoningMergeError("dataset report set is invalid")
        datasets.add(dataset)
        input_sha = sha256_file(input_path)
        if report.get("output_sha256") != input_sha:
            raise OpenCodeReasoningMergeError("verified input hash differs from report")
        report_rows = 0
        with input_path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                report_rows += 1
                counters["input_rows"] += 1
                if row.get("schema") != SCHEMA:
                    raise OpenCodeReasoningMergeError("verified row schema differs")
                if row.get("source_dataset") != dataset:
                    raise OpenCodeReasoningMergeError("row dataset differs from report")
                if row.get("verification") != "execution_verified_source_tests":
                    raise OpenCodeReasoningMergeError("row is not execution verified")
                identity = str(row.get("identity_sha256") or "")
                question = str(row.get("question") or "").strip()
                response = str(row.get("response") or "").strip()
                solution = str(row.get("solution") or "").strip()
                if (
                    len(identity) != 64
                    or not question
                    or not response
                    or not solution
                    or int(row.get("verified_cases") or 0) <= 0
                ):
                    raise OpenCodeReasoningMergeError("verified row is incomplete")
                if identity in selected:
                    raise OpenCodeReasoningMergeError("duplicate verified identity")
                normalized = question_key(question)
                if not normalized:
                    raise OpenCodeReasoningMergeError("question normalizes empty")
                previous_identity = by_question.get(normalized)
                if previous_identity is None:
                    selected[identity] = row
                    by_question[normalized] = identity
                    continue
                counters["duplicate_questions"] += 1
                previous = selected[previous_identity]
                if row_order(row) < row_order(previous):
                    del selected[previous_identity]
                    selected[identity] = row
                    by_question[normalized] = identity
                    counters["duplicate_question_replacements"] += 1
        expected_rows = int(report.get("counters", {}).get("kept") or 0)
        if report_rows != expected_rows:
            raise OpenCodeReasoningMergeError("report kept count differs from input")
        receipts.append(
            {
                "dataset": dataset,
                "input": str(input_path.resolve()),
                "input_sha256": input_sha,
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "source_revision": report["source_revision"],
                "rows": report_rows,
            }
        )
    if datasets != EXPECTED_DATASETS:
        raise OpenCodeReasoningMergeError("all three source datasets are required")
    rows = sorted(selected.values(), key=lambda row: row["identity_sha256"])
    counters["selected_rows"] = len(rows)
    counters["verified_cases"] = sum(int(row["verified_cases"]) for row in rows)
    counters["response_chars"] = sum(len(str(row["response"])) for row in rows)
    counters["solution_chars"] = sum(len(str(row["solution"])) for row in rows)
    return rows, {"counters": dict(sorted(counters.items())), "inputs": receipts}


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise OpenCodeReasoningMergeError(f"refusing to replace {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise OpenCodeReasoningMergeError(f"stale partial exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs=3, type=Path, required=True)
    parser.add_argument("--reports", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise OpenCodeReasoningMergeError("merge output already exists")
    rows, evidence = read_bound_inputs(args.inputs, args.reports)
    atomic_jsonl(args.output, rows)
    report = {
        "schema": MERGE_SCHEMA,
        "status": "complete",
        **evidence,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    partial = args.report.with_suffix(args.report.suffix + ".partial")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, args.report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
