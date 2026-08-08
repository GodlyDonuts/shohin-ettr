#!/usr/bin/env python3
"""Pair preserved B1/QPT1 evaluation transcripts for CVG1 scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "shohin-cvg1-evaluation-pairs-v1"
EVAL_SCHEMA = "shohin-hf-product-reasoning-eval-v3"
TASKS = ("gsm8k", "math500", "humaneval", "mbpp", "gpqa", "bbh_logic", "aime")
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


class CVG1EvaluationPairError(RuntimeError):
    """Preserved evaluation reports do not define matched whole lineages."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CVG1EvaluationPairError(f"missing evaluation report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != EVAL_SCHEMA or report.get("status") != "complete":
        raise CVG1EvaluationPairError(f"evaluation report is incomplete: {path}")
    results = report.get("results")
    if not isinstance(results, list) or len(results) != report.get("total"):
        raise CVG1EvaluationPairError(f"evaluation results differ: {path}")
    return report


def _index_results(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for result in report["results"]:
        identity = result.get("identity_sha256")
        if not isinstance(identity, str) or len(identity) != 64:
            raise CVG1EvaluationPairError("evaluation identity is invalid")
        if identity in indexed:
            raise CVG1EvaluationPairError("evaluation identity is duplicated")
        completion = result.get("completion")
        if not isinstance(completion, str) or not completion.strip():
            raise CVG1EvaluationPairError("evaluation completion is empty")
        if not isinstance(result.get("correct"), bool):
            raise CVG1EvaluationPairError("evaluation correctness is invalid")
        indexed[identity] = result
    return indexed


def build_pairs(
    base_reports: list[Path], expert_reports: list[Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(base_reports) != len(TASKS) or len(expert_reports) != len(TASKS):
        raise CVG1EvaluationPairError("CVG1 requires seven reports per lineage")
    inputs: dict[str, dict[str, dict[str, str]]] = {"base": {}, "expert": {}}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for expected_task, base_path, expert_path in zip(
        TASKS, base_reports, expert_reports, strict=True
    ):
        base = _load_report(base_path)
        expert = _load_report(expert_path)
        if base.get("task") != expected_task or expert.get("task") != expected_task:
            raise CVG1EvaluationPairError("evaluation task order differs")
        for field in COMPARABILITY_FIELDS:
            if base.get(field) != expert.get(field):
                raise CVG1EvaluationPairError(
                    f"unmatched {expected_task} field {field}"
                )
        base_results = _index_results(base)
        expert_results = _index_results(expert)
        if set(base_results) != set(expert_results):
            raise CVG1EvaluationPairError(
                f"{expected_task} lineage identity coverage differs"
            )
        ordered_identities = [row["identity_sha256"] for row in base["results"]]
        if ordered_identities != [row["identity_sha256"] for row in expert["results"]]:
            raise CVG1EvaluationPairError(
                f"{expected_task} lineage identity order differs"
            )
        for identity in ordered_identities:
            if identity in seen:
                raise CVG1EvaluationPairError("identity repeats across tasks")
            seen.add(identity)
            base_row = base_results[identity]
            expert_row = expert_results[identity]
            if base_row.get("question") != expert_row.get("question"):
                raise CVG1EvaluationPairError("paired questions differ")
            if base_row.get("gold") != expert_row.get("gold"):
                raise CVG1EvaluationPairError("paired gold values differ")
            rows.append(
                {
                    "schema": SCHEMA,
                    "identity_sha256": identity,
                    "task": expected_task,
                    "question": base_row["question"],
                    "candidates": [
                        {
                            "lineage": "base",
                            "completion": base_row["completion"],
                            "correct": base_row["correct"],
                            "prediction": base_row.get("prediction"),
                        },
                        {
                            "lineage": "expert",
                            "completion": expert_row["completion"],
                            "correct": expert_row["correct"],
                            "prediction": expert_row.get("prediction"),
                        },
                    ],
                }
            )
        inputs["base"][expected_task] = {
            "path": str(base_path.resolve()),
            "sha256": sha256_file(base_path),
        }
        inputs["expert"][expected_task] = {
            "path": str(expert_path.resolve()),
            "sha256": sha256_file(expert_path),
        }
    return rows, {
        "schema": SCHEMA,
        "status": "complete",
        "rows": len(rows),
        "tasks": list(TASKS),
        "inference_fields": ["question", "completion"],
        "task_or_benchmark_label_at_inference": False,
        "inputs": inputs,
    }


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise CVG1EvaluationPairError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise CVG1EvaluationPairError(f"refusing existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", action="append", type=Path, required=True)
    parser.add_argument("--expert-report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows, report = build_pairs(args.base_report, args.expert_report)
    report["pairs_sha256"] = _atomic_lines(args.output, rows)
    report["pairs"] = str(args.output.resolve())
    _atomic_json(args.report, report)
    print(json.dumps({"rows": len(rows), "pairs_sha256": report["pairs_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
