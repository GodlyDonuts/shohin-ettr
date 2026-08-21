#!/usr/bin/env python3
"""Export and collect pinned EvalPlus scoring for HumanEval+ or MBPP+."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from dense_public_official_scoring import (
    STAGES,
    OfficialScoringError,
    load_bound_benchmark,
    official_score_row,
    write_official_scores,
)

BENCHMARK_DATASET = {
    "humaneval_plus": "humaneval",
    "mbpp_plus": "mbpp",
}
EXPORT_SCHEMA = "shohin-dense-public-evalplus-export-v1"
REPORT_SCHEMA = "shohin-dense-public-evalplus-collect-v1"
TASK_ID_PREFIX = {
    "humaneval_plus": "HumanEval/",
    "mbpp_plus": "Mbpp/",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_python_solution(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def normalize_evalplus_task_id(benchmark: str, raw_task_id: object) -> str:
    """Project frozen assessor IDs into EvalPlus' canonical task namespace."""

    try:
        prefix = TASK_ID_PREFIX[benchmark]
    except KeyError as error:
        raise OfficialScoringError("EvalPlus benchmark differs") from error
    task_id = str(raw_task_id)
    if task_id.startswith(prefix):
        return task_id
    if "/" in task_id or not task_id:
        raise OfficialScoringError("EvalPlus task identity differs")
    return f"{prefix}{task_id}"


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise OfficialScoringError(f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (
                json.dumps(row, ensure_ascii=False, sort_keys=True).encode() + b"\n"
            )
            digest.update(encoded)
            handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def bound(args: argparse.Namespace):
    if args.benchmark not in BENCHMARK_DATASET:
        raise OfficialScoringError("EvalPlus benchmark differs")
    return load_bound_benchmark(
        manifest_path=args.manifest,
        generation_root=args.generation_root,
        assessor_root=args.assessor_root,
        assessor_name=args.assessor_name,
        benchmark=args.benchmark,
    )


def export(args: argparse.Namespace) -> dict[str, Any]:
    questions, assessors, ledgers = bound(args)
    hashes = {}
    for stage in STAGES:
        samples = []
        for question, assessor_row, generated in zip(
            questions, assessors, ledgers[stage], strict=True
        ):
            task_id = normalize_evalplus_task_id(
                args.benchmark, assessor_row["assessor"]["task_id"]
            )
            samples.append(
                {
                    "task_id": task_id,
                    "solution": extract_python_solution(generated["completion"]),
                    "shohin_identity": question["id"],
                }
            )
        # EvalPlus derives `<stem>.eval_results.json` from the sample filename.
        # Keep the stage as the entire stem so collection has one unambiguous path.
        path = args.work_root / args.benchmark / f"{stage}.jsonl"
        hashes[stage] = atomic_jsonl(path, samples)
    report = {
        "schema": EXPORT_SCHEMA,
        "status": "complete",
        "benchmark": args.benchmark,
        "evalplus_dataset": BENCHMARK_DATASET[args.benchmark],
        "rows": len(questions),
        "evalplus_commit": args.evalplus_commit,
        "sample_sha256": hashes,
    }
    report_path = args.work_root / args.benchmark / "export.report.json"
    if report_path.exists():
        raise OfficialScoringError("refusing to replace EvalPlus export report")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def collect(args: argparse.Namespace) -> dict[str, Any]:
    questions, assessors, _ = bound(args)
    hashes = {}
    pass_counts = {}
    for stage in STAGES:
        result_path = args.work_root / args.benchmark / f"{stage}.eval_results.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        evaluated = result.get("eval")
        if not isinstance(evaluated, dict):
            raise OfficialScoringError("EvalPlus result schema differs")
        output = []
        for question, assessor_row in zip(questions, assessors, strict=True):
            task_id = normalize_evalplus_task_id(
                args.benchmark, assessor_row["assessor"]["task_id"]
            )
            task_results = evaluated.get(task_id)
            if not isinstance(task_results, list) or len(task_results) != 1:
                raise OfficialScoringError(
                    f"EvalPlus {task_id} result coverage differs"
                )
            task = task_results[0]
            passed = task.get("base_status") == task.get("plus_status") == "pass"
            output.append(
                official_score_row(
                    stage=stage,
                    identity=question["id"],
                    benchmark=args.benchmark,
                    metric="evalplus_pass_at_1_plus",
                    stratum=assessor_row["stratum"],
                    score=float(passed),
                    details={
                        "task_id": task_id,
                        "base_status": task.get("base_status"),
                        "plus_status": task.get("plus_status"),
                    },
                )
            )
        hashes[stage] = write_official_scores(
            output_root=args.output_root,
            benchmark=args.benchmark,
            stage=stage,
            rows=output,
        )
        pass_counts[stage] = sum(row["score"] for row in output)
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "benchmark": args.benchmark,
        "rows": len(questions),
        "evalplus_commit": args.evalplus_commit,
        "official_score_sha256": hashes,
        "pass_counts": pass_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "collect"))
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_DATASET), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--assessor-root", type=Path, required=True)
    parser.add_argument("--assessor-name", default="full.assessors.jsonl")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--evalplus-commit", required=True)
    args = parser.parse_args()
    if args.mode == "collect" and args.output_root is None:
        parser.error("collect requires --output-root")
    return args


def main() -> int:
    args = parse_args()
    report = export(args) if args.mode == "export" else collect(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
