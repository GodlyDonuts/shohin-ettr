#!/usr/bin/env python3
"""Score pinned LiveCodeBench completions with the official execution harness.

Run this program inside the repository's network-isolated execution sandbox.  The
official harness executes model-generated Python, so an ordinary host invocation
is deliberately rejected unless the caller explicitly acknowledges containment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from dense_public_official_scoring import (
    STAGES,
    OfficialScoringError,
    load_bound_benchmark,
    official_score_row,
    write_official_scores,
)

BENCHMARK = "livecodebench"
REPORT_SCHEMA = "shohin-dense-public-livecodebench-score-v1"


def extract_fenced_code(text: str) -> str:
    """Mirror LiveCodeBench's instructed-code-block extraction deterministically."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.IGNORECASE | re.DOTALL)
    return (blocks[-1] if blocks else text).strip()


def task_passed(result: Any) -> bool:
    """A pass@1 sample passes only when every official public/private test passes."""
    return isinstance(result, list) and bool(result) and all(value is True for value in result)


def score(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("SHOHIN_CODE_SANDBOX") != "1":
        raise OfficialScoringError(
            "LiveCodeBench executes untrusted generated code; use the frozen sandbox wrapper"
        )
    sys.path.insert(0, str(args.livecodebench_root))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    questions, assessors, ledgers = load_bound_benchmark(
        manifest_path=args.manifest,
        generation_root=args.generation_root,
        assessor_root=args.assessor_root,
        assessor_name=args.assessor_name,
        benchmark=BENCHMARK,
    )
    samples = []
    for assessor_row in assessors:
        raw = assessor_row.get("assessor")
        if not isinstance(raw, dict):
            raise OfficialScoringError("LiveCodeBench assessor schema differs")
        samples.append(CodeGenerationProblem(**raw).get_evaluation_sample())

    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for stage in STAGES:
        generations = [[extract_fenced_code(row["completion"])] for row in ledgers[stage]]
        _, results, metadata = codegen_metrics(
            samples,
            generations,
            k_list=[1],
            num_process_evaluate=args.workers,
            timeout=args.timeout,
            debug=False,
        )
        if set(results) != set(range(len(questions))) or len(metadata) != len(questions):
            raise OfficialScoringError(f"{stage} LiveCodeBench result coverage differs")
        rows = []
        for index, (question, assessor_row) in enumerate(
            zip(questions, assessors, strict=True)
        ):
            attempts = results[index]
            if not isinstance(attempts, list) or len(attempts) != 1:
                raise OfficialScoringError(f"{stage} LiveCodeBench attempt coverage differs")
            passed = task_passed(attempts[0])
            rows.append(
                official_score_row(
                    stage=stage,
                    identity=question["id"],
                    benchmark=BENCHMARK,
                    metric="official_pass_at_1",
                    stratum=assessor_row["stratum"],
                    score=float(passed),
                    details={
                        "question_id": assessor_row["assessor"]["question_id"],
                        "tests": len(attempts[0]) if isinstance(attempts[0], list) else 0,
                    },
                )
            )
        hashes[stage] = write_official_scores(
            output_root=args.output_root,
            benchmark=BENCHMARK,
            stage=stage,
            rows=rows,
        )
        counts[stage] = sum(int(row["score"]) for row in rows)
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "benchmark": BENCHMARK,
        "rows": len(questions),
        "livecodebench_commit": args.livecodebench_commit,
        "pass_counts": counts,
        "official_score_sha256": hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--assessor-root", type=Path, required=True)
    parser.add_argument("--assessor-name", default="full.assessors.jsonl")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--livecodebench-root", type=Path, required=True)
    parser.add_argument("--livecodebench-commit", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        parser.error("workers and timeout must be positive")
    return args


def main() -> int:
    report = score(parse_args())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
