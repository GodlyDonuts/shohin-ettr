#!/usr/bin/env python3
"""Score the pinned LiveBench release with its objective ground-truth graders."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from dense_public_official_scoring import (
    STAGES,
    OfficialScoringError,
    load_bound_benchmark,
    official_score_row,
    write_official_scores,
)

BENCHMARK = "livebench"
REPORT_SCHEMA = "shohin-dense-public-livebench-score-v1"


def score(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("SHOHIN_CODE_SANDBOX") != "1":
        raise OfficialScoringError(
            "LiveBench includes generated-code execution; use the frozen sandbox wrapper"
        )
    sys.path.insert(0, str(args.livebench_root))
    from livebench.gen_ground_truth_judgment import play_a_match_gt

    questions, assessors, ledgers = load_bound_benchmark(
        manifest_path=args.manifest,
        generation_root=args.generation_root,
        assessor_root=args.assessor_root,
        assessor_name=args.assessor_name,
        benchmark=BENCHMARK,
    )
    hashes: dict[str, str] = {}
    sums: dict[str, float] = {}
    error_counts: dict[str, int] = {}
    for stage in STAGES:
        rows = []
        errors = 0
        for question, assessor_row, generated in zip(
            questions, assessors, ledgers[stage], strict=True
        ):
            upstream = assessor_row.get("assessor")
            if not isinstance(upstream, dict):
                raise OfficialScoringError("LiveBench assessor schema differs")
            answer = {
                "choices": [{"turns": [generated["completion"]]}],
                "eval_status": "success",
            }
            match = SimpleNamespace(question=upstream, model=f"shohin-{stage}", answer=answer)
            with redirect_stdout(io.StringIO()):
                result = play_a_match_gt(match, output_file=None, debug=False)
            if not isinstance(result, dict):
                raise OfficialScoringError("LiveBench scorer returned no result")
            value = float(result.get("score", -1.0))
            if not 0.0 <= value <= 1.0:
                raise OfficialScoringError("LiveBench score is outside [0, 1]")
            scorer_error = result.get("eval_status") == "eval_error"
            errors += int(scorer_error)
            rows.append(
                official_score_row(
                    stage=stage,
                    identity=question["id"],
                    benchmark=BENCHMARK,
                    metric="official_objective_score",
                    stratum=assessor_row["stratum"],
                    score=value,
                    details={
                        "question_id": upstream["question_id"],
                        "task": result.get("task"),
                        "subtask": result.get("subtask"),
                        "scorer_error": scorer_error,
                        "error_message": result.get("error_msg") if scorer_error else None,
                    },
                )
            )
        hashes[stage] = write_official_scores(
            output_root=args.output_root,
            benchmark=BENCHMARK,
            stage=stage,
            rows=rows,
        )
        sums[stage] = sum(float(row["score"]) for row in rows)
        error_counts[stage] = errors
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "benchmark": BENCHMARK,
        "rows": len(questions),
        "release": args.release,
        "livebench_commit": args.livebench_commit,
        "score_sums": sums,
        "scorer_error_counts": error_counts,
        "official_score_sha256": hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--assessor-root", type=Path, required=True)
    parser.add_argument("--assessor-name", default="full.assessors.jsonl")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--livebench-root", type=Path, required=True)
    parser.add_argument("--livebench-commit", required=True)
    parser.add_argument("--release", required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(score(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
