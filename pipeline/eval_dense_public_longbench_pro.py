#!/usr/bin/env python3
"""Score the admitted LongBench Pro <=64k subset with the pinned official metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from dense_public_official_scoring import (
    STAGES,
    OfficialScoringError,
    load_bound_benchmark,
    official_score_row,
    write_official_scores,
)

BENCHMARK = "longbench_pro"
REPORT_SCHEMA = "shohin-dense-public-longbench-pro-score-v1"
ADMITTED_LENGTHS = {"8k", "16k", "32k", "64k"}


def score(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(args.longbench_root))
    from modules.evaluation import Evaluator

    questions, assessors, ledgers = load_bound_benchmark(
        manifest_path=args.manifest,
        generation_root=args.generation_root,
        assessor_root=args.assessor_root,
        assessor_name=args.assessor_name,
        benchmark=BENCHMARK,
    )
    if len(questions) != 1000:
        raise OfficialScoringError("LongBench Pro admitted subset cardinality differs")
    strata = {str(row["assessor"].get("token_length")) for row in assessors}
    if not strata <= ADMITTED_LENGTHS or not strata:
        raise OfficialScoringError("LongBench Pro admitted context lengths differ")

    evaluator = Evaluator(
        evaluation_samples_num=len(questions),
        embedding_model_path=str(args.embedding_model),
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
                raise OfficialScoringError("LongBench Pro assessor schema differs")
            ok, value = evaluator.calculate_metric(
                str(upstream["secondary_task"]),
                upstream["answer"],
                generated["completion"],
                str(upstream["language"]) == "Chinese",
            )
            value = float(value)
            if not 0.0 <= value <= 1.0:
                raise OfficialScoringError("LongBench Pro score is outside [0, 1]")
            errors += int(not ok)
            rows.append(
                official_score_row(
                    stage=stage,
                    identity=question["id"],
                    benchmark=BENCHMARK,
                    metric="official_task_metric",
                    stratum=assessor_row["stratum"],
                    score=value,
                    details={
                        "upstream_id": upstream["id"],
                        "primary_task": upstream["primary_task"],
                        "secondary_task": upstream["secondary_task"],
                        "language": upstream["language"],
                        "context_length": upstream["token_length"],
                        "metric_success": bool(ok),
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
        "scope": "official_nonthinking_8k_through_64k_subset_one_greedy_run",
        "rows": len(questions),
        "context_lengths": sorted(strata),
        "longbench_commit": args.longbench_commit,
        "embedding_model": str(args.embedding_model),
        "score_sums": sums,
        "metric_error_counts": error_counts,
        "official_score_sha256": hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--assessor-root", type=Path, required=True)
    parser.add_argument("--assessor-name", default="full.assessors.jsonl")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--longbench-root", type=Path, required=True)
    parser.add_argument("--longbench-commit", required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(score(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
