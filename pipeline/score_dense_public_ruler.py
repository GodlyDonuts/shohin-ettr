#!/usr/bin/env python3
"""Score RULER campaign generations with NVIDIA's pinned string metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dense_public_official_scoring import (
    STAGES,
    OfficialScoringError,
    load_bound_benchmark,
    official_score_row,
    write_official_scores,
)

BENCHMARK = "ruler"
REPORT_SCHEMA = "shohin-dense-public-ruler-score-v1"


def ruler_score(task: str, prediction: str, references: list[str]) -> float:
    if not references or not all(isinstance(item, str) for item in references):
        raise OfficialScoringError("RULER references differ")
    lowered = prediction.lower()
    matches = [float(reference.lower() in lowered) for reference in references]
    if task.startswith("qa_"):
        return max(matches)
    return sum(matches) / len(matches)


def run(args: argparse.Namespace) -> dict[str, Any]:
    questions, assessors, ledgers = load_bound_benchmark(
        manifest_path=args.manifest,
        generation_root=args.generation_root,
        assessor_root=args.assessor_root,
        assessor_name=args.assessor_name,
        benchmark=BENCHMARK,
    )
    hashes = {}
    stage_means = {}
    for stage in STAGES:
        output = []
        for question, assessor_row, generated in zip(
            questions, assessors, ledgers[stage], strict=True
        ):
            assessor = assessor_row["assessor"]
            task = str(assessor["task"])
            references = assessor["outputs"]
            score = ruler_score(task, generated["completion"], references)
            output.append(
                official_score_row(
                    stage=stage,
                    identity=question["id"],
                    benchmark=BENCHMARK,
                    metric="nvidia_ruler_string_match",
                    stratum=assessor_row["stratum"],
                    score=score,
                    details={"task": task, "references": len(references)},
                )
            )
        hashes[stage] = write_official_scores(
            output_root=args.output_root,
            benchmark=BENCHMARK,
            stage=stage,
            rows=output,
        )
        stage_means[stage] = sum(row["score"] for row in output) / len(output)
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "rows": len(questions),
        "official_source_commit": args.ruler_commit,
        "official_score_sha256": hashes,
        "stage_means": stage_means,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--assessor-root", type=Path, required=True)
    parser.add_argument("--assessor-name", default="full.assessors.jsonl")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ruler-commit", required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
