#!/usr/bin/env python3
"""Score matched campaign ledgers with official deterministic benchmark logic."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from score_dense_public_benchmark import ifeval_scorer, mmlu_answer, musr_answer

ASSESSOR_SCHEMA = "shohin-dense-public-benchmark-assessor-v1"
LEDGER_SCHEMA = "shohin-dense-public-campaign-ledger-v1"
REPORT_SCHEMA = "shohin-dense-public-campaign-score-v1"


class CampaignScoreError(RuntimeError):
    """Campaign coverage, identity, or scorer binding differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def correctbench_answer(text: str) -> str | None:
    patterns = (
        r"\\boxed\{\s*([A-E])\s*\}",
        r"(?:answer|option)\s*(?:is|:)\s*\(?\s*([A-E])\s*\)?",
        r"\b([A-E])\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


def paired_sign_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(trials, i) for i in range(tail + 1)) / 2**trials)


def summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    rows = len(outcomes)
    if rows == 0:
        raise CampaignScoreError("cannot summarize an empty benchmark")
    unchanged = sum(row["unchanged_correct"] for row in outcomes)
    revision = sum(row["revision_correct"] for row in outcomes)
    wins = sum(row["revision_correct"] and not row["unchanged_correct"] for row in outcomes)
    losses = sum(row["unchanged_correct"] and not row["revision_correct"] for row in outcomes)
    return {
        "rows": rows,
        "unchanged_correct": unchanged,
        "trained_revision_correct": revision,
        "unchanged_score": 100 * unchanged / rows,
        "trained_revision_score": 100 * revision / rows,
        "paired_delta_count": revision - unchanged,
        "paired_delta_points": 100 * (revision - unchanged) / rows,
        "wins": wins,
        "losses": losses,
        "paired_sign_test_two_sided_p": paired_sign_pvalue(wins, losses),
        "baseline_correct_retention": (unchanged - losses) / unchanged if unchanged else 1.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    benchmark_entries = manifest.get("benchmarks")
    if not isinstance(benchmark_entries, list) or not benchmark_entries:
        raise CampaignScoreError("campaign manifest differs")
    expected_ids = []
    benchmark_by_id = {}
    assessor_by_id = {}
    benchmark_order = []
    for entry in benchmark_entries:
        benchmark = str(entry["name"])
        benchmark_order.append(benchmark)
        question_rows = load_jsonl(Path(entry["questions"]))
        assessor_path = args.assessor_root / benchmark / args.assessor_name
        assessor_rows = load_jsonl(assessor_path)
        if len(question_rows) != len(assessor_rows) or len(question_rows) != entry["rows"]:
            raise CampaignScoreError(f"{benchmark} question/assessor coverage differs")
        for question, assessor in zip(question_rows, assessor_rows, strict=True):
            identity = question.get("id")
            if (
                assessor.get("schema") != ASSESSOR_SCHEMA
                or assessor.get("id") != identity
                or assessor.get("benchmark") != benchmark
                or identity in benchmark_by_id
            ):
                raise CampaignScoreError(f"{benchmark} identity binding differs")
            expected_ids.append(identity)
            benchmark_by_id[identity] = benchmark
            assessor_by_id[identity] = assessor
    ledgers = {}
    for stage in ("unchanged_continuation", "trained_revision"):
        path = args.generation_root / f"{stage}.jsonl"
        rows = load_jsonl(path)
        if (
            len(rows) != len(expected_ids)
            or [row.get("id") for row in rows] != expected_ids
            or any(row.get("schema") != LEDGER_SCHEMA or row.get("stage") != stage for row in rows)
        ):
            raise CampaignScoreError(f"{stage} ledger coverage differs")
        ledgers[stage] = {row["id"]: row for row in rows}
    official_ifeval = ifeval_scorer(args.ifeval_root) if "ifeval" in benchmark_order else None
    outcomes_by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    instruction_metrics: dict[str, Counter[str]] = defaultdict(Counter)
    for index, identity in enumerate(expected_ids):
        benchmark = benchmark_by_id[identity]
        assessor = assessor_by_id[identity]["assessor"]
        results = {}
        for stage in ("unchanged_continuation", "trained_revision"):
            completion = ledgers[stage][identity]["completion"]
            if benchmark == "mmlu_pro":
                parsed = mmlu_answer(completion)
                result = {"parsed": parsed, "correct": parsed == assessor["answer"]}
            elif benchmark == "musr":
                parsed = musr_answer(completion, assessor["choice_count"], args.seed + index)
                result = {"parsed": parsed, "correct": parsed == assessor["answer"]}
            elif benchmark == "ifeval":
                assert official_ifeval is not None
                result = official_ifeval(assessor, completion)
                bucket = instruction_metrics[f"{benchmark}:{stage}"]
                bucket["strict_instructions"] += sum(result["strict_instructions"])
                bucket["strict_instruction_total"] += len(result["strict_instructions"])
                bucket["loose_prompt"] += int(result["loose_prompt"])
                bucket["loose_instructions"] += sum(result["loose_instructions"])
            elif benchmark == "correctbench":
                parsed = correctbench_answer(completion)
                result = {"parsed": parsed, "correct": parsed == assessor["answer"]}
            else:
                continue
            results[stage] = result
        if len(results) == 2:
            outcomes_by_benchmark[benchmark].append(
                {
                    "id": identity,
                    "stratum": assessor_by_id[identity]["stratum"],
                    "unchanged_correct": bool(results["unchanged_continuation"]["correct"]),
                    "revision_correct": bool(results["trained_revision"]["correct"]),
                    "arms": results,
                }
            )
    benchmarks = {}
    for name in benchmark_order:
        outcomes = outcomes_by_benchmark.get(name, [])
        if not outcomes:
            benchmarks[name] = {"status": "pending_official_scorer"}
            continue
        metrics = summarize(outcomes)
        if name == "ifeval":
            metrics["official_instruction_metrics"] = {
                stage: dict(instruction_metrics[f"ifeval:{stage}"])
                for stage in ("unchanged_continuation", "trained_revision")
            }
        benchmarks[name] = {
            "status": "complete",
            "primary_metric": "official_accuracy_or_strict_prompt_percentage",
            "metrics": metrics,
            "outcomes": outcomes,
        }
    complete = [row for row in benchmarks.values() if row["status"] == "complete"]
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "generation_root": str(args.generation_root.resolve()),
        "benchmarks": benchmarks,
        "standardized_overall": {
            "method": "unweighted_macro_average_of_each_complete_benchmark_primary_percentage",
            "benchmarks": len(complete),
            "unchanged_score": sum(row["metrics"]["unchanged_score"] for row in complete) / len(complete),
            "trained_revision_score": sum(row["metrics"]["trained_revision_score"] for row in complete) / len(complete),
        },
    }
    report["standardized_overall"]["paired_delta_points"] = (
        report["standardized_overall"]["trained_revision_score"]
        - report["standardized_overall"]["unchanged_score"]
    )
    if args.output.exists():
        raise CampaignScoreError("refusing to replace campaign score")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--assessor-root", type=Path, required=True)
    parser.add_argument("--assessor-name", default="screen.assessors.jsonl")
    parser.add_argument("--ifeval-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081903)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report["standardized_overall"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
