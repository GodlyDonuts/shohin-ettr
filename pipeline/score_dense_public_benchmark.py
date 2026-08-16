#!/usr/bin/env python3
"""Score one matched dense public screen with exact official benchmark logic."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Callable

GENERATION_SCHEMA = "shohin-dense-public-benchmark-generation-v1"
ASSESSOR_SCHEMA = "shohin-dense-public-benchmark-assessor-v1"
REPORT_SCHEMA = "shohin-dense-public-benchmark-score-v1"


class DenseBenchmarkScoreError(RuntimeError):
    """Generation, coverage, assessor, or official scoring evidence differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_assessors(path: Path, benchmark: str) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("id")
            if (
                row.get("schema") != ASSESSOR_SCHEMA
                or row.get("benchmark") != benchmark
                or not isinstance(identity, str)
                or identity in seen
                or not isinstance(row.get("assessor"), dict)
            ):
                raise DenseBenchmarkScoreError("assessor row differs")
            seen.add(identity)
            rows.append(row)
    if len(rows) != 256:
        raise DenseBenchmarkScoreError("screen assessor cardinality differs")
    return rows


def load_generation(
    paths: list[Path], benchmark: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths:
        raise DenseBenchmarkScoreError("generation reports are required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    first = reports[0]
    stable = (
        "host",
        "benchmark",
        "model_revision",
        "model_config_sha256",
        "model_tree_sha256",
        "questions_sha256",
        "draft_checkpoint_sha256",
        "revision_checkpoint_sha256",
        "generation_mode",
        "max_new_tokens_per_stage",
        "seed",
        "shard_count",
        "full_rows",
    )
    for report in reports:
        if (
            report.get("schema") != GENERATION_SCHEMA
            or report.get("status") != "complete"
            or report.get("benchmark") != benchmark
            or any(report.get(key) != first.get(key) for key in stable)
            or report.get("matched_two_pass_budget") is not True
            or report.get("identical_second_pass_decoding") is not True
        ):
            raise DenseBenchmarkScoreError("generation report binding differs")
    count = first.get("shard_count")
    if count != len(paths) or sorted(
        report["shard_index"] for report in reports
    ) != list(range(count)):
        raise DenseBenchmarkScoreError("generation shard coverage differs")
    rows = []
    cursor = 0
    for report in sorted(reports, key=lambda value: value["shard_index"]):
        if report.get("row_start") != cursor or report.get("row_end") - cursor != len(
            report.get("interactions", [])
        ):
            raise DenseBenchmarkScoreError("generation shard bounds differ")
        rows.extend(report["interactions"])
        cursor = report["row_end"]
    if cursor != first["full_rows"] or cursor != 256:
        raise DenseBenchmarkScoreError("generation identity coverage differs")
    if len({row.get("id") for row in rows}) != len(rows):
        raise DenseBenchmarkScoreError("generation identities are duplicated")
    return rows, first


def mmlu_answer(text: str) -> str | None:
    matches = re.findall(r"answer is\s*\(?([A-J])\)?", text, re.IGNORECASE)
    if not matches:
        matches = re.findall(r"\banswer\s*:\s*\(?([A-J])\)?", text, re.IGNORECASE)
    if not matches:
        matches = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", text.upper())
    return matches[-1].upper() if matches else None


def musr_answer(text: str, choices: int, seed: int) -> int:
    matches = re.findall(r"ANSWER\s*:\s*\(?\s*(\d+)\s*\)?", text, re.IGNORECASE)
    if matches and 1 <= int(matches[-1]) <= choices:
        return int(matches[-1])
    return random.Random(seed).randint(1, choices)


def ifeval_scorer(root: Path) -> Callable[[dict[str, Any], str], dict[str, Any]]:
    if not root.is_dir() or (root / ".git").exists():
        raise DenseBenchmarkScoreError("IFEval source must be an exported source tree")
    sys.path.insert(0, str(root.parent))
    try:
        library = importlib.import_module("instruction_following_eval.evaluation_lib")
    except ImportError as exc:
        raise DenseBenchmarkScoreError(
            "pinned IFEval source is not importable"
        ) from exc

    def score(assessor: dict[str, Any], response: str) -> dict[str, Any]:
        example = library.InputExample(
            key=assessor["key"],
            instruction_id_list=assessor["instruction_id_list"],
            prompt=assessor["prompt"],
            kwargs=assessor["kwargs"],
        )
        responses = {assessor["prompt"]: response}
        strict = library.test_instruction_following_strict(example, responses)
        loose = library.test_instruction_following_loose(example, responses)
        return {
            "correct": bool(strict.follow_all_instructions),
            "strict_prompt": bool(strict.follow_all_instructions),
            "strict_instructions": [
                bool(value) for value in strict.follow_instruction_list
            ],
            "loose_prompt": bool(loose.follow_all_instructions),
            "loose_instructions": [
                bool(value) for value in loose.follow_instruction_list
            ],
        }

    return score


def paired_sign_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    return min(
        1.0, 2 * sum(math.comb(trials, index) for index in range(tail + 1)) / 2**trials
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    assessors = load_assessors(args.assessors, args.benchmark)
    interactions, generation = load_generation(args.generation_report, args.benchmark)
    assessor_by_id = {row["id"]: row for row in assessors}
    if [row["id"] for row in interactions] != [row["id"] for row in assessors]:
        raise DenseBenchmarkScoreError("generation/assessor identity order differs")
    official_ifeval = (
        ifeval_scorer(args.ifeval_root) if args.benchmark == "ifeval" else None
    )
    outcomes = []
    for index, interaction in enumerate(interactions):
        assessor_row = assessor_by_id[interaction["id"]]
        assessor = assessor_row["assessor"]
        arms = {}
        for arm in ("unchanged_continuation", "trained_revision"):
            completion = interaction.get(arm)
            if not isinstance(completion, str):
                raise DenseBenchmarkScoreError("completion differs")
            if args.benchmark == "mmlu_pro":
                parsed = mmlu_answer(completion)
                result = {"parsed": parsed, "correct": parsed == assessor["answer"]}
            elif args.benchmark == "musr":
                parsed = musr_answer(
                    completion, assessor["choice_count"], args.seed + index
                )
                result = {"parsed": parsed, "correct": parsed == assessor["answer"]}
            else:
                assert official_ifeval is not None
                result = official_ifeval(assessor, completion)
            arms[arm] = result
        outcomes.append(
            {
                "id": interaction["id"],
                "upstream_id": interaction["upstream_id"],
                "stratum": assessor_row["stratum"],
                "arms": arms,
            }
        )
    unchanged = sum(
        row["arms"]["unchanged_continuation"]["correct"] for row in outcomes
    )
    revision = sum(row["arms"]["trained_revision"]["correct"] for row in outcomes)
    wins = sum(
        row["arms"]["trained_revision"]["correct"]
        and not row["arms"]["unchanged_continuation"]["correct"]
        for row in outcomes
    )
    losses = sum(
        row["arms"]["unchanged_continuation"]["correct"]
        and not row["arms"]["trained_revision"]["correct"]
        for row in outcomes
    )
    strata = defaultdict(
        lambda: Counter(rows=0, unchanged=0, revision=0, wins=0, losses=0)
    )
    for row in outcomes:
        base = row["arms"]["unchanged_continuation"]["correct"]
        treatment = row["arms"]["trained_revision"]["correct"]
        bucket = strata[row["stratum"]]
        bucket["rows"] += 1
        bucket["unchanged"] += int(base)
        bucket["revision"] += int(treatment)
        bucket["wins"] += int(treatment and not base)
        bucket["losses"] += int(base and not treatment)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "label": "prospective_256_row_screen_not_full_benchmark",
        "host": generation["host"],
        "benchmark": args.benchmark,
        "rows": len(outcomes),
        "generation_reports": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.generation_report
        ],
        "assessors": str(args.assessors.resolve()),
        "assessors_sha256": sha256_file(args.assessors),
        "model_revision": generation["model_revision"],
        "draft_checkpoint_sha256": generation["draft_checkpoint_sha256"],
        "revision_checkpoint_sha256": generation["revision_checkpoint_sha256"],
        "metrics": {
            "unchanged_correct": unchanged,
            "trained_revision_correct": revision,
            "paired_delta_count": revision - unchanged,
            "paired_delta_points": 100 * (revision - unchanged) / len(outcomes),
            "wins": wins,
            "losses": losses,
            "paired_sign_test_two_sided_p": paired_sign_pvalue(wins, losses),
            "baseline_correct_retained": unchanged - losses,
            "baseline_correct_retention": (
                (unchanged - losses) / unchanged if unchanged else 1.0
            ),
        },
        "strata": {name: dict(counts) for name, counts in sorted(strata.items())},
        "outcomes": outcomes,
    }
    if args.benchmark == "ifeval":
        for arm in ("unchanged_continuation", "trained_revision"):
            report["metrics"][f"{arm}_strict_instruction"] = sum(
                sum(row["arms"][arm]["strict_instructions"]) for row in outcomes
            )
            report["metrics"][f"{arm}_strict_instruction_total"] = sum(
                len(row["arms"][arm]["strict_instructions"]) for row in outcomes
            )
            report["metrics"][f"{arm}_loose_prompt"] = sum(
                row["arms"][arm]["loose_prompt"] for row in outcomes
            )
            report["metrics"][f"{arm}_loose_instruction"] = sum(
                sum(row["arms"][arm]["loose_instructions"]) for row in outcomes
            )
    if args.output.exists():
        raise DenseBenchmarkScoreError("score output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", choices=("mmlu_pro", "ifeval", "musr"), required=True
    )
    parser.add_argument(
        "--generation-report", type=Path, action="append", required=True
    )
    parser.add_argument("--assessors", type=Path, required=True)
    parser.add_argument("--ifeval-root", type=Path)
    parser.add_argument("--seed", type=int, default=2026081521)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.benchmark == "ifeval" and args.ifeval_root is None:
        parser.error("--ifeval-root is required for IFEval")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "host": report["host"],
                "benchmark": report["benchmark"],
                "metrics": report["metrics"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
