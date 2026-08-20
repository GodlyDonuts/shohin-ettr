#!/usr/bin/env python3
"""Score a prospective two-arm GPT-OSS MMLU-Pro commit confirmation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any

from build_dense_public_benchmark_data import text_digest
from score_dense_public_benchmark import mmlu_answer
from score_nemotron_super_screen import paired_report
from score_q36_mtr_external import _load_jsonl, sha256_file

SCHEMA = "shohin-gpt-oss-120b-commit-confirmation-score-v1"
SOURCE_SCHEMA = "shohin-q36-mtr-external-validation-source-v1"
ASSESSOR_SCHEMA = "shohin-dense-public-benchmark-assessor-v1"
CANDIDATE_SCHEMA = "shohin-gpt-oss-120b-fixed-draft-candidate-v1"
TASK = "mmlu_pro"
ARMS = ("revision", "unchanged")
ROWS = 256
ASSESSOR_ROWS = 12_032
SHARDS = 4


class ConfirmationScoreError(RuntimeError):
    """The prospective GPT-OSS confirmation scoring contract differed."""


def _sources(path: Path, expected_sha256: str) -> dict[str, dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ConfirmationScoreError("confirmation source bytes differ")
    result: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != SOURCE_SCHEMA
            or row.get("split") != "external_validation"
            or row.get("task") != TASK
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in result
            or not isinstance(row.get("source_prompt"), str)
            or not row["source_prompt"].strip()
            or any(
                field in row
                for field in ("assessor", "answer", "gold", "correct", "response")
            )
        ):
            raise ConfirmationScoreError("confirmation source projection differs")
        result[identity] = row
    if len(result) != ROWS:
        raise ConfirmationScoreError("confirmation source coverage differs")
    return result


def _assessors(
    path: Path, expected_sha256: str, sources: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ConfirmationScoreError("confirmation assessor bytes differ")
    full: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        identity = row.get("id")
        assessor = row.get("assessor")
        if (
            row.get("schema") != ASSESSOR_SCHEMA
            or row.get("benchmark") != TASK
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in full
            or not isinstance(row.get("stratum"), str)
            or not isinstance(row.get("question_sha256"), str)
            or not isinstance(assessor, dict)
            or not isinstance(assessor.get("answer"), str)
        ):
            raise ConfirmationScoreError("confirmation assessor row differs")
        full[identity] = row
    if len(full) != ASSESSOR_ROWS or not set(sources).issubset(full):
        raise ConfirmationScoreError("confirmation assessor coverage differs")
    selected = {identity: full[identity] for identity in sources}
    if any(
        row["question_sha256"] != text_digest(sources[identity]["source_prompt"])
        for identity, row in selected.items()
    ):
        raise ConfirmationScoreError("confirmation question/assessor binding differs")
    return selected


def _candidates(
    arm: str, paths: list[Path], identities: set[str]
) -> dict[str, dict[str, Any]]:
    if arm not in ARMS or len(paths) != SHARDS:
        raise ConfirmationScoreError("confirmation candidate geometry differs")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ConfirmationScoreError("confirmation candidate shard differs")
        for row in _load_jsonl(path):
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != CANDIDATE_SCHEMA
                or row.get("arm") != arm
                or row.get("task") != TASK
                or not isinstance(identity, str)
                or identity in result
                or not isinstance(row.get("completion"), str)
                or isinstance(row.get("generated_tokens"), bool)
                or not isinstance(row.get("generated_tokens"), int)
                or row["generated_tokens"] < 0
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise ConfirmationScoreError("confirmation candidate row differs")
            result[identity] = row
    if set(result) != identities:
        raise ConfirmationScoreError("confirmation candidate coverage differs")
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ConfirmationScoreError("confirmation score exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def score(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise ConfirmationScoreError("confirmation score settings differ")
    sources = _sources(args.source, args.expected_source_sha256)
    identities = set(sources)
    assessors = _assessors(args.assessors, args.expected_assessors_sha256, sources)
    paths = {arm: getattr(args, f"{arm}_candidates") for arm in ARMS}
    candidates = {arm: _candidates(arm, paths[arm], identities) for arm in ARMS}
    outcomes: dict[str, dict[str, bool]] = {arm: {} for arm in ARMS}
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    empty = Counter()
    exhausted = Counter()
    for identity in sorted(identities):
        answer = assessors[identity]["assessor"]["answer"]
        stratum = assessors[identity]["stratum"]
        bucket = strata[stratum]
        bucket["rows"] += 1
        for arm in ARMS:
            candidate = candidates[arm][identity]
            correct = mmlu_answer(candidate["completion"]) == answer
            outcomes[arm][identity] = correct
            bucket[f"{arm}_correct"] += int(correct)
            empty[arm] += int(not candidate["completion"].strip())
            exhausted[arm] += int(candidate["max_token_exhausted"])

    unchanged_correct = sum(outcomes["unchanged"].values())
    arm_reports: dict[str, Any] = {}
    for arm in ARMS:
        correct = sum(outcomes[arm].values())
        retained = sum(
            outcomes["unchanged"][identity] and outcomes[arm][identity]
            for identity in identities
        )
        arm_reports[arm] = {
            "correct": correct,
            "total": ROWS,
            "accuracy": correct / ROWS,
            "gain_over_unchanged_count": correct - unchanged_correct,
            "unchanged_correct_retained": retained,
            "unchanged_correct_retention": (
                retained / unchanged_correct if unchanged_correct else None
            ),
            "empty_completions": empty[arm],
            "max_token_exhausted": exhausted[arm],
            "candidate_sha256s": [sha256_file(path) for path in paths[arm]],
        }
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "host": "openai/gpt-oss-120b",
        "benchmark": TASK,
        "rows": ROWS,
        "shards_per_arm": SHARDS,
        "source_sha256": args.expected_source_sha256,
        "assessors_sha256": args.expected_assessors_sha256,
        "assessor_universe_rows": ASSESSOR_ROWS,
        "assessor_selected_rows": ROWS,
        "assessor_open_phase": "post_generation_cpu_score_only",
        "official_scoring": "MMLU-Pro terminal answer extraction",
        "arms": arm_reports,
        "strata": {
            name: dict(sorted(values.items()))
            for name, values in sorted(strata.items())
        },
        "revision_vs_unchanged": paired_report(
            outcomes["revision"], outcomes["unchanged"]
        ),
        "outcomes": [
            {
                "identity_sha256": identity,
                "task": TASK,
                "stratum": assessors[identity]["stratum"],
                "correct": {arm: outcomes[arm][identity] for arm in ARMS},
            }
            for identity in sorted(identities)
        ],
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--assessors", type=Path, required=True)
    parser.add_argument("--expected-assessors-sha256", required=True)
    for arm in ARMS:
        parser.add_argument(
            f"--{arm}-candidates", type=Path, action="append", required=True
        )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    result = score(parse_args())
    print(
        json.dumps(
            {arm: result["arms"][arm]["correct"] for arm in ARMS}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
