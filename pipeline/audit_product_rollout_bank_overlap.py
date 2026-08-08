#!/usr/bin/env python3
"""Audit a rollout bank against training sources without retaining source text."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

WORD = re.compile(r"\w+")


class RolloutBankAuditError(RuntimeError):
    """The rollout-bank audit contract was violated."""


def normalized_words(value: str) -> tuple[str, ...]:
    return tuple(WORD.findall(value.casefold()))


def question_from_row(row: dict[str, Any]) -> str | None:
    for key in ("question", "problem", "prompt", "text", "input"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shingles(words: tuple[str, ...], width: int) -> Iterable[tuple[str, ...]]:
    if len(words) < width:
        return ()
    return (words[index : index + width] for index in range(len(words) - width + 1))


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RolloutBankAuditError(
                    f"malformed JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise RolloutBankAuditError(f"non-object row at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise RolloutBankAuditError(f"empty JSONL source: {path}")
    return rows


def audit(candidate: Path, references: list[Path], ngram: int) -> dict[str, Any]:
    if ngram <= 1:
        raise RolloutBankAuditError("ngram width must exceed one")
    candidate_rows = read_rows(candidate)
    reference_rows = [row for path in references for row in read_rows(path)]

    reference_questions: set[tuple[str, ...]] = set()
    reference_shingles: set[tuple[str, ...]] = set()
    for row in reference_rows:
        question = question_from_row(row)
        if question is None:
            continue
        words = normalized_words(question)
        if words:
            reference_questions.add(words)
            reference_shingles.update(shingles(words, ngram))

    seen: Counter[tuple[str, ...]] = Counter()
    task_counts: Counter[str] = Counter()
    exact_hits = 0
    ngram_hits = 0
    missing_questions = 0
    missing_answers = 0
    for row in candidate_rows:
        question = question_from_row(row)
        if question is None:
            missing_questions += 1
            continue
        words = normalized_words(question)
        if not words:
            missing_questions += 1
            continue
        seen[words] += 1
        task_counts[str(row.get("task") or row.get("training_group") or "unknown")] += 1
        exact_hits += int(words in reference_questions)
        ngram_hits += int(
            any(shingle in reference_shingles for shingle in shingles(words, ngram))
        )
        answer = row.get(
            "expected_answer_normalized", row.get("answer", row.get("target"))
        )
        missing_answers += int(answer is None or not str(answer).strip())

    duplicate_rows = sum(count - 1 for count in seen.values() if count > 1)
    admitted = not any(
        (exact_hits, ngram_hits, missing_questions, missing_answers, duplicate_rows)
    )
    return {
        "schema": "shohin-product-rollout-bank-overlap-audit-v1",
        "status": "pass" if admitted else "fail",
        "candidate": {
            "path": str(candidate.resolve()),
            "sha256": sha256_file(candidate),
        },
        "references": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in references
        ],
        "candidate_rows": len(candidate_rows),
        "reference_rows": len(reference_rows),
        "normalized_unique_questions": len(seen),
        "duplicate_normalized_rows": duplicate_rows,
        "missing_questions": missing_questions,
        "missing_answers": missing_answers,
        "exact_reference_hits": exact_hits,
        "ngram_width": ngram,
        "rows_with_reference_ngram_hit": ngram_hits,
        "task_counts": dict(sorted(task_counts.items())),
        "admitted": admitted,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RolloutBankAuditError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.candidate.is_file() or any(
        not path.is_file() for path in args.reference
    ):
        parser.error("candidate and every reference must exist")
    report = audit(args.candidate, args.reference, args.ngram)
    atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
