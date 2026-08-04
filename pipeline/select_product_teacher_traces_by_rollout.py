#!/usr/bin/env python3
"""Select complete teacher traces for verifier-positive rollout identities."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "shohin-product-rollout-selected-teacher-traces-v1"


class RolloutTeacherSelectionError(RuntimeError):
    """The rollout ledger and teacher trace bank cannot be joined safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    if not normalized:
        raise RolloutTeacherSelectionError("question identity is empty")
    return hashlib.sha256(normalized.encode()).hexdigest()


def _question(row: dict[str, Any]) -> str:
    value = row.get("question") or row.get("problem") or row.get("prompt")
    if not value:
        raise RolloutTeacherSelectionError("row has no question")
    return str(value).strip()


def _expected(row: dict[str, Any]) -> str:
    value = row.get("expected_answer_normalized")
    if value is None:
        raise RolloutTeacherSelectionError("row has no expected answer")
    return str(value)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise RolloutTeacherSelectionError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
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
        raise RolloutTeacherSelectionError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def select_teacher_traces(
    teacher_bank_path: Path,
    rollout_positives_path: Path,
    output: Path,
    report_output: Path,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    teacher_bank_sha256 = _sha256(teacher_bank_path)
    rollout_positives_sha256 = _sha256(rollout_positives_path)
    selected: dict[str, dict[str, Any]] = {}
    with rollout_positives_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counters["rollout_positive_rows"] += 1
            row = json.loads(line)
            if row.get("verification") != "expected_answer_match_v1":
                raise RolloutTeacherSelectionError(
                    "rollout selection contains an unverified row"
                )
            question = _question(row)
            identity = _identity(question)
            if identity != row.get("source_identity_sha256"):
                raise RolloutTeacherSelectionError("rollout identity differs")
            if identity in selected:
                raise RolloutTeacherSelectionError(
                    "rollout selection repeats a question"
                )
            selected[identity] = row

    if not selected:
        raise RolloutTeacherSelectionError("rollout selection is empty")

    materialized: list[dict[str, Any]] = []
    seen_teachers: set[str] = set()
    with teacher_bank_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counters["teacher_bank_rows"] += 1
            row = json.loads(line)
            question = _question(row)
            identity = _identity(question)
            if identity != row.get("source_identity_sha256"):
                raise RolloutTeacherSelectionError("teacher identity differs")
            if identity in seen_teachers:
                raise RolloutTeacherSelectionError("teacher bank repeats a question")
            seen_teachers.add(identity)
            positive = selected.get(identity)
            if positive is None:
                counters["teacher_rows_not_selected"] += 1
                continue
            if _question(positive) != question:
                raise RolloutTeacherSelectionError("selected question text differs")
            if _expected(positive) != _expected(row):
                raise RolloutTeacherSelectionError("selected expected answer differs")
            if row.get("verification") != "expected_answer_match_v1":
                raise RolloutTeacherSelectionError("teacher trace is not verified")
            response = row.get("response") or row.get("solution")
            if not response:
                raise RolloutTeacherSelectionError("teacher trace has no response")
            materialized.append(
                {
                    **row,
                    "rollout_positive_source": str(rollout_positives_path.resolve()),
                    "rollout_positive_source_sha256": rollout_positives_sha256,
                }
            )
            counters["selected_teacher_rows"] += 1

    missing = sorted(set(selected) - seen_teachers)
    if missing:
        raise RolloutTeacherSelectionError(
            f"{len(missing)} rollout identities are absent from teacher bank"
        )
    if len(materialized) != len(selected):
        raise RolloutTeacherSelectionError("teacher selection coverage differs")

    output_sha256 = _atomic_jsonl(output, materialized)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "teacher_bank": str(teacher_bank_path.resolve()),
        "teacher_bank_sha256": teacher_bank_sha256,
        "rollout_positives": str(rollout_positives_path.resolve()),
        "rollout_positives_sha256": rollout_positives_sha256,
        "rows": len(materialized),
        "counters": dict(sorted(counters.items())),
        "output": str(output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(report_output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bank", required=True, type=Path)
    parser.add_argument("--rollout-positives", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    args = parser.parse_args()
    report = select_teacher_traces(
        args.teacher_bank,
        args.rollout_positives,
        args.output,
        args.report_output,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
