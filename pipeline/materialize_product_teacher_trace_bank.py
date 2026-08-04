#!/usr/bin/env python3
"""Restore verified teacher traces for an immutable rollout prompt bank."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "shohin-product-teacher-trace-bank-v1"
BANK_SCHEMA = "shohin-product-rollout-bank-v1"


class TeacherTraceBankError(RuntimeError):
    """The prompt bank cannot be joined to its verified teacher source."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    if not normalized:
        raise TeacherTraceBankError("question identity is empty")
    return hashlib.sha256(normalized.encode()).hexdigest()


def _question(row: dict[str, Any]) -> str:
    value = row.get("question") or row.get("problem") or row.get("prompt")
    if not value:
        raise TeacherTraceBankError("row has no question")
    return str(value).strip()


def _group(row: dict[str, Any]) -> str:
    value = row.get("training_group") or row.get("domain")
    if not value:
        raise TeacherTraceBankError("row has no training group")
    return str(value).strip().casefold()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise TeacherTraceBankError(f"refusing existing output: {path}")
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
        raise TeacherTraceBankError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def materialize_teacher_trace_bank(
    bank_path: Path,
    teacher_source_path: Path,
    output: Path,
    report_output: Path,
) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    bank_sha256 = _sha256(bank_path)
    teacher_source_sha256 = _sha256(teacher_source_path)
    teachers: dict[str, dict[str, Any]] = {}
    with teacher_source_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counters["teacher_source_rows"] += 1
            row = json.loads(line)
            question = _question(row)
            identity = _identity(question)
            if identity in teachers:
                raise TeacherTraceBankError("teacher source repeats a question")
            if row.get("verification") != "expected_answer_match_v1":
                counters["teacher_unverified_rows"] += 1
                continue
            response = row.get("response") or row.get("solution")
            if not response:
                raise TeacherTraceBankError("verified teacher row has no response")
            teachers[identity] = row

    materialized: list[dict[str, Any]] = []
    seen: set[str] = set()
    with bank_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counters["bank_rows"] += 1
            bank = json.loads(line)
            if bank.get("schema") != BANK_SCHEMA:
                raise TeacherTraceBankError("prompt bank schema differs")
            question = _question(bank)
            identity = _identity(question)
            if identity != bank.get("identity_sha256"):
                raise TeacherTraceBankError("prompt bank identity differs")
            if identity in seen:
                raise TeacherTraceBankError("prompt bank repeats a question")
            seen.add(identity)
            teacher = teachers.get(identity)
            if teacher is None:
                raise TeacherTraceBankError(
                    "prompt bank row is absent from teacher source"
                )
            if _question(teacher) != question:
                raise TeacherTraceBankError("teacher question text differs")
            if _group(teacher) != _group(bank):
                raise TeacherTraceBankError("teacher training group differs")
            if str(teacher.get("expected_answer_normalized")) != str(
                bank.get("expected_answer_normalized")
            ):
                raise TeacherTraceBankError("teacher expected answer differs")
            materialized.append(
                {
                    **teacher,
                    "source_identity_sha256": identity,
                    "selection_bank": str(bank_path.resolve()),
                    "selection_bank_sha256": bank_sha256,
                    "teacher_source": str(teacher_source_path.resolve()),
                    "teacher_source_sha256": teacher_source_sha256,
                }
            )
            counters["materialized_rows"] += 1

    if not materialized:
        raise TeacherTraceBankError("prompt bank is empty")
    output_sha256 = _atomic_jsonl(output, materialized)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "bank": str(bank_path.resolve()),
        "bank_sha256": bank_sha256,
        "teacher_source": str(teacher_source_path.resolve()),
        "teacher_source_sha256": teacher_source_sha256,
        "rows": len(materialized),
        "counters": dict(sorted(counters.items())),
        "output": str(output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(report_output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--teacher-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    args = parser.parse_args()
    report = materialize_teacher_trace_bank(
        args.bank,
        args.teacher_source,
        args.output,
        args.report_output,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
