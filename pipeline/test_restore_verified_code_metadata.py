from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipeline.restore_verified_code_metadata import (
    VerifiedCodeError,
    restore_verified_code,
)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, changed_response: bool = False):
    source = tmp_path / "source.jsonl"
    source_sha = _write_jsonl(
        source,
        [
            {
                "question": "Implement add.",
                "response": "def add(a, b): return a + b",
                "full_verified_cases": 12,
                "verified_cases": 3,
                "problem_id": "p1",
            }
        ],
    )
    candidate = tmp_path / "candidate.jsonl"
    candidate_sha = _write_jsonl(
        candidate,
        [
            {
                "question": "Implement add.",
                "response": (
                    "def add(a, b): return a - b"
                    if changed_response
                    else "def add(a, b): return a + b"
                ),
                "training_group": "algorithmic_code",
            }
        ],
    )
    source_report = tmp_path / "source.report.json"
    source_report.write_text(json.dumps({"data_sha256": source_sha}))
    candidate_report = tmp_path / "candidate.report.json"
    candidate_report.write_text(json.dumps({"out_sha256": candidate_sha}))
    return source, source_report, source_sha, candidate, candidate_report, candidate_sha


def test_restore_verified_code_exactly_matches_source(tmp_path: Path) -> None:
    source, source_report, source_sha, candidate, candidate_report, candidate_sha = (
        _fixture(tmp_path)
    )
    output = tmp_path / "output.jsonl"
    report = restore_verified_code(
        candidate,
        candidate_report,
        source,
        source_report,
        output,
        tmp_path / "output.report.json",
        expected_candidate_sha256=candidate_sha,
        expected_verified_source_sha256=source_sha,
    )
    row = json.loads(output.read_text())
    assert row["training_group"] == "code"
    assert row["verification"] == "execution_verified"
    assert row["full_verified_cases"] == 12
    assert report["total_full_verified_cases"] == 12


def test_restore_verified_code_rejects_changed_solution(tmp_path: Path) -> None:
    source, source_report, source_sha, candidate, candidate_report, candidate_sha = (
        _fixture(tmp_path, changed_response=True)
    )
    with pytest.raises(VerifiedCodeError, match="response differs"):
        restore_verified_code(
            candidate,
            candidate_report,
            source,
            source_report,
            tmp_path / "output.jsonl",
            tmp_path / "output.report.json",
            expected_candidate_sha256=candidate_sha,
            expected_verified_source_sha256=source_sha,
        )
