"""Shared custody checks for benchmark-specific official Shohin scorers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "shohin-dense-public-campaign-manifest-v1"
QUESTION_SCHEMA = "shohin-dense-public-benchmark-question-v1"
ASSESSOR_SCHEMA = "shohin-dense-public-benchmark-assessor-v1"
LEDGER_SCHEMA = "shohin-dense-public-campaign-ledger-v1"
SCORE_SCHEMA = "shohin-dense-public-official-score-v1"
STAGES = ("direct_base", "unchanged_continuation", "trained_revision")


class OfficialScoringError(RuntimeError):
    """Manifest, assessor, completion, or scorer custody differs."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_bound_benchmark(
    *,
    manifest_path: Path,
    generation_root: Path,
    assessor_root: Path,
    assessor_name: str,
    benchmark: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise OfficialScoringError("campaign manifest schema differs")
    entries = manifest.get("benchmarks")
    if not isinstance(entries, list) or not entries:
        raise OfficialScoringError("campaign benchmark list differs")
    expected_global: list[tuple[str, str]] = []
    selected_questions = None
    selected_assessors = None
    for entry in entries:
        name = str(entry["name"])
        questions = load_jsonl(Path(entry["questions"]))
        assessors = load_jsonl(assessor_root / name / assessor_name)
        if len(questions) != entry["rows"] or len(assessors) != len(questions):
            raise OfficialScoringError(f"{name} question/assessor coverage differs")
        for question, assessor in zip(questions, assessors, strict=True):
            identity = question.get("id")
            if (
                question.get("schema") != QUESTION_SCHEMA
                or question.get("benchmark") != name
                or assessor.get("schema") != ASSESSOR_SCHEMA
                or assessor.get("benchmark") != name
                or assessor.get("id") != identity
            ):
                raise OfficialScoringError(f"{name} question/assessor binding differs")
            expected_global.append((str(identity), name))
        if name == benchmark:
            if selected_questions is not None:
                raise OfficialScoringError("selected benchmark is duplicated")
            selected_questions = questions
            selected_assessors = assessors
    if selected_questions is None or selected_assessors is None:
        raise OfficialScoringError("selected benchmark is absent")
    if len({identity for identity, _ in expected_global}) != len(expected_global):
        raise OfficialScoringError("campaign identities are duplicated")

    stage_rows: dict[str, list[dict[str, Any]]] = {}
    for stage in STAGES:
        rows = load_jsonl(generation_root / f"{stage}.jsonl")
        if len(rows) != len(expected_global):
            raise OfficialScoringError(f"{stage} generation coverage differs")
        selected = []
        for row, (identity, name) in zip(rows, expected_global, strict=True):
            if (
                row.get("schema") != LEDGER_SCHEMA
                or row.get("stage") != stage
                or row.get("id") != identity
                or row.get("benchmark") != name
                or not isinstance(row.get("completion"), str)
            ):
                raise OfficialScoringError(f"{stage} generation binding differs")
            if name == benchmark:
                selected.append(row)
        stage_rows[stage] = selected
    return selected_questions, selected_assessors, stage_rows


def write_official_scores(
    *,
    output_root: Path,
    benchmark: str,
    stage: str,
    rows: list[dict[str, Any]],
) -> str:
    path = output_root / benchmark / f"{stage}.official-scores.jsonl"
    if path.exists():
        raise OfficialScoringError(f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode() + b"\n"
            digest.update(encoded)
            handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def official_score_row(
    *,
    stage: str,
    identity: str,
    benchmark: str,
    metric: str,
    stratum: str,
    score: float,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not 0.0 <= score <= 1.0:
        raise OfficialScoringError("official score is outside [0, 1]")
    row = {
        "schema": SCORE_SCHEMA,
        "stage": stage,
        "id": identity,
        "benchmark": benchmark,
        "metric": metric,
        "stratum": stratum,
        "score": score,
    }
    if details is not None:
        row["details"] = details
    return row
