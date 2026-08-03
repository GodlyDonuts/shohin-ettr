#!/usr/bin/env python3
"""Admit verified procedural traces into the product-reasoning mix schema."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "shohin-verified-procedural-product-v1"


class ProceduralAdmissionError(RuntimeError):
    """The procedural source cannot be admitted without weakening the gate."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_question(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().casefold()
    if not normalized:
        raise ProceduralAdmissionError("question normalizes to an empty string")
    return normalized


def _question(row: dict[str, Any]) -> str | None:
    for key in ("question", "problem", "prompt", "text", "input"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return None


def _load_eval_questions(paths: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    questions: set[str] = set()
    receipts: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise ProceduralAdmissionError(f"evaluation source does not exist: {path}")
        rows = accepted = 0
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ProceduralAdmissionError(
                        f"malformed evaluation JSONL: {path}"
                    ) from exc
                question = _question(row)
                if question:
                    questions.add(_normalized_question(question))
                    accepted += 1
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "rows": rows,
                "questions": accepted,
            }
        )
    return questions, receipts


def adapt_verified_procedural(
    source: Path,
    output: Path,
    report_path: Path,
    *,
    eval_paths: list[Path],
) -> dict[str, Any]:
    if not source.is_file():
        raise ProceduralAdmissionError(f"source does not exist: {source}")
    if output.exists() or report_path.exists():
        raise ProceduralAdmissionError("output and report must not already exist")
    eval_questions, eval_receipts = _load_eval_questions(eval_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    report_partial = report_path.with_suffix(report_path.suffix + ".partial")
    if partial.exists() or report_partial.exists():
        raise ProceduralAdmissionError("stale partial output exists")

    counters: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as source_handle, partial.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for raw_line in source_handle:
            if not raw_line.strip():
                continue
            counters["raw_rows"] += 1
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ProceduralAdmissionError("source contains malformed JSONL") from exc
            question = _question(row)
            response = str(row.get("response") or "").strip()
            answer = str(row.get("answer") or "").strip()
            family = str(row.get("family") or "").strip()
            if not question or not response or not answer or not family:
                counters["missing_required_field"] += 1
                continue
            if str(row.get("source") or "") != "reasoning_gym_trace":
                counters["unverified_source"] += 1
                continue
            if "<think>" not in response or "</think>" not in response:
                counters["missing_verified_trace"] += 1
                continue
            normalized = _normalized_question(question)
            if normalized in eval_questions:
                counters["eval_overlap_dropped"] += 1
                continue
            if normalized in seen:
                counters["duplicate_question_dropped"] += 1
                continue
            seen.add(normalized)
            admitted = {
                **row,
                "question": question,
                "response": response,
                "answer": answer,
                "family": family,
                "training_group": "procedural",
                "verification": "reasoning_gym_answer_verified",
                "admission_schema": SCHEMA,
            }
            output_handle.write(json.dumps(admitted, ensure_ascii=False) + "\n")
            counters["kept_rows"] += 1
            family_counts[family] += 1
        output_handle.flush()
        os.fsync(output_handle.fileno())

    if counters["kept_rows"] == 0:
        partial.unlink(missing_ok=True)
        raise ProceduralAdmissionError("no verified procedural rows survived")
    os.replace(partial, output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "evaluation_sources": eval_receipts,
        "counters": dict(sorted(counters.items())),
        "family_counts": dict(sorted(family_counts.items())),
    }
    with report_partial.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(report_partial, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--eval", action="append", default=[], type=Path)
    args = parser.parse_args()
    report = adapt_verified_procedural(
        args.source,
        args.output,
        args.report,
        eval_paths=args.eval,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
