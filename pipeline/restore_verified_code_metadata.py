#!/usr/bin/env python3
"""Restore full-test provenance stripped from a decontaminated code derivative."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-restored-execution-verified-code-v1"
VERIFICATION = "execution_verified"


class VerifiedCodeError(RuntimeError):
    """Verified code provenance could not be established exactly."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _declared_sha256(report: dict[str, Any]) -> str:
    for key in ("output_sha256", "out_sha256", "data_sha256"):
        value = report.get(key)
        if value:
            return str(value)
    raise VerifiedCodeError("report does not declare an artifact SHA-256")


def _read_unique(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerifiedCodeError(
                    f"malformed JSONL in {path} at line {line_number}"
                ) from exc
            question = str(row.get("question") or "")
            response = str(row.get("response") or "")
            if not question or not response:
                raise VerifiedCodeError(f"missing question/response in {path}")
            if question in rows:
                raise VerifiedCodeError(f"duplicate exact question in {path}")
            rows[question] = row
    return rows


def restore_verified_code(
    candidate: Path,
    candidate_report: Path,
    verified_source: Path,
    verified_report: Path,
    output: Path,
    report_path: Path,
    *,
    expected_candidate_sha256: str,
    expected_verified_source_sha256: str,
) -> dict[str, Any]:
    paths = (candidate, candidate_report, verified_source, verified_report)
    if not all(path.is_file() for path in paths):
        raise VerifiedCodeError("all inputs and provenance reports must exist")
    if output.exists() or report_path.exists():
        raise VerifiedCodeError("refusing to replace an existing output")

    candidate_metadata = json.loads(candidate_report.read_text(encoding="utf-8"))
    source_metadata = json.loads(verified_report.read_text(encoding="utf-8"))
    if _declared_sha256(candidate_metadata) != expected_candidate_sha256:
        raise VerifiedCodeError("candidate SHA-256 does not match its report")
    if _declared_sha256(source_metadata) != expected_verified_source_sha256:
        raise VerifiedCodeError("verified source SHA-256 does not match its report")
    if _sha256(candidate) != expected_candidate_sha256:
        raise VerifiedCodeError("candidate bytes do not match expected SHA-256")
    if _sha256(verified_source) != expected_verified_source_sha256:
        raise VerifiedCodeError("verified source bytes do not match expected SHA-256")

    candidates = _read_unique(candidate)
    verified_rows = _read_unique(verified_source)
    restored: list[dict[str, Any]] = []
    total_verified_cases = 0
    minimum_verified_cases: int | None = None
    maximum_verified_cases = 0
    for question, row in candidates.items():
        source = verified_rows.get(question)
        if source is None:
            raise VerifiedCodeError("candidate question is absent from verified source")
        if str(source["response"]) != str(row["response"]):
            raise VerifiedCodeError("candidate response differs from verified source")
        full_verified_cases = int(source.get("full_verified_cases") or 0)
        if full_verified_cases <= 0:
            raise VerifiedCodeError("source row lacks full-test verification")
        total_verified_cases += full_verified_cases
        minimum_verified_cases = (
            full_verified_cases
            if minimum_verified_cases is None
            else min(minimum_verified_cases, full_verified_cases)
        )
        maximum_verified_cases = max(maximum_verified_cases, full_verified_cases)
        restored.append(
            {
                **row,
                "training_group": "code",
                "verification": VERIFICATION,
                "verified_cases": int(source.get("verified_cases") or 0),
                "full_verified_cases": full_verified_cases,
                "problem_id": source.get("problem_id"),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for row in restored:
                encoded = (
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                ).encode()
                handle.write(encoded)
                digest.update(encoded)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    report = {
        "schema": SCHEMA,
        "status": "complete",
        "rows": len(restored),
        "training_group": "code",
        "verification": VERIFICATION,
        "total_full_verified_cases": total_verified_cases,
        "minimum_full_verified_cases": minimum_verified_cases,
        "maximum_full_verified_cases": maximum_verified_cases,
        "candidate": str(candidate.resolve()),
        "candidate_sha256": expected_candidate_sha256,
        "candidate_report": str(candidate_report.resolve()),
        "candidate_report_sha256": _sha256(candidate_report),
        "verified_source": str(verified_source.resolve()),
        "verified_source_sha256": expected_verified_source_sha256,
        "verified_report": str(verified_report.resolve()),
        "verified_report_sha256": _sha256(verified_report),
        "output": str(output.resolve()),
        "output_sha256": digest.hexdigest(),
    }
    temporary_report = report_path.with_name(
        f".{report_path.name}.tmp.{os.getpid()}"
    )
    try:
        temporary_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_report, report_path)
    except Exception:
        temporary_report.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--verified-source", type=Path, required=True)
    parser.add_argument("--verified-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-verified-source-sha256", required=True)
    args = parser.parse_args()
    result = restore_verified_code(
        args.candidate,
        args.candidate_report,
        args.verified_source,
        args.verified_report,
        args.output,
        args.report,
        expected_candidate_sha256=args.expected_candidate_sha256,
        expected_verified_source_sha256=args.expected_verified_source_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
