#!/usr/bin/env python3
"""Build a source-disjoint, execution-bound MBPP bank for CVG1 rollouts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_product_rollout_bank import (
    _excluded_constraints,
    _identity,
    _normalized_words,
    _sha256_file,
    _shingles,
)


SCHEMA = "shohin-cvg1-mbpp-rollout-bank-v1"


class CVG1MBPPBankError(RuntimeError):
    """The requested code rollout bank violates its custody contract."""


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise CVG1MBPPBankError(f"refusing existing output: {path}")
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
        raise CVG1MBPPBankError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_bank(
    source: Path,
    excludes: list[Path],
    output: Path,
    report_path: Path,
    *,
    count: int,
    seed: int,
    ngram_width: int = 13,
    max_reference_ngram_document_rate: float = 0.001,
) -> dict[str, Any]:
    if not source.is_file():
        raise CVG1MBPPBankError(f"source is missing: {source}")
    if output.exists() or report_path.exists():
        raise CVG1MBPPBankError("refusing to replace rollout bank outputs")
    if count <= 0 or ngram_width <= 1:
        raise CVG1MBPPBankError("count and ngram width must be positive")
    if not 0.0 <= max_reference_ngram_document_rate < 1.0:
        raise CVG1MBPPBankError("reference ngram rate must be in [0, 1)")

    (
        excluded_identities,
        excluded_shingles,
        exclude_reports,
        informative_document_ceiling,
    ) = _excluded_constraints(
        excludes,
        ngram_width=ngram_width,
        max_reference_ngram_document_rate=max_reference_ngram_document_rate,
    )
    counters: Counter[str] = Counter()
    candidates: dict[str, dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            counters["raw_rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CVG1MBPPBankError(
                    f"malformed source JSONL at line {line_number}"
                ) from exc
            question = str(row.get("text") or "").strip()
            code = str(row.get("code") or "").strip()
            tests = [str(test) for test in row.get("test_list") or ()]
            reference_hash = str(row.get("reference_execution_sha256") or "")
            source_name = str(row.get("source") or "")
            if (
                row.get("task") != "mbpp"
                or source_name not in {"mbpp_train", "mbpp_validation"}
                or not question
                or not code
                or not tests
                or len(reference_hash) != 64
            ):
                counters["schema_rejected"] += 1
                continue
            identity = _identity(question)
            if identity in excluded_identities:
                counters["excluded_overlap"] += 1
                continue
            if _shingles(_normalized_words(question), ngram_width) & excluded_shingles:
                counters["excluded_informative_ngram_overlap"] += 1
                continue
            if identity in candidates:
                counters["duplicate_questions"] += 1
                continue
            candidates[identity] = {
                "schema": SCHEMA,
                "identity_sha256": identity,
                "task": "mbpp",
                "text": question,
                "answer": "execution_pass",
                "code": code,
                "test_list": tests,
                "test_setup_code": str(row.get("test_setup_code") or ""),
                "source": source_name,
                "reference_execution_sha256": reference_hash,
            }
            counters["admissible_rows"] += 1

    ranked = sorted(
        candidates.values(),
        key=lambda row: hashlib.sha256(
            f"{seed}\0mbpp\0{row['identity_sha256']}".encode()
        ).hexdigest(),
    )
    if len(ranked) < count:
        raise CVG1MBPPBankError(
            f"only {len(ranked)} admissible rows remain below requested {count}"
        )
    selected = ranked[:count]
    output_sha256 = _atomic_lines(output, selected)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "seed": seed,
        "ngram_width": ngram_width,
        "max_reference_ngram_document_rate": max_reference_ngram_document_rate,
        "informative_ngram_document_ceiling": informative_document_ceiling,
        "count_requested": count,
        "rows": len(selected),
        "counters": dict(sorted(counters.items())),
        "source": {
            "path": str(source.resolve()),
            "sha256": _sha256_file(source),
        },
        "excludes": exclude_reports,
        "output": str(output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026080715)
    parser.add_argument("--ngram-width", type=int, default=13)
    parser.add_argument(
        "--max-reference-ngram-document-rate", type=float, default=0.001
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build_bank(
        args.source,
        args.exclude,
        args.output,
        args.report,
        count=args.count,
        seed=args.seed,
        ngram_width=args.ngram_width,
        max_reference_ngram_document_rate=args.max_reference_ngram_document_rate,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
