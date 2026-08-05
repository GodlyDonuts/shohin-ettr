#!/usr/bin/env python3
"""Merge disjoint, hash-bound rollout banks into one candidate set."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROLLOUT_SCHEMA = "shohin-hf-product-reasoning-rollouts-v1"
SCHEMA = "shohin-sharded-product-candidate-merge-v1"


class ShardedCandidateMergeError(RuntimeError):
    """Candidate shards do not form one complete canonical task bank."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_bytes().splitlines() if line]
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ShardedCandidateMergeError(f"malformed JSONL: {path}") from exc


def _question(row: dict[str, Any]) -> str:
    for key in ("question", "problem", "prompt", "text", "input"):
        value = row.get(key)
        if value:
            return str(value)
    raise ShardedCandidateMergeError("task row has no question")


def _identity(task: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{task}\0{_question(row)}".encode()).hexdigest()


def merge_shard_reports(
    canonical_bank: Path,
    report_paths: list[Path],
    *,
    task: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical_rows = _rows(canonical_bank)
    canonical_identities = [_identity(task, row) for row in canonical_rows]
    if not canonical_rows or len(set(canonical_identities)) != len(canonical_identities):
        raise ShardedCandidateMergeError("canonical bank is empty or repeats")
    if not report_paths:
        raise ShardedCandidateMergeError("no shard reports supplied")

    contracts: set[str] = set()
    by_identity: dict[str, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []
    expected_samples: int | None = None
    for report_path in report_paths:
        report = json.loads(report_path.read_text())
        if report.get("schema") != ROLLOUT_SCHEMA or report.get("status") != "complete":
            raise ShardedCandidateMergeError("shard report schema or status differs")
        contract = {
            key: report.get(key)
            for key in (
                "model_root",
                "model_revision",
                "adapter_checkpoint",
                "samples",
                "max_new_tokens",
                "seed",
            )
        }
        contracts.add(json.dumps(contract, sort_keys=True))
        samples = int(report.get("samples", -1))
        if samples <= 0 or (expected_samples is not None and samples != expected_samples):
            raise ShardedCandidateMergeError("shard sample counts differ")
        expected_samples = samples

        shard_bank = Path(str(report.get("data") or ""))
        candidate_path = Path(str(report.get("candidates_output") or ""))
        if not shard_bank.is_file() or not candidate_path.is_file():
            raise ShardedCandidateMergeError("shard source file is missing")
        if _sha256(shard_bank) != report.get("data_sha256"):
            raise ShardedCandidateMergeError("shard bank hash differs")
        if _sha256(candidate_path) != report.get("candidates_sha256"):
            raise ShardedCandidateMergeError("candidate hash differs")

        shard_rows = _rows(shard_bank)
        shard_identities = [_identity(task, row) for row in shard_rows]
        if len(shard_rows) != int(report.get("count", -1)) or len(
            set(shard_identities)
        ) != len(shard_identities):
            raise ShardedCandidateMergeError("shard bank cardinality differs")
        candidates = _rows(candidate_path)
        if len(candidates) != len(shard_rows) * samples:
            raise ShardedCandidateMergeError("candidate cardinality differs")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            identity = str(row.get("identity_sha256") or "")
            grouped[identity].append(row)
        if set(grouped) != set(shard_identities):
            raise ShardedCandidateMergeError("candidate identities differ from shard bank")
        for identity, rows in grouped.items():
            if identity in by_identity:
                raise ShardedCandidateMergeError("candidate identity repeats across shards")
            rows.sort(key=lambda row: int(row.get("sample_index", -1)))
            if [int(row.get("sample_index", -1)) for row in rows] != list(
                range(samples)
            ):
                raise ShardedCandidateMergeError("candidate sample indices differ")
            if any(str(row.get("task")) != task for row in rows):
                raise ShardedCandidateMergeError("candidate task differs")
            by_identity[identity] = rows
        if sum(bool(row.get("correct")) for row in candidates) != int(
            (report.get("counters") or {}).get("correct_candidates", -1)
        ):
            raise ShardedCandidateMergeError("correct-candidate counter differs")
        sources.append(
            {
                "report": str(report_path.resolve()),
                "report_sha256": _sha256(report_path),
                "bank": str(shard_bank.resolve()),
                "bank_sha256": _sha256(shard_bank),
                "candidates": str(candidate_path.resolve()),
                "candidates_sha256": _sha256(candidate_path),
                "identities": len(shard_rows),
            }
        )
    if len(contracts) != 1:
        raise ShardedCandidateMergeError("shard rollout contracts differ")
    if set(by_identity) != set(canonical_identities):
        raise ShardedCandidateMergeError("shards do not cover the canonical bank")

    merged = [row for identity in canonical_identities for row in by_identity[identity]]
    return merged, {
        "schema": SCHEMA,
        "status": "complete",
        "task": task,
        "canonical_bank": str(canonical_bank.resolve()),
        "canonical_bank_sha256": _sha256(canonical_bank),
        "identities": len(canonical_identities),
        "samples_per_identity": expected_samples,
        "rows": len(merged),
        "correct_candidates": sum(bool(row.get("correct")) for row in merged),
        "contract": json.loads(next(iter(contracts))),
        "sources": sources,
    }


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ShardedCandidateMergeError(f"refusing existing output: {path}")
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
        raise ShardedCandidateMergeError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--shard-report", type=Path, action="append", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows, report = merge_shard_reports(args.bank, args.shard_report, task=args.task)
    report["output_sha256"] = _atomic_lines(args.output, rows)
    report["output"] = str(args.output.resolve())
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
