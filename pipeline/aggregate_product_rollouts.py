#!/usr/bin/env python3
"""Aggregate a complete, hash-bound product rollout fan into training rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROLLOUT_SCHEMA = "shohin-hf-product-reasoning-rollouts-v1"
SCHEMA = "shohin-product-rollout-aggregate-v1"


class ProductRolloutAggregateError(RuntimeError):
    """The rollout fan cannot be admitted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ProductRolloutAggregateError(f"missing JSONL: {path}")
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ProductRolloutAggregateError(f"malformed JSONL: {path}") from exc


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ProductRolloutAggregateError(f"refusing existing output: {path}")
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
        raise ProductRolloutAggregateError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def aggregate(
    bank_path: Path,
    report_paths: list[Path],
    candidates_output: Path,
    positives_output: Path,
    report_output: Path,
    *,
    min_positive_total: int,
    min_positive_per_group: int,
) -> dict[str, Any]:
    bank_bytes = bank_path.read_bytes()
    bank_rows = [json.loads(line) for line in bank_bytes.splitlines() if line.strip()]
    bank_identities = [str(row.get("identity_sha256")) for row in bank_rows]
    if not bank_rows or any(not identity for identity in bank_identities):
        raise ProductRolloutAggregateError("bank identities are incomplete")
    if len(set(bank_identities)) != len(bank_identities):
        raise ProductRolloutAggregateError("bank identities repeat")
    bank_by_identity = dict(zip(bank_identities, bank_rows, strict=True))
    bank_sha256 = hashlib.sha256(bank_bytes).hexdigest()

    loaded_reports: list[tuple[Path, dict[str, Any]]] = []
    for path in report_paths:
        if not path.is_file():
            raise ProductRolloutAggregateError(f"missing shard report: {path}")
        payload = json.loads(path.read_text())
        if payload.get("schema") != ROLLOUT_SCHEMA or payload.get("status") != "complete":
            raise ProductRolloutAggregateError("shard report schema or status differs")
        loaded_reports.append((path, payload))
    loaded_reports.sort(key=lambda item: int(item[1]["skip"]))
    if not loaded_reports:
        raise ProductRolloutAggregateError("no shard reports supplied")

    contract_keys = (
        "model_root",
        "model_revision",
        "adapter_checkpoint",
        "data_sha256",
        "samples",
        "max_new_tokens",
    )
    contract = {key: loaded_reports[0][1].get(key) for key in contract_keys}
    if contract["data_sha256"] != bank_sha256:
        raise ProductRolloutAggregateError("rollout data hash differs from bank")

    expected_start = 0
    all_candidates: list[dict[str, Any]] = []
    all_positives: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    for report_path, report in loaded_reports:
        if any(report.get(key) != value for key, value in contract.items()):
            raise ProductRolloutAggregateError("shard rollout contracts differ")
        skip = int(report["skip"])
        count = int(report["count"])
        samples = int(report["samples"])
        if skip != expected_start or count <= 0:
            raise ProductRolloutAggregateError("shard slices contain a gap or overlap")
        stop = skip + count
        if stop > len(bank_rows):
            raise ProductRolloutAggregateError("shard slice exceeds bank")
        expected_start = stop

        candidates_path = Path(report["candidates_output"])
        positives_path = Path(report["positives_output"])
        if _sha256(candidates_path) != report.get("candidates_sha256"):
            raise ProductRolloutAggregateError("candidate file hash differs")
        if _sha256(positives_path) != report.get("positives_sha256"):
            raise ProductRolloutAggregateError("positive file hash differs")
        candidates = _jsonl(candidates_path)
        positives = _jsonl(positives_path)
        if len(candidates) != count * samples:
            raise ProductRolloutAggregateError("candidate cardinality differs")
        if len(positives) != int(report["counters"].get("positive_prompts", 0)):
            raise ProductRolloutAggregateError("positive cardinality differs")
        if sum(bool(row.get("correct")) for row in candidates) != int(
            report["counters"].get("correct_candidates", 0)
        ):
            raise ProductRolloutAggregateError("correct-candidate counter differs")

        slice_identities = set(bank_identities[skip:stop])
        candidates_by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            identity = str(candidate.get("identity_sha256"))
            if identity not in slice_identities:
                raise ProductRolloutAggregateError("candidate identity outside shard")
            candidates_by_identity[identity].append(candidate)
        if set(candidates_by_identity) != slice_identities:
            raise ProductRolloutAggregateError("candidate identities do not cover shard")
        for rows in candidates_by_identity.values():
            if len(rows) != samples or {int(row["sample_index"]) for row in rows} != set(
                range(samples)
            ):
                raise ProductRolloutAggregateError("candidate sample indices differ")

        seen_positive: set[str] = set()
        for positive in positives:
            identity = str(positive.get("source_identity_sha256"))
            if identity not in slice_identities or identity in seen_positive:
                raise ProductRolloutAggregateError("positive identity differs or repeats")
            seen_positive.add(identity)
            chosen_index = int(positive["chosen_sample_index"])
            chosen = next(
                row
                for row in candidates_by_identity[identity]
                if int(row["sample_index"]) == chosen_index
            )
            if not chosen.get("correct") or positive.get("response") != chosen.get("completion"):
                raise ProductRolloutAggregateError("positive is not its verified candidate")
            if positive.get("training_group") != bank_by_identity[identity].get(
                "training_group"
            ):
                raise ProductRolloutAggregateError("positive training group differs")
        all_candidates.extend(candidates)
        all_positives.extend(positives)
        source_reports.append(
            {
                "path": str(report_path.resolve()),
                "sha256": _sha256(report_path),
                "skip": skip,
                "count": count,
                "prompt_batch_size": report.get("prompt_batch_size"),
            }
        )

    if expected_start != len(bank_rows):
        raise ProductRolloutAggregateError("shard reports do not cover complete bank")
    positive_identities = [str(row["source_identity_sha256"]) for row in all_positives]
    if len(set(positive_identities)) != len(positive_identities):
        raise ProductRolloutAggregateError("positive identities repeat across shards")
    group_counts = Counter(str(row["training_group"]) for row in all_positives)
    admission_failures: list[str] = []
    if len(all_positives) < min_positive_total:
        admission_failures.append("positive_total_below_minimum")
    bank_groups = sorted({str(row["training_group"]) for row in bank_rows})
    for group in bank_groups:
        if group_counts[group] < min_positive_per_group:
            admission_failures.append(f"positive_{group}_below_minimum")

    candidates_sha256 = _atomic_jsonl(candidates_output, all_candidates)
    positives_sha256 = _atomic_jsonl(positives_output, all_positives)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "admitted": not admission_failures,
        "admission_failures": admission_failures,
        "bank": str(bank_path.resolve()),
        "bank_sha256": bank_sha256,
        "bank_rows": len(bank_rows),
        "source_reports": source_reports,
        "contract": contract,
        "candidates": len(all_candidates),
        "correct_candidates": sum(bool(row.get("correct")) for row in all_candidates),
        "positive_prompts": len(all_positives),
        "prompt_positive_rate": len(all_positives) / len(bank_rows),
        "positive_group_counts": dict(sorted(group_counts.items())),
        "min_positive_total": min_positive_total,
        "min_positive_per_group": min_positive_per_group,
        "candidates_output": str(candidates_output.resolve()),
        "candidates_sha256": candidates_sha256,
        "positives_output": str(positives_output.resolve()),
        "positives_sha256": positives_sha256,
    }
    _atomic_json(report_output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--shard-report", type=Path, action="append", required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--positives-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--min-positive-total", type=int, default=512)
    parser.add_argument("--min-positive-per-group", type=int, default=128)
    args = parser.parse_args()
    if args.min_positive_total <= 0 or args.min_positive_per_group <= 0:
        parser.error("positive minima must be positive")
    return args


def main() -> int:
    args = parse_args()
    report = aggregate(
        args.bank,
        args.shard_report,
        args.candidates_output,
        args.positives_output,
        args.report_output,
        min_positive_total=args.min_positive_total,
        min_positive_per_group=args.min_positive_per_group,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
