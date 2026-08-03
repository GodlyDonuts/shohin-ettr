#!/usr/bin/env python3
"""Merge admitted verified-rollout corpora with deterministic identity priority."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any


AGGREGATE_SCHEMA = "shohin-product-rollout-aggregate-v1"
SCHEMA = "shohin-product-rollout-positive-merge-v1"


class ProductRolloutMergeError(RuntimeError):
    """Verified rollout aggregates cannot be merged under one contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ProductRolloutMergeError(f"missing JSONL: {path}")
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise ProductRolloutMergeError(f"malformed JSONL: {path}") from exc


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ProductRolloutMergeError(f"refusing existing output: {path}")
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
        raise ProductRolloutMergeError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _same_identity_contract(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "question",
        "answer",
        "expected_answer_normalized",
        "training_group",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def merge(
    aggregate_reports: list[Path],
    positives_output: Path,
    report_output: Path,
    *,
    required_groups: dict[str, int],
) -> dict[str, Any]:
    if not aggregate_reports:
        raise ProductRolloutMergeError("no aggregate reports supplied")

    common_contract: dict[str, Any] | None = None
    merged_by_identity: dict[str, dict[str, Any]] = {}
    merged_order: list[str] = []
    sources: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for priority, report_path in enumerate(aggregate_reports):
        if not report_path.is_file():
            raise ProductRolloutMergeError(f"missing aggregate report: {report_path}")
        report = json.loads(report_path.read_text())
        if report.get("schema") != AGGREGATE_SCHEMA or report.get("status") != "complete":
            raise ProductRolloutMergeError("aggregate report is structurally incomplete")
        contract = report.get("contract") or {}
        identity_contract = {
            key: contract.get(key)
            for key in ("model_root", "model_revision", "adapter_checkpoint", "samples")
        }
        if common_contract is None:
            common_contract = identity_contract
        elif identity_contract != common_contract:
            raise ProductRolloutMergeError("aggregate model or sampling contract differs")

        positives_path = Path(report["positives_output"])
        if _sha256(positives_path) != report.get("positives_sha256"):
            raise ProductRolloutMergeError("aggregate positive hash differs")
        rows = _jsonl(positives_path)
        if len(rows) != int(report.get("positive_prompts", -1)):
            raise ProductRolloutMergeError("aggregate positive cardinality differs")
        for row in rows:
            identity = str(row.get("source_identity_sha256") or "")
            if not identity or row.get("verification") != "student_exact_answer_match_v1":
                raise ProductRolloutMergeError("positive provenance contract differs")
            previous = merged_by_identity.get(identity)
            if previous is not None:
                if not _same_identity_contract(previous, row):
                    raise ProductRolloutMergeError("duplicate identity metadata differs")
                counters["duplicate_identity_drops"] += 1
                continue
            merged_by_identity[identity] = row
            merged_order.append(identity)
            counters["unique_positives"] += 1
        counters["input_positives"] += len(rows)
        sources.append(
            {
                "priority": priority,
                "report": str(report_path.resolve()),
                "report_sha256": _sha256(report_path),
                "positives": str(positives_path.resolve()),
                "positives_sha256": report["positives_sha256"],
                "max_new_tokens": contract.get("max_new_tokens"),
                "admitted": bool(report.get("admitted")),
                "admission_failures": report.get("admission_failures"),
                "positive_prompts": len(rows),
                "positive_group_counts": report.get("positive_group_counts"),
            }
        )

    merged = [merged_by_identity[identity] for identity in merged_order]
    group_counts = Counter(str(row.get("training_group")) for row in merged)
    admission_failures = [
        f"positive_{group}_below_minimum"
        for group, minimum in sorted(required_groups.items())
        if group_counts[group] < minimum
    ]
    positives_sha256 = _atomic_jsonl(positives_output, merged)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "admitted": not admission_failures,
        "admission_failures": admission_failures,
        "common_contract": common_contract,
        "required_groups": required_groups,
        "positive_prompts": len(merged),
        "positive_group_counts": dict(sorted(group_counts.items())),
        "counters": dict(sorted(counters.items())),
        "sources": sources,
        "positives_output": str(positives_output.resolve()),
        "positives_sha256": positives_sha256,
    }
    _atomic_json(report_output, payload)
    return payload


def _required_groups(values: list[str]) -> dict[str, int]:
    groups: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("required groups use group=minimum")
        group, raw_minimum = value.split("=", 1)
        group = group.strip()
        try:
            minimum = int(raw_minimum)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("group minimum must be an integer") from exc
        if not group or minimum <= 0 or group in groups:
            raise argparse.ArgumentTypeError("group requirements differ and are positive")
        groups[group] = minimum
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-report", action="append", type=Path, required=True)
    parser.add_argument("--positives-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--require-group", action="append", default=[])
    args = parser.parse_args()
    args.required_groups = _required_groups(args.require_group)
    return args


def main() -> int:
    args = parse_args()
    report = merge(
        args.aggregate_report,
        args.positives_output,
        args.report_output,
        required_groups=args.required_groups,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
