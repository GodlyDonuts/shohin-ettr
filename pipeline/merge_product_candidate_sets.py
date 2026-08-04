#!/usr/bin/env python3
"""Merge independent candidate draws into one contiguous per-prompt sample set."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-product-candidate-set-merge-v1"
INVARIANT_FIELDS = ("task", "question", "gold", "training_group")


class ProductCandidateMergeError(RuntimeError):
    """Candidate sets do not describe compatible independent draws."""


def _group(rows: list[dict[str, Any]], source: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        identity = str(row.get("identity_sha256") or "")
        if not identity or "sample_index" not in row:
            raise ProductCandidateMergeError(
                f"{source} candidate identity is incomplete"
            )
        grouped[identity].append(row)
    if not grouped:
        raise ProductCandidateMergeError(f"{source} candidate set is empty")
    return dict(grouped)


def _validate_groups(grouped: dict[str, list[dict[str, Any]]], source: str) -> int:
    sample_counts = {len(group) for group in grouped.values()}
    if len(sample_counts) != 1:
        raise ProductCandidateMergeError(f"{source} sample counts differ by identity")
    samples = sample_counts.pop()
    for group in grouped.values():
        indices = sorted(int(row["sample_index"]) for row in group)
        if indices != list(range(samples)):
            raise ProductCandidateMergeError(f"{source} sample indices differ")
    return samples


def merge_candidate_sets(
    base_rows: list[dict[str, Any]], supplement_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = _group(base_rows, "base")
    supplement = _group(supplement_rows, "supplement")
    if set(base) != set(supplement):
        raise ProductCandidateMergeError("candidate identity sets differ")
    base_samples = _validate_groups(base, "base")
    supplement_samples = _validate_groups(supplement, "supplement")

    merged: list[dict[str, Any]] = []
    task_counts: dict[str, int] = defaultdict(int)
    for identity, base_group in base.items():
        supplement_group = supplement[identity]
        reference = base_group[0]
        for row in supplement_group:
            for field in INVARIANT_FIELDS:
                if row.get(field) != reference.get(field):
                    raise ProductCandidateMergeError(
                        f"candidate {field} differs for identity {identity}"
                    )
        merged.extend(sorted(base_group, key=lambda row: int(row["sample_index"])))
        for row in sorted(supplement_group, key=lambda item: int(item["sample_index"])):
            updated = dict(row)
            updated["sample_index"] = base_samples + int(row["sample_index"])
            merged.append(updated)
        task_counts[str(reference.get("task") or "")] += 1

    report = {
        "schema": SCHEMA,
        "identities": len(base),
        "base_samples_per_identity": base_samples,
        "supplement_samples_per_identity": supplement_samples,
        "merged_samples_per_identity": base_samples + supplement_samples,
        "rows": len(merged),
        "task_identity_counts": dict(sorted(task_counts.items())),
    }
    return merged, report


def _read_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    source = path.read_bytes()
    return source, [json.loads(line) for line in source.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output, args.report):
        if output.exists():
            raise ProductCandidateMergeError(f"refusing existing output: {output}")

    base_bytes, base_rows = _read_jsonl(args.base)
    supplement_bytes, supplement_rows = _read_jsonl(args.supplement)
    merged, report = merge_candidate_sets(base_rows, supplement_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    output_digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in merged:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            output_digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)

    report.update(
        {
            "base": str(args.base.resolve()),
            "base_sha256": hashlib.sha256(base_bytes).hexdigest(),
            "supplement": str(args.supplement.resolve()),
            "supplement_sha256": hashlib.sha256(supplement_bytes).hexdigest(),
            "output": str(args.output.resolve()),
            "output_sha256": output_digest.hexdigest(),
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report.with_suffix(args.report.suffix + ".partial")
    with temporary_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_report, args.report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
