#!/usr/bin/env python3
"""Merge deterministic frozen-lineage rollouts into one CVG1 pair corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

CANDIDATE_SCHEMA = "shohin-hf-product-reasoning-rollouts-v1"
PAIR_SCHEMA = "shohin-cvg1-whole-lineage-pairs-v1"


class LineageMergeError(RuntimeError):
    """Frozen lineage artifacts do not define a valid paired corpus."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_report(path: Path, candidate_path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != CANDIDATE_SCHEMA or report.get("status") != "complete":
        raise LineageMergeError(f"rollout report is incomplete: {path}")
    if report.get("generation_mode") != "greedy" or int(report.get("samples", 0)) != 1:
        raise LineageMergeError("CVG1 requires one deterministic rollout per lineage")
    if report.get("candidates_sha256") != sha256_file(candidate_path):
        raise LineageMergeError("candidate file hash differs from its rollout report")
    adapter = report.get("adapter_checkpoint")
    adapter_sha256 = report.get("adapter_checkpoint_sha256")
    if not adapter or not adapter_sha256:
        raise LineageMergeError("rollout report lacks a bound adapter checkpoint")
    return report


def _load_lineage(
    candidate_paths: list[Path],
    report_paths: list[Path],
    lineage: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if len(candidate_paths) != len(report_paths) or not candidate_paths:
        raise LineageMergeError("candidate and report shard counts must match")
    rows: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    adapter_sha256: str | None = None
    for candidate_path, report_path in zip(candidate_paths, report_paths, strict=True):
        report = _load_report(report_path, candidate_path)
        if adapter_sha256 not in (None, report["adapter_checkpoint_sha256"]):
            raise LineageMergeError("rollout shards use different lineage checkpoints")
        data_sha256 = str(report["data_sha256"])
        adapter_sha256 = str(report["adapter_checkpoint_sha256"])
        receipts.append(
            {
                "candidate_path": str(candidate_path.resolve()),
                "candidate_sha256": sha256_file(candidate_path),
                "report_path": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "skip": int(report["skip"]),
                "count": int(report["count"]),
                "data_sha256": data_sha256,
                "adapter_checkpoint": str(report["adapter_checkpoint"]),
                "adapter_checkpoint_sha256": adapter_sha256,
            }
        )
        with candidate_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                required = (
                    "schema",
                    "identity_sha256",
                    "question",
                    "task",
                    "sample_index",
                    "completion",
                    "correct",
                )
                if any(key not in row for key in required):
                    raise LineageMergeError(
                        f"candidate schema differs at {candidate_path}:{line_number}"
                    )
                if row["schema"] != CANDIDATE_SCHEMA or int(row["sample_index"]) != 0:
                    raise LineageMergeError(
                        "candidate is not a deterministic sample zero"
                    )
                identity = str(row["identity_sha256"])
                if identity in rows:
                    raise LineageMergeError(f"duplicate {lineage} identity: {identity}")
                completion = str(row["completion"])
                if not completion.strip():
                    raise LineageMergeError("candidate completion is empty")
                rows[identity] = {
                    "lineage": lineage,
                    "prompt_bank_sha256": data_sha256,
                    "question": str(row["question"]),
                    "task": str(row["task"]),
                    "completion": completion,
                    "correct": bool(row["correct"]),
                    "generated_tokens": int(row.get("generated_tokens") or 0),
                    "max_token_exhausted": bool(row.get("max_token_exhausted")),
                }
    receipts.sort(
        key=lambda receipt: (
            receipt["data_sha256"],
            receipt["skip"],
            receipt["candidate_sha256"],
        )
    )
    coverage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        coverage[str(receipt["data_sha256"])].append(receipt)
    row_counts = Counter(str(row["prompt_bank_sha256"]) for row in rows.values())
    for data_sha256, data_receipts in coverage.items():
        expected_skip = 0
        for receipt in data_receipts:
            if receipt["skip"] != expected_skip:
                raise LineageMergeError("rollout shard coverage has a gap or overlap")
            expected_skip += receipt["count"]
        if expected_skip != row_counts[data_sha256]:
            raise LineageMergeError("rollout shard cardinality differs from reports")
    return rows, receipts


def _split(identity: str, seed: int) -> str:
    bucket = (
        int.from_bytes(
            hashlib.sha256(f"{seed}\0{identity}".encode()).digest()[:8], "big"
        )
        % 10
    )
    if bucket == 0:
        return "holdout"
    if bucket == 1:
        return "development"
    return "train"


def merge_lineages(
    *,
    base_candidates: list[Path],
    base_reports: list[Path],
    expert_candidates: list[Path],
    expert_reports: list[Path],
    split_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base, base_receipts = _load_lineage(base_candidates, base_reports, "base")
    expert, expert_receipts = _load_lineage(expert_candidates, expert_reports, "expert")
    if set(base) != set(expert):
        raise LineageMergeError("base and expert prompt identities differ")
    if {receipt["data_sha256"] for receipt in base_receipts} != {
        receipt["data_sha256"] for receipt in expert_receipts
    }:
        raise LineageMergeError("base and expert prompt-bank hashes differ")
    if {receipt["adapter_checkpoint_sha256"] for receipt in base_receipts} == {
        receipt["adapter_checkpoint_sha256"] for receipt in expert_receipts
    }:
        raise LineageMergeError("base and expert checkpoint identities are equal")

    rows: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    for identity in sorted(base):
        base_row = base[identity]
        expert_row = expert[identity]
        if (
            base_row["question"] != expert_row["question"]
            or base_row["task"] != expert_row["task"]
            or base_row["prompt_bank_sha256"] != expert_row["prompt_bank_sha256"]
        ):
            raise LineageMergeError("paired prompt metadata differs")
        if base_row["correct"] and expert_row["correct"]:
            outcome = "both_correct"
        elif base_row["correct"]:
            outcome = "base_only"
        elif expert_row["correct"]:
            outcome = "expert_only"
        else:
            outcome = "both_wrong"
        split = _split(identity, split_seed)
        task = base_row["task"]
        candidates = []
        for row in (base_row, expert_row):
            candidates.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"question", "task", "prompt_bank_sha256"}
                }
            )
        rows.append(
            {
                "schema": PAIR_SCHEMA,
                "identity_sha256": identity,
                "split": split,
                "task": task,
                "question": base_row["question"],
                "prompt_bank_sha256": base_row["prompt_bank_sha256"],
                "outcome_class": outcome,
                "candidates": candidates,
            }
        )
        outcomes[outcome] += 1
        splits[split] += 1
        tasks[task] += 1
    return rows, {
        "schema": PAIR_SCHEMA,
        "status": "complete",
        "rows": len(rows),
        "split_seed": split_seed,
        "outcome_counts": dict(sorted(outcomes.items())),
        "split_counts": dict(sorted(splits.items())),
        "task_counts": dict(sorted(tasks.items())),
        "base_receipts": base_receipts,
        "expert_receipts": expert_receipts,
    }


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise LineageMergeError(f"refusing existing output: {path}")
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
        raise LineageMergeError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-candidates", type=Path, action="append", required=True)
    parser.add_argument("--base-report", type=Path, action="append", required=True)
    parser.add_argument(
        "--expert-candidates", type=Path, action="append", required=True
    )
    parser.add_argument("--expert-report", type=Path, action="append", required=True)
    parser.add_argument("--split-seed", type=int, default=2026080713)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows, report = merge_lineages(
        base_candidates=args.base_candidates,
        base_reports=args.base_report,
        expert_candidates=args.expert_candidates,
        expert_reports=args.expert_report,
        split_seed=args.split_seed,
    )
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = _atomic_lines(args.output, rows)
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
