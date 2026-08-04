#!/usr/bin/env python3
"""Materialize verifier-correct, non-exhausted autonomous rollout drafts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-product-rollout-positive-merge-v1"
CANDIDATE_SCHEMA = "shohin-hf-product-reasoning-rollouts-v1"
AGGREGATE_SCHEMA = "shohin-product-rollout-aggregate-v1"


class CompleteRolloutPositiveError(RuntimeError):
    """The candidate ledger cannot produce a complete positive corpus."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise CompleteRolloutPositiveError(f"refusing existing output: {path}")
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
        raise CompleteRolloutPositiveError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def materialize_complete_positives(
    aggregate_report_path: Path,
    output: Path,
    report_output: Path,
    *,
    min_positive_total: int,
) -> dict[str, Any]:
    aggregate = json.loads(aggregate_report_path.read_text())
    if (
        aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("status") != "complete"
        or not aggregate.get("admitted")
    ):
        raise CompleteRolloutPositiveError("aggregate report is not admitted")
    candidates_path = Path(aggregate["candidates_output"])
    candidates_sha256 = _sha256(candidates_path)
    if candidates_sha256 != aggregate.get("candidates_sha256"):
        raise CompleteRolloutPositiveError("candidate ledger hash differs")

    counters: Counter[str] = Counter()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    contracts: dict[str, tuple[str, str, str, str]] = {}
    sample_indices: dict[str, set[int]] = defaultdict(set)
    with candidates_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counters["candidate_rows"] += 1
            row = json.loads(line)
            if row.get("schema") != CANDIDATE_SCHEMA:
                raise CompleteRolloutPositiveError("candidate schema differs")
            identity = str(row.get("identity_sha256") or "")
            question = str(row.get("question") or "")
            group = str(row.get("training_group") or "")
            gold = str(row.get("gold") or "")
            task = str(row.get("task") or "")
            if not all((identity, question, group, gold, task)):
                raise CompleteRolloutPositiveError("candidate contract is incomplete")
            contract = (question, group, gold, task)
            previous_contract = contracts.setdefault(identity, contract)
            if previous_contract != contract:
                raise CompleteRolloutPositiveError("candidate contract differs")
            sample_index = int(row["sample_index"])
            if sample_index in sample_indices[identity]:
                raise CompleteRolloutPositiveError("candidate sample index repeats")
            sample_indices[identity].add(sample_index)
            grouped[identity].append(row)

    positives: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    for identity, candidates in grouped.items():
        complete = [
            row
            for row in candidates
            if row.get("correct")
            and row.get("explicit_final_answer")
            and not row.get("draft_max_token_exhausted", False)
        ]
        if not complete:
            counters["prompts_without_complete_positive"] += 1
            continue
        chosen = min(
            complete,
            key=lambda row: (
                int(row.get("draft_generated_tokens") or row["generated_tokens"]),
                len(str(row["draft_completion"])),
                int(row["sample_index"]),
            ),
        )
        draft = str(chosen.get("draft_completion") or "")
        if not draft or draft != chosen.get("completion"):
            raise CompleteRolloutPositiveError(
                "complete draft differs from scored completion"
            )
        if chosen.get("finalization") is not None:
            raise CompleteRolloutPositiveError(
                "complete autonomous draft unexpectedly has a finalization"
            )
        negatives = [row for row in candidates if not row.get("correct")]
        positives.append(
            {
                "question": chosen["question"],
                "response": draft,
                "answer": chosen["gold"],
                "expected_answer_normalized": chosen["gold"],
                "training_group": chosen["training_group"],
                "verification": "student_complete_exact_answer_match_v1",
                "source_identity_sha256": identity,
                "source_adapter_checkpoint": aggregate["contract"][
                    "adapter_checkpoint"
                ],
                "chosen_sample_index": chosen["sample_index"],
                "rejected_response": (
                    min(
                        negatives,
                        key=lambda row: (
                            int(row["generated_tokens"]),
                            int(row["sample_index"]),
                        ),
                    )["completion"]
                    if negatives
                    else None
                ),
            }
        )
        group_counts[str(chosen["training_group"])] += 1
        counters["complete_positive_prompts"] += 1

    positives.sort(key=lambda row: str(row["source_identity_sha256"]))
    if len(positives) < min_positive_total:
        raise CompleteRolloutPositiveError(
            f"complete positives {len(positives)} below minimum {min_positive_total}"
        )
    positives_sha256 = _atomic_jsonl(output, positives)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "admitted": True,
        "aggregate_report": str(aggregate_report_path.resolve()),
        "aggregate_report_sha256": _sha256(aggregate_report_path),
        "candidates_output": str(candidates_path.resolve()),
        "candidates_sha256": candidates_sha256,
        "counters": dict(sorted(counters.items())),
        "min_positive_total": min_positive_total,
        "positive_group_counts": dict(sorted(group_counts.items())),
        "positive_prompts": len(positives),
        "positives_output": str(output.resolve()),
        "positives_sha256": positives_sha256,
    }
    _atomic_json(report_output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--min-positive-total", type=int, default=1)
    args = parser.parse_args()
    report = materialize_complete_positives(
        args.aggregate_report,
        args.output,
        args.report_output,
        min_positive_total=args.min_positive_total,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
