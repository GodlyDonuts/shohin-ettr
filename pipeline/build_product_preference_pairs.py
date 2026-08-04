#!/usr/bin/env python3
"""Build verifier-labeled within-prompt preference pairs from rollout reports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


AGGREGATE_SCHEMA = "shohin-product-rollout-aggregate-v1"
SCHEMA = "shohin-product-verifier-preference-pairs-v1"


class ProductPreferencePairError(RuntimeError):
    """The rollout population cannot produce a trustworthy preference set."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ProductPreferencePairError(f"refusing existing output: {path}")
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
        raise ProductPreferencePairError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rank(seed: int, identity: str, chosen: dict[str, Any], rejected: dict[str, Any]) -> str:
    payload = (
        f"{seed}\0{identity}\0{chosen.get('sample_index')}\0"
        f"{rejected.get('sample_index')}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_pairs(
    aggregate_reports: list[Path],
    output: Path,
    report_output: Path,
    *,
    pairs_per_prompt: int,
    seed: int,
) -> dict[str, Any]:
    if not aggregate_reports:
        raise ProductPreferencePairError("at least one aggregate report is required")
    if pairs_per_prompt <= 0:
        raise ProductPreferencePairError("pairs per prompt must be positive")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_receipts: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    checkpoint_contract: tuple[str, str] | None = None
    for report_path in aggregate_reports:
        report = json.loads(report_path.read_text())
        if (
            report.get("schema") != AGGREGATE_SCHEMA
            or not report.get("admitted")
            or report.get("admission_failures")
        ):
            raise ProductPreferencePairError("rollout aggregate is not admitted")
        candidates_path = Path(report["candidates_output"])
        candidates_hash = _sha256(candidates_path)
        if candidates_hash != report.get("candidates_sha256"):
            raise ProductPreferencePairError("rollout candidate hash differs")
        contract = report.get("contract") or {}
        current_contract = (
            str(contract.get("adapter_checkpoint") or ""),
            str(contract.get("model_revision") or ""),
        )
        if not all(current_contract):
            raise ProductPreferencePairError("rollout model contract is incomplete")
        if checkpoint_contract is None:
            checkpoint_contract = current_contract
        elif checkpoint_contract != current_contract:
            raise ProductPreferencePairError("rollout model contracts differ")

        local_rows = 0
        with candidates_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProductPreferencePairError("candidate JSONL is malformed") from exc
                required = (
                    "identity_sha256",
                    "question",
                    "completion",
                    "correct",
                    "sample_index",
                    "training_group",
                )
                if any(key not in row for key in required):
                    raise ProductPreferencePairError("candidate row schema differs")
                identity = str(row["identity_sha256"])
                if not identity or not str(row["question"]).strip():
                    raise ProductPreferencePairError("candidate identity or question is empty")
                if not str(row["completion"]).strip():
                    counters["empty_completion_drops"] += 1
                    local_rows += 1
                    continue
                grouped[identity].append(row)
                local_rows += 1
        if local_rows != int(report.get("candidates", -1)):
            raise ProductPreferencePairError("candidate cardinality differs")
        source_receipts.append(
            {
                "aggregate_report": str(report_path.resolve()),
                "aggregate_report_sha256": _sha256(report_path),
                "candidates": str(candidates_path.resolve()),
                "candidates_sha256": candidates_hash,
                "rows": local_rows,
            }
        )

    selected: list[tuple[str, dict[str, Any]]] = []
    group_counts: Counter[str] = Counter()
    for identity, rows in grouped.items():
        questions = {str(row["question"]) for row in rows}
        groups = {str(row["training_group"]) for row in rows}
        if len(questions) != 1 or len(groups) != 1:
            raise ProductPreferencePairError("candidate identity has conflicting metadata")
        chosen = [row for row in rows if bool(row["correct"])]
        rejected = [row for row in rows if not bool(row["correct"])]
        if not chosen:
            counters["all_wrong_prompts"] += 1
            continue
        if not rejected:
            counters["all_correct_prompts"] += 1
            continue
        counters["mixed_prompts"] += 1
        combinations = [
            (_rank(seed, identity, good, bad), good, bad)
            for good in chosen
            for bad in rejected
        ]
        combinations.sort(key=lambda item: item[0])
        for rank, good, bad in combinations[:pairs_per_prompt]:
            group = next(iter(groups))
            pair = {
                "schema": SCHEMA,
                "identity_sha256": identity,
                "question": next(iter(questions)),
                "chosen": str(good["completion"]),
                "rejected": str(bad["completion"]),
                "training_group": group,
                "chosen_sample_index": int(good["sample_index"]),
                "rejected_sample_index": int(bad["sample_index"]),
                "chosen_generated_tokens": int(good.get("generated_tokens") or 0),
                "rejected_generated_tokens": int(bad.get("generated_tokens") or 0),
                "pair_rank_sha256": rank,
            }
            selected.append((rank, pair))
            group_counts[group] += 1
    if not selected:
        raise ProductPreferencePairError("no mixed-outcome preference pairs exist")
    selected.sort(key=lambda item: item[0])
    rows = [row for _, row in selected]
    output_sha256 = _atomic_jsonl(output, rows)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "seed": seed,
        "pairs_per_prompt": pairs_per_prompt,
        "source_reports": source_receipts,
        "adapter_checkpoint": checkpoint_contract[0] if checkpoint_contract else None,
        "model_revision": checkpoint_contract[1] if checkpoint_contract else None,
        "candidate_prompts": len(grouped),
        "mixed_prompts": counters["mixed_prompts"],
        "all_correct_prompts": counters["all_correct_prompts"],
        "all_wrong_prompts": counters["all_wrong_prompts"],
        "pairs": len(rows),
        "group_counts": dict(sorted(group_counts.items())),
        "output": str(output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(report_output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--pairs-per-prompt", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_pairs(
        args.aggregate_report,
        args.output,
        args.report_output,
        pairs_per_prompt=args.pairs_per_prompt,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
