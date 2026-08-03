#!/usr/bin/env python3
"""Build a fresh, deterministic exact-answer bank for product rollouts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "shohin-product-rollout-bank-v1"
TASK_BY_GROUP = {"math": "math500", "science": "bbh_logic"}


class ProductRolloutBankError(RuntimeError):
    """The rollout bank cannot be constructed without violating its contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _question(row: dict[str, Any]) -> str | None:
    for key in ("question", "problem", "prompt", "text", "input"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return None


def _identity(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    if not normalized:
        raise ProductRolloutBankError("question identity is empty")
    return hashlib.sha256(normalized.encode()).hexdigest()


def _parse_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("counts must use group=integer")
        group, raw_count = value.split("=", 1)
        group = group.strip()
        if group not in TASK_BY_GROUP or group in counts:
            raise argparse.ArgumentTypeError("count groups differ or repeat")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("count must be an integer") from exc
        if count <= 0:
            raise argparse.ArgumentTypeError("count must be positive")
        counts[group] = count
    if not counts:
        raise argparse.ArgumentTypeError("at least one group count is required")
    return counts


def _excluded_identities(paths: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    identities: set[str] = set()
    reports: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise ProductRolloutBankError(f"exclude source is missing: {path}")
        rows = questions = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProductRolloutBankError(
                        f"malformed exclude JSONL: {path}"
                    ) from exc
                question = _question(row)
                if question:
                    identities.add(_identity(question))
                    questions += 1
        reports.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "rows": rows,
                "questions": questions,
            }
        )
    return identities, reports


def build_bank(
    sources: list[Path],
    excludes: list[Path],
    output: Path,
    report_path: Path,
    *,
    counts: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    if not sources:
        raise ProductRolloutBankError("at least one source is required")
    if output.exists() or report_path.exists():
        raise ProductRolloutBankError("refusing to replace rollout bank output")
    excluded, exclude_reports = _excluded_identities(excludes)
    counters: Counter[str] = Counter()
    source_reports: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}

    for path in sources:
        if not path.is_file():
            raise ProductRolloutBankError(f"source is missing: {path}")
        source_counter: Counter[str] = Counter()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                counters["raw_rows"] += 1
                source_counter["raw_rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProductRolloutBankError(f"malformed source JSONL: {path}") from exc
                question = _question(row)
                answer = row.get("expected_answer_normalized")
                group = str(row.get("training_group") or row.get("domain") or "")
                if group not in counts:
                    counters["unrequested_group"] += 1
                    source_counter["unrequested_group"] += 1
                    continue
                if not question or answer is None or not str(answer).strip():
                    counters["schema_rejected"] += 1
                    source_counter["schema_rejected"] += 1
                    continue
                if row.get("verification") != "expected_answer_match_v1":
                    counters["verification_rejected"] += 1
                    source_counter["verification_rejected"] += 1
                    continue
                identity = _identity(question)
                if identity in excluded:
                    counters["excluded_overlap"] += 1
                    source_counter["excluded_overlap"] += 1
                    continue
                candidate = {
                    "schema": SCHEMA,
                    "identity_sha256": identity,
                    "question": question,
                    "answer": f"\\boxed{{{str(answer).strip()}}}",
                    "expected_answer_normalized": str(answer).strip(),
                    "task": TASK_BY_GROUP[group],
                    "training_group": group,
                    "source": row.get("source"),
                    "source_prompt_sha256": row.get("prompt_sha256"),
                    "source_verification": row.get("verification"),
                }
                if group == "science":
                    candidate["target"] = str(answer).strip()
                previous = candidates.get(identity)
                if previous is not None:
                    counters["duplicate_questions"] += 1
                    source_counter["duplicate_questions"] += 1
                    continue
                candidates[identity] = candidate
                counters["admissible_rows"] += 1
                source_counter["admissible_rows"] += 1
        source_reports.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "counters": dict(sorted(source_counter.items())),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates.values():
        grouped[candidate["training_group"]].append(candidate)
    selected: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for group, count in counts.items():
        rows = sorted(
            grouped[group],
            key=lambda row: hashlib.sha256(
                f"{seed}\0{group}\0{row['identity_sha256']}".encode()
            ).hexdigest(),
        )
        if len(rows) < count:
            raise ProductRolloutBankError(
                f"group {group!r} has {len(rows)} rows below requested {count}"
            )
        chosen = rows[:count]
        selected.extend(chosen)
        selected_counts[group] = len(chosen)
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}\0output\0{row['identity_sha256']}".encode()
        ).hexdigest()
    )
    if len({row["identity_sha256"] for row in selected}) != len(selected):
        raise ProductRolloutBankError("selected rollout bank contains duplicates")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in selected:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "seed": seed,
        "counts_requested": counts,
        "counts_selected": selected_counts,
        "rows": len(selected),
        "output": str(output.resolve()),
        "output_sha256": digest.hexdigest(),
        "counters": dict(sorted(counters.items())),
        "sources": source_reports,
        "excludes": exclude_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_tmp = report_path.with_suffix(report_path.suffix + ".partial")
    with report_tmp.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(report_tmp, report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--exclude", action="append", type=Path, default=[])
    parser.add_argument("--count", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    args.counts = _parse_counts(args.count)
    return args


def main() -> int:
    args = parse_args()
    report = build_bank(
        args.source,
        args.exclude,
        args.output,
        args.report,
        counts=args.counts,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
