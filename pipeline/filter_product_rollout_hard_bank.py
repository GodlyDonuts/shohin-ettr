#!/usr/bin/env python3
"""Build an immutable rollout bank containing only previously unsolved prompts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


BANK_SCHEMA = "shohin-product-rollout-bank-v1"
REPORT_SCHEMA = "shohin-product-rollout-hard-bank-v1"
GROUPS = frozenset({"math", "science"})


class ProductRolloutHardBankError(RuntimeError):
    """The hard-negative bank cannot be constructed without violating its contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    if not normalized:
        raise ProductRolloutHardBankError("question identity is empty")
    return hashlib.sha256(normalized.encode()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ProductRolloutHardBankError(f"input is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProductRolloutHardBankError(
                    f"malformed JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ProductRolloutHardBankError(
                    f"non-object JSONL row at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def build_hard_bank(
    banks: list[Path],
    positives: list[Path],
    output: Path,
    report_path: Path,
    *,
    group: str,
    seed: int,
) -> dict[str, Any]:
    if not banks or not positives:
        raise ProductRolloutHardBankError("banks and positives are required")
    if group not in GROUPS:
        raise ProductRolloutHardBankError(f"unsupported training group: {group}")
    if output.exists() or report_path.exists():
        raise ProductRolloutHardBankError("refusing to replace hard-bank output")

    solved: set[str] = set()
    positive_reports: list[dict[str, Any]] = []
    for path in positives:
        rows = _load_jsonl(path)
        for row in rows:
            question = str(row.get("question") or "").strip()
            identity = str(row.get("source_identity_sha256") or "")
            if not question or identity != _identity(question):
                raise ProductRolloutHardBankError(
                    f"positive identity mismatch in {path}"
                )
            solved.add(identity)
        positive_reports.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "rows": len(rows),
            }
        )

    counters: Counter[str] = Counter()
    unique: dict[str, dict[str, Any]] = {}
    bank_reports: list[dict[str, Any]] = []
    for path in banks:
        rows = _load_jsonl(path)
        for row in rows:
            counters["bank_rows"] += 1
            if row.get("schema") != BANK_SCHEMA:
                raise ProductRolloutHardBankError(f"bank schema mismatch in {path}")
            question = str(row.get("question") or "").strip()
            identity = str(row.get("identity_sha256") or "")
            if not question or identity != _identity(question):
                raise ProductRolloutHardBankError(f"bank identity mismatch in {path}")
            training_group = str(row.get("training_group") or "")
            if training_group not in GROUPS:
                raise ProductRolloutHardBankError(
                    f"bank training group mismatch in {path}"
                )
            previous = unique.get(identity)
            if previous is not None:
                if previous != row:
                    raise ProductRolloutHardBankError(
                        f"conflicting duplicate bank identity: {identity}"
                    )
                counters["duplicate_bank_rows"] += 1
                continue
            unique[identity] = row
        bank_reports.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "rows": len(rows),
            }
        )

    selected = [
        row
        for identity, row in unique.items()
        if row["training_group"] == group and identity not in solved
    ]
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}\0{group}\0{row['identity_sha256']}".encode()
        ).hexdigest()
    )
    if not selected:
        raise ProductRolloutHardBankError("hard-negative bank is empty")
    counters["unique_bank_prompts"] = len(unique)
    counters["solved_prompt_identities"] = len(solved)
    counters["selected_unsolved_prompts"] = len(selected)

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
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "group": group,
        "seed": seed,
        "rows": len(selected),
        "output": str(output.resolve()),
        "output_sha256": digest.hexdigest(),
        "counters": dict(sorted(counters.items())),
        "banks": bank_reports,
        "positives": positive_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_temporary = report_path.with_suffix(report_path.suffix + ".partial")
    with report_temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(report_temporary, report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", action="append", type=Path, required=True)
    parser.add_argument("--positive", action="append", type=Path, required=True)
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_hard_bank(
        args.bank,
        args.positive,
        args.output,
        args.report,
        group=args.group,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
