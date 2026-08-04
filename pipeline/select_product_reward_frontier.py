#!/usr/bin/env python3
"""Select exact reward-bank rows where the source demonstrated a solution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-product-reward-frontier-selection-v1"


class RewardFrontierSelectionError(RuntimeError):
    """The reward bank and capability-frontier mask cannot be joined safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RewardFrontierSelectionError(f"non-object row in {path}")
            rows.append(row)
    if not rows:
        raise RewardFrontierSelectionError(f"empty input: {path}")
    return rows


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise RewardFrontierSelectionError(f"refusing existing output: {path}")
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
        raise RewardFrontierSelectionError(f"refusing existing report: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def select_reward_frontier(
    reward_bank: Path,
    frontier_mask: Path,
    output: Path,
    report_output: Path,
) -> dict[str, Any]:
    mask_by_identity: dict[str, dict[str, Any]] = {}
    for row in _rows(frontier_mask):
        identity = str(row.get("source_identity_sha256") or "")
        if not identity or identity in mask_by_identity:
            raise RewardFrontierSelectionError(
                "frontier identities are empty or duplicated"
            )
        if row.get("verification") != "expected_answer_match_v1":
            raise RewardFrontierSelectionError("frontier row is not verified")
        mask_by_identity[identity] = row

    selected: list[dict[str, Any]] = []
    seen_reward: set[str] = set()
    for row in _rows(reward_bank):
        identity = str(row.get("identity_sha256") or "")
        if not identity or identity in seen_reward:
            raise RewardFrontierSelectionError(
                "reward identities are empty or duplicated"
            )
        seen_reward.add(identity)
        mask_row = mask_by_identity.get(identity)
        if mask_row is None:
            continue
        if row.get("task") != "math500":
            raise RewardFrontierSelectionError("selected reward row is not MATH")
        if str(row.get("question") or "") != str(mask_row.get("question") or ""):
            raise RewardFrontierSelectionError("joined question text differs")
        if str(row.get("expected_answer_normalized")) != str(
            mask_row.get("expected_answer_normalized")
        ):
            raise RewardFrontierSelectionError("joined expected answer differs")
        if row.get("answer") is None:
            raise RewardFrontierSelectionError("selected reward row has no answer")
        selected.append(row)

    missing = sorted(set(mask_by_identity) - seen_reward)
    if missing:
        raise RewardFrontierSelectionError(
            f"{len(missing)} frontier identities are absent from reward bank"
        )
    if len(selected) != len(mask_by_identity):
        raise RewardFrontierSelectionError("frontier selection coverage differs")

    output_sha256 = _atomic_jsonl(output, selected)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "reward_bank": str(reward_bank.resolve()),
        "reward_bank_sha256": _sha256(reward_bank),
        "frontier_mask": str(frontier_mask.resolve()),
        "frontier_mask_sha256": _sha256(frontier_mask),
        "reward_rows": len(seen_reward),
        "frontier_rows": len(mask_by_identity),
        "selected_rows": len(selected),
        "output": str(output.resolve()),
        "output_sha256": output_sha256,
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-bank", type=Path, required=True)
    parser.add_argument("--frontier-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    report = select_reward_frontier(
        args.reward_bank,
        args.frontier_mask,
        args.output,
        args.report_output,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
