#!/usr/bin/env python3
"""Score a completed Q36 model-draft shard for engineering feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pcf1_code_sandbox import (
    mbpp_allocation_setup_receipts_sha256,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
    score_completion,
)

TASKS = ("math500", "bbh_logic", "mbpp")


class Q36MTRDraftPreviewError(RuntimeError):
    """Raised when preview inputs or scoring differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRDraftPreviewError(f"unreadable preview input: {path}") from error
    return rows


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRDraftPreviewError(f"refusing existing preview output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def score_preview(
    candidates_path: Path,
    assessor_board_path: Path,
    *,
    split: str = "development",
) -> dict[str, Any]:
    candidates = [
        row for row in load_jsonl(candidates_path) if row.get("split") == split
    ]
    if not candidates:
        raise Q36MTRDraftPreviewError("preview candidate split is empty")
    candidate_ids = [str(row.get("identity_sha256", "")) for row in candidates]
    if (
        len(candidate_ids) != len(set(candidate_ids))
        or any(len(identity) != 64 for identity in candidate_ids)
        or any(
            row.get("schema") != "shohin-q36-mtr-model-draft-v1"
            or row.get("task") not in TASKS
            or not isinstance(row.get("completion"), str)
            or not row["completion"].strip()
            for row in candidates
        )
    ):
        raise Q36MTRDraftPreviewError("preview candidates differ")

    assessor_rows = load_jsonl(assessor_board_path)
    assessors: dict[str, dict[str, Any]] = {}
    for row in assessor_rows:
        identity = str(row.get("identity_sha256", ""))
        assessor = row.get("assessor")
        if (
            len(identity) != 64
            or identity in assessors
            or row.get("task") not in TASKS
            or not isinstance(assessor, dict)
            or assessor.get("identity_sha256") != identity
            or assessor.get("task") != row.get("task")
        ):
            raise Q36MTRDraftPreviewError("preview assessor board differs")
        assessors[identity] = row
    if any(identity not in assessors for identity in candidate_ids):
        raise Q36MTRDraftPreviewError("preview assessor coverage differs")

    allocation_receipt = qualify_allocation()
    subset_assessors = [assessors[identity]["assessor"] for identity in candidate_ids]
    setup_receipts = qualify_mbpp_assessor_setups(subset_assessors)
    outcomes: list[dict[str, Any]] = []
    for candidate in candidates:
        identity = str(candidate["identity_sha256"])
        assessor_row = assessors[identity]
        if assessor_row["task"] != candidate["task"]:
            raise Q36MTRDraftPreviewError("preview task binding differs")
        result = score_completion(assessor_row["assessor"], candidate["completion"])
        outcomes.append(
            {
                "identity_sha256": identity,
                "task": candidate["task"],
                "correct": bool(result["correct"]),
                "explicit_final_answer": bool(
                    result.get("explicit_final_answer", True)
                ),
                "max_token_exhausted": bool(candidate.get("max_token_exhausted")),
            }
        )

    domains = {}
    for task in TASKS:
        task_rows = [row for row in outcomes if row["task"] == task]
        correct = sum(row["correct"] for row in task_rows)
        domains[task] = {
            "rows": len(task_rows),
            "correct": correct,
            "accuracy": correct / len(task_rows) if task_rows else None,
            "max_token_exhausted": sum(row["max_token_exhausted"] for row in task_rows),
        }
    correct = sum(row["correct"] for row in outcomes)
    allocation_probe_sha256 = hashlib.sha256(
        (json.dumps(allocation_receipt, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    return {
        "schema": "shohin-q36-mtr-draft-preview-v1",
        "status": "complete",
        "interpretation": "exploratory_model_owned_draft_only_not_matched_gate",
        "split": split,
        "rows": len(outcomes),
        "correct": correct,
        "accuracy": correct / len(outcomes),
        "explicit_final_answers": sum(row["explicit_final_answer"] for row in outcomes),
        "max_token_exhausted": sum(row["max_token_exhausted"] for row in outcomes),
        "domains": domains,
        "candidates_sha256": sha256_file(candidates_path),
        "assessor_board_sha256": sha256_file(assessor_board_path),
        "allocation_probe_sha256": allocation_probe_sha256,
        "mbpp_setup_receipts": len(setup_receipts),
        "mbpp_setup_receipts_sha256": mbpp_allocation_setup_receipts_sha256(
            setup_receipts
        ),
        "outcomes": outcomes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--assessor-board", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="development")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = score_preview(args.candidates, args.assessor_board, split=args.split)
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
