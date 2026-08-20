#!/usr/bin/env python3
"""Freeze a source-disjoint, label-free MMLU-Pro GPT-OSS confirmation screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

QUESTION_SCHEMA = "shohin-dense-public-benchmark-question-v1"
SOURCE_SCHEMA = "shohin-q36-mtr-external-validation-source-v1"
RECEIPT_SCHEMA = "shohin-gpt-oss-120b-commit-confirmation-inputs-v1"
BENCHMARK = "mmlu_pro"
ROWS = 256
ALLOWED_ROWS = (256, 1_023)
DEFAULT_SELECTION_SEED = 2026082001


class ConfirmationPreparationError(RuntimeError):
    """The prospective confirmation input contract differed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ConfirmationPreparationError(f"missing or linked input: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfirmationPreparationError(f"unreadable input: {path}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ConfirmationPreparationError("confirmation JSONL differs")
    return rows


def _questions(path: Path) -> list[dict[str, Any]]:
    rows = _jsonl(path)
    identities: set[str] = set()
    for row in rows:
        identity = row.get("id")
        if (
            row.get("schema") != QUESTION_SCHEMA
            or row.get("benchmark") != BENCHMARK
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in identities
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or row.get("response_mode") != "general"
            or any(
                field in row
                for field in ("assessor", "answer", "correct", "gold", "response")
            )
        ):
            raise ConfirmationPreparationError("confirmation question differs")
        identities.add(identity)
    return rows


def _rank(identity: str, selection_seed: int) -> bytes:
    return hashlib.sha256(f"{selection_seed}\0{identity}".encode()).digest()


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise ConfirmationPreparationError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ConfirmationPreparationError("confirmation receipt exists")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    questions = _questions(args.questions)
    excluded = _questions(args.excluded_questions)
    prior_q36 = _jsonl(args.prior_q36_source)
    prior_confirmation_value = getattr(args, "prior_confirmation_source", [])
    if isinstance(prior_confirmation_value, Path):
        prior_confirmation_paths = [prior_confirmation_value]
    else:
        prior_confirmation_paths = list(prior_confirmation_value or [])
    prior_confirmation_groups = [_jsonl(path) for path in prior_confirmation_paths]
    prior_confirmation = [row for group in prior_confirmation_groups for row in group]
    question_ids = {row["id"] for row in questions}
    excluded_ids = {row["id"] for row in excluded}
    prior_q36_ids = {row.get("identity_sha256") for row in prior_q36}
    prior_confirmation_ids = {row.get("identity_sha256") for row in prior_confirmation}
    selection_seed = getattr(args, "selection_seed", DEFAULT_SELECTION_SEED)
    expected_rows = getattr(args, "rows", ROWS)
    if (
        len(questions) != 12_032
        or len(excluded) != 256
        or len(prior_q36_ids) != 1_279
        or expected_rows not in ALLOWED_ROWS
        or any(len(group) not in ALLOWED_ROWS for group in prior_confirmation_groups)
        or len(prior_confirmation_ids)
        != sum(len(group) for group in prior_confirmation_groups)
        or isinstance(selection_seed, bool)
        or not isinstance(selection_seed, int)
        or selection_seed <= 0
        or any(
            row.get("schema") != SOURCE_SCHEMA
            or row.get("split") != "external_validation"
            or not isinstance(row.get("identity_sha256"), str)
            or len(row["identity_sha256"]) != 64
            for row in prior_q36
        )
        or any(
            row.get("schema") != SOURCE_SCHEMA
            or row.get("split") != "external_validation"
            or row.get("task") != BENCHMARK
            or not isinstance(row.get("identity_sha256"), str)
            or len(row["identity_sha256"]) != 64
            for row in prior_confirmation
        )
        or not excluded_ids.issubset(question_ids)
    ):
        raise ConfirmationPreparationError("confirmation identity universe differs")
    ineligible_ids = excluded_ids | prior_q36_ids | prior_confirmation_ids
    eligible = [row for row in questions if row["id"] not in ineligible_ids]
    selected = sorted(
        eligible, key=lambda row: (_rank(row["id"], selection_seed), row["id"])
    )[:expected_rows]
    selected.sort(key=lambda row: row["id"])
    selected_ids = {row["id"] for row in selected}
    if (
        len(selected_ids) != expected_rows
        or selected_ids & excluded_ids
        or selected_ids & prior_q36_ids
        or selected_ids & prior_confirmation_ids
    ):
        raise ConfirmationPreparationError("confirmation selection differs")
    source_rows = [
        {
            "schema": SOURCE_SCHEMA,
            "identity_sha256": row["id"],
            "split": "external_validation",
            "task": BENCHMARK,
            "source_prompt": row["question"],
            "runtime_fields": ["source_prompt"],
            "supervisor_only_fields": ["task"],
        }
        for row in selected
    ]
    source_sha256 = _atomic_lines(args.source_output, source_rows)
    identity_digest = hashlib.sha256(
        ("\n".join(sorted(selected_ids)) + "\n").encode()
    ).hexdigest()
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "benchmark": BENCHMARK,
        "selection_seed": selection_seed,
        "selection_rule": "lowest_sha256_seeded_rank_excluding_all_prior_source_identities",
        "rows": expected_rows,
        "question_universe_rows": len(questions),
        "excluded_public_screen_rows": len(excluded),
        "questions_sha256": sha256_file(args.questions),
        "excluded_questions_sha256": sha256_file(args.excluded_questions),
        "prior_q36_source_sha256": sha256_file(args.prior_q36_source),
        "prior_confirmation_source_count": len(prior_confirmation_paths),
        "prior_confirmation_source_sha256s": [
            sha256_file(path) for path in prior_confirmation_paths
        ],
        "source_output": str(args.source_output.resolve()),
        "source_output_sha256": source_sha256,
        "selected_identity_sha256": identity_digest,
        "excluded_identity_overlap": 0,
        "prior_q36_external_identity_overlap": 0,
        "prior_confirmation_identity_overlap": 0,
        "assessor_access_count": 0,
        "label_or_correctness_field_access_count": 0,
    }
    _atomic_json(args.receipt, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--excluded-questions", type=Path, required=True)
    parser.add_argument("--prior-q36-source", type=Path, required=True)
    parser.add_argument(
        "--prior-confirmation-source", type=Path, action="append", default=[]
    )
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--rows", type=int, choices=ALLOWED_ROWS, default=ROWS)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(prepare(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
