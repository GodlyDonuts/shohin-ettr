#!/usr/bin/env python3
"""Materialize the unique IDR1 train identities for predecessor generation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from build_idr1_revision_data import EVAL_SCHEMA, REPORT_SCHEMA, _atomic_json, _atomic_lines
from build_vcr1_revision_data import _load_jsonl, load_source_banks, sha256_file


class KR2PredecessorError(RuntimeError):
    """The IDR1 train lineage differs from the frozen predecessor contract."""


def _bound_path(report: dict[str, Any], key: str, hash_key: str) -> Path:
    path = Path(str(report.get(key, "")))
    if not path.is_file() or report.get(hash_key) != sha256_file(path):
        raise KR2PredecessorError(f"IDR1 {key} is not hash-bound")
    return path


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise KR2PredecessorError("refusing existing KR2 predecessor output")
    source_report = json.loads(args.idr_report.read_text(encoding="utf-8"))
    train_expected = source_report.get("outputs", {}).get("train", {})
    if (
        source_report.get("schema") != REPORT_SCHEMA
        or source_report.get("status") != "complete"
        or Path(train_expected.get("path", "")).resolve() != args.idr_train.resolve()
        or train_expected.get("sha256") != sha256_file(args.idr_train)
        or train_expected.get("rows") != 9655
    ):
        raise KR2PredecessorError("IDR1 train data is not report-bound")
    pairs_path = _bound_path(source_report, "pairs", "pairs_sha256")
    drafts_path = _bound_path(source_report, "drafts", "drafts_sha256")
    bank_entries = source_report.get("banks")
    if not isinstance(bank_entries, list) or len(bank_entries) != 3:
        raise KR2PredecessorError("IDR1 source-bank receipt differs")
    banks: list[Path] = []
    for entry in bank_entries:
        path = Path(str(entry.get("path", "")))
        if not path.is_file() or entry.get("sha256") != sha256_file(path):
            raise KR2PredecessorError("IDR1 source bank is not hash-bound")
        banks.append(path)

    train_rows = _load_jsonl(args.idr_train)
    pairs = {str(row["identity_sha256"]): row for row in _load_jsonl(pairs_path)}
    drafts = {str(row["identity_sha256"]): row for row in _load_jsonl(drafts_path)}
    sources = load_source_banks(banks)
    questions: dict[str, str] = {}
    order: list[str] = []
    for row in train_rows:
        identity = str(row.get("source_identity_sha256"))
        question = str(row.get("question", ""))
        if identity not in questions:
            questions[identity] = question
            order.append(identity)
        elif questions[identity] != question:
            raise KR2PredecessorError("duplicate source has differing IDR1 prompts")
    if len(order) != 5824 or not set(order) <= set(pairs) & set(drafts) & set(sources):
        raise KR2PredecessorError("KR2 predecessor identity geometry differs")

    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for identity in order:
        pair, draft, source = pairs[identity], drafts[identity], sources[identity]
        task = str(source.get("task"))
        if pair.get("task") != task or draft.get("task") != task:
            raise KR2PredecessorError("predecessor task binding differs")
        if str(draft.get("completion", "")) not in questions[identity]:
            raise KR2PredecessorError("original draft is not prompt-bound")
        rows.append(
            {
                "schema": EVAL_SCHEMA,
                "identity_sha256": identity,
                "split": "development",
                "task": task,
                "outcome_class": pair["outcome_class"],
                "question": questions[identity],
                "internal_draft": draft,
                "candidates": pair["candidates"],
                "assessor": source,
                "runtime_fields": ["question"],
                "internal_draft_visible": True,
                "external_candidate_text_visible": False,
                "training_partition_only": True,
            }
        )
        counts[task] += 1

    args.output.mkdir(parents=True)
    data_path = args.output / "development.jsonl"
    data_sha = _atomic_lines(data_path, rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "experiment": "kr2-train-predecessor-generation-v1",
        "training_partition_only": True,
        "source_idr_train": str(args.idr_train.resolve()),
        "source_idr_train_sha256": sha256_file(args.idr_train),
        "source_idr_report": str(args.idr_report.resolve()),
        "source_idr_report_sha256": sha256_file(args.idr_report),
        "counts": {"development": {"pairs": len(rows), **dict(counts)}},
        "outputs": {
            "development": {
                "path": str(data_path.resolve()),
                "sha256": data_sha,
                "rows": len(rows),
            }
        },
        "runtime_fields": ["question"],
        "internal_draft_visible": True,
        "external_candidate_text_visible": False,
        "assessor_fields_visible_to_model": False,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idr-train", type=Path, required=True)
    parser.add_argument("--idr-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = build(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
