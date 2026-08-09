#!/usr/bin/env python3
"""Build selective whole-trajectory commitment data from verified drafts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from build_idr1_revision_data import (
    DRAFT_RECEIPT_SCHEMA,
    DRAFT_SCHEMA,
    IDR1DataError,
    PAIR_SCHEMA,
    SPLIT_SEED,
    _atomic_json,
    _atomic_lines,
)
from build_vcr1_revision_data import (
    SPLITS,
    _load_jsonl,
    load_source_banks,
    sha256_file,
    source_task_prompt,
    training_target,
)
from hf_pcj1_pairwise_judge import assigned_split
from sctr1_commit import (
    KEEP_COMMAND,
    REVISE_COMMAND,
    selective_commit_prompt,
)


TRAIN_SCHEMA = "shohin-sctr1-selective-commit-train-v1"
EVAL_SCHEMA = "shohin-sctr1-selective-commit-eval-v1"
REPORT_SCHEMA = "shohin-sctr1-selective-commit-data-report-v1"


class SCTR1DataError(IDR1DataError):
    """Selective-commit source, draft, or split geometry differs."""


def selective_target(
    pair: dict[str, Any], source: dict[str, Any], draft: dict[str, Any]
) -> tuple[str, str]:
    """Teach preservation when correct and complete replacement when incorrect."""

    if draft.get("correct") is True:
        return KEEP_COMMAND, "keep_verified_draft"
    revision, kind = training_target(pair, source)
    return f"{REVISE_COMMAND}\n{revision}", f"revise_{kind}"


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise SCTR1DataError(f"refusing existing SCTR1 output root: {args.output}")
    receipt = json.loads(args.draft_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != DRAFT_RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("unique_identities") != 8392
        or receipt.get("exact_bank_coverage") is not True
        or receipt.get("output_sha256") != sha256_file(args.drafts)
    ):
        raise SCTR1DataError("SCTR1 draft receipt is incomplete")

    pairs = _load_jsonl(args.pairs)
    sources = load_source_banks(args.banks)
    draft_rows = _load_jsonl(args.drafts)
    drafts = {str(row.get("identity_sha256")): row for row in draft_rows}
    pair_ids = {str(row.get("identity_sha256")) for row in pairs}
    if (
        len(pair_ids) != len(pairs)
        or len(drafts) != len(draft_rows)
        or pair_ids != set(sources)
        or pair_ids != set(drafts)
    ):
        raise SCTR1DataError("SCTR1 pair/source/draft identity coverage differs")

    args.output.mkdir(parents=True)
    train_rows: list[dict[str, Any]] = []
    evaluation: dict[str, list[dict[str, Any]]] = {"development": [], "holdout": []}
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    target_counts: Counter[str] = Counter()
    for pair in pairs:
        identity = str(pair["identity_sha256"])
        source, draft = sources[identity], drafts[identity]
        if pair.get("schema") != PAIR_SCHEMA or draft.get("schema") != DRAFT_SCHEMA:
            raise SCTR1DataError("SCTR1 source schema differs")
        if pair.get("task") != source.get("task") or draft.get("task") != source.get(
            "task"
        ):
            raise SCTR1DataError("SCTR1 task binding differs")
        completion = draft.get("completion")
        if not isinstance(completion, str) or not isinstance(draft.get("correct"), bool):
            raise SCTR1DataError("SCTR1 draft outcome is invalid")

        prompt = selective_commit_prompt(
            source_task_prompt(source), completion, str(source["task"])
        )
        split = assigned_split(identity, args.split_seed)
        command = "keep" if draft["correct"] else "revise"
        counts[split]["pairs"] += 1
        counts[split][str(pair["task"])] += 1
        counts[split][command] += 1
        response, target_kind = selective_target(pair, source, draft)
        if split == "train":
            presentations = 4 if pair["outcome_class"] in ("base_only", "expert_only") else 1
            for presentation in range(presentations):
                train_rows.append(
                    {
                        "schema": TRAIN_SCHEMA,
                        "identity_sha256": hashlib.sha256(
                            f"sctr1\0{identity}\0{presentation}".encode()
                        ).hexdigest(),
                        "source_identity_sha256": identity,
                        "outcome_class": pair["outcome_class"],
                        "presentation": presentation,
                        "question": prompt,
                        "response": response,
                        "target_kind": target_kind,
                        "commit_command": command,
                        "internal_draft_visible": True,
                        "external_candidate_text_visible": False,
                    }
                )
                target_counts[target_kind] += 1
            continue
        evaluation[split].append(
            {
                "schema": EVAL_SCHEMA,
                "identity_sha256": identity,
                "split": split,
                "task": pair["task"],
                "outcome_class": pair["outcome_class"],
                "question": prompt,
                "internal_draft": draft,
                "expected_command": command,
                "candidates": pair["candidates"],
                "assessor": source,
                "runtime_fields": ["question"],
                "internal_draft_visible": True,
                "external_candidate_text_visible": False,
            }
        )

    if (
        len(train_rows) != 9655
        or len(evaluation["development"]) != 1289
        or len(evaluation["holdout"]) != 1279
    ):
        raise SCTR1DataError("SCTR1 matched split geometry differs")
    paths = {
        split: args.output / f"{split}.jsonl"
        for split in ("train", "development", "holdout")
    }
    hashes = {
        "train": _atomic_lines(paths["train"], train_rows),
        "development": _atomic_lines(paths["development"], evaluation["development"]),
        "holdout": _atomic_lines(paths["holdout"], evaluation["holdout"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "drafts": str(args.drafts.resolve()),
        "drafts_sha256": sha256_file(args.drafts),
        "draft_receipt": str(args.draft_receipt.resolve()),
        "draft_receipt_sha256": sha256_file(args.draft_receipt),
        "banks": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.banks
        ],
        "split_seed": args.split_seed,
        "counts": {split: dict(counts[split]) for split in SPLITS},
        "target_counts": dict(target_counts),
        "outputs": {
            split: {
                "path": str(paths[split].resolve()),
                "sha256": hashes[split],
                "rows": len(train_rows) if split == "train" else len(evaluation[split]),
            }
            for split in paths
        },
        "runtime_fields": ["question"],
        "internal_draft_visible": True,
        "external_candidate_text_visible": False,
        "assessor_fields_visible_to_model": False,
        "commitment": "whole_draft_or_whole_revision",
    }
    _atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--draft-receipt", type=Path, required=True)
    parser.add_argument("--bank", dest="banks", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    report = build(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
