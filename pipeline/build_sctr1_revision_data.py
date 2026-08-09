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
SHUFFLE_SEED = 2026080825


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


def shuffled_source_commands(
    sources: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Permute commit labels within task and presentation-count strata."""

    groups: dict[tuple[str, int], list[str]] = {}
    for identity, source in sources.items():
        key = (str(source["task"]), int(source["presentations"]))
        groups.setdefault(key, []).append(identity)
    assigned: dict[str, str] = {}
    for identities in groups.values():
        ordered = sorted(
            identities,
            key=lambda identity: hashlib.sha256(
                f"{SHUFFLE_SEED}\0{identity}".encode()
            ).digest(),
        )
        labels = [str(sources[identity]["command"]) for identity in ordered]
        shifted = labels[1:] + labels[:1] if len(labels) > 1 else labels
        assigned.update(zip(ordered, shifted, strict=True))
    if set(assigned) != set(sources):
        raise SCTR1DataError("SCTR1 shuffled command coverage differs")
    return assigned


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
    train_sources: dict[str, dict[str, Any]] = {}
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
            revision, revision_kind = training_target(pair, source)
            train_sources[identity] = {
                "task": pair["task"],
                "presentations": presentations,
                "command": command,
                "revision": revision,
                "revision_kind": revision_kind,
            }
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

    shuffled_commands = shuffled_source_commands(train_sources)
    shuffled_train_rows: list[dict[str, Any]] = []
    changed_sources = sum(
        shuffled_commands[identity] != source["command"]
        for identity, source in train_sources.items()
    )
    if changed_sources == 0:
        raise SCTR1DataError("SCTR1 shuffled commands did not change")
    for row in train_rows:
        identity = str(row["source_identity_sha256"])
        command = shuffled_commands[identity]
        source = train_sources[identity]
        response = (
            KEEP_COMMAND
            if command == "keep"
            else f"{REVISE_COMMAND}\n{source['revision']}"
        )
        shuffled_train_rows.append(
            {
                **row,
                "schema": TRAIN_SCHEMA,
                "identity_sha256": hashlib.sha256(
                    f"sctr1-shuffled\0{identity}\0{row['presentation']}".encode()
                ).hexdigest(),
                "response": response,
                "target_kind": f"shuffled_{command}",
                "commit_command": command,
                "command_supervision_shuffled": True,
            }
        )

    if (
        len(train_rows) != 9655
        or len(shuffled_train_rows) != 9655
        or len(evaluation["development"]) != 1289
        or len(evaluation["holdout"]) != 1279
    ):
        raise SCTR1DataError("SCTR1 matched split geometry differs")
    paths = {
        split: args.output / f"{split}.jsonl"
        for split in ("train", "development", "holdout")
    }
    paths["train_shuffled"] = args.output / "train_shuffled.jsonl"
    hashes = {
        "train": _atomic_lines(paths["train"], train_rows),
        "train_shuffled": _atomic_lines(
            paths["train_shuffled"], shuffled_train_rows
        ),
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
        "shuffle_seed": SHUFFLE_SEED,
        "shuffled_changed_sources": changed_sources,
        "counts": {split: dict(counts[split]) for split in SPLITS},
        "target_counts": dict(target_counts),
        "shuffled_target_counts": dict(
            sorted(Counter(row["target_kind"] for row in shuffled_train_rows).items())
        ),
        "outputs": {
            split: {
                "path": str(paths[split].resolve()),
                "sha256": hashes[split],
                "rows": (
                    len(train_rows)
                    if split in ("train", "train_shuffled")
                    else len(evaluation[split])
                ),
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
