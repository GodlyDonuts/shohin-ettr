#!/usr/bin/env python3
"""Build matched source-plus-internal-draft IDR1 revision data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_vcr1_revision_data import (
    SPLITS,
    _load_jsonl,
    load_source_banks,
    sha256_file,
    source_task_prompt,
    training_target,
)
from hf_pcj1_pairwise_judge import assigned_split

PAIR_SCHEMA = "shohin-cvg1-whole-lineage-pairs-v1"
DRAFT_SCHEMA = "shohin-idr1-internal-drafts-v1"
DRAFT_RECEIPT_SCHEMA = "shohin-idr1-internal-drafts-receipt-v1"
TRAIN_SCHEMA = "shohin-idr1-revision-train-v1"
EVAL_SCHEMA = "shohin-idr1-revision-eval-v1"
REPORT_SCHEMA = "shohin-idr1-revision-data-report-v1"
SPLIT_SEED = 2026080811


class IDR1DataError(RuntimeError):
    """IDR1 draft, source, or partitioning differs from the frozen contract."""


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise IDR1DataError(f"refusing existing IDR1 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
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
        raise IDR1DataError(f"refusing existing IDR1 report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def internal_revision_prompt(task_prompt: str, draft: str, task: str) -> str:
    format_instruction = (
        "Return only executable Python code, without Markdown fences."
        if task == "mbpp"
        else "Return a complete corrected solution with the exact final answer in \\boxed{}."
    )
    return (
        "Solve the original problem by checking and revising the model's earlier draft. "
        "The draft may contain useful steps or errors; do not merely critique it.\n\n"
        f"Original problem:\n{task_prompt}\n\nInternal draft:\n{draft}\n\n"
        f"{format_instruction}\n\nOriginal problem:\n{task_prompt}"
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise IDR1DataError(f"refusing existing IDR1 output root: {args.output}")
    receipt = json.loads(args.draft_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != DRAFT_RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("unique_identities") != 8392
        or receipt.get("exact_bank_coverage") is not True
        or receipt.get("output_sha256") != sha256_file(args.drafts)
    ):
        raise IDR1DataError("IDR1 draft receipt is incomplete")
    args.output.mkdir(parents=True)
    pairs = _load_jsonl(args.pairs)
    sources = load_source_banks(args.banks)
    drafts = {str(row.get("identity_sha256")): row for row in _load_jsonl(args.drafts)}
    pair_ids = {str(row.get("identity_sha256")) for row in pairs}
    if (
        len(pair_ids) != len(pairs)
        or pair_ids != set(sources)
        or pair_ids != set(drafts)
    ):
        raise IDR1DataError("IDR1 pair/source/draft identity coverage differs")

    train_rows: list[dict[str, Any]] = []
    evaluation: dict[str, list[dict[str, Any]]] = {"development": [], "holdout": []}
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    target_counts: Counter[str] = Counter()
    for pair in pairs:
        if pair.get("schema") != PAIR_SCHEMA:
            raise IDR1DataError("IDR1 pair schema differs")
        identity = str(pair["identity_sha256"])
        source, draft = sources[identity], drafts[identity]
        if draft.get("schema") != DRAFT_SCHEMA:
            raise IDR1DataError("IDR1 draft schema differs")
        if pair.get("task") != source.get("task") or draft.get("task") != source.get(
            "task"
        ):
            raise IDR1DataError("IDR1 task binding differs")
        prompt = internal_revision_prompt(
            source_task_prompt(source),
            str(draft.get("completion", "")),
            str(source["task"]),
        )
        split = assigned_split(identity, args.split_seed)
        counts[split]["pairs"] += 1
        counts[split][str(pair["task"])] += 1
        counts[split][str(pair["outcome_class"])] += 1
        counts[split]["draft_correct"] += int(bool(draft.get("correct")))
        target, target_kind = training_target(pair, source)
        if split == "train":
            presentations = (
                4 if pair["outcome_class"] in ("base_only", "expert_only") else 1
            )
            for presentation in range(presentations):
                train_rows.append(
                    {
                        "schema": TRAIN_SCHEMA,
                        "identity_sha256": hashlib.sha256(
                            f"idr1\0{identity}\0{presentation}".encode()
                        ).hexdigest(),
                        "source_identity_sha256": identity,
                        "outcome_class": pair["outcome_class"],
                        "presentation": presentation,
                        "question": prompt,
                        "response": target,
                        "target_kind": target_kind,
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
        raise IDR1DataError("IDR1 matched split geometry differs")
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
    }
    _atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--draft-receipt", type=Path, required=True)
    parser.add_argument(
        "--bank", dest="banks", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    report = build(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
