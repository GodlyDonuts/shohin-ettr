#!/usr/bin/env python3
"""Build source-only verified-reasoning data matched to VCR1."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_vcr1_revision_data import (
    ORDER_SEED,
    SPLITS,
    _load_jsonl,
    load_source_banks,
    sha256_file,
    source_task_prompt,
    training_target,
)
from hf_pcj1_pairwise_judge import assigned_split

PAIR_SCHEMA = "shohin-cvg1-whole-lineage-pairs-v1"
TRAIN_SCHEMA = "shohin-sdr1-source-only-train-v1"
EVAL_SCHEMA = "shohin-sdr1-source-only-eval-v1"
REPORT_SCHEMA = "shohin-sdr1-source-only-data-report-v1"
SPLIT_SEED = 2026080811


class SDR1DataError(RuntimeError):
    """The SDR1 source data or matching contract differs."""


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise SDR1DataError(f"refusing existing SDR1 output: {path}")
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
        raise SDR1DataError(f"refusing existing SDR1 report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise SDR1DataError(f"refusing existing SDR1 output root: {args.output}")
    args.output.mkdir(parents=True)
    pairs = _load_jsonl(args.pairs)
    sources = load_source_banks(args.banks)
    pair_ids = {str(row.get("identity_sha256")) for row in pairs}
    if len(pair_ids) != len(pairs) or pair_ids != set(sources):
        raise SDR1DataError("SDR1 pair/source identity coverage differs")

    train_rows: list[dict[str, Any]] = []
    evaluation: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "holdout": [],
    }
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    target_counts: Counter[str] = Counter()
    for pair in pairs:
        if pair.get("schema") != PAIR_SCHEMA:
            raise SDR1DataError("SDR1 pair schema differs")
        identity = str(pair["identity_sha256"])
        source = sources[identity]
        if pair.get("task") != source.get("task"):
            raise SDR1DataError("SDR1 task binding differs")
        prompt = source_task_prompt(source)
        if str(pair.get("question")) not in prompt:
            raise SDR1DataError("SDR1 question/source binding differs")
        split = assigned_split(identity, args.split_seed)
        counts[split]["pairs"] += 1
        counts[split][str(pair["task"])] += 1
        counts[split][str(pair["outcome_class"])] += 1
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
                            f"sdr1\0{identity}\0{presentation}".encode()
                        ).hexdigest(),
                        "source_identity_sha256": identity,
                        "outcome_class": pair["outcome_class"],
                        "presentation": presentation,
                        "question": prompt,
                        "response": target,
                        "target_kind": target_kind,
                        "candidate_text_visible": False,
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
                "candidates": pair["candidates"],
                "assessor": source,
                "runtime_fields": ["question"],
                "candidate_text_visible": False,
            }
        )

    if not train_rows or any(not evaluation[split] for split in evaluation):
        raise SDR1DataError("SDR1 output split is empty")
    paths = {
        "train": args.output / "train.jsonl",
        "development": args.output / "development.jsonl",
        "holdout": args.output / "holdout.jsonl",
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
        "banks": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.banks
        ],
        "split_seed": args.split_seed,
        "matched_vcr1_order_seed": ORDER_SEED,
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
        "candidate_text_visible": False,
        "assessor_fields_visible_to_model": False,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
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
