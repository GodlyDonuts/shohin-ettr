#!/usr/bin/env python3
"""Build source-candidate pairs for the TSVC1 semantic commit head."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any


SCHEMA = "shohin-tsvc1-source-candidate-pairs-v1"


class TSVC1Error(RuntimeError):
    """TSVC1 source/candidate custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_order(identity: str, rows: list[dict[str, Any]], seed: int) -> None:
    rng = random.Random(int(hashlib.sha256(f"{seed}\0{identity}".encode()).hexdigest(), 16))
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["sample_index"] = index


def original_problem(question: str) -> str:
    marker = "Original problem:\n"
    end_marker = "\n\nEmit exactly one edit script."
    start = question.rfind(marker)
    if start < 0:
        raise TSVC1Error("original-problem marker is missing")
    start += len(marker)
    end = question.find(end_marker, start)
    if end < 0:
        raise TSVC1Error("edit-script boundary is missing")
    source = question[start:end].strip()
    if not source:
        raise TSVC1Error("extracted original problem is empty")
    return source


def train_rows(path: Path, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
        grouped[str(row["pair_identity_sha256"])].append(row)
    output: list[dict[str, Any]] = []
    for pair_identity, rows in sorted(grouped.items()):
        if len(rows) != 2 or {str(row["pair_member"]) for row in rows} != {"clean", "fault"}:
            raise TSVC1Error("training pair is not one clean/one fault presentation")
        question = original_problem(str(rows[0]["question"]))
        final = str(rows[0]["final_response"])
        if any(
            original_problem(str(row["question"])) != question
            or str(row["final_response"]) != final
            for row in rows
        ):
            raise TSVC1Error("source or target differs within training pair")
        candidates = []
        for row in rows:
            member = str(row["pair_member"])
            completion = str(row["draft"])
            candidates.append(
                {
                    "schema": SCHEMA,
                    "split": "train",
                    "identity_sha256": pair_identity,
                    "pair_identity_sha256": pair_identity,
                    "pair_member": member,
                    "corruption_family": str(row["corruption_family"]),
                    "task": "bbh_logic",
                    "original_task": str(row["task"]),
                    "question": question,
                    "completion": completion,
                    "prediction": completion,
                    "correct": completion == final,
                }
            )
        if sum(int(row["correct"]) for row in candidates) != 1:
            raise TSVC1Error("training pair does not contain exactly one correct trajectory")
        stable_order(pair_identity, candidates, seed)
        output.extend(candidates)
    return output


def diagnostic_rows(
    candidates_path: Path, diagnostic_path: Path, seed: int, shuffled_source: bool
) -> list[dict[str, Any]]:
    gold = {
        (str(row["pair_identity_sha256"]), str(row["pair_member"])): row
        for row in load_jsonl(diagnostic_path)
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_by_identity: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(candidates_path):
        identity = str(row["identity_sha256"])
        key = (str(row["pair_identity_sha256"]), str(row["pair_member"]))
        if key not in gold:
            raise TSVC1Error("diagnostic candidate has no gold source")
        source_by_identity[identity] = gold[key]
        copied = dict(row)
        copied["schema"] = SCHEMA
        copied["split"] = "diagnostic_shuffled" if shuffled_source else "diagnostic_aligned"
        copied["original_task"] = str(copied["task"])
        copied["task"] = "bbh_logic"
        copied["question"] = original_problem(str(gold[key]["question"]))
        grouped[identity].append(copied)

    if shuffled_source:
        buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
        for identity, source in source_by_identity.items():
            buckets[(str(source["corruption_family"]), str(source["pair_member"]))].append(identity)
        replacement: dict[str, str] = {}
        family_members: dict[str, list[str]] = defaultdict(list)
        for identity, source in source_by_identity.items():
            family_members[str(source["corruption_family"])].append(identity)
        for bucket, identities in sorted(buckets.items()):
            ordered = sorted(identities)
            if len(ordered) < 2:
                alternatives = sorted(
                    identity
                    for identity in family_members[bucket[0]]
                    if identity != ordered[0]
                )
                if not alternatives:
                    raise TSVC1Error(f"shuffled-source family is singleton: {bucket[0]}")
                replacement[ordered[0]] = alternatives[seed % len(alternatives)]
                continue
            shift = 1 + seed % (len(ordered) - 1)
            for index, identity in enumerate(ordered):
                replacement[identity] = ordered[(index + shift) % len(ordered)]
        for identity, rows in grouped.items():
            other = source_by_identity[replacement[identity]]
            for row in rows:
                row["question"] = original_problem(str(other["question"]))
                row["source_shuffled_from"] = replacement[identity]

    output: list[dict[str, Any]] = []
    for identity, rows in sorted(grouped.items()):
        if sum(int(bool(row["correct"])) for row in rows) != 1:
            raise TSVC1Error("diagnostic group does not contain exactly one correct trajectory")
        stable_order(identity, rows, seed)
        output.extend(rows)
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "train":
        rows = train_rows(args.source, args.seed)
        sources = {"source": sha256_file(args.source)}
    else:
        if args.diagnostic is None:
            raise TSVC1Error("diagnostic gold is required")
        rows = diagnostic_rows(
            args.source, args.diagnostic, args.seed, args.mode == "diagnostic-shuffled"
        )
        sources = {
            "source": sha256_file(args.source),
            "diagnostic": sha256_file(args.diagnostic),
        }
    if args.output.exists() or args.report.exists():
        raise TSVC1Error("refusing existing output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "mode": args.mode,
        "seed": args.seed,
        "groups": len(rows) // 2,
        "rows": len(rows),
        "correct_rows": sum(int(bool(row["correct"])) for row in rows),
        "sources": sources,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    temporary_report = args.report.with_suffix(args.report.suffix + ".partial")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_report, args.report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "diagnostic-aligned", "diagnostic-shuffled"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
