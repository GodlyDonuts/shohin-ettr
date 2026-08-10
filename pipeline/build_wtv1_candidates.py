#!/usr/bin/env python3
"""Build a sealed whole-trajectory candidate set from completed edit arms."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-wtv1-whole-trajectory-candidates-v1"


class WTV1Error(RuntimeError):
    """The frozen candidate-set contract was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["pair_identity_sha256"]), str(row["pair_member"])


def load_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise WTV1Error(f"result rows are missing: {path}")
    mapped = {row_key(row): row for row in rows}
    if len(mapped) != len(rows):
        raise WTV1Error(f"result identities repeat: {path}")
    return mapped


def load_gold(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    mapped: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row_key(row)
            if key in mapped:
                raise WTV1Error("diagnostic identities repeat")
            mapped[key] = row
    if not mapped:
        raise WTV1Error("diagnostic source is empty")
    return mapped


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise WTV1Error(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    arms = {
        "dset": load_results(args.dset),
        "gset": load_results(args.gset),
        "iset": load_results(args.iset),
    }
    gold = load_gold(args.diagnostic)
    keys = set(gold)
    if any(set(rows) != keys for rows in arms.values()):
        raise WTV1Error("arm and diagnostic identities differ")

    candidates: list[dict[str, Any]] = []
    groups = 0
    oracle_correct = 0
    family_counts: Counter[str] = Counter()
    unanimous_correct = 0
    unanimous_wrong = 0
    for key in sorted(keys):
        source = gold[key]
        trajectories: dict[str, list[str]] = {}
        for arm, rows in arms.items():
            trajectory = str(rows[key]["executed_trajectory"])
            trajectories.setdefault(trajectory, []).append(arm)
        if len(trajectories) == 1:
            only = next(iter(trajectories))
            if only == str(source["final_response"]):
                unanimous_correct += 1
            else:
                unanimous_wrong += 1
            continue
        groups += 1
        family = str(source["corruption_family"])
        family_counts[family] += 1
        group_identity = hashlib.sha256(
            f"{key[0]}\0{key[1]}".encode("utf-8")
        ).hexdigest()
        contextual_question = (
            f"{source['question']}\n\n"
            "Original model-owned draft:\n"
            f"{source['draft']}"
        )
        group_has_gold = False
        for sample_index, (trajectory, origins) in enumerate(
            sorted(trajectories.items(), key=lambda item: tuple(item[1]))
        ):
            correct = trajectory == str(source["final_response"])
            group_has_gold |= correct
            candidates.append(
                {
                    "schema": SCHEMA,
                    "identity_sha256": group_identity,
                    "pair_identity_sha256": key[0],
                    "pair_member": key[1],
                    "corruption_family": family,
                    "task": str(source["task"]),
                    "sample_index": sample_index,
                    "candidate_origins": origins,
                    "question": contextual_question,
                    "completion": trajectory,
                    "prediction": trajectory,
                    "correct": correct,
                }
            )
        oracle_correct += int(group_has_gold)

    if groups != 125 or oracle_correct != groups:
        raise WTV1Error(
            f"frozen disagreement receipt differs: groups={groups}, oracle={oracle_correct}"
        )
    if unanimous_correct != 1769 or unanimous_wrong != 14:
        raise WTV1Error("frozen unanimous receipt differs")

    if args.output.exists():
        raise WTV1Error(f"refusing existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "sources": {
            "dset": {"path": str(args.dset.resolve()), "sha256": sha256_file(args.dset)},
            "gset": {"path": str(args.gset.resolve()), "sha256": sha256_file(args.gset)},
            "iset": {"path": str(args.iset.resolve()), "sha256": sha256_file(args.iset)},
            "diagnostic": {
                "path": str(args.diagnostic.resolve()),
                "sha256": sha256_file(args.diagnostic),
            },
        },
        "total_rows": len(keys),
        "disagreement_groups": groups,
        "candidate_rows": len(candidates),
        "oracle_disagreement_correct": oracle_correct,
        "unanimous_correct": unanimous_correct,
        "unanimous_wrong": unanimous_wrong,
        "family_disagreement_groups": dict(sorted(family_counts.items())),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset", type=Path, required=True)
    parser.add_argument("--gset", type=Path, required=True)
    parser.add_argument("--iset", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
