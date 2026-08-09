#!/usr/bin/env python3
"""Build source-disjoint TCS1 train and development candidate sets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-tcs1-candidate-v1"
REPORT_SCHEMA = "shohin-tcs1-candidate-set-report-v1"


class TCS1DataError(RuntimeError):
    """TCS1 source geometry or immutable output differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TCS1DataError(f"missing source: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise TCS1DataError(f"empty source: {path}")
    return rows


def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get("identity_sha256") or "")
        if len(identity) != 64 or identity in result:
            raise TCS1DataError("candidate identity is invalid or duplicated")
        result[identity] = row
    return result


def normalized_candidate(
    *,
    split: str,
    identity: str,
    task: str,
    question: str,
    index: int,
    lineage: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    completion = candidate.get("completion")
    correct = candidate.get("correct")
    if not isinstance(completion, str) or not isinstance(correct, bool):
        raise TCS1DataError("candidate completion or outcome differs")
    return {
        "schema": SCHEMA,
        "split": split,
        "identity_sha256": identity,
        "task": task,
        "question": question,
        "sample_index": index,
        "lineage": lineage,
        "completion": completion,
        "correct": correct,
        "generated_tokens": int(candidate.get("generated_tokens") or 0),
        "max_token_exhausted": bool(candidate.get("max_token_exhausted")),
    }


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts["candidates"] += 1
        counts[f"lineage:{row['lineage']}"] += 1
        counts[f"correct:{row['lineage']}"] += int(row["correct"])
        counts[f"task:{row['task']}"] += 1
    counts["identities"] = len({row["identity_sha256"] for row in rows})
    return dict(sorted(counts.items()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise TCS1DataError("refusing existing TCS1 output")
    pair_rows = load_jsonl(args.pairs)
    train_pairs = keyed([row for row in pair_rows if row.get("split") == "train"])
    development_pairs = keyed(
        [row for row in pair_rows if row.get("split") == "development"]
    )
    if len(train_pairs) != 5824 or len(development_pairs) != 1289:
        raise TCS1DataError("pair split geometry differs")

    train_depth_one = keyed(load_jsonl(args.train_depth_one))
    dev_sources = {
        "depth1": keyed(load_jsonl(args.development_depth_one)),
        "depth2": keyed(load_jsonl(args.development_depth_two)),
        "direct": keyed(load_jsonl(args.development_direct)),
    }
    if set(train_depth_one) != set(train_pairs):
        raise TCS1DataError("train depth-one coverage differs")
    if any(set(source) != set(development_pairs) for source in dev_sources.values()):
        raise TCS1DataError("development candidate coverage differs")
    if set(train_pairs) & set(development_pairs):
        raise TCS1DataError("train and development identities overlap")

    train_rows: list[dict[str, Any]] = []
    for identity, pair in train_pairs.items():
        candidates = pair.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 2:
            raise TCS1DataError("train source pair differs")
        task, question = str(pair.get("task")), str(pair.get("question"))
        if train_depth_one[identity].get("task") != task:
            raise TCS1DataError("train task binding differs")
        for index, (lineage, candidate) in enumerate(
            zip(("base", "expert"), candidates, strict=True)
        ):
            train_rows.append(
                normalized_candidate(
                    split="train",
                    identity=identity,
                    task=task,
                    question=question,
                    index=index,
                    lineage=lineage,
                    candidate=candidate,
                )
            )
        train_rows.append(
            normalized_candidate(
                split="train",
                identity=identity,
                task=task,
                question=question,
                index=2,
                lineage="depth1",
                candidate=train_depth_one[identity],
            )
        )

    development_rows: list[dict[str, Any]] = []
    for identity, pair in development_pairs.items():
        task, question = str(pair.get("task")), str(pair.get("question"))
        for index, lineage in enumerate(("depth1", "depth2", "direct")):
            candidate = dev_sources[lineage][identity]
            if candidate.get("task") != task:
                raise TCS1DataError("development task binding differs")
            development_rows.append(
                normalized_candidate(
                    split="development",
                    identity=identity,
                    task=task,
                    question=question,
                    index=index,
                    lineage=lineage,
                    candidate=candidate,
                )
            )

    args.output.mkdir(parents=True)
    train_path = args.output / "train.jsonl"
    development_path = args.output / "development.jsonl"
    train_sha = atomic_lines(train_path, train_rows)
    development_sha = atomic_lines(development_path, development_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source_disjoint": True,
        "holdout_used": False,
        "sources": {
            "pairs": {
                "path": str(args.pairs.resolve()),
                "sha256": sha256_file(args.pairs),
            },
            "train_depth_one": {
                "path": str(args.train_depth_one.resolve()),
                "sha256": sha256_file(args.train_depth_one),
            },
            "development_depth_one": {
                "path": str(args.development_depth_one.resolve()),
                "sha256": sha256_file(args.development_depth_one),
            },
            "development_depth_two": {
                "path": str(args.development_depth_two.resolve()),
                "sha256": sha256_file(args.development_depth_two),
            },
            "development_direct": {
                "path": str(args.development_direct.resolve()),
                "sha256": sha256_file(args.development_direct),
            },
        },
        "outputs": {
            "train": {
                "path": str(train_path.resolve()),
                "sha256": train_sha,
                **summarize(train_rows),
            },
            "development": {
                "path": str(development_path.resolve()),
                "sha256": development_sha,
                **summarize(development_rows),
            },
        },
    }
    atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--train-depth-one", type=Path, required=True)
    parser.add_argument("--development-depth-one", type=Path, required=True)
    parser.add_argument("--development-depth-two", type=Path, required=True)
    parser.add_argument("--development-direct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = build(parser.parse_args())
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
