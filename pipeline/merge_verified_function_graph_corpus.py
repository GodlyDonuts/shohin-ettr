#!/usr/bin/env python3
"""Merge a complete set of independently verified function-graph shards."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_verified_function_graph_corpus import SCHEMA, _grams, evaluation_grams


MERGE_SCHEMA = "shohin-verified-function-graph-merge-v2"


class FunctionGraphMergeError(RuntimeError):
    """Function-graph shards cannot be admitted as one complete corpus."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_shards(
    shard_paths: list[Path],
    report_paths: list[Path],
    *,
    expected_shards: int,
    expected_rows: int,
    expected_schema: str = SCHEMA,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(shard_paths) != len(report_paths) or len(shard_paths) != expected_shards:
        raise FunctionGraphMergeError("shard and report cardinality differs")
    reports = [json.loads(path.read_text()) for path in report_paths]
    indices = {int(report.get("shard_index", -1)) for report in reports}
    if indices != set(range(expected_shards)):
        raise FunctionGraphMergeError("shard indices differ")
    seeds = {int(report.get("seed", -1)) for report in reports}
    shard_counts = {int(report.get("shard_count", -1)) for report in reports}
    eval_sources = {
        json.dumps(report.get("evaluation_sources"), sort_keys=True)
        for report in reports
    }
    ngram_widths = {int(report.get("ngram_width", -1)) for report in reports}
    if (
        len(seeds) != 1
        or shard_counts != {expected_shards}
        or len(eval_sources) != 1
        or len(ngram_widths) != 1
        or min(ngram_widths) <= 0
    ):
        raise FunctionGraphMergeError("shard generation contracts differ")

    by_index = {int(report["shard_index"]): report for report in reports}
    inputs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path, report_path in zip(shard_paths, report_paths, strict=True):
        report = json.loads(report_path.read_text())
        if (
            report.get("schema") != expected_schema
            or report.get("status") != "complete"
        ):
            raise FunctionGraphMergeError("shard report is not complete")
        if Path(str(report.get("output"))).resolve() != path.resolve():
            raise FunctionGraphMergeError("shard output path differs")
        if _sha256(path) != str(report.get("output_sha256")):
            raise FunctionGraphMergeError("shard hash differs")
        loaded = [json.loads(line) for line in path.read_bytes().splitlines() if line]
        if len(loaded) != int(report.get("rows", -1)):
            raise FunctionGraphMergeError("shard row count differs")
        counters = report.get("counters") or {}
        if int(counters.get("kept", -1)) != len(loaded):
            raise FunctionGraphMergeError("shard kept counter differs")
        if int(counters.get("generated", -1)) < len(loaded):
            raise FunctionGraphMergeError("shard generated counter differs")
        rows.extend(loaded)
        inputs.append(
            {
                "shard_index": int(report["shard_index"]),
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "report": str(report_path.resolve()),
                "report_sha256": _sha256(report_path),
                "rows": len(loaded),
            }
        )
    if len(rows) != expected_rows:
        raise FunctionGraphMergeError("merged row count differs")
    identities = [int(row.get("global_identity", -1)) for row in rows]
    if set(identities) != set(range(expected_rows)):
        raise FunctionGraphMergeError("merged identities differ")
    questions = [str(row.get("question") or "").strip().casefold() for row in rows]
    if any(not question for question in questions) or len(set(questions)) != len(
        questions
    ):
        raise FunctionGraphMergeError("merged questions differ")
    if any(row.get("schema") != expected_schema for row in rows):
        raise FunctionGraphMergeError("merged row schema differs")
    seed = next(iter(seeds))
    if any(int(row.get("seed", -1)) != seed for row in rows):
        raise FunctionGraphMergeError("merged row seed differs")
    if any(
        row.get("verification") != "generated_reference_passes_randomized_tests"
        for row in rows
    ):
        raise FunctionGraphMergeError("merged verification differs")
    for row in rows:
        verification_sha = hashlib.sha256(
            (str(row["response"]) + "\n" + "\n".join(row["tests"])).encode()
        ).hexdigest()
        if verification_sha != row.get("verification_sha256"):
            raise FunctionGraphMergeError("merged verification hash differs")
    evaluation_sources = json.loads(next(iter(eval_sources)))
    blocked, replayed_sources = evaluation_grams(
        [Path(receipt["path"]) for receipt in evaluation_sources],
        next(iter(ngram_widths)),
    )
    if replayed_sources != evaluation_sources:
        raise FunctionGraphMergeError("evaluation source receipt differs")
    if any(
        blocked.intersection(_grams(row["question"], next(iter(ngram_widths))))
        for row in rows
    ):
        raise FunctionGraphMergeError("merged evaluation overlap differs")
    split_counts = Counter(str(row.get("split")) for row in rows)
    if set(split_counts) != {"train", "confirmation"}:
        raise FunctionGraphMergeError("merged split differs")
    rows.sort(key=lambda row: int(row["global_identity"]))
    return rows, {
        "schema": MERGE_SCHEMA,
        "status": "complete",
        "rows": len(rows),
        "source_schema": expected_schema,
        "seed": seed,
        "shards": expected_shards,
        "ngram_width": next(iter(ngram_widths)),
        "split_counts": dict(sorted(split_counts.items())),
        "family_counts": dict(
            sorted(Counter(str(row["family"]) for row in rows).items())
        ),
        "evaluation_sources": evaluation_sources,
        "inputs": sorted(inputs, key=lambda row: row["shard_index"]),
        "source_report_indices": sorted(by_index),
    }


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise FunctionGraphMergeError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
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
        raise FunctionGraphMergeError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--shard-report", type=Path, action="append", required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--confirmation-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-schema", default=SCHEMA)
    args = parser.parse_args()
    rows, report = merge_shards(
        args.shard,
        args.shard_report,
        expected_shards=args.expected_shards,
        expected_rows=args.expected_rows,
        expected_schema=args.expected_schema,
    )
    train = [row for row in rows if row["split"] == "train"]
    confirmation = [row for row in rows if row["split"] == "confirmation"]
    report["train_output_sha256"] = _atomic_lines(args.train_output, train)
    report["train_output"] = str(args.train_output.resolve())
    report["confirmation_output_sha256"] = _atomic_lines(
        args.confirmation_output, confirmation
    )
    report["confirmation_output"] = str(args.confirmation_output.resolve())
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
