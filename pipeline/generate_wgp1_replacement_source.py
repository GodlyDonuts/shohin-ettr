#!/usr/bin/env python3
"""Generate the prospectively frozen, source-disjoint WGP1 confirmation source."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "shohin-wgp1-replacement-source-report-v1"
ROW_SCHEMA = "shohin-wgp1-replacement-source-v1"
PACKAGE = "reasoning-gym"
PACKAGE_VERSION = "0.1.25"
SEED = 20260810
POOL_SIZE = 8192
ROWS_PER_FAMILY = 500
FAMILIES = (
    "basic_arithmetic",
    "chain_sum",
    "decimal_arithmetic",
    "decimal_chain_sum",
    "products",
)


class WGP1SourceError(ValueError):
    """The replacement source does not satisfy its frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def question_sha256(question: object) -> str:
    if not isinstance(question, str) or not question.strip():
        raise WGP1SourceError("generated question is empty")
    return hashlib.sha256(question.strip().encode()).hexdigest()


def protected_questions(paths: list[Path]) -> set[str]:
    protected = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                protected.add(value["source_question_sha256"])
    return protected


def select_family(
    family: str,
    entries: Iterable[dict[str, Any]],
    protected: set[str],
    global_seen: set[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, entry in enumerate(entries):
        counts["pool_scanned"] += 1
        question, answer = entry.get("question"), entry.get("answer")
        digest = question_sha256(question)
        if digest in protected:
            counts["protected_overlap"] += 1
            continue
        if digest in global_seen:
            counts["duplicate"] += 1
            continue
        if answer is None:
            counts["missing_answer"] += 1
            continue
        global_seen.add(digest)
        selected.append(
            {
                "schema": ROW_SCHEMA,
                "family": family,
                "split": "confirmation_seed",
                "generator_seed": SEED,
                "generator_index": index,
                "question": question,
                "answer": str(answer),
                "source_question_sha256": digest,
            }
        )
        if len(selected) == ROWS_PER_FAMILY:
            break
    counts["selected"] = len(selected)
    if len(selected) != ROWS_PER_FAMILY:
        raise WGP1SourceError(f"{family} has only {len(selected)} disjoint rows")
    return selected, counts


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            payload = (
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            handle.write(payload)
            digest.update(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise WGP1SourceError("refusing existing output root")
    if importlib.metadata.version(PACKAGE) != PACKAGE_VERSION:
        raise WGP1SourceError("reasoning-gym version differs")

    import reasoning_gym as rg

    protected = protected_questions(args.protected)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    family_counts = {}
    for family in FAMILIES:
        dataset = rg.create_dataset(family, size=POOL_SIZE, seed=SEED)
        selected, counts = select_family(family, dataset, protected, seen)
        for entry in selected:
            original = dataset[entry["generator_index"]]
            try:
                verified = (
                    dataset.score_answer(answer=original["answer"], entry=original)
                    >= 1.0
                )
            except Exception as error:
                raise WGP1SourceError(f"{family} answer verification failed") from error
            if not verified:
                raise WGP1SourceError(f"{family} generated answer is not verified")
        rows.extend(selected)
        family_counts[family] = dict(sorted(counts.items()))

    if len(rows) != len(FAMILIES) * ROWS_PER_FAMILY:
        raise WGP1SourceError("replacement source cardinality differs")
    if len({row["source_question_sha256"] for row in rows}) != len(rows):
        raise WGP1SourceError("replacement source contains duplicates")

    args.output_root.mkdir(parents=True)
    source = args.output_root / "source.jsonl"
    source_sha256 = atomic_jsonl(source, rows)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "package": {"name": PACKAGE, "version": PACKAGE_VERSION},
        "seed": SEED,
        "pool_size_per_family": POOL_SIZE,
        "rows_per_family": ROWS_PER_FAMILY,
        "families": list(FAMILIES),
        "rows": len(rows),
        "family_counts": family_counts,
        "protected": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.protected
        ],
        "zero_protected_overlap": True,
        "zero_internal_duplicates": True,
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
