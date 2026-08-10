#!/usr/bin/env python3
"""Independently audit a token-balanced mix against bound evaluation sources."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA = "shohin-token-balanced-mix-eval-overlap-audit-v1"
SOURCE_SCHEMA = "shohin-token-balanced-reasoning-mix-v1"
WORD = re.compile(r"\w+")


class EvalOverlapAuditError(RuntimeError):
    """The mix, report, or evaluation boundary is inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_words(text: str) -> tuple[str, ...]:
    words = tuple(WORD.findall(text.casefold()))
    if not words:
        raise EvalOverlapAuditError("source question normalizes to empty")
    return words


def word_ngrams(text: str, width: int) -> set[tuple[str, ...]]:
    words = normalized_words(text)
    return {
        words[index : index + width]
        for index in range(len(words) - width + 1)
    }


def reference_question(row: dict[str, Any]) -> str:
    for key in ("assessor", "internal_draft"):
        payload = row.get(key)
        if isinstance(payload, dict) and str(payload.get("question", "")).strip():
            return str(payload["question"]).strip()
    question = str(row.get("question", "")).strip()
    if question:
        return question
    raise EvalOverlapAuditError("evaluation row has no source question")


def load_reference_boundary(
    paths: Iterable[Path], width: int
) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]], list[dict[str, Any]]]:
    exact: set[tuple[str, ...]] = set()
    protected: set[tuple[str, ...]] = set()
    receipts = []
    for path in paths:
        frequencies: Counter[tuple[str, ...]] = Counter()
        questions: set[tuple[str, ...]] = set()
        rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                row = json.loads(line)
                question = reference_question(row)
                questions.add(normalized_words(question))
                frequencies.update(word_ngrams(question, width))
        if not rows:
            raise EvalOverlapAuditError(f"empty evaluation reference: {path}")
        unique = {gram for gram, count in frequencies.items() if count == 1}
        exact.update(questions)
        protected.update(unique)
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": rows,
                "normalized_questions": len(questions),
                "unique_word_ngrams": len(unique),
            }
        )
    return exact, protected, receipts


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise EvalOverlapAuditError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.ngram <= 0 or not args.eval_reference:
        raise EvalOverlapAuditError("positive n-gram and references are required")
    report = json.loads(args.report.read_text())
    if report.get("schema") != SOURCE_SCHEMA or report.get("status") != "complete":
        raise EvalOverlapAuditError("source report is not a complete token mix")
    if Path(str(report.get("output", ""))).resolve() != args.data.resolve():
        raise EvalOverlapAuditError("source report binds a different output path")
    data_sha256 = sha256_file(args.data)
    if report.get("output_sha256") != data_sha256:
        raise EvalOverlapAuditError("source report binds a different output hash")

    exact, protected, references = load_reference_boundary(
        args.eval_reference, args.ngram
    )
    declared = report.get("eval_overlap_filter")
    if not isinstance(declared, dict) or declared.get("ngram_size") != args.ngram:
        raise EvalOverlapAuditError("source report lacks the bound overlap filter")
    declared_refs = declared.get("references")
    if not isinstance(declared_refs, list):
        raise EvalOverlapAuditError("source report lacks evaluation receipts")
    declared_path_hash = {
        (str(item.get("path")), str(item.get("sha256"))) for item in declared_refs
    }
    actual_path_hash = {(item["path"], item["sha256"]) for item in references}
    if declared_path_hash != actual_path_hash:
        raise EvalOverlapAuditError("evaluation reference receipts differ")

    rows = 0
    identities: set[tuple[str, ...]] = set()
    exact_hits = 0
    ngram_hits = 0
    groups: Counter[str] = Counter()
    with args.data.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            identity = normalized_words(question)
            if identity in identities:
                raise EvalOverlapAuditError("selected mix repeats a source question")
            identities.add(identity)
            exact_hits += identity in exact
            ngram_hits += bool(word_ngrams(question, args.ngram) & protected)
            groups[str(row.get("training_group", ""))] += 1
    if rows != report.get("selected_rows"):
        raise EvalOverlapAuditError("selected row count differs from source report")

    result = {
        "schema": SCHEMA,
        "status": "complete" if exact_hits == 0 and ngram_hits == 0 else "failed",
        "data": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "source_report": str(args.report.resolve()),
        "source_report_sha256": sha256_file(args.report),
        "ngram_size": args.ngram,
        "rows": rows,
        "unique_normalized_questions": len(identities),
        "groups": dict(sorted(groups.items())),
        "exact_overlap_rows": exact_hits,
        "protected_unique_ngram_overlap_rows": ngram_hits,
        "evaluation_references": references,
        "holdout_used_for_filtering_only": True,
        "capability_output_accessed": False,
    }
    atomic_json(args.output, result)
    if result["status"] != "complete":
        raise EvalOverlapAuditError("selected mix overlaps the evaluation boundary")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--eval-reference", action="append", type=Path, required=True)
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
