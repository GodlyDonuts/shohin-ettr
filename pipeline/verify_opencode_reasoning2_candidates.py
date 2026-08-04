#!/usr/bin/env python3
"""Join OCR2 traces to pinned source problems and independently replay tests."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from curate_apps import build_test_grams, has_eval_overlap, run_solution
from select_opencode_reasoning2_candidates import SCHEMA as CANDIDATE_SCHEMA


SCHEMA = "shohin-opencode-reasoning2-execution-verified-v1"
REPORT_SCHEMA = "shohin-opencode-reasoning2-execution-report-v1"
DATASETS = frozenset({"apps", "taco", "code_contests"})
DANGEROUS_MODULES = frozenset(
    {
        "ctypes",
        "http",
        "multiprocessing",
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "threading",
        "urllib",
    }
)
DANGEROUS_CALLS = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "getattr",
        "open",
        "popen",
        "setattr",
        "system",
    }
)


class OpenCodeReasoningVerificationError(RuntimeError):
    """The candidate/source contract cannot be verified safely."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def solution_is_safe(code: str) -> bool:
    """Reject capabilities unnecessary for an isolated contest program."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.split(".", 1)[0] in DANGEROUS_MODULES for alias in node.names
            ):
                return False
        elif isinstance(node, ast.ImportFrom):
            if str(node.module or "").split(".", 1)[0] in DANGEROUS_MODULES:
                return False
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_CALLS:
                return False
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in DANGEROUS_CALLS
            ):
                return False
    return True


def _json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def stdio_cases(
    value: Any, *, max_tests: int, max_case_chars: int
) -> list[tuple[str, str]] | None:
    item = _json_object(value)
    if item is None or item.get("fn_name"):
        return None
    inputs = item.get("inputs")
    outputs = item.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return None
    cases: list[tuple[str, str]] = []
    for stdin, expected in zip(inputs, outputs):
        stdin_text, expected_text = str(stdin), str(expected)
        if len(stdin_text) > max_case_chars or len(expected_text) > max_case_chars:
            continue
        cases.append((stdin_text, expected_text))
        if max_tests and len(cases) >= max_tests:
            break
    return cases or None


def code_contests_cases(
    row: dict[str, Any], *, max_tests: int, max_case_chars: int
) -> list[tuple[str, str]] | None:
    cases: list[tuple[str, str]] = []
    for key in ("public_tests", "generated_tests", "private_tests"):
        tests = row.get(key) or {}
        if not isinstance(tests, dict):
            continue
        inputs, outputs = tests.get("input") or [], tests.get("output") or []
        for stdin, expected in zip(inputs, outputs):
            stdin_text, expected_text = str(stdin), str(expected)
            if len(stdin_text) > max_case_chars or len(expected_text) > max_case_chars:
                continue
            cases.append((stdin_text, expected_text))
            if max_tests and len(cases) >= max_tests:
                return cases
    return cases or None


def question_and_cases(
    dataset: str,
    row: dict[str, Any],
    *,
    max_tests: int,
    max_case_chars: int,
) -> tuple[str, list[tuple[str, str]] | None]:
    if dataset == "code_contests":
        question = str(row.get("description") or "").strip()
        cases = code_contests_cases(
            row, max_tests=max_tests, max_case_chars=max_case_chars
        )
    else:
        question = str(row.get("question") or "").strip()
        cases = stdio_cases(
            row.get("input_output"),
            max_tests=max_tests,
            max_case_chars=max_case_chars,
        )
    return question, cases


def iter_source_rows(paths: list[Path], batch_size: int) -> Iterable[dict[str, Any]]:
    for path in paths:
        if not path.is_file():
            raise OpenCodeReasoningVerificationError(f"missing source file: {path}")
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.ParquetFile(path)
            for batch in table.iter_batches(batch_size=batch_size):
                yield from batch.to_pylist()
        elif path.suffix in {".json", ".jsonl"}:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        yield json.loads(line)
        else:
            raise OpenCodeReasoningVerificationError(
                f"unsupported source suffix: {path}"
            )


def read_candidates(
    path: Path, *, dataset: str, shard_index: int, shard_count: int
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("schema") != CANDIDATE_SCHEMA:
                raise OpenCodeReasoningVerificationError("candidate schema differs")
            if row.get("source_dataset") != dataset:
                continue
            if row.get("source_split") != "train":
                raise OpenCodeReasoningVerificationError("non-train candidate found")
            identity = str(row.get("identity_sha256") or "")
            if len(identity) != 64:
                raise OpenCodeReasoningVerificationError("invalid candidate identity")
            if int(identity[:16], 16) % shard_count != shard_index:
                continue
            index = int(row["source_index"])
            if index in selected:
                raise OpenCodeReasoningVerificationError(
                    f"duplicate source index for {dataset}: {index}"
                )
            selected[index] = row
    return selected


def verify_match(
    match: tuple[int, dict[str, Any], dict[str, Any]],
    *,
    dataset: str,
    test_grams: set[str],
    ngram: int,
    max_tests: int,
    max_case_chars: int,
    min_tests: int,
    timeout: float,
) -> tuple[dict[str, Any] | None, str | None]:
    source_index, candidate, source_row = match
    question, cases = question_and_cases(
        dataset,
        source_row,
        max_tests=max_tests,
        max_case_chars=max_case_chars,
    )
    if not question:
        return None, "missing_question"
    if has_eval_overlap(question, test_grams, ngram):
        return None, "eval_overlap"
    if not cases or len(cases) < min_tests:
        return None, "insufficient_cases"
    solution = str(candidate.get("solution") or "")
    if not solution_is_safe(solution):
        return None, "unsafe_solution"
    if not run_solution(solution, cases, timeout):
        return None, "execution"
    clean = dict(candidate)
    clean.update(
        {
            "schema": SCHEMA,
            "question": question,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "source_index": source_index,
            "training_group": "verified_code_reasoning",
            "verification": "execution_verified_source_tests",
            "verified_cases": len(cases),
        }
    )
    return clean, None


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise OpenCodeReasoningVerificationError(f"refusing to replace {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise OpenCodeReasoningVerificationError(f"stale partial exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--source-files", nargs="+", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--evals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-tests", type=int, default=0)
    parser.add_argument("--min-tests", type=int, default=2)
    parser.add_argument("--max-case-chars", type=int, default=20000)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--ngram", type=int, default=13)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if (
        min(
            args.shard_count,
            args.workers,
            args.batch_size,
            args.min_tests,
            args.max_case_chars,
            args.ngram,
        )
        <= 0
        or args.max_tests < 0
        or args.timeout <= 0
    ):
        raise ValueError("verification bounds must be positive; max-tests non-negative")
    if args.output.exists() or args.report.exists():
        raise OpenCodeReasoningVerificationError("verification output already exists")

    candidates = read_candidates(
        args.candidates,
        dataset=args.dataset,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    matches: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    remaining = dict(candidates)
    source_rows = 0
    for source_index, source_row in enumerate(
        iter_source_rows(args.source_files, args.batch_size)
    ):
        source_rows += 1
        candidate = remaining.pop(source_index, None)
        if candidate is not None:
            matches.append((source_index, candidate, source_row))
    if remaining:
        raise OpenCodeReasoningVerificationError(
            f"{len(remaining)} candidate source indices were not found"
        )

    test_grams = build_test_grams(args.evals, args.ngram)

    def verify(match: tuple[int, dict[str, Any], dict[str, Any]]):
        return verify_match(
            match,
            dataset=args.dataset,
            test_grams=test_grams,
            ngram=args.ngram,
            max_tests=args.max_tests,
            max_case_chars=args.max_case_chars,
            min_tests=args.min_tests,
            timeout=args.timeout,
        )

    counters: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for clean, drop in executor.map(verify, matches):
            counters["checked"] += 1
            if clean is None:
                counters[f"dropped_{drop}"] += 1
            else:
                kept.append(clean)
                counters["kept"] += 1
    kept.sort(key=lambda row: row["identity_sha256"])
    _atomic_jsonl(args.output, kept)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "candidate_file": str(args.candidates.resolve()),
        "candidate_file_sha256": sha256_file(args.candidates),
        "dataset": args.dataset,
        "source_revision": args.source_revision,
        "source_files": [str(path.resolve()) for path in args.source_files],
        "source_file_sha256": {
            str(path.resolve()): sha256_file(path) for path in args.source_files
        },
        "source_rows": source_rows,
        "selected_candidates": len(candidates),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "max_tests": args.max_tests,
        "min_tests": args.min_tests,
        "max_case_chars": args.max_case_chars,
        "timeout": args.timeout,
        "counters": dict(sorted(counters.items())),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    report_partial = args.report.with_suffix(args.report.suffix + ".partial")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with report_partial.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(report_partial, args.report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
