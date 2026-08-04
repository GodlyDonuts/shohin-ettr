#!/usr/bin/env python3
"""Select deterministic high-confidence OpenCodeReasoning-2 Python candidates.

This is deliberately not an admission verifier. It reduces the 1.4M-row
synthetic source to one bounded candidate per original train problem while
preserving every field required to replay the program against the pinned
upstream tests. A later process performs that execution gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA = "shohin-opencode-reasoning2-candidate-v1"
REPORT_SCHEMA = "shohin-opencode-reasoning2-candidate-report-v1"
MERGE_SCHEMA = "shohin-opencode-reasoning2-candidate-merge-v1"
DEFAULT_DATASETS = frozenset({"apps", "taco", "code_contests"})
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class OpenCodeReasoningCandidateError(RuntimeError):
    """The source cannot be reduced without violating the frozen contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _pass_rate(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if 0.0 <= result <= 1.0 else None


def _complete_reasoning(value: Any, *, min_chars: int, max_chars: int) -> str | None:
    text = str(value or "").strip()
    if not min_chars <= len(text) <= max_chars:
        return None
    if text.count(THINK_OPEN) != 1 or text.count(THINK_CLOSE) != 1:
        return None
    if text.index(THINK_OPEN) > text.index(THINK_CLOSE):
        return None
    if not text.split(THINK_CLOSE, 1)[1].strip():
        return None
    return text


def _syntax_valid_solution(value: Any, *, max_chars: int) -> str | None:
    import ast

    code = str(value or "").strip()
    if not code or len(code) > max_chars:
        return None
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code


def _contains_solution(response: str, solution: str) -> bool:
    compact_response = re.sub(r"\s+", "", response)
    compact_solution = re.sub(r"\s+", "", solution)
    return bool(compact_solution) and compact_solution in compact_response


def _source_key(row: dict[str, Any]) -> str | None:
    dataset = str(row.get("dataset") or "").strip()
    split = str(row.get("split") or "").strip()
    index = str(row.get("index") or "").strip()
    question_id = str(row.get("question_id") or "").strip()
    if not dataset or split != "train" or not index or not question_id:
        return None
    try:
        normalized_index = str(int(index))
    except ValueError:
        return None
    return "\0".join((dataset, split, normalized_index, question_id))


def candidate_from_row(
    row: dict[str, Any],
    *,
    allowed_datasets: frozenset[str],
    min_response_chars: int,
    max_response_chars: int,
    max_code_chars: int,
) -> tuple[dict[str, Any] | None, str | None]:
    dataset = str(row.get("dataset") or "").strip()
    if dataset not in allowed_datasets:
        return None, "dataset"
    if str(row.get("judgement") or "").strip().casefold() != "right":
        return None, "judgement"
    pass_rate = _pass_rate(row.get("pass_rate"))
    if pass_rate is None or pass_rate < 1.0:
        return None, "pass_rate"
    key = _source_key(row)
    if key is None:
        return None, "source_identity"
    response = _complete_reasoning(
        row.get("r1_generation"),
        min_chars=min_response_chars,
        max_chars=max_response_chars,
    )
    if response is None:
        return None, "reasoning"
    solution = _syntax_valid_solution(row.get("solution"), max_chars=max_code_chars)
    if solution is None:
        return None, "solution"
    if not _contains_solution(response, solution):
        return None, "solution_not_embedded"
    license_name = str(row.get("license") or "").strip()
    if not license_name:
        return None, "license"
    dataset_name, split, index, question_id = key.split("\0")
    identity = hashlib.sha256(key.encode()).hexdigest()
    candidate = {
        "schema": SCHEMA,
        "identity_sha256": identity,
        "question_id": question_id,
        "source_dataset": dataset_name,
        "source_split": split,
        "source_index": int(index),
        "source_platform": str(row.get("source") or "").strip(),
        "source_license": license_name,
        "difficulty": str(row.get("difficulty") or "").strip(),
        "ocr2_id": str(row.get("id") or "").strip(),
        "reported_pass_rate": pass_rate,
        "reported_judgement": "right",
        "response": response,
        "solution": solution,
        "response_chars": len(response),
        "solution_chars": len(solution),
        "training_group": "code",
        "verification": "pending_source_test_replay",
    }
    return candidate, None


def _candidate_order(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row["response_chars"]),
        int(row["solution_chars"]),
        str(row["ocr2_id"]),
    )


def select_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    allowed_datasets: frozenset[str] = DEFAULT_DATASETS,
    min_response_chars: int = 256,
    max_response_chars: int = 24_000,
    max_code_chars: int = 20_000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counters: Counter[str] = Counter()
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        counters["source_rows"] += 1
        candidate, drop = candidate_from_row(
            row,
            allowed_datasets=allowed_datasets,
            min_response_chars=min_response_chars,
            max_response_chars=max_response_chars,
            max_code_chars=max_code_chars,
        )
        if candidate is None:
            counters[f"dropped_{drop}"] += 1
            continue
        identity = candidate["identity_sha256"]
        previous = selected.get(identity)
        if previous is None:
            selected[identity] = candidate
            counters["candidate_identities"] += 1
        else:
            counters["duplicate_candidate_rows"] += 1
            if _candidate_order(candidate) < _candidate_order(previous):
                selected[identity] = candidate
                counters["candidate_replacements"] += 1
    result = sorted(selected.values(), key=lambda row: row["identity_sha256"])
    counters["selected_rows"] = len(result)
    return result, dict(sorted(counters.items()))


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise OpenCodeReasoningCandidateError(f"refusing to replace {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise OpenCodeReasoningCandidateError(f"stale partial exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise OpenCodeReasoningCandidateError(f"refusing to replace {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise OpenCodeReasoningCandidateError(f"stale partial exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, path)


def scan_parquet(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow.parquet as pq

    source = Path(args.input)
    output = Path(args.output)
    report_path = Path(args.report)
    if not source.is_file():
        raise OpenCodeReasoningCandidateError(f"missing input parquet: {source}")
    if output.exists() or report_path.exists():
        raise OpenCodeReasoningCandidateError("scan output already exists")
    table = pq.ParquetFile(source)

    def rows() -> Iterable[dict[str, Any]]:
        for batch in table.iter_batches(batch_size=args.batch_size):
            yield from batch.to_pylist()

    selected, counters = select_candidates(
        rows(),
        allowed_datasets=frozenset(args.allowed_datasets.split(",")),
        min_response_chars=args.min_response_chars,
        max_response_chars=args.max_response_chars,
        max_code_chars=args.max_code_chars,
    )
    _atomic_write_jsonl(output, selected)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "shard_index": args.shard_index,
        "input": str(source.resolve()),
        "input_sha256": sha256_file(source),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "allowed_datasets": sorted(frozenset(args.allowed_datasets.split(","))),
        "min_response_chars": args.min_response_chars,
        "max_response_chars": args.max_response_chars,
        "max_code_chars": args.max_code_chars,
        "counters": counters,
    }
    _atomic_write_json(report_path, report)
    return report


def _read_candidate_files(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter()
    for path in paths:
        if not path.is_file():
            raise OpenCodeReasoningCandidateError(f"missing candidate shard: {path}")
        seen_in_file: set[str] = set()
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                counters["input_rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise OpenCodeReasoningCandidateError(
                        f"malformed JSONL: {path}"
                    ) from exc
                if row.get("schema") != SCHEMA:
                    raise OpenCodeReasoningCandidateError(
                        f"candidate schema differs: {path}"
                    )
                identity = str(row.get("identity_sha256") or "")
                if len(identity) != 64:
                    raise OpenCodeReasoningCandidateError(
                        f"invalid candidate identity: {path}"
                    )
                if identity in seen_in_file:
                    raise OpenCodeReasoningCandidateError(
                        f"duplicate identity within {path}"
                    )
                seen_in_file.add(identity)
                previous = selected.get(identity)
                if previous is None:
                    selected[identity] = row
                else:
                    counters["cross_shard_duplicates"] += 1
                    if _candidate_order(row) < _candidate_order(previous):
                        selected[identity] = row
                        counters["cross_shard_replacements"] += 1
    result = sorted(selected.values(), key=lambda row: row["identity_sha256"])
    counters["selected_rows"] = len(result)
    return result, counters


def merge_candidates(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    report_path = Path(args.report)
    paths = sorted(Path().glob(args.inputs_glob))
    if len(paths) != args.expected_shards:
        raise OpenCodeReasoningCandidateError(
            f"expected {args.expected_shards} candidate shards, found {len(paths)}"
        )
    reports = [path.with_suffix(".report.json") for path in paths]
    for path, report_path_for_shard in zip(paths, reports, strict=True):
        if not report_path_for_shard.is_file():
            raise OpenCodeReasoningCandidateError(
                f"missing shard report: {report_path_for_shard}"
            )
        report = json.loads(report_path_for_shard.read_text(encoding="utf-8"))
        if report.get("status") != "complete" or report.get(
            "output_sha256"
        ) != sha256_file(path):
            raise OpenCodeReasoningCandidateError(f"shard report mismatch: {path}")
        if report.get("dataset_revision") != args.dataset_revision:
            raise OpenCodeReasoningCandidateError(f"dataset revision mismatch: {path}")
    selected, counters = _read_candidate_files(paths)
    _atomic_write_jsonl(output, selected)
    report = {
        "schema": MERGE_SCHEMA,
        "status": "complete",
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "input_shards": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths
        ],
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "counters": dict(sorted(counters.items())),
    }
    _atomic_write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan")
    scan.add_argument("--input", required=True)
    scan.add_argument("--output", required=True)
    scan.add_argument("--report", required=True)
    scan.add_argument("--dataset", default="nvidia/OpenCodeReasoning-2")
    scan.add_argument("--dataset-revision", required=True)
    scan.add_argument("--shard-index", type=int, required=True)
    scan.add_argument("--allowed-datasets", default=",".join(sorted(DEFAULT_DATASETS)))
    scan.add_argument("--min-response-chars", type=int, default=256)
    scan.add_argument("--max-response-chars", type=int, default=24_000)
    scan.add_argument("--max-code-chars", type=int, default=20_000)
    scan.add_argument("--batch-size", type=int, default=512)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--inputs-glob", required=True)
    merge.add_argument("--expected-shards", type=int, required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--report", required=True)
    merge.add_argument("--dataset", default="nvidia/OpenCodeReasoning-2")
    merge.add_argument("--dataset-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "scan":
        report = scan_parquet(args)
    else:
        report = merge_candidates(args)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
