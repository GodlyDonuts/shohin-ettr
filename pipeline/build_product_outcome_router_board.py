#!/usr/bin/env python3
"""Freeze held-out verified prompts for paired frozen-expert outcome labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


WORD = re.compile(r"\w+")


def question_from_row(row: dict[str, Any]) -> str | None:
    for key in ("question", "problem", "prompt", "text", "input"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return None


def answer_from_row(row: dict[str, Any]) -> str | None:
    for key in ("expected_answer_normalized", "target", "answer"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def normalized_question(question: str) -> str:
    return " ".join(WORD.findall(question.casefold()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_excluded_questions(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        with path.open(errors="replace") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = question_from_row(row)
                if question:
                    excluded.add(normalized_question(question))
    return excluded


def build_board(
    *,
    source_path: Path,
    excluded_paths: list[Path],
    training_group: str,
    task: str,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if task not in {"math500", "bbh_logic"}:
        raise ValueError("task must be math500 or bbh_logic")
    if count <= 0:
        raise ValueError("count must be positive")
    excluded = read_excluded_questions(excluded_paths)
    seen: set[str] = set()
    candidates: list[tuple[str, dict[str, Any]]] = []
    scanned = malformed = wrong_group = missing = duplicate = excluded_count = 0
    with source_path.open(errors="replace") as source:
        for line in source:
            if not line.strip():
                continue
            scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if str(row.get("training_group") or row.get("domain")) != training_group:
                wrong_group += 1
                continue
            question = question_from_row(row)
            answer = answer_from_row(row)
            if not question or not answer:
                missing += 1
                continue
            identity = normalized_question(question)
            if identity in excluded:
                excluded_count += 1
                continue
            if identity in seen:
                duplicate += 1
                continue
            seen.add(identity)
            output = {
                "source_prompt_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "source": row.get("source"),
                "verification": row.get("verification"),
            }
            if task == "math500":
                output.update({"question": question, "answer": f"\\boxed{{{answer}}}"})
            else:
                output.update({"input": question, "target": answer})
            rank = hashlib.sha256(f"{seed}\0{identity}".encode()).hexdigest()
            candidates.append((rank, output))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < count:
        raise ValueError(
            f"source has only {len(candidates)} eligible unique rows for count={count}"
        )
    selected = [row for _, row in candidates[:count]]
    report = {
        "schema": "shohin-product-outcome-router-board-v1",
        "task": task,
        "training_group": training_group,
        "seed": seed,
        "requested_count": count,
        "selected_count": len(selected),
        "source": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "exclude_sources": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in excluded_paths
        ],
        "scanned": scanned,
        "malformed": malformed,
        "wrong_group": wrong_group,
        "missing_question_or_answer": missing,
        "duplicate_normalized_question": duplicate,
        "excluded_training_question": excluded_count,
    }
    return selected, report


def atomic_write(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--training-group", required=True)
    parser.add_argument("--task", choices=("math500", "bbh_logic"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.source.is_file() or any(not path.is_file() for path in args.exclude):
        parser.error("every source and exclusion path must exist")
    if args.output == args.report or args.output.exists() or args.report.exists():
        parser.error("outputs must be distinct and absent")

    selected, report = build_board(
        source_path=args.source,
        excluded_paths=args.exclude,
        training_group=args.training_group,
        task=args.task,
        count=args.count,
        seed=args.seed,
    )
    output_bytes = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode()
        for row in selected
    )
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = hashlib.sha256(output_bytes).hexdigest()
    atomic_write(args.output, output_bytes)
    atomic_write(
        args.report,
        (json.dumps(report, sort_keys=True, indent=2) + "\n").encode(),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
