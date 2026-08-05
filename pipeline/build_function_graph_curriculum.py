#!/usr/bin/env python3
"""Build a deterministic function-graph curriculum with real-function replay."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any


SCHEMA = "shohin-function-graph-curriculum-v1"


class FunctionGraphCurriculumError(RuntimeError):
    """The requested function curriculum violates its frozen data contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FunctionGraphCurriculumError(f"malformed JSONL: {path}") from exc
    if not rows:
        raise FunctionGraphCurriculumError(f"empty JSONL: {path}")
    return rows


def _question(row: dict[str, Any]) -> str:
    return str(row.get("question") or row.get("problem") or row.get("prompt") or "")


def _response(row: dict[str, Any]) -> str:
    return str(
        row.get("response")
        or row.get("solution")
        or row.get("completion")
        or row.get("answer")
        or ""
    )


def build_curriculum(
    generated_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    *,
    anchor_repeats: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if anchor_repeats <= 0:
        raise FunctionGraphCurriculumError("anchor repetition must be positive")
    generated_questions = [_question(row).strip().casefold() for row in generated_rows]
    anchor_questions = [_question(row).strip().casefold() for row in anchor_rows]
    if any(not value for value in generated_questions + anchor_questions) or any(
        not _response(row).strip() for row in generated_rows + anchor_rows
    ):
        raise FunctionGraphCurriculumError("curriculum row schema differs")
    if len(set(generated_questions)) != len(generated_questions):
        raise FunctionGraphCurriculumError("generated questions are not unique")
    if len(set(anchor_questions)) != len(anchor_questions):
        raise FunctionGraphCurriculumError("anchor questions are not unique")
    if set(generated_questions).intersection(anchor_questions):
        raise FunctionGraphCurriculumError("generated and anchor questions overlap")
    if any(row.get("split") != "train" for row in generated_rows):
        raise FunctionGraphCurriculumError("generated corpus contains non-train rows")

    output: list[dict[str, Any]] = []
    for row in generated_rows:
        output.append({**row, "curriculum_origin": "generated_function_graph"})
    for repetition in range(anchor_repeats):
        for row in anchor_rows:
            output.append(
                {
                    **row,
                    "curriculum_origin": "verified_real_function_anchor",
                    "curriculum_repetition": repetition,
                }
            )
    random.Random(seed).shuffle(output)
    return output, {
        "schema": SCHEMA,
        "status": "complete",
        "seed": seed,
        "rows": len(output),
        "generated_rows": len(generated_rows),
        "anchor_unique_rows": len(anchor_rows),
        "anchor_repeats": anchor_repeats,
        "anchor_materialized_rows": len(anchor_rows) * anchor_repeats,
        "family_counts": dict(
            sorted(
                Counter(
                    str(row.get("family") or "anchor") for row in generated_rows
                ).items()
            )
        ),
    }


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise FunctionGraphCurriculumError(f"refusing existing output: {path}")
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
        raise FunctionGraphCurriculumError(f"refusing existing report: {path}")
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
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--generated-sha256", required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--anchor-sha256", required=True)
    parser.add_argument("--anchor-repeats", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    generated_sha = _sha256(args.generated)
    anchor_sha = _sha256(args.anchor)
    if generated_sha != args.generated_sha256 or anchor_sha != args.anchor_sha256:
        raise FunctionGraphCurriculumError("input hash differs")
    rows, report = build_curriculum(
        _rows(args.generated),
        _rows(args.anchor),
        anchor_repeats=args.anchor_repeats,
        seed=args.seed,
    )
    report.update(
        {
            "generated": str(args.generated.resolve()),
            "generated_sha256": generated_sha,
            "anchor": str(args.anchor.resolve()),
            "anchor_sha256": anchor_sha,
        }
    )
    report["output_sha256"] = _atomic_lines(args.output, rows)
    report["output"] = str(args.output.resolve())
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
