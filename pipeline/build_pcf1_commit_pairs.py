#!/usr/bin/env python3
"""Build PCF1 commit calibration pairs from training identities only."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_pcf1_evaluate import CANDIDATE_SCHEMA, EVAL_SCHEMA
from hf_pcf1_train_commit import OUTCOMES, PAIR_SCHEMA, expected_outcome
from merge_pcf1_evaluation_shards import MERGED_REPORT_SCHEMA

REPORT_SCHEMA = "shohin-pcf1-commit-pair-report-v1"
CALIBRATION_SEED = 2026080820


class PCF1PairError(RuntimeError):
    """PCF1 calibration data or candidate custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_sealed_path(path: Path, *, allow_confirmation: bool = False) -> None:
    blocked = (
        ("holdout", "product", "public")
        if allow_confirmation
        else (
            "confirmation",
            "holdout",
            "product",
            "public",
        )
    )
    rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
    if any(word in rendered for word in blocked):
        raise PCF1PairError(f"sealed path supplied to PCF1 calibration: {path}")


def explicit_candidate_root(root: Path, *, allow_confirmation: bool = False) -> Path:
    """Resolve one narrow caller-supplied root for candidate artifacts."""

    reject_sealed_path(root, allow_confirmation=allow_confirmation)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise PCF1PairError("PCF1 candidate root is not an explicit directory")
    resolved = root.resolve(strict=True)
    if resolved in {Path("/"), Path.home().resolve()}:
        raise PCF1PairError("PCF1 candidate root is too broad")
    return resolved


def explicit_candidate_file(
    path: Path, root: Path, *, allow_confirmation: bool = False
) -> Path:
    """Require a regular, nonsymbolic candidate file below the explicit root."""

    reject_sealed_path(path, allow_confirmation=allow_confirmation)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PCF1PairError("PCF1 candidate is not an explicit regular file")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise PCF1PairError("PCF1 candidate escapes the explicit root") from error
    if not relative.parts:
        raise PCF1PairError("PCF1 candidate path equals its root")
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise PCF1PairError("PCF1 candidate traverses a symbolic directory")
    return resolved


def load_lines(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PCF1PairError(f"unreadable PCF1 input: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PCF1PairError(
                f"malformed PCF1 input at line {line_number}: {path}"
            ) from error
        if not isinstance(row, dict):
            raise PCF1PairError(f"non-object PCF1 input at line {line_number}: {path}")
        rows.append(row)
    if not rows:
        raise PCF1PairError(f"empty PCF1 input: {path}")
    return rows


def load_data(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in load_lines(path):
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != EVAL_SCHEMA
            or row.get("split") != "calibration"
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in rows
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
        ):
            raise PCF1PairError("PCF1 calibration source data differs")
        rows[identity] = row
    return rows


def load_arm(
    report_path: Path,
    candidates_path: Path,
    arm: str,
    split: str = "calibration",
    *,
    candidates_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    safe_root = explicit_candidate_root(
        candidates_root, allow_confirmation=split == "confirmation"
    )
    explicit_candidates = explicit_candidate_file(
        candidates_path, safe_root, allow_confirmation=split == "confirmation"
    )
    if report_path.is_symlink() or not report_path.is_file():
        raise PCF1PairError("PCF1 arm report is not an explicit regular file")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PCF1PairError("PCF1 arm report is unreadable") from error
    if not isinstance(report, dict):
        raise PCF1PairError("PCF1 arm report is not an object")
    if (
        report.get("schema") != MERGED_REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("arm") != arm
        or report.get("split") != split
        or report.get("exact_identity_coverage") is not True
        or report.get("assessment_mode")
        != (
            "calibration_immediate"
            if split == "calibration"
            else "confirmation_deferred"
        )
        or report.get("assessor_board_access_count") != 0
        or (split == "confirmation" and report.get("metrics") is not None)
        or (split == "calibration" and not isinstance(report.get("metrics"), dict))
        or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
    ):
        raise PCF1PairError("PCF1 calibration arm report differs")
    if Path(
        str(report.get("candidates_output", ""))
    ).resolve() != explicit_candidates or sha256_file(
        explicit_candidates
    ) != report.get(
        "candidates_sha256"
    ):
        raise PCF1PairError("PCF1 calibration candidate hash differs")
    candidates: dict[str, dict[str, Any]] = {}
    for row in load_lines(explicit_candidates):
        identity = row.get("identity_sha256")
        label_free_fields = {
            "schema",
            "arm",
            "identity_sha256",
            "task",
            "completion",
            "generated_tokens",
            "max_token_exhausted",
        }
        if (
            row.get("schema") != CANDIDATE_SCHEMA
            or row.get("arm") != arm
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in candidates
            or not isinstance(row.get("completion"), str)
            or isinstance(row.get("generated_tokens"), bool)
            or not isinstance(row.get("generated_tokens"), int)
            or row["generated_tokens"] < 0
            or not isinstance(row.get("max_token_exhausted"), bool)
        ):
            raise PCF1PairError("PCF1 calibration candidate content differs")
        if split == "confirmation":
            if set(row) != label_free_fields:
                raise PCF1PairError("PCF1 confirmation candidate exposes assessment")
        elif not isinstance(row.get("correct"), bool):
            raise PCF1PairError("PCF1 calibration candidate lacks assessment")
        candidates[identity] = row
    return candidates, report


def calibration_split(identity: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}\0{identity}".encode()).digest()
    return (
        "calibration_train"
        if int.from_bytes(digest[:8], "big") % 10_000 < 8_000
        else "calibration_development"
    )


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise PCF1PairError(f"refusing existing PCF1 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise PCF1PairError(f"refusing existing PCF1 output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PCF1PairError(f"refusing existing PCF1 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise PCF1PairError(f"refusing existing PCF1 output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.calibration_data,
        args.revision_report,
        args.revision_candidates,
        args.unchanged_report,
        args.unchanged_candidates,
        args.candidates_root,
        args.output,
        args.report,
    ):
        reject_sealed_path(path)
    if any(path.exists() or path.is_symlink() for path in (args.output, args.report)):
        raise PCF1PairError("PCF1 calibration pair output already exists")
    data = load_data(args.calibration_data)
    revision, revision_receipt = load_arm(
        args.revision_report,
        args.revision_candidates,
        "revision",
        candidates_root=args.candidates_root,
    )
    unchanged, unchanged_receipt = load_arm(
        args.unchanged_report,
        args.unchanged_candidates,
        "unchanged",
        candidates_root=args.candidates_root,
    )
    data_sha256 = sha256_file(args.calibration_data)
    if any(
        Path(str(receipt.get("data", ""))).resolve() != args.calibration_data.resolve()
        or receipt.get("data_sha256") != data_sha256
        for receipt in (revision_receipt, unchanged_receipt)
    ):
        raise PCF1PairError("PCF1 calibration arm/data binding differs")
    if set(data) != set(revision) or set(data) != set(unchanged):
        raise PCF1PairError("PCF1 calibration identity coverage differs")
    output_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    outcomes: dict[str, Counter[str]] = {
        "calibration_train": Counter(),
        "calibration_development": Counter(),
    }
    for identity in sorted(data):
        source, left, right = data[identity], revision[identity], unchanged[identity]
        if left.get("task") != source.get("task") or right.get("task") != source.get(
            "task"
        ):
            raise PCF1PairError("PCF1 calibration task binding differs")
        split = calibration_split(identity, args.seed)
        outcome = expected_outcome(bool(left["correct"]), bool(right["correct"]))
        counts[split] += 1
        outcomes[split][outcome] += 1
        output_rows.append(
            {
                "schema": PAIR_SCHEMA,
                "identity_sha256": identity,
                "split": split,
                "task": source["task"],
                "question": source["question"],
                "outcome_class": outcome,
                "candidates": [
                    {
                        "lineage": "revision",
                        "completion": left["completion"],
                        "correct": left["correct"],
                        "generated_tokens": left.get("generated_tokens"),
                        "max_token_exhausted": left.get("max_token_exhausted"),
                    },
                    {
                        "lineage": "unchanged",
                        "completion": right["completion"],
                        "correct": right["correct"],
                        "generated_tokens": right.get("generated_tokens"),
                        "max_token_exhausted": right.get("max_token_exhausted"),
                    },
                ],
            }
        )
    if set(outcomes["calibration_train"]) != set(OUTCOMES):
        raise PCF1PairError("PCF1 calibration train lacks an outcome class")
    if not counts["calibration_development"]:
        raise PCF1PairError("PCF1 calibration development is empty")
    output_rows.sort(key=lambda row: (str(row["split"]), str(row["identity_sha256"])))
    output_sha256 = atomic_lines(args.output, output_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "seed": args.seed,
        "counts": dict(sorted(counts.items())),
        "outcomes": {
            split: dict(sorted(counter.items())) for split, counter in outcomes.items()
        },
        "inputs": {
            "data": str(args.calibration_data.resolve()),
            "data_sha256": data_sha256,
            "revision_report": str(args.revision_report.resolve()),
            "revision_report_sha256": sha256_file(args.revision_report),
            "revision_candidates_sha256": revision_receipt["candidates_sha256"],
            "unchanged_report": str(args.unchanged_report.resolve()),
            "unchanged_report_sha256": sha256_file(args.unchanged_report),
            "unchanged_candidates_sha256": unchanged_receipt["candidates_sha256"],
        },
        "source_disjoint_from_confirmation": True,
        "confirmation_rows_loaded": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--revision-report", type=Path, required=True)
    parser.add_argument("--revision-candidates", type=Path, required=True)
    parser.add_argument("--unchanged-report", type=Path, required=True)
    parser.add_argument("--unchanged-candidates", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEED)
    args = parser.parse_args()
    report = build(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
