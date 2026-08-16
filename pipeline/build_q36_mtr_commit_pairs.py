#!/usr/bin/env python3
"""Build labeled calibration or label-free development Q36 commit pairs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_q36_mtr_evaluate import CANDIDATE_SCHEMA, DATA_SCHEMA, TASKS
from merge_q36_mtr_evaluations import SCHEMA as MERGED_REPORT_SCHEMA
from q36_mtr_roles import MODEL_REVISION

PAIR_SCHEMA = "shohin-q36-mtr-whole-trajectory-pair-v1"
REPORT_SCHEMA = "shohin-q36-mtr-commit-pair-report-v1"
CALIBRATION_SEED = 2026080820
COUNTS = {"calibration": 5_824, "development": 1_289}
OUTCOMES = ("both_correct", "revision_only", "both_wrong", "unchanged_only")


class Q36MTRPairError(RuntimeError):
    """Q36 commit-pair source, candidate, or visibility custody differs."""


def _reject_protected_path(path: Path) -> None:
    rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
    if any(term in rendered for term in ("holdout", "product", "public")):
        raise Q36MTRPairError(f"protected path supplied to Q36 commit pairs: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRPairError("Q36 commit-pair input is absent or symbolic")
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Q36MTRPairError(f"malformed Q36 pair input line {number}") from error
        if not isinstance(row, dict):
            raise Q36MTRPairError("Q36 pair input row is not an object")
        result.append(row)
    if not result:
        raise Q36MTRPairError("Q36 commit-pair input is empty")
    return result


def _candidate_file(path: Path, root: Path) -> Path:
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise Q36MTRPairError("Q36 candidate root/file boundary differs")
    resolved_root = root.resolve(strict=True)
    if resolved_root in {Path("/"), Path.home().resolve()}:
        raise Q36MTRPairError("Q36 candidate root is too broad")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise Q36MTRPairError("Q36 candidate escapes its explicit root") from error
    if not relative.parts:
        raise Q36MTRPairError("Q36 candidate path equals its root")
    current = resolved_root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise Q36MTRPairError("Q36 candidate traverses a symbolic directory")
    return resolved


def _load_source(path: Path, split: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _rows(path):
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != DATA_SCHEMA
            or row.get("split") != split
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in result
            or row.get("task") not in TASKS
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
        ):
            raise Q36MTRPairError("Q36 commit-pair source differs")
        if split == "development" and any(
            key in row for key in ("assessor", "answer", "correct", "gold", "response")
        ):
            raise Q36MTRPairError("Q36 development pair source exposes supervision")
        result[identity] = row
    if len(result) != COUNTS[split]:
        raise Q36MTRPairError("Q36 commit-pair source count differs")
    return result


def _load_arm(
    report_path: Path,
    candidates_path: Path,
    candidates_root: Path,
    arm: str,
    split: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    explicit = _candidate_file(candidates_path, candidates_root)
    if report_path.is_symlink() or not report_path.is_file():
        raise Q36MTRPairError("Q36 merged report is absent or symbolic")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != MERGED_REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("arm") != arm
        or report.get("split") != split
        or report.get("model_revision") != MODEL_REVISION
        or report.get("rows") != COUNTS[split]
        or report.get("exact_identity_coverage") is not True
        or report.get("duplicate_identities") != 0
        or report.get("assessor_board_access_count") != 0
        or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or Path(str(report.get("output", ""))).resolve() != explicit
        or report.get("output_sha256") != sha256_file(explicit)
        or (split == "development" and report.get("metrics") is not None)
        or (split == "calibration" and not isinstance(report.get("metrics"), dict))
    ):
        raise Q36MTRPairError("Q36 merged arm report differs")
    result: dict[str, dict[str, Any]] = {}
    label_free = {
        "schema",
        "arm",
        "identity_sha256",
        "task",
        "completion",
        "generated_tokens",
        "max_token_exhausted",
    }
    for row in _rows(explicit):
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != CANDIDATE_SCHEMA
            or row.get("arm") != arm
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in result
            or row.get("task") not in TASKS
            or not isinstance(row.get("completion"), str)
            or isinstance(row.get("generated_tokens"), bool)
            or not isinstance(row.get("generated_tokens"), int)
            or row["generated_tokens"] < 0
            or not isinstance(row.get("max_token_exhausted"), bool)
        ):
            raise Q36MTRPairError("Q36 commit candidate differs")
        if split == "development" and set(row) != label_free:
            raise Q36MTRPairError("Q36 development candidate exposes supervision")
        if split == "calibration" and not isinstance(row.get("correct"), bool):
            raise Q36MTRPairError("Q36 calibration candidate lacks correctness")
        result[identity] = row
    return result, report


def expected_outcome(revision: bool, unchanged: bool) -> str:
    if revision and unchanged:
        return "both_correct"
    if revision:
        return "revision_only"
    if unchanged:
        return "unchanged_only"
    return "both_wrong"


def calibration_split(identity: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}\0{identity}".encode()).digest()
    return (
        "calibration_train"
        if int.from_bytes(digest[:8], "big") % 10_000 < 8_000
        else "calibration_development"
    )


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRPairError("Q36 commit-pair output exists")
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
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRPairError("Q36 commit-pair report exists")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.data,
        args.revision_report,
        args.revision_candidates,
        args.unchanged_report,
        args.unchanged_candidates,
        args.candidates_root,
        args.output,
        args.report,
    ):
        _reject_protected_path(path)
    if (
        args.split not in COUNTS
        or args.seed != CALIBRATION_SEED
        or args.output.exists()
        or args.report.exists()
    ):
        raise Q36MTRPairError("Q36 commit-pair settings differ")
    source = _load_source(args.data, args.split)
    revision, revision_report = _load_arm(
        args.revision_report,
        args.revision_candidates,
        args.candidates_root,
        "revision",
        args.split,
    )
    unchanged, unchanged_report = _load_arm(
        args.unchanged_report,
        args.unchanged_candidates,
        args.candidates_root,
        "unchanged",
        args.split,
    )
    data_sha256 = sha256_file(args.data)
    if any(
        report.get("data_sha256") != data_sha256
        for report in (revision_report, unchanged_report)
    ):
        raise Q36MTRPairError("Q36 merged arm/data binding differs")
    if set(source) != set(revision) or set(source) != set(unchanged):
        raise Q36MTRPairError("Q36 commit-pair identity coverage differs")
    rows = []
    counts: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    for identity in sorted(source):
        left, right = revision[identity], unchanged[identity]
        if (
            left["task"] != source[identity]["task"]
            or right["task"] != source[identity]["task"]
        ):
            raise Q36MTRPairError("Q36 commit-pair task binding differs")
        candidates = [
            {"lineage": "revision", "completion": left["completion"]},
            {"lineage": "unchanged", "completion": right["completion"]},
        ]
        row: dict[str, Any] = {
            "schema": PAIR_SCHEMA,
            "identity_sha256": identity,
            "split": args.split,
            "task": source[identity]["task"],
            "question": source[identity]["question"],
            "candidates": candidates,
        }
        if args.split == "calibration":
            local_split = calibration_split(identity, args.seed)
            outcome = expected_outcome(bool(left["correct"]), bool(right["correct"]))
            row["split"] = local_split
            row["outcome_class"] = outcome
            for candidate, original in zip(candidates, (left, right), strict=True):
                candidate.update(
                    {
                        "correct": original["correct"],
                        "generated_tokens": original["generated_tokens"],
                        "max_token_exhausted": original["max_token_exhausted"],
                    }
                )
            counts[local_split] += 1
            outcomes[outcome] += 1
        else:
            counts["development"] += 1
        rows.append(row)
    if args.split == "calibration" and (
        set(outcomes) != set(OUTCOMES)
        or not counts["calibration_train"]
        or not counts["calibration_development"]
    ):
        raise Q36MTRPairError("Q36 calibration outcome/split coverage differs")
    rows.sort(key=lambda row: (str(row["split"]), str(row["identity_sha256"])))
    output_sha256 = _atomic_lines(args.output, rows)
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "pair_schema": PAIR_SCHEMA,
        "source_split": args.split,
        "seed": args.seed,
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "outcomes": dict(sorted(outcomes.items())),
        "labels_or_correctness_fields": 0 if args.split == "development" else len(rows),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "inputs": {
            "data_sha256": data_sha256,
            "revision_report_sha256": sha256_file(args.revision_report),
            "revision_candidates_sha256": revision_report["output_sha256"],
            "unchanged_report_sha256": sha256_file(args.unchanged_report),
            "unchanged_candidates_sha256": unchanged_report["output_sha256"],
        },
        "source_disjoint_from_calibration": args.split == "development",
        "assessor_board_access_count": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.report, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(COUNTS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--revision-report", type=Path, required=True)
    parser.add_argument("--revision-candidates", type=Path, required=True)
    parser.add_argument("--unchanged-report", type=Path, required=True)
    parser.add_argument("--unchanged-candidates", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEED)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(build(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
