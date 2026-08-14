#!/usr/bin/env python3
"""Merge 16 candidate-only Q36 owner selections into one reviser draft view."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from hf_q36_mtr_generate_drafts import (
    DRAFT_IDENTITIES,
    DRAFT_SHARDS,
    MODEL_REVISION,
    SCHEMA as DRAFT_SCHEMA,
    load_sources,
    sha256_file,
)
from merge_q36_mtr_drafts import SCHEMA as MERGED_SCHEMA
from select_q36_mtr_owner_trajectories import REPORT_SCHEMA, RULE


class Q36MTRSelectedTrajectoryMergeError(RuntimeError):
    """Raised when selected owner trajectories do not form an exact draft view."""


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRSelectedTrajectoryMergeError("selected draft output exists")
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
        raise Q36MTRSelectedTrajectoryMergeError("selected draft report exists")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _lineage(receipts: list[dict[str, Any]]) -> str:
    encoded = (
        json.dumps(
            {
                "schema": "shohin-q36-mtr-selected-owner-lineage-v1",
                "rule": RULE,
                "receipts": receipts,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def merge(args: argparse.Namespace) -> dict[str, Any]:
    if (
        len(args.selection_reports) != DRAFT_SHARDS
        or len(args.selected_candidates) != DRAFT_SHARDS
        or args.output.exists()
        or args.report.exists()
    ):
        raise Q36MTRSelectedTrajectoryMergeError(
            "selected draft merge geometry differs"
        )
    sources, freeze_report = load_sources(
        args.train_source, args.development_source, args.freeze_report
    )
    receipts = []
    shard_rows: list[list[dict[str, Any]]] = []
    for index, (report_path, candidates_path) in enumerate(
        zip(args.selection_reports, args.selected_candidates, strict=True)
    ):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in candidates_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
        except (OSError, json.JSONDecodeError) as error:
            raise Q36MTRSelectedTrajectoryMergeError(
                "selected draft shard is unreadable"
            ) from error
        start = DRAFT_IDENTITIES * index // DRAFT_SHARDS
        end = DRAFT_IDENTITIES * (index + 1) // DRAFT_SHARDS
        if (
            report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "complete"
            or report.get("rule") != RULE
            or report.get("rows") != end - start
            or report.get("answer_labels_read") != 0
            or report.get("assessor_fields_read") != 0
            or report.get("output_sha256") != sha256_file(candidates_path)
            or Path(str(report.get("output", ""))).resolve()
            != candidates_path.resolve()
            or len(rows) != end - start
        ):
            raise Q36MTRSelectedTrajectoryMergeError(
                "selected draft selection report differs"
            )
        receipts.append(
            {
                "shard_index": index,
                "row_start": start,
                "row_end": end,
                "selection_report": str(report_path.resolve()),
                "selection_report_sha256": sha256_file(report_path),
                "selected_candidates": str(candidates_path.resolve()),
                "selected_candidates_sha256": report["output_sha256"],
                "first_candidates_sha256": report.get("first_candidates_sha256"),
                "second_candidates_sha256": report.get("second_candidates_sha256"),
            }
        )
        shard_rows.append(rows)

    lineage_sha256 = _lineage(receipts)
    merged = []
    selection_counts = {"first": 0, "second": 0}
    reason_counts = {
        "explicit_final_answer": 0,
        "nonexhausted": 0,
        "retained_first": 0,
    }
    for index, rows in enumerate(shard_rows):
        start = DRAFT_IDENTITIES * index // DRAFT_SHARDS
        end = DRAFT_IDENTITIES * (index + 1) // DRAFT_SHARDS
        for source, row in zip(sources[start:end], rows, strict=True):
            selection = row.get("trajectory_selection")
            tokens = row.get("generated_tokens")
            wall = row.get("wall_seconds")
            if not isinstance(selection, dict):
                raise Q36MTRSelectedTrajectoryMergeError(
                    "selected draft provenance differs"
                )
            choice = selection.get("choice")
            reason = selection.get("reason")
            expected_checkpoint = selection.get(f"{choice}_owner_checkpoint_sha256", "")
            if (
                row.get("schema") != DRAFT_SCHEMA
                or row.get("identity_sha256") != source["identity_sha256"]
                or row.get("split") != source["split"]
                or row.get("task") != source["task"]
                or row.get("prompt_sha256")
                != hashlib.sha256(source["source_prompt"].encode()).hexdigest()
                or row.get("model_revision") != MODEL_REVISION
                or not isinstance(row.get("completion"), str)
                or not row["completion"].strip()
                or isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or tokens <= 0
                or not isinstance(row.get("max_token_exhausted"), bool)
                or row.get("finish_reason")
                != ("length" if row["max_token_exhausted"] else "stop")
                or isinstance(wall, bool)
                or not isinstance(wall, (int, float))
                or not math.isfinite(float(wall))
                or wall < 0
                or selection.get("schema")
                != "shohin-q36-mtr-owner-trajectory-selection-v1"
                or selection.get("rule") != RULE
                or choice not in selection_counts
                or reason not in reason_counts
                or not isinstance(expected_checkpoint, str)
                or len(expected_checkpoint) != 64
                or row.get("owner_checkpoint_sha256") != expected_checkpoint
            ):
                raise Q36MTRSelectedTrajectoryMergeError(
                    "selected draft/source binding differs"
                )
            selected = dict(row)
            selected_selection = dict(selection)
            selected_selection["selected_owner_checkpoint_sha256"] = expected_checkpoint
            selected["trajectory_selection"] = selected_selection
            selected["owner_checkpoint_sha256"] = lineage_sha256
            merged.append(selected)
            selection_counts[choice] += 1
            reason_counts[reason] += 1
    identities = [row["identity_sha256"] for row in merged]
    if len(merged) != DRAFT_IDENTITIES or len(set(identities)) != DRAFT_IDENTITIES:
        raise Q36MTRSelectedTrajectoryMergeError(
            "selected draft identity coverage differs"
        )
    output_sha256 = _atomic_lines(args.output, merged)
    payload = {
        "schema": MERGED_SCHEMA,
        "status": "complete",
        "interpretation": "candidate_only_owner_trajectory_commit_as_reviser_draft",
        "model_revision": MODEL_REVISION,
        "owner_checkpoint_sha256": lineage_sha256,
        "owner_lineage_schema": "shohin-q36-mtr-selected-owner-lineage-v1",
        "selection_rule": RULE,
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "rows": len(merged),
        "ordered_identity_sha256": hashlib.sha256(
            ("\n".join(identities) + "\n").encode()
        ).hexdigest(),
        "freeze_report_identity_receipts": freeze_report["identity_receipts"],
        "input_receipts": receipts,
        "selection_counts": selection_counts,
        "reason_counts": reason_counts,
        "generated_tokens": sum(row["generated_tokens"] for row in merged),
        "max_token_exhausted": sum(row["max_token_exhausted"] for row in merged),
        "exact_identity_coverage": True,
        "duplicate_identities": 0,
        "answer_labels_read": 0,
        "assessor_fields_read": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.report, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument(
        "--selection-report",
        dest="selection_reports",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--selected-candidates",
        dest="selected_candidates",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    payload = merge(parse_args())
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
