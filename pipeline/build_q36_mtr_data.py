#!/usr/bin/env python3
"""Bind merged Q36 owner drafts into matched natural-trajectory role data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from build_pcf1_data import (
    ASSESSOR_SCHEMA,
    CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA,
    DEVELOPMENT_SOURCE_SCHEMA,
    FREEZE_REPORT_SCHEMA,
    TRAIN_SOURCE_SCHEMA,
    _load_source_view,
    revision_presentations,
    revision_prompt,
    sha256_file,
)
from hf_q36_mtr_generate_drafts import MODEL_REVISION, SCHEMA as DRAFT_SCHEMA
from merge_q36_mtr_drafts import SCHEMA as DRAFT_REPORT_SCHEMA
from q36_mtr_roles import DRAFT_IDENTITIES, REVISION_PRESENTATIONS

REVISION_SCHEMA = "shohin-q36-mtr-revision-train-v1"
EVAL_SCHEMA = "shohin-q36-mtr-eval-v1"
REPORT_SCHEMA = "shohin-q36-mtr-data-report-v1"
TRAIN_IDENTITIES = 5_824
DEVELOPMENT_IDENTITIES = 1_289


class Q36MTRDataError(RuntimeError):
    """The Q36-MTR natural-trajectory data boundary differs."""


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_drafts(
    drafts_path: Path,
    draft_report_path: Path,
    sources: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report = json.loads(draft_report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != DRAFT_REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("output_sha256") != sha256_file(drafts_path)
        or Path(str(report.get("output", ""))).resolve() != drafts_path.resolve()
        or report.get("rows") != DRAFT_IDENTITIES
        or report.get("model_revision") != MODEL_REVISION
        or report.get("exact_identity_coverage") is not True
        or report.get("duplicate_identities") != 0
        or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
    ):
        raise Q36MTRDataError("Q36-MTR merged draft report differs")
    drafts: dict[str, dict[str, Any]] = {}
    checkpoint_hashes: set[str] = set()
    for line in drafts_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        identity = str(row.get("identity_sha256", ""))
        source = sources.get(identity)
        if (
            row.get("schema") != DRAFT_SCHEMA
            or source is None
            or identity in drafts
            or row.get("split") != source["split"]
            or row.get("task") != source["task"]
            or row.get("prompt_sha256")
            != hashlib.sha256(source["source_prompt"].encode()).hexdigest()
            or row.get("model_revision") != MODEL_REVISION
            or not isinstance(row.get("completion"), str)
            or not row["completion"].strip()
        ):
            raise Q36MTRDataError("Q36-MTR draft/source binding differs")
        checkpoint_hashes.add(str(row.get("owner_checkpoint_sha256", "")))
        drafts[identity] = row
    if (
        set(drafts) != set(sources)
        or len(checkpoint_hashes) != 1
        or next(iter(checkpoint_hashes), "") != report.get("owner_checkpoint_sha256")
    ):
        raise Q36MTRDataError("Q36-MTR draft lineage differs")
    return drafts, report


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise Q36MTRDataError("Q36-MTR materialized output exists")
    freeze_report = json.loads(
        (args.source_root / "report.json").read_text(encoding="utf-8")
    )
    if (
        freeze_report.get("schema") != FREEZE_REPORT_SCHEMA
        or freeze_report.get("status") != "complete"
        or freeze_report.get("counts")
        != {
            "train": TRAIN_IDENTITIES,
            "development": DEVELOPMENT_IDENTITIES,
            "holdout": 1_279,
        }
        or freeze_report.get("source_disjoint") is not True
        or freeze_report.get("sealed_content_materialized") is not False
    ):
        raise Q36MTRDataError("Q36-MTR source freeze differs")
    train = _load_source_view(
        args.source_root / "train_sources.jsonl", TRAIN_SOURCE_SCHEMA, "train"
    )
    development = _load_source_view(
        args.source_root / "development_sources.jsonl",
        DEVELOPMENT_SOURCE_SCHEMA,
        "development",
    )
    if len(train) != TRAIN_IDENTITIES or len(development) != DEVELOPMENT_IDENTITIES:
        raise Q36MTRDataError("Q36-MTR source counts differ")
    sources = {**train, **development}
    drafts, draft_report = _load_drafts(args.drafts, args.draft_report, sources)
    assessor_receipt = json.loads(args.assessor_receipt.read_text(encoding="utf-8"))
    if (
        assessor_receipt.get("schema") != CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA
        or assessor_receipt.get("status") != "complete"
        or assessor_receipt.get("rows") != DEVELOPMENT_IDENTITIES
        or assessor_receipt.get("semantic_access") != "final_score_only"
        or not isinstance(assessor_receipt.get("board_sha256"), str)
        or len(assessor_receipt["board_sha256"]) != 64
    ):
        raise Q36MTRDataError("Q36-MTR assessor receipt differs")

    revision_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    development_rows: list[dict[str, Any]] = []
    for identity in sorted(sources):
        source = sources[identity]
        draft = drafts[identity]
        completion = str(draft["completion"]).strip()
        question = revision_prompt(str(source["source_prompt"]), completion)
        draft_sha256 = hashlib.sha256(completion.encode()).hexdigest()
        common = {
            "schema": EVAL_SCHEMA,
            "identity_sha256": identity,
            "task": source["task"],
            "question": question,
            "source_prompt": source["source_prompt"],
            "internal_draft": draft,
            "model_owned_draft_sha256": draft_sha256,
            "candidates": [],
            "runtime_fields": ["question", "source_prompt"],
            "internal_draft_visible": True,
            "external_candidate_text_visible": False,
        }
        if source["split"] == "train":
            assessor = source.get("assessor")
            if (
                not isinstance(assessor, dict)
                or assessor.get("schema") != ASSESSOR_SCHEMA
                or assessor.get("identity_sha256") != identity
            ):
                raise Q36MTRDataError("Q36-MTR calibration assessor differs")
            for presentation in range(revision_presentations(source["outcome_class"])):
                revision_rows.append(
                    {
                        "schema": REVISION_SCHEMA,
                        "identity_sha256": hashlib.sha256(
                            f"q36-mtr-revision\0{identity}\0{presentation}".encode()
                        ).hexdigest(),
                        "source_identity_sha256": identity,
                        "task": source["task"],
                        "outcome_class": source["outcome_class"],
                        "presentation": presentation,
                        "question": question,
                        "model_owned_draft_sha256": draft_sha256,
                        "response": source["response"],
                        "target_kind": source["target_kind"],
                        "runtime_fields": ["question"],
                        "internal_draft_visible": True,
                        "external_candidate_text_visible": False,
                        "supervisor_only_fields": [
                            "response",
                            "target_kind",
                            "task",
                            "outcome_class",
                        ],
                    }
                )
            calibration_rows.append(
                {
                    **common,
                    "split": "calibration",
                    "assessor": assessor,
                    "runtime_fields": ["question"],
                }
            )
        else:
            if any(
                field in source
                for field in ("assessor", "answer", "response", "gold", "target")
            ):
                raise Q36MTRDataError("Q36-MTR development source exposes supervision")
            development_rows.append({**common, "split": "development"})
    if (
        len(revision_rows) != REVISION_PRESENTATIONS
        or len(calibration_rows) != TRAIN_IDENTITIES
        or len(development_rows) != DEVELOPMENT_IDENTITIES
    ):
        raise Q36MTRDataError("Q36-MTR materialized geometry differs")

    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        outputs = {
            "revision_train": {
                "path": str((args.output / "revision_train.jsonl").resolve()),
                "sha256": _atomic_lines(
                    temporary / "revision_train.jsonl", revision_rows
                ),
                "rows": len(revision_rows),
            },
            "calibration": {
                "path": str((args.output / "calibration_eval.jsonl").resolve()),
                "sha256": _atomic_lines(
                    temporary / "calibration_eval.jsonl", calibration_rows
                ),
                "rows": len(calibration_rows),
            },
            "development": {
                "path": str((args.output / "development_eval.jsonl").resolve()),
                "sha256": _atomic_lines(
                    temporary / "development_eval.jsonl", development_rows
                ),
                "rows": len(development_rows),
            },
        }
        report = {
            "schema": REPORT_SCHEMA,
            "status": "complete",
            "model_revision": MODEL_REVISION,
            "freeze_report_sha256": sha256_file(args.source_root / "report.json"),
            "draft_report_sha256": sha256_file(args.draft_report),
            "drafts_sha256": sha256_file(args.drafts),
            "owner_checkpoint_sha256": draft_report["owner_checkpoint_sha256"],
            "counts": {
                "train_unique_identities": TRAIN_IDENTITIES,
                "revision_train_presentations": REVISION_PRESENTATIONS,
                "calibration_rows": TRAIN_IDENTITIES,
                "development_rows": DEVELOPMENT_IDENTITIES,
            },
            "revision_presentation_rule": {
                "single_correct": 4,
                "both_correct_or_both_wrong": 1,
            },
            "outputs": outputs,
            "confirmation_assessors": {
                "board_sha256": assessor_receipt["board_sha256"],
                "receipt_sha256": sha256_file(args.assessor_receipt),
                "semantic_reads": 0,
                "authorized_reader": "future_q36_mtr_score_once",
            },
            "source_disjoint": True,
            "model_owned_drafts": True,
            "sealed_content_materialized": False,
            "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        }
        _atomic_json(temporary / "report.json", report)
        os.replace(temporary, args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--draft-report", type=Path, required=True)
    parser.add_argument("--assessor-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(materialize(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
