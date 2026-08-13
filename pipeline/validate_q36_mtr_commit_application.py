#!/usr/bin/env python3
"""Validate Q36 commit/application outputs without opening any assessor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from build_q36_mtr_commit_pairs import REPORT_SCHEMA as PAIR_REPORT_SCHEMA
from q36_mtr_contract import MODEL_REVISION, TOTAL_ROWS
from q36_mtr_roles import TRAINABLE_MASTER_DTYPE
from score_q36_mtr import APPLICATION_SCHEMA, COMMIT_REPORT_SCHEMA, _load_selections

SCHEMA = "shohin-q36-mtr-commit-application-validation-v1"


class Q36MTRApplicationValidationError(RuntimeError):
    """The Q36 learned-commit application is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, schema: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRApplicationValidationError("Q36 commit artifact differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise Q36MTRApplicationValidationError("Q36 commit schema differs")
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRApplicationValidationError("Q36 commit validation exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate(args: argparse.Namespace) -> dict:
    report = _load(args.commit_training_report, COMMIT_REPORT_SCHEMA)
    application = _load(args.application_report, APPLICATION_SCHEMA)
    pairs = _load(args.development_pairs_report, PAIR_REPORT_SCHEMA)
    selections = _load_selections(args.selections)
    truncated = application.get("prompt_truncated")
    malformed = application.get("malformed")
    consistent = application.get("order_consistent")
    maximum_swap_error = application.get("maximum_swap_error")
    if (
        report.get("status") != "complete"
        or report.get("model_revision") != MODEL_REVISION
        or report.get("checkpoint_sha256") != sha256_file(args.commit_checkpoint)
        or Path(str(report.get("checkpoint", ""))).resolve()
        != args.commit_checkpoint.resolve()
        or report.get("development_application_report")
        != str(args.application_report.resolve())
        or report.get("development_selections_sha256") != sha256_file(args.selections)
        or report.get("protected_adapter_unchanged") is not True
        or report.get("trainable_master_dtype") != TRAINABLE_MASTER_DTYPE
        or report.get("trainable_compute_dtype") != "bfloat16"
        or report.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or application.get("status") != "complete"
        or application.get("model_revision") != MODEL_REVISION
        or application.get("commit_checkpoint_sha256")
        != sha256_file(args.commit_checkpoint)
        or Path(str(application.get("commit_checkpoint", ""))).resolve()
        != args.commit_checkpoint.resolve()
        or application.get("development_pairs_sha256")
        != sha256_file(args.development_pairs)
        or Path(str(application.get("development_pairs", ""))).resolve()
        != args.development_pairs.resolve()
        or application.get("development_pairs_report_sha256")
        != sha256_file(args.development_pairs_report)
        or application.get("selections_sha256") != sha256_file(args.selections)
        or Path(str(application.get("selections", ""))).resolve()
        != args.selections.resolve()
        or application.get("rows") != TOTAL_ROWS
        or isinstance(truncated, bool)
        or not isinstance(truncated, int)
        or not 0 <= truncated <= TOTAL_ROWS * 2
        or isinstance(malformed, bool)
        or not isinstance(malformed, int)
        or not 0 <= malformed <= TOTAL_ROWS
        or isinstance(consistent, bool)
        or not isinstance(consistent, int)
        or consistent
        != sum(int(row["order_consistent"]) for row in selections.values())
        or isinstance(maximum_swap_error, bool)
        or not isinstance(maximum_swap_error, (int, float))
        or not math.isfinite(float(maximum_swap_error))
        or maximum_swap_error < 0
        or application.get("inference_fields")
        != ["question", "candidate_a", "candidate_b"]
        or application.get("correctness_or_task_label_visible") is not False
        or application.get("assessor_board_access_count") != 0
        or application.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
        or pairs.get("status") != "complete"
        or pairs.get("source_split") != "development"
        or pairs.get("rows") != TOTAL_ROWS
        or pairs.get("labels_or_correctness_fields") != 0
        or pairs.get("output_sha256") != sha256_file(args.development_pairs)
        or pairs.get("source_disjoint_from_calibration") is not True
        or pairs.get("assessor_board_access_count") != 0
        or pairs.get("sealed_access") != {"holdout": 0, "product": 0, "public": 0}
    ):
        raise Q36MTRApplicationValidationError("Q36 commit application differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "rows": TOTAL_ROWS,
        "commit_checkpoint_sha256": sha256_file(args.commit_checkpoint),
        "commit_training_report_sha256": sha256_file(args.commit_training_report),
        "development_pairs_sha256": sha256_file(args.development_pairs),
        "development_pairs_report_sha256": sha256_file(args.development_pairs_report),
        "application_report_sha256": sha256_file(args.application_report),
        "selections_sha256": sha256_file(args.selections),
        "prompt_truncated": truncated,
        "malformed": malformed,
        "order_consistent": consistent,
        "maximum_swap_error": maximum_swap_error,
        "assessor_board_access_count": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit-checkpoint", type=Path, required=True)
    parser.add_argument("--commit-training-report", type=Path, required=True)
    parser.add_argument("--development-pairs", type=Path, required=True)
    parser.add_argument("--development-pairs-report", type=Path, required=True)
    parser.add_argument("--application-report", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = validate(parser.parse_args())
    print(json.dumps({"status": result["status"], "rows": result["rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
