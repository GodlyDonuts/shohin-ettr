from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from build_q36_mtr_commit_pairs import REPORT_SCHEMA as PAIR_REPORT_SCHEMA
from q36_mtr_contract import MODEL_REVISION
from score_q36_mtr import APPLICATION_SCHEMA, COMMIT_REPORT_SCHEMA, SELECTION_SCHEMA
from validate_q36_mtr_commit_application import (
    Q36MTRApplicationValidationError,
    validate,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> argparse.Namespace:
    checkpoint = tmp_path / "commit.pt"
    checkpoint.write_bytes(b"commit")
    pairs = tmp_path / "development_pairs.jsonl"
    pairs.write_text("pairs\n", encoding="utf-8")
    pairs_report = tmp_path / "development_pairs_report.json"
    pairs_report.write_text(
        json.dumps(
            {
                "schema": PAIR_REPORT_SCHEMA,
                "status": "complete",
                "source_split": "development",
                "rows": 1_289,
                "labels_or_correctness_fields": 0,
                "output_sha256": _sha(pairs),
                "source_disjoint_from_calibration": True,
                "assessor_board_access_count": 0,
                "sealed_access": {"holdout": 0, "product": 0, "public": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selections = tmp_path / "selections.jsonl"
    rows = []
    for index in range(1_289):
        selected = index % 2
        rows.append(
            {
                "schema": SELECTION_SCHEMA,
                "identity_sha256": hashlib.sha256(f"id-{index}".encode()).hexdigest(),
                "task": ("math500", "bbh_logic", "mbpp")[index % 3],
                "selected_index": selected,
                "selected_lineage": ("revision", "unchanged")[selected],
                "order_consistent": True,
                "margin": float(index),
            }
        )
    selections.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    application = tmp_path / "application.json"
    application.write_text(
        json.dumps(
            {
                "schema": APPLICATION_SCHEMA,
                "status": "complete",
                "model_revision": MODEL_REVISION,
                "commit_checkpoint": str(checkpoint.resolve()),
                "commit_checkpoint_sha256": _sha(checkpoint),
                "development_pairs": str(pairs.resolve()),
                "development_pairs_sha256": _sha(pairs),
                "development_pairs_report_sha256": _sha(pairs_report),
                "selections": str(selections.resolve()),
                "selections_sha256": _sha(selections),
                "rows": 1_289,
                "prompt_truncated": 0,
                "malformed": 0,
                "order_consistent": 1_289,
                "maximum_swap_error": 0.0,
                "inference_fields": ["question", "candidate_a", "candidate_b"],
                "correctness_or_task_label_visible": False,
                "assessor_board_access_count": 0,
                "sealed_access": {"holdout": 0, "product": 0, "public": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "commit_report.json"
    report.write_text(
        json.dumps(
            {
                "schema": COMMIT_REPORT_SCHEMA,
                "status": "complete",
                "model_revision": MODEL_REVISION,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha(checkpoint),
                "development_application_report": str(application.resolve()),
                "development_selections_sha256": _sha(selections),
                "protected_adapter_unchanged": True,
                "sealed_access": {"holdout": 0, "product": 0, "public": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return argparse.Namespace(
        commit_checkpoint=checkpoint,
        commit_training_report=report,
        development_pairs=pairs,
        development_pairs_report=pairs_report,
        application_report=application,
        selections=selections,
        output=tmp_path / "validation.json",
    )


def test_q36_commit_application_validation_is_label_free(tmp_path: Path) -> None:
    result = validate(_fixture(tmp_path))
    assert result["status"] == "complete"
    assert result["rows"] == 1_289
    assert result["assessor_board_access_count"] == 0


def test_q36_commit_application_validation_admits_scientific_false_order(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    rows = args.selections.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["order_consistent"] = False
    rows[0] = json.dumps(first)
    args.selections.write_text("\n".join(rows) + "\n", encoding="utf-8")
    application = json.loads(args.application_report.read_text(encoding="utf-8"))
    application["selections_sha256"] = _sha(args.selections)
    application["order_consistent"] = 1_288
    args.application_report.write_text(json.dumps(application) + "\n", encoding="utf-8")
    report = json.loads(args.commit_training_report.read_text(encoding="utf-8"))
    report["development_selections_sha256"] = _sha(args.selections)
    args.commit_training_report.write_text(json.dumps(report) + "\n", encoding="utf-8")
    result = validate(args)
    assert result["order_consistent"] == 1_288


def test_q36_commit_application_validation_rejects_inconsistent_count(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    application = json.loads(args.application_report.read_text(encoding="utf-8"))
    application["order_consistent"] = 1_288
    args.application_report.write_text(json.dumps(application) + "\n", encoding="utf-8")
    with pytest.raises(Q36MTRApplicationValidationError):
        validate(args)
