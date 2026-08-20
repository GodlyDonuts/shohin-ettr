from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import analyze_q36_cross_host_commit as module


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> argparse.Namespace:
    module.HOSTS["gpt_oss_120b_screen"] = {
        "rows": 3,
        "score_schema": "score-v1",
    }
    selections = tmp_path / "selections.jsonl"
    rows = []
    for index, lineage in enumerate(("revision", "unchanged", "revision")):
        rows.append(
            {
                "schema": module.SELECTION_SCHEMA,
                "host": "gpt_oss_120b_screen",
                "identity_sha256": str(index) * 64,
                "task": "math500",
                "selected_index": module.LINEAGES.index(lineage),
                "selected_lineage": lineage,
                "margin": 1.0 if lineage == "revision" else -1.0,
                "order_consistent": True,
            }
        )
    selections.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    application = tmp_path / "application.json"
    _json(
        application,
        {
            "schema": "shohin-q36-cross-host-semantic-commit-application-v1",
            "status": "complete",
            "host": "gpt_oss_120b_screen",
            "rows": 3,
            "selections_sha256": module.sha256_file(selections),
            "task_correctness_or_host_label_visible": False,
            "assessor_access_count": 0,
        },
    )
    score = tmp_path / "score.json"
    _json(
        score,
        {
            "schema": "score-v1",
            "status": "complete",
            "rows": 3,
            "outcomes": [
                {
                    "identity_sha256": "0" * 64,
                    "task": "math500",
                    "correct": {"revision": True, "unchanged": False},
                },
                {
                    "identity_sha256": "1" * 64,
                    "task": "math500",
                    "correct": {"revision": False, "unchanged": True},
                },
                {
                    "identity_sha256": "2" * 64,
                    "task": "math500",
                    "correct": {"revision": False, "unchanged": True},
                },
            ],
        },
    )
    return argparse.Namespace(
        host="gpt_oss_120b_screen",
        selections=selections,
        application_report=application,
        score=score,
        output=tmp_path / "result.json",
    )


def test_analysis_replays_only_frozen_arm_outcomes(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    report = module.analyze(args)
    assert report["selected_correct"] == 2
    assert report["unchanged_correct"] == 2
    assert report["revision_correct"] == 1
    assert report["unchanged_correct_retained"] == 1
    assert report["selected_gain_over_unchanged"] == 0
    assert report["selection_had_score_or_assessor_access"] is False


def test_analyzer_registers_1023_row_gpt_confirmation() -> None:
    assert module.HOSTS["gpt_oss_120b_confirmation_1023"] == {
        "rows": 1_023,
        "score_schema": "shohin-gpt-oss-120b-commit-confirmation-score-v1",
    }


def test_analysis_rejects_selection_score_identity_tamper(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    payload = json.loads(args.score.read_text())
    payload["outcomes"][0]["identity_sha256"] = "f" * 64
    _json(args.score, payload)
    with pytest.raises(module.CrossHostAnalysisError, match="identities differ"):
        module.analyze(args)


def test_analysis_binds_prospective_revision_margin_threshold(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.expected_revision_margin_threshold = 0.703125
    rows = [json.loads(line) for line in args.selections.read_text().splitlines()]
    for row in rows:
        row["revision_margin_threshold"] = 0.703125
    args.selections.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    application = json.loads(args.application_report.read_text())
    application["revision_margin_threshold"] = 0.703125
    application["selections_sha256"] = module.sha256_file(args.selections)
    _json(args.application_report, application)
    assert module.analyze(args)["revision_margin_threshold"] == 0.703125

    args.output.unlink()
    rows[0]["revision_margin_threshold"] = 0.7
    args.selections.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(module.CrossHostAnalysisError, match="selection differs"):
        module.analyze(args)


def test_analysis_binds_revision_reliability_veto(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    args.expected_revision_reliability_veto = "empty_or_exhausted"
    rows = [json.loads(line) for line in args.selections.read_text().splitlines()]
    for row in rows:
        row["revision_reliability_veto"] = "empty_or_exhausted"
    args.selections.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    application = json.loads(args.application_report.read_text())
    application["revision_reliability_veto"] = "empty_or_exhausted"
    application["selections_sha256"] = module.sha256_file(args.selections)
    _json(args.application_report, application)
    assert module.analyze(args)["revision_reliability_veto"] == "empty_or_exhausted"

    args.output.unlink()
    application["revision_reliability_veto"] = "none"
    _json(args.application_report, application)
    with pytest.raises(
        module.CrossHostAnalysisError, match="application or score differs"
    ):
        module.analyze(args)
