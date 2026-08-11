import argparse
import json
from pathlib import Path

from analyze_ndr1_state_failure import analyze


def _write_lines(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_attributes_deficit_to_wrong_exhausted_state(tmp_path: Path) -> None:
    identities = [f"{index:064x}" for index in range(4)]
    data = []
    aligned = []
    shuffled = []
    states = [(False, True), (False, True), (True, False), (False, False)]
    outcomes = [(False, True), (False, False), (True, True), (True, False)]
    for identity, (draft_correct, draft_exhausted), (left, right) in zip(
        identities, states, outcomes, strict=True
    ):
        data.append(
            {
                "identity_sha256": identity,
                "task": "math500",
                "internal_draft": {
                    "identity_sha256": identity,
                    "completion": "draft",
                    "prediction": "1",
                    "correct": draft_correct,
                    "max_token_exhausted": draft_exhausted,
                },
            }
        )
        for rows, correct in ((aligned, left), (shuffled, right)):
            rows.append(
                {
                    "identity_sha256": identity,
                    "task": "math500",
                    "completion": "draft" if correct else "other",
                    "prediction": "1" if correct else "0",
                    "correct": correct,
                    "generated_tokens": 10,
                    "max_token_exhausted": False,
                }
            )
    data_path = tmp_path / "development.jsonl"
    aligned_path = tmp_path / "aligned.jsonl"
    shuffled_path = tmp_path / "shuffled.jsonl"
    report_path = tmp_path / "training.json"
    output = tmp_path / "result.json"
    _write_lines(data_path, data)
    _write_lines(aligned_path, aligned)
    _write_lines(shuffled_path, shuffled)
    report_path.write_text(
        json.dumps(
            {
                "schema": "shohin-ndr1-natural-revision-data-report-v1",
                "status": "complete",
                "natural_drafts_only": True,
                "admitted_rows_per_arm": 10,
                "draft_exhausted_rows_per_arm": 1,
            }
        ),
        encoding="utf-8",
    )
    result = analyze(
        argparse.Namespace(
            development_data=data_path,
            aligned_candidates=aligned_path,
            shuffled_candidates=shuffled_path,
            training_data_report=report_path,
            expected_rows=4,
            output=output,
        )
    )
    assert result["aligned_minus_shuffled_answers"] == 0
    assert result["wrong_exhausted_aligned_minus_shuffled_answers"] == -1
    assert result["evaluation_draft_state"]["exhausted"] == 2
    assert result["by_draft_state"]["draft_wrong_exhausted"]["shuffled_only"] == 1
    assert json.loads(output.read_text())["schema"] == result["schema"]
