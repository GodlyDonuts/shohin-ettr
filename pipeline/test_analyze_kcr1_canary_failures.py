"""Focused tests for KCR1 canary failure attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from analyze_kcr1_canary_failures import KCR1AttributionError, run


ACTIONS = ("<KEEP>", "<CONTINUE>", "<RESTART>")


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    data = tmp_path / "data.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    data_rows = []
    candidate_rows = []
    for source_index in range(522):
        for action_index, action in enumerate(ACTIONS):
            identity = f"{source_index:04d}-{action_index}"
            data_rows.append(
                {
                    "identity_sha256": identity,
                    "source_identity_sha256": f"source-{source_index:04d}",
                    "expected_action": action,
                    "presentation": f"p{action_index}",
                }
            )
            correct = action_index != 2
            candidate_rows.append(
                {
                    "identity_sha256": identity,
                    "expected_action": action,
                    "predicted_action": action,
                    "action_correct": True,
                    "correct": correct,
                    "execution_exact": correct,
                    "valid_transaction": True,
                    "max_token_exhausted": not correct,
                    "generated_tokens": action_index + 1,
                }
            )
    data.write_text("".join(json.dumps(row) + "\n" for row in data_rows))
    candidates.write_text("".join(json.dumps(row) + "\n" for row in candidate_rows))
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "shohin-kcr1-transaction-evaluation-v1",
                "status": "complete",
                "split": "development",
                "merged_from_shards": True,
                "full_row_count": 1566,
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "candidates_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
            }
        )
    )
    return data, candidates, report


def test_attribution_localizes_action_correct_payload_failure(tmp_path: Path) -> None:
    data, candidates, report = _write_fixture(tmp_path)
    result = run(data, candidates, report)
    assert result["holdout_used"] is False
    assert result["overall"]["action_correct"] == 1566
    assert result["by_expected_action"]["<RESTART>"]["action_correct_semantic_wrong"] == 522
    assert result["source_outcomes"]["all_actions_correct"] == 522
    assert result["source_outcomes"]["semantic_inconsistent"] == 522


def test_attribution_rejects_candidate_hash_mismatch(tmp_path: Path) -> None:
    data, candidates, report = _write_fixture(tmp_path)
    candidates.write_text(candidates.read_text() + "\n")
    with pytest.raises(KCR1AttributionError):
        run(data, candidates, report)
