from __future__ import annotations

import json
from pathlib import Path

import pytest

import compare_q36_mtr_semantic_commits as module


def _report(path: Path, correctness: list[bool], *, offset: int = 0) -> Path:
    outcomes = [
        {
            "identity_sha256": f"{offset + index:064x}",
            "task": ("math500", "bbh_logic", "mbpp")[index % 3],
            "correct": correct,
        }
        for index, correct in enumerate(correctness)
    ]
    path.write_text(
        json.dumps(
            {
                "schema": module.INPUT_SCHEMA,
                "status": "complete",
                "split": "development",
                "rows": len(outcomes),
                "correct": sum(correctness),
                "outcomes": outcomes,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_compares_all_variants_and_reports_remaining_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "EXPECTED_ROWS", 4)
    groups = {
        "endpoint": [_report(tmp_path / "endpoint.json", [True, False, True, False])],
        "semantic128": [
            _report(tmp_path / "semantic128.json", [True, True, False, False])
        ],
        "semantic256": [
            _report(tmp_path / "semantic256.json", [True, True, True, False])
        ],
    }
    result = module.compare(groups)
    assert result["rows"] == 4
    assert result["best_variant"] == "semantic256"
    assert result["best_correct"] == 3
    assert result["oracle_correct"] == 3
    pair = result["pairwise"]["endpoint__vs__semantic256"]
    assert pair["first_only_correct"] == 0
    assert pair["second_only_correct"] == 1
    assert pair["first_minus_second_correct"] == -1


def test_rejects_identity_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "EXPECTED_ROWS", 4)
    groups = {
        "a": [_report(tmp_path / "a.json", [True, False, True, False])],
        "b": [_report(tmp_path / "b.json", [True, False, True, False], offset=1)],
    }
    with pytest.raises(module.Q36MTRSemanticComparisonError, match="identities"):
        module.compare(groups)


def test_group_parser_keeps_multiple_reports_per_label() -> None:
    groups = module._group(["owner:/a.json", "owner:/b.json", "commit:/c.json"])
    assert groups == {
        "owner": [Path("/a.json"), Path("/b.json")],
        "commit": [Path("/c.json")],
    }
