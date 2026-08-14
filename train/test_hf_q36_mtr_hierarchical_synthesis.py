from __future__ import annotations

import json
from pathlib import Path

import pytest

import hf_q36_mtr_hierarchical_synthesis as module


def _row(identity: str, schema: str = module.SCHEMA) -> dict:
    return {
        "schema": schema,
        "identity_sha256": identity,
        "split": "development",
        "task": "math500",
        "completion": "A nonempty completion",
        "generated_tokens": 4,
        "max_token_exhausted": False,
    }


def test_hierarchical_prompt_preserves_primary_and_hides_architecture_names() -> None:
    prompt = module.hierarchical_prompt(
        "Original question", "integrated answer", "stacked answer", "review answer"
    )
    assert "Preserve Candidate A unless" in prompt
    assert "integrated answer" in prompt
    assert "stacked answer" in prompt
    assert "review answer" in prompt
    assert "owner_71" not in prompt
    assert "development" not in prompt


def test_hierarchical_prompt_rejects_empty_input() -> None:
    with pytest.raises(module.Q36MTRHierarchicalSynthesisError):
        module.hierarchical_prompt("question", "", "other", "review")


def test_candidate_loader_accepts_model_and_control_schemas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "ROWS", 2)
    path = tmp_path / "candidates.jsonl"
    rows = [_row("0" * 64), _row("1" * 64, "shohin-q36-mtr-candidate-v1")]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    loaded = module.load_candidate_group([path], expected_paths=1)
    assert list(loaded) == ["0" * 64, "1" * 64]


def test_candidate_loader_rejects_duplicate_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "ROWS", 1)
    path = tmp_path / "candidates.jsonl"
    row = _row("0" * 64)
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(module.Q36MTRHierarchicalSynthesisError):
        module.load_candidate_group([path], expected_paths=1)
