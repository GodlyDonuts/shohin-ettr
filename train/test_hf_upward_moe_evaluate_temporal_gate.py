import hashlib
import json
from pathlib import Path

import pytest

import hf_upward_moe_evaluate_temporal_gate as module


def _row(identity: str) -> dict:
    return {
        "schema": module.DATA_SCHEMA,
        "split": "development",
        "identity_sha256": identity,
        "task": "math500",
        "source_prompt": "Solve the problem.",
        "question": "Solve the problem.\n\nInternal model-owned draft:\nDraft",
        "internal_draft": {
            "identity_sha256": identity,
            "completion": "Draft",
        },
        "internal_draft_visible": True,
        "external_candidate_text_visible": False,
    }


def test_development_loader_is_label_free_exact_and_sorted(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "development.jsonl"
    rows = [_row("2" * 64), _row("1" * 64)]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "ROWS", 2)
    loaded = module.load_evaluation_rows(path, digest)
    assert [row["identity_sha256"] for row in loaded] == ["1" * 64, "2" * 64]
    rows[0]["answer"] = "leak"
    path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[1]) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(module.UpwardMoETemporalEvaluationError):
        module.load_evaluation_rows(path, digest)


def test_matched_prompt_projections_are_exact() -> None:
    row = _row("1" * 64)
    assert module.prompt_for("unchanged", row) == row["source_prompt"]
    assert module.prompt_for("owner", row) == row["source_prompt"]
    assert module.prompt_for("aligned_revision", row) == row["question"]
    assert module.prompt_for("temporal_gate", row) == row["question"]
    refined = module.prompt_for("self_refinement", row)
    assert row["source_prompt"] in refined and "Draft" in refined
    with pytest.raises(module.UpwardMoETemporalEvaluationError):
        module.prompt_for("router-selector", row)


def test_upward_temporal_arms_include_conservative_controls() -> None:
    assert module.ARMS == (
        "unchanged",
        "self_refinement",
        "owner",
        "aligned_revision",
        "temporal_gate",
    )
    assert module.ROWS == 1289
    assert module.SHARDS == 16
