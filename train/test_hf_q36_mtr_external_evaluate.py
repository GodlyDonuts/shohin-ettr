from __future__ import annotations

import hashlib
import json

import pytest

import hf_q36_mtr_external_evaluate as module


def _identity(index: int) -> str:
    return hashlib.sha256(f"external-eval-{index}".encode()).hexdigest()


def _source(index: int, task: str) -> dict:
    return {
        "schema": module.SOURCE_SCHEMA,
        "identity_sha256": _identity(index),
        "split": "external_validation",
        "task": task,
        "source_prompt": f"Problem {index}",
        "runtime_fields": ["source_prompt"],
        "supervisor_only_fields": ["task"],
    }


def test_load_sources_and_drafts(tmp_path):
    sources = [_source(0, "math500"), _source(1, "bbh_logic"), _source(2, "mbpp")]
    source_path = tmp_path / "sources.jsonl"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in sources))
    loaded = module.load_sources(source_path, 3)
    draft_path = tmp_path / "drafts.jsonl"
    draft_path.write_text(
        "".join(
            json.dumps(
                {
                    "schema": module.CANDIDATE_SCHEMA,
                    "arm": "unchanged",
                    "identity_sha256": row["identity_sha256"],
                    "task": row["task"],
                    "completion": f"Draft {index}",
                }
            )
            + "\n"
            for index, row in enumerate(sources)
        )
    )
    drafts = module.load_drafts([draft_path], loaded)
    assert set(drafts) == {row["identity_sha256"] for row in sources}


def test_prompt_contracts():
    source = _source(0, "math500")
    draft = {
        "identity_sha256": source["identity_sha256"],
        "completion": "Draft answer",
    }
    assert module.prompt_for("unchanged", source, None) == "Problem 0"
    self_refinement = module.prompt_for("self_refinement", source, draft)
    revision = module.prompt_for("revision", source, draft)
    hidden = module.prompt_for("draft_hidden", source, draft)
    interpolation = module.prompt_for("interpolation", source, draft)
    assert "Draft answer" in self_refinement
    assert revision == hidden == interpolation
    assert "Internal draft:\nDraft answer" in revision


def test_supervision_in_source_fails(tmp_path):
    rows = [_source(0, "math500"), _source(1, "bbh_logic"), _source(2, "mbpp")]
    rows[0]["answer"] = "leak"
    path = tmp_path / "sources.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(module.Q36MTRExternalEvaluationError, match="projection"):
        module.load_sources(path, 3)


def test_external_screen_and_full_geometry_are_distinct():
    assert module.SHARD_COUNTS == {256: 4, 1_023: 16, 1_279: 16}


def test_mmlu_confirmation_source_is_explicit_and_label_free(tmp_path):
    rows = [_source(index, "mmlu_pro") for index in range(3)]
    path = tmp_path / "mmlu.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded = module.load_sources(path, 3, module.MMLU_CONFIRMATION_TASKS)
    assert {row["task"] for row in loaded} == {"mmlu_pro"}
    with pytest.raises(module.Q36MTRExternalEvaluationError, match="source"):
        module.load_sources(path, 3)


def test_mmlu_fixed_draft_uses_aligned_adapter_with_source_only_prompt() -> None:
    assert module.prompt_for("unchanged", _source(0, "mmlu_pro"), None) == "Problem 0"
    assert module.adapter_validation_arm("unchanged", True) == "revision"
    assert module.adapter_validation_arm("unchanged", False) == "unchanged"
    with pytest.raises(
        module.Q36MTRExternalEvaluationError, match="confirmation adapter arm"
    ):
        module.adapter_validation_arm("revision", True)


def test_external_job_accepts_independent_shard_identity() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "train/jobs/q36_mtr_external_evaluate.sbatch").read_text()
    assert "SHARD_INDEX" in wrapper
    assert "task_index=${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-}}" in wrapper
    assert "--confirmation-mmlu-pro" in wrapper
