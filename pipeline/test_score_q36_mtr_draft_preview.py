import json
from pathlib import Path

import pytest

import score_q36_mtr_draft_preview as module


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_preview_scores_exact_development_subset(tmp_path: Path, monkeypatch) -> None:
    candidates = tmp_path / "candidates.jsonl"
    assessors = tmp_path / "assessors.jsonl"
    identities = [f"{index:064x}" for index in range(3)]
    tasks = ["math500", "bbh_logic", "mbpp"]
    _write(
        candidates,
        [
            {
                "schema": "shohin-q36-mtr-model-draft-v1",
                "identity_sha256": identity,
                "split": "development",
                "task": task,
                "completion": f"answer-{index}",
                "max_token_exhausted": index == 2,
            }
            for index, (identity, task) in enumerate(zip(identities, tasks))
        ],
    )
    _write(
        assessors,
        [
            {
                "identity_sha256": identity,
                "task": task,
                "assessor": {
                    "identity_sha256": identity,
                    "task": task,
                    "answer": f"answer-{index}",
                },
            }
            for index, (identity, task) in enumerate(zip(identities, tasks))
        ],
    )
    monkeypatch.setattr(module, "qualify_allocation", lambda: {"status": "pass"})
    monkeypatch.setattr(
        module,
        "qualify_mbpp_assessor_setups",
        lambda rows: [{"receipt_sha256": "b" * 64}],
    )
    monkeypatch.setattr(
        module,
        "mbpp_allocation_setup_receipts_sha256",
        lambda rows: "c" * 64,
    )
    monkeypatch.setattr(
        module,
        "score_completion",
        lambda assessor, completion: {
            "correct": assessor["answer"] == completion,
            "explicit_final_answer": True,
        },
    )
    report = module.score_preview(candidates, assessors)
    assert report["rows"] == report["correct"] == 3
    assert report["accuracy"] == 1.0
    assert report["domains"]["mbpp"]["max_token_exhausted"] == 1
    assert report["interpretation"].startswith("exploratory_")


def test_preview_rejects_missing_assessor_identity(tmp_path: Path, monkeypatch) -> None:
    candidates = tmp_path / "candidates.jsonl"
    assessors = tmp_path / "assessors.jsonl"
    _write(
        candidates,
        [
            {
                "schema": "shohin-q36-mtr-model-draft-v1",
                "identity_sha256": "1" * 64,
                "split": "development",
                "task": "math500",
                "completion": "answer",
            }
        ],
    )
    _write(assessors, [])
    monkeypatch.setattr(
        module, "qualify_allocation", lambda: {"receipt_sha256": "a" * 64}
    )
    with pytest.raises(module.Q36MTRDraftPreviewError, match="coverage"):
        module.score_preview(candidates, assessors)


def test_preview_scores_label_free_evaluation_arm(tmp_path: Path, monkeypatch) -> None:
    candidates = tmp_path / "candidates.jsonl"
    assessors = tmp_path / "assessors.jsonl"
    identity = "a" * 64
    _write(
        candidates,
        [
            {
                "schema": "shohin-q36-mtr-candidate-v1",
                "identity_sha256": identity,
                "arm": "unchanged",
                "task": "math500",
                "completion": "42",
                "max_token_exhausted": False,
            }
        ],
    )
    _write(
        assessors,
        [
            {
                "identity_sha256": identity,
                "task": "math500",
                "assessor": {
                    "identity_sha256": identity,
                    "task": "math500",
                    "answer": "42",
                },
            }
        ],
    )
    monkeypatch.setattr(module, "qualify_allocation", lambda: {"status": "pass"})
    monkeypatch.setattr(module, "qualify_mbpp_assessor_setups", lambda rows: [])
    monkeypatch.setattr(
        module, "mbpp_allocation_setup_receipts_sha256", lambda rows: "c" * 64
    )
    monkeypatch.setattr(
        module,
        "score_completion",
        lambda assessor, completion: {
            "correct": assessor["answer"] == completion,
            "explicit_final_answer": True,
        },
    )
    report = module.score_preview(candidates, assessors, evaluation_arm="unchanged")
    assert report["rows"] == report["correct"] == 1
    assert report["evaluation_arm"] == "unchanged"
    assert report["interpretation"] == "engineering_label_free_evaluation_arm"


def test_preview_wrapper_is_cpu_only_and_nonrequeue() -> None:
    source = Path("pipeline/jobs/q36_mtr_score_draft_preview.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres" not in source
    assert "#SBATCH --no-requeue" in source
    assert "PREVIEW_SCRIPT_SHA256" in source
    assert "--split development" in source
    assert '--evaluation-arm "$EVALUATION_ARM"' in source


def test_split_wrapper_supports_training_without_gpu() -> None:
    source = Path("pipeline/jobs/q36_mtr_score_draft_split.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres" not in source
    assert "#SBATCH --no-requeue" in source
    assert 'case "$SPLIT" in train|development)' in source
    assert '--split "$SPLIT"' in source
