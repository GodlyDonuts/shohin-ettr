from __future__ import annotations

from pathlib import Path

from hf_sctr1_evaluate import summarize


SOURCE = (Path(__file__).resolve().parent / "hf_sctr1_evaluate.py").read_text(
    encoding="utf-8"
)


def _row(index: int, task: str, expected: str) -> dict:
    return {
        "identity_sha256": f"{index:064x}",
        "task": task,
        "expected_command": expected,
        "candidates": [
            {"lineage": "base", "correct": False},
            {"lineage": "expert", "correct": False},
        ],
    }


def test_summary_reports_task_scores_and_commit_decisions() -> None:
    rows = [
        _row(1, "math500", "keep"),
        _row(2, "bbh_logic", "revise"),
        _row(3, "mbpp", "keep"),
    ]
    results = [
        {"identity_sha256": rows[0]["identity_sha256"], "correct": True, "commit_command": "keep"},
        {"identity_sha256": rows[1]["identity_sha256"], "correct": True, "commit_command": "revise"},
        {"identity_sha256": rows[2]["identity_sha256"], "correct": False, "commit_command": "malformed"},
    ]
    report = summarize(rows, results)
    assert report["metrics"]["overall"]["generated_correct"] == 2
    assert report["commitment"]["command_correct"] == 2
    assert report["commitment"]["malformed"] == 1


def test_independent_evaluation_masks_only_when_checkpoint_contract_matches() -> None:
    assert "checkpoint_masks_draft != args.mask_internal_draft" in SOURCE
    assert "tokenize_with_draft_mask" in SOURCE
    assert '"masked_draft_tokens": masked_draft_tokens' in SOURCE
