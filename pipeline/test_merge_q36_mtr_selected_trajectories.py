import argparse
import hashlib
import json
from pathlib import Path

import pytest

import merge_q36_mtr_selected_trajectories as module


def _candidate(identity: str, source_prompt: str, choice: str) -> dict:
    first = "a" * 64
    second = "b" * 64
    return {
        "schema": "shohin-q36-mtr-model-draft-v1",
        "identity_sha256": identity,
        "split": "development",
        "task": "math500",
        "prompt_sha256": hashlib.sha256(source_prompt.encode()).hexdigest(),
        "owner_checkpoint_sha256": first if choice == "first" else second,
        "model_revision": module.MODEL_REVISION,
        "completion": "The final answer is 4.",
        "generated_tokens": 8,
        "max_token_exhausted": False,
        "finish_reason": "stop",
        "wall_seconds": 1.0,
        "trajectory_selection": {
            "schema": "shohin-q36-mtr-owner-trajectory-selection-v1",
            "rule": module.RULE,
            "choice": choice,
            "reason": (
                "explicit_final_answer" if choice == "second" else "retained_first"
            ),
            "first_owner_checkpoint_sha256": first,
            "second_owner_checkpoint_sha256": second,
        },
    }


def _fixture(tmp_path: Path, monkeypatch) -> argparse.Namespace:
    monkeypatch.setattr(module, "DRAFT_IDENTITIES", 2)
    monkeypatch.setattr(module, "DRAFT_SHARDS", 2)
    sources = [
        {
            "identity_sha256": f"{index + 1:064x}",
            "split": "development",
            "task": "math500",
            "source_prompt": f"problem {index}",
        }
        for index in range(2)
    ]
    monkeypatch.setattr(
        module,
        "load_sources",
        lambda *_: (sources, {"identity_receipts": {"development": "c" * 64}}),
    )
    reports = []
    candidates = []
    for index, source in enumerate(sources):
        candidate_path = tmp_path / f"c{index}.jsonl"
        candidate_path.write_text(
            json.dumps(
                _candidate(
                    source["identity_sha256"],
                    source["source_prompt"],
                    "first" if index == 0 else "second",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        report_path = tmp_path / f"r{index}.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema": module.REPORT_SCHEMA,
                    "status": "complete",
                    "rule": module.RULE,
                    "rows": 1,
                    "answer_labels_read": 0,
                    "assessor_fields_read": 0,
                    "output": str(candidate_path.resolve()),
                    "output_sha256": module.sha256_file(candidate_path),
                    "first_candidates_sha256": "d" * 64,
                    "second_candidates_sha256": "e" * 64,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        candidates.append(candidate_path)
        reports.append(report_path)
    return argparse.Namespace(
        train_source=tmp_path / "train",
        development_source=tmp_path / "dev",
        freeze_report=tmp_path / "freeze",
        selection_reports=reports,
        selected_candidates=candidates,
        output=tmp_path / "merged.jsonl",
        report=tmp_path / "report.json",
    )


def test_merges_selected_trajectories_with_synthetic_lineage(
    tmp_path: Path, monkeypatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    result = module.merge(args)
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert result["rows"] == 2
    assert result["selection_counts"] == {"first": 1, "second": 1}
    assert len({row["owner_checkpoint_sha256"] for row in rows}) == 1
    assert (
        rows[0]["trajectory_selection"]["selected_owner_checkpoint_sha256"] == "a" * 64
    )
    assert (
        rows[1]["trajectory_selection"]["selected_owner_checkpoint_sha256"] == "b" * 64
    )


def test_rejects_selection_hash_tamper(tmp_path: Path, monkeypatch) -> None:
    args = _fixture(tmp_path, monkeypatch)
    report = json.loads(args.selection_reports[0].read_text())
    report["output_sha256"] = "0" * 64
    args.selection_reports[0].write_text(json.dumps(report) + "\n")
    with pytest.raises(module.Q36MTRSelectedTrajectoryMergeError, match="report"):
        module.merge(args)
