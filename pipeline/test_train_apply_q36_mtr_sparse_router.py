from __future__ import annotations

import json
from pathlib import Path

import pytest

import train_apply_q36_mtr_sparse_router as module


def _candidate(lineage: str, correct: bool) -> dict:
    completion = (
        "Careful derivation. The answer is 42."
        if correct
        else "Unfinished speculation without a conclusion"
    )
    return {
        "lineage": lineage,
        "completion": completion,
        "correct": correct,
        "generated_tokens": len(completion.split()),
        "max_token_exhausted": not correct,
    }


def _training_rows() -> list[dict]:
    rows = []
    for index in range(8):
        pattern = f"{index:03b}"
        rows.append(
            {
                "schema": module.ROW_SCHEMA,
                "identity_sha256": hashlib_sha(f"train-{index}"),
                "split": (
                    "calibration_development"
                    if index in {1, 6}
                    else "calibration_train"
                ),
                "task": "math500",
                "question": f"Compute the requested value {index}.",
                "correctness_pattern": pattern,
                "candidates": [
                    _candidate(lineage, bit == "1")
                    for lineage, bit in zip(module.LINEAGES, pattern, strict=True)
                ],
            }
        )
    return rows


def hashlib_sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _write_lines(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_features_are_deterministic_and_owner_conditioned() -> None:
    candidate = _candidate("current", True)
    first = module.candidate_features("What is 6*7?", "math500", "current", candidate)
    second = module.candidate_features("What is 6*7?", "math500", "current", candidate)
    other = module.candidate_features("What is 6*7?", "math500", "owner_8", candidate)
    assert first == second
    assert first != other
    assert abs(sum(value * value for value in first.values()) - 1.0) < 1e-9


def test_training_rows_require_all_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "TRAIN_ROWS", 8)
    path = tmp_path / "training.jsonl"
    rows = _training_rows()
    _write_lines(path, rows)
    assert len(module.load_training_rows(path)) == 8
    rows[-1]["correctness_pattern"] = "000"
    _write_lines(path, rows)
    with pytest.raises(module.Q36MTRSparseRouterError):
        module.load_training_rows(path)


def test_end_to_end_router_uses_no_development_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "TRAIN_ROWS", 8)
    monkeypatch.setattr(module, "DEVELOPMENT_ROWS", 1)
    training = tmp_path / "training.jsonl"
    development = tmp_path / "development.jsonl"
    _write_lines(training, _training_rows())
    identity = hashlib_sha("development")
    _write_lines(
        development,
        [
            {
                "schema": module.ROW_SCHEMA,
                "identity_sha256": identity,
                "split": "development",
                "task": "math500",
                "question": "Compute 6*7.",
                "candidates": [
                    {"lineage": lineage, "completion": f"candidate {lineage}"}
                    for lineage in module.LINEAGES
                ],
            }
        ],
    )
    owner_paths: list[list[Path]] = [[], [], []]
    for owner_index, lineage in enumerate(module.LINEAGES):
        for shard in range(16):
            path = tmp_path / f"{lineage}-{shard}.jsonl"
            row = {
                "schema": module.CANDIDATE_SCHEMA,
                "identity_sha256": (
                    identity
                    if shard == 0
                    else hashlib_sha(f"ignored-{lineage}-{shard}")
                ),
                "split": "development" if shard == 0 else "train",
                "task": "math500",
                "completion": (
                    "The answer is 42." if owner_index == 2 else "No conclusion yet"
                ),
                "generated_tokens": 5,
                "max_token_exhausted": owner_index != 2,
            }
            _write_lines(path, [row])
            owner_paths[owner_index].append(path)
    args = type(
        "Args",
        (),
        {
            "training_rows": training,
            "development_rows": development,
            "current_candidates": owner_paths[0],
            "owner71_candidates": owner_paths[1],
            "owner8_candidates": owner_paths[2],
            "model_output": tmp_path / "model.json",
            "output": tmp_path / "candidates.jsonl",
            "selections": tmp_path / "selections.jsonl",
            "report": tmp_path / "report.json",
        },
    )()
    report = module.run(args)
    assert report["status"] == "complete"
    assert report["development_labels_read"] == 0
    assert report["rows"] == 1
    selected = json.loads(args.output.read_text().strip())
    assert "correct" not in selected
    assert selected["sparse_router_selection"]["selected_lineage"] in module.LINEAGES
