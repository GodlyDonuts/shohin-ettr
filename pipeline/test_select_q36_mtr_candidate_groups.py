from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import select_q36_mtr_candidate_groups as module


def _row(identity: str, completion: str, exhausted: bool = False) -> dict:
    return {
        "schema": module.base.INPUT_SCHEMA,
        "identity_sha256": identity,
        "task": "math500",
        "split": "development",
        "prompt_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "model_revision": "revision",
        "completion": completion,
        "generated_tokens": 10,
        "max_token_exhausted": exhausted,
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_group_selection_is_label_free_and_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "EXPECTED_ROWS", 2)
    identities = [
        hashlib.sha256(f"row-{index}".encode()).hexdigest() for index in range(2)
    ]
    first = tmp_path / "first.jsonl"
    second_a = tmp_path / "second-a.jsonl"
    second_b = tmp_path / "second-b.jsonl"
    _write(
        first,
        [
            _row(identities[0], "unfinished attempt", True),
            _row(identities[1], "The answer is 7."),
        ],
    )
    _write(second_a, [_row(identities[0], "The answer is 42.")])
    _write(second_b, [_row(identities[1], "The answer is 8.")])
    args = type(
        "Args",
        (),
        {
            "first_candidates": [first],
            "second_candidates": [second_a, second_b],
            "first_label": "stacked",
            "second_label": "midpoint",
            "split": "development",
            "output": tmp_path / "selected.jsonl",
            "report": tmp_path / "report.json",
        },
    )()
    report = module.select(args)
    assert report["rows"] == 2
    assert report["answer_labels_read"] == 0
    assert report["selection_counts"] == {"first": 1, "second": 1}
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    by_identity = {row["identity_sha256"]: row for row in rows}
    assert by_identity[identities[0]]["completion"] == "The answer is 42."
    assert by_identity[identities[1]]["completion"] == "The answer is 7."
    assert all("correct" not in row for row in rows)


def test_group_overlap_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "EXPECTED_ROWS", 1)
    identity = hashlib.sha256(b"same").hexdigest()
    path = tmp_path / "rows.jsonl"
    _write(path, [_row(identity, "The answer is 1.")])
    with pytest.raises(module.Q36MTRCandidateGroupError):
        module._group([path, path], "development")
