import json
from pathlib import Path

import pytest

from select_q36_mtr_owner_trajectories import (
    Q36MTROwnerTrajectorySelectionError,
    select,
)


def _rows(owner: str) -> list[dict]:
    values = [
        ("partial thought", True),
        ("The final answer is 7.", True),
        ("The final answer is A.", False),
    ]
    return [
        {
            "schema": "shohin-q36-mtr-model-draft-v1",
            "identity_sha256": f"{index + 1:064x}",
            "task": "math500" if index < 2 else "bbh_logic",
            "split": "development",
            "prompt_sha256": "a" * 64,
            "model_revision": "revision",
            "owner_checkpoint_sha256": owner * 64,
            "completion": completion,
            "max_token_exhausted": exhausted,
            "generated_tokens": 10,
        }
        for index, (completion, exhausted) in enumerate(values)
    ]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_candidate_only_selection_rule(tmp_path: Path) -> None:
    first = _rows("a")
    second = _rows("b")
    second[0]["completion"] = "The final answer is 4."
    second[1]["max_token_exhausted"] = False
    second[2]["completion"] = "different partial thought"
    selected, report = select(
        _write(tmp_path / "first.jsonl", first),
        _write(tmp_path / "second.jsonl", second),
    )
    assert [row["trajectory_selection"]["choice"] for row in selected] == [
        "second",
        "second",
        "first",
    ]
    assert report["selection_counts"] == {"first": 1, "second": 2}
    assert report["adaptive_generation"]["second_trajectory_calls"] == 2
    assert report["adaptive_generation"][
        "trajectory_calls_per_identity"
    ] == pytest.approx(5 / 3)
    assert report["answer_labels_read"] == 0
    assert report["assessor_fields_read"] == 0


def test_selection_rejects_identity_reorder(tmp_path: Path) -> None:
    first = _rows("a")
    second = list(reversed(_rows("b")))
    with pytest.raises(Q36MTROwnerTrajectorySelectionError, match="alignment"):
        select(
            _write(tmp_path / "first.jsonl", first),
            _write(tmp_path / "second.jsonl", second),
        )


def test_selection_rejects_duplicate_identity(tmp_path: Path) -> None:
    first = _rows("a")
    first[1]["identity_sha256"] = first[0]["identity_sha256"]
    with pytest.raises(Q36MTROwnerTrajectorySelectionError, match="identities"):
        select(
            _write(tmp_path / "first.jsonl", first),
            _write(tmp_path / "second.jsonl", _rows("b")),
        )


def test_selection_can_project_one_shared_split(tmp_path: Path) -> None:
    first = _rows("a")
    train = dict(first[0])
    train["identity_sha256"] = "f" * 64
    train["split"] = "train"
    first.append(train)
    selected, report = select(
        _write(tmp_path / "first.jsonl", first),
        _write(tmp_path / "second.jsonl", _rows("b")),
        "development",
    )
    assert len(selected) == 3
    assert report["selected_split"] == "development"
    assert {row["split"] for row in selected} == {"development"}
