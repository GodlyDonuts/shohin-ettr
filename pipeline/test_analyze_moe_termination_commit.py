from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyze_moe_termination_commit import TerminationCommitError, replay


def identity(index: int) -> str:
    return f"{index:064x}"


def write_rows(path: Path, arm: str, exhausted: list[bool]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for index, value in enumerate(exhausted, start=1):
            handle.write(
                json.dumps(
                    {
                        "schema": "candidate-v1",
                        "arm": arm,
                        "identity_sha256": identity(index),
                        "task": ("bbh_logic", "math500", "mbpp", "bbh_logic")[
                            index - 1
                        ],
                        "completion": f"answer-{arm}-{index}",
                        "generated_tokens": 8,
                        "max_token_exhausted": value,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def fixture(tmp_path: Path) -> tuple[Path, dict[str, list[Path]]]:
    candidates = {
        "unchanged": [
            write_rows(tmp_path / "u.jsonl", "unchanged", [True, False, False, False])
        ],
        "self_refinement": [
            write_rows(
                tmp_path / "s.jsonl", "self_refinement", [True, True, False, False]
            )
        ],
        "revision": [
            write_rows(tmp_path / "r.jsonl", "revision", [False, False, False, True])
        ],
    }
    outcomes = [
        {
            "identity_sha256": identity(1),
            "task": "bbh_logic",
            "correct": {"unchanged": False, "self_refinement": False, "revision": True},
        },
        {
            "identity_sha256": identity(2),
            "task": "math500",
            "correct": {"unchanged": True, "self_refinement": False, "revision": True},
        },
        {
            "identity_sha256": identity(3),
            "task": "mbpp",
            "correct": {"unchanged": True, "self_refinement": True, "revision": False},
        },
        {
            "identity_sha256": identity(4),
            "task": "bbh_logic",
            "correct": {"unchanged": False, "self_refinement": True, "revision": False},
        },
    ]
    score = tmp_path / "score.json"
    score.write_text(json.dumps({"schema": "score-v1", "outcomes": outcomes}) + "\n")
    return score, candidates


def test_replay_uses_only_exhaustion_and_preserves_completed_baseline(
    tmp_path: Path,
) -> None:
    score, candidates = fixture(tmp_path)
    report = replay(host="fixture", score=score, candidate_paths=candidates)
    self_selector = report["selectors"]["self_refinement"]
    assert self_selector["baseline_correct"] == 2
    assert self_selector["selected_correct"] == 4
    assert self_selector["retained_baseline_correct"] == 2
    assert self_selector["baseline_correct_retention"] == 1.0
    assert self_selector["revision_selected"] == 2
    assert self_selector["paired_wins"] == 2
    assert self_selector["paired_losses"] == 0
    assert report["rule"]["uses_correctness_at_selection"] is False
    assert report["rule"]["uses_task_label"] is False


def test_replay_rejects_identity_coverage_drift(tmp_path: Path) -> None:
    score, candidates = fixture(tmp_path)
    candidates["revision"][0].write_text(
        "\n".join(candidates["revision"][0].read_text().splitlines()[:-1]) + "\n"
    )
    with pytest.raises(TerminationCommitError, match="identity coverage"):
        replay(host="fixture", score=score, candidate_paths=candidates)


def test_replay_rejects_task_binding_drift(tmp_path: Path) -> None:
    score, candidates = fixture(tmp_path)
    rows = [
        json.loads(line) for line in candidates["revision"][0].read_text().splitlines()
    ]
    rows[0]["task"] = "math500"
    candidates["revision"][0].write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(TerminationCommitError, match="task binding"):
        replay(host="fixture", score=score, candidate_paths=candidates)


def test_replay_rejects_untyped_exhaustion(tmp_path: Path) -> None:
    score, candidates = fixture(tmp_path)
    rows = [
        json.loads(line) for line in candidates["revision"][0].read_text().splitlines()
    ]
    rows[0]["max_token_exhausted"] = 0
    candidates["revision"][0].write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(TerminationCommitError, match="candidate"):
        replay(host="fixture", score=score, candidate_paths=candidates)
