import hashlib
import json
from pathlib import Path

import pytest

from select_product_reward_frontier import (
    RewardFrontierSelectionError,
    select_reward_frontier,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _reward(identity: str, answer: str = "2") -> dict:
    return {
        "identity_sha256": identity,
        "question": f"Question {identity}",
        "task": "math500",
        "answer": rf"\boxed{{{answer}}}",
        "expected_answer_normalized": answer,
    }


def _mask(identity: str, answer: str = "2") -> dict:
    return {
        "source_identity_sha256": identity,
        "question": f"Question {identity}",
        "expected_answer_normalized": answer,
        "verification": "expected_answer_match_v1",
    }


def test_selects_exact_reward_rows_in_reward_order(tmp_path: Path) -> None:
    reward = tmp_path / "reward.jsonl"
    mask = tmp_path / "mask.jsonl"
    output = tmp_path / "output.jsonl"
    _write(reward, [_reward("b"), _reward("a"), _reward("c")])
    _write(mask, [_mask("a"), _mask("b")])

    report = select_reward_frontier(
        reward,
        mask,
        output,
        tmp_path / "report.json",
    )

    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert selected == [_reward("b"), _reward("a")]
    assert report["reward_rows"] == 3
    assert report["selected_rows"] == 2
    encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    assert report["output_sha256"] == hashlib.sha256(encoded.encode()).hexdigest()


def test_rejects_frontier_identity_missing_from_reward_bank(tmp_path: Path) -> None:
    reward = tmp_path / "reward.jsonl"
    mask = tmp_path / "mask.jsonl"
    _write(reward, [_reward("a")])
    _write(mask, [_mask("missing")])

    with pytest.raises(RewardFrontierSelectionError, match="absent"):
        select_reward_frontier(
            reward,
            mask,
            tmp_path / "output.jsonl",
            tmp_path / "report.json",
        )


def test_rejects_joined_answer_mismatch(tmp_path: Path) -> None:
    reward = tmp_path / "reward.jsonl"
    mask = tmp_path / "mask.jsonl"
    _write(reward, [_reward("a", "2")])
    _write(mask, [_mask("a", "3")])

    with pytest.raises(RewardFrontierSelectionError, match="answer differs"):
        select_reward_frontier(
            reward,
            mask,
            tmp_path / "output.jsonl",
            tmp_path / "report.json",
        )
