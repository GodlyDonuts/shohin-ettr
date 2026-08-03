import hashlib
import json
from pathlib import Path

import pytest

from build_product_rollout_replay_mix import (
    ProductRolloutReplayMixError,
    build_mix,
)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def _positive(question: str, group: str) -> dict:
    normalized = " ".join(question.split()).casefold()
    return {
        "question": question,
        "response": "reasoning \\boxed{1}",
        "training_group": group,
        "verification": "student_exact_answer_match_v1",
        "source_identity_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
    }


def _positive_report(tmp_path: Path, rows: list[dict], admitted: bool = True) -> Path:
    positives = tmp_path / "positives.jsonl"
    digest = _write_jsonl(positives, rows)
    report = tmp_path / "positives.report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "shohin-product-rollout-positive-merge-v1",
                "status": "complete",
                "admitted": admitted,
                "positive_prompts": len(rows),
                "positives_output": str(positives),
                "positives_sha256": digest,
            }
        )
    )
    return report


def test_builds_balanced_replay_count_and_excludes_overlap(tmp_path: Path) -> None:
    positive_rows = [_positive("Math A", "math"), _positive("Science B", "science")]
    report = _positive_report(tmp_path, positive_rows)
    replay = tmp_path / "replay.jsonl"
    _write_jsonl(
        replay,
        [
            {"question": "Math A", "response": "old overlap"},
            {"question": "Replay 1", "response": "one", "training_group": "code"},
            {"question": "Replay 2", "response": "two", "training_group": "logic"},
            {"question": "Replay 3", "response": "three", "training_group": "math"},
        ],
    )

    result = build_mix(
        report,
        replay,
        tmp_path / "mix.jsonl",
        tmp_path / "mix.report.json",
        replay_multiplier=1.0,
        seed=7,
    )

    rows = [json.loads(line) for line in (tmp_path / "mix.jsonl").read_text().splitlines()]
    assert result["positive_rows"] == 2
    assert result["replay_rows_selected"] == 2
    assert result["rows"] == 4
    assert result["counters"]["replay_positive_overlap_drops"] == 1
    assert len({row["question"] for row in rows}) == 4


def test_rejects_unadmitted_positive_merge(tmp_path: Path) -> None:
    report = _positive_report(tmp_path, [_positive("Math A", "math")], admitted=False)
    replay = tmp_path / "replay.jsonl"
    _write_jsonl(replay, [{"question": "Replay", "response": "one"}])
    with pytest.raises(ProductRolloutReplayMixError, match="not admitted"):
        build_mix(
            report,
            replay,
            tmp_path / "mix.jsonl",
            tmp_path / "mix.report.json",
            replay_multiplier=1.0,
            seed=7,
        )


def test_rejects_insufficient_replay_capacity(tmp_path: Path) -> None:
    report = _positive_report(
        tmp_path,
        [_positive("Math A", "math"), _positive("Science B", "science")],
    )
    replay = tmp_path / "replay.jsonl"
    _write_jsonl(replay, [{"question": "Only replay", "response": "one"}])
    with pytest.raises(ProductRolloutReplayMixError, match="capacity"):
        build_mix(
            report,
            replay,
            tmp_path / "mix.jsonl",
            tmp_path / "mix.report.json",
            replay_multiplier=1.0,
            seed=7,
        )
