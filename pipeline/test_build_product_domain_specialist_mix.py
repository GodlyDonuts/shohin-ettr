import hashlib
import json
from pathlib import Path

import pytest

from build_product_domain_specialist_mix import (
    DomainSpecialistMixError,
    build_domain_mix,
)


def _identity(question: str) -> str:
    return hashlib.sha256(" ".join(question.split()).casefold().encode()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def _positive(question: str, group: str) -> dict:
    return {
        "question": question,
        "response": "reasoning \\boxed{1}",
        "training_group": group,
        "source_identity_sha256": _identity(question),
    }


def _report(tmp_path: Path, rows: list[dict], *, admitted: bool = True) -> Path:
    positives = tmp_path / "positives.jsonl"
    digest = _write_jsonl(positives, rows)
    report = tmp_path / "merge.report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "shohin-product-rollout-positive-merge-v1",
                "status": "complete",
                "admitted": admitted,
                "positives_output": str(positives),
                "positives_sha256": digest,
            }
        )
    )
    return report


def test_builds_one_domain_with_equal_replay(tmp_path: Path) -> None:
    report = _report(
        tmp_path,
        [
            _positive("Math A", "math"),
            _positive("Math B", "math"),
            _positive("Science A", "science"),
        ],
    )
    replay = tmp_path / "replay.jsonl"
    _write_jsonl(
        replay,
        [
            {"question": "Math A", "response": "overlap", "training_group": "math"},
            {"question": "Replay 1", "response": "one", "training_group": "math"},
            {"question": "Replay 2", "response": "two", "domain": "math"},
            {
                "question": "Science replay",
                "response": "three",
                "training_group": "science",
            },
        ],
    )

    result = build_domain_mix(
        report,
        replay,
        tmp_path / "mix.jsonl",
        tmp_path / "mix.report.json",
        domain="MATH",
        replay_multiplier=1.0,
        seed=7,
    )

    rows = [
        json.loads(line) for line in (tmp_path / "mix.jsonl").read_text().splitlines()
    ]
    assert result["positive_rows"] == 2
    assert result["replay_rows_selected"] == 2
    assert result["rows"] == 4
    assert result["counters"]["replay_positive_overlap_drops"] == 1
    assert all(
        (row.get("training_group") or row.get("domain")) == "math" for row in rows
    )
    assert len({_identity(row["question"]) for row in rows}) == 4


def test_rejects_insufficient_domain_replay(tmp_path: Path) -> None:
    report = _report(
        tmp_path, [_positive("Math A", "math"), _positive("Math B", "math")]
    )
    replay = tmp_path / "replay.jsonl"
    _write_jsonl(replay, [{"question": "Only one", "response": "x", "domain": "math"}])

    with pytest.raises(DomainSpecialistMixError, match="capacity"):
        build_domain_mix(
            report,
            replay,
            tmp_path / "mix.jsonl",
            tmp_path / "mix.report.json",
            domain="math",
            replay_multiplier=1.0,
            seed=7,
        )


def test_rejects_hash_mismatch(tmp_path: Path) -> None:
    report = _report(tmp_path, [_positive("Math A", "math")])
    payload = json.loads(report.read_text())
    payload["positives_sha256"] = "0" * 64
    report.write_text(json.dumps(payload))
    replay = tmp_path / "replay.jsonl"
    _write_jsonl(replay, [{"question": "Replay", "response": "x", "domain": "math"}])

    with pytest.raises(DomainSpecialistMixError, match="hash differs"):
        build_domain_mix(
            report,
            replay,
            tmp_path / "mix.jsonl",
            tmp_path / "mix.report.json",
            domain="math",
            replay_multiplier=1.0,
            seed=7,
        )
