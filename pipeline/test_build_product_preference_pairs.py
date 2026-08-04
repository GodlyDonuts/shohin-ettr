import hashlib
import json
from pathlib import Path

import pytest

from build_product_preference_pairs import (
    ProductPreferencePairError,
    build_pairs,
)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def _candidate(identity: str, sample: int, correct: bool, group: str = "math") -> dict:
    return {
        "identity_sha256": identity,
        "question": f"Question {identity}",
        "completion": f"Completion {identity} {sample}",
        "correct": correct,
        "sample_index": sample,
        "generated_tokens": sample + 10,
        "training_group": group,
    }


def _aggregate(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    candidates = tmp_path / f"{name}.jsonl"
    digest = _write_jsonl(candidates, rows)
    report = tmp_path / f"{name}.report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "shohin-product-rollout-aggregate-v1",
                "admitted": True,
                "admission_failures": [],
                "candidates": len(rows),
                "candidates_output": str(candidates),
                "candidates_sha256": digest,
                "contract": {
                    "adapter_checkpoint": "/checkpoint.pt",
                    "model_revision": "revision",
                },
            }
        )
    )
    return report


def test_builds_only_mixed_within_prompt_pairs_deterministically(tmp_path: Path) -> None:
    report = _aggregate(
        tmp_path,
        "math",
        [
            _candidate("mixed", 0, True),
            _candidate("mixed", 1, True),
            _candidate("mixed", 2, False),
            _candidate("mixed", 3, False),
            {**_candidate("empty", 0, False), "completion": ""},
            *[_candidate("all-good", index, True) for index in range(4)],
            *[_candidate("all-bad", index, False) for index in range(4)],
        ],
    )
    output = tmp_path / "pairs.jsonl"
    result = build_pairs(
        [report],
        output,
        tmp_path / "pairs.report.json",
        pairs_per_prompt=2,
        seed=7,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["mixed_prompts"] == 1
    assert result["all_correct_prompts"] == 1
    assert result["all_wrong_prompts"] == 1
    assert result["counters"]["empty_completion_drops"] == 1
    assert result["pairs"] == 2
    assert all(row["identity_sha256"] == "mixed" for row in rows)
    assert all(row["chosen"] != row["rejected"] for row in rows)


def test_rejects_candidate_hash_mismatch(tmp_path: Path) -> None:
    report = _aggregate(
        tmp_path,
        "math",
        [_candidate("mixed", 0, True), _candidate("mixed", 1, False)],
    )
    payload = json.loads(report.read_text())
    payload["candidates_sha256"] = "0" * 64
    report.write_text(json.dumps(payload))
    with pytest.raises(ProductPreferencePairError, match="hash"):
        build_pairs(
            [report],
            tmp_path / "pairs.jsonl",
            tmp_path / "pairs.report.json",
            pairs_per_prompt=1,
            seed=7,
        )


def test_rejects_cross_report_model_contract_mismatch(tmp_path: Path) -> None:
    first = _aggregate(
        tmp_path,
        "first",
        [_candidate("a", 0, True), _candidate("a", 1, False)],
    )
    second = _aggregate(
        tmp_path,
        "second",
        [_candidate("b", 0, True), _candidate("b", 1, False, "science")],
    )
    payload = json.loads(second.read_text())
    payload["contract"]["adapter_checkpoint"] = "/different.pt"
    second.write_text(json.dumps(payload))
    with pytest.raises(ProductPreferencePairError, match="contracts differ"):
        build_pairs(
            [first, second],
            tmp_path / "pairs.jsonl",
            tmp_path / "pairs.report.json",
            pairs_per_prompt=1,
            seed=7,
        )
