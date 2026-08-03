import hashlib
import json
from pathlib import Path

import pytest

from merge_product_rollout_positives import (
    ProductRolloutMergeError,
    merge,
)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def _row(identity: str, group: str, response: str = "\\boxed{1}") -> dict:
    return {
        "source_identity_sha256": identity,
        "question": f"question-{identity}",
        "response": response,
        "answer": "\\boxed{1}",
        "expected_answer_normalized": "1",
        "training_group": group,
        "verification": "student_exact_answer_match_v1",
    }


def _aggregate(tmp_path: Path, name: str, rows: list[dict], **contract_updates) -> Path:
    positives = tmp_path / f"{name}.jsonl"
    digest = _write_jsonl(positives, rows)
    contract = {
        "model_root": "/model",
        "model_revision": "revision",
        "adapter_checkpoint": "/checkpoint",
        "samples": 4,
        "max_new_tokens": 1536,
    }
    contract.update(contract_updates)
    report = tmp_path / f"{name}.report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "shohin-product-rollout-aggregate-v1",
                "status": "complete",
                "admitted": True,
                "contract": contract,
                "positive_prompts": len(rows),
                "positive_group_counts": {},
                "positives_output": str(positives),
                "positives_sha256": digest,
            }
        )
    )
    return report


def test_merge_preserves_priority_and_deduplicates(tmp_path: Path) -> None:
    short = _aggregate(tmp_path, "short", [_row("a", "science"), _row("b", "math")])
    long = _aggregate(
        tmp_path,
        "long",
        [_row("b", "math", "\\boxed{1} longer"), _row("c", "math")],
        max_new_tokens=3072,
    )
    output = tmp_path / "merged.jsonl"
    report_path = tmp_path / "merged.report.json"

    report = merge(
        [short, long],
        output,
        report_path,
        required_groups={"math": 2, "science": 1},
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["admitted"] is True
    assert report["positive_group_counts"] == {"math": 2, "science": 1}
    assert report["counters"]["duplicate_identity_drops"] == 1
    assert [row["source_identity_sha256"] for row in rows] == ["a", "b", "c"]
    assert rows[1]["response"] == "\\boxed{1}"


def test_merge_fails_closed_on_sparse_group(tmp_path: Path) -> None:
    source = _aggregate(tmp_path, "source", [_row("a", "science")])
    report = merge(
        [source],
        tmp_path / "merged.jsonl",
        tmp_path / "merged.report.json",
        required_groups={"math": 1, "science": 1},
    )
    assert report["admitted"] is False
    assert report["admission_failures"] == ["positive_math_below_minimum"]


def test_merge_rejects_model_contract_mismatch(tmp_path: Path) -> None:
    first = _aggregate(tmp_path, "first", [_row("a", "science")])
    second = _aggregate(
        tmp_path,
        "second",
        [_row("b", "math")],
        adapter_checkpoint="/other-checkpoint",
    )
    with pytest.raises(ProductRolloutMergeError, match="sampling contract differs"):
        merge(
            [first, second],
            tmp_path / "merged.jsonl",
            tmp_path / "merged.report.json",
            required_groups={},
        )


def test_merge_rejects_inconsistent_duplicate_metadata(tmp_path: Path) -> None:
    first = _aggregate(tmp_path, "first", [_row("a", "science")])
    inconsistent = _row("a", "math")
    second = _aggregate(tmp_path, "second", [inconsistent], max_new_tokens=3072)
    with pytest.raises(ProductRolloutMergeError, match="metadata differs"):
        merge(
            [first, second],
            tmp_path / "merged.jsonl",
            tmp_path / "merged.report.json",
            required_groups={},
        )
