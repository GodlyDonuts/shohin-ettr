import hashlib
import json
from pathlib import Path

import pytest

from merge_product_lineage_rollouts import LineageMergeError, merge_lineages


def _write_lineage(
    root: Path,
    lineage: str,
    checkpoint_sha256: str,
    correct: tuple[bool, bool],
    *,
    bank: str = "data",
    identity_offset: int = 0,
) -> tuple[list[Path], list[Path]]:
    candidate = root / f"{lineage}_{bank}.jsonl"
    rows = [
        {
            "schema": "shohin-hf-product-reasoning-rollouts-v1",
            "identity_sha256": f"id-{identity_offset + index}",
            "question": f"question {identity_offset + index}",
            "task": "math500",
            "sample_index": 0,
            "completion": f"{lineage} completion {index}",
            "correct": value,
            "generated_tokens": 10,
        }
        for index, value in enumerate(correct)
    ]
    candidate.write_text("".join(json.dumps(row) + "\n" for row in rows))
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    report = root / f"{lineage}_{bank}.report.json"
    report.write_text(
        json.dumps(
            {
                "schema": "shohin-hf-product-reasoning-rollouts-v1",
                "status": "complete",
                "generation_mode": "greedy",
                "samples": 1,
                "candidates_sha256": candidate_sha256,
                "data_sha256": bank,
                "adapter_checkpoint": f"/{lineage}.pt",
                "adapter_checkpoint_sha256": checkpoint_sha256,
                "skip": 0,
                "count": 2,
            }
        )
    )
    return [candidate], [report]


def test_merge_preserves_whole_lineages_and_outcomes(tmp_path: Path) -> None:
    base_candidates, base_reports = _write_lineage(
        tmp_path, "base", "a" * 64, (True, False)
    )
    expert_candidates, expert_reports = _write_lineage(
        tmp_path, "expert", "b" * 64, (False, True)
    )
    rows, report = merge_lineages(
        base_candidates=base_candidates,
        base_reports=base_reports,
        expert_candidates=expert_candidates,
        expert_reports=expert_reports,
        split_seed=7,
    )
    assert report["outcome_counts"] == {"base_only": 1, "expert_only": 1}
    assert [candidate["lineage"] for candidate in rows[0]["candidates"]] == [
        "base",
        "expert",
    ]
    assert all(
        "gold" not in candidate for row in rows for candidate in row["candidates"]
    )


def test_merge_rejects_same_checkpoint_identity(tmp_path: Path) -> None:
    base_candidates, base_reports = _write_lineage(
        tmp_path, "base", "a" * 64, (True, False)
    )
    expert_candidates, expert_reports = _write_lineage(
        tmp_path, "expert", "a" * 64, (False, True)
    )
    with pytest.raises(LineageMergeError, match="checkpoint identities are equal"):
        merge_lineages(
            base_candidates=base_candidates,
            base_reports=base_reports,
            expert_candidates=expert_candidates,
            expert_reports=expert_reports,
            split_seed=7,
        )


def test_merge_accepts_multiple_independently_sharded_banks(tmp_path: Path) -> None:
    base_a, base_a_reports = _write_lineage(
        tmp_path, "base", "a" * 64, (True, False), bank="bank-a"
    )
    base_b, base_b_reports = _write_lineage(
        tmp_path,
        "base",
        "a" * 64,
        (False, True),
        bank="bank-b",
        identity_offset=2,
    )
    expert_a, expert_a_reports = _write_lineage(
        tmp_path, "expert", "b" * 64, (False, True), bank="bank-a"
    )
    expert_b, expert_b_reports = _write_lineage(
        tmp_path,
        "expert",
        "b" * 64,
        (True, False),
        bank="bank-b",
        identity_offset=2,
    )
    rows, report = merge_lineages(
        base_candidates=base_a + base_b,
        base_reports=base_a_reports + base_b_reports,
        expert_candidates=expert_a + expert_b,
        expert_reports=expert_a_reports + expert_b_reports,
        split_seed=7,
    )
    assert len(rows) == 4
    assert report["rows"] == 4
    assert {row["prompt_bank_sha256"] for row in rows} == {"bank-a", "bank-b"}
