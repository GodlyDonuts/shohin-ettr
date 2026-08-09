from __future__ import annotations

from pathlib import Path


SOURCE = (Path(__file__).resolve().parent / "merge_sctr1_evaluation_shards.py").read_text(
    encoding="utf-8"
)


def test_merger_checks_exact_shard_identity_and_commit_accounting() -> None:
    assert "shard_bounds" in SOURCE
    assert "expected_ids" in SOURCE
    assert "results_by_identity" in SOURCE
    assert '"commitment"' in SOURCE
    assert "merged_from_shards" in SOURCE
    assert '"mask_internal_draft"' in SOURCE
    assert '"masked_draft_tokens"' in SOURCE
