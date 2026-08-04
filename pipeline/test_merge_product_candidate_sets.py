"""Tests for merging independent product-candidate draws."""

from __future__ import annotations

import pytest

from merge_product_candidate_sets import (
    ProductCandidateMergeError,
    merge_candidate_sets,
)


def _row(identity: str, sample: int, *, question: str = "q") -> dict[str, object]:
    return {
        "identity_sha256": identity,
        "sample_index": sample,
        "task": "math500",
        "question": question,
        "gold": "1",
        "training_group": "heldout",
        "completion": f"answer-{sample}",
    }


def test_merge_reindexes_supplement_per_identity() -> None:
    base = [_row("a", 0), _row("a", 1), _row("b", 0), _row("b", 1)]
    supplement = [_row("a", 0), _row("a", 1), _row("b", 0), _row("b", 1)]
    merged, report = merge_candidate_sets(base, supplement)
    assert [row["sample_index"] for row in merged if row["identity_sha256"] == "a"] == [
        0,
        1,
        2,
        3,
    ]
    assert report["merged_samples_per_identity"] == 4
    assert report["rows"] == 8


def test_merge_rejects_identity_mismatch() -> None:
    with pytest.raises(ProductCandidateMergeError, match="identity sets differ"):
        merge_candidate_sets([_row("a", 0)], [_row("b", 0)])


def test_merge_rejects_prompt_mismatch() -> None:
    with pytest.raises(ProductCandidateMergeError, match="question differs"):
        merge_candidate_sets([_row("a", 0)], [_row("a", 0, question="different")])


def test_merge_rejects_non_contiguous_samples() -> None:
    with pytest.raises(ProductCandidateMergeError, match="sample indices differ"):
        merge_candidate_sets([_row("a", 1)], [_row("a", 0)])
