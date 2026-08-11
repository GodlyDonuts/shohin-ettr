"""Focused CPU tests for VTE1's set-valued executable objective."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hf_product_reasoning_train import (
    _tokenize_vte1_rows,
    ProductReasoningTrainError,
    kcr1_action_payload_loss,
    vte1_equivalence_loss,
    vte1_rows_with_sha256,
)
from test_kcr1_weighted_loss import OffsetTokenizer


def test_single_candidate_matches_kcr1_loss() -> None:
    logits = torch.randn(1, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3, 4]])
    expected, _ = kcr1_action_payload_loss(logits, labels, [2])
    actual, metrics = vte1_equivalence_loss(logits, labels, [2], [1])
    assert torch.allclose(actual, expected)
    assert metrics["equivalence_candidates"] == 1.0
    assert metrics["equivalence_groups"] == 1.0


def test_soft_min_prefers_lower_verified_candidate_and_backpropagates() -> None:
    logits = torch.zeros(2, 4, 3, requires_grad=True)
    labels = torch.tensor([[-100, 0, 0, 0], [-100, 1, 1, 1]])
    with torch.no_grad():
        logits[0, :3, 0] = 5.0
    loss, _ = vte1_equivalence_loss(logits, labels, [1, 1], [2])
    row_zero, _ = kcr1_action_payload_loss(logits[:1], labels[:1], [1])
    row_one, _ = kcr1_action_payload_loss(logits[1:], labels[1:], [1])
    assert row_zero < loss < row_one
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_loader_and_tokenizer_preserve_candidate_groups(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {
            "question": "DRAFT:\nfirst",
            "responses": ["<KEEP>", "<RESTART>\nanswer"],
            "candidate_count": 2,
        },
        {
            "question": "DRAFT:\nsecond",
            "responses": ["<CONTINUE>\nsuffix"],
            "candidate_count": 1,
        },
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    selected, _ = vte1_rows_with_sha256(path, limit=10, seed=7)
    prompts, responses, actions, attention, groups = _tokenize_vte1_rows(
        OffsetTokenizer(), selected, 4096, 0
    )
    assert sorted(groups) == [1, 2]
    assert len(prompts) == len(responses) == len(actions) == sum(groups)
    assert attention is None


@pytest.mark.parametrize(
    "row",
    [
        {"question": "q", "responses": []},
        {"question": "q", "responses": ["not-a-transaction"]},
        {
            "question": "q",
            "responses": ["<KEEP>", "<KEEP>"],
            "candidate_count": 2,
        },
    ],
)
def test_invalid_equivalence_sets_fail_closed(tmp_path: Path, row: dict) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ProductReasoningTrainError):
        vte1_rows_with_sha256(path, limit=10, seed=7)


def test_group_cardinality_fails_closed() -> None:
    logits = torch.zeros(1, 3, 4)
    labels = torch.tensor([[-100, 1, 2]])
    with pytest.raises(ProductReasoningTrainError):
        vte1_equivalence_loss(logits, labels, [1], [2])
