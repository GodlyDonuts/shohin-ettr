"""Focused CPU tests for KCR1's causal action/payload objective."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from hf_product_reasoning_train import (
    ProductReasoningTrainError,
    kcr1_action_payload_loss,
    kcr1_rows_with_sha256,
)


def test_action_and_payload_receive_equal_presentation_mass() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 0, 1, 2], [-100, 0, 1, 2]])
    loss, metrics = kcr1_action_payload_loss(logits, labels, [1, 3])

    first = torch.nn.functional.cross_entropy(logits[0, :3], labels[0, 1:], reduction="none")
    second = torch.nn.functional.cross_entropy(logits[1, :3], labels[1, 1:], reduction="none")
    expected = (0.5 * first[0] + 0.5 * first[1:].mean() + second.mean()) / 2
    assert torch.allclose(loss, expected)
    assert metrics["action_tokens"] == 4.0
    assert metrics["payload_tokens"] == 2.0
    assert math.isfinite(metrics["action_loss"])
    assert math.isfinite(metrics["payload_loss"])
    loss.backward()
    assert logits.grad is not None


def test_action_boundary_fails_closed() -> None:
    logits = torch.zeros(1, 3, 4)
    labels = torch.tensor([[-100, 1, 2]])
    with pytest.raises(ProductReasoningTrainError):
        kcr1_action_payload_loss(logits, labels, [3])


def test_kcr1_loader_preserves_action_field(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {"question": "q1", "response": "<KEEP>", "action": "<KEEP>"},
        {
            "question": "q2",
            "response": "<CONTINUE>\nanswer",
            "action": "<CONTINUE>",
        },
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    selected, _ = kcr1_rows_with_sha256(path, limit=10, seed=7)
    assert {row["action"] for row in selected} == {"<KEEP>", "<CONTINUE>"}


def test_kcr1_loader_rejects_mismatched_prefix(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {"question": "q", "response": "<RESTART>\nanswer", "action": "<KEEP>"}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProductReasoningTrainError):
        kcr1_rows_with_sha256(path, limit=10, seed=7)
