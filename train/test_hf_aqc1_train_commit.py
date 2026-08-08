#!/usr/bin/env python3
"""AQC1 antisymmetry and selection accounting tests."""

from __future__ import annotations

import hashlib
import json

import torch

from hf_aqc1_train_commit import (
    AntisymmetricCommitHead,
    IndependentCommitHead,
    load_pairs,
    metrics,
)


def test_heads_are_exactly_order_antisymmetric() -> None:
    torch.manual_seed(7)
    left = torch.randn(5, 16)
    right = torch.randn(5, 16)
    for head in (
        IndependentCommitHead(16, 8),
        AntisymmetricCommitHead(16, 8, 4),
    ):
        direct = head.margin(left, right)
        reverse = head.margin(right, left)
        assert torch.equal(direct, -reverse)


def test_pair_loading_and_metrics(tmp_path) -> None:
    rows = []
    selections = {}
    outcomes = (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    )
    names = ("both_correct", "idr1_only", "control_only", "both_wrong")
    for split_index, split in enumerate(("train", "development", "holdout")):
        for index, ((left, right), outcome) in enumerate(zip(outcomes, names)):
            identity = hashlib.sha256(f"{split}:{index}".encode()).hexdigest()
            rows.append(
                {
                    "schema": "shohin-aqc1-whole-trajectory-pair-v1",
                    "identity_sha256": identity,
                    "split": split,
                    "source_split": "development" if split != "holdout" else "holdout",
                    "task": ("math500", "bbh_logic", "mbpp")[(index + split_index) % 3],
                    "question": "question",
                    "outcome_class": outcome,
                    "candidates": [
                        {"lineage": "idr1", "completion": "left", "correct": left},
                        {"lineage": "control", "completion": "right", "correct": right},
                    ],
                }
            )
            if split == "holdout":
                selections[identity] = (1 if right and not left else 0, True)
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    loaded = load_pairs(path)
    result = metrics(loaded, selections, "holdout")
    assert result["overall"]["selected_correct"] == 3
    assert result["overall"]["oracle_correct"] == 3
    assert result["overall"]["order_consistency"] == 1.0
