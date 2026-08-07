#!/usr/bin/env python3
"""Focused mechanics tests for DIVERGE-JRB1."""

from __future__ import annotations

import torch

from diverge_jrb1_data import (
    DEVELOPMENT_SEED,
    augment_evaluation_episode,
    build_training_record,
    render_initial_state,
    render_query,
)
from diverge_jrb1_runtime import (
    JointRegisterBinder,
    tensorize_register_sources,
    tensorize_temporal_without_register_scan,
)
from eval_diverge_jrb1 import (
    _compile_packets,
    _decode_initial_states,
    _execution_score,
)


def main() -> None:
    first = build_training_record(17)
    assert first == build_training_record(17)
    assert first != build_training_record(18)
    assert len(first["evidence_register_targets"]) == 4
    assert sorted(first["initial_register_targets"]) == [0, 1]

    initial_train = {
        render_initial_state(
            ("rax", "reb"), (3, 9), split="train", serial=index, order=(0, 1)
        )[2][0]
        for index in range(32)
    }
    initial_dev = {
        render_initial_state(
            ("rax", "reb"),
            (3, 9),
            split="development",
            serial=index,
            order=(0, 1),
        )[2][0]
        for index in range(32)
    }
    assert initial_train.isdisjoint(initial_dev)
    query_train = {
        render_query(("rax", "reb"), 0, split="train", serial=index)[1]
        for index in range(32)
    }
    query_dev = {
        render_query(("rax", "reb"), 0, split="development", serial=index)[1]
        for index in range(32)
    }
    assert query_train.isdisjoint(query_dev)

    public, assessor = augment_evaluation_episode(3, seed=DEVELOPMENT_SEED)
    assert "initial_state" not in public["transfer"][0]
    assert "symbols" not in public["transfer"][0]
    assert "register_index" not in public["queries"][0]
    assert len(assessor["initial_targets"]) == len(public["transfer"])
    assert len(assessor["query_targets"]) == len(public["queries"])

    model = JointRegisterBinder()
    records = [
        {
            "source": "Initially, rax was 3; meanwhile, reb held 9.",
            "registers": ["rax", "reb"],
        },
        {
            "source": "Initially, reb was 8; meanwhile, rax held 4.",
            "registers": ["rax", "reb"],
        },
    ]
    tensors = tensorize_register_sources(
        records,
        torch.device("cpu"),
        text_key="source",
        mention_count=2,
    )
    mention_logits = model.forward_mentions(*tensors[:4], tensors[4], tensors[5])
    assert mention_logits.shape == (2, 2, 2)
    query_tensors = tensorize_register_sources(
        [
            {"source": "At the end, report the value in rax.", "registers": ["rax", "reb"]},
            {"source": "At the end, report the value in reb.", "registers": ["rax", "reb"]},
        ],
        torch.device("cpu"),
        text_key="source",
        mention_count=None,
    )
    query_logits = model.forward_query(*query_tensors[:4])
    assert query_logits.shape == (2, 2)
    swapped = tensorize_register_sources(
        records,
        torch.device("cpu"),
        text_key="source",
        mention_count=2,
        rotate_register_table=True,
    )
    swapped_logits = model.forward_mentions(
        *swapped[:4], swapped[4], swapped[5]
    )
    torch.testing.assert_close(mention_logits[..., 0], swapped_logits[..., 1])
    torch.testing.assert_close(mention_logits[..., 1], swapped_logits[..., 0])
    swapped_queries = tensorize_register_sources(
        [
            {"source": "At the end, report the value in rax.", "registers": ["rax", "reb"]},
            {"source": "At the end, report the value in reb.", "registers": ["rax", "reb"]},
        ],
        torch.device("cpu"),
        text_key="source",
        mention_count=None,
        rotate_register_table=True,
    )
    swapped_query_logits = model.forward_query(*swapped_queries[:4])
    torch.testing.assert_close(query_logits[:, 0], swapped_query_logits[:, 1])
    torch.testing.assert_close(query_logits[:, 1], swapped_query_logits[:, 0])

    temporal = tensorize_temporal_without_register_scan(
        [{"source": public["evidence"][0]["source_text"]}],
        torch.device("cpu"),
        text_key="source",
    )
    assert temporal[0].shape == (1, 320)
    assert temporal[2].shape == (1, 4, 2)

    model.train()
    loss = torch.nn.functional.cross_entropy(
        mention_logits.flatten(0, 1), torch.tensor([0, 1, 1, 0])
    ) + torch.nn.functional.cross_entropy(query_logits, torch.tensor([0, 1]))
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    board = [augment_evaluation_episode(index, seed=DEVELOPMENT_SEED) for index in range(2)]
    visible = [value[0] for value in board]
    hidden = [value[1] for value in board]
    roles = [
        tuple(item["numeric_role_ids"])
        for episode in hidden
        for item in episode["evidence"]
    ]
    packets = _compile_packets(
        visible,
        roles,
        text_key="source_text",
        hash_key="source_sha256",
        owner_state_sha256="0" * 64,
    )
    initial_mentions = [
        tuple(item["mention_register_targets"])
        for episode in hidden
        for item in episode["initial_targets"]
    ]
    states = _decode_initial_states(
        visible, initial_mentions, text_key="initial_text"
    )
    programs = [
        tuple(item["targets"])
        for episode in hidden
        for item in episode["command_targets"]
    ]
    queries = [
        int(item["register_index"])
        for episode in hidden
        for item in episode["query_targets"]
    ]
    execution = _execution_score(packets, visible, hidden, programs, states, queries)
    assert execution["state_exact_rate"] == 1.0
    assert execution["answer_exact_rate"] == 1.0
    print("diverge JRB1 mechanics tests passed")


if __name__ == "__main__":
    main()
