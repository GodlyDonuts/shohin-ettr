from __future__ import annotations

import torch

from endogenous_typed_theory_reactor import TheoryReactorConfig
from ettr_objectives import ETTRPacketTargets, ETTRTransactionTargets
from probe_ettr_oracle_interfaces import (
    _count_summary,
    packet_targets_to_state,
    policy_masks,
    target_policy,
)


def _packet_targets() -> ETTRPacketTargets:
    return ETTRPacketTargets(
        value_code=torch.tensor([[2, 0]]),
        type_index=torch.tensor([[1, 0]]),
        relations=torch.tensor([[[[False, True], [False, False]]]]),
        active=torch.tensor([[True, True]]),
        root=torch.tensor([[True, False]]),
        committed=torch.tensor([True]),
        halted=torch.tensor([False]),
        slot_mask=torch.tensor([[True, True]]),
        relation_mask=torch.ones(1, 1, 2, 2, dtype=torch.bool),
    )


def _transaction_targets() -> ETTRTransactionTargets:
    return ETTRTransactionTargets(
        opcode=torch.tensor([[0, 3]]),
        source=torch.tensor([[1, 0]]),
        target=torch.tensor([[0, 1]]),
        relation=torch.tensor([[0, 0]]),
        type_index=torch.tensor([[1, 0]]),
        value_code=torch.tensor([[2, 0]]),
        committed=torch.tensor([[False, False]]),
        halted=torch.tensor([[False, False]]),
        step_mask=torch.tensor([[True, True]]),
    )


def test_packet_target_state_is_exact_and_masks_inactive_values() -> None:
    config = TheoryReactorConfig(
        d_model=8,
        state_width=8,
        num_slots=2,
        num_types=2,
        num_relations=1,
        num_value_codes=3,
        max_edges=2,
        num_heads=2,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        max_steps=4,
        stage_after_block=0,
    )
    state = packet_targets_to_state(
        _packet_targets(),
        config,
        step=2,
        dtype=torch.float32,
    )
    assert state.step == 2
    assert state.value_probabilities.argmax(-1).tolist() == [[2, 0]]
    assert state.type_probabilities.argmax(-1).tolist() == [[1, 0]]
    assert state.relations.bool().tolist() == [
        [[[False, True], [False, False]]]
    ]
    assert state.committed.tolist() == [1.0]


def test_target_policy_and_operand_masks_match_opcode_semantics() -> None:
    config = TheoryReactorConfig(
        d_model=8,
        state_width=8,
        num_slots=2,
        num_types=2,
        num_relations=1,
        num_value_codes=3,
        max_edges=2,
        num_heads=2,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        max_steps=4,
        stage_after_block=0,
    )
    targets = _transaction_targets()
    policy = target_policy(targets, config, 1, dtype=torch.float32)
    assert policy.opcode.argmax(-1).tolist() == [3]
    assert policy.target.argmax(-1).tolist() == [1]
    masks = policy_masks(targets)
    assert masks["source"].tolist() == [[True, True]]
    assert masks["target"].tolist() == [[False, True]]
    assert masks["type_index"].tolist() == [[True, False]]
    assert masks["value_code"].tolist() == [[True, False]]


def test_count_summary_handles_zero_support() -> None:
    assert _count_summary({"a": (1, 2), "b": (0, 0)}) == {
        "a": {"accuracy": 0.5, "correct": 1, "total": 2},
        "b": {"accuracy": None, "correct": 0, "total": 0},
    }
