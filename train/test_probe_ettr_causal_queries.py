from __future__ import annotations

import torch

from endogenous_typed_theory_reactor import ReactorTrace, TypedTheoryState
from ettr_objectives import (
    ETTRCausalQueryPair,
    ETTRPacketTargets,
    ETTRTransactionPredictions,
    ETTRTransactionTargets,
)
from probe_ettr_causal_queries import (
    _depth_bucket,
    _packet_geometry_row,
    _packet_geometry_summary,
    _pair_rows,
    _quantile,
    _state_pair_rows,
    _state_summary,
    _summary,
    _trace_pair_rows,
    _trace_summary,
    _transaction_geometry_row,
    _transaction_geometry_summary,
)
from probe_joint_ettr_causal_queries import _causal_shift


def test_pair_rows_measure_exact_difference_in_differences() -> None:
    pair = ETTRCausalQueryPair(
        correct_logits=torch.tensor(
            [[3.0, 0.0, -1.0], [0.0, 2.0, -1.0]]
        ),
        foil_logits=torch.tensor(
            [[0.0, 2.0, -1.0], [0.0, 2.0, -1.0]]
        ),
        correct_target=torch.tensor([0, 1]),
        foil_target=torch.tensor([1, 1]),
    )
    rows = _pair_rows(pair)
    assert rows[0]["contrast"] is True
    assert rows[0]["correct_delta"] == 3.0
    assert rows[0]["foil_delta"] == 2.0
    assert rows[0]["difference_in_differences"] == 5.0
    assert rows[0]["correct_top1"] is True
    assert rows[0]["foil_top1"] is True
    assert rows[1]["contrast"] is False


def test_summary_stratifies_effect_rows_by_depth() -> None:
    rows = [
        {
            **row,
            "depth_bucket": bucket,
        }
        for row, bucket in zip(
            _pair_rows(
                ETTRCausalQueryPair(
                    correct_logits=torch.tensor(
                        [[3.0, 0.0], [0.5, 0.0]]
                    ),
                    foil_logits=torch.tensor(
                        [[0.0, 2.0], [0.0, 0.25]]
                    ),
                    correct_target=torch.tensor([0, 0]),
                    foil_target=torch.tensor([1, 1]),
                )
            ),
            ("1", "3-4"),
            strict=True,
        )
    ]
    summary = _summary(rows)
    assert summary["count"] == 2
    assert summary["margin_rates"]["1"] == 0.5
    assert summary["by_depth"]["1"]["margin_rates"]["1"] == 1.0
    assert summary["by_depth"]["3-4"]["margin_rates"]["1"] == 0.0


def test_quantile_and_depth_buckets_are_deterministic() -> None:
    assert _quantile([1.0, 3.0], 0.5) == 2.0
    assert [_depth_bucket(value) for value in (1, 2, 3, 4, 5, 9, 33)] == [
        "1",
        "2",
        "3-4",
        "3-4",
        "5-8",
        "9-16",
        "33-64",
    ]


def test_state_pair_rows_separate_structure_from_disposition() -> None:
    def state(active: torch.Tensor, committed: torch.Tensor) -> TypedTheoryState:
        batch, slots = active.shape
        return TypedTheoryState(
            value_probabilities=torch.zeros(batch, slots, 2),
            type_probabilities=torch.zeros(batch, slots, 2),
            relations=torch.zeros(batch, 1, slots, slots),
            active=active,
            root=torch.zeros(batch, slots),
            committed=committed,
            halted=torch.zeros(batch),
            step=1,
        )

    correct = state(
        torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        torch.tensor([1.0, 1.0]),
    )
    foil = state(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([0.0, 1.0]),
    )
    rows = _state_pair_rows(correct, foil, torch.tensor([0, 1]))
    assert rows[0]["structural_state_equal"] is True
    assert rows[0]["exact_state_equal"] is False
    assert rows[0]["correct_answer_disposition"] is True
    assert rows[0]["foil_answer_disposition"] is False
    assert rows[1]["structural_state_equal"] is False
    summary = _state_summary(rows)
    assert summary["structural_state_equal_rate"] == 0.5
    assert summary["exact_state_equal_rate"] == 0.0


def test_trace_pair_rows_separate_policy_from_applied_state() -> None:
    def trace(policy_delta: float, applied_delta: float) -> ReactorTrace:
        policy = torch.zeros(2, 1, 2)
        policy[1, 0, 0] = policy_delta
        applied = torch.zeros(2, 1, 2)
        applied[1, 0, 0] = applied_delta
        status = torch.zeros(2, 1)
        return ReactorTrace(
            opcode=policy,
            source=policy.clone(),
            target=policy.clone(),
            relation=policy.clone(),
            type_index=policy.clone(),
            value_code=policy.clone(),
            applied_opcode=applied,
            applied_source=applied.clone(),
            applied_target=applied.clone(),
            applied_relation=applied.clone(),
            applied_type_index=applied.clone(),
            applied_value_code=applied.clone(),
            active=applied.clone(),
            committed=status,
            halted=status.clone(),
        )

    correct = trace(1.0, 0.0)
    foil = trace(0.0, 0.0)
    rows = _trace_pair_rows(correct, foil, torch.tensor([0, 1]))
    assert rows[0]["policy_trace_equal"] is True
    assert rows[0]["applied_trace_equal"] is True
    assert rows[1]["policy_trace_equal"] is False
    assert rows[1]["applied_trace_equal"] is True
    summary = _trace_summary(rows)
    assert summary["policy_trace_equal_rate"] == 0.5
    assert summary["applied_trace_equal_rate"] == 1.0


def test_packet_geometry_exposes_sparse_positive_failures() -> None:
    active = torch.tensor([[True, True], [True, False]])
    targets = ETTRPacketTargets(
        value_code=torch.tensor([[1, 0], [0, 0]]),
        type_index=torch.tensor([[0, 1], [1, 0]]),
        relations=torch.tensor(
            [
                [[[False, True], [False, False]]],
                [[[False, False], [False, False]]],
            ]
        ),
        active=active,
        root=torch.tensor([[True, False], [True, False]]),
        committed=torch.tensor([False, False]),
        halted=torch.tensor([False, False]),
        slot_mask=torch.ones(2, 2, dtype=torch.bool),
        relation_mask=torch.ones(2, 1, 2, 2, dtype=torch.bool),
    )
    predicted_values = torch.tensor([[1, 1], [0, 0]])
    prediction = TypedTheoryState(
        value_probabilities=(
            torch.nn.functional.one_hot(predicted_values, 2).float()
            * active.unsqueeze(-1)
        ),
        type_probabilities=(
            torch.nn.functional.one_hot(targets.type_index, 2).float()
            * active.unsqueeze(-1)
        ),
        relations=torch.zeros(2, 1, 2, 2),
        active=active.float(),
        root=targets.root.float(),
        committed=targets.committed.float(),
        halted=targets.halted.float(),
        step=0,
    )
    summary = _packet_geometry_summary(
        [_packet_geometry_row(prediction, targets)]
    )
    assert summary["complete_packet_accuracy"] == 0.5
    assert summary["fields"]["value_code"]["top1_accuracy"] == 2 / 3
    assert summary["fields"]["relations"]["positive_accuracy"] == 0.0
    assert summary["fields"]["relations"]["positive_support"] == 1
    assert summary["fields"]["relations"]["negative_accuracy"] == 1.0
    assert summary["fields"]["relations"]["negative_support"] == 7


def test_transaction_geometry_localizes_premature_terminal_opcode() -> None:
    def categorical(
        labels: torch.Tensor,
        classes: int,
    ) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            labels,
            num_classes=classes,
        ).float()

    targets = ETTRTransactionTargets(
        opcode=torch.tensor([[1, 6], [1, 6]]),
        source=torch.tensor([[0, 0], [1, 0]]),
        target=torch.zeros(2, 2, dtype=torch.long),
        relation=torch.zeros(2, 2, dtype=torch.long),
        type_index=torch.zeros(2, 2, dtype=torch.long),
        value_code=torch.tensor([[2, 0], [3, 0]]),
        committed=torch.tensor([[False, True], [False, True]]),
        halted=torch.zeros(2, 2, dtype=torch.bool),
        step_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    predicted_opcode = torch.tensor([[6, 6], [1, 6]])
    prediction = ETTRTransactionPredictions(
        opcode=categorical(predicted_opcode, 9),
        source=categorical(targets.source, 2),
        target=categorical(targets.target, 2),
        relation=categorical(targets.relation, 2),
        type_index=categorical(targets.type_index, 2),
        value_code=categorical(targets.value_code, 4),
        active=torch.ones(2, 2, 2),
        committed=targets.committed.float(),
        halted=targets.halted.float(),
    )
    row = _transaction_geometry_row(prediction, targets)
    summary = _transaction_geometry_summary([row, row])
    assert summary["fields"]["opcode"]["top1_accuracy"] == 0.75
    assert summary["fields"]["source"]["top1_accuracy"] == 1.0
    assert summary["complete_transaction_accuracy"] == 0.75
    assert summary["premature_terminal_rate"] == 0.5
    assert summary["predicted_opcode_counts"] == [0, 2, 0, 0, 0, 0, 6, 0, 0]
    assert summary["target_opcode_counts"] == [0, 4, 0, 0, 0, 0, 4, 0, 0]


def test_joint_causal_shift_preserves_signed_metric_deltas() -> None:
    def arm(mean: float, margin: float) -> dict[str, object]:
        query = {
            "difference_in_differences": {
                name: mean
                for name in (
                    "maximum",
                    "mean",
                    "minimum",
                    "p05",
                    "p25",
                    "p50",
                    "p75",
                    "p95",
                )
            },
            "joint_top1_rate": margin,
            "margin_rates": {"0": margin, "1": margin},
            "paired_order_joint_rate": margin,
        }
        return {
            "command": {"query": query},
            "world": {"query": query},
        }

    shift = _causal_shift(arm(0.25, 0.125), arm(0.75, 0.5))
    assert (
        shift["command"]["difference_in_differences_delta"]["p50"]
        == 0.5
    )
    assert shift["world"]["margin_rate_delta"]["1"] == 0.375
