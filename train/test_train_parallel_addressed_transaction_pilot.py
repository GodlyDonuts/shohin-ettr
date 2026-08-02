from __future__ import annotations

from types import SimpleNamespace

import torch

from endogenous_typed_theory_reactor import (
    GenericTransactionReactor,
    TheoryReactorConfig,
    TypedTheoryState,
)
from parallel_addressed_transaction_compiler import AddressedSchedule
from train_parallel_addressed_transaction_pilot import (
    _balanced_binary_brier,
    _balanced_categorical_loss,
    _program_statistics,
    _schedule_counts,
    _schedule_loss,
    _semantic_prefix_loss,
    _state_brier,
    _training_initial_state,
)


def _schedule() -> SimpleNamespace:
    values = {
        "opcode": torch.tensor(
            [
                [
                    [0.9, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02],
                    [0.01, 0.01, 0.01, 0.9, 0.02, 0.01, 0.01, 0.01, 0.02],
                ]
            ]
        ),
        "source": torch.tensor([[[0.7, 0.3], [0.4, 0.6]]]),
        "target": torch.tensor([[[0.6, 0.4], [0.3, 0.7]]]),
        "relation": torch.tensor([[[0.8, 0.2], [0.1, 0.9]]]),
        "type_index": torch.tensor([[[0.9, 0.1], [0.2, 0.8]]]),
        "value_code": torch.tensor([[[0.9, 0.1], [0.2, 0.8]]]),
    }
    return SimpleNamespace(
        **values,
        **{
            f"applied_{name}": torch.nn.functional.one_hot(
                value.argmax(-1),
                value.shape[-1],
            ).float()
            for name, value in values.items()
        },
    )


def _targets() -> SimpleNamespace:
    return SimpleNamespace(
        opcode=torch.tensor([[0, 3]]),
        source=torch.tensor([[0, 1]]),
        target=torch.tensor([[0, 1]]),
        relation=torch.tensor([[0, 1]]),
        type_index=torch.tensor([[0, 1]]),
        value_code=torch.tensor([[0, 1]]),
        step_mask=torch.tensor([[True, True]]),
    )


def test_balanced_categorical_loss_balances_observed_classes() -> None:
    probabilities = torch.tensor(
        [[[0.8, 0.2], [0.9, 0.1], [0.4, 0.6]]],
        requires_grad=True,
    )
    targets = torch.tensor([[0, 0, 1]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    loss = _balanced_categorical_loss(probabilities, targets, mask)
    assert loss is not None
    expected = 0.5 * (-torch.tensor([0.8, 0.9]).log().mean() - torch.tensor(0.6).log())
    assert torch.allclose(loss, expected)
    loss.backward()
    assert bool(torch.isfinite(probabilities.grad).all())


def test_schedule_loss_and_counts_cover_all_heads() -> None:
    schedule = _schedule()
    targets = _targets()
    loss, parts = _schedule_loss(schedule, targets)
    assert bool(torch.isfinite(loss))
    assert set(parts) == {
        "opcode",
        "source",
        "target",
        "relation",
        "type_index",
        "value_code",
    }
    counts = _schedule_counts(schedule, targets)
    assert counts["opcode"] == (2, 2)
    assert counts["joint"] == (2, 2)


def test_schedule_loss_supervises_one_complete_opcode_program() -> None:
    schedule = _schedule()
    schedule.program_probabilities = torch.tensor(
        [[0.8, 0.2]],
        requires_grad=True,
    )
    compiler = SimpleNamespace(
        opcode_program_table=torch.tensor([[0, 3], [1, 6]]),
        opcode_program_step_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    loss, parts = _schedule_loss(schedule, _targets(), compiler)
    assert "program" in parts
    loss.backward()
    assert schedule.program_probabilities.grad is not None
    assert float(schedule.program_probabilities.grad[0, 0]) < 0.0


def test_program_statistics_separates_known_coverage_accuracy_and_entropy() -> None:
    schedule = _schedule()
    schedule.program_probabilities = torch.tensor([[0.8, 0.2]])
    compiler = SimpleNamespace(
        opcode_program_table=torch.tensor([[0, 3], [1, 6]]),
        opcode_program_step_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    statistics = _program_statistics(schedule, _targets(), compiler)
    assert statistics is not None
    assert statistics["correct"] == 1
    assert statistics["known"] == 1
    assert statistics["rows"] == 1
    assert statistics["selected_counts"].tolist() == [1, 0]
    assert 0.0 < statistics["probability_entropy_sum"] < 1.0


def test_program_statistics_reports_unknown_target_without_false_match() -> None:
    schedule = _schedule()
    schedule.program_probabilities = torch.tensor([[0.8, 0.2]])
    compiler = SimpleNamespace(
        opcode_program_table=torch.tensor([[0, 3], [1, 6]]),
        opcode_program_step_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    targets = _targets()
    targets.opcode = torch.tensor([[3, 6]])
    statistics = _program_statistics(schedule, targets, compiler)
    assert statistics is not None
    assert statistics["correct"] == 0
    assert statistics["known"] == 0
    assert statistics["rows"] == 1


def test_balanced_binary_brier_does_not_reward_sparse_zero_collapse() -> None:
    predicted = torch.tensor([[0.2, 0.2, 0.2, 0.2]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = _balanced_binary_brier(predicted, target, mask)
    assert torch.allclose(loss, torch.tensor(0.34))
    loss.backward()
    assert predicted.grad is not None
    assert predicted.grad[0, 0].abs() > predicted.grad[0, 1].abs()


def test_state_brier_carries_gradients_from_every_semantic_field() -> None:
    predicted = SimpleNamespace(
        active=torch.tensor([[0.8, 0.2]], requires_grad=True),
        committed=torch.tensor([0.2], requires_grad=True),
        halted=torch.tensor([0.2], requires_grad=True),
        relations=torch.tensor([[[[0.1, 0.7], [0.2, 0.1]]]], requires_grad=True),
        root=torch.tensor([[0.7, 0.3]], requires_grad=True),
        type_probabilities=torch.tensor([[[0.8, 0.2], [0.4, 0.6]]], requires_grad=True),
        value_probabilities=torch.tensor(
            [[[0.6, 0.4], [0.3, 0.7]]], requires_grad=True
        ),
    )
    target = SimpleNamespace(
        active=torch.tensor([[1.0, 0.0]]),
        committed=torch.tensor([1.0]),
        halted=torch.tensor([0.0]),
        relations=torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
        root=torch.tensor([[1.0, 0.0]]),
        type_probabilities=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
        value_probabilities=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
    )
    loss, parts = _state_brier(
        predicted,
        target,
        slot_mask=torch.ones(1, 2, dtype=torch.bool),
        relation_mask=torch.ones(1, 1, 2, 2, dtype=torch.bool),
    )
    assert set(parts) == {
        "active",
        "committed",
        "halted",
        "relations",
        "root",
        "type_index",
        "value_code",
    }
    assert bool(torch.isfinite(loss))
    loss.backward()
    for value in vars(predicted).values():
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())


def test_semantic_prefix_loss_backpropagates_through_exact_execution() -> None:
    config = TheoryReactorConfig(
        d_model=16,
        state_width=16,
        num_slots=4,
        num_types=3,
        num_relations=2,
        num_value_codes=7,
        max_edges=8,
        num_heads=2,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        ff_multiplier=2,
        max_steps=3,
        stage_after_block=0,
    )
    active = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    initial = TypedTheoryState(
        value_probabilities=torch.nn.functional.one_hot(
            torch.tensor([[1, 0, 0, 0]]), config.num_value_codes
        ).float()
        * active.unsqueeze(-1),
        type_probabilities=torch.nn.functional.one_hot(
            torch.tensor([[0, 0, 0, 0]]), config.num_types
        ).float()
        * active.unsqueeze(-1),
        relations=torch.zeros(1, 2, 4, 4),
        active=active,
        root=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        committed=torch.zeros(1),
        halted=torch.zeros(1),
        step=0,
    )
    opcode_logits = torch.zeros(1, 1, 9, requires_grad=True)
    source_logits = torch.zeros(1, 1, 4, requires_grad=True)
    type_logits = torch.zeros(1, 1, 3, requires_grad=True)
    value_logits = torch.zeros(1, 1, 7, requires_grad=True)
    values = {
        "opcode": opcode_logits.softmax(-1),
        "source": source_logits.softmax(-1),
        "target": torch.full((1, 1, 4), 0.25),
        "relation": torch.full((1, 1, 2), 0.5),
        "type_index": type_logits.softmax(-1),
        "value_code": value_logits.softmax(-1),
    }
    schedule = AddressedSchedule(
        **values,
        **{f"applied_{name}": value for name, value in values.items()},
    )
    targets = SimpleNamespace(
        opcode=torch.tensor([[0]]),
        source=torch.tensor([[1]]),
        target=torch.tensor([[0]]),
        relation=torch.tensor([[0]]),
        type_index=torch.tensor([[2]]),
        value_code=torch.tensor([[5]]),
        step_mask=torch.tensor([[True]]),
    )
    packet_targets = SimpleNamespace(
        slot_mask=torch.ones(1, 4, dtype=torch.bool),
        relation_mask=torch.ones(1, 2, 4, 4, dtype=torch.bool),
    )
    loss, parts = _semantic_prefix_loss(
        schedule,
        GenericTransactionReactor(config),
        initial,
        targets,
        packet_targets,
    )
    assert set(parts) == {
        "active",
        "committed",
        "halted",
        "relations",
        "root",
        "type_index",
        "value_code",
    }
    loss.backward()
    for value in (opcode_logits, source_logits, type_logits, value_logits):
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad).all())


def test_autonomous_training_uses_the_deployed_world_state() -> None:
    config = TheoryReactorConfig(
        d_model=16,
        state_width=16,
        num_slots=4,
        num_types=3,
        num_relations=2,
        num_value_codes=7,
        max_edges=8,
        num_heads=2,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        ff_multiplier=2,
        max_steps=3,
        stage_after_block=0,
    )
    expected = TypedTheoryState(
        value_probabilities=torch.zeros(1, 4, 7),
        type_probabilities=torch.zeros(1, 4, 3),
        relations=torch.zeros(1, 2, 4, 4),
        active=torch.zeros(1, 4),
        root=torch.zeros(1, 4),
        committed=torch.zeros(1),
        halted=torch.zeros(1),
        step=0,
    )

    class FakeModel:
        def __init__(self) -> None:
            self.config = config
            self.calls = []

        def compile_world(self, tokens, *, attention_mask, hard):
            self.calls.append((tokens, attention_mask, hard))
            return expected

    tokens = torch.tensor([[1, 2]])
    attention_mask = torch.tensor([[1, 1]])
    batch = SimpleNamespace(
        episodes=SimpleNamespace(
            world=SimpleNamespace(
                tokens=tokens,
                attention_mask=attention_mask,
            )
        )
    )
    model = FakeModel()
    observed = _training_initial_state(
        model,
        batch,
        source="autonomous",
        dtype=torch.bfloat16,
    )
    assert model.calls == [(tokens, attention_mask, True)]
    assert observed is not expected
    for field in (
        "value_probabilities",
        "type_probabilities",
        "relations",
        "active",
        "root",
        "committed",
        "halted",
    ):
        assert torch.equal(getattr(observed, field), getattr(expected, field))
