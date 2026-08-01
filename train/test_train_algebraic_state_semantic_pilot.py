from __future__ import annotations

import torch

from train_algebraic_state_semantic_pilot import (
    _balanced_probability_loss,
    _semantic_states,
    _set_active_owner,
    _set_training_ownership,
)


class _StateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(3, 3)
        self.compiler = torch.nn.Linear(3, 3)
        self.reactor = torch.nn.Linear(3, 3)


def test_causal_owner_switches_exclude_cross_factor_parameters() -> None:
    model = _StateModel()
    reader = torch.nn.Linear(3, 2)
    trainable, count = _set_training_ownership(model, reader)
    assert count == sum(
        parameter.numel()
        for module in (model.compiler, model.reactor)
        for parameter in module.parameters()
    )
    assert set(trainable) == {
        parameter
        for module in (model.compiler, model.reactor)
        for parameter in module.parameters()
    }
    assert not any(parameter.requires_grad for parameter in model.base.parameters())
    assert not any(parameter.requires_grad for parameter in reader.parameters())

    _set_active_owner(model, "compiler")
    assert all(parameter.requires_grad for parameter in model.compiler.parameters())
    assert not any(parameter.requires_grad for parameter in model.reactor.parameters())

    _set_active_owner(model, "reactor")
    assert not any(parameter.requires_grad for parameter in model.compiler.parameters())
    assert all(parameter.requires_grad for parameter in model.reactor.parameters())

    _set_active_owner(model, None)
    assert all(parameter.requires_grad for parameter in model.compiler.parameters())
    assert all(parameter.requires_grad for parameter in model.reactor.parameters())


class _FakeState:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.active = torch.ones(1)


class _FakeReactor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.applied: list[tuple[int, bool]] = []

    def apply(self, state, policy, *, hard: bool, validate: bool):
        assert not validate
        self.applied.append((policy, hard))
        return _FakeState(f"{state.tag}-step{policy}")


class _FakeSemanticModel:
    def __init__(self) -> None:
        self.config = object()
        self.reactor = _FakeReactor()
        self.compiles = 0
        self.executes: list[str] = []
        self.hard_compiles: list[bool] = []
        self.hard_executes: list[bool] = []

    def compile_world(self, _tokens, *, attention_mask, hard: bool):
        assert attention_mask is not None
        self.compiles += 1
        self.hard_compiles.append(hard)
        return _FakeState("autonomous-initial")

    def execute(self, state, **kwargs):
        self.executes.append(state.tag)
        self.hard_executes.append(kwargs["hard"])
        return _FakeState(f"{state.tag}-autonomous-terminal"), None


class _Batch:
    def __init__(self) -> None:
        self.packet_targets = object()
        self.transaction_targets = type(
            "Transactions",
            (),
            {"opcode": torch.zeros(1, 3, dtype=torch.long)},
        )()
        stage = type(
            "Stage",
            (),
            {
                "tokens": torch.zeros(1, 2, dtype=torch.long),
                "attention_mask": torch.ones(1, 2, dtype=torch.bool),
            },
        )()
        self.episodes = type(
            "Episodes",
            (),
            {"world": stage, "command": stage},
        )()


def test_oracle_factor_bridge_isolates_each_causal_owner(monkeypatch) -> None:
    monkeypatch.setattr(
        "train_algebraic_state_semantic_pilot.packet_targets_to_state",
        lambda *_args, **_kwargs: _FakeState("oracle-initial"),
    )
    monkeypatch.setattr(
        "train_algebraic_state_semantic_pilot.target_policy",
        lambda _targets, _config, step, **_kwargs: step,
    )
    batch = _Batch()

    world_model = _FakeSemanticModel()
    world_initial, world_terminal = _semantic_states(
        world_model,
        batch,
        factor="world",
        owner_state_bridge="oracle-factors",
        semantic_state_mode="hard-st",
    )
    assert world_initial.tag == "autonomous-initial"
    assert world_terminal.tag == "autonomous-initial-step0-step1-step2"
    assert world_model.compiles == 1
    assert world_model.executes == []
    assert world_model.hard_compiles == [True]
    assert world_model.reactor.applied == [(0, True), (1, True), (2, True)]

    command_model = _FakeSemanticModel()
    command_initial, command_terminal = _semantic_states(
        command_model,
        batch,
        factor="command",
        owner_state_bridge="oracle-factors",
        semantic_state_mode="hard-st",
    )
    assert command_initial.tag == "oracle-initial"
    assert command_terminal.tag == "oracle-initial-autonomous-terminal"
    assert command_model.compiles == 0
    assert command_model.executes == ["oracle-initial"]
    assert command_model.hard_executes == [True]
    assert command_model.reactor.applied == []


def test_soft_semantic_bridge_preserves_dense_owner_states(monkeypatch) -> None:
    monkeypatch.setattr(
        "train_algebraic_state_semantic_pilot.packet_targets_to_state",
        lambda *_args, **_kwargs: _FakeState("oracle-initial"),
    )
    monkeypatch.setattr(
        "train_algebraic_state_semantic_pilot.target_policy",
        lambda _targets, _config, step, **_kwargs: step,
    )
    batch = _Batch()

    world_model = _FakeSemanticModel()
    _semantic_states(
        world_model,
        batch,
        factor="world",
        owner_state_bridge="oracle-factors",
        semantic_state_mode="soft",
    )
    assert world_model.hard_compiles == [False]
    assert world_model.reactor.applied == [(0, False), (1, False), (2, False)]

    command_model = _FakeSemanticModel()
    _semantic_states(
        command_model,
        batch,
        factor="command",
        owner_state_bridge="oracle-factors",
        semantic_state_mode="soft",
    )
    assert command_model.hard_executes == [False]


def test_balanced_probability_loss_weights_sparse_classes_equally() -> None:
    predicted = torch.tensor([[0.8, 0.2, 0.2, 0.2]])
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    loss = _balanced_probability_loss(predicted, target)
    torch.testing.assert_close(loss, -torch.log(torch.tensor(0.8)))


def test_balanced_probability_loss_accepts_single_class_batches() -> None:
    predicted = torch.tensor([[0.8, 0.7, 0.6]])
    all_occupied = _balanced_probability_loss(
        predicted,
        torch.ones_like(predicted),
    )
    all_unoccupied = _balanced_probability_loss(
        predicted,
        torch.zeros_like(predicted),
    )
    torch.testing.assert_close(all_occupied, -predicted.log().mean())
    torch.testing.assert_close(
        all_unoccupied,
        -(1.0 - predicted).log().mean(),
    )
