from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    TheoryReactorConfig,
    TheoryReactorError,
    TypedTheoryState,
    validate_deployed_state,
)
from parallel_terminal_state_compiler import (
    ParallelTerminalStateCompiler,
    ParallelTerminalStateReactor,
)
from train_parallel_terminal_state_pilot import causal_terminal_delta_brier


def _config() -> TheoryReactorConfig:
    return TheoryReactorConfig(
        d_model=16,
        state_width=16,
        num_slots=5,
        num_types=3,
        num_relations=2,
        num_value_codes=7,
        max_edges=8,
        num_heads=2,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        ff_multiplier=2,
        max_steps=4,
        stage_after_block=0,
    )


def _state(config: TheoryReactorConfig, batch: int = 2) -> TypedTheoryState:
    active = torch.zeros(batch, config.num_slots)
    active[:, :2] = 1.0
    values = torch.zeros(batch, config.num_slots, dtype=torch.long)
    values[:, 0] = 1
    values[:, 1] = 2
    types = torch.zeros_like(values)
    types[:, 1] = 1
    return TypedTheoryState(
        value_probabilities=(
            F.one_hot(values, config.num_value_codes).float()
            * active.unsqueeze(-1)
        ),
        type_probabilities=(
            F.one_hot(types, config.num_types).float()
            * active.unsqueeze(-1)
        ),
        relations=torch.zeros(
            batch,
            config.num_relations,
            config.num_slots,
            config.num_slots,
        ),
        active=active,
        root=F.one_hot(
            torch.zeros(batch, dtype=torch.long),
            config.num_slots,
        ).float(),
        committed=torch.zeros(batch),
        halted=torch.zeros(batch),
        step=0,
    )


def _inputs(config: TheoryReactorConfig, batch: int = 2):
    return {
        "command_hidden": torch.randn(batch, 6, config.d_model),
        "command_attention_mask": torch.tensor(
            [[True, True, True, True, False, False]] * batch
        ),
        "steps": 3,
    }


def test_terminal_compiler_emits_valid_hard_state() -> None:
    config = _config()
    compiler = ParallelTerminalStateCompiler(
        config,
        width=64,
        layers=2,
        num_heads=2,
        relation_width=16,
    )
    terminal = compiler(_state(config), **_inputs(config), hard=True)
    validate_deployed_state(terminal, config)
    assert terminal.step == 3
    assert bool((terminal.root.sum(-1) <= 1).all())
    assert bool((terminal.root <= terminal.active).all())
    pair_active = (
        terminal.active[:, None, :, None]
        * terminal.active[:, None, None, :]
    )
    assert bool((terminal.relations <= pair_active).all())


def test_terminal_compiler_backpropagates_every_semantic_head() -> None:
    config = _config()
    compiler = ParallelTerminalStateCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
        relation_width=16,
    )
    terminal = compiler(_state(config), **_inputs(config), hard=False)
    loss = sum(
        value.float().mean()
        for value in (
            terminal.value_probabilities.square(),
            terminal.type_probabilities.square(),
            terminal.relations.square(),
            terminal.active.square(),
            terminal.root.square(),
            terminal.committed.square(),
            terminal.halted.square(),
        )
    )
    loss.backward()
    for name in (
        "value_head.weight",
        "type_head.weight",
        "active_head.weight",
        "root_head.weight",
        "no_root_head.weight",
        "relation_left.weight",
        "relation_right.weight",
        "relation_bias",
        "status_head.weight",
        "command_projection.weight",
    ):
        gradient = dict(compiler.named_parameters())[name].grad
        assert gradient is not None, name
        assert bool(torch.isfinite(gradient).all()), name


def test_terminal_runtime_has_no_query_or_policy_interface() -> None:
    parameters = inspect.signature(
        ParallelTerminalStateCompiler.forward
    ).parameters
    assert "query" not in " ".join(parameters).lower()
    assert "targets" not in parameters
    assert not hasattr(
        ParallelTerminalStateReactor(
            ParallelTerminalStateCompiler(
                _config(),
                width=64,
                layers=1,
                num_heads=2,
                relation_width=16,
            ),
            _config(),
        ),
        "policy",
    )


def test_terminal_reactor_returns_explicit_policyless_trace() -> None:
    config = _config()
    compiler = ParallelTerminalStateCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
        relation_width=16,
    )
    reactor = ParallelTerminalStateReactor(compiler, config)
    terminal, trace = reactor(_state(config), **_inputs(config), hard=True)
    assert terminal.step == 3
    assert trace.opcode.shape == (2, 3, 9)
    assert trace.active.shape == (2, 3, config.num_slots)
    assert trace.opcode.count_nonzero() == 0
    assert trace.applied_opcode.count_nonzero() == 0
    assert tuple(reactor.state_dict()) == tuple(
        f"compiler.{name}" for name in compiler.state_dict()
    )


def test_production_terminal_compiler_fits_replacement_budget() -> None:
    compiler = ParallelTerminalStateCompiler(TheoryReactorConfig())
    parameters = sum(parameter.numel() for parameter in compiler.parameters())
    # Exact available headroom after removing the old learned reactor from the
    # protected model plus production algebraic reader.
    assert parameters == 18_520_349
    assert parameters < 44_061_106


def test_terminal_compiler_rejects_wrong_command_geometry() -> None:
    config = _config()
    compiler = ParallelTerminalStateCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
        relation_width=16,
    )
    values = _inputs(config)
    values["command_attention_mask"] = torch.ones(2, 5, dtype=torch.bool)
    with pytest.raises(TheoryReactorError, match="input differs"):
        compiler(_state(config), **values, hard=False)


def test_causal_delta_loss_penalizes_intervention_invariance() -> None:
    config = _config()
    target = _state(config, batch=4)
    target.value_probabilities[2, 0].zero_()
    target.value_probabilities[2, 0, 3] = 1.0
    target.value_probabilities[3, 0].zero_()
    target.value_probabilities[3, 0, 3] = 1.0
    rectangles = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
    slot_mask = torch.ones(4, config.num_slots, dtype=torch.bool)
    relation_mask = torch.ones_like(target.relations, dtype=torch.bool)

    matched_loss, matched_parts, counts = causal_terminal_delta_brier(
        target,
        target,
        rectangle_rows=rectangles,
        slot_mask=slot_mask,
        relation_mask=relation_mask,
    )
    invariant_loss, _, _ = causal_terminal_delta_brier(
        _state(config, batch=4),
        target,
        rectangle_rows=rectangles,
        slot_mask=slot_mask,
        relation_mask=relation_mask,
    )
    assert matched_loss.item() == 0.0
    assert invariant_loss.item() > 0.0
    assert tuple(matched_parts) == ("world.value_code",)
    assert counts["world.value_code"] == 2


def test_causal_delta_loss_reaches_both_rectangle_axes() -> None:
    config = _config()
    target = _state(config, batch=4)
    for row, code in ((1, 3), (2, 4), (3, 5)):
        target.value_probabilities[row, 0].zero_()
        target.value_probabilities[row, 0, code] = 1.0
    predicted_values = target.value_probabilities.detach().clone()
    predicted_values.requires_grad_(True)
    predicted = TypedTheoryState(
        value_probabilities=predicted_values,
        type_probabilities=target.type_probabilities,
        relations=target.relations,
        active=target.active,
        root=target.root,
        committed=target.committed,
        halted=target.halted,
        step=target.step,
    )
    rectangles = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
    loss, parts, counts = causal_terminal_delta_brier(
        predicted,
        target,
        rectangle_rows=rectangles,
        slot_mask=torch.ones(4, config.num_slots, dtype=torch.bool),
        relation_mask=torch.ones_like(target.relations, dtype=torch.bool),
    )
    loss.backward()
    assert "world.value_code" in parts
    assert "command.value_code" in parts
    assert counts["world.value_code"] == 2
    assert counts["command.value_code"] == 2
    assert predicted_values.grad is not None
    assert bool(predicted_values.grad.isfinite().all())
