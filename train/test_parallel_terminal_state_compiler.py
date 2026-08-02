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
