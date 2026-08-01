from __future__ import annotations

import torch
import pytest

from endogenous_typed_theory_reactor import (
    GenericTransactionReactor,
    TheoryReactorConfig,
    TheoryReactorError,
    TypedTheoryState,
)
from parallel_addressed_transaction_compiler import (
    ParallelAddressedTransactionCompiler,
    ParallelScheduledReactor,
)


def _config() -> TheoryReactorConfig:
    return TheoryReactorConfig(
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


def _state(config: TheoryReactorConfig, batch: int = 2) -> TypedTheoryState:
    active = torch.tensor([[1.0, 1.0, 0.0, 0.0]]).expand(batch, -1).clone()
    values = torch.zeros(batch, config.num_slots, config.num_value_codes)
    values[:, 0, 1] = 1.0
    values[:, 1, 2] = 1.0
    types = torch.zeros(batch, config.num_slots, config.num_types)
    types[:, :2, 0] = 1.0
    root = torch.zeros(batch, config.num_slots)
    root[:, 0] = 1.0
    return TypedTheoryState(
        value_probabilities=values,
        type_probabilities=types,
        relations=torch.zeros(
            batch,
            config.num_relations,
            config.num_slots,
            config.num_slots,
        ),
        active=active,
        root=root,
        committed=torch.zeros(batch),
        halted=torch.zeros(batch),
        step=0,
    )


def test_parallel_schedule_is_well_formed_and_hard() -> None:
    config = _config()
    compiler = ParallelAddressedTransactionCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
    )
    schedule = compiler(
        _state(config),
        command_hidden=torch.randn(2, 5, config.d_model),
        command_attention_mask=torch.ones(2, 5, dtype=torch.bool),
        steps=3,
        hard=True,
    )
    expected = {
        "opcode": 9,
        "source": config.num_slots,
        "target": config.num_slots,
        "relation": config.num_relations,
        "type_index": config.num_types,
        "value_code": config.num_value_codes,
    }
    for name, classes in expected.items():
        probability = getattr(schedule, name)
        applied = getattr(schedule, f"applied_{name}")
        assert probability.shape == (2, 3, classes)
        assert applied.shape == probability.shape
        assert torch.equal(applied.sum(-1), torch.ones(2, 3))
        assert bool(((applied == 0) | (applied == 1)).all())


def test_parallel_reactor_uses_one_schedule_for_the_full_rollout() -> None:
    config = _config()
    compiler = ParallelAddressedTransactionCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
    )
    reactor = ParallelScheduledReactor(
        compiler,
        config,
    )
    terminal, trace = reactor(
        _state(config),
        steps=2,
        hard=True,
        command_hidden=torch.randn(2, 5, config.d_model),
        command_attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    assert terminal.step == 2
    assert trace.opcode.shape == (2, 2, 9)
    assert trace.active.shape == (2, 2, config.num_slots)
    assert tuple(reactor.state_dict()) == tuple(
        f"compiler.{name}" for name in compiler.state_dict()
    )


def test_parameterless_reactor_reuses_the_exact_transaction_algebra() -> None:
    config = _config()
    compiler = ParallelAddressedTransactionCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
    )
    exact = ParallelScheduledReactor(compiler, config)
    reference = GenericTransactionReactor(config)
    state = _state(config)
    schedule = compiler(
        state,
        command_hidden=torch.randn(2, 5, config.d_model),
        command_attention_mask=torch.ones(2, 5, dtype=torch.bool),
        steps=1,
        hard=True,
    )
    expected = reference.apply(state, schedule.policy(0), hard=True)
    observed = exact.apply(state, schedule.policy(0), hard=True)
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
    assert observed.step == expected.step


def test_pointer_masks_require_grounded_slot_scores() -> None:
    with pytest.raises(
        TheoryReactorError,
        match="addressed schedule geometry differs",
    ):
        ParallelAddressedTransactionCompiler(
            _config(),
            width=64,
            layers=1,
            num_heads=2,
            valid_pointer_masks=True,
        )


def test_grounded_pointer_masks_follow_transaction_validity() -> None:
    config = _config()
    compiler = ParallelAddressedTransactionCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
        grounded_pointers=True,
        valid_pointer_masks=True,
    )
    with torch.no_grad():
        compiler.source_query.weight.zero_()
        compiler.target_query.weight.zero_()
        compiler.opcode_head.weight.zero_()
        compiler.opcode_head.bias.fill_(-20.0)
        compiler.opcode_head.bias[1] = 20.0
    kwargs = {
        "command_hidden": torch.randn(2, 5, config.d_model),
        "command_attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "steps": 3,
        "hard": True,
    }
    write = compiler(_state(config), **kwargs)
    assert bool((write.source[..., 2:].sum(-1) < 1e-3).all())

    with torch.no_grad():
        compiler.opcode_head.bias.fill_(-20.0)
        compiler.opcode_head.bias[0] = 20.0
    allocate = compiler(_state(config), **kwargs)
    assert bool((allocate.source[..., :2].sum(-1) < 1e-3).all())

    with torch.no_grad():
        compiler.opcode_head.bias.fill_(-20.0)
        compiler.opcode_head.bias[3] = 20.0
    link = compiler(_state(config), **kwargs)
    assert bool((link.target[..., 2:].sum(-1) < 1e-3).all())


def test_grounded_pointer_masks_track_allocations_across_the_schedule() -> None:
    config = _config()
    compiler = ParallelAddressedTransactionCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
        grounded_pointers=True,
        valid_pointer_masks=True,
    )
    with torch.no_grad():
        compiler.source_query.weight.zero_()
        compiler.opcode_head.weight.zero_()
        compiler.opcode_head.bias.fill_(-20.0)
        compiler.opcode_head.bias[0] = 20.0
    schedule = compiler(
        _state(config),
        command_hidden=torch.randn(2, 5, config.d_model),
        command_attention_mask=torch.ones(2, 5, dtype=torch.bool),
        steps=2,
        hard=True,
    )
    selected = schedule.applied_source.argmax(-1)
    assert torch.equal(selected[:, 0], torch.full((2,), 2))
    assert torch.equal(selected[:, 1], torch.full((2,), 3))


def test_grounded_pointer_parameters_replace_fixed_slot_classifiers() -> None:
    compiler = ParallelAddressedTransactionCompiler(
        _config(),
        width=64,
        layers=1,
        num_heads=2,
        grounded_pointers=True,
    )
    names = set(compiler.state_dict())
    assert "source_query.weight" in names
    assert "target_query.weight" in names
    assert "slot_key.weight" in names
    assert "source_head.weight" not in names
    assert "target_head.weight" not in names


def test_grounded_production_geometry_stays_inside_system_cap() -> None:
    compiler = ParallelAddressedTransactionCompiler(
        TheoryReactorConfig(),
        grounded_pointers=True,
        valid_pointer_masks=True,
    )
    parameters = sum(parameter.numel() for parameter in compiler.parameters())
    assert parameters == 6_855_841
    assert 185_696_111 - 29_757_217 + parameters == 162_794_735


def test_production_geometry_stays_inside_remaining_parameter_budget() -> None:
    compiler = ParallelAddressedTransactionCompiler(TheoryReactorConfig())
    parameters = sum(parameter.numel() for parameter in compiler.parameters())
    assert parameters == 6_462_753
    recurrent_policy_parameters = sum(
        parameter.numel()
        for parameter in GenericTransactionReactor(
            TheoryReactorConfig()
        ).parameters()
    )
    assert recurrent_policy_parameters == 29_757_217
    assert (
        185_696_111 - recurrent_policy_parameters + parameters
        == 162_401_647
    )


def test_wide_geometry_uses_the_recovered_budget_without_crossing_cap() -> None:
    config = TheoryReactorConfig()
    compiler = ParallelAddressedTransactionCompiler(
        config,
        width=896,
        layers=4,
        num_heads=14,
    )
    schedule_parameters = sum(
        parameter.numel() for parameter in compiler.parameters()
    )
    assert schedule_parameters == 43_074_721
    assert 185_696_111 - 29_757_217 + schedule_parameters == 199_013_615
