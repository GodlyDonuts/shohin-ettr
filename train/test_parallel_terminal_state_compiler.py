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
    AtomicTypedEdits,
    ParallelTerminalStateCompiler,
    ParallelTerminalStateReactor,
)
from train_parallel_terminal_state_pilot import (
    atomic_typed_edit_loss,
    causal_terminal_delta_brier,
    derive_atomic_edit_targets,
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


def test_sparse_residual_terminal_compiler_emits_valid_hard_state() -> None:
    config = _config()
    compiler = ParallelTerminalStateCompiler(
        config,
        width=64,
        layers=2,
        num_heads=2,
        relation_width=16,
        residual_edits=True,
    )
    terminal = compiler(_state(config), **_inputs(config), hard=True)
    validate_deployed_state(terminal, config)
    assert terminal.step == 3


def test_atomic_terminal_compiler_emits_valid_hard_state() -> None:
    config = _config()
    compiler = ParallelTerminalStateCompiler(
        config,
        width=64,
        layers=2,
        num_heads=2,
        relation_width=16,
        atomic_edits=True,
    )
    terminal = compiler(_state(config), **_inputs(config), hard=True)
    validate_deployed_state(terminal, config)
    assert terminal.step == 3


def test_lexical_atomic_compiler_requires_and_uses_direct_tokens() -> None:
    config = _config()
    compiler = ParallelTerminalStateCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
        relation_width=16,
        atomic_edits=True,
        lexical_command=True,
    )
    inputs = _inputs(config)
    with pytest.raises(TheoryReactorError, match="input differs"):
        compiler(_state(config), **inputs, hard=False)
    inputs["command_lexical"] = torch.randn_like(inputs["command_hidden"])
    _terminal, edits = compiler.forward_with_atomic_edits(
        _state(config),
        **inputs,
        hard=False,
    )
    edits.value_code.square().mean().backward()
    gradient = compiler.command_lexical_projection.weight.grad
    assert gradient is not None
    assert bool(gradient.isfinite().all())


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


def test_production_sparse_residual_compiler_fits_system_cap() -> None:
    compiler = ParallelTerminalStateCompiler(
        TheoryReactorConfig(),
        residual_edits=True,
    )
    parameters = sum(parameter.numel() for parameter in compiler.parameters())
    assert parameters == 19_572_019
    assert parameters < 44_061_106


def test_production_atomic_edit_compiler_fits_system_cap() -> None:
    compiler = ParallelTerminalStateCompiler(
        TheoryReactorConfig(),
        atomic_edits=True,
    )
    parameters = sum(parameter.numel() for parameter in compiler.parameters())
    assert parameters == 19_574_616
    assert parameters < 44_061_106


def test_production_lexical_atomic_compiler_fits_system_cap() -> None:
    compiler = ParallelTerminalStateCompiler(
        TheoryReactorConfig(),
        atomic_edits=True,
        lexical_command=True,
    )
    parameters = sum(parameter.numel() for parameter in compiler.parameters())
    assert parameters == 19_869_528
    assert parameters < 44_061_106


def test_production_occurrence_linked_compiler_fits_system_cap() -> None:
    from ettr_il_v2_token_native_surface import (
        DEFAULT_TOKENIZER_PATH,
        TokenNativeSurfaceCodec,
    )

    codec = TokenNativeSurfaceCodec(DEFAULT_TOKENIZER_PATH)
    compiler = ParallelTerminalStateCompiler(
        TheoryReactorConfig(),
        atomic_edits=True,
        lexical_command=True,
        token_native_command_mask=True,
        token_native_occurrence_command=True,
        token_native_codebook_ids=codec.codebook.token_ids,
        token_native_vocab_size=codec.tokenizer.get_vocab_size(),
    )
    parameters = sum(parameter.numel() for parameter in compiler.parameters())
    assert 19_869_528 < parameters < 44_061_106


def test_syntax_routed_atomic_compiler_ignores_transport_cover() -> None:
    config = _config()
    compiler = ParallelTerminalStateCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
        relation_width=16,
        atomic_edits=True,
        lexical_command=True,
        token_native_command_mask=True,
        token_native_codebook_ids=tuple(range(700)),
        token_native_vocab_size=700,
    )
    tokens = torch.tensor(
        [
            [528, 528, 497, 565, 430, 566, 100, 200],
            [528, 528, 497, 565, 430, 566, 300, 400],
        ],
        dtype=torch.long,
    )
    hidden = torch.randn(2, 8, config.d_model)
    lexical = torch.randn_like(hidden)
    hidden[1, :6] = hidden[0, :6]
    lexical[1, :6] = lexical[0, :6]
    terminal = compiler(
        _state(config),
        command_hidden=hidden,
        command_lexical=lexical,
        command_tokens=tokens,
        command_attention_mask=torch.ones(2, 8, dtype=torch.bool),
        steps=3,
        hard=False,
    )
    assert torch.allclose(
        terminal.value_probabilities[0],
        terminal.value_probabilities[1],
    )


def test_sparse_residual_gate_can_preserve_initial_identity() -> None:
    config = _config()
    compiler = ParallelTerminalStateCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
        relation_width=16,
        residual_edits=True,
    )
    for head in (
        compiler.value_edit_head,
        compiler.type_edit_head,
        compiler.active_edit_head,
        compiler.root_edit_head,
        compiler.status_edit_head,
    ):
        assert head is not None
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.constant_(head.bias, -100.0)
    assert compiler.relation_edit_left is not None
    assert compiler.relation_edit_right is not None
    assert compiler.relation_edit_bias is not None
    torch.nn.init.zeros_(compiler.relation_edit_left.weight)
    torch.nn.init.zeros_(compiler.relation_edit_right.weight)
    torch.nn.init.constant_(compiler.relation_edit_bias, -100.0)
    initial = _state(config)
    terminal = compiler(initial, **_inputs(config), hard=False)
    for name in (
        "value_probabilities",
        "type_probabilities",
        "relations",
        "active",
        "root",
        "committed",
        "halted",
    ):
        assert torch.equal(getattr(terminal, name), getattr(initial, name))


def test_sparse_residual_edit_heads_receive_gradients() -> None:
    config = _config()
    compiler = ParallelTerminalStateCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
        relation_width=16,
        residual_edits=True,
    )
    terminal = compiler(_state(config), **_inputs(config), hard=False)
    loss = sum(
        value.float().square().mean()
        for value in (
            terminal.value_probabilities,
            terminal.type_probabilities,
            terminal.relations,
            terminal.active,
            terminal.root,
            terminal.committed,
            terminal.halted,
        )
    )
    loss.backward()
    parameters = dict(compiler.named_parameters())
    for name in (
        "value_edit_head.weight",
        "type_edit_head.weight",
        "active_edit_head.weight",
        "root_edit_head.weight",
        "relation_edit_left.weight",
        "relation_edit_right.weight",
        "relation_edit_bias",
        "status_edit_head.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert bool(gradient.isfinite().all()), name


def test_atomic_edit_heads_receive_gradients() -> None:
    config = _config()
    compiler = ParallelTerminalStateCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
        relation_width=16,
        atomic_edits=True,
    )
    _terminal, edits = compiler.forward_with_atomic_edits(
        _state(config),
        **_inputs(config),
        hard=False,
    )
    loss = sum(
        value.float().square().mean()
        for value in (
            edits.node_action,
            edits.value_code,
            edits.type_index,
            edits.relation_action,
            edits.root_action,
            edits.disposition_action,
        )
    )
    loss.backward()
    parameters = dict(compiler.named_parameters())
    for name in (
        "node_action_head.weight",
        "value_head.weight",
        "type_head.weight",
        "relation_left.weight",
        "relation_right.weight",
        "relation_unlink_left.weight",
        "relation_unlink_right.weight",
        "relation_action_bias",
        "root_control_head.weight",
        "root_head.weight",
        "disposition_action_head.weight",
        "command_projection.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert bool(gradient.isfinite().all()), name


def test_canonical_atomic_edits_reconstruct_target_state() -> None:
    config = _config()
    compiler = ParallelTerminalStateCompiler(
        config,
        width=64,
        layers=1,
        num_heads=2,
        relation_width=16,
        atomic_edits=True,
    )
    initial = _state(config)
    target = initial.detached_clone()
    target.active[0, 2] = 1.0
    target.value_probabilities[0, 2, 3] = 1.0
    target.type_probabilities[0, 2, 2] = 1.0
    target.relations[0, 0, 0, 2] = 1.0
    target.root[0].zero_()
    target.root[0, 2] = 1.0
    target.committed[0] = 1.0
    target.value_probabilities[1, 1].zero_()
    target.value_probabilities[1, 1, 4] = 1.0
    target.active[1, 0] = 0.0
    target.value_probabilities[1, 0].zero_()
    target.type_probabilities[1, 0].zero_()
    target.root[1].zero_()
    target.halted[1] = 1.0
    labels = derive_atomic_edit_targets(initial, target)
    edits = AtomicTypedEdits(
        node_action=F.one_hot(labels["node_action"], 5).float(),
        value_code=F.one_hot(labels["value_code"], config.num_value_codes).float(),
        type_index=F.one_hot(labels["type_index"], config.num_types).float(),
        relation_action=F.one_hot(labels["relation_action"], 3).float(),
        root_action=F.one_hot(labels["root_action"], 2 + config.num_slots).float(),
        disposition_action=F.one_hot(labels["disposition_action"], 4).float(),
    )
    terminal = compiler.apply_atomic_edits(initial, edits, steps=3, hard=True)
    for name in (
        "value_probabilities",
        "type_probabilities",
        "relations",
        "active",
        "root",
        "committed",
        "halted",
    ):
        assert torch.equal(getattr(terminal, name), getattr(target, name)), name


def test_atomic_action_loss_is_exact_for_canonical_edits() -> None:
    config = _config()
    initial = _state(config)
    target = initial.detached_clone()
    target.value_probabilities[:, 1].zero_()
    target.value_probabilities[:, 1, 4] = 1.0
    labels = derive_atomic_edit_targets(initial, target)
    edits = AtomicTypedEdits(
        node_action=F.one_hot(labels["node_action"], 5).float(),
        value_code=F.one_hot(labels["value_code"], config.num_value_codes).float(),
        type_index=F.one_hot(labels["type_index"], config.num_types).float(),
        relation_action=F.one_hot(labels["relation_action"], 3).float(),
        root_action=F.one_hot(labels["root_action"], 2 + config.num_slots).float(),
        disposition_action=F.one_hot(labels["disposition_action"], 4).float(),
    )
    loss, parts, _counts = atomic_typed_edit_loss(
        edits,
        labels,
        slot_mask=torch.ones_like(initial.active, dtype=torch.bool),
        relation_mask=torch.ones_like(initial.relations, dtype=torch.bool),
    )
    assert loss.item() == 0.0
    assert "node_action" in parts
    assert "value_code" in parts


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
