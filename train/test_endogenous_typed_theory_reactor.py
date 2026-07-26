from __future__ import annotations

from dataclasses import fields

import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
    TypedTheoryState,
    validate_state,
)
from model import GPT, GPTConfig


def _model() -> EndogenousTypedTheoryReactorGPT:
    torch.manual_seed(2026072501)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=32,
            d_ff=64,
            seq_len=32,
            zloss=0.0,
        )
    )
    return EndogenousTypedTheoryReactorGPT(
        base,
        TheoryReactorConfig(
            d_model=32,
            state_width=32,
            num_slots=6,
            num_types=3,
            num_relations=3,
            num_heads=4,
            compiler_layers=1,
            reactor_layers=1,
            query_layers=1,
            ff_multiplier=2,
            max_steps=6,
            stage_after_block=1,
            parameter_cap=1_000_000,
        ),
    )


def test_staged_architecture_is_trainable_end_to_end() -> None:
    model = _model()
    world = torch.randint(0, 64, (2, 9))
    query = torch.randint(0, 64, (2, 5))
    targets = torch.randint(0, 64, (2, 5))
    logits, loss, state, trace = model.forward_staged(
        world,
        query,
        reactor_steps=3,
        targets=targets,
    )
    assert logits.shape == (2, 5, 64)
    assert loss is not None and torch.isfinite(loss)
    validate_state(state, model.config)
    assert trace.opcode.shape == (2, 3, 8)
    loss.backward()
    assert model.compiler.token_projection.weight.grad is not None
    assert model.reactor.opcode_head.weight.grad is not None
    assert model.query_reader.output_projection.weight.grad is not None


def test_source_deleted_state_has_only_allowlisted_tensors() -> None:
    model = _model()
    world = torch.randint(0, 64, (1, 7))
    state = model.compile_world(world)
    assert tuple(field.name for field in fields(TypedTheoryState)) == (
        "values",
        "type_probabilities",
        "relations",
        "active",
        "root",
        "committed",
        "halted",
        "step",
    )
    detached = state.detached_clone()
    for field in fields(TypedTheoryState):
        value = getattr(detached, field.name)
        if isinstance(value, torch.Tensor):
            assert value.grad_fn is None
            assert not value.requires_grad


def test_hard_reactor_emits_exact_transaction_choices() -> None:
    model = _model()
    state = model.compile_world(
        torch.randint(0, 64, (2, 8)),
        hard=True,
    )
    state, trace = model.execute(state, steps=4, hard=True)
    assert torch.equal(
        trace.opcode.detach().sum(-1),
        torch.ones(2, 4),
    )
    assert torch.equal(
        trace.source.detach().sum(-1),
        torch.ones(2, 4),
    )
    assert torch.equal(
        trace.target.detach().sum(-1),
        torch.ones(2, 4),
    )
    assert torch.equal(
        trace.relation.detach().sum(-1),
        torch.ones(2, 4),
    )
    assert torch.equal(
        trace.type_index.detach().sum(-1),
        torch.ones(2, 4),
    )
    validate_state(state, model.config)


def test_parameter_receipt_counts_unique_complete_system() -> None:
    model = _model()
    receipt = model.parameter_receipt()
    assert receipt.base_parameters == model.base.num_params()
    assert receipt.architecture_parameters > 0
    assert receipt.complete_system_parameters == (
        receipt.base_parameters + receipt.architecture_parameters
    )
    assert receipt.complete_system_parameters < receipt.parameter_cap


def test_base_can_be_frozen_without_freezing_architecture() -> None:
    model = _model()
    model.freeze_base()
    assert not any(
        parameter.requires_grad
        for parameter in model.base.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.compiler.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.reactor.parameters()
    )


def test_post_seal_command_tokens_causally_enter_reactor() -> None:
    model = _model().eval()
    world = torch.randint(0, 64, (1, 8))
    first_command = torch.randint(0, 64, (1, 6))
    second_command = first_command.clone()
    second_command[:, 2] = (second_command[:, 2] + 1) % 64
    state = model.compile_world(world)
    first_policy = model.reactor.policy(
        state,
        hard=False,
        command_hidden=model._encode_to_stage(
            first_command,
            pos=0,
        ),
    )
    second_policy = model.reactor.policy(
        state,
        hard=False,
        command_hidden=model._encode_to_stage(
            second_command,
            pos=0,
        ),
    )
    assert not torch.equal(first_policy.opcode, second_policy.opcode)
    assert not torch.equal(first_policy.source, second_policy.source)
