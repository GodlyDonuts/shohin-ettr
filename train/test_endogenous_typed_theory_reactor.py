from __future__ import annotations

from dataclasses import fields, replace

import torch

from endogenous_typed_theory_reactor import (
    DISPOSITION_COUNT,
    HARD_SURROGATE_GRADIENT_CAP,
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
    TypedTheoryState,
    _bounded_hard_adjoint,
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
            num_value_codes=64,
            max_edges=96,
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
    assert trace.opcode.shape == (2, 3, DISPOSITION_COUNT + 5)
    loss.backward()
    for gradient in (
        model.compiler.token_projection.weight.grad,
        model.compiler.value_head.weight.grad,
        model.reactor.opcode_head.weight.grad,
        model.reactor.value_head.weight.grad,
        model.query_reader.output_projection.weight.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient)


def test_source_deleted_state_has_only_allowlisted_tensors() -> None:
    model = _model()
    world = torch.randint(0, 64, (1, 7))
    state = model.compile_world(world)
    assert tuple(field.name for field in fields(TypedTheoryState)) == (
        "value_probabilities",
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


def test_hard_state_adjoint_is_exact_forward_and_bounded_backward() -> None:
    value = torch.tensor([-3.0, 2.0], requires_grad=True)
    observed = _bounded_hard_adjoint(value)
    assert torch.equal(observed, value)
    observed.backward(torch.tensor([1e9, -1e9]))
    torch.testing.assert_close(
        value.grad,
        torch.tensor(
            [
                HARD_SURROGATE_GRADIENT_CAP,
                -HARD_SURROGATE_GRADIENT_CAP,
            ]
        ),
    )


def test_hard_reactor_emits_exact_transaction_choices() -> None:
    model = _model()
    state = model.compile_world(
        torch.randint(0, 64, (2, 8)),
        hard=True,
    )
    state, trace = model.execute(state, steps=4, hard=True)
    assert torch.equal(
        trace.applied_opcode.detach().sum(-1),
        torch.ones(2, 4),
    )
    assert torch.equal(
        trace.applied_source.detach().sum(-1),
        torch.ones(2, 4),
    )
    assert torch.equal(
        trace.applied_target.detach().sum(-1),
        torch.ones(2, 4),
    )
    assert torch.equal(
        trace.applied_relation.detach().sum(-1),
        torch.ones(2, 4),
    )
    assert torch.equal(
        trace.applied_type_index.detach().sum(-1),
        torch.ones(2, 4),
    )
    assert torch.equal(
        trace.applied_value_code.detach().sum(-1),
        torch.ones(2, 4),
    )
    assert torch.equal(
        state.value_probabilities.sum(-1),
        state.active,
    )
    assert bool((state.relations.sum(dim=(1, 2, 3)) <= model.config.max_edges).all())
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
    assert not any(parameter.requires_grad for parameter in model.base.parameters())
    assert all(parameter.requires_grad for parameter in model.compiler.parameters())
    assert all(parameter.requires_grad for parameter in model.reactor.parameters())


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


def test_query_path_is_causal_under_future_token_extension() -> None:
    model = _model().eval()
    state = model.compile_world(
        torch.randint(0, 64, (1, 8)),
        hard=True,
    )
    state = replace(state, committed=torch.ones_like(state.committed))
    prefix = torch.randint(0, 64, (1, 5))
    extended = torch.cat(
        (prefix, torch.randint(0, 64, (1, 4))),
        dim=1,
    )
    with torch.no_grad():
        prefix_logits, _ = model.answer_query(state, prefix)
        extended_logits, _ = model.answer_query(state, extended)
    torch.testing.assert_close(
        prefix_logits,
        extended_logits[:, : prefix.shape[1]],
        atol=1e-6,
        rtol=1e-6,
    )


def test_all_declared_state_fields_enter_the_learned_path() -> None:
    model = _model()
    config = model.config
    tensors = {
        "value_probabilities": torch.rand(
            1,
            config.num_slots,
            config.num_value_codes,
            requires_grad=True,
        ),
        "type_probabilities": torch.rand(
            1,
            config.num_slots,
            config.num_types,
            requires_grad=True,
        ),
        "relations": torch.rand(
            1,
            config.num_relations,
            config.num_slots,
            config.num_slots,
            requires_grad=True,
        ),
        "active": torch.full(
            (1, config.num_slots),
            0.8,
            requires_grad=True,
        ),
        "root": torch.rand(
            1,
            config.num_slots,
            requires_grad=True,
        ),
        "committed": torch.full(
            (1,),
            0.7,
            requires_grad=True,
        ),
        "halted": torch.full(
            (1,),
            0.2,
            requires_grad=True,
        ),
    }
    state = TypedTheoryState(**tensors, step=0)
    output = model.query_reader(
        torch.rand(1, 4, config.d_model),
        state,
    )
    output.square().mean().backward()
    for value in tensors.values():
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad)


def test_degree_preserving_edge_swaps_change_reactor_and_query_reads() -> None:
    model = _model().eval()
    config = model.config
    active = torch.tensor(
        [[True, True, True, True, False, False]],
        dtype=torch.float32,
    )
    value_codes = torch.tensor([[0, 1, 2, 3, 0, 0]])
    type_codes = torch.tensor([[0, 1, 2, 0, 0, 0]])
    relations = torch.zeros(
        1,
        config.num_relations,
        config.num_slots,
        config.num_slots,
    )
    relations[:, 0, 0, 1] = 1
    relations[:, 0, 2, 3] = 1
    swapped_relations = torch.zeros_like(relations)
    swapped_relations[:, 0, 0, 3] = 1
    swapped_relations[:, 0, 2, 1] = 1
    assert torch.equal(relations.sum(-1), swapped_relations.sum(-1))
    assert torch.equal(relations.sum(-2), swapped_relations.sum(-2))

    state = TypedTheoryState(
        value_probabilities=(
            torch.nn.functional.one_hot(
                value_codes,
                config.num_value_codes,
            ).float()
            * active.unsqueeze(-1)
        ),
        type_probabilities=(
            torch.nn.functional.one_hot(
                type_codes,
                config.num_types,
            ).float()
            * active.unsqueeze(-1)
        ),
        relations=relations,
        active=active,
        root=torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.float32),
        committed=torch.zeros(1),
        halted=torch.zeros(1),
        step=0,
    )
    swapped = replace(state, relations=swapped_relations)
    with torch.no_grad():
        policy = model.reactor.policy(state, hard=False)
        swapped_policy = model.reactor.policy(swapped, hard=False)
    assert not torch.equal(policy.source, swapped_policy.source)
    assert not torch.equal(policy.target, swapped_policy.target)

    query_hidden = torch.rand(1, 4, config.d_model)
    committed = replace(state, committed=torch.ones(1))
    swapped_committed = replace(swapped, committed=torch.ones(1))
    with torch.no_grad():
        read = model.query_reader(query_hidden, committed)
        swapped_read = model.query_reader(query_hidden, swapped_committed)
    assert not torch.equal(read, swapped_read)


def test_hard_commit_freezes_structural_state() -> None:
    model = _model().eval()
    initial = model.compile_world(
        torch.randint(0, 64, (2, 8)),
        hard=True,
    )
    committed = replace(
        initial,
        committed=torch.ones_like(initial.committed),
    )
    terminal, _ = model.execute(
        committed,
        steps=1,
        hard=True,
    )
    for name in (
        "value_probabilities",
        "type_probabilities",
        "relations",
        "active",
        "root",
        "committed",
    ):
        assert torch.equal(
            getattr(terminal, name),
            getattr(committed, name),
        )


def test_query_reader_distinguishes_all_four_dispositions() -> None:
    model = _model().eval()
    state = model.compile_world(
        torch.randint(0, 64, (1, 8)),
        hard=True,
    )
    query_hidden = torch.rand(1, 4, model.config.d_model)
    dispositions = (
        replace(state, committed=torch.zeros(1), halted=torch.zeros(1)),
        replace(state, committed=torch.ones(1), halted=torch.zeros(1)),
        replace(state, committed=torch.zeros(1), halted=torch.ones(1)),
        replace(state, committed=torch.ones(1), halted=torch.ones(1)),
    )
    with torch.no_grad():
        reads = tuple(
            model.query_reader(query_hidden, disposition)
            for disposition in dispositions
        )
    for left in range(len(reads)):
        for right in range(left + 1, len(reads)):
            assert not torch.equal(reads[left], reads[right])


def test_hard_policy_keeps_soft_supervision_gradients() -> None:
    model = _model()
    state = model.compile_world(
        torch.randint(0, 64, (1, 8)),
        hard=True,
    )
    policy = model.reactor.policy(state, hard=True)
    wrong_target = (policy.opcode.argmax(-1) + 1) % policy.opcode.shape[-1]
    loss = -policy.opcode_probabilities.gather(
        -1,
        wrong_target[:, None],
    ).log().mean()
    loss.backward()
    gradient = model.reactor.opcode_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient)
    assert torch.equal(
        policy.opcode.sum(-1),
        torch.ones_like(policy.opcode.sum(-1)),
    )
