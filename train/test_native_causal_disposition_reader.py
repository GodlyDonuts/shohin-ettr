from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from endogenous_typed_theory_reactor import (
    SourceDeletedQueryReader,
    TheoryReactorConfig,
    TheoryReactorError,
    TypedTheoryState,
)
from native_causal_disposition_reader import NativeCausalDispositionReader


def _config() -> TheoryReactorConfig:
    return TheoryReactorConfig(
        d_model=16,
        state_width=16,
        num_slots=3,
        num_types=2,
        num_relations=2,
        num_value_codes=5,
        max_edges=4,
        num_heads=2,
        compiler_layers=1,
        reactor_layers=1,
        query_layers=1,
        ff_multiplier=2,
        max_steps=2,
        stage_after_block=0,
    )


def _permuted_state(
    state: TypedTheoryState,
    permutation: torch.Tensor,
) -> TypedTheoryState:
    relations = state.relations.index_select(2, permutation).index_select(
        3,
        permutation,
    )
    return TypedTheoryState(
        value_probabilities=state.value_probabilities.index_select(
            1,
            permutation,
        ),
        type_probabilities=state.type_probabilities.index_select(
            1,
            permutation,
        ),
        relations=relations,
        active=state.active.index_select(1, permutation),
        root=state.root.index_select(1, permutation),
        committed=state.committed,
        halted=state.halted,
        step=state.step,
    )


def _state(*, committed: float, halted: float) -> TypedTheoryState:
    return TypedTheoryState(
        value_probabilities=torch.nn.functional.one_hot(
            torch.zeros((2, 3), dtype=torch.long),
            5,
        ).float(),
        type_probabilities=torch.nn.functional.one_hot(
            torch.zeros((2, 3), dtype=torch.long),
            2,
        ).float(),
        relations=torch.zeros((2, 2, 3, 3)),
        active=torch.ones((2, 3)),
        root=torch.zeros((2, 3)),
        committed=torch.full((2,), committed),
        halted=torch.full((2,), halted),
        step=1,
    )


def test_disposition_reader_exposes_only_bound_answer_tokens() -> None:
    reader = NativeCausalDispositionReader(
        _config(),
        vocab_size=32,
        answer_token_ids=(4, 7, 11, 19),
    )
    logits = reader(
        torch.randn((2, 5, 16)),
        _state(committed=1.0, halted=0.0),
        attention_mask=torch.ones((2, 5), dtype=torch.bool),
    )
    assert logits.shape == (2, 5, 32)
    legal = logits[..., (4, 7, 11, 19)]
    assert torch.isfinite(legal).all()
    illegal = logits.clone()
    illegal[..., (4, 7, 11, 19)] = illegal.amin()
    assert bool((illegal.max(-1).values < legal.min(-1).values).all())
    torch.nn.functional.cross_entropy(
        logits[:, -1].float(),
        torch.tensor((4, 7)),
    ).backward()
    motor_gradients = tuple(
        parameter.grad for parameter in reader.truth_motor.parameters()
    )
    assert all(gradient is not None for gradient in motor_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in motor_gradients
        if gradient is not None
    )


def test_nonlinear_truth_motor_preserves_causal_interface() -> None:
    reader = NativeCausalDispositionReader(
        _config(),
        vocab_size=32,
        answer_token_ids=(4, 7, 11, 19),
        truth_motor_hidden=64,
    )
    assert reader.truth_motor_hidden == 64
    assert sum(
        parameter.numel() for parameter in reader.truth_motor.parameters()
    ) == 16 * 64 + 64 + 64 * 2 + 2
    logits = reader(
        torch.randn((2, 5, 16)),
        _state(committed=1.0, halted=0.0),
        attention_mask=torch.ones((2, 5), dtype=torch.bool),
    )
    assert logits.shape == (2, 5, 32)


def test_unaddressed_reader_is_invariant_to_consistent_slot_permutation() -> None:
    torch.manual_seed(7)
    reader = SourceDeletedQueryReader(_config()).eval()
    query = torch.randn((2, 5, 16))
    state = _state(committed=1.0, halted=0.0)
    state.value_probabilities[0, 0] = torch.tensor((0, 1, 0, 0, 0))
    state.value_probabilities[0, 1] = torch.tensor((0, 0, 1, 0, 0))
    permutation = torch.tensor((1, 0, 2))
    with torch.inference_mode():
        original = reader(query, state)
        permuted = reader(query, _permuted_state(state, permutation))
    torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-6)


def test_addressed_reader_breaks_the_slot_permutation_symmetry() -> None:
    torch.manual_seed(7)
    config = replace(
        _config(),
        reader_slot_addresses=True,
    )
    reader = SourceDeletedQueryReader(config).eval()
    query = torch.randn((2, 5, 16))
    state = _state(committed=1.0, halted=0.0)
    state.value_probabilities[0, 0] = torch.tensor((0, 1, 0, 0, 0))
    state.value_probabilities[0, 1] = torch.tensor((0, 0, 1, 0, 0))
    permutation = torch.tensor((1, 0, 2))
    with torch.inference_mode():
        original = reader(query, state)
        permuted = reader(query, _permuted_state(state, permutation))
    assert not torch.allclose(original, permuted)


def test_native_reader_upgrades_an_unaddressed_warm_start() -> None:
    source = SourceDeletedQueryReader(_config())
    config = replace(
        _config(),
        reader_slot_addresses=True,
    )
    target = NativeCausalDispositionReader(
        config,
        vocab_size=32,
        answer_token_ids=(4, 7, 11, 19),
    )
    initial_addresses = target.reader.slot_embedding.detach().clone()
    target.load_reader_state(source)
    assert target.reader.slot_embedding is not None
    torch.testing.assert_close(
        target.reader.slot_embedding,
        initial_addresses,
    )


def test_motor_hidden_geometry_is_explicit() -> None:
    reader = NativeCausalDispositionReader(
        _config(),
        vocab_size=32,
        answer_token_ids=(4, 7, 11, 19),
    )
    query = torch.randn((2, 5, 16))
    state = _state(committed=1.0, halted=0.0)
    stage_logits = reader.class_logits(query, state)
    late_logits = reader.class_logits(
        query,
        state,
        motor_hidden=query + 1.0,
    )
    assert not torch.equal(stage_logits, late_logits)
    with pytest.raises(TheoryReactorError, match="motor-hidden"):
        reader.class_logits(
            query,
            state,
            motor_hidden=torch.randn((2, 4, 16)),
        )


@pytest.mark.parametrize(
    ("committed", "halted", "expected_class"),
    ((0.0, 1.0, 2), (1.0, 1.0, 3)),
)
def test_terminal_disposition_overrides_truth_motor(
    committed: float,
    halted: float,
    expected_class: int,
) -> None:
    reader = NativeCausalDispositionReader(
        _config(),
        vocab_size=32,
        answer_token_ids=(4, 7, 11, 19),
    )
    with torch.no_grad():
        reader.truth_motor.weight.fill_(100.0)
        reader.truth_motor.bias.copy_(torch.tensor((100.0, -100.0)))
    logits = reader.class_logits(
        torch.randn((2, 5, 16)),
        _state(committed=committed, halted=halted),
        attention_mask=torch.ones((2, 5), dtype=torch.bool),
    )
    assert logits.argmax(-1).eq(expected_class).all()


def test_answer_codebook_must_be_unique_and_in_vocabulary() -> None:
    with pytest.raises(TheoryReactorError, match="codebook"):
        NativeCausalDispositionReader(
            _config(),
            vocab_size=32,
            answer_token_ids=(4, 4, 11, 19),
        )
