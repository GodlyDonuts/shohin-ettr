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
    assert (
        sum(parameter.numel() for parameter in reader.truth_motor.parameters())
        == 16 * 64 + 64 + 64 * 2 + 2
    )
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


def test_temporal_reader_requires_and_uses_initial_state() -> None:
    torch.manual_seed(11)
    config = replace(
        _config(),
        reader_slot_addresses=True,
        reader_initial_state=True,
    )
    reader = SourceDeletedQueryReader(config).eval()
    query = torch.randn((2, 5, 16))
    terminal = _state(committed=1.0, halted=0.0)
    unchanged = _state(committed=0.0, halted=0.0)
    changed = unchanged.detached_clone()
    changed.value_probabilities[0, 0] = torch.tensor((0, 1, 0, 0, 0))
    with pytest.raises(TheoryReactorError, match="initial state is required"):
        reader(query, terminal)
    with torch.inference_mode():
        unchanged_read = reader(query, terminal, initial_state=unchanged)
        changed_read = reader(query, terminal, initial_state=changed)
    assert not torch.allclose(unchanged_read, changed_read)


def test_temporal_reader_requires_slot_addresses() -> None:
    with pytest.raises(TheoryReactorError, match="requires slot addresses"):
        SourceDeletedQueryReader(replace(_config(), reader_initial_state=True))


def test_native_reader_upgrades_to_two_snapshot_state_memory() -> None:
    source = SourceDeletedQueryReader(_config())
    config = replace(
        _config(),
        reader_slot_addresses=True,
        reader_initial_state=True,
    )
    target = NativeCausalDispositionReader(
        config,
        vocab_size=32,
        answer_token_ids=(4, 7, 11, 19),
    )
    initial_addresses = target.reader.slot_embedding.detach().clone()
    initial_phases = target.reader.state_phase_embedding.detach().clone()
    target.load_reader_state(source)
    torch.testing.assert_close(target.reader.slot_embedding, initial_addresses)
    torch.testing.assert_close(
        target.reader.state_phase_embedding,
        initial_phases,
    )


def test_temporal_upgrade_preserves_every_common_initial_parameter() -> None:
    addressed_config = replace(_config(), reader_slot_addresses=True)
    temporal_config = replace(
        addressed_config,
        reader_initial_state=True,
    )
    torch.manual_seed(19)
    addressed = NativeCausalDispositionReader(
        addressed_config,
        vocab_size=32,
        answer_token_ids=(4, 7, 11, 19),
    )
    torch.manual_seed(19)
    temporal = NativeCausalDispositionReader(
        temporal_config,
        vocab_size=32,
        answer_token_ids=(4, 7, 11, 19),
    )
    addressed_state = addressed.state_dict()
    temporal_state = temporal.state_dict()
    assert set(temporal_state) - set(addressed_state) == {
        "reader.state_phase_embedding"
    }
    for name, value in addressed_state.items():
        torch.testing.assert_close(value, temporal_state[name])


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


def test_state_only_motor_cannot_use_the_pretrained_query_bypass() -> None:
    reader = NativeCausalDispositionReader(
        _config(),
        vocab_size=32,
        answer_token_ids=(4, 7, 11, 19),
        state_only_motor=True,
    ).eval()
    query = torch.randn((2, 5, 16))
    state = _state(committed=1.0, halted=0.0)
    with torch.inference_mode():
        first = reader.class_logits(query, state, motor_hidden=query)
        second = reader.class_logits(query, state, motor_hidden=query + 100.0)
    torch.testing.assert_close(first, second)


def test_direct_reader_output_bypasses_the_stuck_scalar_gate() -> None:
    torch.manual_seed(23)
    legacy = SourceDeletedQueryReader(_config()).eval()
    torch.manual_seed(23)
    direct = SourceDeletedQueryReader(
        replace(_config(), reader_direct_output=True)
    ).eval()
    direct.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        legacy.gate.zero_()
        direct.gate.zero_()
    query = torch.randn((2, 5, 16))
    state = _state(committed=1.0, halted=0.0)
    with torch.inference_mode():
        legacy_read = legacy(query, state)
        direct_read = direct(query, state)
    assert torch.count_nonzero(legacy_read) == 0
    assert torch.count_nonzero(direct_read) > 0


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
