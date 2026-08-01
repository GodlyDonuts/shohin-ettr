from __future__ import annotations

import pytest
import torch

from endogenous_typed_theory_reactor import (
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
    assert reader.truth_motor.weight.grad is not None
    assert torch.isfinite(reader.truth_motor.weight.grad).all()


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
