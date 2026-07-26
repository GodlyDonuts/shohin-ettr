from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
    TheoryReactorError,
)
from ettr_episode import (
    CausalETTREpisodeRunner,
    ETTREpisodeBatch,
    ETTREpisodeSegment,
)
from model import GPT, GPTConfig


def _runner() -> CausalETTREpisodeRunner:
    torch.manual_seed(2026072502)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=32,
            d_ff=64,
            seq_len=24,
            zloss=0.0,
        )
    )
    model = EndogenousTypedTheoryReactorGPT(
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
    return CausalETTREpisodeRunner(model)


def _segment(batch: int, tokens: int) -> ETTREpisodeSegment:
    return ETTREpisodeSegment.from_tokens(torch.randint(0, 64, (batch, tokens)))


def _batch(batch: int = 2) -> ETTREpisodeBatch:
    return ETTREpisodeBatch(
        episode_ids=tuple(f"episode-{index}" for index in range(batch)),
        reset_mask=torch.ones(batch, dtype=torch.bool),
        world=_segment(batch, 8),
        command=_segment(batch, 6),
        query=_segment(batch, 5),
    )


def test_segment_targets_never_cross_reset_or_padding() -> None:
    tokens = torch.tensor([[4, 5, 6, 7], [8, 9, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    segment = ETTREpisodeSegment.from_tokens(
        tokens,
        attention_mask=mask,
    )
    assert segment.targets.tolist() == [
        [5, 6, 7, -1],
        [9, -1, -1, -1],
    ]


def test_episode_requires_explicit_unique_resets() -> None:
    batch = _batch()
    with pytest.raises(TheoryReactorError, match="explicitly reset"):
        replace(
            batch,
            reset_mask=torch.tensor([True, False]),
        ).validate()
    with pytest.raises(TheoryReactorError, match="identity"):
        replace(batch, episode_ids=("same", "same")).validate()


def test_complete_episode_has_count_weighted_full_token_loss() -> None:
    runner = _runner()
    batch = _batch()
    output = runner(batch, reactor_steps=3)
    expected_count = sum(
        int(segment.targets.ne(-1).sum())
        for segment in (batch.world, batch.command, batch.query)
    )
    assert int(output.losses.supervised_token_count) == expected_count
    manual = (
        output.losses.world * batch.world.targets.ne(-1).sum()
        + output.losses.command * batch.command.targets.ne(-1).sum()
        + output.losses.query * batch.query.targets.ne(-1).sum()
    ) / expected_count
    torch.testing.assert_close(output.losses.token_lm, manual)
    output.losses.token_lm.backward()
    for parameter in (
        runner.model.compiler.token_projection.weight,
        runner.model.reactor.opcode_head.weight,
        runner.model.query_reader.output_projection.weight,
        runner.model.base.tok.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad)


def test_command_language_model_context_is_reset_from_world() -> None:
    runner = _runner().eval()
    first = _batch(batch=1)
    second = replace(
        first,
        episode_ids=("different-world",),
        world=_segment(1, first.world.tokens.shape[1]),
    )
    with torch.no_grad():
        first_output = runner(first, reactor_steps=2)
        second_output = runner(second, reactor_steps=2)
    torch.testing.assert_close(
        first_output.command_logits,
        second_output.command_logits,
        atol=0,
        rtol=0,
    )
    assert not torch.equal(
        first_output.initial_state.value_probabilities,
        second_output.initial_state.value_probabilities,
    )


def test_query_prefix_is_independent_of_future_query_tokens() -> None:
    runner = _runner().eval()
    batch = _batch(batch=1)
    prefix_tokens = batch.query.tokens[:, :3]
    extended_tokens = torch.cat(
        (prefix_tokens, torch.randint(0, 64, (1, 3))),
        dim=1,
    )
    prefix = replace(
        batch,
        query=ETTREpisodeSegment.from_tokens(prefix_tokens),
    )
    extended = replace(
        batch,
        query=ETTREpisodeSegment.from_tokens(extended_tokens),
    )
    with torch.no_grad():
        prefix_output = runner(prefix, reactor_steps=2)
        extended_output = runner(extended, reactor_steps=2)
    torch.testing.assert_close(
        prefix_output.query_logits,
        extended_output.query_logits[:, : prefix_tokens.shape[1]],
        atol=1e-6,
        rtol=1e-6,
    )


def test_rows_do_not_share_episode_state() -> None:
    runner = _runner().eval()
    batch = _batch(batch=2)
    isolated = ETTREpisodeBatch(
        episode_ids=(batch.episode_ids[1],),
        reset_mask=torch.ones(1, dtype=torch.bool),
        world=ETTREpisodeSegment(
            *(
                value[1:2]
                for value in (
                    batch.world.tokens,
                    batch.world.targets,
                    batch.world.attention_mask,
                )
            )
        ),
        command=ETTREpisodeSegment(
            *(
                value[1:2]
                for value in (
                    batch.command.tokens,
                    batch.command.targets,
                    batch.command.attention_mask,
                )
            )
        ),
        query=ETTREpisodeSegment(
            *(
                value[1:2]
                for value in (
                    batch.query.tokens,
                    batch.query.targets,
                    batch.query.attention_mask,
                )
            )
        ),
    )
    with torch.no_grad():
        together = runner(batch, reactor_steps=2)
        alone = runner(isolated, reactor_steps=2)
    torch.testing.assert_close(
        together.query_logits[1:2],
        alone.query_logits,
        atol=2e-6,
        rtol=2e-6,
    )


def test_validated_episode_core_is_torch_compile_compatible() -> None:
    runner = _runner().eval()
    batch = _batch(batch=1)
    batch.validate()
    compiled = torch.compile(runner, backend="eager")
    with torch.no_grad():
        eager = runner(
            batch,
            reactor_steps=2,
            validate_batch=False,
        )
        traced = compiled(
            batch,
            reactor_steps=2,
            validate_batch=False,
        )
    torch.testing.assert_close(
        eager.query_logits,
        traced.query_logits,
    )
    torch.testing.assert_close(
        eager.losses.token_lm,
        traced.losses.token_lm,
    )
