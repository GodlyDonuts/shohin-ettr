from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
    TheoryReactorError,
    validate_deployed_state,
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
    return CausalETTREpisodeRunner(model)


def _segment(batch: int, tokens: int) -> ETTREpisodeSegment:
    return ETTREpisodeSegment.from_tokens(torch.randint(0, 64, (batch, tokens)))


def _batch(batch: int = 2) -> ETTREpisodeBatch:
    return ETTREpisodeBatch(
        episode_ids=tuple(f"{index + 1:064x}" for index in range(batch)),
        reset_mask=torch.ones(batch, dtype=torch.bool),
        query_read_index=torch.zeros(batch, dtype=torch.long),
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


def test_segment_rejects_nonbinary_mask_before_boolean_conversion() -> None:
    with pytest.raises(TheoryReactorError, match="binary"):
        ETTREpisodeSegment.from_tokens(
            torch.tensor([[4, 5, 6]]),
            attention_mask=torch.tensor([[1, 2, 0]]),
        )


def test_segment_rejects_rows_without_support_and_forged_targets() -> None:
    with pytest.raises(TheoryReactorError, match="row has no supervised"):
        ETTREpisodeSegment.from_tokens(
            torch.tensor([[4, 5, 6], [7, 0, 0]]),
            attention_mask=torch.tensor([[1, 1, 1], [1, 0, 0]]),
        )
    valid = ETTREpisodeSegment.from_tokens(torch.tensor([[4, 5, 6]]))
    with pytest.raises(TheoryReactorError, match="causal token shift"):
        replace(valid, targets=torch.tensor([[6, 5, -1]])).validate()


def test_episode_requires_explicit_unique_resets() -> None:
    batch = _batch()
    with pytest.raises(TheoryReactorError, match="explicitly reset"):
        replace(
            batch,
            reset_mask=torch.tensor([True, False]),
        ).validate()
    with pytest.raises(TheoryReactorError, match="identity"):
        replace(batch, episode_ids=("a" * 64, "a" * 64)).validate()


def test_episode_query_read_index_fails_closed() -> None:
    batch = _batch()
    with pytest.raises(TheoryReactorError, match="geometry"):
        replace(
            batch,
            query_read_index=torch.zeros(2, dtype=torch.int32),
        ).validate()
    with pytest.raises(TheoryReactorError, match="causal query range"):
        replace(
            batch,
            query_read_index=torch.full((2,), batch.query.tokens.shape[1] - 1),
        ).validate()
    mask = batch.query.attention_mask.clone()
    mask[0, 2:] = False
    with pytest.raises(TheoryReactorError, match="must both be valid"):
        replace(
            batch,
            query_read_index=torch.ones(2, dtype=torch.long),
            query=ETTREpisodeSegment.from_tokens(
                batch.query.tokens,
                attention_mask=mask,
            ),
        ).validate()


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
        episode_ids=("f" * 64,),
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
        query_read_index=batch.query_read_index[1:2],
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


def test_intervention_runner_holds_the_orthogonal_cause_fixed() -> None:
    runner = _runner().eval()
    batch = _batch(batch=2)
    world_packet_index = torch.tensor([1, 0])
    world_command_index = torch.tensor([0, 1])
    command_packet_index = torch.tensor([0, 1])
    command_command_index = torch.tensor([1, 0])
    world_query_index = torch.tensor([1, 0])
    command_query_index = torch.tensor([1, 0])
    with torch.no_grad():
        output = runner(batch, reactor_steps=2)
        intervention = runner.intervene(
            batch,
            output.initial_state,
            reactor_steps=2,
            world_packet_index=world_packet_index,
            world_command_index=world_command_index,
            world_query_index=world_query_index,
            command_packet_index=command_packet_index,
            command_command_index=command_command_index,
            command_query_index=command_query_index,
        )
        swapped_state = type(output.initial_state)(
            value_probabilities=(
                output.initial_state.value_probabilities.index_select(
                    0,
                    world_packet_index,
                )
            ),
            type_probabilities=(
                output.initial_state.type_probabilities.index_select(
                    0,
                    world_packet_index,
                )
            ),
            relations=output.initial_state.relations.index_select(
                0,
                world_packet_index,
            ),
            active=output.initial_state.active.index_select(
                0,
                world_packet_index,
            ),
            root=output.initial_state.root.index_select(
                0,
                world_packet_index,
            ),
            committed=output.initial_state.committed.index_select(
                0,
                world_packet_index,
            ),
            halted=output.initial_state.halted.index_select(
                0,
                world_packet_index,
            ),
            step=output.initial_state.step,
        )
        manual_packet, manual_world_trace = runner.model.execute(
            swapped_state,
            steps=2,
            command_idx=batch.command.tokens.index_select(
                0,
                world_command_index,
            ),
            command_attention_mask=(
                batch.command.attention_mask.index_select(
                    0,
                    world_command_index,
                )
            ),
        )
        command_state = type(output.initial_state)(
            value_probabilities=(
                output.initial_state.value_probabilities.index_select(
                    0,
                    command_packet_index,
                )
            ),
            type_probabilities=(
                output.initial_state.type_probabilities.index_select(
                    0,
                    command_packet_index,
                )
            ),
            relations=output.initial_state.relations.index_select(
                0,
                command_packet_index,
            ),
            active=output.initial_state.active.index_select(
                0,
                command_packet_index,
            ),
            root=output.initial_state.root.index_select(
                0,
                command_packet_index,
            ),
            committed=output.initial_state.committed.index_select(
                0,
                command_packet_index,
            ),
            halted=output.initial_state.halted.index_select(
                0,
                command_packet_index,
            ),
            step=output.initial_state.step,
        )
        manual_command, manual_command_trace = runner.model.execute(
            command_state,
            steps=2,
            command_idx=batch.command.tokens.index_select(
                0,
                command_command_index,
            ),
            command_attention_mask=(
                batch.command.attention_mask.index_select(
                    0,
                    command_command_index,
                )
            ),
        )
        manual_world_query, _ = runner.model.answer_query(
            manual_packet,
            batch.query.tokens.index_select(0, world_query_index),
            targets=None,
            attention_mask=batch.query.attention_mask.index_select(
                0,
                world_query_index,
            ),
        )
        manual_command_query, _ = runner.model.answer_query(
            manual_command,
            batch.query.tokens.index_select(0, command_query_index),
            targets=None,
            attention_mask=batch.query.attention_mask.index_select(
                0,
                command_query_index,
            ),
        )
        manual_world_query = manual_world_query[
            torch.arange(world_query_index.shape[0]),
            batch.query_read_index.index_select(0, world_query_index),
        ]
        manual_command_query = manual_command_query[
            torch.arange(command_query_index.shape[0]),
            batch.query_read_index.index_select(0, command_query_index),
        ]
    for name in (
        "value_probabilities",
        "type_probabilities",
        "relations",
        "active",
        "root",
        "committed",
        "halted",
    ):
        torch.testing.assert_close(
            getattr(intervention.world_terminal_state, name),
            getattr(manual_packet, name),
        )
        torch.testing.assert_close(
            getattr(intervention.command_terminal_state, name),
            getattr(manual_command, name),
        )
    for name in (
        "opcode",
        "source",
        "target",
        "relation",
        "type_index",
        "value_code",
        "committed",
        "halted",
    ):
        torch.testing.assert_close(
            getattr(intervention.world_trace, name),
            getattr(manual_world_trace, name),
        )
        torch.testing.assert_close(
            getattr(intervention.command_trace, name),
            getattr(manual_command_trace, name),
        )
    torch.testing.assert_close(
        intervention.world_query_logits,
        manual_world_query,
    )
    torch.testing.assert_close(
        intervention.command_query_logits,
        manual_command_query,
    )


def test_intervention_query_read_ignores_all_tokens_after_read() -> None:
    runner = _runner().eval()
    query = torch.tensor(
        [
            [9, 10, 20, 30, 31],
            [9, 10, 21, 40, 41],
        ]
    )
    batch = replace(
        _batch(batch=2),
        query_read_index=torch.ones(2, dtype=torch.long),
        query=ETTREpisodeSegment.from_tokens(query),
    )
    mutated_query = query.clone()
    mutated_query[:, 2:] = torch.tensor(
        [[50, 51, 52], [53, 54, 55]]
    )
    mutated = replace(
        batch,
        query=ETTREpisodeSegment.from_tokens(mutated_query),
    )
    index = torch.tensor([1, 0])
    fixed = torch.tensor([0, 1])
    with torch.no_grad():
        output = runner(batch, reactor_steps=2)
        first = runner.intervene(
            batch,
            output.initial_state,
            reactor_steps=2,
            world_packet_index=index,
            world_command_index=fixed,
            world_query_index=index,
            command_packet_index=fixed,
            command_command_index=index,
            command_query_index=index,
        )
        second = runner.intervene(
            mutated,
            output.initial_state,
            reactor_steps=2,
            world_packet_index=index,
            world_command_index=fixed,
            world_query_index=index,
            command_packet_index=fixed,
            command_command_index=index,
            command_query_index=index,
        )
    torch.testing.assert_close(
        first.world_query_logits,
        second.world_query_logits,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        first.command_query_logits,
        second.command_query_logits,
        atol=0,
        rtol=0,
    )


def test_intervention_runner_accepts_indices_but_no_answer_targets() -> None:
    parameters = inspect.signature(
        CausalETTREpisodeRunner.intervene
    ).parameters
    assert "world_query_index" in parameters
    assert "command_query_index" in parameters
    assert not any(
        "answer" in name or "target" in name
        for name in parameters
    )


def test_hard_factual_and_intervention_states_pass_deployment_validation() -> None:
    runner = _runner().eval()
    batch = _batch(batch=2)
    world_packet_index = torch.tensor([1, 0])
    world_command_index = torch.tensor([0, 1])
    command_packet_index = torch.tensor([0, 1])
    command_command_index = torch.tensor([1, 0])
    query_index = torch.tensor([1, 0])
    with torch.no_grad():
        output = runner(
            batch,
            reactor_steps=2,
            hard=True,
            compute_losses=False,
        )
        intervention = runner.intervene(
            batch,
            output.initial_state,
            reactor_steps=2,
            world_packet_index=world_packet_index,
            world_command_index=world_command_index,
            world_query_index=query_index,
            command_packet_index=command_packet_index,
            command_command_index=command_command_index,
            command_query_index=query_index,
            hard=True,
        )
    for state in (
        output.initial_state,
        output.terminal_state,
        intervention.world_terminal_state,
        intervention.command_terminal_state,
    ):
        validate_deployed_state(state, runner.model.config)


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
