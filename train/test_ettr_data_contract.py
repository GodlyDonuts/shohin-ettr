from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from endogenous_typed_theory_reactor import TheoryReactorError
from ettr_data_contract import (
    ETTR_CONTINUATION_SCHEMA,
    ETTRCausalRectangle,
    ETTRContinuationBatch,
    ETTRContinuationManifest,
)
from ettr_episode import ETTREpisodeBatch, ETTREpisodeSegment
from ettr_objectives import (
    ETTRCompositeObjective,
    ETTRObjectiveConfig,
    ETTRObjectiveWeights,
    ETTRPacketTargets,
    ETTRTransactionTargets,
    ETTRVariantAlignment,
)
from test_ettr_episode import _batch, _runner


def _packet(batch: int) -> ETTRPacketTargets:
    active = (
        torch.tensor([[True, True, False, False, False, False]])
        .expand(batch, -1)
        .clone()
    )
    relations = torch.zeros(batch, 3, 6, 6, dtype=torch.bool)
    relations[:, 0, 0, 1] = True
    values = torch.tensor([[3, 4, 0, 0, 0, 0]]).expand(batch, -1).clone()
    if batch == 4:
        values[2:, 0] = 7
    return ETTRPacketTargets(
        value_code=values,
        type_index=torch.tensor([[0, 1, 0, 0, 0, 0]]).expand(batch, -1).clone(),
        relations=relations,
        active=active,
        root=torch.tensor([[True, False, False, False, False, False]])
        .expand(batch, -1)
        .clone(),
        committed=torch.zeros(batch, dtype=torch.bool),
        halted=torch.zeros(batch, dtype=torch.bool),
        slot_mask=torch.ones(batch, 6, dtype=torch.bool),
        relation_mask=torch.ones(
            batch,
            3,
            6,
            6,
            dtype=torch.bool,
        ),
    )


def _terminal_packet(batch: int) -> ETTRPacketTargets:
    assert batch == 4
    packet = _packet(batch)
    values = packet.value_code.clone()
    values[:, 0] = torch.tensor([3, 5, 7, 9])
    return replace(
        packet,
        value_code=values,
        committed=torch.tensor([False, True, False, True]),
        halted=torch.tensor([False, False, True, True]),
    )


def _transactions(batch: int, steps: int = 3) -> ETTRTransactionTargets:
    if batch == 4:
        opcode = torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 6],
                [1, 1, 7],
                [1, 1, 8],
            ]
        )
        values = torch.tensor(
            [
                [3, 3, 3],
                [5, 5, 5],
                [7, 7, 7],
                [9, 9, 9],
            ]
        )
        committed = torch.tensor(
            [
                [False, False, False],
                [False, False, True],
                [False, False, False],
                [False, False, True],
            ]
        )
        halted = torch.tensor(
            [
                [False, False, False],
                [False, False, False],
                [False, False, True],
                [False, False, True],
            ]
        )
    else:
        opcode = torch.tensor([[0, 3, 7]]).expand(batch, -1).clone()
        values = torch.full(
            (batch, steps),
            3,
            dtype=torch.long,
        )
        committed = torch.tensor([[False, False, True]]).expand(
            batch,
            -1,
        ).clone()
        halted = torch.tensor([[False, False, True]]).expand(
            batch,
            -1,
        ).clone()
    return ETTRTransactionTargets(
        opcode=opcode,
        source=torch.zeros(batch, steps, dtype=torch.long),
        target=torch.ones(batch, steps, dtype=torch.long),
        relation=torch.zeros(batch, steps, dtype=torch.long),
        type_index=torch.zeros(batch, steps, dtype=torch.long),
        value_code=values,
        committed=committed,
        halted=halted,
        step_mask=torch.ones(batch, steps, dtype=torch.bool),
    )


def _alignment() -> ETTRVariantAlignment:
    return ETTRVariantAlignment(
        left_index=torch.tensor([0]),
        right_index=torch.tensor([0]),
        slot_permutation=torch.arange(6)[None, :],
        type_permutation=torch.arange(3)[None, :],
        relation_permutation=torch.arange(3)[None, :],
        value_permutation=torch.arange(64)[None, :],
        slot_mask=torch.ones(1, 6, dtype=torch.bool),
        relation_mask=torch.ones(1, 3, 6, 6, dtype=torch.bool),
        step_mask=torch.ones(1, 3, dtype=torch.bool),
    )


def _rectangle_episodes() -> ETTREpisodeBatch:
    return _batch(4)


def _rectangles() -> ETTRCausalRectangle:
    return ETTRCausalRectangle(
        rows=torch.tensor([[[0, 1], [2, 3]]]),
    )


def _execute_interventions(
    runner,
    continuation: ETTRContinuationBatch,
    output,
):
    (
        world_packet,
        world_command,
        _world_target,
        command_packet,
        command_command,
        _command_target,
    ) = continuation.causal_rectangles.intervention_indices()
    return runner.intervene(
        continuation.episodes,
        output.initial_state,
        reactor_steps=3,
        world_packet_index=world_packet,
        world_command_index=world_command,
        command_packet_index=command_packet,
        command_command_index=command_command,
    )


def _continuation() -> tuple[
    ETTRContinuationBatch,
    ETTRObjectiveConfig,
]:
    episodes = _rectangle_episodes()
    objective = ETTRObjectiveConfig(
        vocab_size=64,
        num_slots=6,
        num_types=3,
        num_relations=3,
        num_value_codes=64,
        active_slot_budget=6,
        relation_edge_budget=96,
    )
    return (
        ETTRContinuationBatch(
            manifest_sha256="a" * 64,
            dataset_sha256="b" * 64,
            episodes=episodes,
            packet_targets=_packet(4),
            terminal_packet_targets=_terminal_packet(4),
            causal_rectangles=_rectangles(),
            transaction_targets=_transactions(4),
            initial_committed=torch.zeros(4, dtype=torch.bool),
            initial_halted=torch.zeros(4, dtype=torch.bool),
            equivariance=_alignment(),
        ),
        objective,
    )


def test_continuation_batch_builds_reset_safe_objective() -> None:
    continuation, objective_config = _continuation()
    runner = _runner()
    continuation.validate(runner.model.config, objective_config)
    output = runner(
        continuation.episodes,
        reactor_steps=3,
    )
    interventions = _execute_interventions(runner, continuation, output)
    objective_batch = continuation.objective_batch(output, interventions)
    starts = (
        0,
        continuation.episodes.world.tokens.shape[1],
        continuation.episodes.world.tokens.shape[1]
        + continuation.episodes.command.tokens.shape[1],
    )
    assert objective_batch.token_targets.reset_mask.sum().item() == 12
    assert all(
        bool(objective_batch.token_targets.reset_mask[:, start].all())
        for start in starts
    )
    assert objective_batch.transactions.value_code.shape == (
        4,
        3,
        64,
    )
    assert objective_batch.terminal_packet_prediction is output.terminal_state
    assert (
        objective_batch.world_intervention_prediction
        is interventions.world_terminal_state
    )
    (
        world_packet,
        world_command,
        world_target,
        command_packet,
        command_command,
        command_target,
    ) = continuation.causal_rectangles.intervention_indices()
    for source, target, segment in (
        (world_packet, world_target, continuation.episodes.world),
        (world_command, world_target, continuation.episodes.command),
        (command_packet, command_target, continuation.episodes.world),
        (command_command, command_target, continuation.episodes.command),
    ):
        assert torch.all(source != target)
        assert torch.all(
            segment.tokens.index_select(0, source).ne(
                segment.tokens.index_select(0, target)
            ).any(dim=1)
        )
    for field in fields(continuation.terminal_packet_targets):
        torch.testing.assert_close(
            getattr(objective_batch.world_intervention_targets, field.name),
            getattr(continuation.terminal_packet_targets, field.name).index_select(
                0,
                world_target,
            ),
        )
        torch.testing.assert_close(
            getattr(objective_batch.command_intervention_targets, field.name),
            getattr(continuation.terminal_packet_targets, field.name).index_select(
                0,
                command_target,
            ),
        )
    for field in fields(continuation.transaction_targets):
        torch.testing.assert_close(
            getattr(
                objective_batch.world_intervention_transaction_targets,
                field.name,
            ),
            getattr(continuation.transaction_targets, field.name).index_select(
                0,
                world_target,
            ),
        )
        torch.testing.assert_close(
            getattr(
                objective_batch.command_intervention_transaction_targets,
                field.name,
            ),
            getattr(continuation.transaction_targets, field.name).index_select(
                0,
                command_target,
            ),
        )
    declared = {
        field.name
        for value in (
            continuation,
            continuation.packet_targets,
            continuation.terminal_packet_targets,
            continuation.causal_rectangles,
            continuation.transaction_targets,
        )
        for field in fields(value)
    }
    assert not any("family" in name or "ontology" in name for name in declared)


def test_terminal_packet_loss_connects_compiler_through_reactor() -> None:
    continuation, objective_config = _continuation()
    runner = _runner()
    output = runner(
        continuation.episodes,
        reactor_steps=3,
    )
    interventions = _execute_interventions(runner, continuation, output)
    objective_batch = continuation.objective_batch(output, interventions)
    detached_initial = type(output.initial_state)(
        **{
            field.name: (
                getattr(output.initial_state, field.name)
                if field.name == "step"
                else getattr(output.initial_state, field.name).detach()
            )
            for field in fields(output.initial_state)
        }
    )
    objective_batch = replace(
        objective_batch,
        packet_prediction=detached_initial,
    )
    runner.zero_grad(set_to_none=True)
    loss = ETTRCompositeObjective(
        objective_config,
        weights=ETTRObjectiveWeights(
            token_lm=0.0,
            packet=1.0,
            world_intervention=0.0,
            command_intervention=0.0,
            transaction=0.0,
            equivariance=0.0,
            commit_halt=0.0,
            sparsity=0.0,
            anti_bypass=0.0,
        ),
    )(objective_batch).total
    loss.backward()
    assert any(
        parameter.grad is not None
        and bool(parameter.grad.detach().abs().sum().gt(0))
        for parameter in runner.model.compiler.parameters()
    )
    assert any(
        parameter.grad is not None
        and bool(parameter.grad.detach().abs().sum().gt(0))
        for parameter in runner.model.reactor.parameters()
    )


@pytest.mark.parametrize(
    ("world_weight", "command_weight", "component"),
    (
        (1.0, 0.0, "compiler"),
        (0.0, 1.0, "command"),
    ),
)
def test_causal_arms_reach_their_isolated_input_paths(
    world_weight: float,
    command_weight: float,
    component: str,
) -> None:
    continuation, objective_config = _continuation()
    runner = _runner()
    output = runner(
        continuation.episodes,
        reactor_steps=3,
    )
    interventions = _execute_interventions(runner, continuation, output)
    objective_batch = continuation.objective_batch(output, interventions)
    runner.zero_grad(set_to_none=True)
    loss = ETTRCompositeObjective(
        objective_config,
        weights=ETTRObjectiveWeights(
            token_lm=0.0,
            packet=0.0,
            world_intervention=world_weight,
            command_intervention=command_weight,
            transaction=0.0,
            equivariance=0.0,
            commit_halt=0.0,
            sparsity=0.0,
            anti_bypass=0.0,
        ),
    )(objective_batch).total
    loss.backward()
    parameters = (
        runner.model.compiler.parameters()
        if component == "compiler"
        else runner.model.reactor.command_projection.parameters()
    )
    assert any(
        parameter.grad is not None
        and bool(parameter.grad.detach().abs().sum().gt(0))
        for parameter in parameters
    )


def test_padded_segments_restart_validity_only_at_declared_resets() -> None:
    continuation, objective_config = _continuation()

    def padded(segment: ETTREpisodeSegment, valid: int) -> ETTREpisodeSegment:
        mask = torch.zeros_like(segment.attention_mask, dtype=torch.bool)
        mask[:, :valid] = True
        return ETTREpisodeSegment.from_tokens(
            segment.tokens,
            attention_mask=mask,
        )

    episodes = ETTREpisodeBatch(
        episode_ids=continuation.episodes.episode_ids,
        reset_mask=continuation.episodes.reset_mask,
        world=padded(continuation.episodes.world, 6),
        command=padded(continuation.episodes.command, 4),
        query=padded(continuation.episodes.query, 3),
    )
    continuation = replace(continuation, episodes=episodes)
    runner = _runner()
    continuation.validate(runner.model.config, objective_config)
    output = runner(episodes, reactor_steps=3)
    interventions = _execute_interventions(runner, continuation, output)
    objective_batch = continuation.objective_batch(output, interventions)
    mask = objective_batch.token_targets.mask
    reset = objective_batch.token_targets.reset_mask
    rises = mask[:, 1:] & ~mask[:, :-1]
    assert torch.equal(rises, rises & reset[:, 1:])
    loss = runner.model.base.tok.weight.new_zeros(())
    loss = loss + output.losses.token_lm
    loss = loss + torch.nn.functional.cross_entropy(
        objective_batch.token_logits[:, :-1][
            mask[:, :-1] & mask[:, 1:] & ~reset[:, 1:]
        ],
        objective_batch.token_targets.token_ids[:, 1:][
            mask[:, :-1] & mask[:, 1:] & ~reset[:, 1:]
        ],
    )
    assert torch.isfinite(loss)


def test_continuation_geometry_fails_closed() -> None:
    continuation, objective_config = _continuation()
    runner = _runner()
    with pytest.raises(TheoryReactorError, match="geometry"):
        continuation.validate(
            runner.model.config,
            replace(objective_config, num_types=2),
        )


def test_causal_rectangles_require_factorial_sources_and_consequences() -> None:
    continuation, objective_config = _continuation()
    runner = _runner()
    duplicate = replace(
        continuation.causal_rectangles,
        rows=torch.tensor([[[0, 1], [2, 2]]]),
    )
    with pytest.raises(RuntimeError, match="partition"):
        replace(
            continuation,
            causal_rectangles=duplicate,
        ).validate(runner.model.config, objective_config)
    crossed = replace(
        continuation.causal_rectangles,
        rows=torch.tensor([[[0, 2], [1, 3]]]),
    )
    with pytest.raises(RuntimeError, match="packet target differs"):
        replace(
            continuation,
            causal_rectangles=crossed,
        ).validate(runner.model.config, objective_config)
    repeated_world = continuation.episodes.world.tokens.clone()
    repeated_world[1] = repeated_world[0]
    with pytest.raises(RuntimeError, match="raw renderings are identical"):
        replace(
            continuation,
            episodes=replace(
                continuation.episodes,
                world=ETTREpisodeSegment.from_tokens(repeated_world),
            ),
        ).validate(runner.model.config, objective_config)
    with pytest.raises(RuntimeError, match="no terminal consequence"):
        replace(
            continuation,
            terminal_packet_targets=_packet(4),
        ).validate(runner.model.config, objective_config)


def test_continuation_binds_initial_and_terminal_dispositions() -> None:
    continuation, objective_config = _continuation()
    runner = _runner()
    with pytest.raises(RuntimeError, match="compiler reset state"):
        replace(
            continuation,
            initial_committed=torch.tensor([True, False, False, False]),
        ).validate(runner.model.config, objective_config)
    contradictory = replace(
        continuation.transaction_targets,
        committed=torch.zeros_like(
            continuation.transaction_targets.committed,
        ),
    )
    with pytest.raises(RuntimeError, match="labeled recurrence"):
        replace(
            continuation,
            transaction_targets=contradictory,
        ).validate(runner.model.config, objective_config)
    padded_open_mask = continuation.transaction_targets.step_mask.clone()
    padded_open_mask[0, -1] = False
    with pytest.raises(RuntimeError, match="remains open"):
        replace(
            continuation,
            transaction_targets=replace(
                continuation.transaction_targets,
                step_mask=padded_open_mask,
            ),
        ).validate(runner.model.config, objective_config)
    wrong_terminal = replace(
        continuation.terminal_packet_targets,
        value_code=continuation.packet_targets.value_code,
    )
    with pytest.raises(RuntimeError, match="do not realize"):
        replace(
            continuation,
            terminal_packet_targets=wrong_terminal,
        ).validate(runner.model.config, objective_config)


def test_manifest_fails_closed_on_live_or_overlapping_data() -> None:
    manifest = ETTRContinuationManifest(
        schema=ETTR_CONTINUATION_SCHEMA,
        protected_checkpoint_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        qualification_payload_sha256="c" * 64,
        hybrid_payload_sha256="d" * 64,
        train_rows=100,
        validation_rows=20,
        train_payload_sha256="e" * 64,
        validation_payload_sha256="f" * 64,
        source_deleted=True,
        immutable_snapshot=True,
        live_writer_input=False,
        family_label_fields=(),
    )
    manifest.validate()
    with pytest.raises(TheoryReactorError, match="custody"):
        replace(
            manifest,
            live_writer_input=True,
        ).validate()
    with pytest.raises(TheoryReactorError, match="custody"):
        replace(
            manifest,
            validation_payload_sha256=manifest.train_payload_sha256,
        ).validate()
