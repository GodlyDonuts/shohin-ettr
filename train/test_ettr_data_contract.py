from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from endogenous_typed_theory_reactor import TheoryReactorError
from ettr_data_contract import (
    ETTR_CONTINUATION_SCHEMA,
    ETTRContinuationBatch,
    ETTRContinuationManifest,
)
from ettr_episode import ETTREpisodeBatch, ETTREpisodeSegment
from ettr_objectives import (
    ETTRObjectiveConfig,
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
    return ETTRPacketTargets(
        value_code=torch.tensor([[3, 4, 0, 0, 0, 0]]).expand(batch, -1).clone(),
        type_index=torch.tensor([[0, 1, 0, 0, 0, 0]]).expand(batch, -1).clone(),
        relations=relations,
        active=active,
        root=torch.tensor([[True, False, False, False, False, False]])
        .expand(batch, -1)
        .clone(),
        slot_mask=torch.ones(batch, 6, dtype=torch.bool),
        relation_mask=torch.ones(
            batch,
            3,
            6,
            6,
            dtype=torch.bool,
        ),
    )


def _transactions(batch: int, steps: int = 3) -> ETTRTransactionTargets:
    return ETTRTransactionTargets(
        opcode=torch.tensor([[0, 3, 7]]).expand(batch, -1).clone(),
        source=torch.zeros(batch, steps, dtype=torch.long),
        target=torch.ones(batch, steps, dtype=torch.long),
        relation=torch.zeros(batch, steps, dtype=torch.long),
        type_index=torch.zeros(batch, steps, dtype=torch.long),
        value_code=torch.full(
            (batch, steps),
            3,
            dtype=torch.long,
        ),
        committed=torch.tensor([[False, False, True]]).expand(batch, -1).clone(),
        halted=torch.tensor([[False, False, True]]).expand(batch, -1).clone(),
        step_mask=torch.ones(batch, steps, dtype=torch.bool),
    )


def _alignment() -> ETTRVariantAlignment:
    return ETTRVariantAlignment(
        left_index=torch.tensor([0]),
        right_index=torch.tensor([1]),
        slot_permutation=torch.arange(6)[None, :],
        type_permutation=torch.arange(3)[None, :],
        relation_permutation=torch.arange(3)[None, :],
        value_permutation=torch.arange(64)[None, :],
        slot_mask=torch.ones(1, 6, dtype=torch.bool),
        relation_mask=torch.ones(1, 3, 6, 6, dtype=torch.bool),
        step_mask=torch.ones(1, 3, dtype=torch.bool),
    )


def _continuation() -> tuple[
    ETTRContinuationBatch,
    ETTRObjectiveConfig,
]:
    episodes = _batch(2)
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
            episodes=episodes,
            packet_targets=_packet(2),
            transaction_targets=_transactions(2),
            initial_committed=torch.zeros(2, dtype=torch.bool),
            initial_halted=torch.zeros(2, dtype=torch.bool),
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
    objective_batch = continuation.objective_batch(output)
    starts = (
        0,
        continuation.episodes.world.tokens.shape[1],
        continuation.episodes.world.tokens.shape[1]
        + continuation.episodes.command.tokens.shape[1],
    )
    assert objective_batch.token_targets.reset_mask.sum().item() == 6
    assert all(
        bool(objective_batch.token_targets.reset_mask[:, start].all())
        for start in starts
    )
    assert objective_batch.transactions.value_code.shape == (
        2,
        3,
        64,
    )
    declared = {
        field.name
        for value in (
            continuation,
            continuation.packet_targets,
            continuation.transaction_targets,
        )
        for field in fields(value)
    }
    assert not any("family" in name or "ontology" in name for name in declared)


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
    objective_batch = continuation.objective_batch(output)
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
