from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect

import pytest
import torch
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    TheoryReactorConfig,
    TheoryReactorError,
    TypedTheoryState,
)
from ettr_qualification import (
    ETTRQualificationBatch,
    ETTRQualificationHarness,
    ETTRQualificationReadouts,
    ETTR_QUALIFICATION_SCHEMA,
    score_ettr_qualification,
    typed_state_row_sha256,
)
from model import GPT, GPTConfig


ROWS_PER_PACKET = 4
PACKETS = 4
BATCH = ROWS_PER_PACKET * PACKETS
VOCAB = 64


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _model() -> EndogenousTypedTheoryReactorGPT:
    torch.manual_seed(2026072601)
    base = GPT(
        GPTConfig(
            vocab_size=VOCAB,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=32,
            d_ff=64,
            seq_len=8,
            zloss=0.0,
        )
    )
    model = EndogenousTypedTheoryReactorGPT(
        base,
        TheoryReactorConfig(
            d_model=32,
            state_width=32,
            num_slots=2,
            num_types=2,
            num_relations=1,
            num_value_codes=8,
            max_edges=4,
            num_heads=4,
            compiler_layers=1,
            reactor_layers=1,
            query_layers=1,
            ff_multiplier=2,
            max_steps=4,
            stage_after_block=1,
            parameter_cap=1_000_000,
        ),
    )
    return model.eval()


def _state() -> TypedTheoryState:
    packet = torch.arange(PACKETS).repeat_interleave(ROWS_PER_PACKET)
    active = torch.ones(BATCH, 2)
    value = torch.zeros(BATCH, 2, 8)
    value[:, 0] = F.one_hot(packet + 1, 8).float()
    value[:, 1] = F.one_hot((packet % 2) + 5, 8).float()
    types = torch.zeros(BATCH, 2, 2)
    types[:, 0, 0] = 1
    types[:, 1, 1] = 1
    relations = torch.zeros(BATCH, 1, 2, 2)
    relations[packet.remainder(2).bool(), 0, 0, 1] = 1
    root = torch.zeros(BATCH, 2)
    root[:, 0] = 1
    return TypedTheoryState(
        value_probabilities=value,
        type_probabilities=types,
        relations=relations,
        active=active,
        root=root,
        committed=torch.ones(BATCH),
        halted=torch.zeros(BATCH),
        step=2,
    )


def _index(
    packet_map,
    variant_map=lambda value: value,
) -> torch.Tensor:
    values = []
    for packet in range(PACKETS):
        for variant in range(ROWS_PER_PACKET):
            values.append(
                packet_map(packet) * ROWS_PER_PACKET
                + variant_map(variant)
            )
    return torch.tensor(values, dtype=torch.long)


def _batch() -> ETTRQualificationBatch:
    state = _state()
    tokens = torch.zeros(BATCH, 5, dtype=torch.long)
    targets = torch.empty(BATCH, dtype=torch.long)
    packet_ids = tuple(
        typed_state_row_sha256(state, row)
        for row in range(BATCH)
    )
    world_ids = []
    command_ids = []
    semantic_ids = []
    paraphrase_ids = []
    for packet in range(PACKETS):
        world = packet // 2
        command = packet % 2
        for variant in range(ROWS_PER_PACKET):
            semantic = variant // 2
            paraphrase = variant % 2
            row = packet * ROWS_PER_PACKET + variant
            target = 8 + packet * 2 + semantic
            tokens[row] = torch.tensor(
                [
                    1,
                    20 + semantic,
                    30 + paraphrase,
                    target,
                    50,
                ]
            )
            targets[row] = target
            world_ids.append(_digest(f"world-{world}"))
            command_ids.append(_digest(f"command-{command}"))
            semantic_ids.append(_digest(f"semantic-{semantic}"))
            paraphrase_ids.append(_digest(f"paraphrase-{paraphrase}"))
    return ETTRQualificationBatch(
        terminal_state=state,
        query_tokens=tokens,
        query_attention_mask=torch.ones_like(tokens, dtype=torch.bool),
        query_read_index=torch.full((BATCH,), 2, dtype=torch.long),
        targets=targets,
        packet_ids=packet_ids,
        world_factor_ids=tuple(world_ids),
        command_factor_ids=tuple(command_ids),
        query_semantic_ids=tuple(semantic_ids),
        query_paraphrase_ids=tuple(paraphrase_ids),
        shuffled_state_index=_index(lambda packet: packet ^ 3),
        wrong_world_state_index=_index(lambda packet: packet ^ 2),
        wrong_command_state_index=_index(lambda packet: packet ^ 1),
        query_twin_index=_index(
            lambda packet: packet,
            lambda variant: variant ^ 2,
        ),
        target_derangement_index=_index(
            lambda packet: packet,
            lambda variant: variant ^ 2,
        ),
    )


def test_batch_freezes_factorial_state_and_query_twin_geometry() -> None:
    model = _model()
    batch = _batch()
    batch.validate(model.config, vocab_size=VOCAB)
    assert len(set(batch.packet_ids)) == PACKETS
    assert all(
        batch.packet_ids[row]
        == typed_state_row_sha256(batch.terminal_state, row)
        for row in range(BATCH)
    )
    changed_state = replace(
        batch.terminal_state,
        halted=batch.terminal_state.halted.clone(),
    )
    changed_state.halted[0] = 1
    assert replace(
        batch,
        terminal_state=changed_state,
    ).sha256() != batch.sha256()


@pytest.mark.parametrize(
    "mutation, message",
    (
        (
            lambda batch: replace(
                batch,
                packet_ids=(_digest("forged"),) + batch.packet_ids[1:],
            ),
            "packet identities",
        ),
        (
            lambda batch: replace(
                batch,
                shuffled_state_index=torch.arange(BATCH),
            ),
            "derangement",
        ),
        (
            lambda batch: replace(
                batch,
                wrong_world_state_index=batch.wrong_command_state_index,
            ),
            "wrong WORLD",
        ),
        (
            lambda batch: replace(
                batch,
                query_twin_index=_index(
                    lambda packet: packet,
                    lambda variant: variant ^ 1,
                ),
            ),
            "query twin",
        ),
        (
            lambda batch: replace(
                batch,
                target_derangement_index=torch.arange(BATCH),
            ),
            "derangement",
        ),
    ),
)
def test_control_identity_attacks_fail_closed(mutation, message: str) -> None:
    model = _model()
    with pytest.raises(TheoryReactorError, match=message):
        mutation(_batch()).validate(model.config, vocab_size=VOCAB)


def test_query_paraphrase_and_target_attacks_fail_closed() -> None:
    model = _model()
    batch = _batch()
    duplicate = batch.query_tokens.clone()
    duplicate[1, :3] = duplicate[0, :3]
    with pytest.raises(TheoryReactorError, match="paraphrase twin"):
        replace(batch, query_tokens=duplicate).validate(
            model.config,
            vocab_size=VOCAB,
        )

    wrong_target = batch.targets.clone()
    wrong_target[0] = (wrong_target[0] + 1) % VOCAB
    with pytest.raises(TheoryReactorError, match="causal next token"):
        replace(batch, targets=wrong_target).validate(
            model.config,
            vocab_size=VOCAB,
        )


def test_soft_or_incomplete_packet_is_not_qualification_admissible() -> None:
    model = _model()
    batch = _batch()
    soft = replace(
        batch.terminal_state,
        active=batch.terminal_state.active.clone(),
    )
    soft.active[0, 0] = 0.5
    with pytest.raises(TheoryReactorError, match="not binary"):
        replace(batch, terminal_state=soft).validate(
            model.config,
            vocab_size=VOCAB,
        )


def test_harness_runs_every_control_without_passing_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    batch = _batch()
    observed_targets = []
    original = model.answer_query

    def wrapped(*args, **kwargs):
        observed_targets.append(kwargs.get("targets"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "answer_query", wrapped)
    readouts = ETTRQualificationHarness(model).run(batch)
    readouts.validate(rows=BATCH, vocab_size=VOCAB)
    assert observed_targets
    assert observed_targets == [None] * len(observed_targets)
    assert readouts.treatment.shape == (BATCH, VOCAB)
    assert torch.equal(
        readouts.query_twin_targets,
        batch.targets.index_select(0, batch.query_twin_index),
    )
    assert torch.equal(
        readouts.deranged_targets,
        batch.targets.index_select(
            0,
            batch.target_derangement_index,
        ),
    )


def test_answer_and_suffix_tokens_are_physically_absent_from_readout() -> None:
    model = _model()
    batch = _batch()
    exact = ETTRQualificationHarness(model).run(batch)
    poisoned_tokens = batch.query_tokens.clone()
    poisoned_targets = (batch.targets + 17) % VOCAB
    row = torch.arange(BATCH)
    poisoned_tokens[row, batch.query_read_index + 1] = poisoned_targets
    poisoned_tokens[:, -1] = (poisoned_tokens[:, -1] + 9) % VOCAB
    poisoned = replace(
        batch,
        query_tokens=poisoned_tokens,
        targets=poisoned_targets,
    )
    altered = ETTRQualificationHarness(model).run(poisoned)
    for name in (
        "treatment",
        "query_only",
        "zero_reader",
        "shuffled_state",
        "wrong_world_state",
        "wrong_command_state",
        "query_twin",
    ):
        assert torch.equal(getattr(exact, name), getattr(altered, name))


def test_score_is_post_forward_and_reports_all_controls() -> None:
    model = _model()
    batch = _batch()
    readouts = ETTRQualificationHarness(model).run(batch)
    score = score_ettr_qualification(batch, readouts)
    assert score.schema == ETTR_QUALIFICATION_SCHEMA
    assert score.rows == BATCH
    assert score.packet_groups == PACKETS
    for item in fields(score):
        value = getattr(score, item.name)
        if item.name not in {"schema", "rows", "packet_groups"}:
            assert 0 <= value <= BATCH
    assert 0 <= score.strongest_negative_control_exact <= BATCH


def test_score_rejects_nonfinite_or_wrong_shape_readouts() -> None:
    batch = _batch()
    good = ETTRQualificationHarness(_model()).run(batch)
    nonfinite = good.treatment.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(TheoryReactorError, match="treatment"):
        score_ettr_qualification(
            batch,
            replace(good, treatment=nonfinite),
        )
    with pytest.raises(TheoryReactorError, match="query_only"):
        score_ettr_qualification(
            batch,
            replace(good, query_only=good.query_only[:-1]),
        )


def test_score_rejects_batch_reassociation_and_target_mutation() -> None:
    batch = _batch()
    readouts = ETTRQualificationHarness(_model()).run(batch)
    altered_tokens = batch.query_tokens.clone()
    altered_tokens[:, -1] = (altered_tokens[:, -1] + 1) % VOCAB
    with pytest.raises(TheoryReactorError, match="another batch"):
        score_ettr_qualification(
            replace(batch, query_tokens=altered_tokens),
            readouts,
        )

    wrong_targets = readouts.targets.clone()
    wrong_targets[0] = (wrong_targets[0] + 1) % VOCAB
    with pytest.raises(TheoryReactorError, match="targets changed"):
        score_ettr_qualification(
            batch,
            replace(readouts, targets=wrong_targets),
        )

    wrong_twin_targets = readouts.query_twin_targets.clone()
    wrong_twin_targets[0] = (wrong_twin_targets[0] + 1) % VOCAB
    with pytest.raises(
        TheoryReactorError,
        match="query twin targets changed",
    ):
        score_ettr_qualification(
            batch,
            replace(
                readouts,
                query_twin_targets=wrong_twin_targets,
            ),
        )


def test_train_mode_and_candidate_side_channels_fail_closed() -> None:
    model = _model().train()
    with pytest.raises(TheoryReactorError, match="eval mode"):
        ETTRQualificationHarness(model).run(_batch())

    fields_seen = {item.name for item in fields(ETTRQualificationBatch)}
    assert not fields_seen & {
        "world_tokens",
        "command_tokens",
        "trace",
        "oracle",
        "assessor",
        "source",
    }
    source = inspect.getsource(
        __import__("ettr_qualification")
    )
    assert "cross_ontology_" not in source
    assert "execute_sequence" not in source
    assert "execute_closure" not in source
    assert "one_step_reducts" not in source


def test_readout_validation_rejects_targets_with_wrong_dtype() -> None:
    readouts = ETTRQualificationHarness(_model()).run(_batch())
    forged = ETTRQualificationReadouts(
        **{
            item.name: (
                readouts.targets.float()
                if item.name == "targets"
                else getattr(readouts, item.name)
            )
            for item in fields(readouts)
        }
    )
    with pytest.raises(TheoryReactorError, match="targets"):
        forged.validate(rows=BATCH, vocab_size=VOCAB)
