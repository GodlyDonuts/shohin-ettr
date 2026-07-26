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
    ETTRQualificationManifest,
    ETTRQualificationReadouts,
    ETTR_QUALIFICATION_MANIFEST_SCHEMA,
    ETTR_QUALIFICATION_SCHEMA,
    _model_sha256,
    _prefix_bytes,
    _score_ettr_qualification,
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


_PRODUCER_MODEL_SHA256 = _model_sha256(_model())


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
            values.append(packet_map(packet) * ROWS_PER_PACKET + variant_map(variant))
    return torch.tensor(values, dtype=torch.long)


def _batch() -> ETTRQualificationBatch:
    state = _state()
    tokens = torch.zeros(BATCH, 5, dtype=torch.long)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    read_index = torch.full((BATCH,), 2, dtype=torch.long)
    targets = torch.empty(BATCH, dtype=torch.long)
    packet_ids = tuple(typed_state_row_sha256(state, row) for row in range(BATCH))
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
    manifest = ETTRQualificationManifest(
        schema=ETTR_QUALIFICATION_MANIFEST_SCHEMA,
        dataset_sha256=_digest("independently-frozen-dataset"),
        producer_model_sha256=_PRODUCER_MODEL_SHA256,
        row_ids=tuple(_digest(f"row-{row}") for row in range(BATCH)),
        packet_ids=packet_ids,
        world_factor_ids=tuple(world_ids),
        command_factor_ids=tuple(command_ids),
        query_semantic_ids=tuple(semantic_ids),
        query_paraphrase_ids=tuple(paraphrase_ids),
        query_prefix_sha256s=tuple(
            hashlib.sha256(_prefix_bytes(tokens, mask, read_index, row)).hexdigest()
            for row in range(BATCH)
        ),
        target_token_ids=tuple(int(value) for value in targets),
    )
    return ETTRQualificationBatch(
        terminal_state=state,
        manifest=manifest,
        query_tokens=tokens,
        query_attention_mask=mask,
        query_read_index=read_index,
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


def _harness(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRQualificationBatch,
) -> ETTRQualificationHarness:
    return ETTRQualificationHarness(
        model,
        expected_manifest_sha256=batch.manifest.sha256(),
        expected_model_sha256=_model_sha256(model),
    )


def _readouts(
    model: EndogenousTypedTheoryReactorGPT,
    batch: ETTRQualificationBatch,
) -> ETTRQualificationReadouts:
    return _harness(model, batch)._collect_readouts(batch)


def _rebind_manifest(
    batch: ETTRQualificationBatch,
) -> ETTRQualificationBatch:
    manifest = replace(
        batch.manifest,
        dataset_sha256=hashlib.sha256(
            batch.query_tokens.numpy().tobytes() + batch.targets.numpy().tobytes()
        ).hexdigest(),
        packet_ids=batch.packet_ids,
        world_factor_ids=batch.world_factor_ids,
        command_factor_ids=batch.command_factor_ids,
        query_semantic_ids=batch.query_semantic_ids,
        query_paraphrase_ids=batch.query_paraphrase_ids,
        query_prefix_sha256s=tuple(
            hashlib.sha256(
                _prefix_bytes(
                    batch.query_tokens,
                    batch.query_attention_mask,
                    batch.query_read_index,
                    row,
                )
            ).hexdigest()
            for row in range(BATCH)
        ),
        target_token_ids=tuple(int(value) for value in batch.targets),
    )
    return replace(batch, manifest=manifest)


def test_batch_freezes_factorial_state_and_query_twin_geometry() -> None:
    model = _model()
    batch = _batch()
    batch.validate(model.config, vocab_size=VOCAB)
    assert len(set(batch.packet_ids)) == PACKETS
    assert all(
        batch.packet_ids[row] == typed_state_row_sha256(batch.terminal_state, row)
        for row in range(BATCH)
    )
    changed_state = replace(
        batch.terminal_state,
        halted=batch.terminal_state.halted.clone(),
    )
    changed_state.halted[0] = 1
    assert (
        replace(
            batch,
            terminal_state=changed_state,
        ).sha256()
        != batch.sha256()
    )


def test_preregistered_manifest_rejects_semantic_axis_relabeling() -> None:
    model = _model()
    batch = _batch()
    relabeled_world = tuple(
        _digest(f"relabeled-world-{value}") for value in batch.world_factor_ids
    )
    relabeled_command = tuple(
        _digest(f"relabeled-command-{value}") for value in batch.command_factor_ids
    )
    relabeled_manifest = replace(
        batch.manifest,
        world_factor_ids=relabeled_world,
        command_factor_ids=relabeled_command,
    )
    relabeled = replace(
        batch,
        manifest=relabeled_manifest,
        world_factor_ids=relabeled_world,
        command_factor_ids=relabeled_command,
    )
    harness = ETTRQualificationHarness(
        model,
        expected_manifest_sha256=batch.manifest.sha256(),
        expected_model_sha256=_model_sha256(model),
    )
    with pytest.raises(
        TheoryReactorError,
        match="preregistered manifest",
    ):
        harness.evaluate(relabeled)


def test_preregistered_model_rejects_weight_substitution() -> None:
    model = _model()
    batch = _batch()
    expected_model_sha256 = _model_sha256(model)
    with torch.no_grad():
        next(model.parameters()).add_(0.25)
    harness = ETTRQualificationHarness(
        model,
        expected_manifest_sha256=batch.manifest.sha256(),
        expected_model_sha256=expected_model_sha256,
    )
    with pytest.raises(
        TheoryReactorError,
        match="preregistered model",
    ):
        harness.evaluate(batch)


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
    with pytest.raises(TheoryReactorError, match="frozen manifest"):
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

    target_collision = batch.query_tokens.clone()
    collision_targets = batch.targets.clone()
    for packet in range(PACKETS):
        command = packet % 2
        for variant in range(ROWS_PER_PACKET):
            semantic = variant // 2
            row = packet * ROWS_PER_PACKET + variant
            target = 8 + command * 2 + semantic
            target_collision[row, 3] = target
            collision_targets[row] = target
    with pytest.raises(
        TheoryReactorError,
        match="target-changing packet state",
    ):
        _rebind_manifest(
            replace(
                batch,
                query_tokens=target_collision,
                targets=collision_targets,
            )
        ).validate(model.config, vocab_size=VOCAB)


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


def test_harness_runs_every_control_without_passing_targets() -> None:
    model = _model()
    batch = _batch()
    readouts = _readouts(model, batch)
    readouts.validate(rows=BATCH, vocab_size=VOCAB)
    source = inspect.getsource(ETTRQualificationHarness._read)
    assert "targets=None" in source
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
    exact = _readouts(model, batch)
    poisoned_tokens = batch.query_tokens.clone()
    poisoned_targets = (batch.targets + 17) % VOCAB
    row = torch.arange(BATCH)
    poisoned_tokens[row, batch.query_read_index + 1] = poisoned_targets
    poisoned_tokens[:, -1] = (poisoned_tokens[:, -1] + 9) % VOCAB
    poisoned = _rebind_manifest(
        replace(
            batch,
            query_tokens=poisoned_tokens,
            targets=poisoned_targets,
        )
    )
    altered = _readouts(model, poisoned)
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
    result = _harness(model, batch).evaluate(batch)
    score = result.score
    assert not hasattr(result, "treatment")
    assert result.manifest_sha256 == batch.manifest.sha256()
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
    good = _readouts(_model(), batch)
    nonfinite = good.treatment.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(TheoryReactorError, match="treatment"):
        _score_ettr_qualification(
            batch,
            replace(good, treatment=nonfinite),
        )
    with pytest.raises(TheoryReactorError, match="query_only"):
        _score_ettr_qualification(
            batch,
            replace(good, query_only=good.query_only[:-1]),
        )
    forged_logits = good.treatment.clone()
    forged_logits.fill_(-100)
    forged_logits[
        torch.arange(BATCH),
        batch.targets,
    ] = 100
    with pytest.raises(TheoryReactorError, match="sealed readout"):
        _score_ettr_qualification(
            batch,
            replace(good, treatment=forged_logits),
        )


def test_score_rejects_batch_reassociation_and_target_mutation() -> None:
    batch = _batch()
    readouts = _readouts(_model(), batch)
    altered_tokens = batch.query_tokens.clone()
    altered_tokens[:, -1] = (altered_tokens[:, -1] + 1) % VOCAB
    with pytest.raises(TheoryReactorError, match="another batch"):
        _score_ettr_qualification(
            replace(batch, query_tokens=altered_tokens),
            readouts,
        )

    wrong_targets = readouts.targets.clone()
    wrong_targets[0] = (wrong_targets[0] + 1) % VOCAB
    forged_targets = replace(
        readouts,
        targets=wrong_targets,
        readout_sha256="0" * 64,
    )
    forged_targets = replace(
        forged_targets,
        readout_sha256=forged_targets.computed_sha256(),
    )
    with pytest.raises(TheoryReactorError, match="targets changed"):
        _score_ettr_qualification(
            batch,
            forged_targets,
        )

    wrong_twin_targets = readouts.query_twin_targets.clone()
    wrong_twin_targets[0] = (wrong_twin_targets[0] + 1) % VOCAB
    forged_twin_targets = replace(
        readouts,
        query_twin_targets=wrong_twin_targets,
        readout_sha256="0" * 64,
    )
    forged_twin_targets = replace(
        forged_twin_targets,
        readout_sha256=forged_twin_targets.computed_sha256(),
    )
    with pytest.raises(
        TheoryReactorError,
        match="query twin targets changed",
    ):
        _score_ettr_qualification(
            batch,
            forged_twin_targets,
        )


def test_candidate_method_or_hook_override_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    original = model.answer_query

    def mutate_input(state, tokens, *args, **kwargs):
        tokens[0, 0] = (tokens[0, 0] + 1) % VOCAB
        return original(state, tokens, *args, **kwargs)

    monkeypatch.setattr(model, "answer_query", mutate_input)
    with pytest.raises(TheoryReactorError, match="method answer_query"):
        _readouts(model, _batch())

    model = _model()
    reader_forward = model.query_reader.forward
    monkeypatch.setattr(
        model.query_reader,
        "forward",
        lambda *args, **kwargs: reader_forward(*args, **kwargs),
    )
    with pytest.raises(TheoryReactorError, match="child forward override"):
        _readouts(model, _batch())

    model = _model()
    handle = model.query_reader.register_forward_hook(
        lambda _module, _inputs, output: output
    )
    with pytest.raises(TheoryReactorError, match="hooks are forbidden"):
        _readouts(model, _batch())
    handle.remove()


def test_preregistered_model_binds_child_module_implementation() -> None:
    model = _model()
    batch = _batch()
    expected_model_sha256 = _model_sha256(model)
    original_reader_class = type(model.query_reader)

    class ReplacementReader(original_reader_class):
        pass

    model.query_reader.__class__ = ReplacementReader
    harness = ETTRQualificationHarness(
        model,
        expected_manifest_sha256=batch.manifest.sha256(),
        expected_model_sha256=expected_model_sha256,
    )
    with pytest.raises(
        TheoryReactorError,
        match="preregistered model",
    ):
        harness.evaluate(batch)


def test_train_mode_and_candidate_side_channels_fail_closed() -> None:
    model = _model().train()
    with pytest.raises(TheoryReactorError, match="eval mode"):
        _readouts(model, _batch())

    fields_seen = {item.name for item in fields(ETTRQualificationBatch)}
    assert not fields_seen & {
        "world_tokens",
        "command_tokens",
        "trace",
        "oracle",
        "assessor",
        "source",
    }
    source = inspect.getsource(__import__("ettr_qualification"))
    assert "cross_ontology_" not in source
    assert "execute_sequence" not in source
    assert "execute_closure" not in source
    assert "one_step_reducts" not in source


def test_readout_validation_rejects_targets_with_wrong_dtype() -> None:
    readouts = _readouts(_model(), _batch())
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
