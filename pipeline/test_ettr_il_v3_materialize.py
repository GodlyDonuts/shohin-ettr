from __future__ import annotations

from dataclasses import replace

import pytest
from tokenizers import Tokenizer

import ettr_il_v3_materialize as materializer
from ettr_il_v2_token_native_surface import DEFAULT_TOKENIZER_PATH
from ettr_il_v3_horn_resource import (
    CurriculumStage,
    generate_horn_episodes,
    generate_resource_episodes,
)
from ettr_il_v3_materialize import (
    MATERIALIZATION_SCHEMA,
    V3MaterializationError,
    materialize_candidate,
    rematerialize_record,
)
from ettr_il_v3_production import ProductionCell, _candidate_row
from ettr_il_v3_rewrite_episodes import generate_rewrite_episodes
from ettr_il_v3_shards import SemanticCoreRecord


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(DEFAULT_TOKENIZER_PATH))


def _row(family: str):
    stage = CurriculumStage.ATOMIC_TRANSITIONS
    if family == "horn":
        episode = generate_horn_episodes(
            stage=stage,
            theory_index=0,
            limit=1,
        )[0]
    elif family == "resource":
        episode = generate_resource_episodes(
            stage=stage,
            theory_index=0,
            limit=1,
        )[0]
    else:
        episode = generate_rewrite_episodes(
            stage=stage,
            theory_index=0,
            limit=1,
        )[0]
    cell = ProductionCell(
        index=0,
        split="train",
        family=family,
        stage=stage.value,
        depth=1,
        selected_quota=1,
        candidate_target=1,
        owner_skip=0,
    )
    return _candidate_row(cell, episode, ordinal=0)


@pytest.mark.parametrize("family", ("horn", "resource", "local_rewrite"))
def test_materializes_four_views_and_exact_architecture_targets(
    family: str,
    tokenizer: Tokenizer,
) -> None:
    first = materialize_candidate(_row(family), tokenizer)
    second = materialize_candidate(_row(family), tokenizer)
    assert first == second
    assert first.identity.generator_version == MATERIALIZATION_SCHEMA
    assert len(first.source_visible.views) == 4
    assert {view.renderer for view in first.source_visible.views} == {0, 1, 2, 3}
    assert all(len(view.world_sources) == 4 for view in first.source_visible.views)
    assert all(len(view.command_sources) == 4 for view in first.source_visible.views)
    assert all(len(view.query_sources) == 4 for view in first.source_visible.views)
    assert len(first.assessor_only.targets.initial_packets) == 2
    assert len(first.assessor_only.targets.terminal_packets) == 4
    assert len(first.assessor_only.targets.transaction_traces) == 4
    assert len(first.assessor_only.targets.answer_matrix) == 4
    assert SemanticCoreRecord.from_canonical_bytes(first.canonical_bytes()) == first


@pytest.mark.parametrize("family", ("horn", "resource", "local_rewrite"))
def test_stored_record_rematerializes_to_exact_tensor_receipt(
    family: str,
    tokenizer: Tokenizer,
) -> None:
    record = materialize_candidate(_row(family), tokenizer)
    batch = rematerialize_record(record, tokenizer)
    assert batch.episodes.world.tokens.shape == (64, 192)
    assert batch.episodes.command.tokens.shape == (64, 96)
    assert batch.episodes.query.tokens.shape == (64, 48)


def test_rematerializer_rejects_source_and_target_mutation(
    tokenizer: Tokenizer,
) -> None:
    record = materialize_candidate(_row("horn"), tokenizer)
    first_view = record.source_visible.views[0]
    changed_source = replace(
        first_view,
        world_sources=(
            first_view.world_sources[0] + " ",
            *first_view.world_sources[1:],
        ),
    )
    source_mutation = replace(
        record,
        source_visible=replace(
            record.source_visible,
            views=(changed_source, *record.source_visible.views[1:]),
        ),
    )
    with pytest.raises(V3MaterializationError, match="view receipt"):
        rematerialize_record(source_mutation, tokenizer)

    answers = list(record.assessor_only.targets.answer_matrix)
    first_answers = list(answers[0])
    first_answers[0] = not first_answers[0]
    answers[0] = first_answers
    target_mutation = replace(
        record,
        assessor_only=replace(
            record.assessor_only,
            targets=replace(
                record.assessor_only.targets,
                answer_matrix=tuple(answers),
            ),
        ),
    )
    with pytest.raises(
        V3MaterializationError,
        match="materialization receipt",
    ):
        rematerialize_record(target_mutation, tokenizer)


def test_confirmation_requires_separate_sealed_key(tokenizer: Tokenizer) -> None:
    row = _row("resource")
    cell = dict(row["cell"])
    cell["split"] = "confirmation"
    changed = {**row, "cell": cell, "owner": "confirmation"}
    with pytest.raises(V3MaterializationError, match="sealed 32-byte key"):
        materialize_candidate(changed, tokenizer)


@pytest.mark.parametrize(
    ("family", "adapter_name"),
    (
        ("horn", "adapt_horn_semantic_rectangle"),
        ("resource", "adapt_resource_rectangle"),
    ),
)
def test_v3_explicitly_uses_broad_dependency_mode(
    family: str,
    adapter_name: str,
    tokenizer: Tokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = getattr(materializer, adapter_name)
    modes: list[bool] = []

    def checked_adapter(**kwargs: object):
        mode = kwargs.get("require_dependent")
        assert type(mode) is bool
        modes.append(mode)
        return original(**kwargs)

    monkeypatch.setattr(materializer, adapter_name, checked_adapter)
    materialize_candidate(_row(family), tokenizer)
    assert modes == [False] * 4
