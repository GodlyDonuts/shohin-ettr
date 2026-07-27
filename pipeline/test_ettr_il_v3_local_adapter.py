from __future__ import annotations

import hashlib

import pytest

from ettr_il_v2_materialize import (
    MaterializationRequest,
    materialize_ettr_il_v2,
)
from ettr_il_v3_horn_resource import CurriculumStage
from ettr_il_v3_local_adapter import (
    LocalAdapterError,
    adapt_local_rewrite_rectangle,
)
from ettr_il_v3_production import ProductionCell, _candidate_row
from ettr_il_v3_reconstruct import reconstruct_candidate
from ettr_il_v3_rectangles import build_causal_rectangle
from ettr_il_v3_rewrite_episodes import generate_rewrite_episodes


class _ByteTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        return list(text.encode("ascii"))


def _rectangle():
    episode = generate_rewrite_episodes(
        stage=CurriculumStage.DEPENDENT_COMPOSITION,
        theory_index=0,
        depth=2,
        limit=1,
    )[0]
    cell = ProductionCell(
        index=0,
        split="train",
        family="local_rewrite",
        stage=CurriculumStage.DEPENDENT_COMPOSITION.value,
        depth=2,
        selected_quota=1,
        candidate_target=1,
        owner_skip=0,
    )
    candidate = reconstruct_candidate(_candidate_row(cell, episode, ordinal=0))
    return build_causal_rectangle(candidate)


def _sources():
    return (
        ((b"world-00", b"world-01"), (b"world-10", b"world-11")),
        ((b"command-00", b"command-01"), (b"command-10", b"command-11")),
        ((b"query-a= ", b"ask-a= "), (b"query-b= ", b"ask-b= ")),
    )


def test_local_adapter_passes_actual_broad_materializer() -> None:
    world_sources, command_sources, query_prefixes = _sources()
    generic = adapt_local_rewrite_rectangle(
        _rectangle(),
        presentation_id="base",
        world_sources=world_sources,
        command_sources=command_sources,
        query_prefixes=query_prefixes,
        require_query_checkerboard=False,
    )
    batch = materialize_ettr_il_v2(
        MaterializationRequest(
            manifest_sha256=hashlib.sha256(b"manifest").hexdigest(),
            dataset_sha256=hashlib.sha256(b"dataset").hexdigest(),
            vocab_size=256,
            rectangles=(generic,),
            require_query_checkerboard=False,
        ),
        _ByteTokenizer(),
    )
    assert batch.episodes.world.tokens.shape == (16, 192)
    assert batch.episodes.command.tokens.shape == (16, 96)
    assert batch.episodes.query.tokens.shape == (16, 48)
    assert batch.causal_rectangles.rows.shape == (4, 2, 2)


def test_local_adapter_remains_strict_by_default() -> None:
    world_sources, command_sources, query_prefixes = _sources()
    with pytest.raises(LocalAdapterError, match="checkerboard"):
        adapt_local_rewrite_rectangle(
            _rectangle(),
            presentation_id="base",
            world_sources=world_sources,
            command_sources=command_sources,
            query_prefixes=query_prefixes,
        )


def test_local_adapter_rejects_non_boolean_broad_mode() -> None:
    world_sources, command_sources, query_prefixes = _sources()
    with pytest.raises(LocalAdapterError, match="request differs"):
        adapt_local_rewrite_rectangle(
            _rectangle(),
            presentation_id="base",
            world_sources=world_sources,
            command_sources=command_sources,
            query_prefixes=query_prefixes,
            require_query_checkerboard=0,  # type: ignore[arg-type]
        )
