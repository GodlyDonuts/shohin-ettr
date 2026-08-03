from pathlib import Path

import pytest

from capability_floor_corpus import (
    CORPUS_SCHEMA,
    INDEX_SCHEMA,
    CapabilityFloorCorpusError,
    EncodedSource,
    TokenizerSpec,
    canonical_rectangle_id,
    query_dependency_strata,
    token_mask_for_spans,
)


def test_cohort_schema_binds_split_aware_jsonl() -> None:
    assert CORPUS_SCHEMA.endswith("-v2")
    assert INDEX_SCHEMA.endswith("-v2")


def test_encoded_source_binds_offsets_and_context() -> None:
    value = EncodedSource(
        token_ids=(10, 11, 12),
        offsets=((0, 0), (0, 4), (4, 8)),
    )
    value.validate(text_length=8, context_limit=3)
    assert len(value.sha256) == 64
    with pytest.raises(CapabilityFloorCorpusError, match="tokenization"):
        value.validate(text_length=8, context_limit=2)


def test_role_spans_map_by_overlap_without_selecting_bos() -> None:
    offsets = ((0, 0), (0, 3), (3, 7), (7, 10))
    assert token_mask_for_spans(offsets, ((2, 6),)) == (
        False,
        True,
        True,
        False,
    )
    with pytest.raises(CapabilityFloorCorpusError, match="no candidate token"):
        token_mask_for_spans(offsets, ((11, 12),))


def test_query_strata_detect_both_causal_axes() -> None:
    answers = ((0, 0), (1, 0), (1, 1), (0, 1))
    assert query_dependency_strata(answers, 0) == ("WORLD", "COMMAND")
    assert query_dependency_strata(answers, 1) == ("WORLD",)


def test_rectangle_identity_binds_query_and_view() -> None:
    first = canonical_rectangle_id(
        core_id="core",
        view_id="view-a",
        query_variant=0,
    )
    assert first != canonical_rectangle_id(
        core_id="core",
        view_id="view-a",
        query_variant=1,
    )
    assert first != canonical_rectangle_id(
        core_id="core",
        view_id="view-b",
        query_variant=0,
    )


def test_tokenizer_spec_requires_explicit_bos_identity() -> None:
    with pytest.raises(CapabilityFloorCorpusError, match="specification"):
        TokenizerSpec(
            candidate="protected-shohin-125m-step300k",
            path=Path("tokenizer.json"),
            source_revision="a" * 64,
            context_limit=2048,
            add_bos=True,
        ).validate()
