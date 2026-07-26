from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from cross_ontology_horn_board import GroundAtom
from cross_ontology_resource_board import Marking
from cross_ontology_rewrite_board import GroundTerm
from ettr_il_v2_semantics import (
    HornCommand,
    HornPolicy,
    HornWorld,
    QueryOp,
    ResourceCommand,
    ResourcePolicy,
    ResourceWorld,
    RewriteCommand,
    RewritePolicy,
    RewriteWorld,
    SemanticQuery,
)
from ettr_il_v2_surface import (
    SurfaceRenderer,
    render_surface,
)
from ettr_il_v2_surface_adapter import (
    IMPLEMENTED_PRESENTATIONS,
    SemanticSurfaceDocument,
    SurfaceAdapterContext,
    SurfaceAdapterError,
    SurfaceStage,
    build_base_surface_bundle,
    canonical_factor_ast,
    parse_and_semantic_canonicalize,
)


EVIDENCE = "0" * 64
FORBIDDEN = (
    b"horn",
    b"rewrite",
    b"resource",
    b"ontology",
    b"theory",
    b"oracle",
    b"target",
    b"answer",
)


def _prf(label: str, context: bytes) -> bytes:
    return hashlib.sha256(label.encode("ascii") + b"\0" + context).digest()


def _context(
    renderer: SurfaceRenderer,
    *,
    presentation: str = "base",
    rectangle_id: str = "rectangle-0",
) -> SurfaceAdapterContext:
    return SurfaceAdapterContext(
        fold=1,
        split="development",
        semantic_core_id="core-0",
        semantic_rectangle_id=rectangle_id,
        renderer=renderer,
        prf=_prf,
        presentation=presentation,
    )


def _horn_factors() -> tuple[
    tuple[HornWorld, HornWorld],
    tuple[HornCommand, HornCommand],
    tuple[SemanticQuery, SemanticQuery],
]:
    return (
        (
            HornWorld(
                EVIDENCE,
                0,
                (GroundAtom(0, (0,)),),
                HornPolicy.PERSISTENT,
            ),
            HornWorld(
                EVIDENCE,
                1,
                (GroundAtom(0, (1,)),),
                HornPolicy.DERIVED_ONLY,
            ),
        ),
        (
            HornCommand(1, (GroundAtom(0, (1,)),)),
            HornCommand(2, (GroundAtom(0, (0,)), GroundAtom(3, (0, 3)))),
        ),
        (
            SemanticQuery(QueryOp.HORN_HAS, (0, 0)),
            SemanticQuery(QueryOp.HORN_COUNT_GE, (2,)),
        ),
    )


def _rewrite_factors() -> tuple[
    tuple[RewriteWorld, RewriteWorld],
    tuple[RewriteCommand, RewriteCommand],
    tuple[SemanticQuery, SemanticQuery],
]:
    return (
        (
            RewriteWorld(
                EVIDENCE,
                0,
                GroundTerm(0, 0),
                RewritePolicy.CONTEXTUAL,
            ),
            RewriteWorld(
                EVIDENCE,
                4,
                GroundTerm(0, 4, (GroundTerm(0, 1),)),
                RewritePolicy.ROOT_ONLY,
            ),
        ),
        (
            RewriteCommand(1, (0,)),
            RewriteCommand(3, (1, 0, 1)),
        ),
        (
            SemanticQuery(QueryOp.REWRITE_ROOT_IS, (0,)),
            SemanticQuery(QueryOp.REWRITE_NODES_GE, (2,)),
        ),
    )


def _resource_factors() -> tuple[
    tuple[ResourceWorld, ResourceWorld],
    tuple[ResourceCommand, ResourceCommand],
    tuple[SemanticQuery, SemanticQuery],
]:
    return (
        (
            ResourceWorld(
                EVIDENCE,
                0,
                Marking((1, 0, 1, 0)),
                ResourcePolicy.ATOMIC_DEADLOCK,
            ),
            ResourceWorld(
                EVIDENCE,
                7,
                Marking((0, 2, 1, 0)),
                ResourcePolicy.SKIP_BLOCKED,
            ),
        ),
        (
            ResourceCommand(1, (0,)),
            ResourceCommand(3, (2, 1, 0)),
        ),
        (
            SemanticQuery(QueryOp.RESOURCE_PLACE_GE, (0, 1)),
            SemanticQuery(QueryOp.RESOURCE_HALT, ()),
        ),
    )


ONTOLOGY_FACTORS = (
    _horn_factors,
    _rewrite_factors,
    _resource_factors,
)


@pytest.mark.parametrize("renderer", tuple(SurfaceRenderer))
@pytest.mark.parametrize("factor_factory", ONTOLOGY_FACTORS)
def test_all_ontologies_and_renderers_roundtrip_cell_local_sources(
    renderer: SurfaceRenderer,
    factor_factory: object,
) -> None:
    assert callable(factor_factory)
    worlds, commands, queries = factor_factory()
    bundle = build_base_surface_bundle(
        worlds,
        commands,
        queries,
        context=_context(renderer),
    )

    for world_index in range(2):
        variants = bundle.world_variants[world_index]
        assert variants[0].cell_salt == "world-0"
        assert variants[1].cell_salt == "world-1"
        assert variants[0].source != variants[1].source
        assert variants[0].parsed_semantics() == variants[1].parsed_semantics()
        assert variants[0].parsed_semantics() == canonical_factor_ast(
            worlds[world_index]
        )
    for command_index in range(2):
        variants = bundle.command_variants[command_index]
        assert variants[0].cell_salt == "command-0"
        assert variants[1].cell_salt == "command-1"
        assert variants[0].source != variants[1].source
        assert variants[0].parsed_semantics() == variants[1].parsed_semantics()
        assert variants[0].parsed_semantics() == canonical_factor_ast(
            commands[command_index]
        )

    all_sources = tuple(
        document.source
        for pair in (*bundle.world_variants, *bundle.command_variants)
        for document in pair
    )
    assert all(source.isascii() for source in all_sources)
    assert not any(
        marker in source.lower()
        for source in all_sources
        for marker in FORBIDDEN
    )


@pytest.mark.parametrize("factor_factory", ONTOLOGY_FACTORS)
def test_corner_axes_and_query_prefixes_are_shared_exactly(
    factor_factory: object,
) -> None:
    assert callable(factor_factory)
    worlds, commands, queries = factor_factory()
    bundle = build_base_surface_bundle(
        worlds,
        commands,
        queries,
        context=_context(SurfaceRenderer.PREFIX_SEXPR),
    )
    observed_query_objects = []
    for world_index in range(2):
        for command_index in range(2):
            corner = bundle.corner(world_index, command_index)
            assert corner.world is bundle.world_variants[world_index][command_index]
            assert corner.command is bundle.command_variants[command_index][world_index]
            assert corner.query_prefixes is bundle.query_prefixes
            observed_query_objects.append(corner.query_prefixes)
    assert all(item is bundle.query_prefixes for item in observed_query_objects)

    prefixes = tuple(
        query_prefix.prefix
        for query_pair in bundle.query_prefixes
        for query_prefix in query_pair
    )
    assert len(prefixes) == len(set(prefixes)) == 4
    assert all(prefix.endswith(b"\nR=") for prefix in prefixes)
    assert bundle.query_prefixes[0][0].document.cell_salt == "shared-query"
    assert bundle.query_prefixes[0][0].document.parsed_semantics() != (
        bundle.query_prefixes[0][1].document.parsed_semantics()
    )


def test_layout_bit_controls_only_declaration_reversal_schedule() -> None:
    worlds, commands, queries = _horn_factors()
    context = _context(
        SurfaceRenderer.CANONICAL_JSON,
        rectangle_id="layout-a",
    )
    bundle = build_base_surface_bundle(
        worlds,
        commands,
        queries,
        context=context,
    )
    expected_base = (
        hashlib.sha256(b"layout-a|layout").digest()[0] >> 7
    )
    assert tuple(
        document.layout for document in bundle.world_variants[0]
    ) == (expected_base, expected_base ^ 1)
    assert tuple(
        document.layout for document in bundle.command_variants[0]
    ) == (expected_base, expected_base ^ 1)
    assert all(
        prefix.document.layout == 0
        for pair in bundle.query_prefixes
        for prefix in pair
    )


def test_typed_factor_semantics_omit_assessor_only_ids_but_bind_laws() -> None:
    worlds, _, _ = _horn_factors()
    same_semantics_new_receipt = replace(
        worlds[0],
        evidence_id="f" * 64,
    )
    changed_laws = replace(worlds[0], theory_index=1)
    assert canonical_factor_ast(worlds[0]) == canonical_factor_ast(
        same_semantics_new_receipt
    )
    assert canonical_factor_ast(worlds[0]) != canonical_factor_ast(changed_laws)
    payload = render_surface(
        canonical_factor_ast(worlds[0]),
        SurfaceRenderer.CANONICAL_JSON,
    )
    assert EVIDENCE.encode("ascii") not in payload
    assert b"theory_index" not in payload


INVARIANT_PRESENTATIONS = (
    "base",
    "alpha_reorder",
    "alias_split",
    "relation_reification",
    "type_twin",
)


@pytest.mark.parametrize("factor_factory", ONTOLOGY_FACTORS)
@pytest.mark.parametrize("presentation", INVARIANT_PRESENTATIONS)
def test_invariant_presentations_invert_exactly_to_base_semantics(
    factor_factory: object,
    presentation: str,
) -> None:
    assert callable(factor_factory)
    worlds, commands, queries = factor_factory()
    bundle = build_base_surface_bundle(
        worlds,
        commands,
        queries,
        context=_context(
            SurfaceRenderer.PREFIX_SEXPR,
            presentation=presentation,
        ),
    )
    for world_index, variants in enumerate(bundle.world_variants):
        assert all(
            document.presentation == presentation
            and document.parsed_semantics()
            == canonical_factor_ast(worlds[world_index])
            for document in variants
        )
    for command_index, variants in enumerate(bundle.command_variants):
        assert all(
            document.presentation == presentation
            and document.parsed_semantics()
            == canonical_factor_ast(commands[command_index])
            for document in variants
        )


@pytest.mark.parametrize("factor_factory", ONTOLOGY_FACTORS)
def test_execution_semantics_twin_changes_world_policy_only(
    factor_factory: object,
) -> None:
    assert callable(factor_factory)
    worlds, commands, queries = factor_factory()
    bundle = build_base_surface_bundle(
        worlds,
        commands,
        queries,
        context=_context(
            SurfaceRenderer.RECORD_INFIX,
            presentation="execution_semantics_twin",
        ),
    )
    assert all(
        document.parsed_semantics()
        == canonical_factor_ast(
            worlds[world_index],
            presentation="execution_semantics_twin",
        )
        for world_index, variants in enumerate(bundle.world_variants)
        for document in variants
    )
    assert bundle.world_variants[0][0].parsed_semantics() != (
        canonical_factor_ast(worlds[0])
    )
    assert all(
        document.parsed_semantics()
        == canonical_factor_ast(commands[command_index])
        for command_index, variants in enumerate(bundle.command_variants)
        for document in variants
    )


def test_all_preregistered_presentations_are_executable() -> None:
    assert IMPLEMENTED_PRESENTATIONS == (
        "base",
        "alpha_reorder",
        "alias_split",
        "relation_reification",
        "type_twin",
        "execution_semantics_twin",
    )
    worlds, _, _ = _horn_factors()
    for presentation in IMPLEMENTED_PRESENTATIONS:
        _context(
            SurfaceRenderer.CANONICAL_JSON,
            presentation=presentation,
        )
        canonical_factor_ast(worlds[0], presentation=presentation)


def test_mixed_ontologies_and_invalid_corner_indices_fail_closed() -> None:
    horn_worlds, horn_commands, horn_queries = _horn_factors()
    resource_worlds, resource_commands, _ = _resource_factors()
    with pytest.raises(SurfaceAdapterError, match="world pair"):
        build_base_surface_bundle(
            (horn_worlds[0], resource_worlds[0]),
            horn_commands,
            horn_queries,
            context=_context(SurfaceRenderer.CANONICAL_JSON),
        )
    with pytest.raises(SurfaceAdapterError, match="command pair ontology"):
        build_base_surface_bundle(
            horn_worlds,
            resource_commands,
            horn_queries,
            context=_context(SurfaceRenderer.CANONICAL_JSON),
        )
    bundle = build_base_surface_bundle(
        horn_worlds,
        horn_commands,
        horn_queries,
        context=_context(SurfaceRenderer.CANONICAL_JSON),
    )
    with pytest.raises(SurfaceAdapterError, match="exact bits"):
        bundle.corner(True, 0)  # type: ignore[arg-type]
    with pytest.raises(SurfaceAdapterError, match="exact bits"):
        bundle.corner(0, 2)


def test_full_parse_rejects_tampered_or_unbound_surface_source() -> None:
    worlds, commands, queries = _horn_factors()
    bundle = build_base_surface_bundle(
        worlds,
        commands,
        queries,
        context=_context(SurfaceRenderer.PREFIX_SEXPR),
    )
    original = bundle.world_variants[0][0]
    opaque, _ = original.opaque_to_canonical[0]
    replacement = "x" + ("f" * 16)
    assert replacement not in dict(original.opaque_to_canonical)
    tampered_source = original.source.replace(
        opaque.encode("ascii"),
        replacement.encode("ascii"),
        1,
    )
    tampered = SemanticSurfaceDocument(
        stage=SurfaceStage.WORLD,
        presentation=original.presentation,
        cell_salt=original.cell_salt,
        layout=original.layout,
        renderer=original.renderer,
        ast=replace(
            original,
            source=original.source,
        ).ast,
        source=original.source,
        semantic_ast=original.semantic_ast,
        opaque_to_canonical=original.opaque_to_canonical,
    )
    object.__setattr__(tampered, "source", tampered_source)
    with pytest.raises(SurfaceAdapterError, match="unbound opaque"):
        parse_and_semantic_canonicalize(tampered)
