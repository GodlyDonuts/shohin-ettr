"""Architecture-facing materialization of selected ETTR-IL-v3 candidates.

One canonical semantic candidate expands into four renderer-controlled views
of one independently replayed 2x2 WORLD-by-COMMAND rectangle.  The resulting
record keeps candidate-visible token-native source bytes separate from exact
assessor packets, traces, labels, and oracle receipts.  Admission executes the
real ETTR CPU materializer in explicit broad mode and hashes the resulting
source-free tensor batch.

This module is CPU-only.  It does not access a model, checkpoint, optimizer,
accelerator, network, or filesystem unless used by a caller that supplies a
tokenizer object and persists the returned record.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from enum import Enum
import hashlib
from typing import Mapping

from ettr_il_v2_custody import derive_public_split_key, prf
from ettr_il_v2_horn_adapter import adapt_horn_semantic_rectangle
from ettr_il_v2_materialize import (
    GenericCorner,
    GenericInvariantPair,
    GenericOperationTrace,
    GenericPacket,
    GenericSemanticRectangle,
    MaterializationRequest,
    materialize_ettr_il_v2,
)
from ettr_il_v2_resource_adapter import adapt_resource_rectangle
from ettr_il_v2_semantics import (
    HornWorld,
    ResourceWorld,
)
from ettr_il_v2_surface import (
    SurfaceNode,
    SurfaceRenderer,
    call,
    integer,
)
from ettr_il_v2_surface_adapter import (
    SurfaceAdapterContext,
    build_base_surface_bundle,
)
from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec
from ettr_il_v3_local_adapter import adapt_local_rewrite_rectangle
from ettr_il_v3_protocol import (
    COMMAND_WIDTH,
    PROTOCOL,
    QUERY_WIDTH,
    ROWS_PER_CORE,
    VIEWS_PER_CORE,
    WORLD_WIDTH,
)
from ettr_il_v3_reconstruct import (
    ReconstructedCandidate,
    reconstruct_candidate,
)
from ettr_il_v3_rectangles import (
    SemanticRectangleBundle,
    build_causal_rectangle,
)
from ettr_il_v3_rewrite import (
    LOCAL_LAWS,
    THEORIES,
    QueryOp as RewriteQueryOp,
    RewriteWorld,
    StructuralQuery,
)
from ettr_il_v3_shards import (
    AssessorOnly,
    AuditRecord,
    CoreIdentity,
    CounterfactualGroups,
    CoverageRecord,
    OracleChannel,
    OracleRecord,
    SemanticCoreRecord,
    SemanticFactors,
    SourceView,
    SourceVisible,
    TargetRecord,
    canonical_json_bytes,
    canonical_sha256,
    semantic_factors_sha256,
)


MATERIALIZATION_SCHEMA = "r12-ettr-il-v3-architecture-materialization-v1"
PRESENTATION = "base"
RENDERERS = tuple(SurfaceRenderer)


class V3MaterializationError(ValueError):
    """A semantic candidate cannot become one admitted ETTR v3 core."""


def _jsonable(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if type(value) in {tuple, list}:
        return [_jsonable(item) for item in value]  # type: ignore[union-attr]
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    raise V3MaterializationError(
        f"oracle value is not strict JSON: {type(value).__name__}"
    )


def _value_ref(value: object) -> dict[str, object]:
    kind = getattr(value, "kind", None)
    index = getattr(value, "index", None)
    if not isinstance(kind, Enum):
        raise V3MaterializationError("generic value reference differs")
    return {"index": index, "kind": str(kind.value)}


def _packet_value(packet: GenericPacket) -> dict[str, object]:
    return {
        "cells": [
            {
                "slot": cell.slot,
                "type_index": cell.type_index,
                "value": _value_ref(cell.value),
            }
            for cell in packet.cells
        ],
        "committed": packet.committed,
        "edges": [
            {
                "relation": edge.relation,
                "source": edge.source,
                "target": edge.target,
            }
            for edge in packet.edges
        ],
        "halted": packet.halted,
        "root": packet.root,
    }


def _operation_trace_value(trace: GenericOperationTrace) -> dict[str, object]:
    return {
        "cursor": trace.cursor,
        "mutations": [
            {
                "opcode": int(mutation.opcode),
                "relation": mutation.relation,
                "source": mutation.source,
                "target": mutation.target,
                "type_index": mutation.type_index,
                "value": _value_ref(mutation.value),
            }
            for mutation in trace.mutations
        ],
    }


def _corner_value(corner: GenericCorner) -> dict[str, object]:
    return {
        "answers": list(corner.answers),
        "disposition": corner.disposition.value,
        "operation_traces": [
            _operation_trace_value(trace)
            for trace in corner.operation_traces
        ],
        "outcome": _value_ref(corner.outcome),
        "terminal_packet": _packet_value(corner.terminal_packet),
    }


def _query_transport_prefix(
    ast: SurfaceNode,
    codec: TokenNativeSurfaceCodec,
    renderer: SurfaceRenderer,
) -> bytes:
    document = codec.serialize(ast, renderer)
    framing = b"\nR="
    tail_counts = {
        len(
            codec.tokenizer.encode(
                (document.payload + framing + answer + b"\n").decode("ascii"),
                add_special_tokens=False,
            ).ids
        )
        - len(document.token_ids)
        for answer in (b"0", b"1")
    }
    if len(tail_counts) != 1:
        raise V3MaterializationError("query answer tails have unequal widths")
    transport = codec.pack(
        document,
        width=QUERY_WIDTH - tail_counts.pop(),
    )
    prefix = transport.payload + framing
    prefix_ids = codec.tokenizer.encode(
        prefix.decode("ascii"),
        add_special_tokens=False,
    ).ids
    for answer in (b"0", b"1"):
        full_ids = codec.tokenizer.encode(
            (prefix + answer + b"\n").decode("ascii"),
            add_special_tokens=False,
        ).ids
        if (
            full_ids[: len(prefix_ids)] != prefix_ids
            or len(full_ids) != QUERY_WIDTH
        ):
            raise V3MaterializationError(
                "query answer is not one fixed-width token boundary"
            )
    return prefix


def _legacy_sources(
    rectangle: SemanticRectangleBundle,
    *,
    codec: TokenNativeSurfaceCodec,
    renderer: SurfaceRenderer,
    key: bytes,
    owner_split: str,
) -> tuple[
    tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
    tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
    tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
]:
    context = SurfaceAdapterContext(
        fold=0,
        split=owner_split,
        semantic_core_id=rectangle.episode_id,
        semantic_rectangle_id=rectangle.semantic_rectangle_id,
        renderer=renderer,
        prf=lambda label, payload: prf(key, label, payload),
        presentation=PRESENTATION,
    )
    surface = build_base_surface_bundle(
        rectangle.worlds,  # type: ignore[arg-type]
        rectangle.commands,  # type: ignore[arg-type]
        rectangle.queries,  # type: ignore[arg-type]
        context=context,
    )
    worlds = tuple(
        tuple(
            codec.pack(
                codec.serialize(document.ast, renderer),
                width=WORLD_WIDTH,
            ).payload
            for document in variants
        )
        for variants in surface.world_variants
    )
    commands = tuple(
        tuple(
            codec.pack(
                codec.serialize(document.ast, renderer),
                width=COMMAND_WIDTH,
            ).payload
            for document in variants
        )
        for variants in surface.command_variants
    )
    queries = tuple(
        tuple(
            _query_transport_prefix(prefix.document.ast, codec, renderer)
            for prefix in variants
        )
        for variants in surface.query_prefixes
    )
    return worlds, commands, queries  # type: ignore[return-value]


def _local_world_ast(world: RewriteWorld, nuisance: int) -> SurfaceNode:
    theory = THEORIES[world.theory_index]
    laws = tuple(
        call(
            6,
            integer(local_slot),
            integer(LOCAL_LAWS[law_index].forward_source[0]),
            integer(LOCAL_LAWS[law_index].forward_source[1]),
            integer(LOCAL_LAWS[law_index].forward_target[0]),
            integer(LOCAL_LAWS[law_index].forward_target[1]),
        )
        for local_slot, law_index in enumerate(theory.law_indices)
    )
    semantic = call(
        14,
        integer(3),
        call(1, *laws),
        call(7, *(integer(value) for value in world.registers)),
    )
    return call(15, integer(nuisance), semantic)


def _local_command_ast(
    command: object,
    nuisance: int,
) -> SurfaceNode:
    operations = getattr(command, "operations", None)
    if type(operations) is not tuple:
        raise V3MaterializationError("local command operations differ")
    semantic = call(
        13,
        *(
            call(
                4,
                integer(operation.law_slot),
                integer(operation.site),
                integer(0 if operation.direction.value == "forward" else 1),
            )
            for operation in operations
        ),
    )
    return call(15, integer(nuisance), semantic)


def _local_query_ast(query: StructuralQuery, paraphrase: int) -> SurfaceNode:
    operation = tuple(RewriteQueryOp).index(query.op)
    semantic = call(
        4,
        integer(operation),
        *(integer(argument) for argument in query.arguments),
    )
    return (
        call(9, semantic)
        if paraphrase == 0
        else call(10, semantic, integer(1))
    )


def _local_sources(
    rectangle: SemanticRectangleBundle,
    *,
    codec: TokenNativeSurfaceCodec,
    renderer: SurfaceRenderer,
) -> tuple[
    tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
    tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
    tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
]:
    worlds = tuple(
        tuple(
            codec.pack(
                codec.serialize(
                    _local_world_ast(world, nuisance),
                    renderer,
                ),
                width=WORLD_WIDTH,
            ).payload
            for nuisance in range(2)
        )
        for world in rectangle.worlds
    )
    commands = tuple(
        tuple(
            codec.pack(
                codec.serialize(
                    _local_command_ast(command, nuisance),
                    renderer,
                ),
                width=COMMAND_WIDTH,
            ).payload
            for nuisance in range(2)
        )
        for command in rectangle.commands
    )
    queries = tuple(
        tuple(
            _query_transport_prefix(
                _local_query_ast(query, paraphrase),
                codec,
                renderer,
            )
            for paraphrase in range(2)
        )
        for query in rectangle.queries
    )
    return worlds, commands, queries  # type: ignore[return-value]


def _adapt(
    rectangle: SemanticRectangleBundle,
    *,
    renderer: SurfaceRenderer,
    worlds: tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
    commands: tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
    queries: tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
) -> GenericSemanticRectangle:
    presentation_id = f"{PRESENTATION}-renderer-{int(renderer)}"
    if rectangle.family == "horn":
        return adapt_horn_semantic_rectangle(
            semantic_rectangle_id=rectangle.semantic_rectangle_id,
            presentation_id=presentation_id,
            worlds=rectangle.worlds,  # type: ignore[arg-type]
            commands=rectangle.commands,  # type: ignore[arg-type]
            primary_executions=rectangle.primary,  # type: ignore[arg-type]
            replay_executions=rectangle.replay,  # type: ignore[arg-type]
            queries=rectangle.queries,  # type: ignore[arg-type]
            world_sources=worlds,
            command_sources=commands,
            query_prefixes=queries,
            require_query_checkerboard=False,
        )
    if rectangle.family == "resource":
        command_cells = tuple(
            tuple(
                commands[command_index][world_index]
                for command_index in range(2)
            )
            for world_index in range(2)
        )
        return adapt_resource_rectangle(
            semantic_rectangle_id=rectangle.semantic_rectangle_id,
            presentation_id=presentation_id,
            worlds=rectangle.worlds,  # type: ignore[arg-type]
            commands=rectangle.commands,  # type: ignore[arg-type]
            executions=rectangle.primary,  # type: ignore[arg-type]
            queries=rectangle.queries,  # type: ignore[arg-type]
            cell_world_sources=worlds,
            cell_command_sources=command_cells,  # type: ignore[arg-type]
            query_prefixes=queries,
            require_query_checkerboard=False,
        )
    return adapt_local_rewrite_rectangle(
        rectangle,
        presentation_id=presentation_id,
        world_sources=worlds,
        command_sources=commands,
        query_prefixes=queries,
        require_query_checkerboard=False,
    )


def _batch_sha256(batch: object) -> str:
    hasher = hashlib.sha256()

    def visit(value: object, path: str) -> None:
        hasher.update(path.encode("ascii") + b"\0")
        if hasattr(value, "detach") and hasattr(value, "shape"):
            tensor = value.detach().cpu().contiguous()  # type: ignore[union-attr]
            hasher.update(str(tensor.dtype).encode("ascii") + b"\0")
            hasher.update(canonical_json_bytes(list(tensor.shape)))
            hasher.update(tensor.numpy().tobytes())
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name), f"{path}.{field.name}")
        elif type(value) in {tuple, list}:
            for index, item in enumerate(value):  # type: ignore[union-attr]
                visit(item, f"{path}[{index}]")
        elif value is None or type(value) in {bool, int, str}:
            hasher.update(canonical_json_bytes(value))
        else:
            raise V3MaterializationError(
                f"batch contains unsupported value {type(value).__name__}"
            )

    visit(batch, "batch")
    return hasher.hexdigest()


def _factor_value(value: object) -> object:
    method = getattr(value, "to_value", None)
    if callable(method):
        return method()
    method = getattr(value, "assessor_value", None)
    if callable(method):
        return method()
    if type(value) is HornWorld or type(value) is ResourceWorld:
        from ettr_il_v2_candidate_search import semantic_world_value  # noqa: PLC0415

        return semantic_world_value(value)
    from ettr_il_v2_candidate_search import semantic_command_value  # noqa: PLC0415

    return semantic_command_value(value)  # type: ignore[arg-type]


def _semantic_factors(
    candidate: ReconstructedCandidate,
    rectangle: SemanticRectangleBundle,
) -> SemanticFactors:
    world = rectangle.worlds[0]
    if type(world) is RewriteWorld:
        theory = {
            "family": candidate.family,
            "laws": list(THEORIES[world.theory_index].law_indices),
            "theory_index": world.theory_index,
        }
    else:
        theory = {
            "evidence_id": world.evidence_id,
            "family": candidate.family,
            "policy": world.policy.value,
            "theory_index": world.theory_index,
        }
    return SemanticFactors(
        theory=theory,
        worlds=tuple(_factor_value(value) for value in rectangle.worlds),
        commands=tuple(_factor_value(value) for value in rectangle.commands),
        queries=tuple(_factor_value(value) for value in rectangle.queries),
    )


def _source_view(
    rectangle: GenericSemanticRectangle,
    renderer: SurfaceRenderer,
) -> SourceView:
    world_sources = tuple(
        source.decode("ascii")
        for world in rectangle.worlds
        for source in world.sources
    )
    command_sources = tuple(
        source.decode("ascii")
        for command in rectangle.commands
        for source in command.sources
    )
    query_sources = tuple(
        source.decode("ascii")
        for query in rectangle.queries
        for source in query.prefixes
    )
    view_id = hashlib.sha256(
        canonical_json_bytes(
            {
                "renderer": int(renderer),
                "semantic_rectangle_id": rectangle.semantic_rectangle_id,
                "sources": [
                    *world_sources,
                    *command_sources,
                    *query_sources,
                ],
            }
        )
    ).hexdigest()
    return SourceView(
        view_id=view_id,
        presentation=PRESENTATION,
        renderer=int(renderer),
        world_sources=world_sources,
        command_sources=command_sources,
        query_sources=query_sources,
    )


def _targets(rectangle: GenericSemanticRectangle) -> TargetRecord:
    corners = tuple(
        rectangle.corners[world][command]
        for world in range(2)
        for command in range(2)
    )
    return TargetRecord(
        initial_packets=tuple(
            _packet_value(world.initial_packet)
            for world in rectangle.worlds
        ),
        terminal_packets=tuple(
            _packet_value(corner.terminal_packet)
            for corner in corners
        ),
        transaction_traces=tuple(
            [_operation_trace_value(trace) for trace in corner.operation_traces]
            for corner in corners
        ),
        answer_matrix=tuple(list(corner.answers) for corner in corners),
    )


def _oracle_channel(
    executions: object,
    generic: GenericSemanticRectangle,
) -> OracleChannel:
    flat = tuple(
        executions[world][command]  # type: ignore[index]
        for world in range(2)
        for command in range(2)
    )
    return OracleChannel(
        executions=tuple(_jsonable(execution) for execution in flat),
        intermediate_snapshots=tuple(
            _jsonable(getattr(execution, "snapshots"))
            for execution in flat
        ),
        terminal_observations=tuple(
            {
                "answers": list(
                    generic.corners[index // 2][index % 2].answers
                ),
                "terminal_packet": _packet_value(
                    generic.corners[index // 2][index % 2].terminal_packet
                ),
            }
            for index in range(4)
        ),
    )


def _coverage(
    candidate: ReconstructedCandidate,
    generic: GenericSemanticRectangle,
) -> CoverageRecord:
    corners = tuple(
        generic.corners[world][command]
        for world in range(2)
        for command in range(2)
    )
    mutation_count = sum(
        len(trace.mutations)
        for corner in corners
        for trace in corner.operation_traces
    )
    trace_length = max(
        2 * len(corner.operation_traces)
        + sum(len(trace.mutations) for trace in corner.operation_traces)
        + 2
        for corner in corners
    )
    active_slots = max(len(corner.terminal_packet.cells) for corner in corners)
    edge_count = max(len(corner.terminal_packet.edges) for corner in corners)
    topology_signature = canonical_sha256(
        {
            "active": [
                [cell.slot for cell in corner.terminal_packet.cells]
                for corner in corners
            ],
            "edges": [
                [
                    [edge.relation, edge.source, edge.target]
                    for edge in corner.terminal_packet.edges
                ]
                for corner in corners
            ],
            "family": candidate.family,
        }
    )
    return CoverageRecord(
        depth=candidate.depth,
        trace_length=trace_length,
        opcode_histogram={
            "COMMAND_WRITE": 4 * candidate.depth,
            "CURSOR_WRITE": 4 * candidate.depth,
            "ONTOLOGY_MUTATION": mutation_count,
            "OUTCOME_WRITE": 4,
            "TERMINAL": 4,
        },
        active_slot_bin=active_slots // 8,
        edge_count_bin=edge_count // 16,
        topology_signature=topology_signature,
    )


def _counterfactual_groups(
    rectangle: SemanticRectangleBundle,
) -> CounterfactualGroups:
    def digest(axis: str, value: object) -> str:
        return canonical_sha256(
            {
                "axis": axis,
                "episode_id": rectangle.episode_id,
                "factor": _factor_value(value),
                "protocol": PROTOCOL,
            }
        )

    return CounterfactualGroups(
        invariant_orbit_id=canonical_sha256(
            {
                "episode_id": rectangle.episode_id,
                "protocol": PROTOCOL,
                "renderers": list(range(VIEWS_PER_CORE)),
            }
        ),
        world_counterfactual_id=digest("world", rectangle.worlds[1]),
        command_counterfactual_id=digest("command", rectangle.commands[1]),
        query_counterfactual_id=digest("query", rectangle.queries[1]),
        hard_negative_ids=(),
    )


def materialize_candidate(
    value: object,
    tokenizer: object,
    *,
    confirmation_key: bytes | None = None,
) -> SemanticCoreRecord:
    """Reconstruct, render, replay, tensor-admit, and seal one selected core."""

    candidate = reconstruct_candidate(value)
    rectangle = build_causal_rectangle(candidate)
    owner_split = candidate.split.removesuffix("_reserve")
    if owner_split == "confirmation":
        if type(confirmation_key) is not bytes or len(confirmation_key) != 32:
            raise V3MaterializationError(
                "confirmation materialization requires a sealed 32-byte key"
            )
        split_key = confirmation_key
    else:
        if confirmation_key is not None:
            raise V3MaterializationError(
                "public materialization cannot receive a confirmation key"
            )
        split_key = derive_public_split_key(0, owner_split)
    codec = (
        tokenizer
        if isinstance(tokenizer, TokenNativeSurfaceCodec)
        else TokenNativeSurfaceCodec(tokenizer)  # type: ignore[arg-type]
    )
    generic_rectangles: list[GenericSemanticRectangle] = []
    views: list[SourceView] = []
    for renderer in RENDERERS:
        view_rectangle = replace(
            rectangle,
            semantic_rectangle_id=canonical_sha256(
                {
                    "base_semantic_rectangle_id": (
                        rectangle.semantic_rectangle_id
                    ),
                    "renderer": int(renderer),
                    "schema": MATERIALIZATION_SCHEMA,
                }
            ),
        )
        sources = (
            _local_sources(view_rectangle, codec=codec, renderer=renderer)
            if candidate.family == "local_rewrite"
            else _legacy_sources(
                view_rectangle,
                codec=codec,
                renderer=renderer,
                key=split_key,
                owner_split=owner_split,
            )
        )
        generic = _adapt(
            view_rectangle,
            renderer=renderer,
            worlds=sources[0],
            commands=sources[1],
            queries=sources[2],
        )
        generic_rectangles.append(generic)
        views.append(_source_view(generic, renderer))
    source_value = [view.to_value() for view in views]
    dataset_sha256 = canonical_sha256(
        {
            "episode_id": candidate.episode_id,
            "schema": MATERIALIZATION_SCHEMA,
            "source_visible": source_value,
        }
    )
    manifest_sha256 = canonical_sha256(
        {
            "codebook_sha256": codec.codebook_sha256,
            "dataset_sha256": dataset_sha256,
            "protocol": PROTOCOL,
            "tokenizer_sha256": codec.tokenizer_sha256,
        }
    )
    vocab_size = codec.tokenizer.get_vocab_size()
    batch = materialize_ettr_il_v2(
        MaterializationRequest(
            manifest_sha256=manifest_sha256,
            dataset_sha256=dataset_sha256,
            vocab_size=vocab_size,
            rectangles=tuple(generic_rectangles),
            invariant_pairs=(
                GenericInvariantPair(0, 1),
                GenericInvariantPair(2, 3),
            ),
            require_query_checkerboard=False,
        ),
        codec.tokenizer,
    )
    if (
        batch.episodes.world.tokens.shape[0] != ROWS_PER_CORE
        or batch.episodes.world.tokens.shape[1] != WORLD_WIDTH
        or batch.episodes.command.tokens.shape[1] != COMMAND_WIDTH
        or batch.episodes.query.tokens.shape[1] != QUERY_WIDTH
    ):
        raise V3MaterializationError("materialized tensor geometry differs")

    generic = generic_rectangles[0]
    factors = _semantic_factors(candidate, rectangle)
    targets = _targets(generic)
    replay_value = [
        _jsonable(rectangle.replay[world][command])
        for world in range(2)
        for command in range(2)
    ]
    token_hashes = tuple(
        hashlib.sha256(source.encode("ascii")).hexdigest()
        for view in views
        for source in (
            *view.world_sources,
            *view.command_sources,
            *view.query_sources,
        )
    )
    materialization_hash = _batch_sha256(batch)
    audit = AuditRecord(
        raw_hashes=(
            hashlib.sha256(canonical_json_bytes(candidate.row)).hexdigest(),
        ),
        semantic_hash=semantic_factors_sha256(factors),
        graph_iso_hash=canonical_sha256(
            {
                "family": candidate.family,
                "semantic_factors": factors.to_value(),
            }
        ),
        token_hashes=token_hashes,
        materialization_hash=materialization_hash,
        replay_hash=canonical_sha256(replay_value),
    )
    record = SemanticCoreRecord(
        identity=CoreIdentity(
            core_id=candidate.episode_id,
            generator_version=MATERIALIZATION_SCHEMA,
            split=candidate.split,
            curriculum_stage=candidate.stage.value,
            generator_ordinal=candidate.ordinal,
        ),
        source_visible=SourceVisible(tuple(views)),
        assessor_only=AssessorOnly(
            semantic_factors=factors,
            oracle=OracleRecord(
                primary=_oracle_channel(
                    rectangle.primary,
                    generic,
                ),
                replay=_oracle_channel(
                    rectangle.replay,
                    generic,
                ),
            ),
            targets=targets,
            counterfactual_groups=_counterfactual_groups(rectangle),
            coverage=_coverage(candidate, generic),
            audit=audit,
        ),
    )
    record.validate()
    return record


__all__ = [
    "MATERIALIZATION_SCHEMA",
    "PRESENTATION",
    "RENDERERS",
    "V3MaterializationError",
    "materialize_candidate",
]
