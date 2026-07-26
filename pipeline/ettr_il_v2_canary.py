"""End-to-end CPU canary for R12-ETTR-IL-v2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from ettr_il_v2_candidate_search import (
    SemanticCandidate,
    find_first_depth1_checkerboard,
    semantic_core_id,
)
from ettr_il_v2_custody import derive_public_split_key, prf
from ettr_il_v2_horn_adapter import adapt_horn_semantic_rectangle
from ettr_il_v2_materialize import (
    COMMAND_WIDTH,
    GenericSemanticRectangle,
    MaterializationRequest,
    QUERY_WIDTH,
    WORLD_WIDTH,
    materialize_ettr_il_v2,
)
from ettr_il_v2_resource_adapter import adapt_resource_rectangle
from ettr_il_v2_rewrite_adapter import adapt_rewrite_rectangle
from ettr_il_v2_semantics import (
    HornCommand,
    HornExecution,
    HornWorld,
    Ontology,
    ResourceCommand,
    ResourceExecution,
    ResourceWorld,
    RewriteCommand,
    RewriteExecution,
    RewriteWorld,
    replay_semantics,
)
from ettr_il_v2_surface import SurfaceNode, SurfaceRenderer
from ettr_il_v2_surface_adapter import (
    BaseSurfaceBundle,
    SurfaceAdapterContext,
    build_base_surface_bundle,
)
from ettr_il_v2_token_native_surface import (
    CODEWORD_BYTES,
    TokenNativeSurfaceCodec,
)


PROTOCOL = "R12-ETTR-IL-v2"
REPORT_SCHEMA = "r12-ettr-il-v2-end-to-end-canary-v2"
TOKENIZER_SHA256 = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)


class CanaryError(ValueError):
    """The complete CPU canary cannot be admitted."""


@dataclass(frozen=True, slots=True)
class CanaryResult:
    ontology: str
    semantic_core_id: str
    semantic_rectangle_id: str
    renderer: int
    codebook_sha256: str
    row_count: int
    causal_rectangle_count: int
    valid_trace_min: int
    valid_trace_max: int
    world_source_sha256: tuple[str, ...]
    command_source_sha256: tuple[str, ...]
    query_prefix_sha256: tuple[str, ...]
    batch_sha256: str
    source_free_batch: bool
    world_token_count: int
    command_token_count: int
    query_token_count: int
    world_byte_count: int
    command_byte_count: int
    query_prefix_byte_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            field.name: (
                list(value)
                if isinstance((value := getattr(self, field.name)), tuple)
                else value
            )
            for field in fields(self)
        }


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rectangle_id(core_id: str, renderer: SurfaceRenderer) -> str:
    return _digest(
        canonical_json_bytes(
            {
                "semantic_core_id": core_id,
                "presentation": "base",
                "protocol": PROTOCOL,
                "renderer": int(renderer),
            }
        )
    )


def _execution_matrices(
    candidate: SemanticCandidate,
) -> tuple[tuple[tuple[Any, Any], tuple[Any, Any]], ...]:
    primary = (
        (candidate.rectangle.cells[0], candidate.rectangle.cells[1]),
        (candidate.rectangle.cells[2], candidate.rectangle.cells[3]),
    )
    replay = tuple(
        tuple(
            replay_semantics(
                candidate.worlds[world_index],
                candidate.commands[command_index],
                require_dependent=False,
            )
            for command_index in range(2)
        )
        for world_index in range(2)
    )
    if primary != replay:
        raise CanaryError("primary and replay execution matrices differ")
    return primary, replay  # type: ignore[return-value]


def _surface_sources(
    bundle: BaseSurfaceBundle,
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
                codec.serialize(document.ast, renderer),
                width=WORLD_WIDTH,
            ).payload
            for document in variants
        )
        for variants in bundle.world_variants
    )
    commands = tuple(
        tuple(
            codec.pack(
                codec.serialize(document.ast, renderer),
                width=COMMAND_WIDTH,
            ).payload
            for document in variants
        )
        for variants in bundle.command_variants
    )
    queries = tuple(
        tuple(
            _query_transport_prefix(prefix.document.ast, codec, renderer)
            for prefix in variants
        )
        for variants in bundle.query_prefixes
    )
    return worlds, commands, queries  # type: ignore[return-value]


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
        raise CanaryError("query answer tails have different token widths")
    tail_count = tail_counts.pop()
    transport_width = QUERY_WIDTH - tail_count
    transport = codec.pack(document, width=transport_width)
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
            raise CanaryError(
                "query transport does not preserve the one-token answer boundary"
            )
    return prefix


def _adapt_with_sources(
    candidate: SemanticCandidate,
    rectangle_id: str,
    worlds: tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
    commands: tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
    queries: tuple[tuple[bytes, bytes], tuple[bytes, bytes]],
) -> GenericSemanticRectangle:
    primary, replay = _execution_matrices(candidate)
    selected_queries = (
        candidate.queries.slot_0,
        candidate.queries.slot_1,
    )
    if candidate.ontology is Ontology.HORN:
        if (
            any(type(value) is not HornWorld for value in candidate.worlds)
            or any(
                type(value) is not HornCommand
                for value in candidate.commands
            )
            or any(
                type(value) is not HornExecution
                for row in primary
                for value in row
            )
        ):
            raise CanaryError("Horn candidate types differ")
        return adapt_horn_semantic_rectangle(
            semantic_rectangle_id=rectangle_id,
            presentation_id="base",
            worlds=candidate.worlds,  # type: ignore[arg-type]
            commands=candidate.commands,  # type: ignore[arg-type]
            primary_executions=primary,  # type: ignore[arg-type]
            replay_executions=replay,  # type: ignore[arg-type]
            queries=selected_queries,
            world_sources=worlds,
            command_sources=commands,
            query_prefixes=queries,
        )
    if candidate.ontology is Ontology.REWRITE:
        if (
            any(type(value) is not RewriteWorld for value in candidate.worlds)
            or any(
                type(value) is not RewriteCommand
                for value in candidate.commands
            )
            or any(
                type(value) is not RewriteExecution
                for row in primary
                for value in row
            )
        ):
            raise CanaryError("rewrite candidate types differ")
        return adapt_rewrite_rectangle(
            semantic_rectangle_id=rectangle_id,
            presentation_id="base",
            worlds=candidate.worlds,  # type: ignore[arg-type]
            commands=candidate.commands,  # type: ignore[arg-type]
            primary_executions=primary,  # type: ignore[arg-type]
            replay_executions=replay,  # type: ignore[arg-type]
            queries=selected_queries,
            world_sources=worlds,
            command_sources=commands,
            query_prefixes=queries,
        )
    if (
        any(type(value) is not ResourceWorld for value in candidate.worlds)
        or any(
            type(value) is not ResourceCommand
            for value in candidate.commands
        )
        or any(
            type(value) is not ResourceExecution
            for row in primary
            for value in row
        )
    ):
        raise CanaryError("resource candidate types differ")
    command_cells = tuple(
        tuple(commands[command_index][world_index] for command_index in range(2))
        for world_index in range(2)
    )
    return adapt_resource_rectangle(
        semantic_rectangle_id=rectangle_id,
        presentation_id="base",
        worlds=candidate.worlds,  # type: ignore[arg-type]
        commands=candidate.commands,  # type: ignore[arg-type]
        executions=primary,  # type: ignore[arg-type]
        queries=selected_queries,
        cell_world_sources=worlds,
        cell_command_sources=command_cells,  # type: ignore[arg-type]
        query_prefixes=queries,
    )


def _contains_source_bytes(value: object, seen: set[int] | None = None) -> bool:
    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return False
    visited.add(identity)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if is_dataclass(value):
        return any(
            _contains_source_bytes(getattr(value, field.name), visited)
            for field in fields(value)
        )
    if isinstance(value, dict):
        return any(
            _contains_source_bytes(key, visited)
            or _contains_source_bytes(item, visited)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_source_bytes(item, visited) for item in value)
    return False


def _batch_digest(batch: Any) -> str:
    hasher = hashlib.sha256()
    tensors = (
        batch.episodes.world.tokens,
        batch.episodes.world.attention_mask,
        batch.episodes.command.tokens,
        batch.episodes.command.attention_mask,
        batch.episodes.query.tokens,
        batch.episodes.query.attention_mask,
        batch.episodes.query_read_index,
        batch.packet_targets.value_code,
        batch.packet_targets.type_index,
        batch.packet_targets.relations,
        batch.terminal_packet_targets.value_code,
        batch.terminal_packet_targets.type_index,
        batch.terminal_packet_targets.relations,
        batch.transaction_targets.opcode,
        batch.transaction_targets.source,
        batch.transaction_targets.target,
        batch.transaction_targets.relation,
        batch.transaction_targets.type_index,
        batch.transaction_targets.value_code,
        batch.transaction_targets.committed,
        batch.transaction_targets.halted,
        batch.transaction_targets.step_mask,
        batch.causal_rectangles.rows,
    )
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        hasher.update(str(contiguous.dtype).encode("ascii"))
        hasher.update(canonical_json_bytes(list(contiguous.shape)))
        hasher.update(contiguous.numpy().tobytes())
    return hasher.hexdigest()


def run_canary(
    ontology: Ontology,
    tokenizer: Any,
    *,
    fold: int = 0,
    split: str = "train",
    renderer: SurfaceRenderer = SurfaceRenderer.CANONICAL_JSON,
) -> CanaryResult:
    if type(ontology) is not Ontology:
        raise CanaryError("ontology differs")
    if split not in {"train", "development"}:
        raise CanaryError("CPU canary split differs")
    candidate = find_first_depth1_checkerboard(ontology)
    core_id = semantic_core_id(candidate)
    rectangle_id = _rectangle_id(core_id, renderer)
    key = derive_public_split_key(fold, split)
    context = SurfaceAdapterContext(
        fold=fold,
        split=split,
        semantic_core_id=core_id,
        semantic_rectangle_id=rectangle_id,
        renderer=renderer,
        prf=lambda label, payload: prf(key, label, payload),
    )
    surface = build_base_surface_bundle(
        candidate.worlds,
        candidate.commands,
        (candidate.queries.slot_0, candidate.queries.slot_1),
        context=context,
    )
    codec = TokenNativeSurfaceCodec(tokenizer)
    world_sources, command_sources, query_sources = _surface_sources(
        surface,
        codec,
        renderer,
    )
    generic = _adapt_with_sources(
        candidate,
        rectangle_id,
        world_sources,
        command_sources,
        query_sources,
    )
    source_manifest = {
        "commands": [
            _digest(value)
            for pair in command_sources
            for value in pair
        ],
        "ontology": ontology.value,
        "queries": [
            _digest(value) for pair in query_sources for value in pair
        ],
        "semantic_rectangle_id": rectangle_id,
        "worlds": [
            _digest(value) for pair in world_sources for value in pair
        ],
    }
    dataset_sha256 = _digest(canonical_json_bytes(source_manifest))
    manifest_sha256 = _digest(
        canonical_json_bytes(
            {
                "dataset_sha256": dataset_sha256,
                "protocol": PROTOCOL,
                "tokenizer_sha256": TOKENIZER_SHA256,
            }
        )
    )
    try:
        vocab_size = tokenizer.get_vocab_size()
    except AttributeError as exc:
        raise CanaryError("tokenizer lacks get_vocab_size") from exc
    batch = materialize_ettr_il_v2(
        MaterializationRequest(
            manifest_sha256=manifest_sha256,
            dataset_sha256=dataset_sha256,
            vocab_size=vocab_size,
            rectangles=(generic,),
        ),
        tokenizer,
    )
    valid_counts = batch.transaction_targets.step_mask.sum(dim=1)
    result = CanaryResult(
        ontology=ontology.value,
        semantic_core_id=core_id,
        semantic_rectangle_id=rectangle_id,
        renderer=int(renderer),
        codebook_sha256=codec.codebook_sha256,
        row_count=batch.episodes.world.tokens.shape[0],
        causal_rectangle_count=batch.causal_rectangles.rows.shape[0],
        valid_trace_min=int(valid_counts.min().item()),
        valid_trace_max=int(valid_counts.max().item()),
        world_source_sha256=tuple(source_manifest["worlds"]),
        command_source_sha256=tuple(source_manifest["commands"]),
        query_prefix_sha256=tuple(source_manifest["queries"]),
        batch_sha256=_batch_digest(batch),
        source_free_batch=not _contains_source_bytes(batch),
        world_token_count=int(
            batch.episodes.world.attention_mask.sum(dim=1).min().item()
        ),
        command_token_count=int(
            batch.episodes.command.attention_mask.sum(dim=1).min().item()
        ),
        query_token_count=int(
            batch.episodes.query.attention_mask.sum(dim=1).min().item()
        ),
        world_byte_count=len(world_sources[0][0]),
        command_byte_count=len(command_sources[0][0]),
        query_prefix_byte_count=len(query_sources[0][0]),
    )
    if (
        result.row_count != 16
        or result.causal_rectangle_count != 4
        or not result.source_free_batch
        or result.world_token_count != WORLD_WIDTH
        or result.command_token_count != COMMAND_WIDTH
        or result.query_token_count != QUERY_WIDTH
        or result.world_byte_count != WORLD_WIDTH * CODEWORD_BYTES
        or result.command_byte_count != COMMAND_WIDTH * CODEWORD_BYTES
    ):
        raise CanaryError("materialized canary admission differs")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = args.tokenizer.read_bytes()
    if _digest(payload) != TOKENIZER_SHA256:
        raise CanaryError("tokenizer SHA-256 differs")
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise CanaryError("tokenizers runtime is unavailable") from exc
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    results = tuple(run_canary(ontology, tokenizer) for ontology in Ontology)
    report = {
        "protocol": PROTOCOL,
        "results": [result.as_dict() for result in results],
        "schema": REPORT_SCHEMA,
        "status": "pass",
        "tokenizer_sha256": TOKENIZER_SHA256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    print(canonical_json_bytes(report).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CanaryError",
    "CanaryResult",
    "PROTOCOL",
    "REPORT_SCHEMA",
    "TOKENIZER_SHA256",
    "canonical_json_bytes",
    "run_canary",
]
