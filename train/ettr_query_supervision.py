"""Assessor-supervised query programs aligned to immutable ETTR v3 batches.

The supervision in this module is available only while fitting a query
compiler.  Candidate inference receives the original source query tokens and
the architecture-produced typed state; it never receives these labels or an
answer.  Alignment is performed against the exact candidate-visible query
prefix, then erased before model evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from ettr_data_contract import ETTRContinuationBatch
from ettr_v3_streaming import ETTRV3StreamingError, ETTRV3StreamingRelease


QUERY_OPERATIONS = (
    "horn_has",
    "horn_count_ge",
    "resource_place_ge",
    "resource_cursor_ge",
    "resource_halt",
    "slot_is",
    "type_count_ge",
    "adjacent_is",
    "pattern_exists",
    "same_type_slots_equal",
    "slot_changed",
)
QUERY_OPERATION_TO_INDEX = {
    operation: index for index, operation in enumerate(QUERY_OPERATIONS)
}
MAX_QUERY_ARGUMENTS = 3
MAX_QUERY_ARGUMENT_VALUE = 27


class ETTRQuerySupervisionError(ValueError):
    """A sealed query-program label or its source alignment differs."""


@dataclass(frozen=True, slots=True)
class ETTRQuerySpecBatch:
    """Exact query programs used only as compiler training targets."""

    operation: torch.Tensor
    arguments: torch.Tensor
    argument_mask: torch.Tensor

    def validate(self, *, batch_size: int | None = None) -> None:
        if (
            self.operation.ndim != 1
            or self.operation.dtype != torch.long
            or self.arguments.shape != (self.operation.shape[0], MAX_QUERY_ARGUMENTS)
            or self.arguments.dtype != torch.long
            or self.argument_mask.shape != self.arguments.shape
            or self.argument_mask.dtype != torch.bool
            or self.arguments.device != self.operation.device
            or self.argument_mask.device != self.operation.device
            or (batch_size is not None and self.operation.shape[0] != batch_size)
            or not bool(
                (
                    (self.operation >= 0)
                    & (self.operation < len(QUERY_OPERATIONS))
                ).all()
            )
            or not bool(
                (
                    (self.arguments >= 0)
                    & (self.arguments <= MAX_QUERY_ARGUMENT_VALUE)
                ).all()
            )
            or not bool(self.argument_mask[:, :-1].ge(self.argument_mask[:, 1:]).all())
        ):
            raise ETTRQuerySupervisionError("query-program batch differs")

    def to(self, device: torch.device | str) -> "ETTRQuerySpecBatch":
        return ETTRQuerySpecBatch(
            operation=self.operation.to(device),
            arguments=self.arguments.to(device),
            argument_mask=self.argument_mask.to(device),
        )


def _query_specs(record: object) -> tuple[tuple[int, tuple[int, ...]], ...]:
    assessor = getattr(record, "assessor_only", None)
    factors = getattr(assessor, "semantic_factors", None)
    queries = getattr(factors, "queries", None)
    if not isinstance(queries, tuple) or len(queries) != 2:
        raise ETTRQuerySupervisionError("semantic query factor pair differs")
    result: list[tuple[int, tuple[int, ...]]] = []
    for value in queries:
        if not isinstance(value, dict):
            raise ETTRQuerySupervisionError("semantic query factor differs")
        operation = value.get("op")
        arguments = value.get("args", value.get("arguments"))
        if (
            not isinstance(operation, str)
            or operation not in QUERY_OPERATION_TO_INDEX
            or not isinstance(arguments, list)
            or len(arguments) > MAX_QUERY_ARGUMENTS
            or any(
                not isinstance(argument, int)
                or isinstance(argument, bool)
                or not 0 <= argument <= MAX_QUERY_ARGUMENT_VALUE
                for argument in arguments
            )
        ):
            raise ETTRQuerySupervisionError("semantic query program differs")
        result.append(
            (QUERY_OPERATION_TO_INDEX[operation], tuple(arguments))
        )
    return tuple(result)


def query_specs_for_batch(
    record: object,
    batch: ETTRContinuationBatch,
    *,
    tokenizer,
) -> ETTRQuerySpecBatch:
    """Align exact semantic programs to rows by visible query-prefix bytes."""

    programs = _query_specs(record)
    source_visible = getattr(record, "source_visible", None)
    views = getattr(source_visible, "views", None)
    if not isinstance(views, tuple) or len(views) != 4:
        raise ETTRQuerySupervisionError("semantic query view geometry differs")
    prefix_programs: dict[tuple[int, ...], tuple[int, tuple[int, ...]]] = {}
    for view in views:
        sources = getattr(view, "query_sources", None)
        if not isinstance(sources, tuple) or len(sources) != 4:
            raise ETTRQuerySupervisionError("query-prefix source geometry differs")
        for source_index, source in enumerate(sources):
            if not isinstance(source, str):
                raise ETTRQuerySupervisionError("query-prefix source differs")
            prefix = tuple(
                tokenizer.encode(source, add_special_tokens=False).ids
            )
            program = programs[source_index // 2]
            previous = prefix_programs.setdefault(prefix, program)
            if previous != program:
                raise ETTRQuerySupervisionError(
                    "candidate-visible query prefix has conflicting programs"
                )

    operations: list[int] = []
    arguments: list[tuple[int, int, int]] = []
    masks: list[tuple[bool, bool, bool]] = []
    tokens = batch.episodes.query.tokens.detach().cpu()
    read_indices = batch.episodes.query_read_index.detach().cpu()
    for row, read_index in zip(tokens, read_indices, strict=True):
        end = int(read_index) + 1
        prefix = tuple(int(value) for value in row[:end])
        try:
            operation, values = prefix_programs[prefix]
        except KeyError as exc:
            raise ETTRQuerySupervisionError(
                "batch query prefix is absent from its semantic core"
            ) from exc
        width = len(values)
        operations.append(operation)
        arguments.append((*values, *(0 for _ in range(MAX_QUERY_ARGUMENTS - width))))
        masks.append(tuple(index < width for index in range(MAX_QUERY_ARGUMENTS)))
    result = ETTRQuerySpecBatch(
        operation=torch.tensor(operations, dtype=torch.long),
        arguments=torch.tensor(arguments, dtype=torch.long),
        argument_mask=torch.tensor(masks, dtype=torch.bool),
    )
    result.validate(batch_size=tokens.shape[0])
    return result


def _iter_positioned_records(
    stream: ETTRV3StreamingRelease,
    split: str,
    *,
    epoch: int,
    seed: int,
    start_position: int,
) -> Iterator[tuple[int, object]]:
    """Mirror the immutable stream order without materializing tensors."""

    from ettr_v3_streaming import _identity
    from materialize_ettr_il_v3_corpus import _iter_records, _sha256_file

    if split not in {"train", "development"}:
        raise ETTRQuerySupervisionError("query-supervision split differs")
    iterators = []
    identities = []
    for path_value, descriptor in stream._ordered_shards(split, epoch, seed):
        path = stream.data_root / path_value
        before = _identity(
            path,
            "ETTR query-supervision shard",
            require_immutable=True,
        )
        digest, size = _sha256_file(path)
        if digest != descriptor["sha256"] or size != descriptor["bytes"]:
            raise ETTRQuerySupervisionError(
                "query-supervision shard identity differs"
            )
        iterators.append([path_value, descriptor, enumerate(_iter_records(path)), 0])
        identities.append((Path(path), before))

    position = 0
    while iterators:
        next_round = []
        for path_value, descriptor, iterator, observed in iterators:
            try:
                _row_index, (payload, record) = next(iterator)
            except StopIteration:
                if observed != descriptor["rows"]:
                    raise ETTRQuerySupervisionError(
                        "query-supervision shard row count differs"
                    )
                continue
            next_round.append([path_value, descriptor, iterator, observed + 1])
            if (
                record.canonical_bytes() != payload
                or record.identity.split != split
            ):
                raise ETTRQuerySupervisionError(
                    "query-supervision semantic core differs"
                )
            for _batch_index in range(4):
                if position >= start_position:
                    yield position, record
                position += 1
        iterators = next_round
    for path, before in identities:
        if (
            _identity(
                path,
                "ETTR query-supervision shard",
                require_immutable=True,
            )
            != before
        ):
            raise ETTRQuerySupervisionError(
                "query-supervision shard changed while streaming"
            )


def iter_batches_with_query_specs(
    stream: ETTRV3StreamingRelease,
    split: str,
    *,
    epoch: int,
    seed: int,
    start_position: int = 0,
) -> Iterator[tuple[int, ETTRContinuationBatch, ETTRQuerySpecBatch]]:
    """Yield immutable batches plus training-only query-program targets."""

    batches = stream.iter_positioned_batches(
        split,
        rank=0,
        world_size=1,
        epoch=epoch,
        seed=seed,
        start_position=start_position,
    )
    records = _iter_positioned_records(
        stream,
        split,
        epoch=epoch,
        seed=seed,
        start_position=start_position,
    )
    for batch_item, record_item in zip(batches, records, strict=True):
        position, batch = batch_item
        record_position, record = record_item
        if position != record_position:
            raise ETTRV3StreamingError(
                "query-supervision and tensor stream positions differ"
            )
        yield (
            position,
            batch,
            query_specs_for_batch(record, batch, tokenizer=stream.tokenizer),
        )


__all__ = [
    "ETTRQuerySpecBatch",
    "ETTRQuerySupervisionError",
    "MAX_QUERY_ARGUMENTS",
    "MAX_QUERY_ARGUMENT_VALUE",
    "QUERY_OPERATIONS",
    "iter_batches_with_query_specs",
    "query_specs_for_batch",
]
