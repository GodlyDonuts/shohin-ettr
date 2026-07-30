"""Exact CPU projection of stored ETTR v3 packet/query contexts.

The optimizer release normally obtains packet-sufficiency commitments after
fully rebuilding every torch tensor.  Split audits need the same commitments
for millions of rows, but they do not need WORLD/COMMAND tensors, traces, or
equivariance ledgers.  This module reconstructs only the terminal packet and
query prefix that define ``terminal_packet_query_context``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterator

from ettr_il_v2_materialize import (
    NUM_RELATIONS,
    NUM_SLOTS,
    _answer_codes,
    _encode_packet,
    _project_initial,
    _tokenize_query,
)
from ettr_il_v3_materialize import (
    _corner_from_targets,
    _packet_from_value,
)
from ettr_il_v3_protocol import ROWS_PER_CORE
from ettr_il_v3_shards import SemanticCoreRecord


class ETTRV3PacketContextError(ValueError):
    """A stored core cannot produce the deployed packet/query projection."""


@dataclass(frozen=True, slots=True)
class ETTRV3PacketContextRow:
    """One exact row consumed by the packet-sufficiency index."""

    context_digest: bytes
    target: int
    view: int
    query: int
    paraphrase: int
    world: int
    command: int

    def validate(self) -> None:
        if (
            type(self.context_digest) is not bytes
            or len(self.context_digest) != 32
            or type(self.target) is not int
            or not 0 <= self.target <= 0xFFFF
            or any(
                type(value) is not int or not 0 <= value < 4
                for value in (self.view,)
            )
            or any(
                type(value) is not int or not 0 <= value < 2
                for value in (
                    self.query,
                    self.paraphrase,
                    self.world,
                    self.command,
                )
            )
        ):
            raise ETTRV3PacketContextError("packet-context row differs")


def _canonical_packet_context_bytes(value: object) -> bytes:
    """Match ``train.ettr_packet_index._canonical_json_bytes`` exactly."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _dense_relations(
    relations: frozenset[tuple[int, int, int]],
) -> list[list[list[bool]]]:
    result = [
        [[False] * NUM_SLOTS for _ in range(NUM_SLOTS)]
        for _ in range(NUM_RELATIONS)
    ]
    for relation, source, target in relations:
        result[relation][source][target] = True
    return result


def _packet_value(packet: object) -> dict[str, object]:
    active = tuple(bool(value) for value in packet.active)
    return {
        "active": list(active),
        "committed": bool(packet.committed),
        "halted": bool(packet.halted),
        "relations": _dense_relations(packet.relations),
        "root": [bool(value) for value in packet.root],
        "type_index": [
            int(value) if active[index] else 0
            for index, value in enumerate(packet.type_index)
        ],
        "value_code": [
            int(value) if active[index] else 0
            for index, value in enumerate(packet.value_code)
        ],
    }


def _packet_prefix_hasher(packet: object) -> object:
    prefix = (
        b'{"packet":'
        + _canonical_packet_context_bytes(_packet_value(packet))
        + b',"query":'
    )
    return hashlib.sha256(prefix)


def _context_digest(
    packet_prefix: object,
    query_tokens: tuple[int, ...],
    read_index: int,
) -> bytes:
    query = {
        "mask": [True] * len(query_tokens),
        "read_index": read_index,
        "tokens": list(query_tokens),
    }
    digest = packet_prefix.copy()
    digest.update(_canonical_packet_context_bytes(query))
    digest.update(b"}")
    return digest.digest()


def iter_packet_context_rows(
    record: SemanticCoreRecord,
    tokenizer: object,
) -> Iterator[ETTRV3PacketContextRow]:
    """Yield the exact 64 packet/query commitments for one stored core."""

    if not isinstance(record, SemanticCoreRecord):
        raise ETTRV3PacketContextError("semantic-core record type differs")
    record.validate()
    targets = record.assessor_only.targets
    if (
        len(targets.initial_packets) != 2
        or len(targets.terminal_packets) != 4
        or len(targets.transaction_traces) != 4
        or len(targets.answer_matrix) != 4
        or len(record.source_visible.views) != 4
    ):
        raise ETTRV3PacketContextError("semantic-core target geometry differs")

    initial_packets = tuple(
        _packet_from_value(value, f"initial packet {index}")
        for index, value in enumerate(targets.initial_packets)
    )
    terminal_packets = tuple(
        _packet_from_value(value, f"terminal packet {index}")
        for index, value in enumerate(targets.terminal_packets)
    )
    corners = tuple(
        _corner_from_targets(
            terminal_packets[index],
            targets.transaction_traces[index],
            targets.answer_matrix[index],
            f"corner {index}",
        )
        for index in range(4)
    )
    static_ranks = tuple(
        _project_initial(packet, f"initial packet {index}")[1]
        for index, packet in enumerate(initial_packets)
    )
    encoded_terminals = tuple(
        _encode_packet(
            terminal_packets[2 * world + command],
            static_ranks[world],
            f"terminal packet {2 * world + command}",
        )
        for world in range(2)
        for command in range(2)
    )
    packet_prefixes = tuple(
        _packet_prefix_hasher(packet) for packet in encoded_terminals
    )
    answer_codes = tuple(
        _answer_codes(corner, f"corner {index}")
        for index, corner in enumerate(corners)
    )
    vocab_size = tokenizer.get_vocab_size()
    observed = 0
    for view_index, view in enumerate(record.source_visible.views):
        if len(view.query_sources) != 4:
            raise ETTRV3PacketContextError("query source geometry differs")
        for query_index in range(2):
            for paraphrase_index in range(2):
                prefix = view.query_sources[
                    2 * query_index + paraphrase_index
                ].encode("ascii")
                for world_index in range(2):
                    for command_index in range(2):
                        corner_index = 2 * world_index + command_index
                        segment, read_index, target = _tokenize_query(
                            tokenizer,
                            prefix,
                            answer_codes[corner_index][query_index],
                            vocab_size,
                            (
                                f"view {view_index} query "
                                f"{query_index}/{paraphrase_index} corner "
                                f"{world_index}{command_index}"
                            ),
                        )
                        query_tokens = segment.tokens[: read_index + 1]
                        result = ETTRV3PacketContextRow(
                            context_digest=_context_digest(
                                packet_prefixes[corner_index],
                                query_tokens,
                                read_index,
                            ),
                            target=target,
                            view=view_index,
                            query=query_index,
                            paraphrase=paraphrase_index,
                            world=world_index,
                            command=command_index,
                        )
                        result.validate()
                        observed += 1
                        yield result
    if observed != ROWS_PER_CORE:
        raise ETTRV3PacketContextError("packet-context row count differs")


def compact_packet_context_rows(
    record: SemanticCoreRecord,
    tokenizer: object,
) -> tuple[tuple[bytes, int], ...]:
    """Return only the digest/target pairs used by the disk index."""

    return tuple(
        (row.context_digest, row.target)
        for row in iter_packet_context_rows(record, tokenizer)
    )


__all__ = [
    "ETTRV3PacketContextError",
    "ETTRV3PacketContextRow",
    "compact_packet_context_rows",
    "iter_packet_context_rows",
]
