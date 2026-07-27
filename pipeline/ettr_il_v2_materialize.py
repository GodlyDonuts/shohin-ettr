"""CPU-only ETTR IL v2 packet and transaction materialization.

The public input types are deliberately ontology-neutral.  An upstream
assessor supplies canonical packet cells, generic mutations, query prefixes,
and independently computed terminal packets.  This module performs only the
frozen projection described by
``R12_ETTR_IL_V2_MATERIALIZATION_SPEC.md``:

* rank static values into the global categorical codebook;
* project cells into the fixed 64-slot packet;
* synthesize and independently replay 64-position generic transactions;
* expand each semantic rectangle into four causal rectangles and 16 rows;
* enforce one-token answer boundaries and exact segment widths; and
* construct and validate the existing ETTR continuation dataclasses.

There is no ontology parser, oracle, training loop, job submission, or
checkpoint access in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Protocol, Sequence, runtime_checkable


PROTOCOL = "R12-ETTR-IL-v2-materialization-v1"
NUM_SLOTS = 64
NUM_TYPES = 8
NUM_RELATIONS = 16
NUM_VALUE_CODES = 256
MAX_EDGES = 256
MAX_STEPS = 64
WORLD_WIDTH = 192
COMMAND_WIDTH = 96
QUERY_WIDTH = 48
PAD_TOKEN_ID = 0

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATIC_SLOTS = range(0, 32)
_RUNTIME_SLOTS = range(32, 48)
_COMMAND_SLOTS = range(48, 54)
_CURSOR_SLOT = 54
_OUTCOME_SLOT = 55
_RESERVED_SLOTS = range(56, 64)
_IDENTITY_SLOTS = tuple(range(NUM_SLOTS))
_IDENTITY_TYPES = tuple(range(NUM_TYPES))
_IDENTITY_RELATIONS = tuple(range(NUM_RELATIONS))
_IDENTITY_VALUES = tuple(range(NUM_VALUE_CODES))


class MaterializationError(ValueError):
    """An ETTR IL v2 input cannot be represented without information loss."""


class Opcode(IntEnum):
    ALLOC = 0
    WRITE = 1
    CLEAR = 2
    LINK = 3
    UNLINK = 4
    SET_ROOT = 5
    COMMIT = 6
    HALT = 7
    REJECT = 8


class Disposition(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    REJECT = "reject"


class ValueKind(StrEnum):
    EMPTY = "empty"
    STATIC_RAW = "static_raw"
    LOCAL_ID = "local_id"
    SMALL_UINT = "small_uint"
    COMMAND_ATOM = "command_atom"
    EXECUTE = "execute"
    ABSTAIN = "abstain"
    REJECT = "reject"
    PROCESS_HALT = "process_halt"
    PROCESS_DEADLOCK = "process_deadlock"


@dataclass(frozen=True, slots=True)
class ValueRef:
    """A symbolic value whose integer code is fixed by Section 5.4."""

    kind: ValueKind
    index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ValueKind):
            raise MaterializationError("value kind differs")
        indexed_limits = {
            ValueKind.STATIC_RAW: None,
            ValueKind.LOCAL_ID: 32,
            ValueKind.SMALL_UINT: 16,
            ValueKind.COMMAND_ATOM: 64,
        }
        if self.kind in indexed_limits:
            if not _plain_int(self.index):
                raise MaterializationError(f"{self.kind.value} requires an integer")
            limit = indexed_limits[self.kind]
            if limit is not None and not 0 <= self.index < limit:
                raise MaterializationError(f"{self.kind.value} leaves its range")
        elif self.index is not None:
            raise MaterializationError(f"{self.kind.value} cannot carry an index")

    @classmethod
    def empty(cls) -> "ValueRef":
        return cls(ValueKind.EMPTY)

    @classmethod
    def static(cls, raw_value: int) -> "ValueRef":
        return cls(ValueKind.STATIC_RAW, raw_value)

    @classmethod
    def local_id(cls, index: int) -> "ValueRef":
        return cls(ValueKind.LOCAL_ID, index)

    @classmethod
    def small_uint(cls, value: int) -> "ValueRef":
        return cls(ValueKind.SMALL_UINT, value)

    @classmethod
    def command_atom(cls, index: int) -> "ValueRef":
        return cls(ValueKind.COMMAND_ATOM, index)

    @classmethod
    def execute(cls) -> "ValueRef":
        return cls(ValueKind.EXECUTE)

    @classmethod
    def abstain(cls) -> "ValueRef":
        return cls(ValueKind.ABSTAIN)

    @classmethod
    def reject(cls) -> "ValueRef":
        return cls(ValueKind.REJECT)


@dataclass(frozen=True, slots=True)
class GenericCell:
    slot: int
    type_index: int
    value: ValueRef


@dataclass(frozen=True, order=True, slots=True)
class GenericEdge:
    relation: int
    source: int
    target: int


@dataclass(frozen=True, slots=True)
class GenericPacket:
    """Sparse generic packet with optional explicit full-support claims."""

    cells: tuple[GenericCell, ...]
    edges: tuple[GenericEdge, ...] = ()
    root: int | None = None
    committed: bool = False
    halted: bool = False
    slot_support: tuple[bool, ...] | None = None
    relation_support: tuple[bool, ...] | None = None


@dataclass(frozen=True, slots=True)
class GenericWorld:
    # The receiving rectangle contract requires raw WORLD contrast along the
    # COMMAND nuisance axis.  Both bytestrings must therefore be independently
    # supplied, meaning-preserving renderings of this same semantic world.
    sources: tuple[bytes, bytes]
    initial_packet: GenericPacket


@dataclass(frozen=True, slots=True)
class GenericCommand:
    # The receiving rectangle contract likewise requires raw COMMAND contrast
    # along the WORLD nuisance axis.
    sources: tuple[bytes, bytes]
    command_atoms: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GenericQuery:
    prefixes: tuple[bytes, bytes]


@dataclass(frozen=True, slots=True)
class GenericMutation:
    """One ontology-neutral state mutation inside a semantic operation."""

    opcode: Opcode
    source: int
    target: int = 0
    relation: int = 0
    type_index: int = 0
    value: ValueRef = field(default_factory=ValueRef.empty)


@dataclass(frozen=True, slots=True)
class GenericOperationTrace:
    mutations: tuple[GenericMutation, ...]
    cursor: int


@dataclass(frozen=True, slots=True)
class GenericCorner:
    """One independently labeled ``(WORLD, COMMAND)`` execution."""

    operation_traces: tuple[GenericOperationTrace, ...]
    terminal_packet: GenericPacket
    disposition: Disposition
    outcome: ValueRef
    answers: tuple[bool | None, bool | None]


@dataclass(frozen=True, slots=True)
class GenericSemanticRectangle:
    semantic_rectangle_id: str
    presentation_id: str
    worlds: tuple[GenericWorld, GenericWorld]
    commands: tuple[GenericCommand, GenericCommand]
    queries: tuple[GenericQuery, GenericQuery]
    # Nested in WORLD-major, COMMAND-minor order.
    corners: tuple[
        tuple[GenericCorner, GenericCorner],
        tuple[GenericCorner, GenericCorner],
    ]


@dataclass(frozen=True, slots=True)
class GenericInvariantPair:
    """A post-quotient, lossless right-to-left rectangle alignment."""

    left_rectangle: int
    right_rectangle: int
    slot_permutation: tuple[int, ...] = _IDENTITY_SLOTS
    type_permutation: tuple[int, ...] = _IDENTITY_TYPES
    relation_permutation: tuple[int, ...] = _IDENTITY_RELATIONS
    value_permutation: tuple[int, ...] = _IDENTITY_VALUES


@dataclass(frozen=True, slots=True)
class MaterializationRequest:
    manifest_sha256: str
    dataset_sha256: str
    vocab_size: int
    rectangles: tuple[GenericSemanticRectangle, ...]
    invariant_pairs: tuple[GenericInvariantPair, ...] = ()
    require_query_checkerboard: bool = True


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Minimal injected tokenizer surface; no tokenizer file is opened here."""

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class _PacketImage:
    value_code: tuple[int, ...]
    type_index: tuple[int, ...]
    relations: frozenset[tuple[int, int, int]]
    active: tuple[bool, ...]
    root: tuple[bool, ...]
    committed: bool
    halted: bool


@dataclass(frozen=True, slots=True)
class _EncodedStep:
    opcode: int
    source: int
    target: int
    relation: int
    type_index: int
    value_code: int


@dataclass(frozen=True, slots=True)
class _Trace:
    opcode: tuple[int, ...]
    source: tuple[int, ...]
    target: tuple[int, ...]
    relation: tuple[int, ...]
    type_index: tuple[int, ...]
    value_code: tuple[int, ...]
    committed: tuple[bool, ...]
    halted: tuple[bool, ...]
    step_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class _Segment:
    tokens: tuple[int, ...]
    mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class _PreparedRow:
    episode_id: str
    world: _Segment
    command: _Segment
    query: _Segment
    query_read_index: int
    answer_token: int
    initial: _PacketImage
    terminal: _PacketImage
    trace: _Trace
    source_identity: tuple[bytes, bytes, bytes]


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_ascii(value: bytes, name: str) -> str:
    if not isinstance(value, bytes) or not value:
        raise MaterializationError(f"{name} must be nonempty bytes")
    try:
        return value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise MaterializationError(f"{name} must be strict ASCII") from exc


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


def _token_ids(tokenizer: TokenizerProtocol, source: bytes, name: str) -> tuple[int, ...]:
    text = _require_ascii(source, name)
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    ids = encoded.ids if hasattr(encoded, "ids") else encoded
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
        raise MaterializationError(f"{name} tokenizer output differs")
    result = tuple(ids)
    if not result or any(not _plain_int(item) or item < 0 for item in result):
        raise MaterializationError(f"{name} tokenizer IDs differ")
    return result


def _pad(ids: tuple[int, ...], width: int, name: str, vocab_size: int) -> _Segment:
    if len(ids) < 2:
        raise MaterializationError(f"{name} has fewer than two tokens")
    if len(ids) > width:
        raise MaterializationError(f"{name} exceeds its exact {width}-token width")
    if any(item >= vocab_size for item in ids):
        raise MaterializationError(f"{name} token leaves the declared vocabulary")
    padding = width - len(ids)
    return _Segment(
        tokens=ids + (PAD_TOKEN_ID,) * padding,
        mask=(True,) * len(ids) + (False,) * padding,
    )


def _tokenize_query(
    tokenizer: TokenizerProtocol,
    prefix: bytes,
    answer_code: int,
    vocab_size: int,
    name: str,
) -> tuple[_Segment, int, int]:
    prefix_text = _require_ascii(prefix, f"{name} prefix")
    code_text = str(answer_code)
    prefix_ids = _token_ids(tokenizer, prefix, f"{name} prefix")
    boundary_ids = _token_ids(
        tokenizer,
        (prefix_text + code_text).encode("ascii"),
        f"{name} answer boundary",
    )
    if (
        len(boundary_ids) != len(prefix_ids) + 1
        or boundary_ids[: len(prefix_ids)] != prefix_ids
    ):
        raise MaterializationError(
            f"{name} answer boundary is not exactly one next token"
        )
    full_ids = _token_ids(
        tokenizer,
        (prefix_text + code_text + "\n").encode("ascii"),
        f"{name} full query",
    )
    if full_ids[: len(boundary_ids)] != boundary_ids:
        raise MaterializationError(
            f"{name} newline changes the one-token answer boundary"
        )
    segment = _pad(full_ids, QUERY_WIDTH, f"{name} QUERY", vocab_size)
    return segment, len(prefix_ids) - 1, boundary_ids[-1]


def _value_code(value: ValueRef, static_ranks: dict[int, int]) -> int:
    if not isinstance(value, ValueRef):
        raise MaterializationError("cell value type differs")
    if value.kind is ValueKind.EMPTY:
        return 0
    if value.kind is ValueKind.STATIC_RAW:
        try:
            return 1 + static_ranks[value.index]  # type: ignore[index]
        except KeyError as exc:
            raise MaterializationError(
                "terminal static value is absent from the WORLD codebook"
            ) from exc
    if value.kind is ValueKind.LOCAL_ID:
        return 33 + int(value.index)
    if value.kind is ValueKind.SMALL_UINT:
        return 65 + int(value.index)
    if value.kind is ValueKind.COMMAND_ATOM:
        return 81 + int(value.index)
    return {
        ValueKind.EXECUTE: 145,
        ValueKind.ABSTAIN: 146,
        ValueKind.REJECT: 147,
        ValueKind.PROCESS_HALT: 148,
        ValueKind.PROCESS_DEADLOCK: 149,
    }[value.kind]


def _validate_support(packet: GenericPacket, name: str) -> None:
    if packet.slot_support is not None:
        if (
            len(packet.slot_support) != NUM_SLOTS
            or any(not isinstance(value, bool) for value in packet.slot_support)
            or not all(packet.slot_support)
        ):
            raise MaterializationError(f"{name} has partial slot support")
    if packet.relation_support is not None:
        expected = NUM_RELATIONS * NUM_SLOTS * NUM_SLOTS
        if (
            len(packet.relation_support) != expected
            or any(
                not isinstance(value, bool)
                for value in packet.relation_support
            )
            or not all(packet.relation_support)
        ):
            raise MaterializationError(f"{name} has partial relation support")


def _validate_cell_projection(cell: GenericCell, name: str) -> None:
    if (
        not isinstance(cell, GenericCell)
        or not _plain_int(cell.slot)
        or not _plain_int(cell.type_index)
        or not isinstance(cell.value, ValueRef)
    ):
        raise MaterializationError(f"{name} cell differs")
    if cell.slot in _STATIC_SLOTS:
        if not 0 <= cell.type_index < 4 or cell.value.kind is not ValueKind.STATIC_RAW:
            raise MaterializationError(f"{name} static-cell projection differs")
    elif cell.slot in _RUNTIME_SLOTS:
        if cell.type_index != 4 or cell.value.kind not in {
            ValueKind.EMPTY,
            ValueKind.LOCAL_ID,
            ValueKind.SMALL_UINT,
        }:
            raise MaterializationError(f"{name} runtime-cell projection differs")
    elif cell.slot in _COMMAND_SLOTS:
        if cell.type_index != 5 or cell.value.kind not in {
            ValueKind.EMPTY,
            ValueKind.COMMAND_ATOM,
        }:
            raise MaterializationError(f"{name} command-cell projection differs")
    elif cell.slot == _CURSOR_SLOT:
        if cell.type_index != 6 or cell.value.kind not in {
            ValueKind.EMPTY,
            ValueKind.SMALL_UINT,
        }:
            raise MaterializationError(f"{name} cursor-cell projection differs")
    elif cell.slot == _OUTCOME_SLOT:
        if cell.type_index != 6 or cell.value.kind not in {
            ValueKind.EMPTY,
            ValueKind.EXECUTE,
            ValueKind.ABSTAIN,
            ValueKind.REJECT,
            ValueKind.PROCESS_HALT,
            ValueKind.PROCESS_DEADLOCK,
        }:
            raise MaterializationError(f"{name} outcome-cell projection differs")
    elif cell.slot in _RESERVED_SLOTS:
        raise MaterializationError(f"{name} activates a reserved slot")
    else:
        raise MaterializationError(f"{name} slot leaves the packet")


def _static_ranks(packet: GenericPacket, name: str) -> dict[int, int]:
    values = sorted(
        {
            cell.value.index
            for cell in packet.cells
            if cell.slot in _STATIC_SLOTS
            and isinstance(cell.value, ValueRef)
            and cell.value.kind is ValueKind.STATIC_RAW
        }
    )
    if not values:
        raise MaterializationError(f"{name} contains no static values")
    if len(values) > 32:
        raise MaterializationError(f"{name} exceeds 32 static value ranks")
    return {int(value): index for index, value in enumerate(values)}


def _project_initial(
    packet: GenericPacket,
    name: str,
) -> tuple[_PacketImage, dict[int, int]]:
    if packet.committed or packet.halted:
        raise MaterializationError(f"{name} initial packet is not open")
    if any(48 <= cell.slot < 56 for cell in packet.cells):
        raise MaterializationError(
            f"{name} initial packet must not supply materializer-owned controls"
        )
    projected = GenericPacket(
        cells=packet.cells
        + tuple(
            GenericCell(slot, 5, ValueRef.empty())
            for slot in _COMMAND_SLOTS
        )
        + (
            GenericCell(_CURSOR_SLOT, 6, ValueRef.empty()),
            GenericCell(_OUTCOME_SLOT, 6, ValueRef.empty()),
        ),
        edges=packet.edges,
        root=packet.root,
        committed=False,
        halted=False,
        slot_support=packet.slot_support,
        relation_support=packet.relation_support,
    )
    ranks = _static_ranks(projected, name)
    return _encode_packet(projected, ranks, name), ranks


def _encode_packet(
    packet: GenericPacket,
    static_ranks: dict[int, int],
    name: str,
) -> _PacketImage:
    if not isinstance(packet, GenericPacket):
        raise MaterializationError(f"{name} packet type differs")
    if not isinstance(packet.committed, bool) or not isinstance(
        packet.halted, bool
    ):
        raise MaterializationError(f"{name} packet status type differs")
    _validate_support(packet, name)
    cells: dict[int, GenericCell] = {}
    for cell in packet.cells:
        _validate_cell_projection(cell, name)
        if cell.slot in cells:
            raise MaterializationError(f"{name} has a duplicate slot")
        cells[cell.slot] = cell
    values = [0] * NUM_SLOTS
    types = [0] * NUM_SLOTS
    active = [False] * NUM_SLOTS
    for slot, cell in cells.items():
        values[slot] = _value_code(cell.value, static_ranks)
        types[slot] = cell.type_index
        active[slot] = True
    root = [False] * NUM_SLOTS
    if packet.root is not None:
        if not _plain_int(packet.root) or packet.root not in cells:
            raise MaterializationError(f"{name} root leaves active cells")
        root[packet.root] = True
    edges: set[tuple[int, int, int]] = set()
    for edge in packet.edges:
        if (
            not isinstance(edge, GenericEdge)
            or not _plain_int(edge.relation)
            or not 0 <= edge.relation < NUM_RELATIONS
            or not _plain_int(edge.source)
            or not _plain_int(edge.target)
            or edge.source not in cells
            or edge.target not in cells
        ):
            raise MaterializationError(f"{name} relation edge differs")
        encoded = (edge.relation, edge.source, edge.target)
        if encoded in edges:
            raise MaterializationError(f"{name} has a duplicate relation edge")
        edges.add(encoded)
    if len(edges) > MAX_EDGES:
        raise MaterializationError(f"{name} exceeds the 256-edge capacity")
    return _PacketImage(
        value_code=tuple(values),
        type_index=tuple(types),
        relations=frozenset(edges),
        active=tuple(active),
        root=tuple(root),
        committed=packet.committed,
        halted=packet.halted,
    )


def _canonical_operands(step: _EncodedStep, name: str) -> None:
    if (
        not 0 <= step.opcode <= int(Opcode.REJECT)
        or not 0 <= step.source < NUM_SLOTS
        or not 0 <= step.target < NUM_SLOTS
        or not 0 <= step.relation < NUM_RELATIONS
        or not 0 <= step.type_index < NUM_TYPES
        or not 0 <= step.value_code < NUM_VALUE_CODES
    ):
        raise MaterializationError(f"{name} operand leaves frozen geometry")
    opcode = Opcode(step.opcode)
    if opcode is Opcode.ALLOC:
        canonical = step.target == 0 and step.relation == 0
    elif opcode is Opcode.WRITE:
        canonical = (
            step.target == 0
            and step.relation == 0
            and step.type_index == 0
        )
    elif opcode in {Opcode.CLEAR, Opcode.SET_ROOT}:
        canonical = (
            step.target == 0
            and step.relation == 0
            and step.type_index == 0
            and step.value_code == 0
        )
    elif opcode in {Opcode.LINK, Opcode.UNLINK}:
        canonical = step.type_index == 0 and step.value_code == 0
    else:
        canonical = (
            step.source == 0
            and step.target == 0
            and step.relation == 0
            and step.type_index == 0
            and step.value_code == 0
        )
    if not canonical:
        raise MaterializationError(f"{name} has noncanonical unused operands")


def _encode_mutation(
    mutation: GenericMutation,
    static_ranks: dict[int, int],
    name: str,
) -> _EncodedStep:
    if not isinstance(mutation, GenericMutation) or not isinstance(
        mutation.opcode, Opcode
    ):
        raise MaterializationError(f"{name} mutation type differs")
    if mutation.opcode not in {Opcode.WRITE, Opcode.LINK, Opcode.UNLINK}:
        raise MaterializationError(
            f"{name} uses a non-v2 mutation opcode"
        )
    step = _EncodedStep(
        opcode=int(mutation.opcode),
        source=mutation.source,
        target=mutation.target,
        relation=mutation.relation,
        type_index=mutation.type_index,
        value_code=_value_code(mutation.value, static_ranks),
    )
    _canonical_operands(step, name)
    if mutation.opcode is Opcode.WRITE and mutation.source not in _RUNTIME_SLOTS:
        raise MaterializationError(f"{name} writes outside runtime registers")
    return step


def _independent_replay(
    initial: _PacketImage,
    steps: tuple[_EncodedStep, ...],
    name: str,
) -> tuple[_PacketImage, tuple[bool, ...], tuple[bool, ...]]:
    """Replay the nine opcodes without importing ETTR's torch recurrence."""

    if not steps or len(steps) > MAX_STEPS:
        raise MaterializationError(f"{name} leaves the 64-step capacity")
    values = list(initial.value_code)
    types = list(initial.type_index)
    active = list(initial.active)
    root = list(initial.root)
    relations = set(initial.relations)
    committed = initial.committed
    halted = initial.halted
    committed_trace: list[bool] = []
    halted_trace: list[bool] = []
    for position, step in enumerate(steps):
        step_name = f"{name} step {position}"
        _canonical_operands(step, step_name)
        if committed or halted:
            raise MaterializationError(f"{step_name} mutates a terminal packet")
        opcode = Opcode(step.opcode)
        before = (
            tuple(values),
            tuple(types),
            tuple(active),
            tuple(root),
            frozenset(relations),
            committed,
            halted,
        )
        if opcode is Opcode.ALLOC:
            if active[step.source]:
                raise MaterializationError(f"{step_name} allocates an active slot")
            active[step.source] = True
            values[step.source] = step.value_code
            types[step.source] = step.type_index
        elif opcode is Opcode.WRITE:
            if not active[step.source]:
                raise MaterializationError(f"{step_name} writes an inactive slot")
            values[step.source] = step.value_code
        elif opcode is Opcode.CLEAR:
            if not active[step.source]:
                raise MaterializationError(f"{step_name} clears an inactive slot")
            active[step.source] = False
            values[step.source] = 0
            types[step.source] = 0
            root[step.source] = False
            relations = {
                edge
                for edge in relations
                if edge[1] != step.source and edge[2] != step.source
            }
        elif opcode in {Opcode.LINK, Opcode.UNLINK}:
            if not active[step.source] or not active[step.target]:
                raise MaterializationError(f"{step_name} touches an inactive endpoint")
            edge = (step.relation, step.source, step.target)
            if opcode is Opcode.LINK:
                if edge in relations:
                    raise MaterializationError(f"{step_name} repeats an existing edge")
                relations.add(edge)
            else:
                if edge not in relations:
                    raise MaterializationError(f"{step_name} removes a missing edge")
                relations.remove(edge)
        elif opcode is Opcode.SET_ROOT:
            if not active[step.source]:
                raise MaterializationError(f"{step_name} roots an inactive slot")
            root = [index == step.source for index in range(NUM_SLOTS)]
        elif opcode is Opcode.COMMIT:
            committed = True
        elif opcode is Opcode.HALT:
            halted = True
        elif opcode is Opcode.REJECT:
            committed = True
            halted = True
        if len(relations) > MAX_EDGES:
            raise MaterializationError(f"{step_name} exceeds the 256-edge capacity")
        after = (
            tuple(values),
            tuple(types),
            tuple(active),
            tuple(root),
            frozenset(relations),
            committed,
            halted,
        )
        if before == after:
            raise MaterializationError(f"{step_name} has no generic state effect")
        committed_trace.append(committed)
        halted_trace.append(halted)
    return (
        _PacketImage(
            value_code=tuple(values),
            type_index=tuple(types),
            relations=frozenset(relations),
            active=tuple(active),
            root=tuple(root),
            committed=committed,
            halted=halted,
        ),
        tuple(committed_trace),
        tuple(halted_trace),
    )


def _expected_status(disposition: Disposition) -> tuple[bool, bool, Opcode]:
    if disposition is Disposition.ANSWER:
        return True, False, Opcode.COMMIT
    if disposition is Disposition.ABSTAIN:
        return False, True, Opcode.HALT
    if disposition is Disposition.REJECT:
        return True, True, Opcode.REJECT
    raise MaterializationError("corner disposition differs")


def _answer_codes(corner: GenericCorner, name: str) -> tuple[int, int]:
    if (
        not isinstance(corner.disposition, Disposition)
        or len(corner.answers) != 2
    ):
        raise MaterializationError(f"{name} answer contract differs")
    if corner.disposition is Disposition.ANSWER:
        if any(not isinstance(value, bool) for value in corner.answers):
            raise MaterializationError(f"{name} answer target differs")
        return tuple(int(value) for value in corner.answers)  # type: ignore[return-value]
    if any(value is not None for value in corner.answers):
        raise MaterializationError(f"{name} non-answer target must be null")
    code = 2 if corner.disposition is Disposition.ABSTAIN else 3
    return code, code


def _build_trace(
    initial: _PacketImage,
    command: GenericCommand,
    corner: GenericCorner,
    static_ranks: dict[int, int],
    name: str,
) -> tuple[_PacketImage, _Trace]:
    if (
        not isinstance(command, GenericCommand)
        or not 1 <= len(command.command_atoms) <= 6
        or any(
            not _plain_int(atom) or not 0 <= atom < 64
            for atom in command.command_atoms
        )
    ):
        raise MaterializationError(f"{name} command depth or atom differs")
    if len(corner.operation_traces) != len(command.command_atoms):
        raise MaterializationError(f"{name} operation trace depth differs")
    deadlock_cursor: int | None = None
    if corner.outcome.kind is ValueKind.PROCESS_DEADLOCK:
        candidate = corner.operation_traces[-1].cursor
        if (
            not _plain_int(candidate)
            or not 0 <= candidate < len(command.command_atoms)
        ):
            raise MaterializationError(f"{name} deadlock cursor differs")
        deadlock_cursor = candidate
    encoded: list[_EncodedStep] = []
    for index, (atom, operation) in enumerate(
        zip(command.command_atoms, corner.operation_traces, strict=True)
    ):
        expected_cursor = (
            min(index + 1, deadlock_cursor)
            if deadlock_cursor is not None
            else index + 1
        )
        if (
            not isinstance(operation, GenericOperationTrace)
            or not _plain_int(operation.cursor)
            or operation.cursor != expected_cursor
        ):
            raise MaterializationError(f"{name} operation cursor differs")
        if (
            deadlock_cursor is not None
            and index >= deadlock_cursor
            and operation.mutations
        ):
            raise MaterializationError(
                f"{name} mutates state after the deadlock cursor"
            )
        encoded.append(
            _EncodedStep(
                int(Opcode.WRITE),
                48 + index,
                0,
                0,
                0,
                _value_code(ValueRef.command_atom(atom), static_ranks),
            )
        )
        encoded.extend(
            _encode_mutation(mutation, static_ranks, name)
            for mutation in operation.mutations
        )
        encoded.append(
            _EncodedStep(
                int(Opcode.WRITE),
                _CURSOR_SLOT,
                0,
                0,
                0,
                _value_code(ValueRef.small_uint(operation.cursor), static_ranks),
            )
        )
    if corner.disposition is Disposition.ABSTAIN:
        if corner.outcome.kind is not ValueKind.ABSTAIN:
            raise MaterializationError(f"{name} abstention outcome differs")
    elif corner.disposition is Disposition.REJECT:
        if corner.outcome.kind is not ValueKind.REJECT:
            raise MaterializationError(f"{name} rejection outcome differs")
    elif corner.outcome.kind not in {
        ValueKind.EXECUTE,
        ValueKind.PROCESS_HALT,
        ValueKind.PROCESS_DEADLOCK,
    }:
        raise MaterializationError(f"{name} answer outcome differs")
    encoded.append(
        _EncodedStep(
            int(Opcode.WRITE),
            _OUTCOME_SLOT,
            0,
            0,
            0,
            _value_code(corner.outcome, static_ranks),
        )
    )
    expected_committed, expected_halted, final_opcode = _expected_status(
        corner.disposition
    )
    encoded.append(_EncodedStep(int(final_opcode), 0, 0, 0, 0, 0))
    if len(encoded) > MAX_STEPS:
        raise MaterializationError(f"{name} trace exceeds 64 steps")
    replayed, committed_trace, halted_trace = _independent_replay(
        initial,
        tuple(encoded),
        name,
    )
    expected = _encode_packet(corner.terminal_packet, static_ranks, f"{name} terminal")
    if (
        expected.committed != expected_committed
        or expected.halted != expected_halted
    ):
        raise MaterializationError(f"{name} terminal disposition differs")
    if replayed != expected:
        raise MaterializationError(
            f"{name} independent replay differs from the oracle terminal"
        )
    padding = MAX_STEPS - len(encoded)
    return (
        replayed,
        _Trace(
            opcode=tuple(step.opcode for step in encoded) + (0,) * padding,
            source=tuple(step.source for step in encoded) + (0,) * padding,
            target=tuple(step.target for step in encoded) + (0,) * padding,
            relation=tuple(step.relation for step in encoded) + (0,) * padding,
            type_index=tuple(step.type_index for step in encoded) + (0,) * padding,
            value_code=tuple(step.value_code for step in encoded) + (0,) * padding,
            committed=committed_trace + (expected_committed,) * padding,
            halted=halted_trace + (expected_halted,) * padding,
            step_mask=(True,) * len(encoded) + (False,) * padding,
        ),
    )


def _packets_differ(left: _PacketImage, right: _PacketImage) -> bool:
    return left != right


def _validate_rectangle_shape(rectangle: GenericSemanticRectangle, index: int) -> None:
    name = f"rectangle {index}"
    if (
        not isinstance(rectangle, GenericSemanticRectangle)
        or not isinstance(rectangle.semantic_rectangle_id, str)
        or not rectangle.semantic_rectangle_id
        or not rectangle.semantic_rectangle_id.isascii()
        or not isinstance(rectangle.presentation_id, str)
        or not rectangle.presentation_id
        or not rectangle.presentation_id.isascii()
        or len(rectangle.worlds) != 2
        or len(rectangle.commands) != 2
        or len(rectangle.queries) != 2
        or len(rectangle.corners) != 2
        or any(len(row) != 2 for row in rectangle.corners)
    ):
        raise MaterializationError(f"{name} geometry differs")
    for query_index, query in enumerate(rectangle.queries):
        if not isinstance(query, GenericQuery) or len(query.prefixes) != 2:
            raise MaterializationError(
                f"{name} query {query_index} geometry differs"
            )
        if query.prefixes[0] == query.prefixes[1]:
            raise MaterializationError(
                f"{name} query {query_index} paraphrases are identical"
            )


def _prepare_rectangle(
    rectangle: GenericSemanticRectangle,
    rectangle_index: int,
    tokenizer: TokenizerProtocol,
    vocab_size: int,
    *,
    require_query_checkerboard: bool,
) -> list[_PreparedRow]:
    _validate_rectangle_shape(rectangle, rectangle_index)
    name = f"rectangle {rectangle_index}"
    # WORLD segments are indexed ``2 * world + command_nuisance``.
    world_segments: list[_Segment] = []
    initial_packets: list[_PacketImage] = []
    static_ranks: list[dict[int, int]] = []
    for world_index, world in enumerate(rectangle.worlds):
        if not isinstance(world, GenericWorld) or len(world.sources) != 2:
            raise MaterializationError(f"{name} WORLD {world_index} differs")
        variants: list[_Segment] = []
        for variant_index, source in enumerate(world.sources):
            ids = _token_ids(
                tokenizer,
                source,
                f"{name} WORLD {world_index}/{variant_index}",
            )
            variants.append(
                _pad(
                    ids,
                    WORLD_WIDTH,
                    f"{name} WORLD {world_index}/{variant_index}",
                    vocab_size,
                )
            )
        if variants[0] == variants[1]:
            raise MaterializationError(
                f"{name} WORLD {world_index} nuisance renderings tokenize "
                "identically"
            )
        world_segments.extend(variants)
        packet, ranks = _project_initial(
            world.initial_packet,
            f"{name} WORLD {world_index}",
        )
        initial_packets.append(packet)
        static_ranks.append(ranks)
    if any(
        world_segments[command_index] == world_segments[2 + command_index]
        for command_index in range(2)
    ):
        raise MaterializationError(f"{name} WORLD factors tokenize identically")
    if not _packets_differ(initial_packets[0], initial_packets[1]):
        raise MaterializationError(f"{name} WORLD packets are identical")

    # COMMAND segments are indexed ``2 * command + world_nuisance``.
    command_segments: list[_Segment] = []
    for command_index, command in enumerate(rectangle.commands):
        if not isinstance(command, GenericCommand) or len(command.sources) != 2:
            raise MaterializationError(f"{name} COMMAND {command_index} differs")
        variants = []
        for variant_index, source in enumerate(command.sources):
            ids = _token_ids(
                tokenizer,
                source,
                f"{name} COMMAND {command_index}/{variant_index}",
            )
            variants.append(
                _pad(
                    ids,
                    COMMAND_WIDTH,
                    f"{name} COMMAND {command_index}/{variant_index}",
                    vocab_size,
                )
            )
        if variants[0] == variants[1]:
            raise MaterializationError(
                f"{name} COMMAND {command_index} nuisance renderings tokenize "
                "identically"
            )
        command_segments.extend(variants)
    if any(
        command_segments[world_index] == command_segments[2 + world_index]
        for world_index in range(2)
    ):
        raise MaterializationError(f"{name} COMMAND factors tokenize identically")

    terminals: list[list[_PacketImage]] = [[], []]
    traces: list[list[_Trace]] = [[], []]
    answer_codes: list[list[tuple[int, int]]] = [[], []]
    for world_index in range(2):
        for command_index in range(2):
            corner = rectangle.corners[world_index][command_index]
            if not isinstance(corner, GenericCorner):
                raise MaterializationError(
                    f"{name} corner {world_index}{command_index} differs"
                )
            codes = _answer_codes(
                corner,
                f"{name} corner {world_index}{command_index}",
            )
            terminal, trace = _build_trace(
                initial_packets[world_index],
                rectangle.commands[command_index],
                corner,
                static_ranks[world_index],
                f"{name} corner {world_index}{command_index}",
            )
            terminals[world_index].append(terminal)
            traces[world_index].append(trace)
            answer_codes[world_index].append(codes)
    for left, right, edge_name in (
        (terminals[0][0], terminals[1][0], "WORLD/C0"),
        (terminals[0][1], terminals[1][1], "WORLD/C1"),
        (terminals[0][0], terminals[0][1], "COMMAND/W0"),
        (terminals[1][0], terminals[1][1], "COMMAND/W1"),
    ):
        if not _packets_differ(left, right):
            raise MaterializationError(
                f"{name} {edge_name} has no terminal packet contrast"
            )

    rows: list[_PreparedRow] = []
    for query_index in range(2):
        query = rectangle.queries[query_index]
        for paraphrase_index in range(2):
            prefix = query.prefixes[paraphrase_index]
            query_by_corner: list[list[tuple[_Segment, int, int]]] = [[], []]
            for world_index in range(2):
                for command_index in range(2):
                    query_by_corner[world_index].append(
                        _tokenize_query(
                            tokenizer,
                            prefix,
                            answer_codes[world_index][command_index][query_index],
                            vocab_size,
                            (
                                f"{name} query {query_index}/{paraphrase_index} "
                                f"corner {world_index}{command_index}"
                            ),
                        )
                    )
            read_indices = {
                item[1]
                for world in query_by_corner
                for item in world
            }
            if len(read_indices) != 1:
                raise MaterializationError(f"{name} query read indices differ")
            labels = [
                [query_by_corner[w][c][2] for c in range(2)]
                for w in range(2)
            ]
            if require_query_checkerboard:
                for left, right, edge_name in (
                    (labels[0][0], labels[1][0], "WORLD/C0"),
                    (labels[0][1], labels[1][1], "WORLD/C1"),
                    (labels[0][0], labels[0][1], "COMMAND/W0"),
                    (labels[1][0], labels[1][1], "COMMAND/W1"),
                ):
                    if left == right:
                        raise MaterializationError(
                            f"{name} {edge_name} has no query-label contrast"
                        )
            for world_index in range(2):
                for command_index in range(2):
                    query_segment, read_index, answer_token = (
                        query_by_corner[world_index][command_index]
                    )
                    identity = {
                        "protocol": PROTOCOL,
                        "semantic_rectangle_id": rectangle.semantic_rectangle_id,
                        "presentation_id": rectangle.presentation_id,
                        "semantic_index": query_index,
                        "paraphrase_index": paraphrase_index,
                        "world_index": world_index,
                        "command_index": command_index,
                    }
                    rows.append(
                        _PreparedRow(
                            episode_id=hashlib.sha256(
                                _canonical_json_bytes(identity)
                            ).hexdigest(),
                            world=world_segments[
                                2 * world_index + command_index
                            ],
                            command=command_segments[
                                2 * command_index + world_index
                            ],
                            query=query_segment,
                            query_read_index=read_index,
                            answer_token=answer_token,
                            initial=initial_packets[world_index],
                            terminal=terminals[world_index][command_index],
                            trace=traces[world_index][command_index],
                            source_identity=(
                                rectangle.worlds[world_index].sources[
                                    command_index
                                ],
                                rectangle.commands[command_index].sources[
                                    world_index
                                ],
                                prefix,
                            ),
                        )
                    )
    if len(rows) != 16:
        raise MaterializationError(f"{name} did not expand to 16 rows")
    return rows


def _validate_alignment_permutation(
    value: tuple[int, ...],
    identity: tuple[int, ...],
    name: str,
) -> None:
    if value != identity:
        if len(value) != len(identity) or sorted(value) != list(identity):
            raise MaterializationError(f"{name} alignment is lossy")
        raise MaterializationError(f"{name} alignment is not canonical identity")


def _alignment_rows(
    request: MaterializationRequest,
    rows: list[_PreparedRow],
) -> tuple[list[int], list[int], list[tuple[bool, ...]]]:
    if not request.invariant_pairs:
        return [], [], []
    covered: set[int] = set()
    left_rows: list[int] = []
    right_rows: list[int] = []
    step_masks: list[tuple[bool, ...]] = []
    rectangle_count = len(request.rectangles)
    for pair_index, pair in enumerate(request.invariant_pairs):
        name = f"invariant pair {pair_index}"
        if (
            not isinstance(pair, GenericInvariantPair)
            or not _plain_int(pair.left_rectangle)
            or not _plain_int(pair.right_rectangle)
            or not 0 <= pair.left_rectangle < rectangle_count
            or not 0 <= pair.right_rectangle < rectangle_count
            or pair.left_rectangle == pair.right_rectangle
        ):
            raise MaterializationError(f"{name} rectangle index differs")
        if pair.left_rectangle in covered or pair.right_rectangle in covered:
            raise MaterializationError(f"{name} reuses a semantic rectangle")
        covered.update((pair.left_rectangle, pair.right_rectangle))
        _validate_alignment_permutation(
            pair.slot_permutation, _IDENTITY_SLOTS, f"{name} slot"
        )
        _validate_alignment_permutation(
            pair.type_permutation, _IDENTITY_TYPES, f"{name} type"
        )
        _validate_alignment_permutation(
            pair.relation_permutation, _IDENTITY_RELATIONS, f"{name} relation"
        )
        _validate_alignment_permutation(
            pair.value_permutation, _IDENTITY_VALUES, f"{name} value"
        )
        for local_row in range(16):
            left_index = 16 * pair.left_rectangle + local_row
            right_index = 16 * pair.right_rectangle + local_row
            left = rows[left_index]
            right = rows[right_index]
            if (
                left.initial != right.initial
                or left.terminal != right.terminal
                or left.trace != right.trace
                or left.answer_token != right.answer_token
            ):
                raise MaterializationError(
                    f"{name} target coordinates are not invariant"
                )
            if left.source_identity == right.source_identity:
                raise MaterializationError(
                    f"{name} pairs identical candidate source bytes"
                )
            left_rows.append(left_index)
            right_rows.append(right_index)
            step_masks.append(left.trace.step_mask)
    if covered != set(range(rectangle_count)):
        raise MaterializationError(
            "invariant pairs do not cover every semantic rectangle"
        )
    return left_rows, right_rows, step_masks


def _import_torch_contracts() -> tuple[Any, ...]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise MaterializationError(
            "torch is unavailable; ETTRContinuationBatch cannot be constructed"
        ) from exc
    train_path = Path(__file__).resolve().parents[1] / "train"
    train_text = str(train_path)
    if train_text not in sys.path:
        sys.path.insert(0, train_text)
    try:
        from endogenous_typed_theory_reactor import TheoryReactorConfig
        from ettr_data_contract import ETTRCausalRectangle, ETTRContinuationBatch
        from ettr_episode import ETTREpisodeBatch, ETTREpisodeSegment
        from ettr_objectives import (
            ETTRObjectiveConfig,
            ETTRPacketTargets,
            ETTRTransactionTargets,
            ETTRVariantAlignment,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise MaterializationError("ETTR receiving contract is unavailable") from exc
    return (
        torch,
        TheoryReactorConfig,
        ETTRCausalRectangle,
        ETTRContinuationBatch,
        ETTREpisodeBatch,
        ETTREpisodeSegment,
        ETTRObjectiveConfig,
        ETTRPacketTargets,
        ETTRTransactionTargets,
        ETTRVariantAlignment,
    )


def _packet_targets(torch: Any, packet_rows: Sequence[_PacketImage], cls: Any) -> Any:
    batch = len(packet_rows)
    relations = torch.zeros(
        batch,
        NUM_RELATIONS,
        NUM_SLOTS,
        NUM_SLOTS,
        dtype=torch.bool,
    )
    for row, packet in enumerate(packet_rows):
        for relation, source, target in packet.relations:
            relations[row, relation, source, target] = True
    return cls(
        value_code=torch.tensor(
            [packet.value_code for packet in packet_rows],
            dtype=torch.long,
        ),
        type_index=torch.tensor(
            [packet.type_index for packet in packet_rows],
            dtype=torch.long,
        ),
        relations=relations,
        active=torch.tensor(
            [packet.active for packet in packet_rows],
            dtype=torch.bool,
        ),
        root=torch.tensor(
            [packet.root for packet in packet_rows],
            dtype=torch.bool,
        ),
        committed=torch.tensor(
            [packet.committed for packet in packet_rows],
            dtype=torch.bool,
        ),
        halted=torch.tensor(
            [packet.halted for packet in packet_rows],
            dtype=torch.bool,
        ),
        slot_mask=torch.ones(batch, NUM_SLOTS, dtype=torch.bool),
        relation_mask=torch.ones(
            batch,
            NUM_RELATIONS,
            NUM_SLOTS,
            NUM_SLOTS,
            dtype=torch.bool,
        ),
    )


def materialize_ettr_il_v2(
    request: MaterializationRequest,
    tokenizer: TokenizerProtocol,
) -> Any:
    """Construct and validate one CPU ``ETTRContinuationBatch``.

    The function is intentionally side-effect free apart from importing the
    receiving classes.  It reads no files and cannot start training.
    """

    if (
        not isinstance(request, MaterializationRequest)
        or _SHA256.fullmatch(request.manifest_sha256) is None
        or _SHA256.fullmatch(request.dataset_sha256) is None
        or not _plain_int(request.vocab_size)
        or request.vocab_size <= 1
        or not request.rectangles
        or not isinstance(request.require_query_checkerboard, bool)
        or not isinstance(tokenizer, TokenizerProtocol)
    ):
        raise MaterializationError("materialization request differs")
    rectangle_ids = [
        rectangle.semantic_rectangle_id for rectangle in request.rectangles
    ]
    if len(rectangle_ids) != len(set(rectangle_ids)):
        raise MaterializationError("semantic rectangle IDs are not unique")

    rows: list[_PreparedRow] = []
    for index, rectangle in enumerate(request.rectangles):
        rows.extend(
            _prepare_rectangle(
                rectangle,
                index,
                tokenizer,
                request.vocab_size,
                require_query_checkerboard=request.require_query_checkerboard,
            )
        )
    if len({row.episode_id for row in rows}) != len(rows):
        raise MaterializationError("episode IDs are not unique")
    left_rows, right_rows, alignment_steps = _alignment_rows(request, rows)

    (
        torch,
        TheoryReactorConfig,
        ETTRCausalRectangle,
        ETTRContinuationBatch,
        ETTREpisodeBatch,
        ETTREpisodeSegment,
        ETTRObjectiveConfig,
        ETTRPacketTargets,
        ETTRTransactionTargets,
        ETTRVariantAlignment,
    ) = _import_torch_contracts()

    def segment(name: str) -> Any:
        values = [getattr(row, name) for row in rows]
        return ETTREpisodeSegment.from_tokens(
            torch.tensor([value.tokens for value in values], dtype=torch.long),
            attention_mask=torch.tensor(
                [value.mask for value in values],
                dtype=torch.bool,
            ),
        )

    episodes = ETTREpisodeBatch(
        episode_ids=tuple(row.episode_id for row in rows),
        reset_mask=torch.ones(len(rows), dtype=torch.bool),
        query_read_index=torch.tensor(
            [row.query_read_index for row in rows],
            dtype=torch.long,
        ),
        world=segment("world"),
        command=segment("command"),
        query=segment("query"),
    )
    packet_targets = _packet_targets(
        torch,
        [row.initial for row in rows],
        ETTRPacketTargets,
    )
    terminal_packet_targets = _packet_targets(
        torch,
        [row.terminal for row in rows],
        ETTRPacketTargets,
    )
    transaction_targets = ETTRTransactionTargets(
        **{
            name: torch.tensor(
                [getattr(row.trace, name) for row in rows],
                dtype=(
                    torch.bool
                    if name in {"committed", "halted", "step_mask"}
                    else torch.long
                ),
            )
            for name in (
                "opcode",
                "source",
                "target",
                "relation",
                "type_index",
                "value_code",
                "committed",
                "halted",
                "step_mask",
            )
        }
    )
    causal_rows = []
    for rectangle_index in range(len(request.rectangles)):
        for query_index in range(2):
            for paraphrase_index in range(2):
                base = (
                    16 * rectangle_index
                    + 8 * query_index
                    + 4 * paraphrase_index
                )
                causal_rows.append(
                    [[base, base + 1], [base + 2, base + 3]]
                )
    causal_rectangles = ETTRCausalRectangle(
        rows=torch.tensor(causal_rows, dtype=torch.long)
    )
    equivariance = None
    if left_rows:
        pairs = len(left_rows)
        equivariance = ETTRVariantAlignment(
            left_index=torch.tensor(left_rows, dtype=torch.long),
            right_index=torch.tensor(right_rows, dtype=torch.long),
            slot_permutation=torch.arange(NUM_SLOTS)[None, :].expand(
                pairs, -1
            ).clone(),
            type_permutation=torch.arange(NUM_TYPES)[None, :].expand(
                pairs, -1
            ).clone(),
            relation_permutation=torch.arange(NUM_RELATIONS)[None, :].expand(
                pairs, -1
            ).clone(),
            value_permutation=torch.arange(NUM_VALUE_CODES)[None, :].expand(
                pairs, -1
            ).clone(),
            slot_mask=torch.ones(pairs, NUM_SLOTS, dtype=torch.bool),
            relation_mask=torch.ones(
                pairs,
                NUM_RELATIONS,
                NUM_SLOTS,
                NUM_SLOTS,
                dtype=torch.bool,
            ),
            step_mask=torch.tensor(alignment_steps, dtype=torch.bool),
        )
    initial_status = torch.zeros(len(rows), dtype=torch.bool)
    batch = ETTRContinuationBatch(
        manifest_sha256=request.manifest_sha256,
        dataset_sha256=request.dataset_sha256,
        episodes=episodes,
        packet_targets=packet_targets,
        terminal_packet_targets=terminal_packet_targets,
        causal_rectangles=causal_rectangles,
        transaction_targets=transaction_targets,
        initial_committed=initial_status.clone(),
        initial_halted=initial_status.clone(),
        equivariance=equivariance,
    )
    reactor_config = TheoryReactorConfig()
    objective_config = ETTRObjectiveConfig(
        vocab_size=request.vocab_size,
        require_equivariance_pairs=bool(left_rows),
    )
    batch.validate(reactor_config, objective_config)
    if any(
        tensor.device.type != "cpu"
        for tensor in (
            batch.episodes.world.tokens,
            batch.episodes.command.tokens,
            batch.episodes.query.tokens,
            batch.packet_targets.active,
            batch.terminal_packet_targets.active,
            batch.transaction_targets.opcode,
            batch.causal_rectangles.rows,
        )
    ):
        raise MaterializationError("materialized tensors left the CPU")
    return batch


__all__ = [
    "COMMAND_WIDTH",
    "Disposition",
    "GenericCell",
    "GenericCommand",
    "GenericCorner",
    "GenericEdge",
    "GenericInvariantPair",
    "GenericMutation",
    "GenericOperationTrace",
    "GenericPacket",
    "GenericQuery",
    "GenericSemanticRectangle",
    "GenericWorld",
    "MAX_EDGES",
    "MAX_STEPS",
    "MaterializationError",
    "MaterializationRequest",
    "NUM_RELATIONS",
    "NUM_SLOTS",
    "NUM_TYPES",
    "NUM_VALUE_CODES",
    "Opcode",
    "PROTOCOL",
    "QUERY_WIDTH",
    "TokenizerProtocol",
    "ValueKind",
    "ValueRef",
    "WORLD_WIDTH",
    "materialize_ettr_il_v2",
]
