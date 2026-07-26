"""Learned sparse latent-law compiler without a family-specific solver.

Deterministic preprocessing recognizes the public domain header, decimal
state literals, opaque action equality, and record/query boundaries. It does
not parse renderer direction or import the exact sparse-law compiler. A
shared byte encoder predicts record direction. Query-state representations
then attend over the oriented demonstrations for one opaque action and emit
the complete transition permutation. The late query executes only from the
sealed predicted packet.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Sequence

import torch
import torch.nn as nn


BYTE_PAD_ID = 256
ACTION_MASK_ID = 257
NUMBER_MASK_ID = 258
BYTE_VOCAB_SIZE = 259
MAX_CARDINALITY = 16
MAX_ACTIONS = 4
MAX_RECORDS = 20
MAX_RECORD_UNITS = 128
PROTECTED_SHOHIN_PARAMETERS = 125_081_664
GLOBAL_PARAMETER_LIMIT = 200_000_000
_ACTION = re.compile(rb"(?<![A-Za-z0-9])h[0-9a-f]{20}(?![A-Za-z0-9])")
_NUMBER = re.compile(rb"(?<![A-Za-z0-9])(?:0|[1-9][0-9]?)(?![A-Za-z0-9])")
_HEADER = re.compile(rb"domain-size=(8|16)\Z")


class SparseLawCompilerError(ValueError):
    """Raised when candidate preprocessing or execution leaves its contract."""


@dataclass(frozen=True, slots=True)
class ScannedSparseRecord:
    payload: bytes
    units: tuple[int, ...]
    action_position: int
    number_positions: tuple[int, int]
    number_values: tuple[int, int]
    action_key: bytes

    def __post_init__(self) -> None:
        if (
            not self.payload
            or not self.units
            or len(self.units) > MAX_RECORD_UNITS
            or len(self.number_positions) != 2
            or len(self.number_values) != 2
            or not 0 <= self.action_position < len(self.units)
            or self.units[self.action_position] != ACTION_MASK_ID
            or any(
                not 0 <= position < len(self.units)
                or self.units[position] != NUMBER_MASK_ID
                for position in self.number_positions
            )
            or not _ACTION.fullmatch(self.action_key)
        ):
            raise SparseLawCompilerError("scanned sparse record differs")


@dataclass(frozen=True, slots=True)
class ScannedSparseSource:
    cardinality: int
    records: tuple[ScannedSparseRecord, ...]
    action_keys: tuple[bytes, ...]
    record_action_indices: tuple[int, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        if (
            self.cardinality not in {8, 16}
            or not 2 <= len(self.action_keys) <= MAX_ACTIONS
            or len(set(self.action_keys)) != len(self.action_keys)
            or not self.records
            or len(self.records) > MAX_RECORDS
            or len(self.record_action_indices) != len(self.records)
            or any(
                index not in range(len(self.action_keys))
                for index in self.record_action_indices
            )
        ):
            raise SparseLawCompilerError("scanned sparse source differs")


@dataclass(frozen=True, slots=True)
class ScannedSparseQuery:
    start: int
    action_keys: tuple[bytes, ...]
    query_sha256: str

    def __post_init__(self) -> None:
        if (
            not 0 <= self.start < MAX_CARDINALITY
            or not self.action_keys
            or len(self.action_keys) > 8
            or any(not _ACTION.fullmatch(key) for key in self.action_keys)
        ):
            raise SparseLawCompilerError("scanned sparse query differs")


@dataclass(frozen=True, slots=True)
class SparseSourceBatch:
    unit_ids: torch.Tensor
    unit_valid: torch.Tensor
    record_valid: torch.Tensor
    action_positions: torch.Tensor
    number_positions: torch.Tensor
    number_values: torch.Tensor
    record_action_indices: torch.Tensor
    action_valid: torch.Tensor
    cardinalities: torch.Tensor
    action_keys: tuple[tuple[bytes, ...], ...]
    source_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.unit_ids.ndim != 3 or self.unit_ids.dtype != torch.long:
            raise SparseLawCompilerError("sparse source unit tensor differs")
        batch = self.unit_ids.shape[0]
        expected = {
            "unit_valid": (self.unit_ids.shape, torch.bool),
            "record_valid": ((batch, MAX_RECORDS), torch.bool),
            "action_positions": ((batch, MAX_RECORDS), torch.long),
            "number_positions": (
                (batch, MAX_RECORDS, 2),
                torch.long,
            ),
            "number_values": ((batch, MAX_RECORDS, 2), torch.long),
            "record_action_indices": (
                (batch, MAX_RECORDS),
                torch.long,
            ),
            "action_valid": ((batch, MAX_ACTIONS), torch.bool),
            "cardinalities": ((batch,), torch.long),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype:
                raise SparseLawCompilerError(f"{name} tensor differs")
        if (
            len(self.action_keys) != batch
            or len(self.source_sha256) != batch
        ):
            raise SparseLawCompilerError("sparse source metadata differs")


@dataclass(frozen=True, slots=True)
class SparseCompilerOutput:
    direction_logits: torch.Tensor
    transition_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class SparseParameterReceipt:
    protected_shohin: int
    learned_compiler: int
    complete_system: int
    global_limit: int
    headroom: int

    def __post_init__(self) -> None:
        if (
            self.complete_system
            != self.protected_shohin + self.learned_compiler
            or self.headroom
            != self.global_limit - self.complete_system
            or self.headroom <= 0
        ):
            raise SparseLawCompilerError(
                "sparse parameter receipt differs"
            )


@dataclass(frozen=True, slots=True)
class SealedLearnedSparseMachine:
    cardinality: int
    action_keys: tuple[bytes, ...]
    transition: tuple[tuple[int, ...], ...]
    compiler_state_sha256: str

    def __post_init__(self) -> None:
        expected = set(range(self.cardinality))
        if (
            self.cardinality not in {8, 16}
            or not 2 <= len(self.action_keys) <= MAX_ACTIONS
            or len(set(self.action_keys)) != len(self.action_keys)
            or len(self.transition) != len(self.action_keys)
            or any(
                len(row) != self.cardinality or set(row) != expected
                for row in self.transition
            )
            or len(self.compiler_state_sha256) != 64
        ):
            raise SparseLawCompilerError(
                "sealed learned sparse machine differs"
            )

    @property
    def packet_sha256(self) -> str:
        digest = sha256(b"LEARNED-SPARSE-LAW-MACHINE-V1\0")
        digest.update(bytes([self.cardinality]))
        for key in self.action_keys:
            digest.update(key)
        for row in self.transition:
            digest.update(bytes(row))
        digest.update(bytes.fromhex(self.compiler_state_sha256))
        return digest.hexdigest()

    def deployed_wire(self) -> bytes:
        return (
            json.dumps(
                {
                    "action_keys": [
                        key.hex() for key in self.action_keys
                    ],
                    "cardinality": self.cardinality,
                    "compiler_state_sha256": self.compiler_state_sha256,
                    "schema": "LEARNED-SPARSE-LAW-MACHINE-V1",
                    "transition": [
                        list(row) for row in self.transition
                    ],
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")

    @classmethod
    def from_deployed_wire(
        cls,
        wire: bytes,
    ) -> SealedLearnedSparseMachine:
        try:
            payload = json.loads(wire)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SparseLawCompilerError(
                "learned sparse wire is not JSON"
            ) from exc
        canonical = (
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        if (
            canonical != wire
            or payload.get("schema")
            != "LEARNED-SPARSE-LAW-MACHINE-V1"
        ):
            raise SparseLawCompilerError(
                "learned sparse wire is not canonical"
            )
        try:
            return cls(
                cardinality=int(payload["cardinality"]),
                action_keys=tuple(
                    bytes.fromhex(value)
                    for value in payload["action_keys"]
                ),
                transition=tuple(
                    tuple(int(value) for value in row)
                    for row in payload["transition"]
                ),
                compiler_state_sha256=payload[
                    "compiler_state_sha256"
                ],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SparseLawCompilerError(
                "learned sparse wire fields differ"
            ) from exc


def _mask_record(payload: bytes) -> ScannedSparseRecord:
    matches = [
        (match.start(), match.end(), ACTION_MASK_ID, match.group())
        for match in _ACTION.finditer(payload)
    ] + [
        (match.start(), match.end(), NUMBER_MASK_ID, match.group())
        for match in _NUMBER.finditer(payload)
    ]
    matches.sort()
    action_matches = [
        match for match in matches if match[2] == ACTION_MASK_ID
    ]
    number_matches = [
        match for match in matches if match[2] == NUMBER_MASK_ID
    ]
    if len(action_matches) != 1 or len(number_matches) != 2:
        raise SparseLawCompilerError(
            "record lexical occurrence geometry differs"
        )
    units: list[int] = []
    action_position = -1
    number_positions: list[int] = []
    number_values: list[int] = []
    cursor = 0
    for start, end, mask, value in matches:
        units.extend(payload[cursor:start])
        position = len(units)
        units.append(mask)
        if mask == ACTION_MASK_ID:
            action_position = position
        else:
            number_positions.append(position)
            number_values.append(int(value))
        cursor = end
    units.extend(payload[cursor:])
    return ScannedSparseRecord(
        payload=payload,
        units=tuple(units),
        action_position=action_position,
        number_positions=tuple(number_positions),
        number_values=tuple(number_values),
        action_key=action_matches[0][3],
    )


def scan_sparse_source(payload: bytes) -> ScannedSparseSource:
    if not isinstance(payload, bytes) or not payload:
        raise SparseLawCompilerError("sparse source payload differs")
    lines = payload.splitlines()
    if len(lines) < 2 or not (header := _HEADER.fullmatch(lines[0])):
        raise SparseLawCompilerError("sparse source header differs")
    cardinality = int(header.group(1))
    records = tuple(_mask_record(line) for line in lines[1:])
    if any(
        value >= cardinality
        for record in records
        for value in record.number_values
    ):
        raise SparseLawCompilerError("record state leaves public domain")
    action_keys = tuple(
        sorted({record.action_key for record in records})
    )
    if not 2 <= len(action_keys) <= MAX_ACTIONS:
        raise SparseLawCompilerError("sparse action geometry differs")
    action_index = {
        key: index for index, key in enumerate(action_keys)
    }
    return ScannedSparseSource(
        cardinality=cardinality,
        records=records,
        action_keys=action_keys,
        record_action_indices=tuple(
            action_index[record.action_key]
            for record in records
        ),
        source_sha256=sha256(payload).hexdigest(),
    )


def scan_sparse_query(payload: bytes) -> ScannedSparseQuery:
    if (
        not isinstance(payload, bytes)
        or not payload
        or b"\n" in payload
    ):
        raise SparseLawCompilerError("sparse query payload differs")
    actions = tuple(match.group() for match in _ACTION.finditer(payload))
    numbers = tuple(int(match.group()) for match in _NUMBER.finditer(payload))
    if len(numbers) != 1:
        raise SparseLawCompilerError(
            "query numeric occurrence geometry differs"
        )
    return ScannedSparseQuery(
        start=numbers[0],
        action_keys=actions,
        query_sha256=sha256(payload).hexdigest(),
    )


def collate_sparse_sources(
    sources: Sequence[ScannedSparseSource],
    *,
    device: torch.device | str = "cpu",
) -> SparseSourceBatch:
    if not sources:
        raise SparseLawCompilerError("sparse source batch is empty")
    max_units = max(
        len(record.units)
        for source in sources
        for record in source.records
    )
    batch = len(sources)
    unit_ids = torch.full(
        (batch, MAX_RECORDS, max_units),
        BYTE_PAD_ID,
        dtype=torch.long,
        device=device,
    )
    unit_valid = torch.zeros_like(unit_ids, dtype=torch.bool)
    record_valid = torch.zeros(
        (batch, MAX_RECORDS),
        dtype=torch.bool,
        device=device,
    )
    action_positions = torch.zeros(
        (batch, MAX_RECORDS),
        dtype=torch.long,
        device=device,
    )
    number_positions = torch.zeros(
        (batch, MAX_RECORDS, 2),
        dtype=torch.long,
        device=device,
    )
    number_values = torch.zeros_like(number_positions)
    record_action_indices = torch.zeros(
        (batch, MAX_RECORDS),
        dtype=torch.long,
        device=device,
    )
    action_valid = torch.zeros(
        (batch, MAX_ACTIONS),
        dtype=torch.bool,
        device=device,
    )
    cardinalities = torch.tensor(
        [source.cardinality for source in sources],
        dtype=torch.long,
        device=device,
    )
    for row, source in enumerate(sources):
        action_valid[row, : len(source.action_keys)] = True
        for record_index, (record, action_index) in enumerate(
            zip(
                source.records,
                source.record_action_indices,
                strict=True,
            )
        ):
            values = torch.tensor(
                record.units,
                dtype=torch.long,
                device=device,
            )
            unit_ids[row, record_index, : len(record.units)] = values
            unit_valid[row, record_index, : len(record.units)] = True
            record_valid[row, record_index] = True
            action_positions[row, record_index] = record.action_position
            number_positions[row, record_index] = torch.tensor(
                record.number_positions,
                dtype=torch.long,
                device=device,
            )
            number_values[row, record_index] = torch.tensor(
                record.number_values,
                dtype=torch.long,
                device=device,
            )
            record_action_indices[row, record_index] = action_index
    return SparseSourceBatch(
        unit_ids=unit_ids,
        unit_valid=unit_valid,
        record_valid=record_valid,
        action_positions=action_positions,
        number_positions=number_positions,
        number_values=number_values,
        record_action_indices=record_action_indices,
        action_valid=action_valid,
        cardinalities=cardinalities,
        action_keys=tuple(source.action_keys for source in sources),
        source_sha256=tuple(source.source_sha256 for source in sources),
    )


class SparseLatentLawCompiler(nn.Module):
    """Shared direction parser and set-attention transition completer."""

    def __init__(
        self,
        *,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if (
            width < 32
            or width % 2
            or width % heads
            or layers < 1
        ):
            raise SparseLawCompilerError(
                "sparse compiler geometry differs"
            )
        self.width = int(width)
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, width)
        self.position = nn.Embedding(MAX_RECORD_UNITS, width)
        self.record_encoder = nn.GRU(
            input_size=width,
            hidden_size=width // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
        )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.state_embedding = nn.Embedding(MAX_CARDINALITY, width)
        self.cardinality_embedding = nn.Embedding(2, width)
        self.pair_encoder = nn.Sequential(
            nn.Linear(width * 2, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            batch_first=True,
        )
        self.transition_decoder = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, MAX_CARDINALITY),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_receipt(self) -> SparseParameterReceipt:
        learned = self.parameter_count()
        return SparseParameterReceipt(
            protected_shohin=PROTECTED_SHOHIN_PARAMETERS,
            learned_compiler=learned,
            complete_system=PROTECTED_SHOHIN_PARAMETERS + learned,
            global_limit=GLOBAL_PARAMETER_LIMIT,
            headroom=(
                GLOBAL_PARAMETER_LIMIT
                - PROTECTED_SHOHIN_PARAMETERS
                - learned
            ),
        )

    @staticmethod
    def _gather(
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return hidden.gather(
            1,
            positions[..., None].expand(
                -1,
                -1,
                hidden.shape[-1],
            ),
        )

    def forward(
        self,
        batch: SparseSourceBatch,
        *,
        direction_sign: float = 1.0,
        observation_target_shift: int = 0,
        observations_zeroed: bool = False,
    ) -> SparseCompilerOutput:
        if direction_sign not in {-1.0, 1.0}:
            raise SparseLawCompilerError("direction sign differs")
        if observation_target_shift not in {0, 1}:
            raise SparseLawCompilerError(
                "observation target shift differs"
            )
        rows, records, units = batch.unit_ids.shape
        positions = torch.arange(units, device=batch.unit_ids.device)
        hidden = (
            self.embedding(
                batch.unit_ids.reshape(rows * records, units)
            )
            + self.position(positions)[None]
        )
        hidden, _ = self.record_encoder(hidden)
        number_hidden = self._gather(
            hidden,
            batch.number_positions.reshape(rows * records, 2),
        )
        direction_logits = self.direction_head(
            number_hidden.reshape(rows * records, self.width * 2)
        ).reshape(rows, records)
        direction_logits = direction_sign * direction_logits
        direction_probability = direction_logits.sigmoid()

        number_values = batch.number_values
        first = self.state_embedding(number_values[..., 0])
        second = self.state_embedding(number_values[..., 1])
        if observation_target_shift:
            shifted_first = self.state_embedding(
                (
                    number_values[..., 0]
                    + observation_target_shift
                )
                % batch.cardinalities[:, None]
            )
            shifted_second = self.state_embedding(
                (
                    number_values[..., 1]
                    + observation_target_shift
                )
                % batch.cardinalities[:, None]
            )
        else:
            shifted_first = first
            shifted_second = second
        forward_pair = self.pair_encoder(
            torch.cat((first, shifted_second), dim=-1)
        )
        reverse_pair = self.pair_encoder(
            torch.cat((second, shifted_first), dim=-1)
        )
        pair_tokens = (
            direction_probability[..., None] * forward_pair
            + (1.0 - direction_probability[..., None]) * reverse_pair
        )
        if observations_zeroed:
            pair_tokens = torch.zeros_like(pair_tokens)

        states = torch.arange(
            MAX_CARDINALITY,
            device=batch.unit_ids.device,
        )
        state_queries = self.state_embedding(states)[None].expand(
            rows,
            -1,
            -1,
        )
        cardinality_index = (batch.cardinalities == 16).long()
        state_queries = (
            state_queries
            + self.cardinality_embedding(cardinality_index)[:, None]
        )
        all_logits: list[torch.Tensor] = []
        for action in range(MAX_ACTIONS):
            action_records = (
                batch.record_valid
                & batch.record_action_indices.eq(action)
            )
            safe_action_records = action_records.clone()
            safe_action_records[
                ~safe_action_records.any(dim=1),
                0,
            ] = True
            attended, _ = self.cross_attention(
                state_queries,
                pair_tokens,
                pair_tokens,
                key_padding_mask=~safe_action_records,
                need_weights=False,
            )
            logits = self.transition_decoder(
                torch.cat((state_queries, attended), dim=-1)
            )
            all_logits.append(logits)
        transition_logits = torch.stack(all_logits, dim=1)
        target_valid = (
            states[None, None, None, :]
            < batch.cardinalities[:, None, None, None]
        )
        transition_logits = transition_logits.masked_fill(
            ~target_valid,
            -1.0e4,
        )
        return SparseCompilerOutput(
            direction_logits=direction_logits,
            transition_logits=transition_logits,
        )


def seal_sparse_machine(
    batch: SparseSourceBatch,
    output: SparseCompilerOutput,
    *,
    row: int,
) -> SealedLearnedSparseMachine:
    if (
        not 0 <= row < batch.unit_ids.shape[0]
        or output.direction_logits.shape
        != (batch.unit_ids.shape[0], MAX_RECORDS)
        or output.transition_logits.shape
        != (
            batch.unit_ids.shape[0],
            MAX_ACTIONS,
            MAX_CARDINALITY,
            MAX_CARDINALITY,
        )
    ):
        raise SparseLawCompilerError(
            "sparse compiler output geometry differs"
        )
    cardinality = int(batch.cardinalities[row])
    action_count = int(batch.action_valid[row].sum())
    transition = tuple(
        tuple(
            int(
                output.transition_logits[
                    row,
                    action,
                    source,
                    :cardinality,
                ].argmax()
            )
            for source in range(cardinality)
        )
        for action in range(action_count)
    )
    expected = set(range(cardinality))
    if any(set(action) != expected for action in transition):
        raise SparseLawCompilerError(
            "predicted sparse transition is not a permutation"
        )
    digest = sha256(
        output.transition_logits[
            row,
            :action_count,
            :cardinality,
            :cardinality,
        ]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .tobytes()
    ).hexdigest()
    return SealedLearnedSparseMachine(
        cardinality=cardinality,
        action_keys=batch.action_keys[row],
        transition=transition,
        compiler_state_sha256=digest,
    )


def execute_sparse_query(
    machine: SealedLearnedSparseMachine,
    query: ScannedSparseQuery,
) -> int:
    if not 0 <= query.start < machine.cardinality:
        raise SparseLawCompilerError("query start leaves machine")
    try:
        actions = tuple(
            machine.action_keys.index(key)
            for key in query.action_keys
        )
    except ValueError as exc:
        raise SparseLawCompilerError(
            "query action leaves machine"
        ) from exc
    state = query.start
    for action in actions:
        state = machine.transition[action][state]
    return state


__all__ = [
    "MAX_ACTIONS",
    "MAX_CARDINALITY",
    "MAX_RECORDS",
    "SealedLearnedSparseMachine",
    "SparseCompilerOutput",
    "SparseLatentLawCompiler",
    "SparseLawCompilerError",
    "SparseParameterReceipt",
    "SparseSourceBatch",
    "collate_sparse_sources",
    "execute_sparse_query",
    "scan_sparse_query",
    "scan_sparse_source",
    "seal_sparse_machine",
]
