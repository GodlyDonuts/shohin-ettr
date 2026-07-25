"""Raw-byte neural compiler for the source-deleted multi-family board.

Deterministic preprocessing recognizes only record boundaries, an opaque key
codec, and exact key equality. It does not parse renderer grammar or assign
semantic roles. A shared neural encoder predicts source/action/target roles
for every source record and start/action roles for a late query. Hard
constrained projections then seal one anonymous transition machine and execute
the late action word after source deletion.

The exact board parser is intentionally not imported by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
import re
from typing import Sequence

import torch
import torch.nn as nn


BYTE_PAD_ID = 256
KEY_MASK_ID = 257
BYTE_VOCAB_SIZE = 258
MAX_RECORDS = 48
MAX_RECORD_UNITS = 192
MAX_SOURCE_KEYS = 19
SOURCE_OCCURRENCES_PER_RECORD = 3
MAX_QUERY_OCCURRENCES = 9
MAX_QUERY_UNITS = 256
SOURCE_ROLES = 3
ROLE_SOURCE = 0
ROLE_ACTION = 1
ROLE_TARGET = 2
QUERY_ROLES = 2
QUERY_START = 0
QUERY_ACTION = 1
PROTECTED_SHOHIN_PARAMETERS = 125_081_664
GLOBAL_PARAMETER_LIMIT = 200_000_000
_OPAQUE_KEY = re.compile(rb"(?<![A-Za-z0-9])h[0-9a-f]{20}(?![A-Za-z0-9])")
_SOURCE_ROLE_PERMUTATIONS = tuple(itertools.permutations(range(SOURCE_ROLES)))


class MultiFamilyCompilerError(ValueError):
    """Raised when candidate preprocessing or sealing violates the contract."""


@dataclass(frozen=True, slots=True)
class ScannedRecord:
    payload: bytes
    units: tuple[int, ...]
    unit_byte_bounds: tuple[tuple[int, int], ...]
    occurrence_positions: tuple[int, ...]
    occurrence_keys: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if (
            not self.units
            or len(self.units) > MAX_RECORD_UNITS
            or len(self.unit_byte_bounds) != len(self.units)
            or len(self.occurrence_positions) != SOURCE_OCCURRENCES_PER_RECORD
            or len(self.occurrence_keys) != SOURCE_OCCURRENCES_PER_RECORD
        ):
            raise MultiFamilyCompilerError("source record geometry differs")
        if any(
            not 0 <= start < end <= len(self.payload)
            for start, end in self.unit_byte_bounds
        ):
            raise MultiFamilyCompilerError("source unit byte bounds differ")
        if any(
            not 0 <= position < len(self.units)
            or self.units[position] != KEY_MASK_ID
            for position in self.occurrence_positions
        ):
            raise MultiFamilyCompilerError("source occurrence position differs")


@dataclass(frozen=True, slots=True)
class ScannedSource:
    payload_sha256: str
    records: tuple[ScannedRecord, ...]
    unique_keys: tuple[bytes, ...]
    occurrence_to_unique: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            not self.records
            or len(self.records) > MAX_RECORDS
            or not self.unique_keys
            or len(self.unique_keys) > MAX_SOURCE_KEYS
            or len(set(self.unique_keys)) != len(self.unique_keys)
            or len(self.occurrence_to_unique) != len(self.records)
        ):
            raise MultiFamilyCompilerError("scanned source geometry differs")
        for row in self.occurrence_to_unique:
            if (
                len(row) != SOURCE_OCCURRENCES_PER_RECORD
                or any(index not in range(len(self.unique_keys)) for index in row)
            ):
                raise MultiFamilyCompilerError(
                    "source equality partition differs"
                )


@dataclass(frozen=True, slots=True)
class ScannedQuery:
    payload_sha256: str
    payload: bytes
    units: tuple[int, ...]
    unit_byte_bounds: tuple[tuple[int, int], ...]
    occurrence_positions: tuple[int, ...]
    occurrence_keys: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if (
            not self.units
            or len(self.units) > MAX_QUERY_UNITS
            or len(self.unit_byte_bounds) != len(self.units)
            or not 2 <= len(self.occurrence_positions) <= MAX_QUERY_OCCURRENCES
            or len(self.occurrence_positions) != len(self.occurrence_keys)
        ):
            raise MultiFamilyCompilerError("scanned query geometry differs")
        if any(
            not 0 <= start < end <= len(self.payload)
            for start, end in self.unit_byte_bounds
        ):
            raise MultiFamilyCompilerError("query unit byte bounds differ")
        if any(
            not 0 <= position < len(self.units)
            or self.units[position] != KEY_MASK_ID
            for position in self.occurrence_positions
        ):
            raise MultiFamilyCompilerError("query occurrence position differs")


@dataclass(frozen=True, slots=True)
class SourceTensorBatch:
    unit_ids: torch.Tensor
    unit_valid: torch.Tensor
    record_valid: torch.Tensor
    occurrence_positions: torch.Tensor
    occurrence_to_unique: torch.Tensor
    unique_key_valid: torch.Tensor
    unique_keys: tuple[tuple[bytes, ...], ...]
    source_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.unit_ids.ndim != 3 or self.unit_ids.dtype != torch.long:
            raise MultiFamilyCompilerError("source unit tensor differs")
        batch = self.unit_ids.shape[0]
        expected = {
            "unit_valid": (self.unit_ids.shape, torch.bool),
            "record_valid": ((batch, MAX_RECORDS), torch.bool),
            "occurrence_positions": (
                (batch, MAX_RECORDS, SOURCE_OCCURRENCES_PER_RECORD),
                torch.long,
            ),
            "occurrence_to_unique": (
                (batch, MAX_RECORDS, SOURCE_OCCURRENCES_PER_RECORD),
                torch.long,
            ),
            "unique_key_valid": ((batch, MAX_SOURCE_KEYS), torch.bool),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype:
                raise MultiFamilyCompilerError(f"{name} tensor differs")
        if (
            len(self.unique_keys) != batch
            or len(self.source_sha256) != batch
        ):
            raise MultiFamilyCompilerError("source metadata batch differs")


@dataclass(frozen=True, slots=True)
class QueryTensorBatch:
    unit_ids: torch.Tensor
    unit_valid: torch.Tensor
    occurrence_positions: torch.Tensor
    occurrence_valid: torch.Tensor
    occurrence_keys: tuple[tuple[bytes, ...], ...]
    query_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.unit_ids.ndim != 2 or self.unit_ids.dtype != torch.long:
            raise MultiFamilyCompilerError("query unit tensor differs")
        batch = self.unit_ids.shape[0]
        expected = {
            "unit_valid": (self.unit_ids.shape, torch.bool),
            "occurrence_positions": (
                (batch, MAX_QUERY_OCCURRENCES),
                torch.long,
            ),
            "occurrence_valid": (
                (batch, MAX_QUERY_OCCURRENCES),
                torch.bool,
            ),
        }
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype:
                raise MultiFamilyCompilerError(f"{name} tensor differs")
        if (
            len(self.occurrence_keys) != batch
            or len(self.query_sha256) != batch
        ):
            raise MultiFamilyCompilerError("query metadata batch differs")


@dataclass(frozen=True, slots=True)
class CompilerOutput:
    source_role_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class QueryOutput:
    query_role_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class SealedAnonymousMachine:
    """The only source-derived object retained after compilation."""

    state_keys: tuple[bytes, ...]
    action_keys: tuple[bytes, ...]
    transition: tuple[tuple[int, ...], ...]
    compiler_state_sha256: str

    def __post_init__(self) -> None:
        cardinality = len(self.state_keys)
        if (
            cardinality not in {8, 16}
            or len(set(self.state_keys)) != cardinality
            or len(self.action_keys) != 3
            or len(set(self.action_keys)) != 3
            or set(self.state_keys) & set(self.action_keys)
            or len(self.transition) != 3
        ):
            raise MultiFamilyCompilerError("sealed anonymous geometry differs")
        expected = set(range(cardinality))
        if any(len(row) != cardinality or set(row) != expected for row in self.transition):
            raise MultiFamilyCompilerError("sealed transition is not a permutation")
        if (
            len(self.compiler_state_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.compiler_state_sha256
            )
        ):
            raise MultiFamilyCompilerError("compiler state digest differs")

    @property
    def packet_sha256(self) -> str:
        digest = sha256(b"MULTIFAMILY-SEALED-MACHINE-V1\0")
        for key in (*self.state_keys, *self.action_keys):
            digest.update(len(key).to_bytes(2, "big"))
            digest.update(key)
        for row in self.transition:
            digest.update(bytes(row))
        digest.update(bytes.fromhex(self.compiler_state_sha256))
        return digest.hexdigest()

    def deployed_wire(self) -> bytes:
        payload = {
            "action_keys": [key.hex() for key in self.action_keys],
            "compiler_state_sha256": self.compiler_state_sha256,
            "schema": "MULTIFAMILY-SEALED-MACHINE-V1",
            "state_keys": [key.hex() for key in self.state_keys],
            "transition": [list(row) for row in self.transition],
        }
        return (
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")

    @classmethod
    def from_deployed_wire(cls, wire: bytes) -> SealedAnonymousMachine:
        try:
            payload = json.loads(wire)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MultiFamilyCompilerError("sealed wire is not JSON") from exc
        canonical = (
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        if canonical != wire or payload.get("schema") != "MULTIFAMILY-SEALED-MACHINE-V1":
            raise MultiFamilyCompilerError("sealed wire is not canonical")
        try:
            return cls(
                state_keys=tuple(
                    bytes.fromhex(value) for value in payload["state_keys"]
                ),
                action_keys=tuple(
                    bytes.fromhex(value) for value in payload["action_keys"]
                ),
                transition=tuple(
                    tuple(int(value) for value in row)
                    for row in payload["transition"]
                ),
                compiler_state_sha256=payload["compiler_state_sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MultiFamilyCompilerError("sealed wire fields differ") from exc


@dataclass(frozen=True, slots=True)
class ParameterReceipt:
    protected_shohin: int
    learned_compiler: int
    complete_system: int
    global_limit: int
    headroom: int

    def __post_init__(self) -> None:
        if (
            self.complete_system != self.protected_shohin + self.learned_compiler
            or self.headroom != self.global_limit - self.complete_system
            or self.headroom <= 0
        ):
            raise MultiFamilyCompilerError("parameter receipt arithmetic differs")


def _masked_units(
    payload: bytes,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    tuple[int, ...],
    tuple[bytes, ...],
]:
    units: list[int] = []
    bounds: list[tuple[int, int]] = []
    positions: list[int] = []
    keys: list[bytes] = []
    cursor = 0
    for match in _OPAQUE_KEY.finditer(payload):
        for index in range(cursor, match.start()):
            units.append(payload[index])
            bounds.append((index, index + 1))
        positions.append(len(units))
        units.append(KEY_MASK_ID)
        bounds.append(match.span())
        keys.append(match.group())
        cursor = match.end()
    for index in range(cursor, len(payload)):
        units.append(payload[index])
        bounds.append((index, index + 1))
    if not keys:
        raise MultiFamilyCompilerError("payload contains no opaque keys")
    return tuple(units), tuple(bounds), tuple(positions), tuple(keys)


def scan_source(payload: bytes) -> ScannedSource:
    if not isinstance(payload, bytes) or not payload:
        raise MultiFamilyCompilerError("source payload differs")
    raw_records = payload.splitlines()
    if not raw_records or len(raw_records) > MAX_RECORDS:
        raise MultiFamilyCompilerError("source record count differs")
    records = tuple(
        ScannedRecord(record, *_masked_units(record))
        for record in raw_records
    )
    unique_keys: list[bytes] = []
    unique_index: dict[bytes, int] = {}
    occurrence_to_unique: list[tuple[int, ...]] = []
    for record in records:
        indices: list[int] = []
        for key in record.occurrence_keys:
            if key not in unique_index:
                unique_index[key] = len(unique_keys)
                unique_keys.append(key)
            indices.append(unique_index[key])
        occurrence_to_unique.append(tuple(indices))
    return ScannedSource(
        payload_sha256=sha256(payload).hexdigest(),
        records=records,
        unique_keys=tuple(unique_keys),
        occurrence_to_unique=tuple(occurrence_to_unique),
    )


def scan_query(payload: bytes) -> ScannedQuery:
    if not isinstance(payload, bytes) or not payload or b"\n" in payload:
        raise MultiFamilyCompilerError("query payload differs")
    units, bounds, positions, keys = _masked_units(payload)
    return ScannedQuery(
        payload_sha256=sha256(payload).hexdigest(),
        payload=payload,
        units=units,
        unit_byte_bounds=bounds,
        occurrence_positions=positions,
        occurrence_keys=keys,
    )


def collate_sources(
    sources: Sequence[ScannedSource],
    *,
    device: torch.device | str = "cpu",
) -> SourceTensorBatch:
    if not sources:
        raise MultiFamilyCompilerError("source batch is empty")
    max_units = max(len(record.units) for source in sources for record in source.records)
    batch = len(sources)
    unit_ids = torch.full(
        (batch, MAX_RECORDS, max_units),
        BYTE_PAD_ID,
        dtype=torch.long,
        device=device,
    )
    unit_valid = torch.zeros_like(unit_ids, dtype=torch.bool)
    record_valid = torch.zeros((batch, MAX_RECORDS), dtype=torch.bool, device=device)
    occurrence_positions = torch.zeros(
        (batch, MAX_RECORDS, SOURCE_OCCURRENCES_PER_RECORD),
        dtype=torch.long,
        device=device,
    )
    occurrence_to_unique = torch.zeros_like(occurrence_positions)
    unique_key_valid = torch.zeros(
        (batch, MAX_SOURCE_KEYS),
        dtype=torch.bool,
        device=device,
    )
    for row, source in enumerate(sources):
        unique_key_valid[row, : len(source.unique_keys)] = True
        for record_index, (record, equality) in enumerate(
            zip(source.records, source.occurrence_to_unique, strict=True)
        ):
            values = torch.tensor(record.units, dtype=torch.long, device=device)
            unit_ids[row, record_index, : len(record.units)] = values
            unit_valid[row, record_index, : len(record.units)] = True
            record_valid[row, record_index] = True
            occurrence_positions[row, record_index] = torch.tensor(
                record.occurrence_positions,
                dtype=torch.long,
                device=device,
            )
            occurrence_to_unique[row, record_index] = torch.tensor(
                equality,
                dtype=torch.long,
                device=device,
            )
    return SourceTensorBatch(
        unit_ids=unit_ids,
        unit_valid=unit_valid,
        record_valid=record_valid,
        occurrence_positions=occurrence_positions,
        occurrence_to_unique=occurrence_to_unique,
        unique_key_valid=unique_key_valid,
        unique_keys=tuple(source.unique_keys for source in sources),
        source_sha256=tuple(source.payload_sha256 for source in sources),
    )


def collate_queries(
    queries: Sequence[ScannedQuery],
    *,
    device: torch.device | str = "cpu",
) -> QueryTensorBatch:
    if not queries:
        raise MultiFamilyCompilerError("query batch is empty")
    max_units = max(len(query.units) for query in queries)
    batch = len(queries)
    unit_ids = torch.full(
        (batch, max_units),
        BYTE_PAD_ID,
        dtype=torch.long,
        device=device,
    )
    unit_valid = torch.zeros_like(unit_ids, dtype=torch.bool)
    occurrence_positions = torch.zeros(
        (batch, MAX_QUERY_OCCURRENCES),
        dtype=torch.long,
        device=device,
    )
    occurrence_valid = torch.zeros(
        (batch, MAX_QUERY_OCCURRENCES),
        dtype=torch.bool,
        device=device,
    )
    for row, query in enumerate(queries):
        values = torch.tensor(query.units, dtype=torch.long, device=device)
        unit_ids[row, : len(query.units)] = values
        unit_valid[row, : len(query.units)] = True
        count = len(query.occurrence_positions)
        occurrence_positions[row, :count] = torch.tensor(
            query.occurrence_positions,
            dtype=torch.long,
            device=device,
        )
        occurrence_valid[row, :count] = True
    return QueryTensorBatch(
        unit_ids=unit_ids,
        unit_valid=unit_valid,
        occurrence_positions=occurrence_positions,
        occurrence_valid=occurrence_valid,
        occurrence_keys=tuple(query.occurrence_keys for query in queries),
        query_sha256=tuple(query.payload_sha256 for query in queries),
    )


def project_byte_features_to_units(
    *,
    unit_byte_bounds: Sequence[tuple[int, int]],
    byte_features: torch.Tensor,
) -> torch.Tensor:
    """Average byte-aligned frozen features into role-neutral masked units."""

    if (
        byte_features.ndim != 2
        or not byte_features.is_floating_point()
        or not bool(torch.isfinite(byte_features).all())
    ):
        raise MultiFamilyCompilerError("frozen byte features differ")
    projected: list[torch.Tensor] = []
    for start, end in unit_byte_bounds:
        if not 0 <= start < end <= byte_features.shape[0]:
            raise MultiFamilyCompilerError(
                "unit projection leaves frozen byte features"
            )
        projected.append(byte_features[start:end].mean(0))
    if not projected:
        raise MultiFamilyCompilerError("unit projection is empty")
    return torch.stack(projected)


class SharedRawMachineCompiler(nn.Module):
    """One renderer/family-shared byte encoder and semantic role compiler."""

    def __init__(
        self,
        *,
        width: int = 192,
        layers: int = 2,
        external_width: int = 0,
    ) -> None:
        super().__init__()
        if width < 32 or width % 2 or layers < 1 or external_width < 0:
            raise MultiFamilyCompilerError("compiler geometry differs")
        self.width = int(width)
        self.external_width = int(external_width)
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, width)
        self.position = nn.Embedding(MAX_QUERY_UNITS, width)
        self.external_projection = (
            nn.Sequential(
                nn.LayerNorm(external_width),
                nn.Linear(external_width, width),
                nn.GELU(),
                nn.Linear(width, width, bias=False),
            )
            if external_width
            else None
        )
        self.encoder = nn.GRU(
            input_size=width,
            hidden_size=width // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
        )
        self.source_role_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, SOURCE_ROLES),
        )
        self.query_role_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, QUERY_ROLES),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_receipt(self) -> ParameterReceipt:
        learned = self.parameter_count()
        return ParameterReceipt(
            protected_shohin=PROTECTED_SHOHIN_PARAMETERS,
            learned_compiler=learned,
            complete_system=PROTECTED_SHOHIN_PARAMETERS + learned,
            global_limit=GLOBAL_PARAMETER_LIMIT,
            headroom=GLOBAL_PARAMETER_LIMIT - PROTECTED_SHOHIN_PARAMETERS - learned,
        )

    def _encode(
        self,
        unit_ids: torch.Tensor,
        unit_valid: torch.Tensor,
    ) -> torch.Tensor:
        if (
            unit_ids.ndim != 2
            or unit_ids.shape != unit_valid.shape
            or unit_ids.dtype != torch.long
            or unit_valid.dtype != torch.bool
            or unit_ids.shape[1] > MAX_QUERY_UNITS
        ):
            raise MultiFamilyCompilerError("encoder input differs")
        positions = torch.arange(unit_ids.shape[1], device=unit_ids.device)
        hidden = self.embedding(unit_ids) + self.position(positions)[None]
        hidden, _ = self.encoder(hidden)
        return hidden * unit_valid[..., None]

    @staticmethod
    def _gather_occurrences(
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return hidden.gather(
            1,
            positions[..., None].expand(-1, -1, hidden.shape[-1]),
        )

    def compile_source(
        self,
        batch: SourceTensorBatch,
        *,
        external_unit_features: torch.Tensor | None = None,
    ) -> CompilerOutput:
        rows, records, units = batch.unit_ids.shape
        if (
            external_unit_features is not None
            and external_unit_features.shape
            != (*batch.unit_ids.shape, self.external_width)
        ):
            raise MultiFamilyCompilerError(
                "source frozen unit feature geometry differs"
            )
        hidden = self._encode(
            batch.unit_ids.reshape(rows * records, units),
            batch.unit_valid.reshape(rows * records, units),
        )
        positions = batch.occurrence_positions.reshape(
            rows * records,
            SOURCE_OCCURRENCES_PER_RECORD,
        )
        occurrences = self._gather_occurrences(hidden, positions)
        if self.external_projection is None:
            if external_unit_features is not None:
                raise MultiFamilyCompilerError(
                    "standalone compiler received frozen features"
                )
        else:
            if (
                external_unit_features is None
                or not external_unit_features.is_floating_point()
                or external_unit_features.device != batch.unit_ids.device
                or not bool(torch.isfinite(external_unit_features).all())
            ):
                raise MultiFamilyCompilerError(
                    "connected source features differ"
                )
            flat_external = external_unit_features.reshape(
                rows * records,
                units,
                self.external_width,
            )
            occurrence_external = self._gather_occurrences(
                flat_external,
                positions,
            )
            occurrences = occurrences + self.external_projection(
                occurrence_external
            )
        logits = self.source_role_head(occurrences).reshape(
            rows,
            records,
            SOURCE_OCCURRENCES_PER_RECORD,
            SOURCE_ROLES,
        )
        return CompilerOutput(source_role_logits=logits)

    def parse_query(
        self,
        batch: QueryTensorBatch,
        *,
        external_unit_features: torch.Tensor | None = None,
    ) -> QueryOutput:
        if (
            external_unit_features is not None
            and external_unit_features.shape
            != (*batch.unit_ids.shape, self.external_width)
        ):
            raise MultiFamilyCompilerError(
                "query frozen unit feature geometry differs"
            )
        hidden = self._encode(
            batch.unit_ids,
            batch.unit_valid,
        )
        occurrences = self._gather_occurrences(
            hidden,
            batch.occurrence_positions,
        )
        if self.external_projection is None:
            if external_unit_features is not None:
                raise MultiFamilyCompilerError(
                    "standalone query parser received frozen features"
                )
        else:
            if (
                external_unit_features is None
                or not external_unit_features.is_floating_point()
                or external_unit_features.device != batch.unit_ids.device
                or not bool(torch.isfinite(external_unit_features).all())
            ):
                raise MultiFamilyCompilerError(
                    "connected query features differ"
                )
            occurrence_external = self._gather_occurrences(
                external_unit_features,
                batch.occurrence_positions,
            )
            occurrences = occurrences + self.external_projection(
                occurrence_external
            )
        return QueryOutput(query_role_logits=self.query_role_head(occurrences))


def _best_source_role_permutation(logits: torch.Tensor) -> tuple[int, int, int]:
    if logits.shape != (SOURCE_OCCURRENCES_PER_RECORD, SOURCE_ROLES):
        raise MultiFamilyCompilerError("record role logits differ")
    scores = [
        sum(float(logits[occurrence, role]) for occurrence, role in enumerate(permutation))
        for permutation in _SOURCE_ROLE_PERMUTATIONS
    ]
    return _SOURCE_ROLE_PERMUTATIONS[max(range(len(scores)), key=scores.__getitem__)]


def seal_machine(
    batch: SourceTensorBatch,
    output: CompilerOutput,
    *,
    row: int,
    binding_shuffle: bool = False,
) -> SealedAnonymousMachine:
    if (
        not 0 <= row < batch.unit_ids.shape[0]
        or output.source_role_logits.shape
        != (
            batch.unit_ids.shape[0],
            MAX_RECORDS,
            SOURCE_OCCURRENCES_PER_RECORD,
            SOURCE_ROLES,
        )
    ):
        raise MultiFamilyCompilerError("compiler output geometry differs")
    record_count = int(batch.record_valid[row].sum())
    role_records: list[tuple[int, int, int]] = []
    action_unique: set[int] = set()
    state_unique: set[int] = set()
    for record in range(record_count):
        assignment = _best_source_role_permutation(
            output.source_role_logits[row, record]
        )
        occurrence_for_role = {
            role: occurrence for occurrence, role in enumerate(assignment)
        }
        equality = batch.occurrence_to_unique[row, record]
        source = int(equality[occurrence_for_role[ROLE_SOURCE]])
        action = int(equality[occurrence_for_role[ROLE_ACTION]])
        target = int(equality[occurrence_for_role[ROLE_TARGET]])
        role_records.append((source, action, target))
        action_unique.add(action)
        state_unique.update((source, target))
    if len(action_unique) != 3 or len(state_unique) not in {8, 16}:
        raise MultiFamilyCompilerError("predicted key partition differs")
    if action_unique & state_unique:
        raise MultiFamilyCompilerError("predicted action/state keys overlap")
    state_slots = tuple(sorted(state_unique))
    action_slots = tuple(sorted(action_unique))
    state_index = {unique: index for index, unique in enumerate(state_slots)}
    action_index = {unique: index for index, unique in enumerate(action_slots)}
    transitions = [[-1 for _ in state_slots] for _ in action_slots]
    for source, action, target in role_records:
        location = (action_index[action], state_index[source])
        target_index = state_index[target]
        previous = transitions[location[0]][location[1]]
        if previous not in {-1, target_index}:
            raise MultiFamilyCompilerError("predicted transition conflict")
        transitions[location[0]][location[1]] = target_index
    if any(target < 0 for transition in transitions for target in transition):
        raise MultiFamilyCompilerError("predicted transition table is incomplete")
    if binding_shuffle:
        transitions = transitions[1:] + transitions[:1]
    keys = batch.unique_keys[row]
    digest = sha256(
        output.source_role_logits[row, :record_count]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .tobytes()
    ).hexdigest()
    return SealedAnonymousMachine(
        state_keys=tuple(keys[index] for index in state_slots),
        action_keys=tuple(keys[index] for index in action_slots),
        transition=tuple(tuple(transition) for transition in transitions),
        compiler_state_sha256=digest,
    )


def execute_query(
    machine: SealedAnonymousMachine,
    batch: QueryTensorBatch,
    output: QueryOutput,
    *,
    row: int,
) -> bytes:
    if (
        not 0 <= row < batch.unit_ids.shape[0]
        or output.query_role_logits.shape
        != (
            batch.unit_ids.shape[0],
            MAX_QUERY_OCCURRENCES,
            QUERY_ROLES,
        )
    ):
        raise MultiFamilyCompilerError("query output geometry differs")
    count = int(batch.occurrence_valid[row].sum())
    logits = output.query_role_logits[row, :count]
    start_scores = logits[:, QUERY_START] - logits[:, QUERY_ACTION]
    start_occurrence = int(start_scores.argmax())
    keys = batch.occurrence_keys[row]
    start_key = keys[start_occurrence]
    try:
        state = machine.state_keys.index(start_key)
        actions = tuple(
            machine.action_keys.index(key)
            for occurrence, key in enumerate(keys)
            if occurrence != start_occurrence
        )
    except ValueError as exc:
        raise MultiFamilyCompilerError(
            "predicted query roles reference the wrong key class"
        ) from exc
    if not actions:
        raise MultiFamilyCompilerError("predicted late action word is empty")
    for action in actions:
        state = machine.transition[action][state]
    return machine.state_keys[state]


__all__ = [
    "CompilerOutput",
    "GLOBAL_PARAMETER_LIMIT",
    "KEY_MASK_ID",
    "MAX_QUERY_OCCURRENCES",
    "MAX_RECORDS",
    "MultiFamilyCompilerError",
    "ParameterReceipt",
    "PROTECTED_SHOHIN_PARAMETERS",
    "QUERY_ACTION",
    "QUERY_START",
    "QueryOutput",
    "QueryTensorBatch",
    "ROLE_ACTION",
    "ROLE_SOURCE",
    "ROLE_TARGET",
    "ScannedQuery",
    "ScannedSource",
    "SealedAnonymousMachine",
    "SharedRawMachineCompiler",
    "SourceTensorBatch",
    "collate_queries",
    "collate_sources",
    "execute_query",
    "project_byte_features_to_units",
    "scan_query",
    "scan_source",
    "seal_machine",
]
