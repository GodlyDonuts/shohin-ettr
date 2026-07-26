"""Episode-local generator closure and sparse constraint intersection.

The model learns record direction from raw bytes.  Complete opaque actions
become episode-local generators; a fixed tensor closure constructs candidate
programs from those generators.  Sparse actions are identified by intersecting
their record constraints over that temporary program space.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re

import torch

from sparse_latent_law_compiler import (
    MAX_ACTIONS,
    MAX_CARDINALITY,
    SparseCompilerOutput,
    SparseLatentLawCompiler,
    SparseLawCompilerError,
    ScannedSparseQuery,
    ScannedSparseSource,
    SparseSourceBatch,
    scan_sparse_source,
)


_HEADER_PATTERNS = (
    re.compile(rb"domain-size=(?P<n>8|16)\Z"),
    re.compile(rb"There are (?P<n>8|16) states\.\Z"),
    re.compile(rb"\(domain\|(?P<n>8|16)\)\Z"),
    re.compile(rb"states=0\.\.(?P<m>7|15)\Z"),
    re.compile(rb"maximum-state=(?P<m>7|15)\Z"),
    re.compile(
        rb"The state domain contains (?P<n>8|16) values\.\Z"
    ),
)


@dataclass(frozen=True, slots=True)
class EpisodicClosureReceipt:
    generators: int
    maximum_depth: int
    syntactic_programs: int


@dataclass(frozen=True, slots=True)
class SealedEpisodicGeneratorPacket:
    cardinality: int
    target_keys: tuple[bytes, bytes]
    transition: tuple[tuple[int, ...], tuple[int, ...]]
    compiler_state_sha256: str

    def __post_init__(self) -> None:
        expected = set(range(self.cardinality))
        if (
            self.cardinality not in {8, 16}
            or len(self.target_keys) != 2
            or len(set(self.target_keys)) != 2
            or any(len(key) != 21 or not key.startswith(b"h") for key in self.target_keys)
            or len(self.compiler_state_sha256) != 64
            or any(
                len(row) != self.cardinality or set(row) != expected
                for row in self.transition
            )
        ):
            raise SparseLawCompilerError(
                "sealed episodic packet differs"
            )

    def deployed_wire(self) -> bytes:
        return (
            json.dumps(
                {
                    "cardinality": self.cardinality,
                    "compiler_state_sha256": self.compiler_state_sha256,
                    "schema": "EPISODIC-GENERATOR-PACKET-V1",
                    "target_keys": [
                        key.hex() for key in self.target_keys
                    ],
                    "transition": self.transition,
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
    ) -> SealedEpisodicGeneratorPacket:
        try:
            value = json.loads(wire)
            if value["schema"] != "EPISODIC-GENERATOR-PACKET-V1":
                raise KeyError("schema")
            result = cls(
                cardinality=int(value["cardinality"]),
                target_keys=tuple(
                    bytes.fromhex(key)
                    for key in value["target_keys"]
                ),
                transition=tuple(
                    tuple(int(target) for target in row)
                    for row in value["transition"]
                ),
                compiler_state_sha256=value[
                    "compiler_state_sha256"
                ],
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise SparseLawCompilerError(
                "episodic packet wire differs"
            ) from exc
        if result.deployed_wire() != wire:
            raise SparseLawCompilerError(
                "episodic packet wire is not canonical"
            )
        return result


def scan_episodic_generator_source(
    payload: bytes,
) -> ScannedSparseSource:
    if not isinstance(payload, bytes) or not payload:
        raise SparseLawCompilerError(
            "episodic source payload differs"
        )
    lines = payload.splitlines()
    if len(lines) < 2:
        raise SparseLawCompilerError("episodic source is empty")
    matches = [
        match
        for pattern in _HEADER_PATTERNS
        if (match := pattern.fullmatch(lines[0]))
    ]
    if len(matches) != 1:
        raise SparseLawCompilerError(
            "episodic source header differs"
        )
    fields = matches[0].groupdict()
    cardinality = (
        int(fields["n"])
        if fields.get("n") is not None
        else int(fields["m"]) + 1
    )
    normalized = b"\n".join(
        [f"domain-size={cardinality}".encode("ascii"), *lines[1:]]
    )
    scanned = scan_sparse_source(normalized)
    return ScannedSparseSource(
        cardinality=scanned.cardinality,
        records=scanned.records,
        action_keys=scanned.action_keys,
        record_action_indices=scanned.record_action_indices,
        source_sha256=sha256(payload).hexdigest(),
    )


def seal_episodic_generator_packet(
    batch: SparseSourceBatch,
    output: SparseCompilerOutput,
    *,
    row: int,
) -> SealedEpisodicGeneratorPacket:
    if not 0 <= row < batch.unit_ids.shape[0]:
        raise SparseLawCompilerError("episodic row differs")
    cardinality = int(batch.cardinalities[row])
    counts = torch.zeros(
        MAX_ACTIONS,
        dtype=torch.long,
        device=batch.unit_ids.device,
    )
    counts.scatter_add_(
        0,
        batch.record_action_indices[row],
        batch.record_valid[row].long(),
    )
    target_indices = torch.nonzero(
        (counts > 0) & (counts < cardinality),
        as_tuple=False,
    ).flatten()
    if target_indices.numel() != 2:
        raise SparseLawCompilerError(
            "episodic target roles differ"
        )
    predictions = output.transition_logits[row].argmax(dim=-1)
    expected = set(range(cardinality))
    transition = tuple(
        tuple(
            int(target)
            for target in predictions[index, :cardinality]
        )
        for index in target_indices.tolist()
    )
    if any(set(row_values) != expected for row_values in transition):
        raise SparseLawCompilerError(
            "episodic target map is not a permutation"
        )
    target_keys = tuple(
        batch.action_keys[row][index]
        for index in target_indices.tolist()
    )
    receipt = sha256()
    receipt.update(b"EPISODIC-GENERATOR-COMPILER-STATE-V1\0")
    receipt.update(batch.source_sha256[row].encode("ascii"))
    for row_values in transition:
        receipt.update(bytes(row_values))
    return SealedEpisodicGeneratorPacket(
        cardinality=cardinality,
        target_keys=target_keys,
        transition=transition,
        compiler_state_sha256=receipt.hexdigest(),
    )


def execute_episodic_generator_query(
    packet: SealedEpisodicGeneratorPacket,
    query: ScannedSparseQuery,
) -> int:
    if (
        not 0 <= query.start < packet.cardinality
        or not query.action_keys
    ):
        raise SparseLawCompilerError(
            "episodic query leaves packet"
        )
    try:
        actions = tuple(
            packet.target_keys.index(action)
            for action in query.action_keys
        )
    except ValueError as exc:
        raise SparseLawCompilerError(
            "episodic query action differs"
        ) from exc
    state = query.start
    for action in actions:
        state = packet.transition[action][state]
    return state


class EpisodicGeneratorConstraintCompiler(SparseLatentLawCompiler):
    """Construct and intersect an episode-local permutation program space."""

    def __init__(
        self,
        *,
        width: int = 128,
        layers: int = 2,
        heads: int = 4,
        generators: int = 2,
        maximum_depth: int = 6,
        evidence_temperature: float = 24.0,
    ) -> None:
        super().__init__(width=width, layers=layers, heads=heads)
        if (
            generators != 2
            or maximum_depth < 2
            or maximum_depth > 8
            or not 4.0 <= evidence_temperature <= 64.0
        ):
            raise SparseLawCompilerError(
                "episodic closure geometry differs"
            )
        del self.state_embedding
        del self.cardinality_embedding
        del self.pair_encoder
        del self.cross_attention
        del self.transition_decoder
        self.generators = int(generators)
        self.maximum_depth = int(maximum_depth)
        self.evidence_temperature = float(evidence_temperature)

    @property
    def closure_receipt(self) -> EpisodicClosureReceipt:
        return EpisodicClosureReceipt(
            generators=self.generators,
            maximum_depth=self.maximum_depth,
            syntactic_programs=sum(
                self.generators**depth
                for depth in range(self.maximum_depth + 1)
            ),
        )

    def _direction(
        self,
        batch: SparseSourceBatch,
        direction_sign: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        logits = self.direction_head(
            number_hidden.reshape(rows * records, self.width * 2)
        ).reshape(rows, records)
        logits = direction_sign * logits
        soft = logits.sigmoid()
        hard = logits.gt(0).to(soft.dtype)
        straight_through = hard + soft - soft.detach()
        return logits, straight_through

    def _direct_action_matrices(
        self,
        *,
        batch: SparseSourceBatch,
        row: int,
        cardinality: int,
        direction: torch.Tensor,
        observation_target_shift: int,
        observations_zeroed: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        first = (
            batch.number_values[row, :, 0] + observation_target_shift
        ) % cardinality
        second = (
            batch.number_values[row, :, 1] + observation_target_shift
        ) % cardinality
        source = (
            direction[:, None]
            * torch.nn.functional.one_hot(
                first,
                num_classes=cardinality,
            ).to(direction.dtype)
            + (1.0 - direction[:, None])
            * torch.nn.functional.one_hot(
                second,
                num_classes=cardinality,
            ).to(direction.dtype)
        )
        target = (
            direction[:, None]
            * torch.nn.functional.one_hot(
                second,
                num_classes=cardinality,
            ).to(direction.dtype)
            + (1.0 - direction[:, None])
            * torch.nn.functional.one_hot(
                first,
                num_classes=cardinality,
            ).to(direction.dtype)
        )
        record_matrix = torch.einsum("ri,rj->rij", source, target)
        record_valid = batch.record_valid[row].to(direction.dtype)
        if observations_zeroed:
            record_matrix = torch.zeros_like(record_matrix)
        action_one_hot = torch.nn.functional.one_hot(
            batch.record_action_indices[row],
            num_classes=MAX_ACTIONS,
        ).to(direction.dtype)
        action_one_hot = action_one_hot * record_valid[:, None]
        direct = torch.einsum(
            "ra,rij->aij",
            action_one_hot,
            record_matrix,
        )
        counts = action_one_hot.sum(dim=0)
        support_indices = counts.topk(self.generators).indices
        return direct, counts, support_indices, record_valid

    def _program_bank(
        self,
        supports: torch.Tensor,
        cardinality: int,
    ) -> torch.Tensor:
        identity = torch.eye(
            cardinality,
            dtype=supports.dtype,
            device=supports.device,
        )[None]
        programs = [identity]
        frontier = identity
        for _depth in range(self.maximum_depth):
            frontier = torch.einsum(
                "pij,gjk->pgik",
                frontier,
                supports,
            ).reshape(-1, cardinality, cardinality)
            programs.append(frontier)
        return torch.cat(programs, dim=0)

    def forward(
        self,
        batch: SparseSourceBatch,
        *,
        direction_sign: float = 1.0,
        observation_target_shift: int = 0,
        observations_zeroed: bool = False,
        support_order_reversed: bool = False,
        support_semantics_deranged: bool = False,
    ) -> SparseCompilerOutput:
        if direction_sign not in {-1.0, 1.0}:
            raise SparseLawCompilerError("direction sign differs")
        if observation_target_shift not in {0, 1}:
            raise SparseLawCompilerError(
                "observation target shift differs"
            )
        direction_logits, direction = self._direction(
            batch,
            direction_sign,
        )
        rows = batch.unit_ids.shape[0]
        transition_logits = torch.full(
            (
                rows,
                MAX_ACTIONS,
                MAX_CARDINALITY,
                MAX_CARDINALITY,
            ),
            -1.0e4,
            dtype=direction.dtype,
            device=direction.device,
        )
        for row in range(rows):
            cardinality = int(batch.cardinalities[row])
            direct, counts, support_indices, _record_valid = (
                self._direct_action_matrices(
                    batch=batch,
                    row=row,
                    cardinality=cardinality,
                    direction=direction[row],
                    observation_target_shift=observation_target_shift,
                    observations_zeroed=observations_zeroed,
                )
            )
            supports = direct[support_indices]
            if support_order_reversed:
                supports = supports.flip(0)
            if support_semantics_deranged:
                target_permutation = torch.arange(
                    cardinality,
                    device=supports.device,
                )
                target_permutation[0] = 1
                target_permutation[1] = 0
                supports = supports.clone()
                supports[0] = supports[0, :, target_permutation]
            bank = self._program_bank(supports, cardinality)
            first = (
                batch.number_values[row, :, 0]
                + observation_target_shift
            ) % cardinality
            second = (
                batch.number_values[row, :, 1]
                + observation_target_shift
            ) % cardinality
            forward_match = bank[:, first, second]
            reverse_match = bank[:, second, first]
            record_match = (
                direction[row][None] * forward_match
                + (1.0 - direction[row][None]) * reverse_match
            )
            if observations_zeroed:
                record_match = torch.zeros_like(record_match)
            evidence: list[torch.Tensor] = []
            for action in range(MAX_ACTIONS):
                valid = (
                    batch.record_valid[row]
                    & batch.record_action_indices[row].eq(action)
                ).to(record_match.dtype)
                evidence.append(
                    torch.einsum("pr,r->p", record_match, valid)
                )
            scores = torch.stack(evidence)
            posterior = (
                scores * self.evidence_temperature
            ).softmax(dim=-1)
            transition_probability = torch.einsum(
                "ap,pst->ast",
                posterior,
                bank,
            )
            transition_logits[
                row,
                :,
                :cardinality,
                :cardinality,
            ] = transition_probability.clamp_min(1e-12).log()
            if bool((counts[support_indices] != cardinality).any()):
                transition_logits[row] = -1.0e4
        return SparseCompilerOutput(
            direction_logits=direction_logits,
            transition_logits=transition_logits,
        )


__all__ = [
    "EpisodicClosureReceipt",
    "EpisodicGeneratorConstraintCompiler",
    "SealedEpisodicGeneratorPacket",
    "execute_episodic_generator_query",
    "scan_episodic_generator_source",
    "seal_episodic_generator_packet",
]
