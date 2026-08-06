"""Whole-mention compiler and source-sealed scalar version space for NFE1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import (
    BYTE_OFFSET,
    BYTE_VOCAB_SIZE,
    CLS_ID,
    MAX_SEGMENT_BYTES,
    PAD_ID,
)
from diverge_nfe1_data import (
    ROLE_NAMES,
    SCALAR_OPERATIONS,
    apply_scalar,
    scan_signed_integer_spans,
)


ANSWER = "answer"
ABSTAIN = "abstain"
REJECT = "reject"
SCHEMA = "shohin-diverge-nfe1-runtime-v1"
MAX_RUNTIME_ABS = 2**63 - 1


class NFE1RuntimeError(RuntimeError):
    """The NFE1 source-sealed runtime contract was violated."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_bytes(payload: object) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def encode_source(text: str, max_bytes: int = MAX_SEGMENT_BYTES) -> tuple[int, ...]:
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as error:
        raise NFE1RuntimeError("NFE1 source is not ASCII") from error
    if not raw or len(raw) + 1 > max_bytes:
        raise NFE1RuntimeError("NFE1 source length differs")
    return (CLS_ID, *(value + BYTE_OFFSET for value in raw))


@dataclass(frozen=True, slots=True)
class MentionConfig:
    width: int = 128
    layers: int = 2
    max_bytes: int = MAX_SEGMENT_BYTES

    def validate(self) -> None:
        if self.width != 128 or self.layers != 2 or self.width % 2:
            raise NFE1RuntimeError("NFE1 mention geometry differs")
        if self.max_bytes != MAX_SEGMENT_BYTES:
            raise NFE1RuntimeError("NFE1 source width differs")


class WholeMentionRoleModel(nn.Module):
    """Assign three complete numeric mentions to LHS/argument/RHS."""

    def __init__(self, config: MentionConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, config.width)
        self.encoder = nn.GRU(
            input_size=config.width,
            hidden_size=config.width // 2,
            num_layers=config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.role_head = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, len(ROLE_NAMES)),
        )

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mention_bounds: torch.Tensor,
    ) -> torch.Tensor:
        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or byte_ids.shape[1] != self.config.max_bytes
            or byte_ids.dtype != torch.long
            or mention_bounds.shape != (byte_ids.shape[0], 3, 2)
        ):
            raise NFE1RuntimeError("NFE1 mention tensor interface differs")
        lengths = attention_mask.bool().sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise NFE1RuntimeError("NFE1 source mask or CLS differs")
        embedded = self.embedding(byte_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=self.config.max_bytes,
        )
        hidden = self.output_norm(hidden)
        positions = torch.arange(self.config.max_bytes, device=byte_ids.device).view(
            1, 1, -1
        )
        starts = mention_bounds[:, :, 0].unsqueeze(-1)
        ends = mention_bounds[:, :, 1].unsqueeze(-1)
        mention_mask = (positions >= starts) & (positions < ends)
        if torch.any(mention_mask.sum(dim=-1) < 1):
            raise NFE1RuntimeError("NFE1 mention span is empty")
        pooled = torch.einsum(
            "bms,bsw->bmw", mention_mask.to(hidden.dtype), hidden
        ) / mention_mask.sum(dim=-1, keepdim=True).to(hidden.dtype)
        return self.role_head(pooled).float()

    def record(self) -> dict[str, object]:
        return {"schema": SCHEMA, "config": asdict(self.config)}


def tensorize_sources(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    max_bytes: int = MAX_SEGMENT_BYTES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(records)
    byte_ids = torch.full((batch, max_bytes), PAD_ID, dtype=torch.long)
    attention = torch.zeros((batch, max_bytes), dtype=torch.bool)
    bounds = torch.zeros((batch, 3, 2), dtype=torch.long)
    targets = torch.zeros((batch, 3), dtype=torch.long)
    for row_index, record in enumerate(records):
        source = str(record.get("source_text", record.get("text", "")))
        encoded = encode_source(source, max_bytes)
        byte_ids[row_index, : len(encoded)] = torch.tensor(encoded)
        attention[row_index, : len(encoded)] = True
        spans = scan_signed_integer_spans(source)
        if len(spans) != 3:
            raise NFE1RuntimeError("NFE1 source does not expose three mentions")
        for mention_index, (start, end) in enumerate(spans):
            bounds[row_index, mention_index] = torch.tensor((start + 1, end + 1))
        role_ids = record.get("role_ids", (0, 1, 2))
        targets[row_index] = torch.tensor(tuple(int(value) for value in role_ids))
    return (
        byte_ids.to(device),
        attention.to(device),
        bounds.to(device),
        targets.to(device),
    )


_ROLE_PERMUTATIONS = tuple(itertools.permutations(range(len(ROLE_NAMES))))


def hard_role_permutation(logits: torch.Tensor) -> tuple[int, int, int]:
    if logits.shape != (3, 3):
        raise NFE1RuntimeError("NFE1 role logits differ")
    scores = [
        sum(float(logits[index, role]) for index, role in enumerate(permutation))
        for permutation in _ROLE_PERMUTATIONS
    ]
    best = max(range(len(scores)), key=lambda index: (scores[index], -index))
    return _ROLE_PERMUTATIONS[best]


@dataclass(frozen=True, slots=True)
class CompiledMention:
    role: str
    start: int
    end: int
    value: int
    score: float

    def record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EquationPacket:
    step_index: int
    source_sha256: str
    mention_commitment: str
    lhs: int
    argument: int
    rhs: int
    operation_support: tuple[float, float, float]

    def record(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "source_sha256": self.source_sha256,
            "mention_commitment": self.mention_commitment,
            "lhs": self.lhs,
            "argument": self.argument,
            "rhs": self.rhs,
            "operation_support": list(self.operation_support),
            "candidate_operations": list(SCALAR_OPERATIONS),
        }


@dataclass(frozen=True, slots=True)
class EpisodePacket:
    identity_sha256: str
    source_commitment: str
    steps: tuple[EquationPacket, ...]
    commitment: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "identity_sha256": self.identity_sha256,
            "source_commitment": self.source_commitment,
            "steps": [step.record() for step in self.steps],
            "commitment": self.commitment,
        }


@dataclass(frozen=True, slots=True)
class EvidenceReceipt:
    packet_commitment: str
    step_index: int
    source_sha256: str
    mention_commitment: str
    lhs: int
    rhs: int
    commitment: str

    def unsigned_record(self) -> dict[str, object]:
        return {
            "packet_commitment": self.packet_commitment,
            "step_index": self.step_index,
            "source_sha256": self.source_sha256,
            "mention_commitment": self.mention_commitment,
            "lhs": self.lhs,
            "rhs": self.rhs,
        }

    def record(self) -> dict[str, object]:
        return {**self.unsigned_record(), "commitment": self.commitment}


@dataclass(frozen=True, slots=True)
class QueryPacket:
    packet_commitment: str
    query_sha256: str
    commitment: str

    def record(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StateGroup:
    state: int
    assignment_mask: int

    def record(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    packet_commitment: str
    groups: tuple[StateGroup, ...]
    represented_worlds: int
    logical_applications: int
    unique_applications: int
    peak_groups: int
    rejected: bool = False
    rejection_reason: str | None = None

    def record(self) -> dict[str, object]:
        return {
            "packet_commitment": self.packet_commitment,
            "groups": [group.record() for group in self.groups],
            "represented_worlds": self.represented_worlds,
            "logical_applications": self.logical_applications,
            "unique_applications": self.unique_applications,
            "peak_groups": self.peak_groups,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True, slots=True)
class QueryDecision:
    disposition: str
    answer: int | None = None
    reason: str | None = None

    def record(self) -> dict[str, object]:
        return asdict(self)


@torch.no_grad()
def compile_mentions_batch(
    model: WholeMentionRoleModel,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> list[tuple[CompiledMention, ...]]:
    model.eval()
    byte_ids, attention, bounds, _ = tensorize_sources(records, device)
    logits = model(byte_ids, attention, bounds).cpu()
    output: list[tuple[CompiledMention, ...]] = []
    for row_index, record in enumerate(records):
        source = str(record.get("source_text", record.get("text", "")))
        spans = scan_signed_integer_spans(source)
        assignment = hard_role_permutation(logits[row_index])
        mentions = tuple(
            CompiledMention(
                ROLE_NAMES[role],
                start,
                end,
                int(source[start:end]),
                float(logits[row_index, mention_index, role]),
            )
            for mention_index, ((start, end), role) in enumerate(
                zip(spans, assignment, strict=True)
            )
        )
        output.append(mentions)
    return output


def mention_values(mentions: Sequence[CompiledMention]) -> tuple[int, int, int]:
    by_role = {mention.role: mention.value for mention in mentions}
    if set(by_role) != set(ROLE_NAMES) or len(mentions) != 3:
        raise NFE1RuntimeError("NFE1 mention assignment is not one-to-one")
    return by_role["LHS"], by_role["ARGUMENT"], by_role["RHS"]


def compile_episode(
    identity_sha256: str,
    sources: Sequence[str],
    compiled_mentions: Sequence[Sequence[CompiledMention]],
    operation_support: Sequence[Sequence[float]],
) -> EpisodePacket:
    if not (
        len(sources) == len(compiled_mentions) == len(operation_support)
        and 2 <= len(sources) <= 5
    ):
        raise NFE1RuntimeError("NFE1 episode geometry differs")
    steps: list[EquationPacket] = []
    source_hashes: list[str] = []
    for step_index, (source, mentions, support) in enumerate(
        zip(sources, compiled_mentions, operation_support, strict=True)
    ):
        source_sha256 = hashlib.sha256(source.encode("ascii")).hexdigest()
        source_hashes.append(source_sha256)
        lhs, argument, rhs = mention_values(mentions)
        mention_record = [mention.record() for mention in mentions]
        mention_commitment = canonical_sha256(
            {"source_sha256": source_sha256, "mentions": mention_record}
        )
        if len(support) != 3 or not all(
            math.isfinite(float(value)) for value in support
        ):
            raise NFE1RuntimeError("NFE1 operation support differs")
        steps.append(
            EquationPacket(
                step_index,
                source_sha256,
                mention_commitment,
                lhs,
                argument,
                rhs,
                tuple(float(value) for value in support),
            )
        )
    source_commitment = canonical_sha256(
        {"identity_sha256": identity_sha256, "source_hashes": source_hashes}
    )
    unsigned = {
        "schema": SCHEMA,
        "identity_sha256": identity_sha256,
        "source_commitment": source_commitment,
        "steps": [step.record() for step in steps],
    }
    return EpisodePacket(
        identity_sha256,
        source_commitment,
        tuple(steps),
        canonical_sha256(unsigned),
    )


def issue_evidence(packet: EpisodePacket) -> tuple[EvidenceReceipt, ...]:
    receipts: list[EvidenceReceipt] = []
    for step in packet.steps:
        unsigned = {
            "packet_commitment": packet.commitment,
            "step_index": step.step_index,
            "source_sha256": step.source_sha256,
            "mention_commitment": step.mention_commitment,
            "lhs": step.lhs,
            "rhs": step.rhs,
        }
        receipts.append(
            EvidenceReceipt(**unsigned, commitment=canonical_sha256(unsigned))
        )
    return tuple(receipts)


def compile_query(packet: EpisodePacket, text: str) -> QueryPacket:
    query_sha256 = hashlib.sha256(text.encode("ascii")).hexdigest()
    unsigned = {
        "packet_commitment": packet.commitment,
        "query_sha256": query_sha256,
    }
    return QueryPacket(**unsigned, commitment=canonical_sha256(unsigned))


def verify_evidence(
    packet: EpisodePacket,
    receipts: Sequence[EvidenceReceipt],
) -> tuple[bool, str | None]:
    if len(receipts) != len(packet.steps):
        return False, "evidence_count"
    for step, receipt in zip(packet.steps, receipts, strict=True):
        if canonical_sha256(receipt.unsigned_record()) != receipt.commitment:
            return False, "evidence_commitment"
        if receipt.packet_commitment != packet.commitment:
            return False, "evidence_packet"
        if receipt.step_index != step.step_index:
            return False, "evidence_step"
        if receipt.source_sha256 != step.source_sha256:
            return False, "evidence_source"
        if receipt.mention_commitment != step.mention_commitment:
            return False, "evidence_mention"
        if receipt.lhs != step.lhs or receipt.rhs != step.rhs:
            return False, "evidence_value"
    return True, None


def assignments(depth: int) -> tuple[tuple[int, ...], ...]:
    if not 1 <= depth <= 8:
        raise NFE1RuntimeError("NFE1 assignment depth differs")
    return tuple(itertools.product(range(3), repeat=depth))


def _option_mask(depth: int, step_index: int, option: int) -> int:
    mask = 0
    for index, assignment in enumerate(assignments(depth)):
        if assignment[step_index] == option:
            mask |= 1 << index
    return mask


def _rejected(packet: EpisodePacket, reason: str) -> ExecutionReceipt:
    return ExecutionReceipt(packet.commitment, (), 0, 0, 0, 0, True, reason)


def _merge_groups(groups: Iterable[StateGroup]) -> tuple[StateGroup, ...]:
    merged: dict[int, int] = {}
    for group in groups:
        if group.assignment_mask <= 0:
            continue
        merged[group.state] = merged.get(group.state, 0) | group.assignment_mask
    return tuple(StateGroup(state, merged[state]) for state in sorted(merged))


def execute_factorized(
    packet: EpisodePacket,
    evidence: Sequence[EvidenceReceipt] | None = None,
    *,
    reset_initial_state: bool = False,
    operand_semantic_shift: bool = False,
) -> ExecutionReceipt:
    if evidence is not None:
        valid, reason = verify_evidence(packet, evidence)
        if not valid:
            return _rejected(packet, reason or "evidence")
    depth = len(packet.steps)
    world_count = 3**depth
    initial = packet.steps[0].lhs + (1 if reset_initial_state else 0)
    groups = (StateGroup(initial, (1 << world_count) - 1),)
    logical = 0
    unique = 0
    peak = 1
    for step_index, step in enumerate(packet.steps):
        next_groups: list[StateGroup] = []
        argument = step.argument + (1 if operand_semantic_shift else 0)
        for group in groups:
            for option, operation in enumerate(SCALAR_OPERATIONS):
                mask = group.assignment_mask & _option_mask(depth, step_index, option)
                if not mask:
                    continue
                logical += mask.bit_count()
                unique += 1
                successor = apply_scalar(operation, group.state, argument)
                if abs(successor) > MAX_RUNTIME_ABS:
                    return _rejected(packet, "overflow")
                if evidence is not None:
                    receipt = evidence[step_index]
                    if group.state != receipt.lhs or successor != receipt.rhs:
                        continue
                next_groups.append(StateGroup(successor, mask))
        groups = _merge_groups(next_groups)
        peak = max(peak, len(groups))
        if not groups:
            return _rejected(packet, "empty_version_space")
    return ExecutionReceipt(
        packet.commitment,
        groups,
        sum(group.assignment_mask.bit_count() for group in groups),
        logical,
        unique,
        peak,
    )


def query_receipt(
    packet: EpisodePacket,
    receipt: ExecutionReceipt,
    query: QueryPacket,
) -> QueryDecision:
    unsigned = {
        "packet_commitment": query.packet_commitment,
        "query_sha256": query.query_sha256,
    }
    if canonical_sha256(unsigned) != query.commitment:
        return QueryDecision(REJECT, reason="query_commitment")
    if query.packet_commitment != packet.commitment:
        return QueryDecision(REJECT, reason="query_packet")
    if receipt.packet_commitment != packet.commitment or receipt.rejected:
        return QueryDecision(REJECT, reason=receipt.rejection_reason or "execution")
    values = {group.state for group in receipt.groups}
    if len(values) != 1:
        return QueryDecision(ABSTAIN, reason="query_not_invariant")
    return QueryDecision(ANSWER, answer=next(iter(values)))


def execute_assignment(
    packet: EpisodePacket,
    assignment: Sequence[int],
    evidence: Sequence[EvidenceReceipt] | None = None,
    *,
    reset_initial_state: bool = False,
    operand_semantic_shift: bool = False,
) -> int | None:
    if len(assignment) != len(packet.steps):
        raise NFE1RuntimeError("NFE1 assignment length differs")
    if evidence is not None:
        valid, _ = verify_evidence(packet, evidence)
        if not valid:
            return None
    state = packet.steps[0].lhs + (1 if reset_initial_state else 0)
    for step_index, (step, option) in enumerate(
        zip(packet.steps, assignment, strict=True)
    ):
        if int(option) not in range(3):
            raise NFE1RuntimeError("NFE1 assignment option differs")
        argument = step.argument + (1 if operand_semantic_shift else 0)
        successor = apply_scalar(SCALAR_OPERATIONS[int(option)], state, argument)
        if abs(successor) > MAX_RUNTIME_ABS:
            return None
        if evidence is not None:
            receipt = evidence[step_index]
            if state != receipt.lhs or successor != receipt.rhs:
                return None
        state = successor
    return state


def receipt_extensional_map(
    packet: EpisodePacket,
    receipt: ExecutionReceipt,
) -> dict[tuple[int, ...], int]:
    if receipt.rejected:
        return {}
    output: dict[tuple[int, ...], int] = {}
    universe = assignments(len(packet.steps))
    for group in receipt.groups:
        for index, assignment in enumerate(universe):
            if group.assignment_mask & (1 << index):
                output[assignment] = group.state
    return output


def enumerate_extensional_map(
    packet: EpisodePacket,
    evidence: Sequence[EvidenceReceipt] | None = None,
    *,
    reset_initial_state: bool = False,
    operand_semantic_shift: bool = False,
) -> dict[tuple[int, ...], int]:
    output: dict[tuple[int, ...], int] = {}
    for assignment in assignments(len(packet.steps)):
        state = execute_assignment(
            packet,
            assignment,
            evidence,
            reset_initial_state=reset_initial_state,
            operand_semantic_shift=operand_semantic_shift,
        )
        if state is not None:
            output[assignment] = state
    return output


def assignment_score(packet: EpisodePacket, assignment: Sequence[int]) -> float:
    return sum(
        step.operation_support[int(option)]
        for step, option in zip(packet.steps, assignment, strict=True)
    )


def ranked_assignments(packet: EpisodePacket) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(
            assignments(len(packet.steps)),
            key=lambda assignment: (-assignment_score(packet, assignment), assignment),
        )
    )


def query_particles(
    packet: EpisodePacket,
    query: QueryPacket,
    candidates: Sequence[Sequence[int]],
    evidence: Sequence[EvidenceReceipt] | None,
) -> QueryDecision:
    states = {
        state
        for assignment in candidates
        if (state := execute_assignment(packet, assignment, evidence)) is not None
    }
    if not states:
        return QueryDecision(REJECT, reason="particles_empty")
    if len(states) != 1:
        return QueryDecision(ABSTAIN, reason="particles_disagree")
    synthetic = ExecutionReceipt(
        packet.commitment,
        (StateGroup(next(iter(states)), 1),),
        len(candidates),
        0,
        0,
        1,
    )
    return query_receipt(packet, synthetic, query)


def query_soft_answers(
    packet: EpisodePacket,
    query: QueryPacket,
    evidence: Sequence[EvidenceReceipt] | None,
) -> QueryDecision:
    weighted: dict[int, float] = {}
    for assignment in assignments(len(packet.steps)):
        state = execute_assignment(packet, assignment, evidence)
        if state is None:
            continue
        weighted[state] = weighted.get(state, 0.0) + math.exp(
            min(60.0, assignment_score(packet, assignment))
        )
    if not weighted:
        return QueryDecision(REJECT, reason="soft_empty")
    answer = min(weighted, key=lambda value: (-weighted[value], value))
    synthetic = ExecutionReceipt(
        packet.commitment,
        (StateGroup(answer, 1),),
        len(weighted),
        0,
        0,
        len(weighted),
    )
    return query_receipt(packet, synthetic, query)


def factorized_total_bytes(
    packet: EpisodePacket,
    receipt: ExecutionReceipt,
    evidence: Sequence[EvidenceReceipt],
    query: QueryPacket,
) -> int:
    return canonical_bytes(
        {
            "packet": packet.record(),
            "execution": receipt.record(),
            "evidence": [item.record() for item in evidence],
            "query": query.record(),
        }
    )


def whole_particle_record(
    packet: EpisodePacket,
    assignment: Sequence[int],
    evidence: Sequence[EvidenceReceipt],
    query: QueryPacket,
) -> dict[str, object]:
    return {
        "packet_commitment": packet.commitment,
        "source_commitment": packet.source_commitment,
        "initial_state": packet.steps[0].lhs,
        "operations": [SCALAR_OPERATIONS[int(option)] for option in assignment],
        "arguments": [step.argument for step in packet.steps],
        "source_hashes": [step.source_sha256 for step in packet.steps],
        "evidence": [item.record() for item in evidence],
        "query": query.record(),
    }


def particle_capacity_for_bytes(
    packet: EpisodePacket,
    candidates: Sequence[Sequence[int]],
    evidence: Sequence[EvidenceReceipt],
    query: QueryPacket,
    budget: int,
) -> tuple[int, int]:
    used = 0
    capacity = 0
    for assignment in candidates:
        charge = canonical_bytes(
            whole_particle_record(packet, assignment, evidence, query)
        )
        if used + charge > budget:
            break
        used += charge
        capacity += 1
    return capacity, used


def all_particle_bytes(
    packet: EpisodePacket,
    evidence: Sequence[EvidenceReceipt],
    query: QueryPacket,
) -> int:
    return sum(
        canonical_bytes(whole_particle_record(packet, assignment, evidence, query))
        for assignment in assignments(len(packet.steps))
    )


def mutate_receipt(receipt: EvidenceReceipt, field: str) -> EvidenceReceipt:
    if field == "source":
        mutated = replace(receipt, source_sha256="0" * 64)
    elif field == "step":
        mutated = replace(receipt, step_index=receipt.step_index + 1)
    elif field == "mention":
        mutated = replace(receipt, mention_commitment="0" * 64)
    elif field == "value":
        mutated = replace(receipt, lhs=receipt.lhs + 1)
    else:
        raise NFE1RuntimeError("unknown evidence mutation")
    return replace(mutated, commitment=canonical_sha256(mutated.unsigned_record()))


__all__ = [
    "ABSTAIN",
    "ANSWER",
    "REJECT",
    "CompiledMention",
    "EpisodePacket",
    "EvidenceReceipt",
    "ExecutionReceipt",
    "MentionConfig",
    "NFE1RuntimeError",
    "QueryDecision",
    "QueryPacket",
    "StateGroup",
    "WholeMentionRoleModel",
    "all_particle_bytes",
    "assignment_score",
    "assignments",
    "canonical_bytes",
    "canonical_sha256",
    "compile_episode",
    "compile_mentions_batch",
    "compile_query",
    "encode_source",
    "enumerate_extensional_map",
    "execute_assignment",
    "execute_factorized",
    "factorized_total_bytes",
    "hard_role_permutation",
    "issue_evidence",
    "mention_values",
    "mutate_receipt",
    "particle_capacity_for_bytes",
    "query_particles",
    "query_receipt",
    "query_soft_answers",
    "ranked_assignments",
    "receipt_extensional_map",
    "tensorize_sources",
    "verify_evidence",
    "whole_particle_record",
]
