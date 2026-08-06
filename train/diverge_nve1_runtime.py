"""Learned natural-evidence compiler and sealing adapter for DIVERGE-NVE1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_OFFSET, BYTE_VOCAB_SIZE, CLS_ID, PAD_ID
from diverge_nve1_data import (
    MAX_EVIDENCE_BYTES,
    NUMERIC_ROLES,
    SYMBOL_ROLES,
    scan_rational_spans,
    symbol_occurrence_groups,
)
from diverge_tfs1_data import FAULT_LINES
from diverge_tfs1_runtime import (
    CompiledPacket,
    FactorizedReceipt,
    TFS1RuntimeError,
    execute_factorized,
)
from diverge_tol1_ir import TOL1IRError, format_fraction, parse_fraction


SCHEMA = "shohin-diverge-nve1-runtime-v1"


class NVE1RuntimeError(RuntimeError):
    """A learned NVE1 evidence packet or seal is invalid."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def module_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def encode_source(
    text: str,
    max_bytes: int = MAX_EVIDENCE_BYTES,
) -> tuple[int, ...]:
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as error:
        raise NVE1RuntimeError("NVE1 evidence is not ASCII") from error
    if not raw or len(raw) + 1 > max_bytes:
        raise NVE1RuntimeError("NVE1 evidence source width differs")
    return (CLS_ID, *(value + BYTE_OFFSET for value in raw))


@dataclass(frozen=True, slots=True)
class EvidenceCompilerConfig:
    width: int = 192
    layers: int = 2
    max_bytes: int = MAX_EVIDENCE_BYTES

    def validate(self) -> None:
        if self.width != 192 or self.layers != 2 or self.width % 2:
            raise NVE1RuntimeError("NVE1 compiler geometry differs")
        if self.max_bytes != MAX_EVIDENCE_BYTES:
            raise NVE1RuntimeError("NVE1 compiler source width differs")


class NaturalEvidenceCompiler(nn.Module):
    """Assign complete numeric mentions and source-owned identity groups."""

    def __init__(self, config: EvidenceCompilerConfig) -> None:
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
        self.numeric_head = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, len(NUMERIC_ROLES)),
        )
        self.symbol_head = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, len(SYMBOL_ROLES)),
        )

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_bounds: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or byte_ids.shape[1] != self.config.max_bytes
            or byte_ids.dtype != torch.long
            or numeric_bounds.shape != (byte_ids.shape[0], 2, 2)
            or symbol_masks.shape != (byte_ids.shape[0], 2, self.config.max_bytes)
        ):
            raise NVE1RuntimeError("NVE1 compiler tensor interface differs")
        lengths = attention_mask.bool().sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise NVE1RuntimeError("NVE1 source mask or CLS differs")
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
        numeric_mask = (positions >= numeric_bounds[:, :, 0].unsqueeze(-1)) & (
            positions < numeric_bounds[:, :, 1].unsqueeze(-1)
        )
        if torch.any(numeric_mask.sum(dim=-1) < 1) or torch.any(
            symbol_masks.sum(dim=-1) < 1
        ):
            raise NVE1RuntimeError("NVE1 evidence group is empty")
        numeric_hidden = torch.einsum(
            "bms,bsw->bmw", numeric_mask.to(hidden.dtype), hidden
        ) / numeric_mask.sum(dim=-1, keepdim=True).to(hidden.dtype)
        symbol_hidden = torch.einsum(
            "bms,bsw->bmw", symbol_masks.to(hidden.dtype), hidden
        ) / symbol_masks.sum(dim=-1, keepdim=True).to(hidden.dtype)
        return self.numeric_head(numeric_hidden).float(), self.symbol_head(
            symbol_hidden
        ).float()

    def record(self) -> dict[str, object]:
        return {"schema": SCHEMA, "config": asdict(self.config)}


def tensorize_sources(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch = len(records)
    byte_ids = torch.full((batch, MAX_EVIDENCE_BYTES), PAD_ID, dtype=torch.long)
    attention = torch.zeros((batch, MAX_EVIDENCE_BYTES), dtype=torch.bool)
    numeric_bounds = torch.zeros((batch, 2, 2), dtype=torch.long)
    symbol_masks = torch.zeros((batch, 2, MAX_EVIDENCE_BYTES), dtype=torch.bool)
    numeric_targets = torch.zeros((batch, 2), dtype=torch.long)
    symbol_targets = torch.zeros((batch, 2), dtype=torch.long)
    for row_index, record in enumerate(records):
        text = str(record["source_text"])
        encoded = encode_source(text)
        byte_ids[row_index, : len(encoded)] = torch.tensor(encoded)
        attention[row_index, : len(encoded)] = True
        spans = scan_rational_spans(text)
        if len(spans) != 2:
            raise NVE1RuntimeError("NVE1 evidence does not expose two numbers")
        for mention_index, (start, end) in enumerate(spans):
            numeric_bounds[row_index, mention_index] = torch.tensor(
                (start + 1, end + 1)
            )
        symbols = tuple(str(value) for value in record["symbols"])
        groups = symbol_occurrence_groups(text, symbols)
        if len(groups) != 2:
            raise NVE1RuntimeError("NVE1 evidence does not expose two symbol groups")
        for group_index, (_, occurrences) in enumerate(groups):
            for start, end in occurrences:
                symbol_masks[row_index, group_index, start + 1 : end + 1] = True
        numeric_targets[row_index] = torch.tensor(
            tuple(int(value) for value in record.get("numeric_role_ids", (0, 1)))
        )
        symbol_targets[row_index] = torch.tensor(
            tuple(int(value) for value in record.get("symbol_role_ids", (0, 1)))
        )
    return (
        byte_ids.to(device),
        attention.to(device),
        numeric_bounds.to(device),
        symbol_masks.to(device),
        numeric_targets.to(device),
        symbol_targets.to(device),
    )


_PERMUTATIONS = tuple(itertools.permutations(range(2)))


def hard_role_permutation(logits: torch.Tensor) -> tuple[int, int]:
    if logits.shape != (2, 2):
        raise NVE1RuntimeError("NVE1 role logits differ")
    scores = [
        sum(float(logits[index, role]) for index, role in enumerate(permutation))
        for permutation in _PERMUTATIONS
    ]
    best = max(range(len(scores)), key=lambda index: (scores[index], -index))
    return _PERMUTATIONS[best]


@dataclass(frozen=True, slots=True)
class NaturalEvidenceReceipt:
    index: int
    packet_commitment: str
    source_commitment: str
    evidence_source_sha256: str
    compiler_commitment: str
    step_index: int
    target: str
    distractor: str
    value: str
    numeric_provenance: tuple[tuple[str, int, int, str], ...]
    symbol_provenance: tuple[tuple[str, str, tuple[tuple[int, int], ...]], ...]
    commitment: str

    def payload(self) -> dict[str, object]:
        return {
            "index": self.index,
            "packet_commitment": self.packet_commitment,
            "source_commitment": self.source_commitment,
            "evidence_source_sha256": self.evidence_source_sha256,
            "compiler_commitment": self.compiler_commitment,
            "step_index": self.step_index,
            "target": self.target,
            "distractor": self.distractor,
            "value": self.value,
            "numeric_provenance": [list(value) for value in self.numeric_provenance],
            "symbol_provenance": [
                [role, symbol, [list(span) for span in spans]]
                for role, symbol, spans in self.symbol_provenance
            ],
        }

    def record(self) -> dict[str, object]:
        return {**self.payload(), "commitment": self.commitment}


@dataclass(frozen=True, slots=True)
class EvidenceCompilation:
    receipt: NaturalEvidenceReceipt | None
    error: str | None

    def record(self) -> dict[str, object]:
        return {
            "receipt": None if self.receipt is None else self.receipt.record(),
            "error": self.error,
        }


def _compile_one(
    text: str,
    packet: CompiledPacket,
    index: int,
    numeric_assignment: Sequence[int],
    symbol_assignment: Sequence[int],
    *,
    compiler_commitment: str,
) -> NaturalEvidenceReceipt:
    numeric_spans = scan_rational_spans(text)
    symbol_groups = symbol_occurrence_groups(text, packet.symbols)
    if len(numeric_spans) != 2 or len(symbol_groups) != 2:
        raise NVE1RuntimeError("NVE1 candidate geometry differs")
    numeric = {
        NUMERIC_ROLES[int(role)]: (start, end, text[start:end])
        for (start, end), role in zip(numeric_spans, numeric_assignment, strict=True)
    }
    symbols = {
        SYMBOL_ROLES[int(role)]: (symbol, spans)
        for (symbol, spans), role in zip(symbol_groups, symbol_assignment, strict=True)
    }
    step_text = numeric["STEP"][2]
    if "/" in step_text:
        raise NVE1RuntimeError("NVE1 step assignment is not an integer")
    try:
        step_ordinal = int(step_text)
        value = format_fraction(parse_fraction(numeric["VALUE"][2]))
    except (ValueError, TOL1IRError) as error:
        raise NVE1RuntimeError("NVE1 numeric assignment cannot be parsed") from error
    if step_ordinal <= 0:
        raise NVE1RuntimeError("NVE1 step ordinal is not positive")
    target, target_spans = symbols["TARGET"]
    distractor, distractor_spans = symbols["DISTRACTOR"]
    numeric_provenance = tuple(
        (
            role,
            numeric[role][0],
            numeric[role][1],
            hashlib.sha256(numeric[role][2].encode("ascii")).hexdigest(),
        )
        for role in NUMERIC_ROLES
    )
    symbol_provenance = (
        ("TARGET", target, target_spans),
        ("DISTRACTOR", distractor, distractor_spans),
    )
    provisional = NaturalEvidenceReceipt(
        index=index,
        packet_commitment=packet.commitment,
        source_commitment=packet.source_commitment,
        evidence_source_sha256=hashlib.sha256(text.encode("ascii")).hexdigest(),
        compiler_commitment=compiler_commitment,
        step_index=step_ordinal - 1,
        target=target,
        distractor=distractor,
        value=value,
        numeric_provenance=numeric_provenance,
        symbol_provenance=symbol_provenance,
        commitment="",
    )
    return replace(provisional, commitment=canonical_sha256(provisional.payload()))


@torch.no_grad()
def compile_evidence_batch(
    model: NaturalEvidenceCompiler,
    texts: Sequence[str],
    packets: Sequence[CompiledPacket],
    indices: Sequence[int],
    *,
    compiler_commitment: str,
    device: torch.device,
    swap_numeric_roles: bool = False,
    swap_symbol_roles: bool = False,
) -> tuple[EvidenceCompilation, ...]:
    if not (len(texts) == len(packets) == len(indices)):
        raise NVE1RuntimeError("NVE1 compilation batch geometry differs")
    records = [
        {"source_text": text, "symbols": list(packet.symbols)}
        for text, packet in zip(texts, packets, strict=True)
    ]
    byte_ids, attention, numeric_bounds, symbol_masks, _, _ = tensorize_sources(
        records, device
    )
    model.eval()
    numeric_logits, symbol_logits = model(
        byte_ids, attention, numeric_bounds, symbol_masks
    )
    output = []
    for row, (text, packet, index) in enumerate(
        zip(texts, packets, indices, strict=True)
    ):
        numeric_assignment = hard_role_permutation(numeric_logits[row])
        symbol_assignment = hard_role_permutation(symbol_logits[row])
        if swap_numeric_roles:
            numeric_assignment = tuple(1 - value for value in numeric_assignment)
        if swap_symbol_roles:
            symbol_assignment = tuple(1 - value for value in symbol_assignment)
        try:
            receipt = _compile_one(
                text,
                packet,
                index,
                numeric_assignment,
                symbol_assignment,
                compiler_commitment=compiler_commitment,
            )
            output.append(EvidenceCompilation(receipt, None))
        except NVE1RuntimeError as error:
            output.append(EvidenceCompilation(None, str(error)))
    return tuple(output)


def _typed_commitment(
    source_commitment: str,
    index: int,
    step_index: int,
    register: str,
    value: str,
) -> str:
    return canonical_sha256(
        {
            "source_commitment": source_commitment,
            "index": index,
            "step_index": step_index,
            "register": register,
            "value": value,
        }
    )


def seal_natural_evidence(
    packet: CompiledPacket,
    receipts: Sequence[NaturalEvidenceReceipt],
    *,
    expected_compiler_commitment: str,
) -> tuple[dict[str, object], ...]:
    if len(receipts) > FAULT_LINES:
        raise NVE1RuntimeError("NVE1 evidence count exceeds fault lines")
    fault_steps = {
        step.fault.index: step_index
        for step_index, step in enumerate(packet.steps)
        if step.fault is not None
    }
    typed = []
    for index, receipt in enumerate(receipts):
        if receipt.index != index:
            raise NVE1RuntimeError("NVE1 evidence order differs")
        if receipt.packet_commitment != packet.commitment:
            raise NVE1RuntimeError("NVE1 packet commitment differs")
        if receipt.source_commitment != packet.source_commitment:
            raise NVE1RuntimeError("NVE1 source commitment differs")
        if receipt.compiler_commitment != expected_compiler_commitment:
            raise NVE1RuntimeError("NVE1 compiler commitment differs")
        if receipt.commitment != canonical_sha256(receipt.payload()):
            raise NVE1RuntimeError("NVE1 receipt commitment differs")
        if fault_steps.get(index) != receipt.step_index:
            raise NVE1RuntimeError("NVE1 step provenance differs")
        fault = packet.steps[receipt.step_index].fault
        assert fault is not None and fault.options[0].action is not None
        if receipt.target != fault.options[0].action.target:
            raise NVE1RuntimeError("NVE1 target differs from the fault target")
        if (
            receipt.distractor not in packet.symbols
            or receipt.distractor == receipt.target
        ):
            raise NVE1RuntimeError("NVE1 distractor provenance differs")
        try:
            value = format_fraction(parse_fraction(receipt.value))
        except TOL1IRError as error:
            raise NVE1RuntimeError("NVE1 value differs") from error
        if value != receipt.value:
            raise NVE1RuntimeError("NVE1 value is not canonical")
        if tuple(value[0] for value in receipt.numeric_provenance) != NUMERIC_ROLES:
            raise NVE1RuntimeError("NVE1 numeric provenance differs")
        if tuple(value[0] for value in receipt.symbol_provenance) != SYMBOL_ROLES:
            raise NVE1RuntimeError("NVE1 symbol provenance differs")
        if (
            receipt.symbol_provenance[0][1] != receipt.target
            or receipt.symbol_provenance[1][1] != receipt.distractor
        ):
            raise NVE1RuntimeError("NVE1 symbol provenance binding differs")
        typed.append(
            {
                "source_commitment": packet.source_commitment,
                "index": index,
                "step_index": receipt.step_index,
                "register": receipt.target,
                "value": receipt.value,
                "commitment": _typed_commitment(
                    packet.source_commitment,
                    index,
                    receipt.step_index,
                    receipt.target,
                    receipt.value,
                ),
                "natural_receipt_commitment": receipt.commitment,
                "evidence_source_sha256": receipt.evidence_source_sha256,
                "distractor": receipt.distractor,
            }
        )
    return tuple(typed)


def _rejected(reason: str) -> FactorizedReceipt:
    return FactorizedReceipt((), 0, 0, 0, 0, 0, 0, True, reason)


def execute_natural_evidence(
    packet: CompiledPacket,
    receipts: Sequence[NaturalEvidenceReceipt],
    *,
    expected_compiler_commitment: str,
    reset_after_declarations: bool = False,
    shift_fault_operations: bool = False,
) -> FactorizedReceipt:
    try:
        typed = seal_natural_evidence(
            packet,
            receipts,
            expected_compiler_commitment=expected_compiler_commitment,
        )
    except NVE1RuntimeError as error:
        return _rejected(str(error))
    try:
        return execute_factorized(
            packet,
            typed,
            reset_after_declarations=reset_after_declarations,
            shift_fault_operations=shift_fault_operations,
        )
    except TFS1RuntimeError as error:
        return _rejected(str(error))


def mutate_receipt(
    receipt: NaturalEvidenceReceipt,
    field: str,
    *,
    alternate_symbol: str | None = None,
) -> NaturalEvidenceReceipt:
    if field == "source":
        return replace(receipt, source_commitment="0" * 64)
    if field == "packet":
        return replace(receipt, packet_commitment="0" * 64)
    if field == "evidence":
        return replace(receipt, evidence_source_sha256="0" * 64)
    if field == "step":
        return replace(receipt, step_index=receipt.step_index + 1)
    if field == "target":
        return replace(receipt, target=alternate_symbol or receipt.distractor)
    if field == "distractor":
        return replace(receipt, distractor=receipt.target)
    if field == "value":
        changed = format_fraction(parse_fraction(receipt.value) + 1)
        return replace(receipt, value=changed)
    raise NVE1RuntimeError(f"unknown NVE1 receipt mutation: {field}")


__all__ = [
    "EvidenceCompilation",
    "EvidenceCompilerConfig",
    "NVE1RuntimeError",
    "NaturalEvidenceCompiler",
    "NaturalEvidenceReceipt",
    "compile_evidence_batch",
    "encode_source",
    "execute_natural_evidence",
    "hard_role_permutation",
    "module_state_sha256",
    "mutate_receipt",
    "seal_natural_evidence",
    "tensorize_sources",
]
