"""Typed fixed-slot compiler mechanics for the FSTC1 architecture."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_SLOTS = 5
MAX_SOURCE_NUMBERS = 7
MAX_NUMERATOR_DIGITS = 23
MAX_DENOMINATOR_DIGITS = 11
DIGIT_CLASSES = 11  # 0..9 plus PAD/EOS.
DIGIT_PAD = 10
SOURCE_REFERENCE = 0
STATE_REFERENCE = 1
IDENTITY_POLARITY = 0
NEGATE_POLARITY = 1
OPERATIONS = ("ADD", "SUB", "MUL", "DIV")
OP_TO_ID = {operation: index for index, operation in enumerate(OPERATIONS)}
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])(?:\d+\.\d+|\d+|\.\d+)")


class FixedSlotCompilerError(ValueError):
    """Raised when a row cannot be represented by the frozen typed schema."""


@dataclass(frozen=True, slots=True)
class NumberSpan:
    start: int
    end: int
    surface: str
    magnitude: Fraction


@dataclass(frozen=True, slots=True)
class TypedReference:
    kind: int
    index: int
    polarity: int


@dataclass(frozen=True, slots=True)
class DigitState:
    sign: int
    numerator: tuple[int, ...]
    denominator: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TypedSlot:
    operation: int
    left: TypedReference
    right: TypedReference
    result: DigitState


@dataclass(frozen=True, slots=True)
class TypedProgram:
    identity_sha256: str
    question: str
    number_spans: tuple[NumberSpan, ...]
    slots: tuple[TypedSlot, ...]


@dataclass(slots=True)
class SkeletonOutput:
    active_logits: torch.Tensor
    operation_logits: torch.Tensor
    left_reference_logits: torch.Tensor
    right_reference_logits: torch.Tensor
    left_polarity_logits: torch.Tensor
    right_polarity_logits: torch.Tensor
    slot_states: torch.Tensor


def number_spans(question: str) -> tuple[NumberSpan, ...]:
    spans = []
    for match in NUMBER_RE.finditer(question):
        surface = match.group(0)
        spans.append(NumberSpan(match.start(), match.end(), surface, Fraction(surface)))
    if len(spans) > MAX_SOURCE_NUMBERS:
        raise FixedSlotCompilerError("source numeric-span count exceeds schema")
    return tuple(spans)


def _fraction(value: dict[str, Any]) -> Fraction:
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise FixedSlotCompilerError("fraction differs")
    return Fraction(numerator, denominator)


def encode_unsigned_digits(value: int, width: int) -> tuple[int, ...]:
    if value < 0:
        raise FixedSlotCompilerError("unsigned digit value is negative")
    digits = tuple(int(character) for character in str(value))
    if len(digits) > width:
        raise FixedSlotCompilerError("digit value exceeds schema width")
    return digits + (DIGIT_PAD,) * (width - len(digits))


def decode_unsigned_digits(digits: Sequence[int]) -> int:
    values = []
    padded = False
    for digit in digits:
        if digit == DIGIT_PAD:
            padded = True
            continue
        if padded or not 0 <= digit <= 9:
            raise FixedSlotCompilerError("digit state is noncanonical")
        values.append(str(digit))
    if not values:
        raise FixedSlotCompilerError("digit state is empty")
    if len(values) > 1 and values[0] == "0":
        raise FixedSlotCompilerError("digit state has a leading zero")
    return int("".join(values))


def encode_fraction(value: Fraction) -> DigitState:
    return DigitState(
        sign=int(value < 0),
        numerator=encode_unsigned_digits(abs(value.numerator), MAX_NUMERATOR_DIGITS),
        denominator=encode_unsigned_digits(value.denominator, MAX_DENOMINATOR_DIGITS),
    )


def decode_fraction(state: DigitState) -> Fraction:
    numerator = decode_unsigned_digits(state.numerator)
    denominator = decode_unsigned_digits(state.denominator)
    if denominator == 0:
        raise FixedSlotCompilerError("digit denominator is zero")
    value = Fraction(numerator, denominator)
    return -value if state.sign and numerator else value


def compile_typed_program(row: dict[str, Any]) -> TypedProgram:
    identity = row.get("identity_sha256")
    question = row.get("question")
    records = row.get("records")
    if not isinstance(identity, str) or len(identity) != 64 or not isinstance(question, str):
        raise FixedSlotCompilerError("row identity or question differs")
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_SLOTS:
        raise FixedSlotCompilerError("record count differs")
    spans = number_spans(question)
    used_source: set[int] = set()
    prior_results: list[Fraction] = []
    slots = []
    for slot_index, record in enumerate(records):
        operation = record.get("operation")
        if operation not in OP_TO_ID:
            raise FixedSlotCompilerError("operation differs")
        dependencies = {
            dependency.get("operand_role"): dependency.get("record_index")
            for dependency in record.get("dependencies", [])
        }
        references = []
        for role, raw_value in zip(("left", "right"), record.get("operands", []), strict=True):
            value = _fraction(raw_value)
            if role in dependencies:
                dependency = dependencies[role]
                if type(dependency) is not int or not 0 <= dependency < slot_index:
                    raise FixedSlotCompilerError("dependency is not causal")
                if prior_results[dependency] != value:
                    raise FixedSlotCompilerError("dependency value differs")
                references.append(
                    TypedReference(STATE_REFERENCE, dependency, IDENTITY_POLARITY)
                )
                continue
            matches = [
                index for index, span in enumerate(spans) if span.magnitude == abs(value)
            ]
            if matches:
                unused = [index for index in matches if index not in used_source]
                chosen = unused[0] if unused else matches[0]
                used_source.add(chosen)
                references.append(
                    TypedReference(
                        SOURCE_REFERENCE,
                        chosen,
                        NEGATE_POLARITY if value < 0 else IDENTITY_POLARITY,
                    )
                )
                continue
            negated = [
                index for index, prior in enumerate(prior_results) if value == -prior
            ]
            if not negated:
                raise FixedSlotCompilerError("operand has no source or prior-state owner")
            references.append(
                TypedReference(STATE_REFERENCE, negated[-1], NEGATE_POLARITY)
            )
        if len(references) != 2:
            raise FixedSlotCompilerError("operand geometry differs")
        result = _fraction(record.get("result", {}))
        slots.append(
            TypedSlot(
                operation=OP_TO_ID[operation],
                left=references[0],
                right=references[1],
                result=encode_fraction(result),
            )
        )
        prior_results.append(result)
    return TypedProgram(identity, question, spans, tuple(slots))


def reference_class(reference: TypedReference) -> int:
    if reference.kind == SOURCE_REFERENCE:
        if not 0 <= reference.index < MAX_SOURCE_NUMBERS:
            raise FixedSlotCompilerError("source reference exceeds schema")
        return reference.index
    if reference.kind == STATE_REFERENCE:
        if not 0 <= reference.index < MAX_SLOTS:
            raise FixedSlotCompilerError("state reference exceeds schema")
        return MAX_SOURCE_NUMBERS + reference.index
    raise FixedSlotCompilerError("reference kind differs")


def skeleton_labels(programs: Sequence[TypedProgram], device: torch.device) -> dict[str, torch.Tensor]:
    batch = len(programs)
    active = torch.zeros(batch, MAX_SLOTS, dtype=torch.long, device=device)
    operation = torch.zeros(batch, MAX_SLOTS, dtype=torch.long, device=device)
    left_reference = torch.zeros(batch, MAX_SLOTS, dtype=torch.long, device=device)
    right_reference = torch.zeros(batch, MAX_SLOTS, dtype=torch.long, device=device)
    left_polarity = torch.zeros(batch, MAX_SLOTS, dtype=torch.long, device=device)
    right_polarity = torch.zeros(batch, MAX_SLOTS, dtype=torch.long, device=device)
    candidate_count = torch.tensor(
        [len(program.number_spans) for program in programs], dtype=torch.long, device=device
    )
    for row, program in enumerate(programs):
        active[row, : len(program.slots)] = 1
        for column, slot in enumerate(program.slots):
            operation[row, column] = slot.operation
            left_reference[row, column] = reference_class(slot.left)
            right_reference[row, column] = reference_class(slot.right)
            left_polarity[row, column] = slot.left.polarity
            right_polarity[row, column] = slot.right.polarity
    return {
        "active": active,
        "operation": operation,
        "left_reference": left_reference,
        "right_reference": right_reference,
        "left_polarity": left_polarity,
        "right_polarity": right_polarity,
        "candidate_count": candidate_count,
    }


class FixedSlotSkeletonCompiler(nn.Module):
    """Tied recurrent typed decoder over one separately encoded source stream."""

    def __init__(
        self,
        source_width: int,
        *,
        width: int = 512,
        encoder_layers: int = 4,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if width % heads or encoder_layers <= 0:
            raise FixedSlotCompilerError("compiler geometry differs")
        self.width = width
        self.source_projection = nn.Linear(source_width, width, bias=False)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(
            layer, encoder_layers, enable_nested_tensor=False
        )
        self.source_norm = nn.LayerNorm(width)
        self.initial_state = nn.Linear(width, width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.recurrent_cell = nn.GRUCell(width, width)
        self.active_head = nn.Linear(width, 2)
        self.operation_head = nn.Linear(width, len(OPERATIONS))
        self.reference_query = nn.ModuleList(
            [nn.Linear(width, width, bias=False), nn.Linear(width, width, bias=False)]
        )
        self.reference_source_key = nn.Linear(width, width, bias=False)
        self.reference_state_key = nn.Linear(width, width, bias=False)
        self.polarity_heads = nn.ModuleList([nn.Linear(width, 2), nn.Linear(width, 2)])
        self.operation_embedding = nn.Embedding(len(OPERATIONS), width)
        self.reference_kind_embedding = nn.Embedding(2, width)
        self.reference_index_embedding = nn.Embedding(
            MAX_SOURCE_NUMBERS + MAX_SLOTS, width
        )
        self.polarity_embedding = nn.Embedding(2, width)
        self.slot_writer = nn.Sequential(
            nn.LayerNorm(8 * width),
            nn.Linear(8 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def _candidate_states(
        self, memory: torch.Tensor, candidate_token_mask: torch.Tensor
    ) -> torch.Tensor:
        if candidate_token_mask.shape[:2] != (
            memory.shape[0],
            MAX_SOURCE_NUMBERS,
        ) or candidate_token_mask.shape[2] != memory.shape[1]:
            raise FixedSlotCompilerError("candidate token mask geometry differs")
        weights = candidate_token_mask.to(memory.dtype)
        states = torch.einsum("bcl,blh->bch", weights, memory)
        return states / weights.sum(-1, keepdim=True).clamp_min(1.0)

    def forward(
        self,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        candidate_count: torch.Tensor,
        *,
        gold: dict[str, torch.Tensor] | None = None,
        feedback: str = "hard",
        reset_recurrence: bool = False,
    ) -> SkeletonOutput:
        if feedback not in {"hard", "gold"} or (feedback == "gold" and gold is None):
            raise FixedSlotCompilerError("feedback mode differs")
        memory = self.source_projection(source_features)
        memory = self.source_encoder(memory, src_key_padding_mask=~source_mask.bool())
        memory = self.source_norm(memory)
        global_state = self._masked_mean(memory, source_mask.bool())
        state = torch.tanh(self.initial_state(global_state))
        candidates = self._candidate_states(memory, candidate_token_mask)
        slot_states: list[torch.Tensor] = []
        active_logits = []
        operation_logits = []
        left_reference_logits = []
        right_reference_logits = []
        left_polarity_logits = []
        right_polarity_logits = []
        batch = source_features.shape[0]

        for slot_index in range(MAX_SLOTS):
            if reset_recurrence and slot_index:
                state = torch.tanh(self.initial_state(global_state))
            context, _ = self.cross_attention(
                state.unsqueeze(1),
                memory,
                memory,
                key_padding_mask=~source_mask.bool(),
                need_weights=False,
            )
            state = self.recurrent_cell(context[:, 0], state)
            active = self.active_head(state)
            operation = self.operation_head(state)
            polarities = [head(state) for head in self.polarity_heads]

            if slot_states:
                prior = torch.stack(slot_states, dim=1)
            else:
                prior = memory.new_zeros(batch, 0, self.width)
            references = torch.cat((candidates, prior), dim=1)
            reference_keys = torch.cat(
                (
                    self.reference_source_key(candidates),
                    self.reference_state_key(prior),
                ),
                dim=1,
            )
            reference_scores = [
                torch.einsum("bh,brh->br", query(state), reference_keys)
                / (self.width**0.5)
                for query in self.reference_query
            ]
            valid_source = (
                torch.arange(MAX_SOURCE_NUMBERS, device=state.device)[None, :]
                < candidate_count[:, None]
            )
            valid = torch.cat(
                (
                    valid_source,
                    torch.ones(batch, slot_index, dtype=torch.bool, device=state.device),
                ),
                dim=1,
            )
            reference_scores = [score.masked_fill(~valid, -1e9) for score in reference_scores]

            if feedback == "gold":
                selected_operation = gold["operation"][:, slot_index]
                selected_references = [
                    gold["left_reference"][:, slot_index],
                    gold["right_reference"][:, slot_index],
                ]
                selected_polarities = [
                    gold["left_polarity"][:, slot_index],
                    gold["right_polarity"][:, slot_index],
                ]
            else:
                selected_operation = operation.argmax(-1)
                selected_references = [score.argmax(-1) for score in reference_scores]
                selected_polarities = [logits.argmax(-1) for logits in polarities]
            reference_kinds = [
                (selected >= MAX_SOURCE_NUMBERS).long()
                for selected in selected_references
            ]
            pieces = [state, self.operation_embedding(selected_operation)]
            for selected, kind, polarity in zip(
                selected_references,
                reference_kinds,
                selected_polarities,
                strict=True,
            ):
                pieces.extend(
                    (
                        self.reference_kind_embedding(kind),
                        self.reference_index_embedding(selected),
                        self.polarity_embedding(polarity),
                    )
                )
            # State plus seven typed field embeddings form one addressable slot.
            slot_state = self.slot_writer(torch.cat(pieces, dim=-1))
            slot_states.append(slot_state)
            state = state + slot_state

            active_logits.append(active)
            operation_logits.append(operation)
            left_reference_logits.append(reference_scores[0])
            right_reference_logits.append(reference_scores[1])
            left_polarity_logits.append(polarities[0])
            right_polarity_logits.append(polarities[1])

        def stack_and_pad(values: list[torch.Tensor]) -> torch.Tensor:
            width = MAX_SOURCE_NUMBERS + MAX_SLOTS
            padded = [F.pad(value, (0, width - value.shape[-1]), value=-1e9) for value in values]
            return torch.stack(padded, dim=1)

        return SkeletonOutput(
            active_logits=torch.stack(active_logits, dim=1),
            operation_logits=torch.stack(operation_logits, dim=1),
            left_reference_logits=stack_and_pad(left_reference_logits),
            right_reference_logits=stack_and_pad(right_reference_logits),
            left_polarity_logits=torch.stack(left_polarity_logits, dim=1),
            right_polarity_logits=torch.stack(right_polarity_logits, dim=1),
            slot_states=torch.stack(slot_states, dim=1),
        )


def skeleton_loss(
    output: SkeletonOutput, labels: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active_mask = labels["active"].bool()
    components = {
        "active": F.cross_entropy(
            output.active_logits.flatten(0, 1), labels["active"].flatten()
        )
    }
    for name, logits in (
        ("operation", output.operation_logits),
        ("left_reference", output.left_reference_logits),
        ("right_reference", output.right_reference_logits),
        ("left_polarity", output.left_polarity_logits),
        ("right_polarity", output.right_polarity_logits),
    ):
        if not active_mask.any():
            raise FixedSlotCompilerError("batch has no active slots")
        components[name] = F.cross_entropy(logits[active_mask], labels[name][active_mask])
    return sum(components.values()), components
