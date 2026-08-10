"""Causal typed slot decoder for TMC1 natural microcode graphs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from typed_microcode_graph import (
    LITERAL,
    OPERATIONS,
    SOURCE,
    STATE,
    Instruction,
    Operand,
    TypedMicrocodeGraph,
    TypedMicrocodeGraphError,
    number_spans,
)

MAX_STEPS = 15
MAX_SOURCE_SPANS = 15
LITERAL_NUMERATOR_DIGITS = 10
LITERAL_DENOMINATOR_DIGITS = 3
DIGIT_PAD = 10
DIGIT_CLASSES = 11
LITERAL_REFERENCE = MAX_SOURCE_SPANS + MAX_STEPS
REFERENCE_CLASSES = LITERAL_REFERENCE + 1
OP_TO_ID = {operation: index for index, operation in enumerate(OPERATIONS)}


@dataclass(slots=True)
class TypedCompilerOutput:
    length_logits: torch.Tensor
    operation_logits: torch.Tensor
    left_reference_logits: torch.Tensor
    right_reference_logits: torch.Tensor
    literal_sign_logits: torch.Tensor
    literal_numerator_logits: torch.Tensor
    literal_denominator_logits: torch.Tensor


def encode_digits(value: int, width: int) -> tuple[int, ...]:
    if value < 0:
        raise TypedMicrocodeGraphError("unsigned literal differs")
    digits = tuple(int(character) for character in str(value))
    if len(digits) > width:
        raise TypedMicrocodeGraphError("literal exceeds digit schema")
    return digits + (DIGIT_PAD,) * (width - len(digits))


def decode_digits(values: Sequence[int], *, denominator: bool = False) -> int:
    digits = []
    for value in values:
        if value == DIGIT_PAD:
            break
        if not 0 <= value <= 9:
            raise TypedMicrocodeGraphError("literal digit differs")
        digits.append(str(value))
    if not digits or (len(digits) > 1 and digits[0] == "0"):
        raise TypedMicrocodeGraphError("literal digit sequence is noncanonical")
    decoded = int("".join(digits))
    if denominator and decoded == 0:
        raise TypedMicrocodeGraphError("literal denominator is zero")
    return decoded


def _reference_targets(operand: Operand) -> tuple[int, ...]:
    if operand.kind == SOURCE:
        if not operand.indices or any(
            not 0 <= index < MAX_SOURCE_SPANS for index in operand.indices
        ):
            raise TypedMicrocodeGraphError("source reference exceeds schema")
        return operand.indices
    if operand.kind == STATE:
        if len(operand.indices) != 1 or not 0 <= operand.indices[0] < MAX_STEPS:
            raise TypedMicrocodeGraphError("state reference exceeds schema")
        return (MAX_SOURCE_SPANS + operand.indices[0],)
    if operand.kind == LITERAL and operand.literal is not None:
        return (LITERAL_REFERENCE,)
    raise TypedMicrocodeGraphError("reference differs")


def graph_labels(
    graphs: Sequence[TypedMicrocodeGraph], device: torch.device
) -> dict[str, torch.Tensor]:
    batch = len(graphs)
    length = torch.empty(batch, dtype=torch.long, device=device)
    operation = torch.zeros(batch, MAX_STEPS, dtype=torch.long, device=device)
    active = torch.zeros(batch, MAX_STEPS, dtype=torch.bool, device=device)
    right_present = torch.zeros_like(active)
    reference_targets = torch.zeros(
        batch, MAX_STEPS, 2, REFERENCE_CLASSES, dtype=torch.bool, device=device
    )
    literal_mask = torch.zeros(batch, MAX_STEPS, 2, dtype=torch.bool, device=device)
    literal_sign = torch.zeros(batch, MAX_STEPS, 2, dtype=torch.long, device=device)
    literal_numerator = torch.full(
        (batch, MAX_STEPS, 2, LITERAL_NUMERATOR_DIGITS),
        DIGIT_PAD,
        dtype=torch.long,
        device=device,
    )
    literal_denominator = torch.full(
        (batch, MAX_STEPS, 2, LITERAL_DENOMINATOR_DIGITS),
        DIGIT_PAD,
        dtype=torch.long,
        device=device,
    )
    source_count = torch.empty(batch, dtype=torch.long, device=device)
    for row, graph in enumerate(graphs):
        if not 1 <= len(graph.instructions) <= MAX_STEPS:
            raise TypedMicrocodeGraphError("instruction count exceeds schema")
        if len(graph.number_spans) > MAX_SOURCE_SPANS:
            raise TypedMicrocodeGraphError("source span count exceeds schema")
        length[row] = len(graph.instructions) - 1
        source_count[row] = len(graph.number_spans)
        active[row, : len(graph.instructions)] = True
        for step, instruction in enumerate(graph.instructions):
            operation[row, step] = OP_TO_ID[instruction.operation]
            operands = (instruction.left, instruction.right)
            for side, operand in enumerate(operands):
                if operand is None:
                    continue
                if side:
                    right_present[row, step] = True
                reference_targets[
                    row, step, side, list(_reference_targets(operand))
                ] = True
                if operand.kind != LITERAL:
                    continue
                assert operand.literal is not None
                literal_mask[row, step, side] = True
                literal_sign[row, step, side] = int(operand.literal < 0)
                literal_numerator[row, step, side] = torch.tensor(
                    encode_digits(
                        abs(operand.literal.numerator), LITERAL_NUMERATOR_DIGITS
                    ),
                    device=device,
                )
                literal_denominator[row, step, side] = torch.tensor(
                    encode_digits(
                        operand.literal.denominator, LITERAL_DENOMINATOR_DIGITS
                    ),
                    device=device,
                )
    return {
        "length": length,
        "operation": operation,
        "active": active,
        "right_present": right_present,
        "reference_targets": reference_targets,
        "literal_mask": literal_mask,
        "literal_sign": literal_sign,
        "literal_numerator": literal_numerator,
        "literal_denominator": literal_denominator,
        "source_count": source_count,
    }


class TypedMicrocodeCompiler(nn.Module):
    """One source encoding followed by causal typed instruction slots."""

    def __init__(
        self,
        source_width: int,
        *,
        width: int = 512,
        source_layers: int = 2,
        decoder_layers: int = 4,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if width % heads or source_layers <= 0 or decoder_layers <= 0:
            raise TypedMicrocodeGraphError("compiler geometry differs")
        self.width = width
        self.source_projection = nn.Linear(source_width, width, bias=False)
        source_layer = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(
            source_layer, source_layers, enable_nested_tensor=False
        )
        decoder_layer = nn.TransformerDecoderLayer(
            width,
            heads,
            4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.source_norm = nn.LayerNorm(width)
        self.slot_norm = nn.LayerNorm(width)
        self.slot_queries = nn.Parameter(torch.empty(MAX_STEPS, width))
        nn.init.normal_(self.slot_queries, std=0.02)
        self.length_head = nn.Linear(width, MAX_STEPS)
        self.operation_head = nn.Linear(width, len(OPERATIONS))
        self.reference_queries = nn.ModuleList(
            [nn.Linear(width, width, bias=False), nn.Linear(width, width, bias=False)]
        )
        self.source_reference_key = nn.Linear(width, width, bias=False)
        self.state_reference_key = nn.Linear(width, width, bias=False)
        self.literal_reference_keys = nn.Parameter(torch.empty(2, width))
        nn.init.normal_(self.literal_reference_keys, std=0.02)
        self.literal_sign_head = nn.Linear(width, 2 * 2)
        self.literal_numerator_head = nn.Linear(
            width, 2 * LITERAL_NUMERATOR_DIGITS * DIGIT_CLASSES
        )
        self.literal_denominator_head = nn.Linear(
            width, 2 * LITERAL_DENOMINATOR_DIGITS * DIGIT_CLASSES
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _candidate_states(
        source: torch.Tensor, candidate_mask: torch.Tensor
    ) -> torch.Tensor:
        weights = candidate_mask.to(source.dtype)
        return torch.einsum("bct,btw->bcw", weights, source) / weights.sum(
            -1, keepdim=True
        ).clamp_min(1.0)

    def forward(
        self,
        source_states: torch.Tensor,
        source_attention: torch.Tensor,
        candidate_mask: torch.Tensor,
        source_count: torch.Tensor,
    ) -> TypedCompilerOutput:
        batch, tokens, _ = source_states.shape
        if (
            source_attention.shape != (batch, tokens)
            or candidate_mask.shape != (batch, MAX_SOURCE_SPANS, tokens)
            or source_count.shape != (batch,)
        ):
            raise TypedMicrocodeGraphError("compiler input geometry differs")
        source = self.source_norm(self.source_projection(source_states))
        source = self.source_encoder(
            source, src_key_padding_mask=~source_attention.bool()
        )
        slots = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        causal_mask = torch.triu(
            torch.ones(MAX_STEPS, MAX_STEPS, dtype=torch.bool, device=source.device),
            diagonal=1,
        )
        slots = self.slot_norm(
            self.decoder(
                slots,
                source,
                tgt_mask=causal_mask,
                memory_key_padding_mask=~source_attention.bool(),
            )
        )
        candidate_states = self._candidate_states(source, candidate_mask)
        source_keys = self.source_reference_key(candidate_states)
        state_keys = self.state_reference_key(slots)
        reference_logits = []
        source_indices = torch.arange(MAX_SOURCE_SPANS, device=source.device)
        source_valid = source_indices.unsqueeze(0) < source_count.unsqueeze(1)
        state_indices = torch.arange(MAX_STEPS, device=source.device)
        state_valid = state_indices.unsqueeze(0) < state_indices.unsqueeze(1)
        for side, query_layer in enumerate(self.reference_queries):
            query = query_layer(slots)
            source_scores = torch.einsum("bsw,bcw->bsc", query, source_keys)
            state_scores = torch.einsum("bsw,btw->bst", query, state_keys)
            literal_scores = torch.einsum(
                "bsw,w->bs", query, self.literal_reference_keys[side]
            ).unsqueeze(-1)
            logits = torch.cat((source_scores, state_scores, literal_scores), -1)
            valid = torch.cat(
                (
                    source_valid[:, None, :].expand(-1, MAX_STEPS, -1),
                    state_valid[None, :, :].expand(batch, -1, -1),
                    torch.ones(
                        batch, MAX_STEPS, 1, dtype=torch.bool, device=source.device
                    ),
                ),
                -1,
            )
            reference_logits.append(
                logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
                / math.sqrt(self.width)
            )
        numerator = self.literal_numerator_head(slots).view(
            batch,
            MAX_STEPS,
            2,
            LITERAL_NUMERATOR_DIGITS,
            DIGIT_CLASSES,
        )
        denominator = self.literal_denominator_head(slots).view(
            batch,
            MAX_STEPS,
            2,
            LITERAL_DENOMINATOR_DIGITS,
            DIGIT_CLASSES,
        )
        numerator[..., 0, DIGIT_PAD] = torch.finfo(numerator.dtype).min
        denominator[..., 0, 0] = torch.finfo(denominator.dtype).min
        denominator[..., 0, DIGIT_PAD] = torch.finfo(denominator.dtype).min
        return TypedCompilerOutput(
            length_logits=self.length_head(slots[:, 0]),
            operation_logits=self.operation_head(slots),
            left_reference_logits=reference_logits[0],
            right_reference_logits=reference_logits[1],
            literal_sign_logits=self.literal_sign_head(slots).view(
                batch, MAX_STEPS, 2, 2
            ),
            literal_numerator_logits=numerator,
            literal_denominator_logits=denominator,
        )


def _set_target_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probabilities = logits.log_softmax(-1)
    selected = log_probabilities.masked_fill(~targets, float("-inf"))
    return -torch.logsumexp(selected, -1)


def typed_compiler_loss(
    output: TypedCompilerOutput, labels: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active = labels["active"]
    right = active & labels["right_present"]
    literal = labels["literal_mask"]
    components = {
        "length": F.cross_entropy(output.length_logits, labels["length"]),
        "operation": F.cross_entropy(
            output.operation_logits[active], labels["operation"][active]
        ),
        "left_reference": _set_target_loss(
            output.left_reference_logits[active],
            labels["reference_targets"][:, :, 0][active],
        ).mean(),
        "right_reference": _set_target_loss(
            output.right_reference_logits[right],
            labels["reference_targets"][:, :, 1][right],
        ).mean(),
    }
    if literal.any():
        sign = F.cross_entropy(
            output.literal_sign_logits[literal], labels["literal_sign"][literal]
        )
        numerator = F.cross_entropy(
            output.literal_numerator_logits[literal].flatten(0, 1),
            labels["literal_numerator"][literal].flatten(),
        )
        denominator = F.cross_entropy(
            output.literal_denominator_logits[literal].flatten(0, 1),
            labels["literal_denominator"][literal].flatten(),
        )
        components["literal"] = (sign + numerator + denominator) / 3
    else:
        components["literal"] = output.length_logits.sum() * 0
    return sum(components.values()), components


def _decode_literal(
    output: TypedCompilerOutput, row: int, step: int, side: int
) -> Fraction:
    sign = int(output.literal_sign_logits[row, step, side].argmax())
    numerator = decode_digits(
        output.literal_numerator_logits[row, step, side].argmax(-1).tolist()
    )
    denominator = decode_digits(
        output.literal_denominator_logits[row, step, side].argmax(-1).tolist(),
        denominator=True,
    )
    value = Fraction(numerator, denominator)
    return -value if sign and numerator else value


def _decode_operand(
    output: TypedCompilerOutput,
    row: int,
    step: int,
    side: int,
    spans: Sequence,
) -> Operand:
    logits = (
        output.left_reference_logits if side == 0 else output.right_reference_logits
    )
    reference = int(logits[row, step].argmax())
    if reference < MAX_SOURCE_SPANS:
        if reference >= len(spans):
            raise TypedMicrocodeGraphError("decoded source reference differs")
        return Operand(SOURCE, (reference,))
    if reference < LITERAL_REFERENCE:
        state = reference - MAX_SOURCE_SPANS
        if state >= step:
            raise TypedMicrocodeGraphError("decoded state reference is not causal")
        return Operand(STATE, (state,))
    return Operand(LITERAL, literal=_decode_literal(output, row, step, side))


def decode_graphs(
    output: TypedCompilerOutput, sources: Sequence[str]
) -> list[TypedMicrocodeGraph]:
    graphs = []
    for row, source in enumerate(sources):
        spans = number_spans(source)
        count = int(output.length_logits[row].argmax()) + 1
        instructions = []
        for step in range(count):
            operation = OPERATIONS[int(output.operation_logits[row, step].argmax())]
            left = _decode_operand(output, row, step, 0, spans)
            right = (
                None
                if operation in {"NEG", "COPY"}
                else _decode_operand(output, row, step, 1, spans)
            )
            instructions.append(Instruction(operation, left, right))
        graphs.append(
            TypedMicrocodeGraph(
                source,
                tuple(spans),
                tuple(instructions),
                Operand(STATE, (count - 1,)),
            )
        )
    return graphs
