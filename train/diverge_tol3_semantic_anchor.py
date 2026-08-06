"""Learned position-free operation and comparator anchors for DIVERGE-TOL3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_tol1_data import source_candidates
from diverge_tol1_ir import Action, Atom, COMPARATORS, DIRECT_OPS
from diverge_tol1_product import row_clauses
from diverge_tol2_anchor_decoder import ACTION_ANCHORS, split_guard


class TOL3AnchorError(RuntimeError):
    """A local semantic anchor violates the frozen TOL3 contract."""


SCHEMA = "shohin-diverge-tol3-anchor-v1"
PAD_ID = 0
CLS_ID = 1
BYTE_OFFSET = 2
BYTE_VOCAB_SIZE = 130
MAX_TEXT_BYTES = 32

OPERATION_NAMES = ("NONE", *DIRECT_OPS, "SWAP", "QUERY")
OPERATION_TO_ID = {name: index for index, name in enumerate(OPERATION_NAMES)}
COMPARATOR_NAMES = tuple(COMPARATORS)
COMPARATOR_TO_ID = {name: index for index, name in enumerate(COMPARATOR_NAMES)}


@dataclass(frozen=True, slots=True)
class AnchorExample:
    task: str
    text: str
    label: int


@dataclass(frozen=True, slots=True)
class AnchorPrediction:
    operation: str
    text: str
    start: int
    end: int
    margin: float


@dataclass(frozen=True, slots=True)
class TOL3Config:
    width: int = 64
    layers: int = 1
    max_bytes: int = MAX_TEXT_BYTES

    def validate(self) -> None:
        if self.width <= 0 or self.width % 2 or self.layers <= 0:
            raise TOL3AnchorError("local anchor geometry differs")
        if self.max_bytes != MAX_TEXT_BYTES:
            raise TOL3AnchorError("local anchor text width differs")


def encode_text(text: str) -> tuple[int, ...]:
    try:
        payload = text.strip().lower().encode("ascii")
    except UnicodeEncodeError as error:
        raise TOL3AnchorError("local anchor must be ASCII") from error
    if not payload or len(payload) + 1 > MAX_TEXT_BYTES:
        raise TOL3AnchorError("local anchor text length differs")
    return (CLS_ID, *(value + BYTE_OFFSET for value in payload))


def _operation_examples(text: str, operation: str) -> set[AnchorExample]:
    if operation not in OPERATION_TO_ID or operation == "NONE":
        raise TOL3AnchorError("unknown local operation label")
    anchors = set(ACTION_ANCHORS[operation])
    output = set()
    positives = 0
    for candidate in source_candidates(text):
        if candidate.kind != "WORD":
            continue
        label = operation if candidate.text in anchors else "NONE"
        positives += int(label != "NONE")
        output.add(AnchorExample("operation", candidate.text, OPERATION_TO_ID[label]))
    if positives != 1:
        raise TOL3AnchorError("supervised operation anchor is missing or ambiguous")
    return output


def comparator_phrase(predicate_text: str, left: str, right: str) -> str:
    candidates = source_candidates(predicate_text)
    left_matches = [value for value in candidates if value.text == left]
    right_matches = [value for value in candidates if value.text == right]
    if len(left_matches) != 1 or len(right_matches) != 1:
        raise TOL3AnchorError("predicate operands are missing or ambiguous")
    left_span = left_matches[0]
    right_span = right_matches[0]
    if left_span.end >= right_span.start:
        raise TOL3AnchorError("predicate operand order differs")
    phrase = predicate_text[left_span.end : right_span.start].strip()
    encode_text(phrase)
    return phrase


def runtime_comparator_phrase(
    predicate_text: str,
    symbols: Sequence[str],
) -> str:
    symbol_set = set(symbols)
    candidates = tuple(
        value
        for value in source_candidates(predicate_text)
        if value.kind == "NUMBER" or value.text in symbol_set
    )
    if len(candidates) != 2 or candidates[0].kind != "WORD":
        raise TOL3AnchorError("predicate does not expose two typed operands")
    if candidates[0].end >= candidates[1].start:
        raise TOL3AnchorError("predicate operand order differs")
    phrase = predicate_text[candidates[0].end : candidates[1].start].strip()
    encode_text(phrase)
    return phrase


def select_operation_anchor(
    text: str,
    logits_by_word: Mapping[str, Sequence[float]],
) -> AnchorPrediction:
    predictions = []
    for candidate in source_candidates(text):
        if candidate.kind != "WORD":
            continue
        logits = tuple(float(value) for value in logits_by_word[candidate.text])
        if len(logits) != len(OPERATION_NAMES):
            raise TOL3AnchorError("operation logit width differs")
        label = max(range(1, len(logits)), key=logits.__getitem__)
        predictions.append(
            AnchorPrediction(
                OPERATION_NAMES[label],
                candidate.text,
                candidate.start,
                candidate.end,
                logits[label] - logits[OPERATION_TO_ID["NONE"]],
            )
        )
    if not predictions:
        raise TOL3AnchorError("action has no source words")
    prediction = max(predictions, key=lambda value: (value.margin, -value.start))
    if prediction.margin <= 0.0:
        raise TOL3AnchorError("action has no positive model-owned anchor")
    return prediction


def _relation_position(text: str, word: str) -> tuple[int, int] | None:
    matches = list(re.finditer(rf"\b{re.escape(word)}\b", text))
    if not matches:
        return None
    if len(matches) != 1:
        raise TOL3AnchorError("argument relation is ambiguous")
    return matches[0].start(), matches[0].end()


def decode_direct_action_from_anchor(
    text: str,
    operation: str,
    symbols: Sequence[str] | None,
    anchor_end: int,
) -> Action:
    """Decode arguments without consulting the supervisor operation lexicon."""

    if operation not in DIRECT_OPS:
        raise TOL3AnchorError("direct action opcode differs")
    symbol_set = set(symbols or ())
    candidates = tuple(
        value
        for value in source_candidates(text)
        if symbols is None or value.kind == "NUMBER" or value.text in symbol_set
    )
    relation_word = {
        "SET": "into",
        "ADD": "to",
        "SUBTRACT": "from",
        "MULTIPLY": None,
    }[operation]
    relation = _relation_position(text, relation_word) if relation_word else None
    position = relation[1] if relation is not None else anchor_end
    targets = [
        value
        for value in candidates
        if value.kind == "WORD" and value.start >= position
    ]
    if not targets:
        raise TOL3AnchorError("typed relation has no following target")
    target = min(targets, key=lambda value: (value.start, value.end))
    remaining = [value for value in candidates if value is not target]
    if symbols is None:
        numeric = [value for value in remaining if value.kind == "NUMBER"]
        if len(numeric) != 1:
            raise TOL3AnchorError("declaration does not expose one constant")
        operand = numeric[0]
    else:
        if len(remaining) != 1:
            raise TOL3AnchorError("direct action does not expose two typed arguments")
        operand = remaining[0]
    action = Action(
        operation,
        target.text,
        Atom("CONST" if operand.kind == "NUMBER" else "REF", operand.text),
    )
    action.validate()
    return action


def build_anchor_examples(rows: Sequence[dict[str, object]]) -> tuple[AnchorExample, ...]:
    output: set[AnchorExample] = set()
    for row in rows:
        for clause in row_clauses(row):
            instruction = clause.instruction
            if instruction.operation in OPERATION_TO_ID and instruction.operation != "NONE":
                output.update(_operation_examples(clause.text, instruction.operation))
            elif instruction.operation == "GUARD":
                assert instruction.predicate and instruction.true_action and instruction.false_action
                regions = split_guard(clause.text)
                output.update(
                    _operation_examples(
                        regions.true_action, instruction.true_action.operation
                    )
                )
                output.update(
                    _operation_examples(
                        regions.false_action, instruction.false_action.operation
                    )
                )
                phrase = comparator_phrase(
                    regions.predicate,
                    instruction.predicate.left,
                    instruction.predicate.right.value,
                )
                output.add(
                    AnchorExample(
                        "comparator",
                        phrase,
                        COMPARATOR_TO_ID[instruction.predicate.comparator],
                    )
                )
            else:
                raise TOL3AnchorError("unknown clause in anchor supervision")
    if not output:
        raise TOL3AnchorError("empty local anchor supervision")
    covered_operations = {
        value.label for value in output if value.task == "operation"
    }
    covered_comparators = {
        value.label for value in output if value.task == "comparator"
    }
    if covered_operations != set(range(len(OPERATION_NAMES))):
        raise TOL3AnchorError("local operation supervision lacks a class")
    if covered_comparators != set(range(len(COMPARATOR_NAMES))):
        raise TOL3AnchorError("local comparator supervision lacks a class")
    return tuple(sorted(output, key=lambda value: (value.task, value.label, value.text)))


def tensorize_texts(texts: Sequence[str], device: torch.device):
    encoded = [encode_text(value) for value in texts]
    ids = torch.full(
        (len(encoded), MAX_TEXT_BYTES), PAD_ID, dtype=torch.long, device=device
    )
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, values in enumerate(encoded):
        ids[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[row, : len(values)] = True
    return ids, mask


class LocalSemanticAnchor(nn.Module):
    def __init__(self, config: TOL3Config) -> None:
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
        self.norm = nn.LayerNorm(config.width)
        self.operation_head = nn.Linear(config.width, len(OPERATION_NAMES))
        self.comparator_head = nn.Linear(config.width, len(COMPARATOR_NAMES))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[1] != self.config.max_bytes:
            raise TOL3AnchorError("local anchor tensor interface differs")
        lengths = mask.bool().sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(ids[:, 0].eq(CLS_ID)):
            raise TOL3AnchorError("local anchor mask or CLS differs")
        packed = pack_padded_sequence(
            self.embedding(ids), lengths.detach().cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=self.config.max_bytes
        )
        hidden = self.norm(hidden)
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.operation_head(pooled).float(), self.comparator_head(pooled).float()

    def record(self) -> dict[str, object]:
        return {"schema": SCHEMA, "config": asdict(self.config)}


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("ascii"))
        digest.update(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


__all__ = [
    "AnchorExample",
    "AnchorPrediction",
    "COMPARATOR_NAMES",
    "COMPARATOR_TO_ID",
    "LocalSemanticAnchor",
    "OPERATION_NAMES",
    "OPERATION_TO_ID",
    "SCHEMA",
    "TOL3AnchorError",
    "TOL3Config",
    "build_anchor_examples",
    "comparator_phrase",
    "decode_direct_action_from_anchor",
    "encode_text",
    "module_state_sha256",
    "runtime_comparator_phrase",
    "select_operation_anchor",
    "tensorize_texts",
]
