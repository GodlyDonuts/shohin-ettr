"""Finite-state span compiler and typed decoder for DIVERGE-TOL1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_tol1_data import (
    ACTION_NAMES,
    BYTE_VOCAB_SIZE,
    CLAUSE_OPS,
    CLS_ID,
    COMPARATOR_NAMES,
    MAX_CLAUSE_BYTES,
    ROLE_NAMES,
    SourceCandidate,
)
from diverge_tol1_ir import (
    Action,
    Atom,
    DIRECT_OPS,
    Instruction,
    Predicate,
    TOL1IRError,
)


class TOL1RuntimeError(RuntimeError):
    """The TOL1 neural or structured decoder contract was violated."""


SCHEMA = "shohin-diverge-tol1-runtime-v1"


@dataclass(frozen=True, slots=True)
class TOL1Config:
    width: int = 192
    layers: int = 2
    max_bytes: int = MAX_CLAUSE_BYTES

    def validate(self) -> None:
        if self.width <= 0 or self.width % 2 or self.layers <= 0:
            raise TOL1RuntimeError("invalid finite-state compiler geometry")
        if self.max_bytes != MAX_CLAUSE_BYTES:
            raise TOL1RuntimeError("TOL1 clause width differs")


class TypedOperationCompiler(nn.Module):
    """Encode one clause and score its typed source spans."""

    def __init__(self, config: TOL1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.byte_embedding = nn.Embedding(BYTE_VOCAB_SIZE, config.width)
        self.encoder = nn.GRU(
            input_size=config.width,
            hidden_size=config.width // 2,
            num_layers=config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.span_projection = nn.Sequential(
            nn.LayerNorm(3 * config.width),
            nn.Linear(3 * config.width, config.width),
            nn.GELU(),
        )
        self.role_head = nn.Linear(config.width, len(ROLE_NAMES))
        self.clause_projection = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
        )
        self.operation_head = nn.Linear(config.width, len(CLAUSE_OPS))
        self.comparator_head = nn.Linear(config.width, len(COMPARATOR_NAMES))
        self.true_action_head = nn.Linear(config.width, len(ACTION_NAMES))
        self.false_action_head = nn.Linear(config.width, len(ACTION_NAMES))

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_batch: torch.Tensor,
        candidate_start: torch.Tensor,
        candidate_end: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or byte_ids.shape[1] != self.config.max_bytes
            or byte_ids.dtype != torch.long
        ):
            raise TOL1RuntimeError("TOL1 compiler tensor interface differs")
        if not (
            candidate_batch.ndim
            == candidate_start.ndim
            == candidate_end.ndim
            == 1
            and len(candidate_batch) == len(candidate_start) == len(candidate_end)
        ):
            raise TOL1RuntimeError("TOL1 candidate tensor interface differs")
        active = attention_mask.bool()
        lengths = active.sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise TOL1RuntimeError("TOL1 source mask or CLS differs")
        if len(candidate_batch):
            if (
                torch.any(candidate_batch < 0)
                or torch.any(candidate_batch >= len(byte_ids))
                or torch.any(candidate_start < 0)
                or torch.any(candidate_end <= candidate_start)
                or torch.any(candidate_end + 1 >= lengths[candidate_batch] + 1)
            ):
                raise TOL1RuntimeError("TOL1 candidate span escaped source")
        embedded = self.byte_embedding(byte_ids)
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
        if len(candidate_batch):
            starts = candidate_start + 1
            ends = candidate_end
            prefix = torch.cat(
                (
                    torch.zeros(
                        hidden.shape[0], 1, hidden.shape[2],
                        dtype=hidden.dtype, device=hidden.device,
                    ),
                    hidden.cumsum(dim=1),
                ),
                dim=1,
            )
            means = (
                prefix[candidate_batch, ends + 1]
                - prefix[candidate_batch, starts]
            ) / (ends - starts + 1).to(hidden.dtype).unsqueeze(-1)
            span_hidden = self.span_projection(
                torch.cat(
                    (
                        hidden[candidate_batch, starts],
                        hidden[candidate_batch, ends],
                        means,
                    ),
                    dim=-1,
                )
            )
            role_logits = self.role_head(span_hidden).float()
        else:
            role_logits = hidden.new_zeros((0, len(ROLE_NAMES)), dtype=torch.float32)
        weights = active.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        clause = self.clause_projection(pooled)
        return (
            role_logits,
            self.operation_head(clause).float(),
            self.comparator_head(clause).float(),
            self.true_action_head(clause).float(),
            self.false_action_head(clause).float(),
        )

    def initialize_fta1_encoder(self, state: dict[str, torch.Tensor]) -> tuple[str, ...]:
        own = self.state_dict()
        prefixes = ("byte_embedding.", "encoder.", "output_norm.")
        loaded = []
        for name, value in state.items():
            if name.startswith(prefixes) and name in own and own[name].shape == value.shape:
                own[name].copy_(value)
                loaded.append(name)
        expected = [name for name in own if name.startswith(prefixes)]
        if set(loaded) != set(expected):
            raise TOL1RuntimeError("FTA1 encoder warm start is incomplete")
        return tuple(sorted(loaded))

    def record(self) -> dict[str, object]:
        return {"schema": SCHEMA, "config": asdict(self.config)}


_REQUIRED_ROLES = {
    "SET": ("TARGET", "OPERAND"),
    "ADD": ("TARGET", "OPERAND"),
    "SUBTRACT": ("TARGET", "OPERAND"),
    "MULTIPLY": ("TARGET", "OPERAND"),
    "SWAP": ("TARGET", "OPERAND"),
    "GUARD": (
        "PRED_LEFT", "PRED_RIGHT", "TRUE_TARGET", "TRUE_OPERAND",
        "FALSE_TARGET", "FALSE_OPERAND",
    ),
    "QUERY": ("QUERY_REF",),
}


def _compatible(role: str, candidate: SourceCandidate) -> bool:
    if role in {"TARGET", "PRED_LEFT", "TRUE_TARGET", "FALSE_TARGET", "QUERY_REF"}:
        return candidate.kind == "WORD"
    return candidate.kind in {"WORD", "NUMBER"}


def _structured_assignment(
    candidates: Sequence[SourceCandidate],
    role_logits: torch.Tensor,
    roles: Sequence[str],
) -> dict[str, SourceCandidate]:
    if tuple(role_logits.shape) != (len(candidates), len(ROLE_NAMES)):
        raise TOL1RuntimeError("role-logit shape differs")
    none = ROLE_NAMES.index("NONE")
    role_ids = [ROLE_NAMES.index(role) for role in roles]
    # Candidate-by-candidate dynamic program gives the exact maximum-weight
    # injective role assignment in O(candidates * roles * 2^roles).
    frontier: dict[int, tuple[float, tuple[int, ...]]] = {
        0: (0.0, (-1,) * len(roles))
    }
    for candidate_index, candidate in enumerate(candidates):
        updated = dict(frontier)
        for mask, (score, assignment) in frontier.items():
            for role_offset, (role, role_id) in enumerate(zip(roles, role_ids, strict=True)):
                bit = 1 << role_offset
                if mask & bit or not _compatible(role, candidate):
                    continue
                margin = float(role_logits[candidate_index, role_id] - role_logits[candidate_index, none])
                assigned = list(assignment)
                assigned[role_offset] = candidate_index
                proposal = (score + margin, tuple(assigned))
                new_mask = mask | bit
                incumbent = updated.get(new_mask)
                if incumbent is None or proposal[0] > incumbent[0]:
                    updated[new_mask] = proposal
        frontier = updated
    full = (1 << len(roles)) - 1
    if full not in frontier:
        raise TOL1RuntimeError("no legal typed role assignment")
    indices = frontier[full][1]
    if len(indices) != len(roles) or any(index < 0 for index in indices):
        raise TOL1RuntimeError("typed assignment reconstruction differs")
    return {role: candidates[index] for role, index in zip(roles, indices, strict=True)}


def _raw_assignment(
    candidates: Sequence[SourceCandidate],
    role_logits: torch.Tensor,
    roles: Sequence[str],
) -> dict[str, SourceCandidate]:
    predicted = role_logits.argmax(dim=-1).tolist()
    output = {}
    for role in roles:
        role_id = ROLE_NAMES.index(role)
        matches = [
            candidate for candidate, prediction in zip(candidates, predicted, strict=True)
            if prediction == role_id and _compatible(role, candidate)
        ]
        if len(matches) != 1:
            raise TOL1RuntimeError("independent role argmax is not a legal packet")
        output[role] = matches[0]
    return output


def _atom_from_candidate(candidate: SourceCandidate) -> Atom:
    return Atom("CONST" if candidate.kind == "NUMBER" else "REF", candidate.text)


def decode_instruction(
    candidates: Sequence[SourceCandidate],
    role_logits: torch.Tensor,
    operation_logits: torch.Tensor,
    comparator_logits: torch.Tensor,
    true_action_logits: torch.Tensor,
    false_action_logits: torch.Tensor,
    *,
    structured: bool,
) -> Instruction:
    if operation_logits.shape != (len(CLAUSE_OPS),):
        raise TOL1RuntimeError("operation-logit shape differs")
    operation = CLAUSE_OPS[int(operation_logits.argmax())]
    roles = _REQUIRED_ROLES[operation]
    assignment = (
        _structured_assignment(candidates, role_logits, roles)
        if structured
        else _raw_assignment(candidates, role_logits, roles)
    )
    if operation in DIRECT_OPS:
        instruction = Instruction(
            operation,
            action=Action(
                operation,
                assignment["TARGET"].text,
                _atom_from_candidate(assignment["OPERAND"]),
            ),
        )
    elif operation == "SWAP":
        right = assignment["OPERAND"]
        if right.kind != "WORD":
            raise TOL1RuntimeError("swap operand is not a register")
        instruction = Instruction(
            operation,
            swap_left=assignment["TARGET"].text,
            swap_right=right.text,
        )
    elif operation == "GUARD":
        comparator = COMPARATOR_NAMES[int(comparator_logits.argmax())]
        true_operation = ACTION_NAMES[int(true_action_logits.argmax())]
        false_operation = ACTION_NAMES[int(false_action_logits.argmax())]
        if "NONE" in {comparator, true_operation, false_operation}:
            raise TOL1RuntimeError("guard subtype is missing")
        instruction = Instruction(
            operation,
            predicate=Predicate(
                comparator,
                assignment["PRED_LEFT"].text,
                _atom_from_candidate(assignment["PRED_RIGHT"]),
            ),
            true_action=Action(
                true_operation,
                assignment["TRUE_TARGET"].text,
                _atom_from_candidate(assignment["TRUE_OPERAND"]),
            ),
            false_action=Action(
                false_operation,
                assignment["FALSE_TARGET"].text,
                _atom_from_candidate(assignment["FALSE_OPERAND"]),
            ),
        )
    else:
        instruction = Instruction(operation, query=assignment["QUERY_REF"].text)
    try:
        instruction.validate()
    except TOL1IRError as error:
        raise TOL1RuntimeError("decoded instruction is invalid") from error
    return instruction


__all__ = [
    "SCHEMA",
    "TOL1Config",
    "TOL1RuntimeError",
    "TypedOperationCompiler",
    "decode_instruction",
]
