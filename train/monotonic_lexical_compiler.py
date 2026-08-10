"""Candidate-local monotonic lexical transducer for MLTC1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_CANDIDATES = 64
ROLES = ("IGNORE", "NUMBER", "NEGATE", "ADD", "SUB", "MUL", "DIV", "LPAREN", "RPAREN")
ROLE_TO_ID = {name: index for index, name in enumerate(ROLES)}
SURFACES = ("NUMBER", "PLUS", "MINUS", "MUL", "DIV", "LPAREN", "RPAREN")
SURFACE_TO_ID = {name: index for index, name in enumerate(SURFACES)}
ALLOWED = {
    "NUMBER": {"IGNORE", "NUMBER"},
    "PLUS": {"IGNORE", "ADD"},
    "MINUS": {"IGNORE", "NEGATE", "SUB"},
    "MUL": {"IGNORE", "MUL"},
    "DIV": {"IGNORE", "DIV"},
    "LPAREN": {"IGNORE", "LPAREN"},
    "RPAREN": {"IGNORE", "RPAREN"},
}


class MonotonicLexicalCompilerError(ValueError):
    """Raised when MLTC1 geometry or labels differ."""


@dataclass(frozen=True, slots=True)
class LexicalCandidate:
    start: int
    end: int
    surface: str
    surface_type: int
    role: int
    source_index: int


@dataclass(frozen=True, slots=True)
class LexicalProgram:
    identity_sha256: str
    family: str
    question: str
    number_spans: tuple[dict[str, Any], ...]
    gold_actions: tuple[dict[str, Any], ...]
    candidates: tuple[LexicalCandidate, ...]


@dataclass(slots=True)
class LexicalCompilerOutput:
    role_logits: torch.Tensor
    chosen_roles: torch.Tensor


def load_lexical_program(row: dict[str, Any]) -> LexicalProgram:
    if row.get("schema") != "shohin-mltc1-lexical-supervision-v1":
        raise MonotonicLexicalCompilerError("lexical schema differs")
    identity, family, question = row.get("identity_sha256"), row.get("family"), row.get("question")
    raw_candidates = row.get("candidates")
    spans, actions = row.get("number_spans"), row.get("gold_actions")
    if not isinstance(identity, str) or len(identity) != 64 or not isinstance(family, str) or not isinstance(question, str):
        raise MonotonicLexicalCompilerError("lexical identity differs")
    if not isinstance(raw_candidates, list) or not isinstance(spans, list) or not isinstance(actions, list):
        raise MonotonicLexicalCompilerError("lexical fields differ")
    if not 1 <= len(raw_candidates) <= MAX_CANDIDATES:
        raise MonotonicLexicalCompilerError("candidate count exceeds schema")
    candidates = []
    previous_end = -1
    for raw in raw_candidates:
        start, end = raw.get("start"), raw.get("end")
        surface, surface_type, role = raw.get("surface"), raw.get("surface_type"), raw.get("role")
        source_index = raw.get("source_index")
        if type(start) is not int or type(end) is not int or start < previous_end or question[start:end] != surface:
            raise MonotonicLexicalCompilerError("candidate source custody differs")
        if surface_type not in SURFACE_TO_ID or role not in ROLE_TO_ID or role not in ALLOWED[surface_type]:
            raise MonotonicLexicalCompilerError("candidate lexical role differs")
        if type(source_index) is not int or (role == "NUMBER" and not 0 <= source_index < len(spans)):
            raise MonotonicLexicalCompilerError("candidate pointer differs")
        candidates.append(
            LexicalCandidate(start, end, surface, SURFACE_TO_ID[surface_type], ROLE_TO_ID[role], source_index)
        )
        previous_end = end
    return LexicalProgram(identity, family, question, tuple(spans), tuple(actions), tuple(candidates))


def lexical_labels(programs: Sequence[LexicalProgram], device: torch.device) -> dict[str, torch.Tensor]:
    roles = torch.full((len(programs), MAX_CANDIDATES), -100, dtype=torch.long, device=device)
    surfaces = torch.zeros_like(roles)
    counts = torch.tensor([len(program.candidates) for program in programs], dtype=torch.long, device=device)
    for row, program in enumerate(programs):
        for column, candidate in enumerate(program.candidates):
            roles[row, column] = candidate.role
            surfaces[row, column] = candidate.surface_type
    return {"role": roles, "surface": surfaces, "candidate_count": counts}


class MonotonicLexicalCompiler(nn.Module):
    """Classify source-ordered lexical candidates without autoregressive decoding."""

    def __init__(self, source_width: int, *, width: int = 384, encoder_layers: int = 4, heads: int = 8) -> None:
        super().__init__()
        if width % heads or encoder_layers <= 0:
            raise MonotonicLexicalCompilerError("compiler geometry differs")
        self.width = width
        self.source_projection = nn.Linear(source_width, width, bias=False)
        self.surface_embedding = nn.Embedding(len(SURFACES), width)
        self.position_embedding = nn.Embedding(MAX_CANDIDATES, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, encoder_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.role_head = nn.Linear(width, len(ROLES))
        allowed = torch.zeros(len(SURFACES), len(ROLES), dtype=torch.bool)
        for surface, names in ALLOWED.items():
            for name in names:
                allowed[SURFACE_TO_ID[surface], ROLE_TO_ID[name]] = True
        self.register_buffer("allowed_roles", allowed, persistent=False)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        source_features: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        surface_ids: torch.Tensor,
        candidate_count: torch.Tensor,
        *,
        permute_candidate_states: bool = False,
    ) -> LexicalCompilerOutput:
        source = self.source_projection(source_features)
        weights = candidate_token_mask.to(source.dtype)
        candidate = torch.einsum("bcl,blh->bch", weights, source)
        candidate = candidate / weights.sum(-1, keepdim=True).clamp_min(1.0)
        if permute_candidate_states and candidate.shape[0] > 1:
            candidate = candidate.roll(1, 0)
        positions = torch.arange(MAX_CANDIDATES, device=source.device)[None, :]
        state = candidate + self.surface_embedding(surface_ids) + self.position_embedding(positions)
        active = positions < candidate_count[:, None]
        state = self.encoder(state, src_key_padding_mask=~active)
        logits = self.role_head(self.norm(state))
        allowed = self.allowed_roles[surface_ids]
        logits = logits.masked_fill(~allowed, -1e9)
        chosen = logits.argmax(-1)
        chosen = torch.where(active, chosen, torch.full_like(chosen, ROLE_TO_ID["IGNORE"]))
        return LexicalCompilerOutput(logits, chosen)


def lexical_loss(output: LexicalCompilerOutput, labels: dict[str, torch.Tensor]) -> torch.Tensor:
    active = labels["role"] != -100
    weights = torch.ones(len(ROLES), dtype=output.role_logits.dtype, device=output.role_logits.device)
    weights[ROLE_TO_ID["IGNORE"]] = 0.25
    return F.cross_entropy(output.role_logits[active], labels["role"][active], weight=weights)
