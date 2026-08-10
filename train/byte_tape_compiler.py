"""Raw byte-tape classifier for BTT1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_BYTES = 512
PAD_BYTE = 256
ROLES = ("IGNORE", "NUM_BEGIN", "NUM_CONT", "NEGATE", "ADD", "SUB", "MUL", "DIV", "LPAREN", "RPAREN")
ROLE_TO_ID = {name: index for index, name in enumerate(ROLES)}


class ByteTapeCompilerError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ByteProgram:
    identity_sha256: str
    family: str
    question: str
    byte_roles: tuple[int, ...]
    gold_actions: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class ByteCompilerOutput:
    role_logits: torch.Tensor
    chosen_roles: torch.Tensor


def load_byte_program(row: dict[str, Any]) -> ByteProgram:
    if row.get("schema") != "shohin-btt1-byte-supervision-v1":
        raise ByteTapeCompilerError("byte schema differs")
    identity, family, question = row.get("identity_sha256"), row.get("family"), row.get("question")
    raw_roles, actions = row.get("byte_roles"), row.get("gold_actions")
    if not isinstance(identity, str) or len(identity) != 64 or not isinstance(family, str) or not isinstance(question, str):
        raise ByteTapeCompilerError("byte identity differs")
    if not isinstance(raw_roles, list) or len(raw_roles) != len(question) or not 1 <= len(raw_roles) <= MAX_BYTES:
        raise ByteTapeCompilerError("byte-role geometry differs")
    if not isinstance(actions, list):
        raise ByteTapeCompilerError("gold actions differ")
    try:
        question.encode("ascii")
    except UnicodeEncodeError as error:
        raise ByteTapeCompilerError("non-ASCII source differs") from error
    if any(role not in ROLE_TO_ID for role in raw_roles):
        raise ByteTapeCompilerError("byte role differs")
    return ByteProgram(identity, family, question, tuple(ROLE_TO_ID[role] for role in raw_roles), tuple(actions))


def byte_batch(programs: Sequence[ByteProgram], device: torch.device) -> dict[str, torch.Tensor]:
    maximum = max(len(program.question) for program in programs)
    byte_ids = torch.full((len(programs), maximum), PAD_BYTE, dtype=torch.long, device=device)
    roles = torch.full_like(byte_ids, -100)
    mask = torch.zeros_like(byte_ids, dtype=torch.bool)
    for row, program in enumerate(programs):
        values = torch.tensor(list(program.question.encode("ascii")), dtype=torch.long, device=device)
        byte_ids[row, : len(values)] = values
        roles[row, : len(values)] = torch.tensor(program.byte_roles, dtype=torch.long, device=device)
        mask[row, : len(values)] = True
    return {"byte_ids": byte_ids, "role": roles, "mask": mask}


class ByteTapeCompiler(nn.Module):
    def __init__(self, *, width: int = 256, encoder_layers: int = 6, heads: int = 8) -> None:
        super().__init__()
        if width % heads or encoder_layers <= 0:
            raise ByteTapeCompilerError("compiler geometry differs")
        self.byte_embedding = nn.Embedding(PAD_BYTE + 1, width, padding_idx=PAD_BYTE)
        self.position_embedding = nn.Embedding(MAX_BYTES, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, encoder_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.role_head = nn.Linear(width, len(ROLES))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, byte_ids: torch.Tensor, mask: torch.Tensor, *, zero_bytes: bool = False) -> ByteCompilerOutput:
        positions = torch.arange(byte_ids.shape[1], device=byte_ids.device)[None, :]
        input_ids = torch.where(mask, torch.zeros_like(byte_ids), byte_ids) if zero_bytes else byte_ids
        state = self.byte_embedding(input_ids) + self.position_embedding(positions)
        state = self.encoder(state, src_key_padding_mask=~mask)
        logits = self.role_head(self.norm(state))
        chosen = logits.argmax(-1)
        chosen = torch.where(mask, chosen, torch.full_like(chosen, ROLE_TO_ID["IGNORE"]))
        return ByteCompilerOutput(logits, chosen)


def byte_loss(output: ByteCompilerOutput, labels: torch.Tensor) -> torch.Tensor:
    active = labels != -100
    weights = torch.ones(len(ROLES), dtype=output.role_logits.dtype, device=output.role_logits.device)
    weights[ROLE_TO_ID["IGNORE"]] = 0.1
    return F.cross_entropy(output.role_logits[active], labels[active], weight=weights)
