"""Candidate-only source compiler and algebraic typed-state runtime for ATS1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_ats1_data import (
    BYTE_OFFSET,
    BYTE_VOCAB_SIZE,
    CLS_ID,
    MAX_SEGMENT_BYTES,
    OPERATION_NAMES,
    ROLE_NAMES,
    ROLE_TO_ID,
)


class ATS1RuntimeError(RuntimeError):
    """The source-sealed typed packet is malformed or unsupported."""


SCHEMA = "shohin-diverge-ats1-runtime-v1"
MODULI = (17, 19, 23, 29, 31)
CRT_PRODUCT = 6_678_671
MAX_SYMBOLS = 24


@dataclass(frozen=True, slots=True)
class ATS1Config:
    width: int = 128
    layers: int = 2
    heads: int = 4
    ff_multiplier: int = 4
    max_bytes: int = MAX_SEGMENT_BYTES

    def validate(self) -> None:
        if min(self.width, self.layers, self.heads, self.ff_multiplier) <= 0:
            raise ATS1RuntimeError("compiler geometry must be positive")
        if self.width % self.heads:
            raise ATS1RuntimeError("compiler width must divide attention heads")
        if self.max_bytes != MAX_SEGMENT_BYTES:
            raise ATS1RuntimeError("ATS1 v1 fixes the source-segment width")


@dataclass(frozen=True, slots=True)
class TypedState:
    family: str
    residues_a: tuple[int, ...] = ()
    residues_b: tuple[int, ...] = ()
    symbols: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.family == "scalar":
            if len(self.residues_a) != len(MODULI) or self.residues_b or self.symbols:
                raise ATS1RuntimeError("scalar packet shape differs")
        elif self.family == "register":
            if (
                len(self.residues_a) != len(MODULI)
                or len(self.residues_b) != len(MODULI)
                or self.symbols
            ):
                raise ATS1RuntimeError("register packet shape differs")
        elif self.family == "symbolic":
            if self.residues_a or self.residues_b or not self.symbols:
                raise ATS1RuntimeError("symbol packet shape differs")
            if len(self.symbols) > MAX_SYMBOLS or any(not 0 <= value < 26 for value in self.symbols):
                raise ATS1RuntimeError("symbol packet leaves the closed alphabet")
        else:
            raise ATS1RuntimeError("unknown typed-state family")
        for values in (self.residues_a, self.residues_b):
            if any(not 0 <= value < modulus for value, modulus in zip(values, MODULI)):
                raise ATS1RuntimeError("numeric packet leaves a residue domain")

    def record(self) -> dict[str, object]:
        return {
            "family": self.family,
            "residues_a": list(self.residues_a),
            "residues_b": list(self.residues_b),
            "symbols": list(self.symbols),
        }


@dataclass(frozen=True, slots=True)
class CompiledSegment:
    operation_id: int
    lhs: TypedState
    rhs_claim: TypedState
    arguments: tuple[int, ...]
    provenance_positions: tuple[tuple[int, ...], ...]


class SourceRoleCompiler(nn.Module):
    """Predict source-owned roles and one complete transaction class."""

    def __init__(self, config: ATS1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.byte_embedding = nn.Embedding(BYTE_VOCAB_SIZE, config.width)
        self.position_embedding = nn.Parameter(torch.empty(config.max_bytes, config.width))
        self.cls_embedding = nn.Parameter(torch.empty(config.width))
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.cls_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=config.width * config.ff_multiplier,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.role_head = nn.Linear(config.width, len(ROLE_NAMES))
        self.operation_head = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, len(OPERATION_NAMES)),
        )

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or byte_ids.shape[1] != self.config.max_bytes
            or byte_ids.dtype != torch.long
        ):
            raise ATS1RuntimeError("compiler tensor interface differs")
        active = attention_mask.bool()
        if torch.any(active.sum(dim=1) < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise ATS1RuntimeError("compiler source mask or CLS differs")
        hidden = self.byte_embedding(byte_ids) + self.position_embedding[None]
        hidden = hidden.clone()
        hidden[:, 0] = hidden[:, 0] + self.cls_embedding
        hidden = self.encoder(hidden, src_key_padding_mask=~active)
        hidden = self.output_norm(hidden)
        role_logits = self.role_head(hidden).float()
        operation_logits = self.operation_head(hidden[:, 0]).float()
        return role_logits, operation_logits

    def record(self) -> dict[str, object]:
        return {"schema": SCHEMA, "config": asdict(self.config)}


def compiler_loss(
    role_logits: torch.Tensor,
    operation_logits: torch.Tensor,
    role_targets: torch.Tensor,
    operation_targets: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if (
        role_logits.shape[:2] != role_targets.shape
        or role_logits.shape[2] != len(ROLE_NAMES)
        or operation_logits.shape != (role_targets.shape[0], len(OPERATION_NAMES))
        or operation_targets.shape != (role_targets.shape[0],)
        or attention_mask.shape != role_targets.shape
    ):
        raise ATS1RuntimeError("compiler loss tensors differ")
    weights = torch.ones(len(ROLE_NAMES), device=role_logits.device)
    weights[ROLE_TO_ID["OTHER"]] = 0.05
    per_token = F.cross_entropy(
        role_logits.flatten(0, 1),
        role_targets.flatten(),
        weight=weights,
        reduction="none",
    ).reshape_as(role_targets)
    active = attention_mask.to(per_token.dtype)
    role_loss = (per_token * active).sum() / active.sum().clamp_min(1.0)
    operation_loss = F.cross_entropy(operation_logits, operation_targets)
    total = role_loss + operation_loss
    return total, {
        "loss": float(total.detach()),
        "role_loss": float(role_loss.detach()),
        "operation_loss": float(operation_loss.detach()),
    }


def _source_char(raw_id: int) -> str:
    value = int(raw_id)
    if not BYTE_OFFSET <= value < BYTE_OFFSET + 128:
        raise ATS1RuntimeError("selected role does not point to a source byte")
    return chr(value - BYTE_OFFSET)


def _extract(
    byte_ids: Sequence[int],
    role_ids: Sequence[int],
    role_name: str,
) -> tuple[str, tuple[int, ...]]:
    if len(byte_ids) != len(role_ids):
        raise ATS1RuntimeError("source bytes and roles differ")
    role = ROLE_TO_ID[role_name]
    positions = tuple(index for index, value in enumerate(role_ids) if int(value) == role)
    if not positions:
        return "", ()
    if any(right != left + 1 for left, right in zip(positions, positions[1:])):
        raise ATS1RuntimeError(f"{role_name} source copy is noncontiguous")
    return "".join(_source_char(byte_ids[index]) for index in positions), positions


def _ascii_unsigned(text: str) -> int:
    if not text or any(character < "0" or character > "9" for character in text):
        raise ATS1RuntimeError("copied argument is not unsigned decimal")
    value = 0
    for character in text:
        value = 10 * value + (ord(character) - ord("0"))
    return value


def decimal_to_residues(text: str) -> tuple[int, ...]:
    if not text:
        raise ATS1RuntimeError("copied numeric state is empty")
    negative = text[0] == "-"
    digits = text[1:] if negative else text
    if not digits or any(character < "0" or character > "9" for character in digits):
        raise ATS1RuntimeError("copied numeric state is not canonical decimal")
    output: list[int] = []
    for modulus in MODULI:
        residue = 0
        for character in digits:
            residue = (10 * residue + ord(character) - ord("0")) % modulus
        output.append((-residue) % modulus if negative else residue)
    return tuple(output)


def _symbol_codes(text: str) -> tuple[int, ...]:
    if not text or len(text) > MAX_SYMBOLS or any(not "a" <= value <= "z" for value in text):
        raise ATS1RuntimeError("copied symbol tape is malformed")
    return tuple(ord(value) - ord("a") for value in text)


def typed_state_from_surfaces(
    family: str,
    value_a: str,
    value_b: str = "",
    symbol: str = "",
) -> TypedState:
    if family == "scalar":
        if value_b or symbol:
            raise ATS1RuntimeError("scalar source fields differ")
        return TypedState(family="scalar", residues_a=decimal_to_residues(value_a))
    if family == "register":
        if not value_b or symbol:
            raise ATS1RuntimeError("register source fields differ")
        return TypedState(
            family="register",
            residues_a=decimal_to_residues(value_a),
            residues_b=decimal_to_residues(value_b),
        )
    if family == "symbolic":
        if value_a or value_b:
            raise ATS1RuntimeError("symbol source fields differ")
        return TypedState(family="symbolic", symbols=_symbol_codes(symbol))
    raise ATS1RuntimeError("unknown source-state family")


def _family_for_operation(operation_id: int) -> str:
    value = int(operation_id)
    if 0 <= value <= 2:
        return "scalar"
    if 3 <= value <= 7:
        return "register"
    if 8 <= value <= 10:
        return "symbolic"
    raise ATS1RuntimeError("operation class leaves the vocabulary")


def compile_segment(
    byte_ids: Sequence[int],
    role_ids: Sequence[int],
    operation_id: int,
) -> CompiledSegment:
    family = _family_for_operation(operation_id)
    extracted = {
        name: _extract(byte_ids, role_ids, name)
        for name in ROLE_NAMES
        if name != "OTHER"
    }
    lhs_a, lhs_a_positions = extracted["LHS_A"]
    lhs_b, lhs_b_positions = extracted["LHS_B"]
    lhs_symbol, lhs_symbol_positions = extracted["LHS_SYMBOL"]
    rhs_a, rhs_a_positions = extracted["RHS_A"]
    rhs_b, rhs_b_positions = extracted["RHS_B"]
    rhs_symbol, rhs_symbol_positions = extracted["RHS_SYMBOL"]
    argument_text = [extracted["ARG1"][0], extracted["ARG2"][0]]
    arguments = tuple(_ascii_unsigned(value) for value in argument_text if value)

    if family == "scalar":
        if any((lhs_b, lhs_symbol, rhs_b, rhs_symbol)) or len(arguments) != 1:
            raise ATS1RuntimeError("scalar compiled fields differ")
        lhs = typed_state_from_surfaces("scalar", lhs_a)
        rhs = typed_state_from_surfaces("scalar", rhs_a)
    elif family == "register":
        if any((lhs_symbol, rhs_symbol, *argument_text)):
            raise ATS1RuntimeError("register compiled fields differ")
        lhs = typed_state_from_surfaces("register", lhs_a, lhs_b)
        rhs = typed_state_from_surfaces("register", rhs_a, rhs_b)
    else:
        expected = 0 if operation_id == 8 else (1 if operation_id == 9 else 2)
        if any((lhs_a, lhs_b, rhs_a, rhs_b)) or len(arguments) != expected:
            raise ATS1RuntimeError("symbolic compiled fields differ")
        lhs = typed_state_from_surfaces("symbolic", "", symbol=lhs_symbol)
        rhs = typed_state_from_surfaces("symbolic", "", symbol=rhs_symbol)

    provenance = (
        lhs_a_positions,
        lhs_b_positions,
        lhs_symbol_positions,
        rhs_a_positions,
        rhs_b_positions,
        rhs_symbol_positions,
        extracted["ARG1"][1],
        extracted["ARG2"][1],
    )
    return CompiledSegment(
        operation_id=int(operation_id),
        lhs=lhs,
        rhs_claim=rhs,
        arguments=arguments,
        provenance_positions=provenance,
    )


def _binary(
    left: Sequence[int],
    right: Sequence[int],
    kind: str,
) -> tuple[int, ...]:
    if len(left) != len(MODULI) or len(right) != len(MODULI):
        raise ATS1RuntimeError("numeric transaction packet differs")
    output: list[int] = []
    for lhs, rhs, modulus in zip(left, right, MODULI):
        if kind == "add":
            value = lhs + rhs
        elif kind == "subtract":
            value = lhs - rhs
        elif kind == "multiply":
            value = lhs * rhs
        else:
            raise ATS1RuntimeError("unknown algebraic kernel")
        output.append(value % modulus)
    return tuple(output)


def execute_step(state: TypedState, operation_id: int, arguments: Sequence[int]) -> TypedState:
    operation_id = int(operation_id)
    if _family_for_operation(operation_id) != state.family:
        raise ATS1RuntimeError("transaction family differs from state")
    if state.family == "scalar":
        if len(arguments) != 1:
            raise ATS1RuntimeError("scalar transaction argument differs")
        argument = tuple(int(arguments[0]) % modulus for modulus in MODULI)
        kind = ("add", "subtract", "multiply")[operation_id]
        return TypedState(
            family="scalar",
            residues_a=_binary(state.residues_a, argument, kind),
        )
    if state.family == "register":
        if arguments:
            raise ATS1RuntimeError("register transaction unexpectedly has arguments")
        a, b = state.residues_a, state.residues_b
        if operation_id == 3:
            a = _binary(a, b, "add")
        elif operation_id == 4:
            b = _binary(b, a, "subtract")
        elif operation_id == 5:
            a, b = b, a
        elif operation_id == 6:
            two = tuple(2 % modulus for modulus in MODULI)
            a = _binary(a, two, "multiply")
        elif operation_id == 7:
            b = _binary(b, a, "add")
        else:
            raise ATS1RuntimeError("register transaction leaves its vocabulary")
        return TypedState(family="register", residues_a=a, residues_b=b)

    values = list(state.symbols)
    if operation_id == 8:
        if arguments:
            raise ATS1RuntimeError("reverse transaction unexpectedly has arguments")
        values.reverse()
    elif operation_id == 9:
        if len(arguments) != 1:
            raise ATS1RuntimeError("rotate transaction argument differs")
        offset = int(arguments[0]) % len(values)
        values = values[offset:] + values[:offset]
    elif operation_id == 10:
        if len(arguments) != 2:
            raise ATS1RuntimeError("swap transaction arguments differ")
        left, right = int(arguments[0]) - 1, int(arguments[1]) - 1
        if not 0 <= left < len(values) or not 0 <= right < len(values):
            raise ATS1RuntimeError("swap transaction leaves the symbol tape")
        values[left], values[right] = values[right], values[left]
    else:
        raise ATS1RuntimeError("symbol transaction leaves its vocabulary")
    return TypedState(family="symbolic", symbols=tuple(values))


def crt_signed(residues: Sequence[int]) -> int:
    """Independent exact readout used by reports, never by execute_step."""

    if len(residues) != len(MODULI):
        raise ATS1RuntimeError("CRT readout packet differs")
    value = 0
    for residue, modulus in zip(residues, MODULI):
        partial = CRT_PRODUCT // modulus
        inverse = pow(partial, -1, modulus)
        value = (value + int(residue) * partial * inverse) % CRT_PRODUCT
    return value - CRT_PRODUCT if value > CRT_PRODUCT // 2 else value


def render_typed_state(state: TypedState) -> str:
    if state.family == "scalar":
        return str(crt_signed(state.residues_a))
    if state.family == "register":
        return f"{crt_signed(state.residues_a)},{crt_signed(state.residues_b)}"
    return "".join(chr(value + ord("a")) for value in state.symbols)


def shifted_operation(operation_id: int) -> int:
    groups = ((0, 1, 2), (3, 4, 5, 6, 7), (8, 9, 10))
    for group in groups:
        if operation_id in group:
            index = group.index(operation_id)
            return group[(index + 1) % len(group)]
    raise ATS1RuntimeError("operation shift leaves the vocabulary")


__all__ = [
    "ATS1Config",
    "ATS1RuntimeError",
    "CRT_PRODUCT",
    "CompiledSegment",
    "MODULI",
    "SCHEMA",
    "SourceRoleCompiler",
    "TypedState",
    "compile_segment",
    "compiler_loss",
    "crt_signed",
    "decimal_to_residues",
    "execute_step",
    "render_typed_state",
    "shifted_operation",
    "typed_state_from_surfaces",
]
