"""Pretrained antisymmetric query grounding for DIVERGE-PQI1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Literal, Mapping, Sequence

import torch
import torch.nn as nn
from tokenizers import Tokenizer

from diverge_ats1_data import BYTE_OFFSET, CLS_ID, PAD_ID


SCHEMA = "shohin-diverge-pqi1-runtime-v1"
PLACEHOLDERS = ("alpha", "beta")
QueryControl = Literal["normal", "role_slot_swap", "scrub_context"]


class PQI1RuntimeError(RuntimeError):
    """A PQI1 source, mention, or model contract differs."""


@dataclass(frozen=True, slots=True)
class PQI1Config:
    layer: int = 19
    width: int = 384
    max_bytes: int = 192

    def validate(self) -> None:
        if self.layer != 19 or self.width != 384 or self.max_bytes != 192:
            raise PQI1RuntimeError("PQI1 frozen geometry differs")


@dataclass(frozen=True, slots=True)
class CanonicalQuery:
    text: str
    mention_spans: tuple[tuple[tuple[int, int], ...], ...]


def _contiguous_spans(mask: Sequence[bool]) -> tuple[tuple[int, int], ...]:
    spans = []
    start = None
    for index, active in enumerate((*mask, False)):
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append((start, index))
            start = None
    return tuple(spans)


def canonicalize_query(
    text: str,
    mention_masks: Sequence[Sequence[bool]],
    *,
    scrub_context: bool = False,
) -> CanonicalQuery:
    """Replace source names by fixed mention identities without role leakage."""

    if len(mention_masks) != 2 or any(len(mask) != len(text) for mask in mention_masks):
        raise PQI1RuntimeError("PQI1 mention mask geometry differs")
    events = []
    for group, mask in enumerate(mention_masks):
        spans = _contiguous_spans(mask)
        if not spans:
            raise PQI1RuntimeError("PQI1 mention group is empty")
        events.extend((left, right, group) for left, right in spans)
    events.sort()
    if any(right > next_left for (_, right, _), (next_left, _, _) in zip(events, events[1:])):
        raise PQI1RuntimeError("PQI1 mention groups overlap")

    if scrub_context:
        pieces = []
        output_spans: list[list[tuple[int, int]]] = [[], []]
        for event_index, (_, _, group) in enumerate(events):
            if event_index:
                pieces.append(" then ")
            left = sum(len(piece) for piece in pieces)
            pieces.append(PLACEHOLDERS[group])
            output_spans[group].append((left, left + len(PLACEHOLDERS[group])))
        return CanonicalQuery(
            "".join(pieces),
            tuple(tuple(spans) for spans in output_spans),
        )

    pieces = []
    output_spans = [[], []]
    cursor = 0
    length = 0
    for left, right, group in events:
        if left < cursor:
            raise PQI1RuntimeError("PQI1 mention ordering differs")
        prefix = text[cursor:left]
        pieces.append(prefix)
        length += len(prefix)
        replacement = PLACEHOLDERS[group]
        pieces.append(replacement)
        output_spans[group].append((length, length + len(replacement)))
        length += len(replacement)
        cursor = right
    pieces.append(text[cursor:])
    return CanonicalQuery(
        "".join(pieces),
        tuple(tuple(spans) for spans in output_spans),
    )


def adapter_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
        if not name.startswith("backbone.")
    }


def adapter_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter_state_dict(module).items()):
        digest.update(name.encode("ascii"))
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_adapter_state(module: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    own = module.state_dict()
    expected = {name for name in own if not name.startswith("backbone.")}
    if set(state) != expected:
        raise PQI1RuntimeError("PQI1 adapter tensor names differ")
    for name in expected:
        if own[name].shape != state[name].shape:
            raise PQI1RuntimeError(f"PQI1 adapter tensor shape differs: {name}")
        own[name].copy_(state[name])


class PretrainedQueryGrounder(nn.Module):
    """Use frozen language residuals and one shared candidate scoring rule."""

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Tokenizer,
        config: PQI1Config,
    ) -> None:
        super().__init__()
        config.validate()
        if config.layer >= len(backbone.blocks):
            raise PQI1RuntimeError("PQI1 layer exceeds the backbone")
        self.backbone = backbone.requires_grad_(False)
        self.tokenizer = tokenizer
        self.config = config
        backbone_width = int(backbone.cfg.d_model)
        self.memory_norm = nn.LayerNorm(backbone_width)
        self.memory_projection = nn.Linear(backbone_width, config.width, bias=False)
        self.candidate = nn.Sequential(
            nn.LayerNorm(config.width * 4),
            nn.Linear(config.width * 4, config.width),
            nn.GELU(),
            nn.Linear(config.width, 1),
        )

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone."):
                yield parameter

    def _canonical_batch(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
        *,
        scrub_context: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or byte_ids.shape[1] != self.config.max_bytes
            or byte_ids.dtype != torch.long
            or attention_mask.dtype != torch.bool
            or symbol_masks.shape != (byte_ids.shape[0], 2, self.config.max_bytes)
            or symbol_masks.dtype != torch.bool
        ):
            raise PQI1RuntimeError("PQI1 tensor interface differs")
        if not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise PQI1RuntimeError("PQI1 CLS differs")

        encoded_rows = []
        token_group_masks = []
        cpu_ids = byte_ids.detach().cpu()
        cpu_attention = attention_mask.detach().cpu()
        cpu_symbols = symbol_masks.detach().cpu()
        for row in range(byte_ids.shape[0]):
            length = int(cpu_attention[row].sum().item())
            if length < 2 or torch.any(cpu_attention[row, :length].logical_not()):
                raise PQI1RuntimeError("PQI1 source mask differs")
            values = cpu_ids[row, 1:length] - BYTE_OFFSET
            if torch.any((values < 0) | (values > 127)):
                raise PQI1RuntimeError("PQI1 source is not ASCII")
            text = bytes(int(value) for value in values.tolist()).decode("ascii")
            mention_masks = [
                cpu_symbols[row, group, 1:length].tolist() for group in range(2)
            ]
            canonical = canonicalize_query(
                text, mention_masks, scrub_context=scrub_context
            )
            encoded = self.tokenizer.encode(
                canonical.text, add_special_tokens=False
            )
            if not encoded.ids or len(encoded.ids) > int(self.backbone.cfg.seq_len):
                raise PQI1RuntimeError("PQI1 tokenized source length differs")
            groups = torch.zeros((2, len(encoded.ids)), dtype=torch.bool)
            for group, spans in enumerate(canonical.mention_spans):
                for token, (left, right) in enumerate(encoded.offsets):
                    if left != right and any(left < end and right > start for start, end in spans):
                        groups[group, token] = True
            if torch.any(groups.sum(dim=-1) < 1) or torch.any(groups[0] & groups[1]):
                raise PQI1RuntimeError("PQI1 token mention projection differs")
            encoded_rows.append(list(encoded.ids))
            token_group_masks.append(groups)

        width = max(len(row) for row in encoded_rows)
        ids = torch.full((len(encoded_rows), width), PAD_ID, dtype=torch.long)
        valid = torch.zeros((len(encoded_rows), width), dtype=torch.bool)
        groups = torch.zeros((len(encoded_rows), 2, width), dtype=torch.bool)
        for index, (row, row_groups) in enumerate(zip(encoded_rows, token_group_masks, strict=True)):
            ids[index, : len(row)] = torch.tensor(row)
            valid[index, : len(row)] = True
            groups[index, :, : len(row)] = row_groups
        return ids.to(byte_ids.device), valid.to(byte_ids.device), groups.to(byte_ids.device)

    def _memory(self, ids: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            hidden = self.backbone.tok(ids)
            cosine = self.backbone.cos[: ids.shape[1]].to(hidden.device)
            sine = self.backbone.sin[: ids.shape[1]].to(hidden.device)
            for block in self.backbone.blocks[: self.config.layer + 1]:
                hidden, _ = block(hidden, cosine, sine)
        hidden = hidden.detach().to(self.memory_projection.weight.dtype)
        return self.memory_projection(self.memory_norm(hidden))

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
        *,
        control: QueryControl = "normal",
    ) -> torch.Tensor:
        if control not in ("normal", "role_slot_swap", "scrub_context"):
            raise PQI1RuntimeError(f"unknown PQI1 control: {control}")
        ids, valid, groups = self._canonical_batch(
            byte_ids,
            attention_mask,
            symbol_masks,
            scrub_context=control == "scrub_context",
        )
        memory = self._memory(ids)
        weights = valid.unsqueeze(-1).to(memory.dtype)
        global_hidden = (memory * weights).sum(dim=1) / weights.sum(dim=1)
        group_weights = groups.to(memory.dtype)
        mention = torch.einsum("bgs,bsw->bgw", group_weights, memory)
        mention = mention / group_weights.sum(dim=-1, keepdim=True)
        context = global_hidden[:, None, :].expand(-1, 2, -1)
        features = torch.cat(
            (mention, context, mention * context, mention - context), dim=-1
        )
        target_score = self.candidate(features).squeeze(-1).float()
        logits = torch.stack((target_score, -target_score), dim=-1)
        if control == "role_slot_swap":
            logits = logits.flip(-1)
        return logits


__all__ = [
    "CanonicalQuery",
    "PLACEHOLDERS",
    "PQI1Config",
    "PQI1RuntimeError",
    "PretrainedQueryGrounder",
    "QueryControl",
    "SCHEMA",
    "adapter_state_dict",
    "adapter_state_sha256",
    "canonicalize_query",
    "load_adapter_state",
]
