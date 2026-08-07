"""Counterfactual whole-world selector for DIVERGE-CWC1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_OFFSET, BYTE_VOCAB_SIZE, CLS_ID, PAD_ID
from diverge_cwc1_data import MAX_SOURCE_BYTES, counterfactual_source


SCHEMA = "shohin-diverge-cwc1-runtime-v1"
ProjectionMode = Literal["involution", "duplicate"]


class CWC1RuntimeError(RuntimeError):
    """A CWC1 tensor or projected decision violates the contract."""


def module_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CWC1Config:
    width: int = 192
    layers: int = 2
    max_bytes: int = MAX_SOURCE_BYTES
    projection_mode: ProjectionMode = "involution"

    def validate(self) -> None:
        if self.width != 192 or self.layers != 2 or self.width % 2:
            raise CWC1RuntimeError("CWC1 encoder geometry differs")
        if self.max_bytes != MAX_SOURCE_BYTES:
            raise CWC1RuntimeError("CWC1 source width differs")
        if self.projection_mode not in ("involution", "duplicate"):
            raise CWC1RuntimeError("CWC1 projection mode differs")


@dataclass(frozen=True, slots=True)
class CWC1Batch:
    byte_ids: torch.Tensor
    attention_mask: torch.Tensor
    candidate_masks: torch.Tensor
    label_masks: torch.Tensor
    targets: torch.Tensor


class CounterfactualWorldCommitter(nn.Module):
    """Score two complete world lineages through one shared byte encoder."""

    def __init__(self, config: CWC1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.width
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, width)
        self.encoder = nn.GRU(
            input_size=width,
            hidden_size=width // 2,
            num_layers=config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.output_norm = nn.LayerNorm(width)
        self.block_projection = nn.Linear(width, width)
        self.label_projection = nn.Linear(width, width)
        self.global_projection = nn.Linear(width, width)
        self.score = nn.Linear(width, 1)

    def _encode(self, batch: CWC1Batch) -> torch.Tensor:
        if (
            batch.byte_ids.ndim != 2
            or batch.byte_ids.shape != batch.attention_mask.shape
            or batch.byte_ids.shape[1] != self.config.max_bytes
            or batch.candidate_masks.shape != (batch.byte_ids.shape[0], 2, self.config.max_bytes)
            or batch.label_masks.shape != batch.candidate_masks.shape
        ):
            raise CWC1RuntimeError("CWC1 tensor interface differs")
        lengths = batch.attention_mask.sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(batch.byte_ids[:, 0].eq(CLS_ID)):
            raise CWC1RuntimeError("CWC1 source mask differs")
        packed = pack_padded_sequence(
            self.embedding(batch.byte_ids),
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=self.config.max_bytes
        )
        return self.output_norm(hidden)

    @staticmethod
    def _pool(hidden: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        counts = masks.sum(dim=-1, keepdim=True)
        if torch.any(counts < 1):
            raise CWC1RuntimeError("CWC1 candidate group is empty")
        return torch.einsum("bms,bsw->bmw", masks.to(hidden.dtype), hidden) / counts.to(hidden.dtype)

    def raw_scores(self, batch: CWC1Batch) -> torch.Tensor:
        hidden = self._encode(batch)
        block = self._pool(hidden, batch.candidate_masks)
        labels = self._pool(hidden, batch.label_masks)
        features = torch.tanh(
            self.block_projection(block)
            + self.label_projection(labels)
            + self.global_projection(hidden[:, 0]).unsqueeze(1)
        )
        return self.score(features).squeeze(-1).float()

    def projected_scores(self, normal: CWC1Batch, partner: CWC1Batch) -> torch.Tensor:
        first = self.raw_scores(normal)
        second = self.raw_scores(partner if self.config.projection_mode == "involution" else normal)
        if self.config.projection_mode == "involution":
            return 0.5 * (first + second.flip(dims=(-1,)))
        return 0.5 * (first + second)

    def record(self) -> dict[str, object]:
        return {"schema": SCHEMA, "config": asdict(self.config)}


def _encode(text: str) -> tuple[int, ...]:
    raw = text.encode("ascii")
    if not raw or len(raw) + 1 > MAX_SOURCE_BYTES:
        raise CWC1RuntimeError("CWC1 source width differs")
    return (CLS_ID, *(value + BYTE_OFFSET for value in raw))


def _label_masks(text: str, labels: Sequence[str]) -> tuple[list[list[bool]], ...]:
    output = []
    for label in labels:
        mask = [False] * MAX_SOURCE_BYTES
        pattern = re.compile(rf"(?<![a-z]){re.escape(label)}(?![a-z])")
        for match in pattern.finditer(text):
            mask[match.start() + 1 : match.end() + 1] = [True] * (match.end() - match.start())
        output.append(mask)
    return tuple(output)


def tensorize_records(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    counterfactual: bool = False,
) -> CWC1Batch:
    if not records:
        raise CWC1RuntimeError("CWC1 batch is empty")
    count = len(records)
    byte_ids = torch.full((count, MAX_SOURCE_BYTES), PAD_ID, dtype=torch.long)
    attention = torch.zeros((count, MAX_SOURCE_BYTES), dtype=torch.bool)
    candidate_masks = torch.zeros((count, 2, MAX_SOURCE_BYTES), dtype=torch.bool)
    label_masks = torch.zeros((count, 2, MAX_SOURCE_BYTES), dtype=torch.bool)
    targets = torch.zeros(count, dtype=torch.long)
    for row_index, record in enumerate(records):
        text = counterfactual_source(record) if counterfactual else str(record["source_text"])
        encoded = _encode(text)
        byte_ids[row_index, : len(encoded)] = torch.tensor(encoded)
        attention[row_index, : len(encoded)] = True
        for candidate, bounds in enumerate(record["candidate_bounds"]):
            left, right = (int(value) for value in bounds)
            candidate_masks[row_index, candidate, left + 1 : right + 1] = True
        for candidate, mask in enumerate(_label_masks(text, record["candidate_labels"])):
            label_masks[row_index, candidate] = torch.tensor(mask)
        targets[row_index] = int(record.get("target_position", 0))
    return CWC1Batch(
        byte_ids=byte_ids.to(device),
        attention_mask=attention.to(device),
        candidate_masks=candidate_masks.to(device),
        label_masks=label_masks.to(device),
        targets=targets.to(device),
    )


__all__ = [
    "CWC1Batch",
    "CWC1Config",
    "CWC1RuntimeError",
    "CounterfactualWorldCommitter",
    "ProjectionMode",
    "module_state_sha256",
    "tensorize_records",
]
