"""Stage-owned neural interfaces for the DIVERGE-SOT1 transaction machine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_VOCAB_SIZE
from diverge_iem1_runtime import (
    CLS_ID,
    IEM1RuntimeError,
    MAX_QUERY_BYTES,
    module_state_sha256,
)
from diverge_nve1_runtime import EvidenceCompilerConfig, NaturalEvidenceCompiler
from diverge_tol3_semantic_anchor import LocalSemanticAnchor, TOL3Config


SCHEMA = "shohin-diverge-sot1-runtime-v1"


class SOT1RuntimeError(RuntimeError):
    """A stage-owned transaction violates the SOT1 contract."""


@dataclass(frozen=True, slots=True)
class SOT1Config:
    query_width: int = 192
    query_layers: int = 2
    query_max_bytes: int = MAX_QUERY_BYTES

    def validate(self) -> None:
        if self.query_width != 192 or self.query_layers != 2:
            raise SOT1RuntimeError("SOT1 query owner geometry differs")
        if self.query_width % 2 or self.query_max_bytes != MAX_QUERY_BYTES:
            raise SOT1RuntimeError("SOT1 query owner width differs")


class NaturalQueryOwner(nn.Module):
    """Assign two complete source-owned symbol groups to query roles."""

    def __init__(self, config: SOT1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.query_width
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, width)
        self.encoder = nn.GRU(
            input_size=width,
            hidden_size=width // 2,
            num_layers=config.query_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.output_norm = nn.LayerNorm(width)
        self.query_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> torch.Tensor:
        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or byte_ids.shape[1] != self.config.query_max_bytes
            or byte_ids.dtype != torch.long
            or attention_mask.dtype != torch.bool
            or symbol_masks.shape != (byte_ids.shape[0], 2, self.config.query_max_bytes)
        ):
            raise SOT1RuntimeError("SOT1 query tensor interface differs")
        lengths = attention_mask.sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise SOT1RuntimeError("SOT1 query mask or CLS differs")
        if torch.any(symbol_masks.sum(dim=-1) < 1):
            raise SOT1RuntimeError("SOT1 query symbol group is empty")
        packed = pack_padded_sequence(
            self.embedding(byte_ids),
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=self.config.query_max_bytes,
        )
        hidden = self.output_norm(hidden)
        pooled = torch.einsum(
            "bms,bsw->bmw", symbol_masks.to(hidden.dtype), hidden
        ) / symbol_masks.sum(dim=-1, keepdim=True).to(hidden.dtype)
        return self.query_head(pooled).float()


class StageOwnedEpistemicMachine(nn.Module):
    """One checkpoint with disjoint WORLD, EVIDENCE, and QUERY owners."""

    def __init__(self, config: SOT1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.source_owner = LocalSemanticAnchor(TOL3Config())
        self.evidence_owner = NaturalEvidenceCompiler(EvidenceCompilerConfig())
        self.query_owner = NaturalQueryOwner(config)

    def forward_query(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.query_owner(byte_ids, attention_mask, symbol_masks)

    def freeze_qualified_owners(self) -> None:
        self.source_owner.requires_grad_(False)
        self.evidence_owner.requires_grad_(False)
        self.query_owner.requires_grad_(True)

    def owner_hashes(self) -> dict[str, str]:
        return {
            "WORLD": module_state_sha256(self.source_owner),
            "EVIDENCE": module_state_sha256(self.evidence_owner),
            "QUERY": module_state_sha256(self.query_owner),
        }

    def owner_manifest(self) -> dict[str, object]:
        validate_owner_isolation(self)
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "owner_hashes": self.owner_hashes(),
            "transaction_order": ["WORLD", "EVIDENCE", "EXECUTE", "QUERY"],
        }


def _storage_pointers(parameters: Iterable[torch.nn.Parameter]) -> set[int]:
    return {parameter.untyped_storage().data_ptr() for parameter in parameters}


def validate_owner_isolation(model: StageOwnedEpistemicMachine) -> None:
    owners = {
        "WORLD": _storage_pointers(model.source_owner.parameters()),
        "EVIDENCE": _storage_pointers(model.evidence_owner.parameters()),
        "QUERY": _storage_pointers(model.query_owner.parameters()),
    }
    for left_name, left in owners.items():
        for right_name, right in owners.items():
            if left_name < right_name and left & right:
                raise SOT1RuntimeError(
                    f"SOT1 owners {left_name}/{right_name} alias parameter storage"
                )
    frozen = tuple(model.source_owner.parameters()) + tuple(
        model.evidence_owner.parameters()
    )
    if any(parameter.requires_grad for parameter in frozen):
        raise SOT1RuntimeError("SOT1 qualified owner is plastic")
    if not all(parameter.requires_grad for parameter in model.query_owner.parameters()):
        raise SOT1RuntimeError("SOT1 query owner is not plastic")


def query_owner_parameters(model: StageOwnedEpistemicMachine):
    model.freeze_qualified_owners()
    validate_owner_isolation(model)
    return tuple(model.query_owner.parameters())


def compile_query_batch(*args, **kwargs):
    """Use the unchanged IEM1 receipt compiler with the isolated query owner."""

    from diverge_iem1_runtime import compile_query_batch as compile_impl

    try:
        return compile_impl(*args, **kwargs)
    except IEM1RuntimeError as error:
        raise SOT1RuntimeError(str(error)) from error
