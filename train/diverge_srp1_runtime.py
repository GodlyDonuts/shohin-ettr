"""Semantic-primitive ownership for DIVERGE-SRP1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable, Mapping

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_VOCAB_SIZE
from diverge_iem1_runtime import CLS_ID, MAX_QUERY_BYTES, module_state_sha256
from diverge_nve1_runtime import EvidenceCompilerConfig, NaturalEvidenceCompiler
from diverge_tol3_semantic_anchor import LocalSemanticAnchor, TOL3Config


SCHEMA = "shohin-diverge-srp1-runtime-v1"


class SRP1RuntimeError(RuntimeError):
    """An SRP1 owner or tensor violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class SRP1Config:
    width: int = 192
    layers: int = 2
    max_bytes: int = MAX_QUERY_BYTES

    def validate(self) -> None:
        if self.width != 192 or self.layers != 2 or self.width % 2:
            raise SRP1RuntimeError("SRP1 referent geometry differs")
        if self.max_bytes != MAX_QUERY_BYTES:
            raise SRP1RuntimeError("SRP1 source width differs")


class SemanticReferentOwner(nn.Module):
    """Exchange-equivariant complete TARGET/DISTRACTOR assignment."""

    def __init__(self, config: SRP1Config) -> None:
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
        self.candidate_score = nn.Sequential(
            nn.LayerNorm(width * 5),
            nn.Linear(width * 5, width),
            nn.GELU(),
            nn.Linear(width, 1),
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
            or byte_ids.shape[1] != self.config.max_bytes
            or byte_ids.dtype != torch.long
            or attention_mask.dtype != torch.bool
            or symbol_masks.shape != (byte_ids.shape[0], 2, self.config.max_bytes)
        ):
            raise SRP1RuntimeError("SRP1 referent tensor interface differs")
        lengths = attention_mask.sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise SRP1RuntimeError("SRP1 referent mask or CLS differs")
        if torch.any(symbol_masks.sum(dim=-1) < 1):
            raise SRP1RuntimeError("SRP1 referent group is empty")
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
            total_length=self.config.max_bytes,
        )
        hidden = self.output_norm(hidden)
        group_hidden = torch.einsum(
            "bms,bsw->bmw", symbol_masks.to(hidden.dtype), hidden
        ) / symbol_masks.sum(dim=-1, keepdim=True).to(hidden.dtype)
        global_hidden = torch.einsum(
            "bs,bsw->bw", attention_mask.to(hidden.dtype), hidden
        ) / attention_mask.sum(dim=-1, keepdim=True).to(hidden.dtype)
        other_hidden = group_hidden.flip(1)
        global_expanded = global_hidden.unsqueeze(1).expand(-1, 2, -1)
        features = torch.cat(
            (
                global_expanded,
                group_hidden,
                other_hidden,
                group_hidden - other_hidden,
                group_hidden * other_hidden,
            ),
            dim=-1,
        )
        scores = self.candidate_score(features).squeeze(-1)
        delta = scores[:, 0] - scores[:, 1]
        return torch.stack(
            (
                torch.stack((delta, -delta), dim=-1),
                torch.stack((-delta, delta), dim=-1),
            ),
            dim=1,
        ).float()


def warm_start_referent(
    referent: SemanticReferentOwner,
    evidence_state: Mapping[str, torch.Tensor],
) -> None:
    required = {
        *{f"embedding.{name}" for name in referent.embedding.state_dict()},
        *{f"encoder.{name}" for name in referent.encoder.state_dict()},
        *{f"output_norm.{name}" for name in referent.output_norm.state_dict()},
    }
    missing = sorted(required - set(evidence_state))
    if missing:
        raise SRP1RuntimeError(f"SRP1 NVE1 warm start is missing {missing}")
    own = referent.state_dict()
    with torch.no_grad():
        for name in required:
            if own[name].shape != evidence_state[name].shape:
                raise SRP1RuntimeError(f"SRP1 warm-start tensor differs: {name}")
            own[name].copy_(evidence_state[name])


def _combined_sha256(payload: Mapping[str, str]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class SemanticPrimitiveEpistemicMachine(nn.Module):
    """Qualified WORLD/numeric owners plus one cross-stage REFERENT owner."""

    def __init__(self, config: SRP1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.source_owner = LocalSemanticAnchor(TOL3Config())
        self.numeric_evidence_owner = NaturalEvidenceCompiler(EvidenceCompilerConfig())
        self.referent_owner = SemanticReferentOwner(config)

    @property
    def evidence_owner(self) -> SRP1EvidenceView:
        return SRP1EvidenceView(self)

    def forward_query(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.referent_owner(byte_ids, attention_mask, symbol_masks)

    def forward_evidence(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_bounds: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        numeric_logits, _ = self.numeric_evidence_owner(
            byte_ids,
            attention_mask,
            numeric_bounds,
            symbol_masks,
        )
        symbol_logits = self.referent_owner(byte_ids, attention_mask, symbol_masks)
        return numeric_logits, symbol_logits

    def freeze_qualified_owners(self) -> None:
        self.source_owner.requires_grad_(False)
        self.numeric_evidence_owner.requires_grad_(False)
        self.referent_owner.requires_grad_(True)

    def owner_hashes(self) -> dict[str, str]:
        world = module_state_sha256(self.source_owner)
        numeric = module_state_sha256(self.numeric_evidence_owner)
        referent = module_state_sha256(self.referent_owner)
        return {
            "WORLD": world,
            "NUMERIC_EVIDENCE": numeric,
            "REFERENT": referent,
            "EVIDENCE": _combined_sha256(
                {"NUMERIC_EVIDENCE": numeric, "REFERENT": referent}
            ),
            "QUERY": referent,
        }

    def owner_manifest(self) -> dict[str, object]:
        validate_owner_contract(self)
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "owner_hashes": self.owner_hashes(),
            "transaction_order": ["WORLD", "EVIDENCE", "EXECUTE", "QUERY"],
            "semantic_sharing": {
                "REFERENT": ["EVIDENCE.TARGET_DISTRACTOR", "QUERY.TARGET_DISTRACTOR"]
            },
        }


class SRP1EvidenceView:
    """Expose hybrid evidence logits without registering aliased parameters."""

    def __init__(self, model: SemanticPrimitiveEpistemicMachine) -> None:
        self.model = model

    def eval(self) -> SRP1EvidenceView:
        self.model.eval()
        return self

    def __call__(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_bounds: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_evidence(
            byte_ids,
            attention_mask,
            numeric_bounds,
            symbol_masks,
        )


def _storage_pointers(parameters: Iterable[torch.nn.Parameter]) -> set[int]:
    return {parameter.untyped_storage().data_ptr() for parameter in parameters}


def validate_owner_contract(model: SemanticPrimitiveEpistemicMachine) -> None:
    owners = {
        "WORLD": _storage_pointers(model.source_owner.parameters()),
        "NUMERIC_EVIDENCE": _storage_pointers(
            model.numeric_evidence_owner.parameters()
        ),
        "REFERENT": _storage_pointers(model.referent_owner.parameters()),
    }
    for left_name, left in owners.items():
        for right_name, right in owners.items():
            if left_name < right_name and left & right:
                raise SRP1RuntimeError(
                    f"SRP1 owners {left_name}/{right_name} alias storage"
                )
    if any(parameter.requires_grad for parameter in model.source_owner.parameters()):
        raise SRP1RuntimeError("SRP1 WORLD owner is plastic")
    if any(
        parameter.requires_grad
        for parameter in model.numeric_evidence_owner.parameters()
    ):
        raise SRP1RuntimeError("SRP1 numeric owner is plastic")
    if not all(parameter.requires_grad for parameter in model.referent_owner.parameters()):
        raise SRP1RuntimeError("SRP1 REFERENT owner is not plastic")


def referent_parameters(model: SemanticPrimitiveEpistemicMachine):
    model.freeze_qualified_owners()
    validate_owner_contract(model)
    return tuple(model.referent_owner.parameters())


__all__ = [
    "SRP1Config",
    "SRP1RuntimeError",
    "SemanticPrimitiveEpistemicMachine",
    "SemanticReferentOwner",
    "referent_parameters",
    "validate_owner_contract",
    "warm_start_referent",
]

