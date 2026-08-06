"""Counterfactual candidate-relative semantic ownership for DIVERGE-CCR1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable, Literal, Mapping

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_VOCAB_SIZE
from diverge_iem1_runtime import CLS_ID, MAX_QUERY_BYTES, module_state_sha256
from diverge_nve1_runtime import EvidenceCompilerConfig, NaturalEvidenceCompiler
from diverge_tol3_semantic_anchor import LocalSemanticAnchor, TOL3Config


SCHEMA = "shohin-diverge-ccr1-runtime-v1"
MarkerControl = Literal["normal", "swap", "delete"]


class CCR1RuntimeError(RuntimeError):
    """A CCR1 owner or tensor violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class CCR1Config:
    width: int = 192
    layers: int = 2
    max_bytes: int = MAX_QUERY_BYTES

    def validate(self) -> None:
        if self.width != 192 or self.layers != 2 or self.width % 2:
            raise CCR1RuntimeError("CCR1 referent geometry differs")
        if self.max_bytes != MAX_QUERY_BYTES:
            raise CCR1RuntimeError("CCR1 source width differs")


class CounterfactualCandidateReferent(nn.Module):
    """Score each referent after replacing names by SELF/OTHER markers."""

    def __init__(self, config: CCR1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.width
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, width)
        self.self_marker = nn.Parameter(torch.empty(width))
        self.other_marker = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.self_marker, std=0.02)
        nn.init.normal_(self.other_marker, std=0.02)
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

    def _markers(self, control: MarkerControl) -> tuple[torch.Tensor, torch.Tensor]:
        if control == "normal":
            return self.self_marker, self.other_marker
        if control == "swap":
            return self.other_marker, self.self_marker
        if control == "delete":
            neutral = 0.5 * (self.self_marker + self.other_marker)
            return neutral, neutral
        raise CCR1RuntimeError(f"unknown CCR1 marker control: {control}")

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
        *,
        marker_control: MarkerControl = "normal",
    ) -> torch.Tensor:
        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or byte_ids.shape[1] != self.config.max_bytes
            or byte_ids.dtype != torch.long
            or attention_mask.dtype != torch.bool
            or symbol_masks.shape != (byte_ids.shape[0], 2, self.config.max_bytes)
        ):
            raise CCR1RuntimeError("CCR1 referent tensor interface differs")
        lengths = attention_mask.sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise CCR1RuntimeError("CCR1 referent mask or CLS differs")
        if torch.any(symbol_masks.sum(dim=-1) < 1):
            raise CCR1RuntimeError("CCR1 referent group is empty")
        if torch.any(symbol_masks[:, 0] & symbol_masks[:, 1]):
            raise CCR1RuntimeError("CCR1 referent groups overlap")

        base = self.embedding(byte_ids)
        self_marker, other_marker = self._markers(marker_control)
        conditioned = []
        for candidate in range(2):
            candidate_hidden = base.clone()
            self_mask = symbol_masks[:, candidate]
            other_mask = symbol_masks[:, 1 - candidate]
            candidate_hidden = torch.where(
                self_mask.unsqueeze(-1), self_marker.view(1, 1, -1), candidate_hidden
            )
            candidate_hidden = torch.where(
                other_mask.unsqueeze(-1), other_marker.view(1, 1, -1), candidate_hidden
            )
            conditioned.append(candidate_hidden)
        stacked = torch.stack(conditioned, dim=1)
        batch, candidates, sequence, width = stacked.shape
        flattened = stacked.reshape(batch * candidates, sequence, width)
        repeated_lengths = lengths.unsqueeze(1).expand(-1, candidates).reshape(-1)
        packed = pack_padded_sequence(
            flattened,
            repeated_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=self.config.max_bytes,
        )
        hidden = self.output_norm(hidden).reshape(batch, candidates, sequence, width)
        attention = attention_mask[:, None, :, None].to(hidden.dtype)
        global_hidden = (hidden * attention).sum(dim=2) / attention.sum(dim=2)
        candidate_masks = symbol_masks[:, :, :, None].to(hidden.dtype)
        self_hidden = (hidden * candidate_masks).sum(dim=2) / candidate_masks.sum(dim=2)
        other_masks = symbol_masks.flip(1)[:, :, :, None].to(hidden.dtype)
        other_hidden = (hidden * other_masks).sum(dim=2) / other_masks.sum(dim=2)
        if marker_control == "delete":
            union = symbol_masks.any(dim=1)[:, None, :, None].to(hidden.dtype)
            union = union.expand(-1, candidates, -1, -1)
            mention_hidden = (hidden * union).sum(dim=2) / union.sum(dim=2)
            self_hidden = mention_hidden
            other_hidden = mention_hidden
        cls_hidden = hidden[:, :, 0]
        features = torch.cat(
            (
                cls_hidden,
                global_hidden,
                self_hidden,
                self_hidden - other_hidden,
                self_hidden * other_hidden,
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


def _combined_sha256(payload: Mapping[str, str]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class CounterfactualReferentMachine(nn.Module):
    """Qualified WORLD/numeric owners plus the CCR1 referent owner."""

    def __init__(self, config: CCR1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.source_owner = LocalSemanticAnchor(TOL3Config())
        self.numeric_evidence_owner = NaturalEvidenceCompiler(EvidenceCompilerConfig())
        self.referent_owner = CounterfactualCandidateReferent(config)

    @property
    def evidence_owner(self) -> CCR1EvidenceView:
        return CCR1EvidenceView(self)

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
        return numeric_logits, self.referent_owner(byte_ids, attention_mask, symbol_masks)

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
            "candidate_canonicalization": "SELF_OTHER_BEFORE_ENCODER",
        }


class CCR1EvidenceView:
    """Expose hybrid evidence logits without registering aliased parameters."""

    def __init__(self, model: CounterfactualReferentMachine) -> None:
        self.model = model

    def eval(self) -> CCR1EvidenceView:
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
            byte_ids, attention_mask, numeric_bounds, symbol_masks
        )


def _storage_pointers(parameters: Iterable[torch.nn.Parameter]) -> set[int]:
    return {parameter.untyped_storage().data_ptr() for parameter in parameters}


def validate_owner_contract(model: CounterfactualReferentMachine) -> None:
    owners = {
        "WORLD": _storage_pointers(model.source_owner.parameters()),
        "NUMERIC_EVIDENCE": _storage_pointers(model.numeric_evidence_owner.parameters()),
        "REFERENT": _storage_pointers(model.referent_owner.parameters()),
    }
    for left_name, left in owners.items():
        for right_name, right in owners.items():
            if left_name < right_name and left & right:
                raise CCR1RuntimeError(
                    f"CCR1 owners {left_name}/{right_name} alias storage"
                )
    if any(parameter.requires_grad for parameter in model.source_owner.parameters()):
        raise CCR1RuntimeError("CCR1 WORLD owner is plastic")
    if any(
        parameter.requires_grad for parameter in model.numeric_evidence_owner.parameters()
    ):
        raise CCR1RuntimeError("CCR1 numeric owner is plastic")
    if not all(parameter.requires_grad for parameter in model.referent_owner.parameters()):
        raise CCR1RuntimeError("CCR1 REFERENT owner is not plastic")


def referent_parameters(model: CounterfactualReferentMachine):
    model.freeze_qualified_owners()
    validate_owner_contract(model)
    return tuple(model.referent_owner.parameters())


__all__ = [
    "CCR1Config",
    "CCR1RuntimeError",
    "CounterfactualCandidateReferent",
    "CounterfactualReferentMachine",
    "referent_parameters",
    "validate_owner_contract",
]
