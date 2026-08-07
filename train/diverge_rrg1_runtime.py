"""Length-free relational role grounding for DIVERGE-RRG1."""

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


SCHEMA = "shohin-diverge-rrg1-runtime-v1"
ReferentControl = Literal["normal", "role_slot_swap", "marker_delete"]


class RRG1RuntimeError(RuntimeError):
    """An RRG1 owner or canonical referent tensor violates the contract."""


@dataclass(frozen=True, slots=True)
class RRG1Config:
    width: int = 192
    layers: int = 2
    max_bytes: int = MAX_QUERY_BYTES

    def validate(self) -> None:
        if self.width != 192 or self.layers != 2 or self.width % 2:
            raise RRG1RuntimeError("RRG1 referent geometry differs")
        if self.max_bytes != MAX_QUERY_BYTES:
            raise RRG1RuntimeError("RRG1 source width differs")


class RelationalRoleGrounder(nn.Module):
    """Collapse names, encode once, and jointly match role slots to mentions."""

    def __init__(self, config: RRG1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.width
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, width)
        self.mention_marker = nn.Parameter(torch.empty(width))
        self.role_slots = nn.Parameter(torch.empty(2, width))
        nn.init.normal_(self.mention_marker, std=0.02)
        nn.init.normal_(self.role_slots, std=0.02)
        self.encoder = nn.GRU(
            input_size=width,
            hidden_size=width // 2,
            num_layers=config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.output_norm = nn.LayerNorm(width)
        self.context = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, width),
            nn.GELU(),
        )
        self.mention_projection = nn.Sequential(
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, width),
            nn.GELU(),
        )
        self.compatibility = nn.Sequential(
            nn.LayerNorm(width * 4),
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def canonicalize(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
        *,
        marker_delete: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove every mention byte and retain one anonymous token per span."""

        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or byte_ids.shape[1] != self.config.max_bytes
            or byte_ids.dtype != torch.long
            or attention_mask.dtype != torch.bool
            or symbol_masks.shape != (byte_ids.shape[0], 2, self.config.max_bytes)
            or symbol_masks.dtype != torch.bool
        ):
            raise RRG1RuntimeError("RRG1 referent tensor interface differs")
        if not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise RRG1RuntimeError("RRG1 CLS differs")
        if torch.any(symbol_masks & ~attention_mask[:, None, :]):
            raise RRG1RuntimeError("RRG1 mention exceeds the source")
        if torch.any(symbol_masks[:, 0] & symbol_masks[:, 1]):
            raise RRG1RuntimeError("RRG1 mention groups overlap")
        if torch.any(symbol_masks.sum(dim=-1) < 1):
            raise RRG1RuntimeError("RRG1 mention group is empty")

        previous = torch.zeros_like(symbol_masks)
        previous[:, :, 1:] = symbol_masks[:, :, :-1]
        starts = symbol_masks & ~previous
        if torch.any(starts.sum(dim=-1) < 1):
            raise RRG1RuntimeError("RRG1 mention starts are absent")
        union = symbol_masks.any(dim=1)
        start_any = starts.any(dim=1)
        keep = attention_mask & (~union | start_any)
        lengths = keep.sum(dim=1)
        if torch.any(lengths < 3):
            raise RRG1RuntimeError("RRG1 canonical source is empty")
        destination = keep.long().cumsum(dim=1) - 1
        destination = destination.clamp_min(0)

        embedded = self.embedding(byte_ids)
        marker = (
            torch.zeros_like(self.mention_marker)
            if marker_delete
            else self.mention_marker
        )
        source = torch.where(
            start_any.unsqueeze(-1),
            marker.view(1, 1, -1),
            embedded,
        )
        compact = torch.zeros_like(embedded)
        compact.scatter_add_(
            1,
            destination.unsqueeze(-1).expand_as(source),
            source * keep.unsqueeze(-1).to(source.dtype),
        )
        compact_groups = torch.zeros_like(symbol_masks, dtype=torch.long)
        compact_groups.scatter_add_(
            2,
            destination[:, None, :].expand_as(symbol_masks),
            starts.long(),
        )
        compact_group_mask = compact_groups.gt(0)
        positions = torch.arange(
            self.config.max_bytes, device=byte_ids.device
        ).unsqueeze(0)
        compact_attention = positions < lengths.unsqueeze(1)
        if not torch.equal(
            compact_group_mask.sum(dim=-1), starts.sum(dim=-1)
        ):
            raise RRG1RuntimeError("RRG1 canonical mention count differs")
        if torch.any(compact_group_mask & ~compact_attention[:, None, :]):
            raise RRG1RuntimeError("RRG1 canonical mention is out of bounds")
        return compact, compact_attention, compact_group_mask

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
        *,
        control: ReferentControl = "normal",
    ) -> torch.Tensor:
        if control not in ("normal", "role_slot_swap", "marker_delete"):
            raise RRG1RuntimeError(f"unknown RRG1 control: {control}")
        embedded, compact_attention, compact_groups = self.canonicalize(
            byte_ids,
            attention_mask,
            symbol_masks,
            marker_delete=control == "marker_delete",
        )
        lengths = compact_attention.sum(dim=1)
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
        attention = compact_attention.unsqueeze(-1).to(hidden.dtype)
        global_hidden = (hidden * attention).sum(dim=1) / attention.sum(dim=1)
        cls_hidden = hidden[:, 0]
        context = self.context(torch.cat((cls_hidden, global_hidden), dim=-1))
        group_weights = compact_groups.to(hidden.dtype)
        mention_hidden = torch.einsum(
            "bgs,bsw->bgw", group_weights, hidden
        ) / group_weights.sum(dim=-1, keepdim=True)
        expanded_context = context[:, None, :].expand(-1, 2, -1)
        mention = self.mention_projection(
            torch.cat(
                (
                    mention_hidden,
                    expanded_context,
                    mention_hidden * expanded_context,
                    mention_hidden - expanded_context,
                ),
                dim=-1,
            )
        )
        roles = self.role_slots
        if control == "role_slot_swap":
            roles = roles.flip(0)
        mention_matrix = mention[:, :, None, :].expand(-1, -1, 2, -1)
        role_matrix = roles[None, None, :, :].expand(
            mention.shape[0], 2, -1, -1
        )
        context_matrix = context[:, None, None, :].expand(-1, 2, 2, -1)
        features = torch.cat(
            (
                mention_matrix,
                role_matrix,
                mention_matrix * role_matrix,
                context_matrix,
            ),
            dim=-1,
        )
        return self.compatibility(features).squeeze(-1).float()


def permutation_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[1:] != (2, 2):
        raise RRG1RuntimeError("RRG1 role logits differ")
    return torch.stack(
        (logits[:, 0, 0] + logits[:, 1, 1], logits[:, 0, 1] + logits[:, 1, 0]),
        dim=-1,
    )


def permutation_targets(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise RRG1RuntimeError("RRG1 role targets differ")
    valid = targets.eq(torch.tensor((0, 1), device=targets.device)).all(dim=1) | targets.eq(
        torch.tensor((1, 0), device=targets.device)
    ).all(dim=1)
    if not torch.all(valid):
        raise RRG1RuntimeError("RRG1 target is not a complete permutation")
    return targets[:, 0]


def _combined_sha256(payload: Mapping[str, str]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class RelationalReferentMachine(nn.Module):
    """Qualified WORLD/numeric owners plus the RRG1 referent owner."""

    def __init__(self, config: RRG1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.source_owner = LocalSemanticAnchor(TOL3Config())
        self.numeric_evidence_owner = NaturalEvidenceCompiler(EvidenceCompilerConfig())
        self.referent_owner = RelationalRoleGrounder(config)

    @property
    def evidence_owner(self) -> RRG1EvidenceView:
        return RRG1EvidenceView(self)

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
        return numeric_logits, self.referent_owner(
            byte_ids, attention_mask, symbol_masks
        )

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
            "candidate_canonicalization": "ANONYMOUS_MENTION_BEFORE_SINGLE_ENCODER",
            "assignment": "HARD_COMPLETE_TWO_WAY_PERMUTATION",
        }


class RRG1EvidenceView:
    """Expose hybrid evidence logits without registering aliased parameters."""

    def __init__(self, model: RelationalReferentMachine) -> None:
        self.model = model

    def eval(self) -> RRG1EvidenceView:
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


def validate_owner_contract(model: RelationalReferentMachine) -> None:
    owners = {
        "WORLD": _storage_pointers(model.source_owner.parameters()),
        "NUMERIC_EVIDENCE": _storage_pointers(model.numeric_evidence_owner.parameters()),
        "REFERENT": _storage_pointers(model.referent_owner.parameters()),
    }
    for left_name, left in owners.items():
        for right_name, right in owners.items():
            if left_name < right_name and left & right:
                raise RRG1RuntimeError(
                    f"RRG1 owners {left_name}/{right_name} alias storage"
                )
    if any(parameter.requires_grad for parameter in model.source_owner.parameters()):
        raise RRG1RuntimeError("RRG1 WORLD owner is plastic")
    if any(
        parameter.requires_grad for parameter in model.numeric_evidence_owner.parameters()
    ):
        raise RRG1RuntimeError("RRG1 numeric owner is plastic")
    if not all(parameter.requires_grad for parameter in model.referent_owner.parameters()):
        raise RRG1RuntimeError("RRG1 REFERENT owner is not plastic")


def referent_parameters(model: RelationalReferentMachine):
    model.freeze_qualified_owners()
    validate_owner_contract(model)
    return tuple(model.referent_owner.parameters())


__all__ = [
    "RRG1Config",
    "RRG1RuntimeError",
    "ReferentControl",
    "RelationalReferentMachine",
    "RelationalRoleGrounder",
    "permutation_scores",
    "permutation_targets",
    "referent_parameters",
    "validate_owner_contract",
]
