"""Stage-typed composition with a pretrained PQI1 query owner."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

import torch
import torch.nn as nn

from diverge_iem1_runtime import module_state_sha256


SCHEMA = "shohin-diverge-pqi1-composite-v1"


class PQI1CompositeError(RuntimeError):
    """A PQI1 composite owner or isolation contract differs."""


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _storage_pointers(parameters: Iterable[nn.Parameter]) -> set[int]:
    return {parameter.untyped_storage().data_ptr() for parameter in parameters}


class PretrainedStageTypedMachine(nn.Module):
    """Route immutable TOL3/NVE1 owners and one frozen PQI1 query owner."""

    def __init__(
        self,
        source_owner: nn.Module,
        evidence_owner: nn.Module,
        query_owner: nn.Module,
        *,
        tokenizer_sha256: str,
    ) -> None:
        super().__init__()
        if len(tokenizer_sha256) != 64:
            raise PQI1CompositeError("PQI1 tokenizer receipt width differs")
        self.source_owner = source_owner
        self.numeric_evidence_owner = evidence_owner
        self.query_owner = query_owner
        self.tokenizer_sha256 = tokenizer_sha256
        self.requires_grad_(False)
        self._initial_hashes = self.owner_hashes()
        validate_composite(self)

    @property
    def evidence_owner(self) -> nn.Module:
        return self.numeric_evidence_owner

    def forward_query(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.query_owner(byte_ids, attention_mask, symbol_masks)

    def owner_hashes(self) -> dict[str, str]:
        query_module = module_state_sha256(self.query_owner)
        return {
            "WORLD": module_state_sha256(self.source_owner),
            "EVIDENCE": module_state_sha256(self.numeric_evidence_owner),
            "QUERY": _canonical_sha256(
                {
                    "module_state_sha256": query_module,
                    "tokenizer_sha256": self.tokenizer_sha256,
                }
            ),
        }

    def owner_manifest(self) -> dict[str, object]:
        validate_composite(self)
        return {
            "schema": SCHEMA,
            "owner_hashes": self.owner_hashes(),
            "initial_owner_hashes": dict(self._initial_hashes),
            "transaction_order": ["WORLD", "EVIDENCE", "EXECUTE", "QUERY"],
            "stage_routes": {
                "WORLD": "TOL3",
                "EVIDENCE": "NVE1_NUMERIC_AND_SYMBOL",
                "QUERY": "PQI1_PRETRAINED_ANTISYMMETRIC_POINTER",
            },
            "tokenizer_sha256": self.tokenizer_sha256,
            "cross_stage_parameter_sharing": False,
            "trainable_parameters": 0,
        }


def validate_composite(model: PretrainedStageTypedMachine) -> None:
    owners = {
        "WORLD": _storage_pointers(model.source_owner.parameters()),
        "EVIDENCE": _storage_pointers(model.numeric_evidence_owner.parameters()),
        "QUERY": _storage_pointers(model.query_owner.parameters()),
    }
    for left_name, left in owners.items():
        for right_name, right in owners.items():
            if left_name < right_name and left & right:
                raise PQI1CompositeError(
                    f"PQI1 owners {left_name}/{right_name} alias storage"
                )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise PQI1CompositeError("PQI1 composite contains a plastic parameter")


__all__ = [
    "PQI1CompositeError",
    "PretrainedStageTypedMachine",
    "SCHEMA",
    "validate_composite",
]
