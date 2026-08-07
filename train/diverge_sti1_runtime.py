"""Stage-typed semantic ownership for DIVERGE-STI1."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import torch
import torch.nn as nn

from diverge_iem1_runtime import module_state_sha256
from diverge_nve1_runtime import EvidenceCompilerConfig, NaturalEvidenceCompiler
from diverge_rrg1_runtime import RRG1Config, RelationalRoleGrounder
from diverge_tol3_semantic_anchor import LocalSemanticAnchor, TOL3Config


SCHEMA = "shohin-diverge-sti1-runtime-v1"


class STI1RuntimeError(RuntimeError):
    """An STI1 owner or typed routing contract differs."""


class StageTypedInterfaceMachine(nn.Module):
    """Route EVIDENCE to NVE1 and QUERY to the qualified RRG1 owner."""

    def __init__(self, config: RRG1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.source_owner = LocalSemanticAnchor(TOL3Config())
        self.numeric_evidence_owner = NaturalEvidenceCompiler(EvidenceCompilerConfig())
        self.referent_owner = RelationalRoleGrounder(config)

    @property
    def evidence_owner(self) -> NaturalEvidenceCompiler:
        return self.numeric_evidence_owner

    def forward_query(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        symbol_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.referent_owner(byte_ids, attention_mask, symbol_masks)

    def freeze_owners(self) -> None:
        self.requires_grad_(False)

    def owner_hashes(self) -> dict[str, str]:
        return {
            "WORLD": module_state_sha256(self.source_owner),
            "EVIDENCE": module_state_sha256(self.numeric_evidence_owner),
            "QUERY": module_state_sha256(self.referent_owner),
        }

    def owner_manifest(self) -> dict[str, object]:
        validate_owner_contract(self)
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "owner_hashes": self.owner_hashes(),
            "transaction_order": ["WORLD", "EVIDENCE", "EXECUTE", "QUERY"],
            "stage_routes": {
                "WORLD": "TOL3",
                "EVIDENCE": "NVE1_NUMERIC_AND_SYMBOL",
                "QUERY": "RRG1_RELATIONAL_REFERENT",
            },
            "cross_stage_parameter_sharing": False,
            "trainable_parameters": 0,
        }


def _storage_pointers(parameters: Iterable[torch.nn.Parameter]) -> set[int]:
    return {parameter.untyped_storage().data_ptr() for parameter in parameters}


def validate_owner_contract(model: StageTypedInterfaceMachine) -> None:
    owners = {
        "WORLD": _storage_pointers(model.source_owner.parameters()),
        "EVIDENCE": _storage_pointers(model.numeric_evidence_owner.parameters()),
        "QUERY": _storage_pointers(model.referent_owner.parameters()),
    }
    for left_name, left in owners.items():
        for right_name, right in owners.items():
            if left_name < right_name and left & right:
                raise STI1RuntimeError(
                    f"STI1 owners {left_name}/{right_name} alias storage"
                )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise STI1RuntimeError("STI1 contains a plastic parameter")


__all__ = [
    "STI1RuntimeError",
    "StageTypedInterfaceMachine",
    "validate_owner_contract",
]
