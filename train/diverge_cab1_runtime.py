#!/usr/bin/env python3
"""Content-addressed table-relative register bus for DIVERGE-CAB1."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from diverge_eal1_runtime import module_state_sha256, sha256_path
from diverge_jrb1_runtime import (
    JointRegisterBinder,
    JointRegisterBinderConfig,
    tensorize_register_sources,
    tensorize_temporal_without_register_scan,
)


SCHEMA = "shohin-diverge-cab1-runtime-v1"
CHECKPOINT_SCHEMA = "shohin-diverge-cab1-checkpoint-v1"


class CAB1RuntimeError(RuntimeError):
    """A content-addressed register bus violates its frozen contract."""


class ContentAddressedRegisterBus(JointRegisterBinder):
    """Point every owner into one table-relative basis."""

    def __init__(self, config: JointRegisterBinderConfig | None = None) -> None:
        super().__init__(config)
        self.query_token_gate = nn.Linear(self.config.width, 1)

    def forward_query(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        register_ids: torch.Tensor,
        register_mask: torch.Tensor,
    ) -> torch.Tensor:
        source_hidden, register_hidden = self._encode(
            source_ids, source_mask, register_ids, register_mask
        )
        source_key = torch.nn.functional.normalize(
            self.query_projection(source_hidden), dim=-1
        )
        register_key = torch.nn.functional.normalize(
            self.register_projection(register_hidden), dim=-1
        )
        pointer = torch.einsum("bsw,brw->bsr", source_key, register_key)
        pointer = pointer * self.logit_scale.exp().clamp(max=100.0)
        evidence = pointer + self.query_token_gate(source_hidden)
        evidence = evidence.masked_fill(~source_mask.unsqueeze(-1), float("-inf"))
        return torch.logsumexp(evidence.float(), dim=1)

    def record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "state_sha256": module_state_sha256(self),
        }


def load_bus(
    path: Path, expected_sha256: str
) -> tuple[ContentAddressedRegisterBus, Mapping[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise CAB1RuntimeError("CAB1 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise CAB1RuntimeError("CAB1 checkpoint schema differs")
    model = ContentAddressedRegisterBus(JointRegisterBinderConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if module_state_sha256(model) != payload["model_state_sha256"]:
        raise CAB1RuntimeError("CAB1 model state hash differs")
    return model, payload


__all__ = [
    "CAB1RuntimeError",
    "CHECKPOINT_SCHEMA",
    "ContentAddressedRegisterBus",
    "JointRegisterBinderConfig",
    "load_bus",
    "tensorize_register_sources",
    "tensorize_temporal_without_register_scan",
]
