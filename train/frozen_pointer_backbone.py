"""Load frozen pointer-compiler backbones from admitted Shohin schemas.

The historical pointer compiler predates the SmolLM2 control and assumes the
plain flagship ``{cfg, model}`` checkpoint.  This module keeps the compiler
itself unchanged while admitting either that protected checkpoint or an ETTR
parent whose language backbone is stored under ``base.*`` tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from model import GPT, GPTConfig


@dataclass(frozen=True, slots=True)
class FrozenPointerBackboneReceipt:
    checkpoint_format: str
    base_step: int | None
    initialization: str | None
    base_import: Mapping[str, Any] | None
    base_rms_norm_eps: float | None


def _load_state(model: GPT, state: Mapping[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Tied checkpoints occasionally serialize only the token embedding.  The
    # language head is the same parameter object, so this is not missing data.
    allowed_missing = {"head.weight"} if model.cfg.tie_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(
            "frozen backbone tensor mismatch missing={} unexpected={}".format(
                sorted(set(missing) - allowed_missing), sorted(unexpected)
            )
        )


def load_frozen_pointer_backbone(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[GPT, GPTConfig, FrozenPointerBackboneReceipt]:
    """Load a plain Shohin checkpoint or an exact imported ETTR parent."""

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ValueError("unsupported frozen backbone payload")

    if isinstance(payload.get("cfg"), Mapping):
        config = GPTConfig(**dict(payload["cfg"]))
        model = GPT(config)
        _load_state(model, payload["model"])
        receipt = FrozenPointerBackboneReceipt(
            checkpoint_format="plain-shohin",
            base_step=(int(payload["step"]) if payload.get("step") is not None else None),
            initialization=(
                str(payload["initialization"])
                if payload.get("initialization") is not None
                else None
            ),
            base_import=None,
            base_rms_norm_eps=None,
        )
    elif isinstance(payload.get("base_config"), Mapping):
        config = GPTConfig(**dict(payload["base_config"]))
        model = GPT(config)
        epsilon = payload.get("base_rms_norm_eps")
        if epsilon is not None:
            model.set_rms_norm_eps(float(epsilon))
        base_state = {
            name.removeprefix("base."): tensor
            for name, tensor in payload["model"].items()
            if name.startswith("base.")
        }
        if not base_state:
            raise ValueError("ETTR parent contains no base.* tensors")
        _load_state(model, base_state)
        initialization = payload.get("initialization")
        initialization_name = (
            str(initialization.get("mode"))
            if isinstance(initialization, Mapping) and initialization.get("mode") is not None
            else str(initialization)
            if isinstance(initialization, str)
            else None
        )
        base_import = payload.get("base_import")
        receipt = FrozenPointerBackboneReceipt(
            checkpoint_format="ettr-parent-base",
            base_step=None,
            initialization=initialization_name,
            base_import=(dict(base_import) if isinstance(base_import, Mapping) else None),
            base_rms_norm_eps=(float(epsilon) if epsilon is not None else None),
        )
    else:
        raise ValueError("checkpoint has neither cfg nor base_config")

    del payload
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, config, receipt


__all__ = ["FrozenPointerBackboneReceipt", "load_frozen_pointer_backbone"]
