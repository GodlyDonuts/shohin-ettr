"""Host-agnostic trainable residuals after frozen decoder MLP/MoE blocks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch
import torch.nn as nn


class SharedPostMLPError(RuntimeError):
    """The shared post-MLP revision contract was violated."""


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise SharedPostMLPError("trainable residual state is empty")
    return state


def trainable_state_sha256(state: dict[str, torch.Tensor]) -> str:
    if not state:
        raise SharedPostMLPError("trainable residual state is empty")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SharedPostMLPConfig:
    hidden_size: int
    controlled_layers: int = 16
    rank: int = 18
    alpha: float = 18.0

    def validate(self) -> None:
        if (
            min(self.hidden_size, self.controlled_layers, self.rank) <= 0
            or self.alpha <= 0
        ):
            raise SharedPostMLPError("shared post-MLP dimensions differ")


class SharedPostMLPResidual(nn.Module):
    """Frozen MLP/MoE followed by one low-rank tokenwise residual."""

    def __init__(self, base: nn.Module, config: SharedPostMLPConfig) -> None:
        super().__init__()
        config.validate()
        self.base = base
        self.base.requires_grad_(False)
        self.config = config
        device = next(base.parameters()).device
        # Keep the small trainable surface in FP32.  In particular, the frozen
        # commit LR is 2e-6, far below BF16's ULP at ordinary initialized
        # adapter magnitudes; BF16 master parameters can therefore turn valid
        # optimizer steps into exact no-ops.  CUDA forwards remain BF16 under
        # autocast while AdamW updates these FP32 masters.
        self.adapter_a = nn.Linear(config.hidden_size, config.rank, bias=False).to(
            device=device, dtype=torch.float32
        )
        self.adapter_b = nn.Linear(config.rank, config.hidden_size, bias=False).to(
            device=device, dtype=torch.float32
        )
        nn.init.kaiming_uniform_(self.adapter_a.weight, a=5**0.5)
        nn.init.zeros_(self.adapter_b.weight)
        self.scale = config.alpha / config.rank
        self.reset_receipt()

    def reset_receipt(self) -> None:
        self._tokens = 0
        self._residual_norm = 0.0
        self._native_norm = 0.0

    def receipt(self) -> dict[str, float | int]:
        if not self._tokens:
            return {"tokens": 0}
        return {
            "tokens": self._tokens,
            "mean_residual_norm": self._residual_norm / self._tokens,
            "mean_native_output_norm": self._native_norm / self._tokens,
        }

    def forward(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        native = self.base(hidden_states, *args, **kwargs)
        if not isinstance(native, torch.Tensor) or native.shape != hidden_states.shape:
            raise SharedPostMLPError("base MLP output geometry differs")
        if hidden_states.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                residual = self.adapter_b(self.adapter_a(hidden_states)) * self.scale
        else:
            residual = (
                self.adapter_b(self.adapter_a(hidden_states.to(torch.float32)))
                * self.scale
            )
        with torch.no_grad():
            tokens = int(native.numel() // native.shape[-1])
            self._tokens += tokens
            self._residual_norm += float(residual.float().norm(dim=-1).sum().cpu())
            self._native_norm += float(native.float().norm(dim=-1).sum().cpu())
        return native + residual.to(native.dtype)


class SharedPostMLPProductModel(nn.Module):
    """Frozen decoder backbone with revision residuals in its final layers."""

    def __init__(
        self,
        backbone: nn.Module,
        config: SharedPostMLPConfig,
        *,
        draft_control: str = "normal",
    ) -> None:
        super().__init__()
        if draft_control not in {"normal", "draft_unavailable"}:
            raise SharedPostMLPError("draft control differs")
        from hf_product_reasoning_train import resolve_product_backbone_layout

        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.text_model, self.lm_head, hidden, self.backbone_layout = (
            resolve_product_backbone_layout(backbone)
        )
        if (
            hidden != config.hidden_size
            or len(self.text_model.layers) < config.controlled_layers
        ):
            raise SharedPostMLPError("backbone geometry differs")
        self.config = config
        self.draft_control = draft_control
        blocks = []
        for layer in self.text_model.layers[-config.controlled_layers :]:
            block = SharedPostMLPResidual(layer.mlp, config)
            layer.mlp = block
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self._generation_prompt_attention: torch.Tensor | None = None
        self._generation_position_ids: torch.Tensor | None = None
        self._generation_prompt_ids: torch.Tensor | None = None

    def sequence_workspace_slots(self) -> int:
        return 0

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_parameter_name_sha256(self) -> str:
        names = sorted(name for name, p in self.named_parameters() if p.requires_grad)
        return hashlib.sha256("\n".join(names).encode()).hexdigest()

    def reset_routing_receipt(self) -> None:
        for block in self.blocks:
            block.reset_receipt()

    def routing_receipt(self) -> dict[str, Any]:
        return {
            "controlled_layers": len(self.blocks),
            "layers": [block.receipt() for block in self.blocks],
        }

    def prepare_generation_draft_attention(
        self,
        tokenizer: Any,
        rendered: list[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(~attention_mask.bool(), 0)
        self._generation_position_ids = position_ids
        self._generation_prompt_ids = input_ids.detach().clone()
        if self.draft_control == "normal":
            self._generation_prompt_attention = attention_mask
            return
        from ttr1_revision import tokenize_with_draft_mask

        masked = attention_mask.clone()
        for row_index, prompt in enumerate(rendered):
            token_ids, draft_attention, _ = tokenize_with_draft_mask(tokenizer, prompt)
            positions = attention_mask[row_index].bool().nonzero().flatten()
            if input_ids[row_index, positions].tolist() != token_ids:
                raise SharedPostMLPError("generation prompt tokenization differs")
            masked[row_index, positions] = torch.tensor(
                draft_attention, device=masked.device, dtype=masked.dtype
            )
        self._generation_prompt_attention = masked

    def generation_position_ids(self) -> torch.Tensor:
        if self._generation_position_ids is None:
            raise SharedPostMLPError("generation positions are absent")
        return self._generation_position_ids

    def generation_embeddings(
        self, prompt_ids: torch.Tensor, prompt_attention: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self._generation_prompt_attention
        prompt_ids_receipt = self._generation_prompt_ids
        if (
            attention is None
            or attention.shape != prompt_attention.shape
            or prompt_ids_receipt is None
            or prompt_ids_receipt.shape != prompt_ids.shape
            or not torch.equal(prompt_ids_receipt, prompt_ids)
        ):
            raise SharedPostMLPError("generation draft attention is absent")
        return self.text_model.embed_tokens(prompt_ids), attention
