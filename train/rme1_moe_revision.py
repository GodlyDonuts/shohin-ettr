"""Dedicated routed micro-experts for model-owned temporal revision."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class RME1Error(RuntimeError):
    """The frozen RME1 contract was violated."""


@dataclass(frozen=True)
class RME1Config:
    hidden_size: int
    num_experts: int
    experts_per_token: int
    controlled_layers: int = 16
    rank: int = 8
    alpha: float = 8.0
    mode: str = "routed"
    revision_experts: int = 4
    revision_top_k: int = 2
    balance_weight: float = 0.01

    def validate(self) -> None:
        if self.mode not in {"routed", "shared"}:
            raise RME1Error("RME1 mode differs")
        if min(
            self.hidden_size,
            self.num_experts,
            self.experts_per_token,
            self.controlled_layers,
            self.rank,
            self.revision_experts,
            self.revision_top_k,
        ) <= 0 or self.alpha <= 0 or self.balance_weight < 0:
            raise RME1Error("RME1 dimensions differ")
        if self.experts_per_token > self.num_experts:
            raise RME1Error("native active experts exceed bank")
        if self.revision_top_k > self.revision_experts:
            raise RME1Error("revision active experts exceed bank")


class RevisionMicroExpertMoE(nn.Module):
    """Frozen native MoE plus a separately routed revision expert bank."""

    INTERVENTIONS = {"normal", "zero", "uniform", "permutation"}

    def __init__(self, base: nn.Module, config: RME1Config) -> None:
        super().__init__()
        config.validate()
        required = ("hidden_dim", "num_experts", "top_k", "norm_topk_prob")
        if not hasattr(base, "gate") or not hasattr(base, "experts"):
            raise RME1Error("native sparse interface differs")
        if any(not hasattr(base.gate, name) for name in required):
            raise RME1Error("native top-k router interface differs")
        if (
            int(base.gate.hidden_dim) != config.hidden_size
            or int(base.gate.num_experts) != config.num_experts
            or int(base.gate.top_k) != config.experts_per_token
        ):
            raise RME1Error("native sparse geometry differs")
        self.base = base
        self.base.requires_grad_(False)
        self.config = config
        hidden, rank = config.hidden_size, config.rank
        device = next(base.parameters()).device
        dtype = next(base.parameters()).dtype
        if config.mode == "routed":
            self.revision_router = nn.Linear(
                hidden, config.revision_experts, bias=False
            ).to(device=device, dtype=dtype)
            nn.init.normal_(self.revision_router.weight, mean=0.0, std=0.01)
            self.adapter_a = nn.Parameter(
                torch.empty(
                    config.revision_experts,
                    rank,
                    hidden,
                    device=device,
                    dtype=dtype,
                )
            )
            self.adapter_b = nn.Parameter(
                torch.zeros(
                    config.revision_experts,
                    hidden,
                    rank,
                    device=device,
                    dtype=dtype,
                )
            )
            nn.init.kaiming_uniform_(self.adapter_a.view(-1, hidden), a=math.sqrt(5))
            self.shared_a = self.shared_b = None
        else:
            self.revision_router = None
            self.register_parameter("adapter_a", None)
            self.register_parameter("adapter_b", None)
            self.shared_a = nn.Linear(hidden, rank, bias=False).to(
                device=device, dtype=dtype
            )
            self.shared_b = nn.Linear(rank, hidden, bias=False).to(
                device=device, dtype=dtype
            )
            nn.init.kaiming_uniform_(self.shared_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.shared_b.weight)
        self.scale = config.alpha / config.rank
        self._intervention = "normal"
        self._balance_loss: torch.Tensor | None = None
        self.reset_receipt()

    def set_code_intervention(self, intervention: str) -> None:
        if intervention not in self.INTERVENTIONS:
            raise RME1Error("RME1 intervention differs")
        if self.config.mode == "shared" and intervention != "normal":
            raise RME1Error("shared control has no revision router")
        self._intervention = intervention

    def reset_receipt(self) -> None:
        self._receipt: dict[str, Any] = {
            "forwards": 0,
            "tokens": 0,
            "probability_sum": None,
            "token_entropy_sum": 0.0,
            "top2_top3_margin_sum": 0.0,
            "residual_norm_sum": 0.0,
            "native_output_norm_sum": 0.0,
            "expert_counts": None,
        }

    def balance_loss(self) -> torch.Tensor:
        if self._balance_loss is None:
            return next(self.parameters()).new_zeros(())
        return self._balance_loss

    def receipt(self) -> dict[str, Any]:
        tokens = int(self._receipt["tokens"])
        if tokens == 0:
            return {"forwards": 0, "tokens": 0}
        result: dict[str, Any] = {
            "forwards": int(self._receipt["forwards"]),
            "tokens": tokens,
            "mean_residual_norm": self._receipt["residual_norm_sum"] / tokens,
            "mean_native_output_norm": self._receipt["native_output_norm_sum"] / tokens,
        }
        if self.config.mode == "routed":
            mean_probability = self._receipt["probability_sum"].float().cpu() / tokens
            load_entropy = -(
                mean_probability * mean_probability.clamp_min(1e-12).log()
            ).sum() / math.log(self.config.revision_experts)
            counts = self._receipt["expert_counts"].to(torch.int64).cpu()
            result.update(
                {
                    "load_entropy": float(load_entropy),
                    "mean_token_entropy_normalized": self._receipt[
                        "token_entropy_sum"
                    ]
                    / tokens,
                    "mean_top2_top3_logit_margin": self._receipt[
                        "top2_top3_margin_sum"
                    ]
                    / tokens,
                    "active_revision_experts": int((counts > 0).sum()),
                    "revision_expert_counts": counts.tolist(),
                }
            )
        return result

    def _record(
        self,
        probabilities: torch.Tensor | None,
        indices: torch.Tensor | None,
        logits: torch.Tensor | None,
        native_output: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            tokens = int(native_output.shape[0])
            self._receipt["forwards"] += 1
            self._receipt["tokens"] += tokens
            self._receipt["residual_norm_sum"] += float(
                residual.float().norm(dim=-1).sum().cpu()
            )
            self._receipt["native_output_norm_sum"] += float(
                native_output.float().norm(dim=-1).sum().cpu()
            )
            if probabilities is None or indices is None or logits is None:
                return
            entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(dim=-1) / math.log(self.config.revision_experts)
            top3 = logits.float().topk(self.config.revision_top_k + 1, dim=-1).values
            margin = top3[:, self.config.revision_top_k - 1] - top3[:, self.config.revision_top_k]
            counts = torch.bincount(
                indices.reshape(-1), minlength=self.config.revision_experts
            )
            probability_sum = probabilities.sum(dim=0)
            self._receipt["token_entropy_sum"] += float(entropy.sum().cpu())
            self._receipt["top2_top3_margin_sum"] += float(margin.sum().cpu())
            self._receipt["probability_sum"] = (
                probability_sum.detach()
                if self._receipt["probability_sum"] is None
                else self._receipt["probability_sum"] + probability_sum.detach()
            )
            self._receipt["expert_counts"] = (
                counts.detach()
                if self._receipt["expert_counts"] is None
                else self._receipt["expert_counts"] + counts.detach()
            )

    def _routed_residual(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.revision_router is not None
        assert self.adapter_a is not None and self.adapter_b is not None
        logits = self.revision_router(hidden)
        probabilities = logits.float().softmax(dim=-1)
        self._balance_loss = self.config.revision_experts * probabilities.mean(dim=0).square().sum()
        if self._intervention == "uniform":
            logits = torch.zeros_like(logits)
            probabilities = logits.float().softmax(dim=-1)
        weights, indices = probabilities.topk(self.config.revision_top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        bank_indices = indices
        if self._intervention == "permutation":
            bank_indices = (indices + 1) % self.config.revision_experts
        selected_a = self.adapter_a[bank_indices]
        selected_b = self.adapter_b[bank_indices]
        low = torch.einsum("th,tkrh->tkr", hidden, selected_a)
        components = torch.einsum("tkr,tkhr->tkh", low, selected_b)
        residual = (weights.unsqueeze(-1) * components.float()).sum(dim=1).to(hidden.dtype)
        if self._intervention == "zero":
            residual = torch.zeros_like(residual)
        return residual, probabilities, indices, logits

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden_size = hidden_states.shape
        flattened = hidden_states.reshape(-1, hidden_size)
        _, native_weights, native_indices = self.base.gate(flattened)
        native_output = self.base.experts(flattened, native_indices, native_weights)
        if self.config.mode == "routed":
            residual, probabilities, indices, logits = self._routed_residual(flattened)
        else:
            assert self.shared_a is not None and self.shared_b is not None
            residual = self.shared_b(self.shared_a(flattened))
            probabilities = indices = logits = None
            self._balance_loss = residual.new_zeros(())
        residual = residual * self.scale
        self._record(probabilities, indices, logits, native_output, residual)
        return (native_output + residual.to(native_output.dtype)).reshape(
            batch, sequence, hidden_size
        )


def install_rme1_blocks(layers: Any, config: RME1Config) -> list[RevisionMicroExpertMoE]:
    if len(layers) < config.controlled_layers:
        raise RME1Error("backbone has too few sparse layers")
    wrappers = []
    for layer in layers[-config.controlled_layers :]:
        wrapper = RevisionMicroExpertMoE(layer.mlp, config)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    return wrappers


class RME1ProductModel(nn.Module):
    """Frozen OLMoE with separately routed revision micro-experts."""

    def __init__(self, backbone: nn.Module, config: RME1Config, *, draft_control: str = "normal") -> None:
        super().__init__()
        if draft_control not in {"normal", "draft_unavailable"}:
            raise RME1Error("draft control differs")
        from hf_product_reasoning_train import resolve_product_backbone_layout

        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.text_model, self.lm_head, hidden, self.backbone_layout = resolve_product_backbone_layout(backbone)
        if hidden != config.hidden_size:
            raise RME1Error("backbone width differs")
        self.config = config
        self.draft_control = draft_control
        self.blocks = nn.ModuleList(install_rme1_blocks(self.text_model.layers, config))
        self._generation_prompt_attention: torch.Tensor | None = None
        self._generation_position_ids: torch.Tensor | None = None

    def sequence_workspace_slots(self) -> int:
        return 0

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_name_sha256(self) -> str:
        names = sorted(name for name, parameter in self.named_parameters() if parameter.requires_grad)
        return hashlib.sha256("\n".join(names).encode()).hexdigest()

    def protected_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not parameter.requires_grad and (".base.gate." in name or ".base.experts." in name)
        )

    def set_code_intervention(self, intervention: str) -> None:
        for block in self.blocks:
            block.set_code_intervention(intervention)

    def reset_routing_receipt(self) -> None:
        for block in self.blocks:
            block.reset_receipt()

    def routing_receipt(self) -> dict[str, Any]:
        return {"controlled_layers": len(self.blocks), "layers": [block.receipt() for block in self.blocks]}

    def forward_batch(self, prompt_rows, response_rows, pad_token_id, prompt_attention_rows):
        from hf_product_reasoning_train import pack_training_embeddings

        masks = prompt_attention_rows if self.draft_control == "draft_unavailable" else None
        inputs, attention, labels, charged = pack_training_embeddings(
            self.text_model.embed_tokens,
            prompt_rows,
            response_rows,
            None,
            pad_token_id,
            prompt_attention_rows=masks,
        )
        outputs = self.text_model(inputs_embeds=inputs, attention_mask=attention, use_cache=False)
        logits = self.lm_head(outputs.last_hidden_state)
        language_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        balance_loss = torch.stack([block.balance_loss() for block in self.blocks]).mean()
        total = language_loss + self.config.balance_weight * balance_loss
        return total, {
            "language_loss": float(language_loss.detach()),
            "balance_loss": float(balance_loss.detach()),
            "charged_tokens": float(charged),
        }

    def prepare_generation_draft_attention(self, tokenizer, rendered, input_ids, attention_mask) -> None:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(~attention_mask.bool(), 0)
        self._generation_position_ids = position_ids
        if self.draft_control == "normal":
            self._generation_prompt_attention = attention_mask
            return
        from ttr1_revision import tokenize_with_draft_mask

        masked = attention_mask.clone()
        for row_index, prompt in enumerate(rendered):
            token_ids, draft_attention, _ = tokenize_with_draft_mask(tokenizer, prompt)
            positions = attention_mask[row_index].bool().nonzero().flatten()
            if input_ids[row_index, positions].tolist() != token_ids:
                raise RME1Error("generation prompt tokenization differs")
            masked[row_index, positions] = torch.tensor(
                draft_attention, device=masked.device, dtype=masked.dtype
            )
        self._generation_prompt_attention = masked

    def generation_position_ids(self) -> torch.Tensor:
        if self._generation_position_ids is None:
            raise RME1Error("generation positions are absent")
        return self._generation_position_ids

    def generation_embeddings(self, prompt_ids, prompt_attention):
        attention = self._generation_prompt_attention
        if attention is None or attention.shape != prompt_attention.shape:
            raise RME1Error("generation draft attention is absent")
        return self.text_model.embed_tokens(prompt_ids), attention
