"""Selected-expert residual transforms for MoE temporal revision."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SER1Error(RuntimeError):
    """The frozen SER1 sparse-block contract was violated."""


@dataclass(frozen=True)
class SER1Config:
    hidden_size: int
    num_experts: int
    experts_per_token: int
    controlled_layers: int = 16
    rank: int = 1
    alpha: float = 1.0
    mode: str = "selected_expert"

    def validate(self) -> None:
        if self.mode not in {"selected_expert", "shared"}:
            raise SER1Error("SER1 mode differs")
        if min(
            self.hidden_size,
            self.num_experts,
            self.experts_per_token,
            self.controlled_layers,
            self.rank,
        ) <= 0 or self.alpha <= 0:
            raise SER1Error("SER1 dimensions differ")
        if self.experts_per_token > self.num_experts:
            raise SER1Error("active experts exceed the expert bank")


def _bank_diagnostics(adapter_a: torch.Tensor, adapter_b: torch.Tensor) -> dict[str, float]:
    vectors = torch.cat(
        (adapter_a.float().flatten(1), adapter_b.float().flatten(1)), dim=1
    )
    normalized = F.normalize(vectors, dim=-1, eps=1e-12)
    cosine = normalized @ normalized.T
    mask = ~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
    gram = vectors @ vectors.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    return {
        "pairwise_cosine_mean": float(cosine[mask].mean()),
        "pairwise_cosine_abs_mean": float(cosine[mask].abs().mean()),
        "effective_rank": float(effective_rank),
    }


class SelectedExpertResidualMoE(nn.Module):
    """Frozen native MoE plus whole residual transforms owned by experts."""

    INTERVENTIONS = {"normal", "zero", "mean", "permutation"}

    def __init__(self, base: nn.Module, config: SER1Config) -> None:
        super().__init__()
        config.validate()
        required = ("hidden_dim", "num_experts", "top_k", "norm_topk_prob")
        if not hasattr(base, "gate") or not hasattr(base, "experts"):
            raise SER1Error("sparse MoE block interface differs")
        if any(not hasattr(base.gate, name) for name in required):
            raise SER1Error("native top-k router interface differs")
        if (
            int(base.gate.hidden_dim) != config.hidden_size
            or int(base.gate.num_experts) != config.num_experts
            or int(base.gate.top_k) != config.experts_per_token
        ):
            raise SER1Error("native sparse geometry differs")
        self.base = base
        self.base.requires_grad_(False)
        self.config = config
        hidden, rank = config.hidden_size, config.rank
        device = next(base.parameters()).device
        dtype = next(base.parameters()).dtype
        if config.mode == "selected_expert":
            self.adapter_a = nn.Parameter(
                torch.empty(config.num_experts, rank, hidden, device=device, dtype=dtype)
            )
            self.adapter_b = nn.Parameter(
                torch.zeros(config.num_experts, hidden, rank, device=device, dtype=dtype)
            )
            nn.init.kaiming_uniform_(self.adapter_a.view(-1, hidden), a=math.sqrt(5))
            self.shared_a = self.shared_b = None
        else:
            self.register_parameter("adapter_a", None)
            self.register_parameter("adapter_b", None)
            self.shared_a = nn.Linear(hidden, rank, bias=False).to(device=device, dtype=dtype)
            self.shared_b = nn.Linear(rank, hidden, bias=False).to(device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.shared_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.shared_b.weight)
        self.scale = config.alpha / config.rank
        self._intervention = "normal"
        self.reset_receipt()

    def set_code_intervention(self, intervention: str) -> None:
        if intervention not in self.INTERVENTIONS:
            raise SER1Error("SER1 intervention differs")
        if self.config.mode == "shared" and intervention != "normal":
            raise SER1Error("shared residual has no expert ownership")
        self._intervention = intervention

    def reset_receipt(self) -> None:
        self._receipt: dict[str, Any] = {
            "forwards": 0,
            "tokens": 0,
            "load_probability_sum": None,
            "token_entropy_sum": 0.0,
            "top8_top9_margin_sum": 0.0,
            "residual_norm_sum": 0.0,
            "native_output_norm_sum": 0.0,
            "expert_counts": None,
        }

    def receipt(self) -> dict[str, Any]:
        tokens = int(self._receipt["tokens"])
        if tokens == 0:
            return {"forwards": 0, "tokens": 0}
        probability_sum = self._receipt["load_probability_sum"].float().cpu()
        mean_probability = probability_sum / tokens
        load_entropy = -(
            mean_probability * mean_probability.clamp_min(1e-12).log()
        ).sum() / math.log(self.config.num_experts)
        counts = self._receipt["expert_counts"].to(torch.int64).cpu()
        result = {
            "forwards": int(self._receipt["forwards"]),
            "tokens": tokens,
            "load_entropy": float(load_entropy),
            "mean_token_entropy_normalized": self._receipt["token_entropy_sum"] / tokens,
            "mean_top8_top9_logit_margin": self._receipt["top8_top9_margin_sum"] / tokens,
            "mean_residual_norm": self._receipt["residual_norm_sum"] / tokens,
            "mean_native_output_norm": self._receipt["native_output_norm_sum"] / tokens,
            "active_experts": int((counts > 0).sum()),
            "expert_counts": counts.tolist(),
        }
        if self.adapter_a is not None and self.adapter_b is not None:
            result["expert_adapter_diagnostics"] = _bank_diagnostics(
                self.adapter_a.detach(), self.adapter_b.detach()
            )
        return result

    def _record(
        self,
        probabilities: torch.Tensor,
        indices: torch.Tensor,
        router_logits: torch.Tensor,
        native_output: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            token_count = int(probabilities.shape[0])
            token_entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(dim=-1) / math.log(self.config.num_experts)
            top9 = router_logits.float().topk(self.config.experts_per_token + 1, dim=-1).values
            margin = top9[:, self.config.experts_per_token - 1] - top9[:, self.config.experts_per_token]
            counts = torch.bincount(indices.reshape(-1), minlength=self.config.num_experts)
            probability_sum = probabilities.sum(dim=0)
            self._receipt["forwards"] += 1
            self._receipt["tokens"] += token_count
            self._receipt["token_entropy_sum"] += float(token_entropy.sum().cpu())
            self._receipt["top8_top9_margin_sum"] += float(margin.sum().cpu())
            self._receipt["residual_norm_sum"] += float(residual.float().norm(dim=-1).sum().cpu())
            self._receipt["native_output_norm_sum"] += float(native_output.float().norm(dim=-1).sum().cpu())
            self._receipt["load_probability_sum"] = (
                probability_sum.detach()
                if self._receipt["load_probability_sum"] is None
                else self._receipt["load_probability_sum"] + probability_sum.detach()
            )
            self._receipt["expert_counts"] = (
                counts.detach()
                if self._receipt["expert_counts"] is None
                else self._receipt["expert_counts"] + counts.detach()
            )

    def _selected_residual(
        self, hidden: torch.Tensor, indices: torch.Tensor, native_weights: torch.Tensor
    ) -> torch.Tensor:
        assert self.adapter_a is not None and self.adapter_b is not None
        if self._intervention == "zero":
            return torch.zeros_like(hidden)
        bank_indices = indices
        if self._intervention == "permutation":
            bank_indices = (indices + 1) % self.config.num_experts
        if self._intervention == "mean":
            selected_a = self.adapter_a.mean(dim=0)[None, None].expand(
                hidden.shape[0], indices.shape[1], -1, -1
            )
            selected_b = self.adapter_b.mean(dim=0)[None, None].expand(
                hidden.shape[0], indices.shape[1], -1, -1
            )
        else:
            selected_a = self.adapter_a[bank_indices]
            selected_b = self.adapter_b[bank_indices]
        low = torch.einsum("th,tkrh->tkr", hidden, selected_a)
        components = torch.einsum("tkr,tkhr->tkh", low, selected_b)
        q = native_weights.float()
        q = (q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)).detach()
        return (q.unsqueeze(-1) * components.float()).sum(dim=1).to(hidden.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden_size = hidden_states.shape
        flattened = hidden_states.reshape(-1, hidden_size)
        router_logits, native_weights, native_indices = self.base.gate(flattened)
        native_output = self.base.experts(flattened, native_indices, native_weights)
        if self.adapter_a is not None:
            residual = self._selected_residual(flattened, native_indices, native_weights)
        else:
            assert self.shared_a is not None and self.shared_b is not None
            residual = self.shared_b(self.shared_a(flattened))
        residual = residual * self.scale
        probabilities = F.softmax(router_logits.float(), dim=-1)
        self._record(probabilities, native_indices, router_logits, native_output, residual)
        return (native_output + residual.to(native_output.dtype)).reshape(
            batch, sequence, hidden_size
        )


def install_ser1_blocks(layers: Any, config: SER1Config) -> list[SelectedExpertResidualMoE]:
    if len(layers) < config.controlled_layers:
        raise SER1Error("backbone has too few sparse layers")
    wrappers = []
    for layer in layers[-config.controlled_layers :]:
        wrapper = SelectedExpertResidualMoE(layer.mlp, config)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    return wrappers


class SER1ProductModel(nn.Module):
    """Frozen OLMoE with selected-expert revision residuals."""

    def __init__(self, backbone: nn.Module, config: SER1Config, *, draft_control: str = "normal") -> None:
        super().__init__()
        if draft_control not in {"normal", "draft_unavailable"}:
            raise SER1Error("draft control differs")
        from hf_product_reasoning_train import resolve_product_backbone_layout

        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.text_model, self.lm_head, hidden, self.backbone_layout = resolve_product_backbone_layout(backbone)
        if hidden != config.hidden_size:
            raise SER1Error("backbone width differs")
        self.config = config
        self.draft_control = draft_control
        self.blocks = nn.ModuleList(install_ser1_blocks(self.text_model.layers, config))
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
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return loss, {"language_loss": float(loss.detach()), "charged_tokens": float(charged)}

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
                raise SER1Error("generation prompt tokenization differs")
            masked[row_index, positions] = torch.tensor(
                draft_attention, device=masked.device, dtype=masked.dtype
            )
        self._generation_prompt_attention = masked

    def generation_position_ids(self) -> torch.Tensor:
        if self._generation_position_ids is None:
            raise SER1Error("generation positions are absent")
        return self._generation_position_ids

    def generation_embeddings(self, prompt_ids, prompt_attention):
        attention = self._generation_prompt_attention
        if attention is None or attention.shape != prompt_attention.shape:
            raise SER1Error("generation draft attention is absent")
        return self.text_model.embed_tokens(prompt_ids), attention
