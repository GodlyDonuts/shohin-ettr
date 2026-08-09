"""Draft-conditioned recurrent expert modulation for sparse revision models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DREM1Error(RuntimeError):
    """The MoE block, controller state, or causal mask contract differs."""


@dataclass(frozen=True)
class DREM1Config:
    hidden_size: int
    num_experts: int
    experts_per_token: int
    controlled_layers: int = 4
    controller_width: int = 256
    adapter_rank: int = 8
    recurrent_steps: int = 4
    router_scale: float = 1.0
    entropy_floor: float = 0.80

    def validate(self) -> None:
        positive = (
            self.hidden_size,
            self.num_experts,
            self.experts_per_token,
            self.controlled_layers,
            self.controller_width,
            self.adapter_rank,
            self.recurrent_steps,
        )
        if any(value <= 0 for value in positive):
            raise DREM1Error("DREM1 dimensions must be positive")
        if self.experts_per_token > self.num_experts:
            raise DREM1Error("active experts exceed the expert bank")
        if self.router_scale <= 0 or not 0 < self.entropy_floor < 1:
            raise DREM1Error("DREM1 routing bounds differ")


def pool_source_and_draft(
    features: torch.Tensor,
    attention_mask: torch.Tensor,
    draft_indicator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool disjoint source and exact-draft features from one causal prompt."""

    if (
        features.ndim != 3
        or attention_mask.shape != features.shape[:2]
        or draft_indicator.shape != attention_mask.shape
    ):
        raise DREM1Error("prompt feature and mask geometry differs")
    valid = attention_mask.bool()
    draft = valid & draft_indicator.bool()
    source = valid & ~draft
    if not bool(draft.any(dim=1).all()) or not bool(source.any(dim=1).all()):
        raise DREM1Error("every prompt requires source and draft tokens")

    def masked_mean(mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(features.dtype).unsqueeze(-1)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    return masked_mean(source), masked_mean(draft)


def aligned_generation_draft_indicator(
    tokenizer: Any,
    rendered: list[str],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct exact draft-token indicators under left-padded generation."""

    from ttr1_revision import tokenize_with_draft_mask

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise DREM1Error("generation token geometry differs")
    if len(rendered) != input_ids.shape[0]:
        raise DREM1Error("generation prompt batch differs")
    indicator = torch.zeros_like(attention_mask)
    for row_index, prompt in enumerate(rendered):
        token_ids, draft_attention, _ = tokenize_with_draft_mask(tokenizer, prompt)
        active_positions = attention_mask[row_index].bool().nonzero().flatten()
        active_ids = input_ids[row_index, active_positions].tolist()
        if active_ids != token_ids:
            raise DREM1Error("rendered prompt tokenization differs at generation")
        draft = torch.tensor(
            [1 - value for value in draft_attention],
            device=indicator.device,
            dtype=indicator.dtype,
        )
        indicator[row_index, active_positions] = draft
    return indicator


class DraftStateController(nn.Module):
    """Compile source/draft discrepancy into tied recurrent per-layer states."""

    def __init__(self, config: DREM1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.controller_width),
            nn.SiLU(),
            nn.LayerNorm(config.controller_width),
        )
        self.recurrence = nn.GRUCell(config.controller_width, config.controller_width)
        self.layer_codes = nn.Parameter(
            torch.empty(config.controlled_layers, config.controller_width)
        )
        self.state_norm = nn.LayerNorm(config.controller_width)
        nn.init.normal_(self.layer_codes, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        draft_indicator: torch.Tensor,
    ) -> list[torch.Tensor]:
        source, draft = pool_source_and_draft(
            features, attention_mask, draft_indicator
        )
        state = self.input_projection(
            torch.cat((source, draft, draft - source, draft * source), dim=-1)
        )
        states: list[torch.Tensor] = []
        for layer in range(self.config.controlled_layers):
            drive = state + self.layer_codes[layer]
            for _ in range(self.config.recurrent_steps):
                state = self.recurrence(drive, state)
            states.append(self.state_norm(state))
        return states


class DraftConditionedMoEBlock(nn.Module):
    """Frozen sparse experts plus recurrent-state router and expert adapters."""

    MODES = {"full", "router_only", "expert_only"}

    def __init__(self, base: nn.Module, config: DREM1Config) -> None:
        super().__init__()
        config.validate()
        if not hasattr(base, "gate") or not hasattr(base, "experts"):
            raise DREM1Error("sparse MoE block interface differs")
        required = ("hidden_dim", "num_experts", "top_k", "norm_topk_prob")
        if any(not hasattr(base.gate, name) for name in required):
            raise DREM1Error("top-k router interface differs")
        if (
            int(base.gate.hidden_dim) != config.hidden_size
            or int(base.gate.num_experts) != config.num_experts
            or int(base.gate.top_k) != config.experts_per_token
        ):
            raise DREM1Error("sparse MoE geometry differs from DREM1")
        self.base = base
        self.base.requires_grad_(False)
        self.config = config
        self.norm_topk_prob = bool(base.gate.norm_topk_prob)
        rank = config.adapter_rank
        hidden = config.hidden_size
        experts = config.num_experts
        self.route_token = nn.Linear(hidden, rank, bias=False)
        self.route_state = nn.Linear(config.controller_width, rank, bias=False)
        self.route_out = nn.Linear(rank, experts, bias=False)
        self.expert_down = nn.Linear(hidden, rank, bias=False)
        self.expert_state = nn.Linear(config.controller_width, rank, bias=False)
        self.expert_up = nn.Parameter(torch.zeros(experts, hidden, rank))
        nn.init.kaiming_uniform_(self.route_token.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.route_state.weight, a=math.sqrt(5))
        nn.init.zeros_(self.route_out.weight)
        nn.init.kaiming_uniform_(self.expert_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.expert_state.weight, a=math.sqrt(5))
        device = next(base.parameters()).device
        dtype = next(base.parameters()).dtype
        self.to(device=device, dtype=dtype)
        self._controller_state: torch.Tensor | None = None
        self._mode = "full"
        self.last_metrics: dict[str, torch.Tensor] = {}

    def set_controller_state(self, state: torch.Tensor, mode: str = "full") -> None:
        if mode not in self.MODES:
            raise DREM1Error("DREM1 intervention mode differs")
        if state.ndim != 2 or state.shape[1] != self.config.controller_width:
            raise DREM1Error("controller state geometry differs")
        self._controller_state = state
        self._mode = mode

    def clear_controller_state(self) -> None:
        self._controller_state = None
        self.last_metrics = {}

    def _expert_residual(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        rank_state = self.expert_down(hidden_states) * torch.sigmoid(
            self.expert_state(state)
        )
        output = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(
                top_k_index, num_classes=self.config.num_experts
            ).permute(2, 1, 0)
            active = expert_mask.sum(dim=(-1, -2)).gt(0).nonzero().flatten()
        for expert_index in active.tolist():
            top_k_position, token_index = torch.where(expert_mask[expert_index])
            selected = F.linear(rank_state[token_index], self.expert_up[expert_index])
            selected = selected * top_k_weights[
                token_index, top_k_position, None
            ]
            output.index_add_(0, token_index, selected.to(output.dtype))
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._controller_state is None:
            return self.base(hidden_states)
        batch, sequence, hidden = hidden_states.shape
        state = self._controller_state
        if state.shape[0] != batch:
            raise DREM1Error("controller and hidden-state batch differ")
        flattened = hidden_states.reshape(-1, hidden)
        expanded_state = (
            state[:, None, :].expand(batch, sequence, -1).reshape(-1, state.shape[-1])
        )
        base_logits, _, _ = self.base.gate(flattened)
        if self._mode == "expert_only":
            router_logits = base_logits
        else:
            interaction = self.route_token(flattened) * torch.tanh(
                self.route_state(expanded_state)
            )
            router_logits = base_logits + torch.tanh(self.route_out(interaction)) * (
                self.config.router_scale / self.config.adapter_rank
            )
        probabilities = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_k_weights, top_k_index = probabilities.topk(
            self.config.experts_per_token, dim=-1
        )
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(router_logits.dtype)
        output = self.base.experts(flattened, top_k_index, top_k_weights)
        if self._mode != "router_only":
            output = output + self._expert_residual(
                flattened, expanded_state, top_k_index, top_k_weights
            )
        mean_probability = probabilities.mean(dim=0)
        normalized_entropy = -(
            mean_probability * mean_probability.clamp_min(1e-12).log()
        ).sum() / math.log(self.config.num_experts)
        base_probabilities = F.softmax(base_logits.float(), dim=-1)
        self.last_metrics = {
            "route_probability_l1": (probabilities - base_probabilities)
            .abs()
            .sum(dim=-1)
            .mean(),
            "route_entropy_normalized": normalized_entropy,
            "collapse_penalty": F.relu(
                normalized_entropy.new_tensor(self.config.entropy_floor)
                - normalized_entropy
            ).square(),
        }
        return output.reshape(batch, sequence, hidden)


def install_drem1_blocks(
    layers: Any,
    config: DREM1Config,
) -> list[DraftConditionedMoEBlock]:
    if len(layers) < config.controlled_layers:
        raise DREM1Error("backbone has fewer layers than the DREM1 controller")
    wrappers = []
    for layer in layers[-config.controlled_layers :]:
        wrapper = DraftConditionedMoEBlock(layer.mlp, config)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    return wrappers


class DREM1ProductModel(nn.Module):
    """Frozen causal MoE with source/draft state controlling sparse execution."""

    def __init__(
        self,
        backbone: nn.Module,
        config: DREM1Config,
        mode: str = "full",
        collapse_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if mode not in DraftConditionedMoEBlock.MODES or collapse_weight < 0:
            raise DREM1Error("DREM1 mode or collapse weight differs")
        from hf_product_reasoning_train import resolve_product_backbone_layout

        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.text_model, self.lm_head, hidden, self.backbone_layout = (
            resolve_product_backbone_layout(backbone)
        )
        if hidden != config.hidden_size:
            raise DREM1Error("backbone width differs from DREM1")
        self.config = config
        self.mode = mode
        self.collapse_weight = collapse_weight
        self.controller = DraftStateController(config)
        self.blocks = nn.ModuleList(install_drem1_blocks(self.text_model.layers, config))
        device = self.text_model.embed_tokens.weight.device
        dtype = self.text_model.embed_tokens.weight.dtype
        self.controller.to(device=device, dtype=dtype)
        self._generation_draft_indicator: torch.Tensor | None = None

    def sequence_workspace_slots(self) -> int:
        return 0

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_name_sha256(self) -> str:
        names = sorted(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )
        return hashlib.sha256("\n".join(names).encode()).hexdigest()

    def protected_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not parameter.requires_grad and (".base.gate." in name or ".base.experts." in name)
        )

    def _set_context(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention: torch.Tensor,
        draft_indicator: torch.Tensor,
    ) -> None:
        for block in self.blocks:
            block.clear_controller_state()
        with torch.no_grad():
            features = self.text_model(
                input_ids=prompt_ids,
                attention_mask=prompt_attention,
                use_cache=False,
            ).last_hidden_state
        states = self.controller(features, prompt_attention, draft_indicator)
        for block, state in zip(self.blocks, states, strict=True):
            block.set_controller_state(state, self.mode)

    def clear_context(self) -> None:
        for block in self.blocks:
            block.clear_controller_state()

    def forward_batch(
        self,
        prompt_rows: list[list[int]],
        response_rows: list[list[int]],
        pad_token_id: int,
        draft_indicator_rows: list[list[int]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        from hf_product_reasoning_train import _pad_token_rows, pack_training_embeddings

        if len(prompt_rows) != len(draft_indicator_rows) or any(
            len(prompt) != len(indicator)
            for prompt, indicator in zip(prompt_rows, draft_indicator_rows, strict=True)
        ):
            raise DREM1Error("training draft-indicator geometry differs")
        prompt_ids, prompt_attention = _pad_token_rows(prompt_rows, pad_token_id)
        draft_indicator, _ = _pad_token_rows(draft_indicator_rows, 0)
        device = self.text_model.embed_tokens.weight.device
        prompt_ids = prompt_ids.to(device)
        prompt_attention = prompt_attention.to(device)
        draft_indicator = draft_indicator.to(device)
        self._set_context(prompt_ids, prompt_attention, draft_indicator)
        inputs, attention, labels, charged = pack_training_embeddings(
            self.text_model.embed_tokens,
            prompt_rows,
            response_rows,
            None,
            pad_token_id,
        )
        outputs = self.text_model(
            inputs_embeds=inputs,
            attention_mask=attention,
            use_cache=False,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        language_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        collapse = torch.stack(
            [block.last_metrics["collapse_penalty"] for block in self.blocks]
        ).mean()
        route_l1 = torch.stack(
            [block.last_metrics["route_probability_l1"] for block in self.blocks]
        ).mean()
        entropy = torch.stack(
            [block.last_metrics["route_entropy_normalized"] for block in self.blocks]
        ).mean()
        loss = language_loss + collapse * self.collapse_weight
        metrics = {
            "language_loss": float(language_loss.detach()),
            "collapse_loss": float(collapse.detach()),
            "route_probability_l1": float(route_l1.detach()),
            "route_entropy_normalized": float(entropy.detach()),
            "charged_tokens": float(charged),
        }
        self.clear_context()
        return loss, metrics

    def set_generation_draft_indicator(
        self,
        indicator: torch.Tensor,
    ) -> None:
        self._generation_draft_indicator = indicator

    def prepare_generation_draft_indicator(
        self,
        tokenizer: Any,
        rendered: list[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        self.set_generation_draft_indicator(
            aligned_generation_draft_indicator(
                tokenizer, rendered, input_ids, attention_mask
            )
        )

    def generation_embeddings(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indicator = self._generation_draft_indicator
        if indicator is None or indicator.shape != prompt_ids.shape:
            raise DREM1Error("generation draft indicator is absent or misaligned")
        device = self.text_model.embed_tokens.weight.device
        prompt_ids = prompt_ids.to(device)
        prompt_attention = prompt_attention.to(device)
        indicator = indicator.to(device)
        self._set_context(prompt_ids, prompt_attention, indicator)
        return self.text_model.embed_tokens(prompt_ids), prompt_attention
