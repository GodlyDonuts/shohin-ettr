"""Causal token-stream state routing for sparse temporal revision."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTSR1Error(RuntimeError):
    """The frozen CTSR1 model or streaming contract was violated."""


@dataclass(frozen=True)
class CTSR1Config:
    hidden_size: int
    num_experts: int
    experts_per_token: int
    controlled_layers: int = 16
    state_width: int = 64
    head_width: int = 32
    residual_rank: int = 18
    residual_alpha: float = 18.0
    mode: str = "temporal_router"
    router_scale: float = 1.0
    entropy_floor: float = 0.80
    collapse_weight: float = 0.01
    projection_seed: int = 2026080917

    def validate(self) -> None:
        if self.mode not in {"temporal_router", "temporal_shared"}:
            raise CTSR1Error("CTSR1 mode differs")
        dimensions = (
            self.hidden_size,
            self.num_experts,
            self.experts_per_token,
            self.controlled_layers,
            self.state_width,
            self.head_width,
            self.residual_rank,
        )
        if any(value <= 0 for value in dimensions):
            raise CTSR1Error("CTSR1 dimensions differ")
        if self.experts_per_token > self.num_experts:
            raise CTSR1Error("native active experts exceed bank")
        if self.residual_alpha <= 0 or self.router_scale <= 0:
            raise CTSR1Error("CTSR1 scales differ")
        if not 0 < self.entropy_floor < 1 or self.collapse_weight < 0:
            raise CTSR1Error("CTSR1 regularization differs")


class TemporalStateHead(nn.Module):
    def __init__(self, state_width: int, head_width: int, output_width: int) -> None:
        super().__init__()
        self.down = nn.Linear(state_width, head_width, bias=False)
        self.up = nn.Linear(head_width, output_width, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(state)))


def _fixed_projection(rank: int, experts: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value = torch.randn(rank, experts, generator=generator, dtype=torch.float32)
    return F.normalize(value, dim=-1)


class CausalTemporalMoEBlock(nn.Module):
    """Frozen native MoE controlled by a shared token-causal state."""

    def __init__(
        self,
        base: nn.Module,
        temporal_core: nn.GRU,
        route_head: TemporalStateHead,
        residual_head: TemporalStateHead,
        config: CTSR1Config,
        layer_index: int,
    ) -> None:
        super().__init__()
        config.validate()
        required = ("hidden_dim", "num_experts", "top_k", "norm_topk_prob")
        if not hasattr(base, "gate") or not hasattr(base, "experts"):
            raise CTSR1Error("native sparse interface differs")
        if any(not hasattr(base.gate, name) for name in required):
            raise CTSR1Error("native top-k router interface differs")
        if (
            int(base.gate.hidden_dim) != config.hidden_size
            or int(base.gate.num_experts) != config.num_experts
            or int(base.gate.top_k) != config.experts_per_token
        ):
            raise CTSR1Error("native sparse geometry differs")
        self.base = base
        self.base.requires_grad_(False)
        self.config = config
        self.layer_index = layer_index
        self.norm_topk_prob = bool(base.gate.norm_topk_prob)
        object.__setattr__(self, "_temporal_core", temporal_core)
        object.__setattr__(self, "_route_head", route_head)
        object.__setattr__(self, "_residual_head", residual_head)
        device = next(base.parameters()).device
        dtype = next(base.parameters()).dtype
        self.layer_code = nn.Parameter(
            torch.empty(config.state_width, device=device, dtype=dtype)
        )
        nn.init.normal_(self.layer_code, std=0.02)
        self.adapter_a = nn.Linear(
            config.hidden_size, config.residual_rank, bias=False
        ).to(device=device, dtype=dtype)
        self.adapter_b = nn.Linear(
            config.residual_rank, config.hidden_size, bias=False
        ).to(device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.adapter_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_b.weight)
        projection = _fixed_projection(
            config.residual_rank,
            config.num_experts,
            config.projection_seed + layer_index,
        )
        self.register_buffer(
            "residual_projection", projection.to(device=device, dtype=dtype)
        )
        self.scale = config.residual_alpha / config.residual_rank
        self._streaming = False
        self._state: torch.Tensor | None = None
        self._sequence_mask: torch.Tensor | None = None
        self._collapse_loss: torch.Tensor | None = None
        self.reset_receipt()

    @property
    def temporal_core(self) -> nn.GRU:
        return object.__getattribute__(self, "_temporal_core")

    @property
    def route_head(self) -> TemporalStateHead:
        return object.__getattribute__(self, "_route_head")

    @property
    def residual_head(self) -> TemporalStateHead:
        return object.__getattribute__(self, "_residual_head")

    def begin_sequence(self, attention_mask: torch.Tensor, *, streaming: bool) -> None:
        if attention_mask.ndim != 2:
            raise CTSR1Error("sequence attention geometry differs")
        self._streaming = streaming
        self._state = None
        self._sequence_mask = attention_mask.bool()

    def reset_receipt(self) -> None:
        self._receipt: dict[str, Any] = {
            "forwards": 0,
            "tokens": 0,
            "probability_sum": None,
            "token_entropy_sum": 0.0,
            "route_l1_sum": 0.0,
            "top1_changes": 0,
            "expert_counts": None,
            "state_norm_sum": 0.0,
            "residual_norm_sum": 0.0,
        }

    def collapse_loss(self) -> torch.Tensor:
        if self._collapse_loss is None:
            return self.layer_code.new_zeros(())
        return self._collapse_loss

    def receipt(self) -> dict[str, Any]:
        tokens = int(self._receipt["tokens"])
        if not tokens:
            return {"forwards": 0, "tokens": 0}
        mean_probability = self._receipt["probability_sum"].float().cpu() / tokens
        load_entropy = -(
            mean_probability * mean_probability.clamp_min(1e-12).log()
        ).sum() / math.log(self.config.num_experts)
        counts = self._receipt["expert_counts"].to(torch.int64).cpu()
        return {
            "forwards": int(self._receipt["forwards"]),
            "tokens": tokens,
            "load_entropy": float(load_entropy),
            "mean_token_entropy_normalized": self._receipt["token_entropy_sum"] / tokens,
            "route_probability_l1_mean": self._receipt["route_l1_sum"] / tokens,
            "top1_change_rate": self._receipt["top1_changes"] / tokens,
            "active_experts": int((counts > 0).sum()),
            "expert_counts": counts.tolist(),
            "mean_state_norm": self._receipt["state_norm_sum"] / tokens,
            "mean_residual_norm": self._receipt["residual_norm_sum"] / tokens,
        }

    def _causal_states(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        if self._state is not None:
            if self._state.shape != (1, batch, self.config.state_width):
                raise CTSR1Error("streaming state geometry differs")
            initial = self._state
        else:
            initial = self.layer_code.view(1, 1, -1).expand(1, batch, -1).contiguous()
        mask = self._sequence_mask
        if mask is not None and mask.shape != hidden.shape[:2]:
            raise CTSR1Error("stored attention mask geometry differs")
        if mask is None or bool(mask.all()):
            states, final = self.temporal_core(hidden, initial)
        else:
            states = hidden.new_zeros(batch, sequence, self.config.state_width)
            finals = []
            for row in range(batch):
                positions = mask[row].nonzero().flatten()
                if not len(positions):
                    raise CTSR1Error("empty causal sequence")
                expected = torch.arange(
                    int(positions[0]), sequence, device=positions.device
                )
                if not torch.equal(positions, expected):
                    raise CTSR1Error("only contiguous left padding is supported")
                active, row_final = self.temporal_core(
                    hidden[row : row + 1, positions], initial[:, row : row + 1]
                )
                states[row, positions] = active[0]
                finals.append(row_final)
            final = torch.cat(finals, dim=1)
        if self._streaming:
            self._state = final.detach()
        self._sequence_mask = None
        return states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden = hidden_states.shape
        states = self._causal_states(hidden_states)
        flattened = hidden_states.reshape(-1, hidden)
        flat_states = states.reshape(-1, self.config.state_width)
        base_logits, _, base_indices = self.base.gate(flattened)
        route_signal = self.route_head(flat_states)
        residual_signal = self.residual_head(flat_states)
        if self.config.mode == "temporal_router":
            router_logits = base_logits + torch.tanh(route_signal) * self.config.router_scale
            gate_source = torch.tanh(residual_signal)
        else:
            router_logits = base_logits + route_signal * 0.0
            gate_source = torch.tanh((route_signal + residual_signal) * 0.5)
        probabilities = F.softmax(router_logits.float(), dim=-1)
        weights, indices = probabilities.topk(self.config.experts_per_token, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = weights.to(router_logits.dtype)
        native_output = self.base.experts(flattened, indices, weights)
        residual_gate = F.linear(gate_source.to(self.residual_projection.dtype), self.residual_projection)
        low = self.adapter_a(flattened) * (1.0 + torch.tanh(residual_gate))
        residual = self.adapter_b(low) * self.scale
        output = native_output + residual.to(native_output.dtype)

        mean_probability = probabilities.mean(dim=0)
        load_entropy = -(
            mean_probability * mean_probability.clamp_min(1e-12).log()
        ).sum() / math.log(self.config.num_experts)
        self._collapse_loss = F.relu(
            load_entropy.new_tensor(self.config.entropy_floor) - load_entropy
        ).square()
        with torch.no_grad():
            base_probability = F.softmax(base_logits.float(), dim=-1)
            entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(dim=-1) / math.log(self.config.num_experts)
            counts = torch.bincount(indices.reshape(-1), minlength=self.config.num_experts)
            probability_sum = probabilities.sum(dim=0)
            self._receipt["forwards"] += 1
            self._receipt["tokens"] += int(flattened.shape[0])
            self._receipt["token_entropy_sum"] += float(entropy.sum().cpu())
            self._receipt["route_l1_sum"] += float(
                (probabilities - base_probability).abs().sum(dim=-1).sum().cpu()
            )
            self._receipt["top1_changes"] += int(
                (indices[:, 0] != base_indices[:, 0]).sum().cpu()
            )
            self._receipt["state_norm_sum"] += float(
                flat_states.float().norm(dim=-1).sum().cpu()
            )
            self._receipt["residual_norm_sum"] += float(
                residual.float().norm(dim=-1).sum().cpu()
            )
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
        return output.reshape(batch, sequence, hidden)


class CTSR1ProductModel(nn.Module):
    """Frozen OLMoE with causal state persisted across generated tokens."""

    def __init__(self, backbone: nn.Module, config: CTSR1Config) -> None:
        super().__init__()
        config.validate()
        from hf_product_reasoning_train import resolve_product_backbone_layout

        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.text_model, self.lm_head, hidden, self.backbone_layout = (
            resolve_product_backbone_layout(backbone)
        )
        if hidden != config.hidden_size:
            raise CTSR1Error("backbone width differs")
        self.config = config
        device = self.text_model.embed_tokens.weight.device
        dtype = self.text_model.embed_tokens.weight.dtype
        self.temporal_core = nn.GRU(
            config.hidden_size, config.state_width, batch_first=True
        ).to(device=device, dtype=dtype)
        self.route_head = TemporalStateHead(
            config.state_width, config.head_width, config.num_experts
        ).to(device=device, dtype=dtype)
        self.residual_head = TemporalStateHead(
            config.state_width, config.head_width, config.num_experts
        ).to(device=device, dtype=dtype)
        if len(self.text_model.layers) < config.controlled_layers:
            raise CTSR1Error("backbone has too few sparse layers")
        wrappers = []
        for index, layer in enumerate(
            self.text_model.layers[-config.controlled_layers :]
        ):
            wrapper = CausalTemporalMoEBlock(
                layer.mlp,
                self.temporal_core,
                self.route_head,
                self.residual_head,
                config,
                index,
            )
            layer.mlp = wrapper
            wrappers.append(wrapper)
        self.blocks = nn.ModuleList(wrappers)
        self._generation_position_ids: torch.Tensor | None = None

    def sequence_workspace_slots(self) -> int:
        return 0

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_parameter_name_sha256(self) -> str:
        names = sorted(name for name, p in self.named_parameters() if p.requires_grad)
        return hashlib.sha256("\n".join(names).encode()).hexdigest()

    def protected_parameter_count(self) -> int:
        return sum(
            p.numel()
            for name, p in self.named_parameters()
            if not p.requires_grad and (".base.gate." in name or ".base.experts." in name)
        )

    def reset_routing_receipt(self) -> None:
        for block in self.blocks:
            block.reset_receipt()

    def routing_receipt(self) -> dict[str, Any]:
        return {"controlled_layers": len(self.blocks), "layers": [b.receipt() for b in self.blocks]}

    def _begin_sequence(self, attention: torch.Tensor, *, streaming: bool) -> None:
        for block in self.blocks:
            block.begin_sequence(attention, streaming=streaming)

    def forward_batch(self, prompt_rows, response_rows, pad_token_id, prompt_attention_rows):
        from hf_product_reasoning_train import pack_training_embeddings

        inputs, attention, labels, charged = pack_training_embeddings(
            self.text_model.embed_tokens,
            prompt_rows,
            response_rows,
            None,
            pad_token_id,
        )
        self._begin_sequence(attention, streaming=False)
        outputs = self.text_model(inputs_embeds=inputs, attention_mask=attention, use_cache=False)
        logits = self.lm_head(outputs.last_hidden_state)
        language_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        collapse = torch.stack([block.collapse_loss() for block in self.blocks]).mean()
        loss = language_loss + self.config.collapse_weight * collapse
        return loss, {
            "language_loss": float(language_loss.detach()),
            "collapse_loss": float(collapse.detach()),
            "charged_tokens": float(charged),
        }

    def prepare_generation_draft_attention(self, tokenizer, rendered, input_ids, attention_mask) -> None:
        del tokenizer, rendered, input_ids
        positions = attention_mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(~attention_mask.bool(), 0)
        self._generation_position_ids = positions
        self._begin_sequence(attention_mask, streaming=True)

    def generation_position_ids(self) -> torch.Tensor:
        if self._generation_position_ids is None:
            raise CTSR1Error("generation positions are absent")
        return self._generation_position_ids

    def generation_embeddings(self, prompt_ids, prompt_attention):
        return self.text_model.embed_tokens(prompt_ids), prompt_attention

