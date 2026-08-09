"""Expert-conditioned post-MoE residuals for temporal revision."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECR1Error(RuntimeError):
    """The frozen ECR1 sparse-block contract was violated."""


@dataclass(frozen=True)
class ECR1Config:
    hidden_size: int
    num_experts: int
    experts_per_token: int
    controlled_layers: int = 4
    rank: int = 31
    alpha: float = 31.0
    mode: str = "expert_conditioned"

    def validate(self) -> None:
        if self.mode not in {"expert_conditioned", "shared"}:
            raise ECR1Error("ECR1 mode differs")
        if min(
            self.hidden_size,
            self.num_experts,
            self.experts_per_token,
            self.controlled_layers,
            self.rank,
        ) <= 0 or self.alpha <= 0:
            raise ECR1Error("ECR1 dimensions differ")
        if self.experts_per_token > self.num_experts:
            raise ECR1Error("active experts exceed the expert bank")


class ExpertConditionedResidualMoE(nn.Module):
    """Frozen native MoE plus a bounded residual keyed by selected experts."""

    INTERVENTIONS = {"normal", "zero", "mean", "permutation"}

    def __init__(self, base: nn.Module, config: ECR1Config) -> None:
        super().__init__()
        config.validate()
        required = ("hidden_dim", "num_experts", "top_k", "norm_topk_prob")
        if not hasattr(base, "gate") or not hasattr(base, "experts"):
            raise ECR1Error("sparse MoE block interface differs")
        if any(not hasattr(base.gate, name) for name in required):
            raise ECR1Error("native top-k router interface differs")
        if (
            int(base.gate.hidden_dim) != config.hidden_size
            or int(base.gate.num_experts) != config.num_experts
            or int(base.gate.top_k) != config.experts_per_token
        ):
            raise ECR1Error("native sparse geometry differs")
        self.base = base
        self.base.requires_grad_(False)
        self.config = config
        hidden, rank = config.hidden_size, config.rank
        device = next(base.parameters()).device
        dtype = next(base.parameters()).dtype
        self.adapter_a = nn.Linear(hidden, rank, bias=False).to(device=device, dtype=dtype)
        self.adapter_b = nn.Linear(rank, hidden, bias=False).to(device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.adapter_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_b.weight)
        if config.mode == "expert_conditioned":
            self.expert_codes = nn.Parameter(
                torch.zeros(config.num_experts, rank, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("expert_codes", None)
        self.scale = config.alpha / config.rank
        self._code_intervention = "normal"
        self.reset_receipt()

    def set_code_intervention(self, intervention: str) -> None:
        if intervention not in self.INTERVENTIONS:
            raise ECR1Error("ECR1 code intervention differs")
        if self.config.mode == "shared" and intervention != "normal":
            raise ECR1Error("shared residual has no expert code")
        self._code_intervention = intervention

    def _selected_codes(self, indices: torch.Tensor) -> torch.Tensor:
        assert self.expert_codes is not None
        codes = self.expert_codes
        if self._code_intervention == "zero":
            codes = torch.zeros_like(codes)
        elif self._code_intervention == "mean":
            codes = codes.mean(dim=0, keepdim=True).expand_as(codes)
        elif self._code_intervention == "permutation":
            codes = torch.roll(codes, shifts=1, dims=0)
        return torch.tanh(codes[indices])

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
            "load_entropy_normalized": float(load_entropy),
            "mean_token_entropy_normalized": self._receipt["token_entropy_sum"] / tokens,
            "mean_top8_top9_logit_margin": self._receipt["top8_top9_margin_sum"] / tokens,
            "mean_residual_norm": self._receipt["residual_norm_sum"] / tokens,
            "mean_native_output_norm": self._receipt["native_output_norm_sum"] / tokens,
            "active_experts": int((counts > 0).sum()),
            "expert_counts": counts.tolist(),
        }
        if self.expert_codes is not None:
            result["expert_code_diagnostics"] = expert_code_diagnostics(
                self.expert_codes.detach().float()
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
            self._receipt["native_output_norm_sum"] += float(
                native_output.float().norm(dim=-1).sum().cpu()
            )
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence, hidden = hidden_states.shape
        flattened = hidden_states.reshape(-1, hidden)
        router_logits, native_weights, native_indices = self.base.gate(flattened)
        native_output = self.base.experts(flattened, native_indices, native_weights)
        low = self.adapter_a(flattened)
        if self.expert_codes is not None:
            q = native_weights.float()
            q = (q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)).detach()
            code = (q.unsqueeze(-1) * self._selected_codes(native_indices).float()).sum(dim=1)
            low = low * (1.0 + code.to(low.dtype))
        residual = self.adapter_b(low) * self.scale
        probabilities = F.softmax(router_logits.float(), dim=-1)
        self._record(
            probabilities,
            native_indices,
            router_logits,
            native_output,
            residual,
        )
        return (native_output + residual.to(native_output.dtype)).reshape(batch, sequence, hidden)


def expert_code_diagnostics(codes: torch.Tensor) -> dict[str, float]:
    normalized = F.normalize(codes, dim=-1, eps=1e-12)
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)]
    singular = torch.linalg.svdvals(codes)
    probabilities = singular.square()
    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    )
    return {
        "pairwise_cosine_mean": float(off_diagonal.mean()),
        "pairwise_cosine_abs_mean": float(off_diagonal.abs().mean()),
        "effective_rank": float(effective_rank),
    }


def install_ecr1_blocks(layers: Any, config: ECR1Config) -> list[ExpertConditionedResidualMoE]:
    if len(layers) < config.controlled_layers:
        raise ECR1Error("backbone has too few sparse layers")
    wrappers = []
    for layer in layers[-config.controlled_layers :]:
        wrapper = ExpertConditionedResidualMoE(layer.mlp, config)
        layer.mlp = wrapper
        wrappers.append(wrapper)
    return wrappers


class ECR1ProductModel(nn.Module):
    """Frozen OLMoE with same-pass expert-conditioned residual correction."""

    def __init__(
        self,
        backbone: nn.Module,
        config: ECR1Config,
        *,
        draft_control: str = "normal",
    ) -> None:
        super().__init__()
        if draft_control not in {"normal", "draft_unavailable"}:
            raise ECR1Error("draft control differs")
        from hf_product_reasoning_train import resolve_product_backbone_layout

        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.text_model, self.lm_head, hidden, self.backbone_layout = (
            resolve_product_backbone_layout(backbone)
        )
        if hidden != config.hidden_size:
            raise ECR1Error("backbone width differs")
        self.config = config
        self.draft_control = draft_control
        self.blocks = nn.ModuleList(install_ecr1_blocks(self.text_model.layers, config))
        self._generation_prompt_attention: torch.Tensor | None = None

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

    def forward_batch(
        self,
        prompt_rows: list[list[int]],
        response_rows: list[list[int]],
        pad_token_id: int,
        prompt_attention_rows: list[list[int]],
    ) -> tuple[torch.Tensor, dict[str, float]]:
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

    def prepare_generation_draft_attention(
        self,
        tokenizer: Any,
        rendered: list[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        if self.draft_control == "normal":
            self._generation_prompt_attention = attention_mask
            return
        from ttr1_revision import tokenize_with_draft_mask

        masked = attention_mask.clone()
        for row_index, prompt in enumerate(rendered):
            token_ids, draft_attention, _ = tokenize_with_draft_mask(tokenizer, prompt)
            positions = attention_mask[row_index].bool().nonzero().flatten()
            if input_ids[row_index, positions].tolist() != token_ids:
                raise ECR1Error("generation prompt tokenization differs")
            masked[row_index, positions] = torch.tensor(
                draft_attention, device=masked.device, dtype=masked.dtype
            )
        self._generation_prompt_attention = masked

    def generation_embeddings(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self._generation_prompt_attention
        if attention is None or attention.shape != prompt_attention.shape:
            raise ECR1Error("generation draft attention is absent")
        return self.text_model.embed_tokens(prompt_ids), attention
