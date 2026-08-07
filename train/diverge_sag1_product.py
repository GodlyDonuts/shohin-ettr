"""Counterfactual advantage-gated pointer transactions for product reasoning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_qpt1_product import (
    QPT1Output,
    QPT1ProductModel,
    frozen_parameter_sha256,
)
from hf_product_reasoning_train import (
    ProductReasoningTrainError,
    _pad_token_rows,
    pack_training_embeddings,
)


@dataclass
class SAG1Output:
    """One coherent base/expert choice plus the complete expert trace."""

    expert: QPT1Output
    router_logits: torch.Tensor
    router_probabilities: torch.Tensor
    router_decisions: torch.Tensor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sag1_architecture_sha256(
    model: "SAG1ProductModel",
    *,
    advantage_margin: float,
    router_threshold: float,
) -> str:
    payload = {
        "architecture": "diverge-sag1-counterfactual-advantage-gate-v1",
        "workspace_config": model.workspace_config.__dict__,
        "router_hidden": model.router_hidden,
        "advantage_margin": advantage_margin,
        "router_threshold": router_threshold,
        "invariants": [
            "qualified_base_adapter_is_frozen",
            "base_path_is_exact_when_router_abstains",
            "router_observes_prompt_only",
            "advantage_targets_are_detached",
            "whole_lineage_hard_commit",
            "no_fieldwise_or_logit_averaging",
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class SAG1ProductModel(QPT1ProductModel):
    """A frozen B1 lineage plus a prompt-gated pointer-transaction expert."""

    architecture = "diverge-sag1"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        base_checkpoint: Path,
        base_checkpoint_sha256: str,
        model_revision: str,
        lora_layers: int,
        lora_rank: int,
        lora_alpha: float,
        workspace_width: int,
        source_slots: int,
        query_slots: int,
        recurrent_steps: int,
        attention_heads: int,
        ff_multiplier: int,
        pointer_temperature: float,
        binding_weight: float,
        coverage_weight: float,
        reset_weight: float,
        router_hidden: int,
        advantage_margin: float,
        router_threshold: float,
        router_weight: float,
        risk_weight: float,
        sparsity_weight: float,
    ) -> None:
        if router_hidden <= 0:
            raise ProductReasoningTrainError("SAG1 router width must be positive")
        if advantage_margin < 0.0:
            raise ProductReasoningTrainError("SAG1 advantage margin must be nonnegative")
        if not 0.0 < router_threshold < 1.0:
            raise ProductReasoningTrainError("SAG1 router threshold must be in (0, 1)")
        if min(router_weight, risk_weight, sparsity_weight) < 0.0:
            raise ProductReasoningTrainError("SAG1 loss weights must be nonnegative")
        super().__init__(
            backbone,
            lora_layers=lora_layers,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            workspace_width=workspace_width,
            source_slots=source_slots,
            query_slots=query_slots,
            recurrent_steps=recurrent_steps,
            attention_heads=attention_heads,
            ff_multiplier=ff_multiplier,
            pointer_temperature=pointer_temperature,
            binding_weight=binding_weight,
            coverage_weight=coverage_weight,
            reset_weight=reset_weight,
        )
        self.arm = "diverge_sag1"
        self.router_hidden = router_hidden
        self.advantage_margin = advantage_margin
        self.router_threshold = router_threshold
        self.router_weight = router_weight
        self.risk_weight = risk_weight
        self.sparsity_weight = sparsity_weight
        self.base_checkpoint = base_checkpoint.resolve()
        self.base_checkpoint_sha256 = base_checkpoint_sha256
        self.base_metadata = self._load_and_freeze_base(
            self.base_checkpoint,
            expected_sha256=base_checkpoint_sha256,
            model_revision=model_revision,
            lora_layers=lora_layers,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        hidden_size = self.workspace_config.backbone_width
        self.router = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, router_hidden),
            nn.SiLU(),
            nn.Linear(router_hidden, 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, -2.0)

    def _load_and_freeze_base(
        self,
        checkpoint: Path,
        *,
        expected_sha256: str,
        model_revision: str,
        lora_layers: int,
        lora_rank: int,
        lora_alpha: float,
    ) -> dict[str, Any]:
        if not checkpoint.is_file():
            raise ProductReasoningTrainError("SAG1 base checkpoint is missing")
        actual_sha256 = _sha256_file(checkpoint)
        if actual_sha256 != expected_sha256:
            raise ProductReasoningTrainError("SAG1 base checkpoint hash differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != "shohin-hf-product-reasoning-checkpoint-v1":
            raise ProductReasoningTrainError("SAG1 base checkpoint schema differs")
        metadata = payload.get("metadata")
        saved = payload.get("trainable_state")
        if not isinstance(metadata, dict) or not isinstance(saved, dict):
            raise ProductReasoningTrainError("SAG1 base checkpoint is incomplete")
        expected = {
            "arm": "baseline",
            "model_revision": model_revision,
            "lora_layers": lora_layers,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
        }
        mismatches = {
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ProductReasoningTrainError(
                f"SAG1 base metadata differs: {json.dumps(mismatches, sort_keys=True)}"
            )
        current = dict(self.named_parameters())
        lora_names = {name for name in current if ".lora_" in name}
        if set(saved) != lora_names:
            raise ProductReasoningTrainError("SAG1 base LoRA parameter contract differs")
        with torch.no_grad():
            for name in sorted(lora_names):
                tensor = saved[name]
                parameter = current[name]
                if tensor.shape != parameter.shape:
                    raise ProductReasoningTrainError(
                        f"SAG1 base tensor shape differs: {name}"
                    )
                parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
                parameter.requires_grad_(False)
        return metadata

    @staticmethod
    def _per_example_cross_entropy(
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shifted_labels = labels[:, 1:]
        losses = F.cross_entropy(
            logits[:, :-1].transpose(1, 2),
            shifted_labels,
            ignore_index=-100,
            reduction="none",
        )
        mask = shifted_labels.ne(-100)
        denominator = mask.sum(dim=1).clamp_min(1)
        return (losses * mask).sum(dim=1) / denominator

    def _expert_and_router(
        self,
        prompt_rows: list[list[int]],
        pad_token_id: int,
    ) -> SAG1Output:
        embedding = self.text_model.embed_tokens
        prompt_ids, prompt_mask = _pad_token_rows(prompt_rows, pad_token_id)
        prompt_ids = prompt_ids.to(embedding.weight.device)
        prompt_mask = prompt_mask.to(embedding.weight.device)
        with torch.no_grad():
            prompt_features = self.text_model(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                use_cache=False,
            ).last_hidden_state
        expert = self.workspace(prompt_features, prompt_mask, control="normal")
        weights = prompt_mask.to(prompt_features.dtype).unsqueeze(-1)
        prompt_summary = (prompt_features * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
        router_logits = self.router(prompt_summary.float()).squeeze(-1)
        probabilities = torch.sigmoid(router_logits)
        decisions = probabilities.ge(self.router_threshold).to(probabilities.dtype)
        return SAG1Output(
            expert=expert,
            router_logits=router_logits,
            router_probabilities=probabilities,
            router_decisions=decisions,
        )

    def forward_batch(
        self,
        prompt_rows: list[list[int]],
        response_rows: list[list[int]],
        pad_token_id: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if len(prompt_rows) != len(response_rows) or not prompt_rows:
            raise ProductReasoningTrainError("SAG1 batch geometry differs")
        embedding = self.text_model.embed_tokens
        output = self._expert_and_router(prompt_rows, pad_token_id)
        residuals = output.expert.prompt_residuals.to(dtype=embedding.weight.dtype)
        base_inputs, attention, labels, charged = pack_training_embeddings(
            embedding,
            prompt_rows,
            response_rows,
            None,
            pad_token_id,
        )
        expert_inputs, expert_attention, expert_labels, expert_charged = (
            pack_training_embeddings(
                embedding,
                prompt_rows,
                response_rows,
                None,
                pad_token_id,
                prompt_residuals=residuals,
            )
        )
        if (
            expert_charged != charged
            or not torch.equal(attention, expert_attention)
            or not torch.equal(labels, expert_labels)
        ):
            raise ProductReasoningTrainError("SAG1 base/expert packing differs")
        with torch.no_grad():
            base_hidden = self.text_model(
                inputs_embeds=base_inputs,
                attention_mask=attention,
                use_cache=False,
            ).last_hidden_state
            base_logits = self.lm_head(base_hidden)
            base_losses = self._per_example_cross_entropy(base_logits, labels)
        expert_hidden = self.text_model(
            inputs_embeds=expert_inputs,
            attention_mask=expert_attention,
            use_cache=False,
        ).last_hidden_state
        expert_logits = self.lm_head(expert_hidden)
        expert_losses = self._per_example_cross_entropy(expert_logits, labels)
        language_loss = expert_losses.mean()

        advantage = (base_losses - expert_losses).detach()
        advantage_targets = advantage.gt(self.advantage_margin).to(
            output.router_logits.dtype
        )
        router_loss = F.binary_cross_entropy_with_logits(
            output.router_logits, advantage_targets
        )
        avoidable_risk = F.relu(
            expert_losses.detach() - base_losses + self.advantage_margin
        )
        risk_loss = (output.router_probabilities * avoidable_risk).mean()
        sparsity_loss = output.router_probabilities.mean()
        binding_loss, coverage_loss, reset_loss = self._auxiliary_losses(
            output.expert
        )
        loss = (
            language_loss
            + self.binding_weight * binding_loss
            + self.coverage_weight * coverage_loss
            + self.reset_weight * reset_loss
            + self.router_weight * router_loss
            + self.risk_weight * risk_loss
            + self.sparsity_weight * sparsity_loss
        )
        return loss, {
            "language_loss": float(language_loss.detach()),
            "base_language_loss": float(base_losses.mean()),
            "expert_advantage": float(advantage.mean()),
            "advantage_target_rate": float(advantage_targets.mean()),
            "router_loss": float(router_loss.detach()),
            "risk_loss": float(risk_loss.detach()),
            "sparsity_loss": float(sparsity_loss.detach()),
            "router_probability": float(output.router_probabilities.detach().mean()),
            "router_commit_rate": float(output.router_decisions.detach().mean()),
            "binding_loss": float(binding_loss.detach()),
            "coverage_loss": float(coverage_loss.detach()),
            "reset_loss": float(reset_loss.detach()),
            "mean_step_delta": float(output.expert.update_norms.detach().mean()),
            "mean_commit_gate": float(output.expert.commit_gates.detach().mean()),
            "mean_release_gate": float(output.expert.release_gates.detach().mean()),
            "logical_charged_tokens": float(charged),
        }

    def generation_embeddings(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_ids.ndim != 2 or prompt_attention.shape != prompt_ids.shape:
            raise ProductReasoningTrainError("SAG1 generation geometry differs")
        embedding = self.text_model.embed_tokens
        prompt_ids = prompt_ids.to(embedding.weight.device)
        prompt_attention = prompt_attention.to(embedding.weight.device)
        prompt_embeddings = embedding(prompt_ids)
        output = self._expert_and_router(
            [
                prompt_ids[index, prompt_attention[index].bool()].tolist()
                for index in range(prompt_ids.shape[0])
            ],
            pad_token_id=0,
        )
        residuals = output.expert.prompt_residuals.to(dtype=prompt_embeddings.dtype)
        result = prompt_embeddings.clone()
        for batch_index in range(result.shape[0]):
            valid = torch.nonzero(
                prompt_attention[batch_index], as_tuple=False
            ).flatten()
            count = min(int(valid.numel()), int(residuals.shape[1]))
            if count and bool(output.router_decisions[batch_index].item()):
                result[batch_index, valid[-count:]] += residuals[batch_index, :count]
        return result, prompt_attention


__all__ = [
    "SAG1Output",
    "SAG1ProductModel",
    "frozen_parameter_sha256",
    "sag1_architecture_sha256",
]
