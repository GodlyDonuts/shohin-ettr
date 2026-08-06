"""Qwen-hosted product path for verified multi-trajectory matching."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_ltm1_product import frozen_parameter_sha256
from diverge_vmt1_workspace import (
    FactorizedLatentTrajectoryWorkspace,
    LatentTrajectoryConfig,
    complete_trace_cost_matrix,
    paired_ordered_trace_targets,
    verified_pair_assignment_objective,
)
from hf_product_reasoning_train import (
    ProductReasoningTrainError,
    install_lora,
    pack_training_embeddings,
    resolve_product_backbone_layout,
)


class VMT1ProductModel(nn.Module):
    """LoRA backbone with two coherent trajectories and a learned verifier."""

    architecture = "diverge-vmt1"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        lora_layers: int,
        lora_rank: int,
        lora_alpha: float,
        latent_width: int,
        trajectory_slots: int,
        recurrent_steps: int,
        attention_heads: int,
        ff_multiplier: int,
        assignment_temperature: float,
        validity_margin: float,
        trace_weight: float,
        validity_weight: float,
        halting_weight: float,
    ) -> None:
        super().__init__()
        if assignment_temperature <= 0.0 or validity_margin < 0.0:
            raise ProductReasoningTrainError("VMT1 assignment settings differ")
        if min(trace_weight, validity_weight, halting_weight) < 0.0:
            raise ProductReasoningTrainError("VMT1 loss weights must be nonnegative")
        self.backbone = backbone
        self.arm = "diverge_vmt1"
        self.backbone.requires_grad_(False)
        (
            self.text_model,
            self.lm_head,
            hidden_size,
            self.backbone_layout,
        ) = resolve_product_backbone_layout(backbone)
        layers = self.text_model.layers
        if not 0 < lora_layers <= len(layers):
            raise ProductReasoningTrainError("LoRA layer count differs")
        self.lora_projection_count = 0
        for layer in layers[-lora_layers:]:
            self.lora_projection_count += install_lora(layer, lora_rank, lora_alpha)
        if self.lora_projection_count == 0:
            raise ProductReasoningTrainError("no text projections received LoRA")

        self.lora_layers = lora_layers
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.workspace_config = LatentTrajectoryConfig(
            backbone_width=hidden_size,
            latent_width=latent_width,
            trajectory_slots=trajectory_slots,
            recurrent_steps=recurrent_steps,
            fault_bits=1,
            attention_heads=attention_heads,
            ff_multiplier=ff_multiplier,
        )
        self.workspace = FactorizedLatentTrajectoryWorkspace(self.workspace_config)
        self.validity_head = nn.Linear(hidden_size, 1)
        self.assignment_temperature = assignment_temperature
        self.validity_margin = validity_margin
        self.trace_weight = trace_weight
        self.validity_weight = validity_weight
        self.halting_weight = halting_weight
        self.selection_strategy = "validity"

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def sequence_workspace_slots(self) -> int:
        return self.workspace_config.trajectory_slots

    def set_selection_strategy(self, strategy: str) -> None:
        if strategy not in {"validity", "swapped_validity", "reset"}:
            raise ProductReasoningTrainError("VMT1 selection strategy differs")
        self.selection_strategy = strategy

    def _prompt_trajectory_output(
        self,
        prompt_rows: list[list[int]],
        pad_token_id: int,
    ):
        from hf_product_reasoning_train import _pad_token_rows

        embedding = self.text_model.embed_tokens
        prompt_ids, prompt_mask = _pad_token_rows(prompt_rows, pad_token_id)
        prompt_ids = prompt_ids.to(embedding.weight.device)
        prompt_mask = prompt_mask.to(embedding.weight.device)
        # Unlike LTM1, source-side LoRA receives direct trace/validity gradients.
        prompt_features = self.text_model(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            use_cache=False,
        ).last_hidden_state
        output = self.workspace(prompt_features, prompt_mask)
        validity_logits = self.validity_head(
            output.trajectory_probes[:, :, -1]
        ).squeeze(-1)
        return prompt_ids, prompt_mask, output, validity_logits

    @staticmethod
    def _validate_pair_rows(
        prompt_rows: Sequence[Sequence[int]],
        response_pair_rows: Sequence[Sequence[Sequence[int]]],
        target_correct: Sequence[Sequence[bool]],
    ) -> None:
        if not prompt_rows or len(prompt_rows) != len(response_pair_rows):
            raise ProductReasoningTrainError("VMT1 batch geometry differs")
        if len(prompt_rows) != len(target_correct):
            raise ProductReasoningTrainError("VMT1 correctness batch differs")
        for responses, correctness in zip(
            response_pair_rows, target_correct, strict=True
        ):
            if len(responses) != 2 or any(not response for response in responses):
                raise ProductReasoningTrainError(
                    "VMT1 requires two nonempty complete responses"
                )
            if len(correctness) != 2 or sum(map(bool, correctness)) != 1:
                raise ProductReasoningTrainError(
                    "VMT1 requires one verified-correct response"
                )

    def forward_batch(
        self,
        prompt_rows: list[list[int]],
        response_pair_rows: list[list[list[int]]],
        target_correct: list[list[bool]],
        pad_token_id: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self._validate_pair_rows(prompt_rows, response_pair_rows, target_correct)
        embedding = self.text_model.embed_tokens
        _, _, workspace_output, validity_logits = self._prompt_trajectory_output(
            prompt_rows, pad_token_id
        )
        batch = len(prompt_rows)
        candidates = self.workspace_config.candidate_count
        if candidates != 2:
            raise ProductReasoningTrainError("VMT1 internal trajectory count differs")

        correct_indices = [
            0 if bool(correctness[0]) else 1 for correctness in target_correct
        ]
        correct_responses = [
            list(response_pair_rows[index][correct_index])
            for index, correct_index in enumerate(correct_indices)
        ]
        expanded_prompts = [row for row in prompt_rows for _ in range(candidates)]
        expanded_responses = [
            row for row in correct_responses for _ in range(candidates)
        ]
        prefixes = workspace_output.candidate_prefixes.reshape(
            batch * candidates,
            self.workspace_config.trajectory_slots,
            self.workspace_config.backbone_width,
        ).to(dtype=embedding.weight.dtype)
        inputs, attention, labels, candidate_charged = pack_training_embeddings(
            embedding,
            expanded_prompts,
            expanded_responses,
            prefixes,
            pad_token_id,
        )
        outputs = self.text_model(
            inputs_embeds=inputs,
            attention_mask=attention,
            use_cache=False,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        shifted_logits = logits[:, :-1]
        shifted_labels = labels[:, 1:]
        token_losses = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(batch * candidates, -1)
        active_tokens = shifted_labels.ne(-100)
        correct_response_nll = (
            token_losses.sum(dim=-1) / active_tokens.sum(dim=-1).clamp_min(1)
        ).view(batch, candidates)

        trace_targets, trace_active = paired_ordered_trace_targets(
            embedding,
            response_pair_rows,
            self.workspace_config.recurrent_steps,
        )
        trace_cost = complete_trace_cost_matrix(
            workspace_output.trajectory_probes,
            trace_targets,
            trace_active,
        )
        correctness = torch.tensor(
            target_correct,
            device=trace_cost.device,
            dtype=torch.bool,
        )
        assignment = verified_pair_assignment_objective(
            trace_cost,
            validity_logits,
            correctness,
            correct_response_nll,
            assignment_temperature=self.assignment_temperature,
            validity_margin=self.validity_margin,
            trace_weight=self.trace_weight,
            validity_weight=self.validity_weight,
        )
        halting_loss = self.workspace.halting_regularizer(workspace_output)
        loss = assignment.loss + self.halting_weight * halting_loss

        assignment_entropy = -(
            assignment.assignment_posterior
            * assignment.assignment_posterior.clamp_min(1e-9).log()
        ).sum(dim=-1)
        selected_indices = validity_logits.argmax(dim=-1)
        final_probes = F.normalize(
            workspace_output.trajectory_probes[:, :, -1].float(), dim=-1
        )
        internal_similarity = (final_probes[:, 0] * final_probes[:, 1]).sum(dim=-1)
        return loss, {
            "assignment_loss": float(assignment.assignment_loss.detach()),
            "validity_loss": float(assignment.validity_loss.detach()),
            "correct_response_nll": float(assignment.correct_response_nll.detach()),
            "selected_correct_response_nll": float(
                assignment.selected_correct_response_nll.detach().mean()
            ),
            "selected_correct_response_nll_rows": (
                assignment.selected_correct_response_nll.detach().float().cpu().tolist()
            ),
            "matched_trace_cosine": float(
                assignment.matched_trace_cosine.detach().mean()
            ),
            "matched_trace_cosine_rows": (
                assignment.matched_trace_cosine.detach().float().cpu().tolist()
            ),
            "crossed_trace_cosine": float(
                assignment.crossed_trace_cosine.detach().mean()
            ),
            "crossed_trace_cosine_rows": (
                assignment.crossed_trace_cosine.detach().float().cpu().tolist()
            ),
            "selector_correct_rows": (
                assignment.selector_correct.detach().cpu().tolist()
            ),
            "swapped_selector_correct_rows": (
                assignment.swapped_selector_correct.detach().cpu().tolist()
            ),
            "best_assignments": assignment.best_assignment.detach().cpu().tolist(),
            "assignment_entropy": float(assignment_entropy.detach().mean()),
            "internal_trajectory_cosine": float(internal_similarity.detach().mean()),
            "internal_trajectory_cosine_rows": (
                internal_similarity.detach().cpu().tolist()
            ),
            "selected_indices": selected_indices.detach().cpu().tolist(),
            "correct_indices": correct_indices,
            "validity_logits_rows": validity_logits.detach().float().cpu().tolist(),
            "halting_loss": float(halting_loss.detach()),
            "logical_charged_tokens": float(sum(len(row) for row in correct_responses)),
            "candidate_charged_tokens": float(candidate_charged),
            "trace_target_tokens": float(
                sum(len(response) for pair in response_pair_rows for response in pair)
            ),
            "final_stop_probability": float(
                workspace_output.stop_logits[:, :, -1].sigmoid().detach().mean()
            ),
            "mean_step_delta": float(workspace_output.step_delta_norms.detach().mean()),
        }

    def generation_embeddings(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_ids.ndim != 2 or prompt_attention.shape != prompt_ids.shape:
            raise ProductReasoningTrainError("generation prompt geometry differs")
        embedding = self.text_model.embed_tokens
        prompt_ids = prompt_ids.to(embedding.weight.device)
        prompt_attention = prompt_attention.to(embedding.weight.device)
        prompt_embeddings = embedding(prompt_ids)
        prompt_features = self.text_model(
            input_ids=prompt_ids,
            attention_mask=prompt_attention,
            use_cache=False,
        ).last_hidden_state
        output = self.workspace(prompt_features, prompt_attention)
        validity_logits = self.validity_head(
            output.trajectory_probes[:, :, -1]
        ).squeeze(-1)
        indices = validity_logits.argmax(dim=-1)
        if self.selection_strategy == "swapped_validity":
            indices = validity_logits.flip(dims=(1,)).argmax(dim=-1)
        batch = torch.arange(indices.shape[0], device=indices.device)
        prefix = output.candidate_prefixes[batch, indices]
        if self.selection_strategy == "reset":
            prefix = torch.zeros_like(prefix)
        prefix = prefix.to(dtype=prompt_embeddings.dtype)
        prefix_attention = torch.ones(
            prefix.shape[:2],
            device=prompt_attention.device,
            dtype=prompt_attention.dtype,
        )
        return (
            torch.cat((prompt_embeddings, prefix), dim=1),
            torch.cat((prompt_attention, prefix_attention), dim=1),
        )


__all__ = ["VMT1ProductModel", "frozen_parameter_sha256"]
