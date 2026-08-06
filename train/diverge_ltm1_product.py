"""Qwen-hosted model-owned product path for DIVERGE-LTM1."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_ltm1_workspace import (
    FactorizedLatentTrajectoryWorkspace,
    LatentTrajectoryConfig,
    complete_trajectory_marginal_loss,
    ordered_trace_targets,
    trajectory_alignment_energy,
)
from hf_product_reasoning_train import (
    ProductReasoningTrainError,
    install_lora,
    pack_training_embeddings,
    resolve_product_backbone_layout,
)


class LTM1ProductModel(nn.Module):
    """LoRA backbone plus factorized, sticky complete latent trajectories."""

    architecture = "diverge-ltm1"

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
        fault_bits: int,
        attention_heads: int,
        ff_multiplier: int,
        trace_weight: float,
        balance_weight: float,
        halting_weight: float,
    ) -> None:
        super().__init__()
        if min(trace_weight, balance_weight, halting_weight) < 0.0:
            raise ProductReasoningTrainError("LTM1 loss weights must be nonnegative")
        self.backbone = backbone
        self.arm = "diverge_ltm1"
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
            fault_bits=fault_bits,
            attention_heads=attention_heads,
            ff_multiplier=ff_multiplier,
        )
        self.workspace = FactorizedLatentTrajectoryWorkspace(self.workspace_config)
        self.trace_weight = trace_weight
        self.balance_weight = balance_weight
        self.halting_weight = halting_weight
        self.selection_strategy = "highest_prior"

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def sequence_workspace_slots(self) -> int:
        return self.workspace_config.trajectory_slots

    def set_selection_strategy(self, strategy: str) -> None:
        if strategy not in {"highest_prior", "lowest_prior", "reset"}:
            raise ProductReasoningTrainError("LTM1 selection strategy differs")
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
        # The prompt representation is a detached observation.  Trainable LoRA
        # and LTM state are optimized through the candidate response paths.
        with torch.no_grad():
            prompt_features = self.text_model(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                use_cache=False,
            ).last_hidden_state
        output = self.workspace(prompt_features, prompt_mask)
        return prompt_ids, prompt_mask, output

    def forward_batch(
        self,
        prompt_rows: list[list[int]],
        response_rows: list[list[int]],
        pad_token_id: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if len(prompt_rows) != len(response_rows) or not prompt_rows:
            raise ProductReasoningTrainError("LTM1 batch geometry differs")
        embedding = self.text_model.embed_tokens
        _, _, workspace_output = self._prompt_trajectory_output(
            prompt_rows, pad_token_id
        )
        batch = len(prompt_rows)
        candidates = self.workspace_config.candidate_count
        expanded_prompts = [
            row for row in prompt_rows for _ in range(candidates)
        ]
        expanded_responses = [
            row for row in response_rows for _ in range(candidates)
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
        per_sequence_nll = token_losses.sum(dim=-1) / active_tokens.sum(
            dim=-1
        ).clamp_min(1)
        candidate_nll = per_sequence_nll.view(batch, candidates)

        trace_targets, trace_active = ordered_trace_targets(
            embedding,
            response_rows,
            self.workspace_config.recurrent_steps,
        )
        trace_energy = trajectory_alignment_energy(
            workspace_output.trajectory_probes,
            trace_targets,
            trace_active,
        )
        marginal = complete_trajectory_marginal_loss(
            candidate_nll,
            trace_energy,
            workspace_output.prior_logits,
            trace_weight=self.trace_weight,
            balance_weight=self.balance_weight,
        )
        halting_loss = self.workspace.halting_regularizer(workspace_output)
        loss = marginal.loss + self.halting_weight * halting_loss

        prior_indices = workspace_output.prior_logits.argmax(dim=-1)
        posterior_indices = marginal.posterior.argmax(dim=-1)
        rows = torch.arange(batch, device=prior_indices.device)
        prior_nll = candidate_nll[rows, prior_indices]
        prior_trace = trace_energy[rows, prior_indices]
        posterior_entropy = -(
            marginal.posterior
            * marginal.posterior.clamp_min(1e-9).log()
        ).sum(dim=-1)
        final_probes = F.normalize(
            workspace_output.trajectory_probes[:, :, -1].float(), dim=-1
        )
        similarity = torch.matmul(final_probes, final_probes.transpose(1, 2))
        off_diagonal = ~torch.eye(
            candidates, device=similarity.device, dtype=torch.bool
        ).unsqueeze(0)
        candidate_similarity = similarity.masked_select(off_diagonal).mean()

        return loss, {
            "marginal_energy": float(marginal.marginal_energy.detach()),
            "mean_candidate_nll": float(candidate_nll.detach().mean()),
            "best_candidate_nll": float(candidate_nll.detach().min(dim=-1).values.mean()),
            "prior_selected_nll": float(prior_nll.detach().mean()),
            "prior_selected_nll_rows": prior_nll.detach().float().cpu().tolist(),
            "trace_cosine": float((1.0 - prior_trace).detach().mean()),
            "trace_cosine_rows": (1.0 - prior_trace).detach().float().cpu().tolist(),
            "balance_loss": float(marginal.balance_loss.detach()),
            "halting_loss": float(halting_loss.detach()),
            "posterior_entropy": float(posterior_entropy.detach().mean()),
            "candidate_similarity": float(candidate_similarity.detach()),
            "prior_indices": prior_indices.detach().cpu().tolist(),
            "posterior_indices": posterior_indices.detach().cpu().tolist(),
            "logical_charged_tokens": float(sum(len(row) for row in response_rows)),
            "candidate_charged_tokens": float(candidate_charged),
            "final_stop_probability": float(
                workspace_output.stop_logits[:, :, -1].sigmoid().detach().mean()
            ),
            "mean_step_delta": float(
                workspace_output.step_delta_norms.detach().mean()
            ),
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
        prefix, _ = self.workspace.select_prefix(
            output,
            strategy=self.selection_strategy,
        )
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


def frozen_parameter_sha256(model: nn.Module) -> str:
    """Hash every non-trainable parameter through raw bytes, including BF16."""

    digest = hashlib.sha256()
    frozen = 0
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            continue
        frozen += 1
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(str(parameter.dtype).encode())
        raw = parameter.detach().to("cpu").contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    if frozen == 0:
        raise ProductReasoningTrainError("model exposes no frozen parameters")
    return digest.hexdigest()
