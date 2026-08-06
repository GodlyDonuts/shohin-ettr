"""Protected product generator plus a guarded causal-revision packet."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_crp1_workspace import CausalRevisionConfig, CausalRevisionPacket
from hf_product_reasoning_train import (
    ProductReasoningModel,
    ProductReasoningTrainError,
    load_trainable_checkpoint,
    pack_training_embeddings,
)


CRP1_CHECKPOINT_SCHEMA = "shohin-diverge-crp1-checkpoint-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pad_rows(
    rows: Sequence[Sequence[int | bool]],
    *,
    width: int,
    fill: int | bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    output = torch.full((len(rows), width), fill, dtype=dtype)
    for index, row in enumerate(rows):
        if len(row) > width:
            raise ProductReasoningTrainError("revision row exceeds padded width")
        output[index, : len(row)] = torch.tensor(row, dtype=dtype)
    return output


def _pad_step_rows(
    rows: Sequence[Sequence[Sequence[bool]]],
    *,
    max_steps: int,
    width: int,
) -> torch.Tensor:
    output = torch.zeros(len(rows), max_steps, width, dtype=torch.bool)
    for batch, step_rows in enumerate(rows):
        if len(step_rows) > max_steps:
            raise ProductReasoningTrainError("revision trace exceeds packet width")
        for step, row in enumerate(step_rows):
            if len(row) > width:
                raise ProductReasoningTrainError("revision step exceeds padded width")
            output[batch, step, : len(row)] = torch.tensor(row, dtype=torch.bool)
    return output


class CRP1ProductModel(nn.Module):
    """Frozen generator conditioned by one selected coherent revision packet."""

    architecture = "diverge-crp1"

    def __init__(
        self,
        backbone: nn.Module,
        source_checkpoint: Path,
        *,
        source_checkpoint_sha256: str,
        source_revision: str,
        unguarded: bool,
        workspace_width: int = 256,
        workspace_slots: int = 6,
        recurrent_steps: int = 4,
        attention_heads: int = 8,
        ff_multiplier: int = 4,
        max_trace_steps: int = 12,
        localization_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if not source_checkpoint.is_file():
            raise ProductReasoningTrainError("protected source checkpoint is missing")
        actual_sha256 = _sha256_file(source_checkpoint)
        if actual_sha256 != source_checkpoint_sha256:
            raise ProductReasoningTrainError("protected source checkpoint hash differs")
        if localization_weight < 0.0:
            raise ProductReasoningTrainError("localization weight must be nonnegative")
        payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ProductReasoningTrainError("protected source metadata is missing")
        expected = {
            "arm": "baseline",
            "model_revision": source_revision,
            "lora_layers": 4,
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "unfreeze_layers": 2,
        }
        mismatches = {
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ProductReasoningTrainError(
                f"protected source contract differs: {mismatches}"
            )
        self.source = ProductReasoningModel(
            backbone=backbone,
            arm="baseline",
            lora_layers=4,
            lora_rank=8,
            lora_alpha=16.0,
            workspace_width=512,
            workspace_slots=16,
            recurrent_steps=8,
            unfreeze_layers=2,
        )
        source_update, restored_metadata = load_trainable_checkpoint(
            source_checkpoint, self.source
        )
        if restored_metadata != metadata:
            raise ProductReasoningTrainError("protected source metadata replay differs")
        self.source.requires_grad_(False)
        self.text_model = self.source.text_model
        self.lm_head = self.source.lm_head
        self.backbone = self.source.backbone
        hidden_size = int(self.text_model.embed_tokens.weight.shape[1])
        self.packet_config = CausalRevisionConfig(
            backbone_width=hidden_size,
            workspace_width=workspace_width,
            workspace_slots=workspace_slots,
            recurrent_steps=recurrent_steps,
            attention_heads=attention_heads,
            ff_multiplier=ff_multiplier,
            max_trace_steps=max_trace_steps,
        )
        self.packet = CausalRevisionPacket(self.packet_config)
        self.unguarded = bool(unguarded)
        self.localization_weight = float(localization_weight)
        self.source_checkpoint_sha256 = actual_sha256
        self.source_metadata = metadata
        self.source_update = source_update
        self.ablation = "normal"

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def set_ablation(self, ablation: str) -> None:
        if ablation not in {"normal", "reset", "force_no_error", "shift", "packet_swap"}:
            raise ProductReasoningTrainError("CRP1 ablation differs")
        self.ablation = ablation

    def frozen_source_sha256(self) -> str:
        digest = hashlib.sha256()
        count = 0
        for name, parameter in sorted(self.source.named_parameters()):
            if parameter.requires_grad:
                raise ProductReasoningTrainError("protected source became trainable")
            count += 1
            digest.update(name.encode())
            digest.update(str(tuple(parameter.shape)).encode())
            digest.update(str(parameter.dtype).encode())
            raw = parameter.detach().to("cpu").contiguous().view(torch.uint8)
            digest.update(raw.numpy().tobytes())
        if count == 0:
            raise ProductReasoningTrainError("protected source exposes no parameters")
        return digest.hexdigest()

    def _revision_output(
        self,
        prompt_rows: list[list[int]],
        problem_masks: list[list[bool]],
        step_masks: list[list[list[bool]]],
        final_masks: list[list[bool]],
        pad_token_id: int,
        *,
        selection_targets: torch.Tensor | None = None,
    ):
        if not prompt_rows or not (
            len(prompt_rows)
            == len(problem_masks)
            == len(step_masks)
            == len(final_masks)
        ):
            raise ProductReasoningTrainError("CRP1 revision batch differs")
        width = max(len(row) for row in prompt_rows)
        ids = _pad_rows(
            prompt_rows, width=width, fill=pad_token_id, dtype=torch.long
        ).to(self.text_model.embed_tokens.weight.device)
        attention = _pad_rows(
            [[True] * len(row) for row in prompt_rows],
            width=width,
            fill=False,
            dtype=torch.bool,
        ).to(ids.device)
        problem = _pad_rows(
            problem_masks, width=width, fill=False, dtype=torch.bool
        ).to(ids.device)
        steps = _pad_step_rows(
            step_masks, max_steps=self.packet_config.max_trace_steps, width=width
        ).to(ids.device)
        final = _pad_rows(
            final_masks, width=width, fill=False, dtype=torch.bool
        ).to(ids.device)
        with torch.no_grad():
            features = self.text_model(
                input_ids=ids,
                attention_mask=attention,
                use_cache=False,
            ).last_hidden_state
        return self.packet(
            features.detach(),
            attention,
            problem,
            steps,
            final,
            unguarded=self.unguarded,
            selection_targets=selection_targets,
            ablation=self.ablation,
        )

    def forward_batch(
        self,
        prompt_rows: list[list[int]],
        response_rows: list[list[int]],
        problem_masks: list[list[bool]],
        step_masks: list[list[list[bool]]],
        final_masks: list[list[bool]],
        error_targets: list[int],
        pad_token_id: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch = len(prompt_rows)
        if batch == 0 or batch % 2:
            raise ProductReasoningTrainError("CRP1 requires complete trace pairs")
        if not (
            len(response_rows)
            == len(problem_masks)
            == len(step_masks)
            == len(final_masks)
            == len(error_targets)
            == batch
        ):
            raise ProductReasoningTrainError("CRP1 supervised batch differs")
        for offset in range(0, batch, 2):
            if error_targets[offset] <= 0 or error_targets[offset + 1] != 0:
                raise ProductReasoningTrainError(
                    "CRP1 pairs must be ordered wrong then correct"
                )
        targets = torch.tensor(
            error_targets,
            device=self.text_model.embed_tokens.weight.device,
            dtype=torch.long,
        )
        revision = self._revision_output(
            prompt_rows,
            problem_masks,
            step_masks,
            final_masks,
            pad_token_id,
            selection_targets=targets,
        )
        prefix = revision.prefix_states.to(
            dtype=self.text_model.embed_tokens.weight.dtype
        )
        inputs, attention, labels, charged_tokens = pack_training_embeddings(
            self.text_model.embed_tokens,
            prompt_rows,
            response_rows,
            prefix,
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
        localization_loss = F.cross_entropy(revision.candidate_logits, targets)
        loss = language_loss + self.localization_weight * localization_loss
        predicted = revision.candidate_logits.argmax(dim=1)
        wrong = torch.arange(batch, device=targets.device) % 2 == 0
        correct = ~wrong
        candidate_prefixes = revision.all_candidate_prefixes.float().flatten(2)
        selected_prefix = candidate_prefixes[
            torch.arange(batch, device=targets.device), targets
        ]
        no_error_prefix = candidate_prefixes[:, 0]
        separation = 1.0 - F.cosine_similarity(selected_prefix, no_error_prefix, dim=1)
        return loss, {
            "language_loss": float(language_loss.detach()),
            "localization_loss": float(localization_loss.detach()),
            "localization_accuracy": float((predicted == targets).float().mean()),
            "wrong_localization_accuracy": float(
                (predicted[wrong] == targets[wrong]).float().mean()
            ),
            "correct_no_error_accuracy": float(
                (predicted[correct] == 0).float().mean()
            ),
            "mean_selected_no_error_separation": float(separation.mean().detach()),
            "mean_step_delta": float(revision.step_delta_norms.detach().mean()),
            "charged_tokens": float(charged_tokens),
            "candidate_predictions": predicted.detach().cpu().tolist(),
        }

    def revision_generation_embeddings(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention: torch.Tensor,
        problem_mask: torch.Tensor,
        step_mask: torch.Tensor,
        final_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if prompt_ids.ndim != 2 or prompt_attention.shape != prompt_ids.shape:
            raise ProductReasoningTrainError("CRP1 generation prompt differs")
        if (
            problem_mask.shape != prompt_ids.shape
            or final_mask.shape != prompt_ids.shape
            or step_mask.ndim != 3
            or step_mask.shape[0] != prompt_ids.shape[0]
            or step_mask.shape[2] != prompt_ids.shape[1]
        ):
            raise ProductReasoningTrainError("CRP1 generation segments differ")
        prompt_rows: list[list[int]] = []
        problem_rows: list[list[bool]] = []
        step_rows: list[list[list[bool]]] = []
        final_rows: list[list[bool]] = []
        for ids, active, problem, steps, final in zip(
            prompt_ids,
            prompt_attention,
            problem_mask,
            step_mask,
            final_mask,
            strict=True,
        ):
            keep = active.bool()
            prompt_rows.append(ids[keep].tolist())
            problem_rows.append(problem[keep].bool().tolist())
            step_rows.append([row[keep].bool().tolist() for row in steps])
            final_rows.append(final[keep].bool().tolist())
        revision = self._revision_output(
            prompt_rows,
            problem_rows,
            step_rows,
            final_rows,
            pad_token_id=0,
        )
        prompt_embeddings = self.text_model.embed_tokens(prompt_ids)
        prefix = revision.prefix_states.to(dtype=prompt_embeddings.dtype)
        prefix_attention = torch.ones(
            prefix.shape[:2],
            device=prompt_attention.device,
            dtype=prompt_attention.dtype,
        )
        return (
            torch.cat((prompt_embeddings, prefix), dim=1),
            torch.cat((prompt_attention, prefix_attention), dim=1),
            revision.candidate_logits,
            revision.selected_candidates,
        )


def save_crp1_checkpoint(
    path: Path,
    model: CRP1ProductModel,
    optimizer: torch.optim.Optimizer,
    update: int,
    metadata: dict[str, Any],
) -> None:
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.packet.named_parameters()
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": CRP1_CHECKPOINT_SCHEMA,
            "update": int(update),
            "packet_state": state,
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)


def load_crp1_checkpoint(
    path: Path,
    model: CRP1ProductModel,
    *,
    load_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CRP1_CHECKPOINT_SCHEMA:
        raise ProductReasoningTrainError("CRP1 checkpoint schema differs")
    metadata = payload.get("metadata")
    saved = payload.get("packet_state")
    if not isinstance(metadata, dict) or not isinstance(saved, dict):
        raise ProductReasoningTrainError("CRP1 checkpoint is incomplete")
    current = dict(model.packet.named_parameters())
    if set(saved) != set(current):
        raise ProductReasoningTrainError("CRP1 packet parameter contract differs")
    with torch.no_grad():
        for name, parameter in current.items():
            tensor = saved[name]
            if tensor.shape != parameter.shape:
                raise ProductReasoningTrainError(
                    f"CRP1 checkpoint tensor differs: {name}"
                )
            parameter.copy_(tensor.to(parameter.device, parameter.dtype))
    if load_optimizer is not None:
        optimizer_state = payload.get("optimizer")
        if not isinstance(optimizer_state, dict):
            raise ProductReasoningTrainError("CRP1 optimizer state is missing")
        load_optimizer.load_state_dict(optimizer_state)
    return int(payload["update"]), metadata


__all__ = [
    "CRP1_CHECKPOINT_SCHEMA",
    "CRP1ProductModel",
    "load_crp1_checkpoint",
    "save_crp1_checkpoint",
]
