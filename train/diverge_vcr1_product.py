"""Protected product generator plus a trainable temporal correction reactor."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_vcr1_workspace import (
    TemporalCorrectionConfig,
    TemporalCorrectionReactor,
)
from hf_product_reasoning_train import (
    ProductReasoningModel,
    ProductReasoningTrainError,
    load_trainable_checkpoint,
    pack_training_embeddings,
)


VCR1_CHECKPOINT_SCHEMA = "shohin-diverge-vcr1-checkpoint-v1"


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
            raise ProductReasoningTrainError("correction row exceeds padded width")
        output[index, : len(row)] = torch.tensor(row, dtype=dtype)
    return output


class VCR1ProductModel(nn.Module):
    """Frozen protected source with a temporally asymmetric correction prefix."""

    architecture = "diverge-vcr1"

    def __init__(
        self,
        backbone: nn.Module,
        source_checkpoint: Path,
        *,
        source_checkpoint_sha256: str,
        source_revision: str,
        role_blind: bool,
        workspace_width: int = 384,
        workspace_slots: int = 8,
        recurrent_steps: int = 4,
        attention_heads: int = 8,
        ff_multiplier: int = 4,
        validity_weight: float = 0.20,
        correction_margin_weight: float = 0.10,
        correction_margin: float = 0.25,
    ) -> None:
        super().__init__()
        if not source_checkpoint.is_file():
            raise ProductReasoningTrainError("protected source checkpoint is missing")
        actual_sha256 = _sha256_file(source_checkpoint)
        if actual_sha256 != source_checkpoint_sha256:
            raise ProductReasoningTrainError("protected source checkpoint hash differs")
        if min(validity_weight, correction_margin_weight, correction_margin) < 0.0:
            raise ProductReasoningTrainError("VCR1 loss settings must be nonnegative")

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
        self.workspace_config = TemporalCorrectionConfig(
            backbone_width=hidden_size,
            workspace_width=workspace_width,
            workspace_slots=workspace_slots,
            recurrent_steps=recurrent_steps,
            attention_heads=attention_heads,
            ff_multiplier=ff_multiplier,
        )
        self.reactor = TemporalCorrectionReactor(self.workspace_config)
        self.role_blind = bool(role_blind)
        self.validity_weight = validity_weight
        self.correction_margin_weight = correction_margin_weight
        self.correction_margin = correction_margin
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
        if ablation not in {"normal", "reset", "swap_roles"}:
            raise ProductReasoningTrainError("VCR1 ablation differs")
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

    def _correction_output(
        self,
        prompt_rows: list[list[int]],
        question_masks: list[list[bool]],
        draft_masks: list[list[bool]],
        pad_token_id: int,
    ):
        if not prompt_rows or not (
            len(prompt_rows) == len(question_masks) == len(draft_masks)
        ):
            raise ProductReasoningTrainError("VCR1 correction batch differs")
        width = max(len(row) for row in prompt_rows)
        ids = _pad_rows(
            prompt_rows,
            width=width,
            fill=pad_token_id,
            dtype=torch.long,
        ).to(self.text_model.embed_tokens.weight.device)
        attention = _pad_rows(
            [[True] * len(row) for row in prompt_rows],
            width=width,
            fill=False,
            dtype=torch.bool,
        ).to(ids.device)
        question = _pad_rows(
            question_masks,
            width=width,
            fill=False,
            dtype=torch.bool,
        ).to(ids.device)
        draft = _pad_rows(
            draft_masks,
            width=width,
            fill=False,
            dtype=torch.bool,
        ).to(ids.device)
        with torch.no_grad():
            features = self.text_model(
                input_ids=ids,
                attention_mask=attention,
                use_cache=False,
            ).last_hidden_state
        return self.reactor(
            features.detach(),
            attention,
            question,
            draft,
            role_blind=self.role_blind,
            swap_roles=self.ablation == "swap_roles",
            reset_prefix=self.ablation == "reset",
        )

    def forward_batch(
        self,
        prompt_rows: list[list[int]],
        response_rows: list[list[int]],
        question_masks: list[list[bool]],
        draft_masks: list[list[bool]],
        draft_is_correct: list[bool],
        pad_token_id: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch = len(prompt_rows)
        if batch == 0 or batch % 2:
            raise ProductReasoningTrainError("VCR1 requires complete draft pairs")
        if not (
            len(response_rows)
            == len(question_masks)
            == len(draft_masks)
            == len(draft_is_correct)
            == batch
        ):
            raise ProductReasoningTrainError("VCR1 supervised batch differs")
        for offset in range(0, batch, 2):
            if list(map(bool, draft_is_correct[offset : offset + 2])) != [False, True]:
                raise ProductReasoningTrainError(
                    "VCR1 pairs must be ordered wrong then correct"
                )

        correction = self._correction_output(
            prompt_rows, question_masks, draft_masks, pad_token_id
        )
        prefix = correction.prefix_states.to(
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
        validity_target = torch.tensor(
            draft_is_correct,
            device=correction.validity_logits.device,
            dtype=correction.validity_logits.dtype,
        )
        validity_loss = F.binary_cross_entropy_with_logits(
            correction.validity_logits, validity_target
        )
        strengths = correction.correction_strength.view(-1, 2)
        margin_loss = F.relu(
            self.correction_margin - strengths[:, 0] + strengths[:, 1]
        ).mean()
        loss = (
            language_loss
            + self.validity_weight * validity_loss
            + self.correction_margin_weight * margin_loss
        )
        predicted = correction.validity_logits >= 0
        target_bool = validity_target.bool()
        return loss, {
            "language_loss": float(language_loss.detach()),
            "validity_loss": float(validity_loss.detach()),
            "margin_loss": float(margin_loss.detach()),
            "validity_accuracy": float((predicted == target_bool).float().mean()),
            "wrong_correction_strength": float(strengths[:, 0].detach().mean()),
            "correct_correction_strength": float(strengths[:, 1].detach().mean()),
            "mean_step_delta": float(correction.step_delta_norms.detach().mean()),
            "charged_tokens": float(charged_tokens),
            "validity_logits": correction.validity_logits.detach()
            .float()
            .cpu()
            .tolist(),
        }

    def correction_generation_embeddings(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention: torch.Tensor,
        question_mask: torch.Tensor,
        draft_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prompt_ids.ndim != 2 or prompt_attention.shape != prompt_ids.shape:
            raise ProductReasoningTrainError("VCR1 generation prompt differs")
        if (
            question_mask.shape != prompt_ids.shape
            or draft_mask.shape != prompt_ids.shape
        ):
            raise ProductReasoningTrainError("VCR1 generation segments differ")
        prompt_rows: list[list[int]] = []
        question_rows: list[list[bool]] = []
        draft_rows: list[list[bool]] = []
        for ids, active, question, draft in zip(
            prompt_ids,
            prompt_attention,
            question_mask,
            draft_mask,
            strict=True,
        ):
            keep = active.bool()
            prompt_rows.append(ids[keep].tolist())
            question_rows.append(question[keep].bool().tolist())
            draft_rows.append(draft[keep].bool().tolist())
        correction = self._correction_output(
            prompt_rows,
            question_rows,
            draft_rows,
            pad_token_id=0,
        )
        prompt_embeddings = self.text_model.embed_tokens(prompt_ids)
        prefix = correction.prefix_states.to(dtype=prompt_embeddings.dtype)
        prefix_attention = torch.ones(
            prefix.shape[:2],
            device=prompt_attention.device,
            dtype=prompt_attention.dtype,
        )
        return (
            torch.cat((prompt_embeddings, prefix), dim=1),
            torch.cat((prompt_attention, prefix_attention), dim=1),
            correction.validity_logits,
        )


def save_vcr1_checkpoint(
    path: Path,
    model: VCR1ProductModel,
    optimizer: torch.optim.Optimizer,
    update: int,
    metadata: dict[str, Any],
) -> None:
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.reactor.named_parameters()
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": VCR1_CHECKPOINT_SCHEMA,
            "update": int(update),
            "reactor_state": state,
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)


def load_vcr1_checkpoint(
    path: Path,
    model: VCR1ProductModel,
    *,
    load_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != VCR1_CHECKPOINT_SCHEMA:
        raise ProductReasoningTrainError("VCR1 checkpoint schema differs")
    metadata = payload.get("metadata")
    saved = payload.get("reactor_state")
    if not isinstance(metadata, dict) or not isinstance(saved, dict):
        raise ProductReasoningTrainError("VCR1 checkpoint is incomplete")
    current = dict(model.reactor.named_parameters())
    if set(saved) != set(current):
        raise ProductReasoningTrainError("VCR1 reactor parameter contract differs")
    with torch.no_grad():
        for name, parameter in current.items():
            tensor = saved[name]
            if tensor.shape != parameter.shape:
                raise ProductReasoningTrainError(
                    f"VCR1 checkpoint tensor differs: {name}"
                )
            parameter.copy_(tensor.to(parameter.device, parameter.dtype))
    if load_optimizer is not None:
        optimizer_state = payload.get("optimizer")
        if not isinstance(optimizer_state, dict):
            raise ProductReasoningTrainError("VCR1 optimizer state is missing")
        load_optimizer.load_state_dict(optimizer_state)
    return int(payload["update"]), metadata


__all__ = [
    "VCR1_CHECKPOINT_SCHEMA",
    "VCR1ProductModel",
    "load_vcr1_checkpoint",
    "save_vcr1_checkpoint",
]
