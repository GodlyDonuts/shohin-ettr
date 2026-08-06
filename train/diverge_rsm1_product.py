"""Frozen CRP1 packet plus a trainable persistent discrete state replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_crp1_product import CRP1ProductModel, load_crp1_checkpoint
from diverge_rsm1_workspace import (
    PersistentReplayConfig,
    PersistentReplayOutput,
    PersistentStateReplay,
)
from hf_product_reasoning_train import ProductReasoningTrainError


RSM1_CHECKPOINT_SCHEMA = "shohin-diverge-rsm1-checkpoint-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(module.state_dict().items()):
        count += 1
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        raw = tensor.detach().to("cpu").contiguous().view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    if count == 0:
        raise ProductReasoningTrainError("RSM1 module exposes no state")
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
            raise ProductReasoningTrainError("RSM1 row exceeds padded width")
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
            raise ProductReasoningTrainError("RSM1 trace exceeds replay width")
        for step, row in enumerate(step_rows):
            if len(row) > width:
                raise ProductReasoningTrainError("RSM1 step exceeds padded width")
            output[batch, step, : len(row)] = torch.tensor(row, dtype=torch.bool)
    return output


@dataclass
class RSM1InferenceOutput:
    replay: PersistentReplayOutput
    candidate_logits: torch.Tensor
    selected_candidates: torch.Tensor


class RSM1ProductModel(nn.Module):
    """Train only state replay while keeping source and CRP1 packet immutable."""

    architecture = "diverge-rsm1"

    def __init__(
        self,
        backbone: nn.Module,
        source_checkpoint: Path,
        crp_checkpoint: Path,
        *,
        source_checkpoint_sha256: str,
        crp_checkpoint_sha256: str,
        source_revision: str,
        packet_arm: str,
        state_width: int = 256,
        state_slots: int = 24,
        packet_slots: int = 6,
        max_trace_steps: int = 12,
        attention_heads: int = 8,
        ff_multiplier: int = 4,
    ) -> None:
        super().__init__()
        if packet_arm not in {"guarded", "unguarded"}:
            raise ProductReasoningTrainError("RSM1 packet arm differs")
        if _sha256_file(crp_checkpoint) != crp_checkpoint_sha256:
            raise ProductReasoningTrainError("RSM1 CRP1 checkpoint hash differs")
        self.crp = CRP1ProductModel(
            backbone,
            source_checkpoint,
            source_checkpoint_sha256=source_checkpoint_sha256,
            source_revision=source_revision,
            unguarded=packet_arm == "unguarded",
            workspace_width=256,
            workspace_slots=packet_slots,
            recurrent_steps=4,
            attention_heads=8,
            ff_multiplier=4,
            max_trace_steps=max_trace_steps,
            localization_weight=0.25,
        )
        crp_update, crp_metadata = load_crp1_checkpoint(crp_checkpoint, self.crp)
        expected = {
            "architecture": "diverge-crp1",
            "arm": packet_arm,
            "model_revision": source_revision,
            "source_checkpoint_sha256": source_checkpoint_sha256,
        }
        mismatches = {
            key: {"expected": value, "actual": crp_metadata.get(key)}
            for key, value in expected.items()
            if crp_metadata.get(key) != value
        }
        if mismatches or crp_update != 200:
            raise ProductReasoningTrainError(
                f"RSM1 frozen CRP1 contract differs: {mismatches}, update={crp_update}"
            )
        packet_config = crp_metadata.get("packet_config")
        if packet_config != {
            "backbone_width": self.crp.packet_config.backbone_width,
            "workspace_width": 256,
            "workspace_slots": packet_slots,
            "recurrent_steps": 4,
            "attention_heads": 8,
            "ff_multiplier": 4,
            "max_trace_steps": max_trace_steps,
        }:
            raise ProductReasoningTrainError("RSM1 frozen packet geometry differs")
        self.crp.requires_grad_(False)
        self.crp.eval()
        self.text_model = self.crp.text_model
        hidden_size = int(self.text_model.embed_tokens.weight.shape[1])
        self.replay_config = PersistentReplayConfig(
            backbone_width=hidden_size,
            state_width=state_width,
            state_slots=state_slots,
            packet_slots=packet_slots,
            max_trace_steps=max_trace_steps,
            attention_heads=attention_heads,
            ff_multiplier=ff_multiplier,
        )
        self.replay = PersistentStateReplay(self.replay_config)
        self.packet_arm = packet_arm
        self.crp_checkpoint_sha256 = crp_checkpoint_sha256
        self.crp_checkpoint_update = crp_update
        self.crp_metadata = crp_metadata
        self.ablation = "normal"

    def train(self, mode: bool = True):
        super().train(mode)
        self.crp.eval()
        return self

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def set_ablation(self, ablation: str) -> None:
        if ablation not in {
            "normal",
            "reset",
            "force_no_error",
            "shift",
            "packet_swap",
        }:
            raise ProductReasoningTrainError("RSM1 ablation differs")
        self.ablation = ablation

    def frozen_crp_sha256(self) -> str:
        if any(parameter.requires_grad for parameter in self.crp.parameters()):
            raise ProductReasoningTrainError("frozen CRP1 state became trainable")
        digest = hashlib.sha256()
        digest.update(self.crp.frozen_source_sha256().encode())
        digest.update(module_state_sha256(self.crp.packet).encode())
        return digest.hexdigest()

    def _frozen_context(
        self,
        prompt_rows: list[list[int]],
        problem_masks: list[list[bool]],
        packet_step_masks: list[list[list[bool]]],
        operation_masks: list[list[list[bool]]],
        final_masks: list[list[bool]],
        pad_token_id: int,
        *,
        selection_targets: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = len(prompt_rows)
        if batch == 0 or not (
            len(problem_masks)
            == len(packet_step_masks)
            == len(operation_masks)
            == len(final_masks)
            == batch
        ):
            raise ProductReasoningTrainError("RSM1 frozen input batch differs")
        width = max(len(row) for row in prompt_rows)
        device = self.text_model.embed_tokens.weight.device
        ids = _pad_rows(
            prompt_rows, width=width, fill=pad_token_id, dtype=torch.long
        ).to(device)
        attention = _pad_rows(
            [[True] * len(row) for row in prompt_rows],
            width=width,
            fill=False,
            dtype=torch.bool,
        ).to(device)
        problem = _pad_rows(
            problem_masks, width=width, fill=False, dtype=torch.bool
        ).to(device)
        packet_steps = _pad_step_rows(
            packet_step_masks,
            max_steps=self.replay_config.max_trace_steps,
            width=width,
        ).to(device)
        operations = _pad_step_rows(
            operation_masks,
            max_steps=self.replay_config.max_trace_steps,
            width=width,
        ).to(device)
        final = _pad_rows(
            final_masks, width=width, fill=False, dtype=torch.bool
        ).to(device)
        if torch.any(operations & ~packet_steps):
            raise ProductReasoningTrainError("RSM1 operation escaped its packet step")
        with torch.no_grad():
            features = self.text_model(
                input_ids=ids,
                attention_mask=attention,
                use_cache=False,
            ).last_hidden_state
            revision = self.crp.packet(
                features,
                attention,
                problem,
                packet_steps,
                final,
                unguarded=self.packet_arm == "unguarded",
                selection_targets=selection_targets,
                ablation=self.ablation,
            )
        return (
            revision.prefix_states.detach(),
            features.detach(),
            attention,
            problem,
            operations,
            revision.candidate_logits.detach(),
            revision.selected_candidates.detach(),
        )

    def run_replay(
        self,
        prompt_rows: list[list[int]],
        problem_masks: list[list[bool]],
        packet_step_masks: list[list[list[bool]]],
        operation_masks: list[list[list[bool]]],
        final_masks: list[list[bool]],
        pad_token_id: int,
        *,
        selection_targets: torch.Tensor | None = None,
    ) -> RSM1InferenceOutput:
        (
            packet_prefix,
            memory,
            attention,
            problem,
            operations,
            candidate_logits,
            selected,
        ) = self._frozen_context(
            prompt_rows,
            problem_masks,
            packet_step_masks,
            operation_masks,
            final_masks,
            pad_token_id,
            selection_targets=selection_targets,
        )
        replay = self.replay(
            packet_prefix,
            memory,
            attention,
            problem,
            operations,
            selected,
        )
        return RSM1InferenceOutput(
            replay=replay,
            candidate_logits=candidate_logits,
            selected_candidates=selected,
        )

    def forward_batch(
        self,
        prompt_rows: list[list[int]],
        problem_masks: list[list[bool]],
        packet_step_masks: list[list[list[bool]]],
        operation_masks: list[list[list[bool]]],
        final_masks: list[list[bool]],
        selection_targets: list[int],
        initial_targets: torch.Tensor,
        free_targets: torch.Tensor,
        free_active: torch.Tensor,
        oracle_predecessors: torch.Tensor,
        oracle_targets: torch.Tensor,
        oracle_active: torch.Tensor,
        terminal_targets: torch.Tensor,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch = len(prompt_rows)
        device = self.text_model.embed_tokens.weight.device
        selected = torch.tensor(selection_targets, device=device, dtype=torch.long)
        (
            packet_prefix,
            memory,
            attention,
            problem,
            operations,
            _,
            selected,
        ) = self._frozen_context(
            prompt_rows,
            problem_masks,
            packet_step_masks,
            operation_masks,
            final_masks,
            pad_token_id,
            selection_targets=selected,
        )
        replay = self.replay(
            packet_prefix,
            memory,
            attention,
            problem,
            operations,
            selected,
        )
        shapes = {
            "initial": (batch, self.replay_config.state_slots),
            "trace": (
                batch,
                self.replay_config.max_trace_steps,
                self.replay_config.state_slots,
            ),
            "active": (batch, self.replay_config.max_trace_steps),
        }
        if (
            initial_targets.shape != shapes["initial"]
            or terminal_targets.shape != shapes["initial"]
            or free_targets.shape != shapes["trace"]
            or oracle_predecessors.shape != shapes["trace"]
            or oracle_targets.shape != shapes["trace"]
            or free_active.shape != shapes["active"]
            or oracle_active.shape != shapes["active"]
        ):
            raise ProductReasoningTrainError("RSM1 supervision tensor shape differs")
        initial_targets = initial_targets.to(device=device, dtype=torch.long)
        terminal_targets = terminal_targets.to(device=device, dtype=torch.long)
        free_targets = free_targets.to(device=device, dtype=torch.long)
        oracle_predecessors = oracle_predecessors.to(device=device, dtype=torch.long)
        oracle_targets = oracle_targets.to(device=device, dtype=torch.long)
        free_active = free_active.to(device=device, dtype=torch.bool)
        oracle_active = oracle_active.to(device=device, dtype=torch.bool)
        if not torch.equal(free_active, replay.replay_active):
            raise ProductReasoningTrainError("RSM1 free-running mask differs")
        if not torch.any(free_active) or not torch.any(oracle_active):
            raise ProductReasoningTrainError("RSM1 supervision contains no transition")

        initial_loss = F.cross_entropy(
            replay.initial_logits.reshape(-1, self.replay_config.state_vocab_size),
            initial_targets.reshape(-1),
        )
        free_loss = F.cross_entropy(
            replay.transition_logits[free_active].reshape(
                -1, self.replay_config.state_vocab_size
            ),
            free_targets[free_active].reshape(-1),
        )
        oracle_logits = self.replay.oracle_transition_logits(
            memory,
            attention,
            problem,
            operations,
            oracle_predecessors,
        )
        oracle_loss = F.cross_entropy(
            oracle_logits[oracle_active].reshape(
                -1, self.replay_config.state_vocab_size
            ),
            oracle_targets[oracle_active].reshape(-1),
        )
        loss = (initial_loss + free_loss + oracle_loss) / 3.0

        initial_exact = replay.state_trace_tokens[:, 0].eq(initial_targets).all(dim=1)
        free_exact = replay.state_trace_tokens[:, 1:].eq(free_targets).all(dim=2)
        free_correct = free_exact | ~free_active
        trajectory_exact = initial_exact & free_correct.all(dim=1)
        terminal_exact = replay.terminal_tokens.eq(terminal_targets).all(dim=1)
        oracle_exact = oracle_logits.argmax(dim=-1).eq(oracle_targets).all(dim=2)
        return loss, {
            "initial_loss": float(initial_loss.detach()),
            "free_loss": float(free_loss.detach()),
            "oracle_loss": float(oracle_loss.detach()),
            "initial_exact": float(initial_exact.float().mean()),
            "free_transition_exact": float(free_exact[free_active].float().mean()),
            "trajectory_exact": float(trajectory_exact.float().mean()),
            "terminal_exact": float(terminal_exact.float().mean()),
            "oracle_transition_exact": float(
                oracle_exact[oracle_active].float().mean()
            ),
            "mean_step_delta": float(replay.step_delta_norms.detach().mean()),
            "source_tokens": int(sum(len(row) for row in prompt_rows)),
            "state_target_tokens": int(
                initial_targets.numel()
                + free_active.sum().item() * self.replay_config.state_slots
                + oracle_active.sum().item() * self.replay_config.state_slots
            ),
            "candidate_predictions": selected.cpu().tolist(),
            "terminal_predictions": replay.terminal_tokens.detach().cpu().tolist(),
        }


def save_rsm1_checkpoint(
    path: Path,
    model: RSM1ProductModel,
    optimizer: torch.optim.Optimizer,
    update: int,
    metadata: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": RSM1_CHECKPOINT_SCHEMA,
            "update": int(update),
            "replay_state": {
                name: tensor.detach().cpu()
                for name, tensor in model.replay.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)


def load_rsm1_checkpoint(
    path: Path,
    model: RSM1ProductModel,
    *,
    load_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != RSM1_CHECKPOINT_SCHEMA:
        raise ProductReasoningTrainError("RSM1 checkpoint schema differs")
    metadata = payload.get("metadata")
    saved = payload.get("replay_state")
    if not isinstance(metadata, dict) or not isinstance(saved, dict):
        raise ProductReasoningTrainError("RSM1 checkpoint is incomplete")
    current = model.replay.state_dict()
    if set(saved) != set(current):
        raise ProductReasoningTrainError("RSM1 replay parameter contract differs")
    with torch.no_grad():
        for name, tensor in current.items():
            source = saved[name]
            if source.shape != tensor.shape:
                raise ProductReasoningTrainError(
                    f"RSM1 checkpoint tensor differs: {name}"
                )
            tensor.copy_(source.to(tensor.device, tensor.dtype))
    if load_optimizer is not None:
        optimizer_state = payload.get("optimizer")
        if not isinstance(optimizer_state, dict):
            raise ProductReasoningTrainError("RSM1 optimizer state is missing")
        load_optimizer.load_state_dict(optimizer_state)
    return int(payload["update"]), metadata


__all__ = [
    "RSM1_CHECKPOINT_SCHEMA",
    "RSM1InferenceOutput",
    "RSM1ProductModel",
    "load_rsm1_checkpoint",
    "module_state_sha256",
    "save_rsm1_checkpoint",
]
