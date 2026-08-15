#!/usr/bin/env python3
"""Train the causal-only temporal gate on one pinned upward MoE host."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from hf_product_reasoning_train import _batches
from hf_q36_mtr_train_role import tokenize_role_rows, training_consumption_receipt
from hf_upward_moe_train_aligned import (
    ALIGNED_CONSUMED_PRESENTATIONS,
    ALIGNED_GRADIENT_ACCUMULATION,
    ALIGNED_MAX_SEQUENCE_LENGTH,
    ALIGNED_UPDATES,
    load_aligned_rows,
)
from hf_upward_moe_train_owner import (
    UpwardMoEOwnerTrainingError,
    _atomic_json,
    _load_host,
)
from upward_moe_role_lineage import (
    UpwardMoERoleLineageError,
    load_role_pair,
    sha256_file,
)
from upward_moe_temporal_gate import (
    MIXTRAL_SPEC,
    NEMOTRON_SPEC,
    MixtralTemporalGateModel,
    NemotronSuperTemporalGateModel,
    UpwardMoETemporalGateError,
)

SCHEMA = "shohin-upward-moe-temporal-gate-training-v1"
CHECKPOINT_SCHEMA = "shohin-upward-moe-temporal-gate-checkpoint-v1"
GATE_SEED = 2026081511
GATE_LEARNING_RATE = 2e-4
GATE_INITIAL_REVISION_WEIGHT = 0.1
GATE_CAUSAL_LOSS_WEIGHT = 1.0
GATE_ROUTING_SUPERVISION_WEIGHT = 0.0


class UpwardMoETemporalTrainingError(RuntimeError):
    """The upward-MoE temporal training contract differed."""


def host_spec(host: str) -> Any:
    if host == "nemotron-super":
        return NEMOTRON_SPEC
    if host == "mixtral-8x22b":
        return MIXTRAL_SPEC
    raise UpwardMoETemporalTrainingError("upward temporal host differs")


def _validate_gate_state(state: Any, spec: Any) -> dict[str, torch.Tensor]:
    if (
        not isinstance(state, dict)
        or len(state) != 2 * len(spec.controlled_layer_indices)
        or sum(tensor.numel() for tensor in state.values())
        != spec.gate_trainable_parameters
        or any(
            not isinstance(name, str)
            or not name.endswith(("gate_weight", "gate_bias"))
            or not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
            for name, tensor in state.items()
        )
    ):
        raise UpwardMoETemporalTrainingError("upward temporal gate state differs")
    return state


def save_gate_checkpoint(
    path: Path,
    model: Any,
    metadata: dict[str, Any],
    spec: Any,
) -> str:
    if path.exists() or path.is_symlink():
        raise UpwardMoETemporalTrainingError("upward temporal checkpoint exists")
    state = _validate_gate_state(model.trainable_state(), spec)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "update": ALIGNED_UPDATES,
        "trainable_state": state,
        "metadata": metadata,
    }
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def restore_gate_checkpoint(path: Path, model: Any, spec: Any) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise UpwardMoETemporalTrainingError("upward temporal checkpoint is absent")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "update", "trainable_state", "metadata"}
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("update") != ALIGNED_UPDATES
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise UpwardMoETemporalTrainingError("upward temporal checkpoint differs")
    saved = _validate_gate_state(payload["trainable_state"], spec)
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(saved) != set(current):
        raise UpwardMoETemporalTrainingError("upward temporal restore names differ")
    with torch.no_grad():
        for name, parameter in current.items():
            tensor = saved[name]
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise UpwardMoETemporalTrainingError(
                    "upward temporal restore geometry differs"
                )
            parameter.copy_(tensor.to(parameter.device))
    if model.trainable_state_sha256() != payload["metadata"].get(
        "final_trainable_state_sha256"
    ):
        raise UpwardMoETemporalTrainingError("upward temporal restore hash differs")
    return payload["metadata"]


def static_gate_contract() -> dict[str, Any]:
    return {
        "architecture": "host-owned-owner-aligned-causal-temporal-gate",
        "updates": ALIGNED_UPDATES,
        "gradient_accumulation": ALIGNED_GRADIENT_ACCUMULATION,
        "consumed_presentations": ALIGNED_CONSUMED_PRESENTATIONS,
        "max_sequence_length": ALIGNED_MAX_SEQUENCE_LENGTH,
        "learning_rate": GATE_LEARNING_RATE,
        "seed": GATE_SEED,
        "initial_revision_weight": GATE_INITIAL_REVISION_WEIGHT,
        "causal_loss_weight": GATE_CAUSAL_LOSS_WEIGHT,
        "routing_supervision_weight": GATE_ROUTING_SUPERVISION_WEIGHT,
        "native_router_expert_trainables": 0,
        "hosts": [NEMOTRON_SPEC.receipt(), MIXTRAL_SPEC.receipt()],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    spec = host_spec(args.host)
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.seed != GATE_SEED
        or args.learning_rate != GATE_LEARNING_RATE
        or args.initial_revision_weight != GATE_INITIAL_REVISION_WEIGHT
        or args.causal_loss_weight != GATE_CAUSAL_LOSS_WEIGHT
        or args.routing_supervision_weight != GATE_ROUTING_SUPERVISION_WEIGHT
    ):
        raise UpwardMoETemporalTrainingError("upward temporal settings differ")
    rows, data_sha256 = load_aligned_rows(args.data, args.expected_data_sha256)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    try:
        loaded = _load_host(args, attach_revision=False)
        owner_state, revision_state, role_receipt = load_role_pair(
            args.owner_checkpoint, args.revision_checkpoint, spec
        )
    except (UpwardMoEOwnerTrainingError, UpwardMoERoleLineageError) as error:
        raise UpwardMoETemporalTrainingError(str(error)) from error
    if loaded.spec != spec:
        raise UpwardMoETemporalTrainingError("upward temporal loaded host differs")
    if loaded.tokenizer.pad_token_id is None:
        loaded.tokenizer.pad_token_id = loaded.tokenizer.eos_token_id
    try:
        model = (
            NemotronSuperTemporalGateModel(loaded.model, owner_state, revision_state)
            if args.host == "nemotron-super"
            else MixtralTemporalGateModel(loaded.model, owner_state, revision_state)
        )
    except UpwardMoETemporalGateError as error:
        raise UpwardMoETemporalTrainingError(str(error)) from error
    prompts, responses, draft_masks, sequence_receipt = tokenize_role_rows(
        loaded.tokenizer,
        rows,
        role="aligned",
        max_sequence_length=ALIGNED_MAX_SEQUENCE_LENGTH,
    )
    examples = list(zip(prompts, responses, draft_masks, strict=True))
    consumption = training_consumption_receipt(
        examples,
        updates=ALIGNED_UPDATES,
        gradient_accumulation=ALIGNED_GRADIENT_ACCUMULATION,
        batch_size=1,
    )
    if consumption["consumed_presentations"] != ALIGNED_CONSUMED_PRESENTATIONS:
        raise UpwardMoETemporalTrainingError("upward temporal consumption differs")
    initial_state_sha256 = model.trainable_state_sha256()
    trainables = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if model.trainable_parameter_count() != spec.gate_trainable_parameters or any(
        parameter.dtype != torch.float32 for parameter in trainables
    ):
        raise UpwardMoETemporalTrainingError("upward temporal trainables differ")
    backbone = model.backbone
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        trainables,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )
    optimizer.zero_grad(set_to_none=True)
    batches = list(_batches(examples, 1))
    update = microstep = charged_tokens = 0
    trace = []
    model.train()
    model.reset_receipt()
    torch.cuda.reset_peak_memory_stats()
    training_started = time.monotonic()
    while update < ALIGNED_UPDATES:
        prompt, response, _ = batches[microstep % len(batches)][0]
        tokens = prompt + response
        labels = [-100] * len(prompt) + response
        input_ids = torch.tensor([tokens], dtype=torch.long, device=loaded.input_device)
        attention_mask = torch.ones_like(input_ids)
        label_tensor = torch.tensor(
            [labels], dtype=torch.long, device=loaded.input_device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                label_tensor[:, 1:].to(logits.device).reshape(-1),
                ignore_index=-100,
            )
            scaled_loss = loss / ALIGNED_GRADIENT_ACCUMULATION
        if not bool(torch.isfinite(loss)):
            raise UpwardMoETemporalTrainingError("upward temporal loss is nonfinite")
        scaled_loss.backward()
        charged_tokens += len(tokens)
        microstep += 1
        if microstep % ALIGNED_GRADIENT_ACCUMULATION:
            continue
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainables, 1.0)
        progress = update / max(ALIGNED_UPDATES - 1, 1)
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update == 1 or update % 8 == 0:
            event = {
                "update": update,
                "loss": float(loss.detach()),
                "causal_loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "charged_tokens": charged_tokens,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
    final_state_sha256 = model.trainable_state_sha256()
    if final_state_sha256 == initial_state_sha256:
        raise UpwardMoETemporalTrainingError("upward temporal update is absent")
    metadata = {
        "schema": SCHEMA,
        "architecture": spec.architecture,
        "host_contract": spec.receipt(),
        "model_revision": spec.model_revision,
        "model_config_sha256": spec.model_config_sha256,
        "controlled_layer_indices": list(spec.controlled_layer_indices),
        "gate_parameters": spec.gate_trainable_parameters,
        "initial_revision_weight": args.initial_revision_weight,
        "causal_loss_weight": args.causal_loss_weight,
        "routing_supervision_weight": args.routing_supervision_weight,
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "initial_trainable_state_sha256": initial_state_sha256,
        "final_trainable_state_sha256": final_state_sha256,
        "data_sha256": data_sha256,
        "sequence_receipt": sequence_receipt,
        "training_consumption": consumption,
        "role_receipt": role_receipt,
        "model_receipt": loaded.model_receipt,
        "mechanics_report_sha256": sha256_file(args.mechanics_report),
        "optimizer_state_serialized": False,
        "router_expert_checkpoint_tensors": 0,
        "native_router_expert_trainables": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    args.output.mkdir(parents=True)
    checkpoint = args.output / "checkpoint_0000256.pt"
    checkpoint_sha256 = save_gate_checkpoint(checkpoint, model, metadata, spec)
    with torch.no_grad():
        for parameter in trainables:
            parameter.zero_()
    restored = restore_gate_checkpoint(checkpoint, model, spec)
    if restored != metadata:
        raise UpwardMoETemporalTrainingError("upward temporal metadata restore differs")
    torch.cuda.synchronize()
    report = {
        **metadata,
        "status": "complete",
        "updates": update,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": ALIGNED_GRADIENT_ACCUMULATION,
        "batch_size": 1,
        "seed": args.seed,
        "charged_tokens": charged_tokens,
        "elapsed_seconds": time.monotonic() - training_started,
        "total_elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in range(2)
        },
        "routing_receipt": model.receipt(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "serialization_restore_exact": True,
        "trace": trace,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", choices=("nemotron-super", "mixtral-8x22b"), required=True
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--expected-model-manifest-sha256")
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--overlay-manifest", type=Path)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--revision-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=GATE_SEED)
    parser.add_argument("--learning-rate", type=float, default=GATE_LEARNING_RATE)
    parser.add_argument(
        "--initial-revision-weight", type=float, default=GATE_INITIAL_REVISION_WEIGHT
    )
    parser.add_argument(
        "--causal-loss-weight", type=float, default=GATE_CAUSAL_LOSS_WEIGHT
    )
    parser.add_argument(
        "--routing-supervision-weight",
        type=float,
        default=GATE_ROUTING_SUPERVISION_WEIGHT,
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
