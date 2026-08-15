#!/usr/bin/env python3
"""Warm-start and train the draft-visible aligned trajectory on an upward MoE host."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from hf_product_reasoning_train import _batches, reservoir_rows_with_sha256
from hf_q36_mtr_train_role import tokenize_role_rows, training_consumption_receipt
from hf_upward_moe_train_owner import (
    UpwardMoEOwnerTrainingError,
    _atomic_json,
    _load_host,
)
from upward_moe_role_lineage import (
    UpwardMoERoleLineageError,
    load_role_checkpoint,
    save_role_checkpoint,
    sha256_file,
)
from upward_moe_temporal_gate import MIXTRAL_SPEC, NEMOTRON_SPEC, ULTRA_SPEC

SCHEMA = "shohin-upward-moe-aligned-training-v1"
DATA_SCHEMA = "shohin-q36-mtr-revision-train-v1"
ALIGNED_PRESENTATIONS = 9_655
ALIGNED_UPDATES = 256
ALIGNED_GRADIENT_ACCUMULATION = 8
ALIGNED_CONSUMED_PRESENTATIONS = ALIGNED_UPDATES * ALIGNED_GRADIENT_ACCUMULATION
ALIGNED_MAX_SEQUENCE_LENGTH = 4_096
ALIGNED_LEARNING_RATE = 2e-5
ALIGNED_SEED = 2026080815
ALIGNED_DATA_SEED = 2026080814


class UpwardMoEAlignedTrainingError(RuntimeError):
    """The upward MoE aligned trajectory or its owner warm start differed."""


def _hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def static_aligned_contract() -> dict[str, Any]:
    return {
        "role": "aligned",
        "data_kind": "host_owned_natural_trajectory_revision",
        "warm_start_role": "owner",
        "presentations": ALIGNED_PRESENTATIONS,
        "updates": ALIGNED_UPDATES,
        "gradient_accumulation": ALIGNED_GRADIENT_ACCUMULATION,
        "consumed_presentations": ALIGNED_CONSUMED_PRESENTATIONS,
        "max_sequence_length": ALIGNED_MAX_SEQUENCE_LENGTH,
        "learning_rate": ALIGNED_LEARNING_RATE,
        "seed": ALIGNED_SEED,
        "data_seed": ALIGNED_DATA_SEED,
        "internal_draft_visible": True,
        "external_proposer": False,
        "task_router": False,
        "native_router_expert_trainables": 0,
        "hosts": [
            NEMOTRON_SPEC.receipt(),
            MIXTRAL_SPEC.receipt(),
            ULTRA_SPEC.receipt(),
        ],
    }


def validate_aligned_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != ALIGNED_PRESENTATIONS:
        raise UpwardMoEAlignedTrainingError("aligned row population differs")
    identities = set()
    for row in rows:
        identity = row.get("identity_sha256") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("schema") != DATA_SCHEMA
            or not _hex_digest(identity)
            or identity in identities
            or row.get("internal_draft_visible") is not True
            or row.get("external_candidate_text_visible") is not False
            or row.get("runtime_fields") != ["question"]
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or "Internal model-owned draft:" not in row["question"]
            or not isinstance(row.get("response"), str)
            or not row["response"].strip()
        ):
            raise UpwardMoEAlignedTrainingError("aligned row differs")
        identities.add(identity)
    return rows


def load_aligned_rows(
    path: Path, expected_sha256: str
) -> tuple[list[dict[str, Any]], str]:
    if (
        not _hex_digest(expected_sha256)
        or path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != expected_sha256
    ):
        raise UpwardMoEAlignedTrainingError("aligned data bytes differ")
    rows, digest = reservoir_rows_with_sha256(
        path, ALIGNED_PRESENTATIONS, ALIGNED_DATA_SEED
    )
    if digest != expected_sha256:
        raise UpwardMoEAlignedTrainingError("aligned data digest differs")
    return validate_aligned_rows(rows), digest


def restore_exact_owner(
    model: Any, owner_checkpoint: Path, spec: Any
) -> dict[str, Any]:
    try:
        payload = load_role_checkpoint(owner_checkpoint, spec)
    except UpwardMoERoleLineageError as error:
        raise UpwardMoEAlignedTrainingError(str(error)) from error
    metadata = payload["metadata"]
    saved = payload["trainable_state"]
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if metadata["role"] != "owner" or set(current) != set(saved):
        raise UpwardMoEAlignedTrainingError("aligned owner state names differ")
    with torch.no_grad():
        for name, parameter in current.items():
            tensor = saved[name]
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise UpwardMoEAlignedTrainingError(
                    "aligned owner state geometry differs"
                )
            parameter.copy_(tensor.to(device=parameter.device))
    observed = model.trainable_state_sha256()
    if observed != metadata["final_trainable_state_sha256"]:
        raise UpwardMoEAlignedTrainingError("aligned owner restore differs")
    return {
        "owner_checkpoint_sha256": sha256_file(owner_checkpoint),
        "owner_state_sha256": observed,
        "owner_restore_exact": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.output.exists() or args.output.is_symlink():
        raise UpwardMoEAlignedTrainingError("aligned output exists")
    rows, data_sha256 = load_aligned_rows(args.data, args.expected_data_sha256)
    random.seed(ALIGNED_SEED)
    torch.manual_seed(ALIGNED_SEED)
    torch.cuda.manual_seed_all(ALIGNED_SEED)
    try:
        loaded = _load_host(args)
    except UpwardMoEOwnerTrainingError as error:
        raise UpwardMoEAlignedTrainingError(str(error)) from error
    if loaded.tokenizer.pad_token_id is None:
        loaded.tokenizer.pad_token_id = loaded.tokenizer.eos_token_id
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
    if (
        consumption["consumed_presentations"] != ALIGNED_CONSUMED_PRESENTATIONS
        or sequence_receipt["draft_masked_tokens"] <= 0
    ):
        raise UpwardMoEAlignedTrainingError("aligned consumption differs")
    model = loaded.model
    owner_receipt = restore_exact_owner(model, args.owner_checkpoint, loaded.spec)
    initial_state_sha256 = model.trainable_state_sha256()
    if initial_state_sha256 != owner_receipt["owner_state_sha256"]:
        raise UpwardMoEAlignedTrainingError("aligned initial state differs")
    trainables = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    backbone = model.backbone
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        trainables,
        lr=ALIGNED_LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )
    optimizer.zero_grad(set_to_none=True)
    batches = list(_batches(examples, 1))
    trace = []
    charged_tokens = 0
    model.train()
    model.reset_receipt()
    training_started = time.monotonic()
    for microstep in range(1, ALIGNED_CONSUMED_PRESENTATIONS + 1):
        prompt, response, _ = batches[(microstep - 1) % len(batches)][0]
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
            raise UpwardMoEAlignedTrainingError("aligned loss is nonfinite")
        scaled_loss.backward()
        charged_tokens += len(tokens)
        if microstep % ALIGNED_GRADIENT_ACCUMULATION:
            continue
        update = microstep // ALIGNED_GRADIENT_ACCUMULATION
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainables, 1.0)
        progress = (update - 1) / max(ALIGNED_UPDATES - 1, 1)
        learning_rate = (
            ALIGNED_LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if update == 1 or update % 8 == 0:
            event = {
                "update": update,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "charged_tokens": charged_tokens,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
    final_state = model.trainable_state()
    training_receipt = {
        "schema": SCHEMA,
        "role": "aligned",
        "host": loaded.spec.host,
        "data_sha256": data_sha256,
        "sequence_receipt": sequence_receipt,
        "consumption_receipt": consumption,
        "model_receipt": loaded.model_receipt,
        "mechanics_report_sha256": sha256_file(args.mechanics_report),
        "owner_lineage": owner_receipt,
        "trace": trace,
        "charged_tokens": charged_tokens,
        "elapsed_seconds": time.monotonic() - training_started,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    checkpoint = args.output / "checkpoint_0000256.pt"
    restored = save_role_checkpoint(
        checkpoint,
        role="aligned",
        state=final_state,
        spec=loaded.spec,
        initial_state_sha256=initial_state_sha256,
        training_receipt=training_receipt,
        warm_start_checkpoint=args.owner_checkpoint,
    )
    torch.cuda.synchronize()
    report = {
        **training_receipt,
        "status": "complete",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "initial_trainable_state_sha256": initial_state_sha256,
        "final_trainable_state_sha256": restored["metadata"][
            "final_trainable_state_sha256"
        ],
        "serialization_restore_exact": True,
        "total_elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in range(torch.cuda.device_count())
        },
        "routing_receipt": model.receipt(),
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        choices=("nemotron-super", "mixtral-8x22b", "nemotron-ultra"),
        required=True,
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--expected-model-manifest-sha256")
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--overlay-manifest", type=Path)
    parser.add_argument("--causal-conv-root", type=Path)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            },
            sort_keys=True,
        )
    )
