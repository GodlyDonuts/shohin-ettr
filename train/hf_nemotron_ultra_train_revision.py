#!/usr/bin/env python3
"""Train the fixed-draft Shohin revision residual directly on Nemotron Ultra."""

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

from hf_nemotron_super_train_revision import (
    CONSUMED_PRESENTATIONS,
    DATA_PRESENTATIONS,
    DATA_SCHEMA,
    DATA_SHA256,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    SEED,
    UPDATES,
    _state_sha256,
    load_revision_rows,
    tokenize_consumed_rows,
)
from hf_nemotron_ultra_mechanics import (
    CAUSAL_CONV_VERSION,
    CUDA_VERSION,
    MAMBA_VERSION,
    MODELOPT_VERSION,
    OVERLAY_MANIFEST_SHA256,
    SCHEMA as MECHANICS_SCHEMA,
    TORCH_VERSION,
    _package_versions,
    verify_manifest,
)
from nemotron_ultra_post_mixer_revision import NemotronUltraRevisionModel
from q36_upward_moe_ultra_host import (
    MINIMUM_H100S,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    sha256_file,
)

SCHEMA = "shohin-nemotron-ultra-direct-revision-training-v1"
CHECKPOINT_SCHEMA = "shohin-nemotron-ultra-direct-revision-checkpoint-v1"


class NemotronUltraTrainingError(RuntimeError):
    """The directly trained 550B-A55B Shohin contract differed."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise NemotronUltraTrainingError("refusing existing training report")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_mechanics_report(path: Path, manifest_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NemotronUltraTrainingError("mechanics report is absent")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NemotronUltraTrainingError("mechanics report is unreadable") from error
    receipt = payload.get("model_receipt") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != MECHANICS_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("score_rows_read") != 0
        or payload.get("benchmark_rows_read") != 0
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("trainable_parameters") != TRAINABLE_PARAMETERS_PER_ROLE
        or payload.get("native_router_expert_trainables") != 0
        or payload.get("serialization_restore_exact") is not True
        or len(payload.get("devices", [])) != MINIMUM_H100S
        or not isinstance(receipt, dict)
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("exact_membership") is not True
    ):
        raise NemotronUltraTrainingError("mechanics authorization differs")
    return payload


def _module_origins(overlay_root: Path, causal_conv_root: Path) -> dict[str, str]:
    import causal_conv1d
    import mamba_ssm
    import modelopt

    origins = {
        "causal_conv1d": str(Path(causal_conv1d.__file__).resolve()),
        "mamba_ssm": str(Path(mamba_ssm.__file__).resolve()),
        "modelopt": str(Path(modelopt.__file__).resolve()),
    }
    if (
        not Path(origins["causal_conv1d"]).is_relative_to(causal_conv_root)
        or not Path(origins["mamba_ssm"]).is_relative_to(overlay_root)
        or not Path(origins["modelopt"]).is_relative_to(overlay_root)
    ):
        raise NemotronUltraTrainingError("training module origin differs")
    return origins


def run(args: argparse.Namespace) -> dict[str, Any]:
    from modelopt.torch.opt.plugins.huggingface import enable_huggingface_checkpointing
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    if args.output.exists() or args.output.is_symlink():
        raise NemotronUltraTrainingError("training output already exists")
    model_root = args.model_root.resolve(strict=True)
    overlay_root = args.overlay_root.resolve(strict=True)
    causal_conv_root = args.causal_conv_root.resolve(strict=True)
    if causal_conv_root.is_symlink() or not causal_conv_root.is_dir():
        raise NemotronUltraTrainingError("causal-conv root differs")
    model_receipt = verify_manifest(
        model_root,
        args.model_manifest,
        args.expected_model_manifest_sha256,
        exact_membership=True,
    )
    overlay_receipt = verify_manifest(
        overlay_root,
        args.overlay_manifest,
        OVERLAY_MANIFEST_SHA256,
        exact_membership=False,
    )
    validate_mechanics_report(
        args.mechanics_report, args.expected_model_manifest_sha256
    )
    if (
        sha256_file(model_root / "config.json") != MODEL_CONFIG_SHA256
        or (model_root / "SOURCE_REVISION").read_text(encoding="utf-8").strip()
        != MODEL_REVISION
    ):
        raise NemotronUltraTrainingError("Ultra model identity differs")
    load_pinned_config(model_root / "config.json")
    expected_versions = {
        "mamba-ssm": MAMBA_VERSION,
        "nvidia-modelopt": MODELOPT_VERSION,
        "causal-conv1d": CAUSAL_CONV_VERSION,
        "torch": TORCH_VERSION,
        "cuda": CUDA_VERSION,
    }
    package_versions = _package_versions()
    if package_versions != expected_versions:
        raise NemotronUltraTrainingError("training package versions differ")
    module_origins = _module_origins(overlay_root, causal_conv_root)
    if torch.cuda.device_count() != MINIMUM_H100S or any(
        "H100" not in torch.cuda.get_device_name(index).upper()
        for index in range(MINIMUM_H100S)
    ):
        raise NemotronUltraTrainingError("training requires exactly eight H100s")

    try:
        rows = load_revision_rows(args.data)
    except RuntimeError as error:
        raise NemotronUltraTrainingError(str(error)) from error
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        examples, sequence_receipt = tokenize_consumed_rows(tokenizer, rows)
    except RuntimeError as error:
        raise NemotronUltraTrainingError(str(error)) from error
    if len(examples) != CONSUMED_PRESENTATIONS:
        raise NemotronUltraTrainingError("consumed presentation count differs")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    enable_huggingface_checkpointing()
    backbone = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        device_map="balanced",
        max_memory={index: "77GiB" for index in range(MINIMUM_H100S)},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    device_map = getattr(backbone, "hf_device_map", None)
    if not isinstance(device_map, dict) or set(device_map.values()) != set(
        range(MINIMUM_H100S)
    ):
        raise NemotronUltraTrainingError("training device map differs")
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    model = NemotronUltraRevisionModel(backbone)
    if model.trainable_parameter_count() != TRAINABLE_PARAMETERS_PER_ROLE:
        raise NemotronUltraTrainingError("trainable parameter count differs")
    initial_state_sha256 = model.trainable_state_sha256()
    trainables = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainables,
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )
    optimizer.zero_grad(set_to_none=True)
    for index in range(MINIMUM_H100S):
        torch.cuda.reset_peak_memory_stats(index)
    trace: list[dict[str, Any]] = []
    charged_tokens = 0
    model.train()
    model.reset_receipt()
    input_device = backbone.model.embeddings.weight.device
    training_started = time.monotonic()
    for microstep, (prompt, response) in enumerate(examples, start=1):
        tokens = prompt + response
        labels = [-100] * len(prompt) + response
        input_ids = torch.tensor([tokens], dtype=torch.long, device=input_device)
        attention_mask = torch.ones_like(input_ids)
        label_tensor = torch.tensor([labels], dtype=torch.long, device=input_device)
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
                label_tensor.to(logits.device)[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            scaled_loss = loss / GRADIENT_ACCUMULATION
        if not bool(torch.isfinite(loss)):
            raise NemotronUltraTrainingError("training loss is nonfinite")
        scaled_loss.backward()
        charged_tokens += len(tokens)
        if microstep % GRADIENT_ACCUMULATION:
            continue
        update = microstep // GRADIENT_ACCUMULATION
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainables, 1.0)
        progress = (update - 1) / max(UPDATES - 1, 1)
        learning_rate = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))
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
    final_state_sha256 = _state_sha256(final_state)
    if final_state_sha256 == initial_state_sha256:
        raise NemotronUltraTrainingError("training produced no parameter update")
    args.output.mkdir(parents=True)
    checkpoint = args.output / "checkpoint_0000256.pt"
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp.{os.getpid()}")
    metadata = {
        "schema": SCHEMA,
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": args.expected_model_manifest_sha256,
        "data_schema": DATA_SCHEMA,
        "data_sha256": DATA_SHA256,
        "data_presentations": DATA_PRESENTATIONS,
        "updates": UPDATES,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "seed": SEED,
        "trainable_parameters": TRAINABLE_PARAMETERS_PER_ROLE,
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "native_router_expert_trainables": 0,
        "initial_trainable_state_sha256": initial_state_sha256,
        "final_trainable_state_sha256": final_state_sha256,
        "sequence_receipt": sequence_receipt,
        "mechanics_report_sha256": sha256_file(args.mechanics_report),
        "optimizer_state_serialized": False,
        "checkpoint_trainable_only": True,
        "revision_source": "direct_host_training",
    }
    with temporary.open("xb") as handle:
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "update": UPDATES,
                "trainable_state": final_state,
                "metadata": metadata,
            },
            handle,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "update", "trainable_state", "metadata"}
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("update") != UPDATES
        or payload.get("metadata") != metadata
        or _state_sha256(payload.get("trainable_state", {})) != final_state_sha256
    ):
        raise NemotronUltraTrainingError("training checkpoint restore differs")
    torch.cuda.synchronize()
    report = {
        **metadata,
        "status": "complete",
        "model_receipt": model_receipt,
        "overlay_receipt": overlay_receipt,
        "package_versions": package_versions,
        "module_origins": module_origins,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "serialization_restore_exact": True,
        "charged_tokens": charged_tokens,
        "elapsed_seconds": time.monotonic() - training_started,
        "total_elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in range(MINIMUM_H100S)
        },
        "routing_receipt": model.receipt(),
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--causal-conv-root", type=Path, required=True)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
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
