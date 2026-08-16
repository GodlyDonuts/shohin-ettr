#!/usr/bin/env python3
"""Qualify Mixtral Shohin mechanics over two independent one-H100 nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any

import torch
import torch.distributed as dist

from hf_mixtral_8x22b_mechanics import (
    EXPECTED_PACKAGES,
    SEED,
    MixtralMechanicsError,
    _atomic_json,
    _package_versions,
    _restore_trainables,
    _router_receipt,
    _state_sha256,
    verify_model_manifest,
)
from mixtral_post_mlp_revision import MixtralRevisionError, MixtralRevisionModel
from q36_upward_moe_mixtral_host import (
    MODEL_REVISION,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    sha256_file,
)

SCHEMA = "shohin-mixtral-8x22b-two-node-tp-mechanics-v1"
CHECKPOINT_SCHEMA = "shohin-mixtral-8x22b-two-node-tp-checkpoint-v1"
EXPECTED_WORLD_SIZE = 2


def _rank() -> int:
    return int(os.environ.get("RANK", "-1"))


def _require_world() -> tuple[int, int]:
    if not dist.is_available():
        raise MixtralMechanicsError("torch distributed is unavailable")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world != EXPECTED_WORLD_SIZE or rank not in range(world) or local_rank != 0:
        raise MixtralMechanicsError("distributed Mixtral world geometry differs")
    if torch.cuda.device_count() != 1:
        raise MixtralMechanicsError("each Mixtral rank requires exactly one H100")
    torch.cuda.set_device(0)
    device = torch.cuda.get_device_properties(0)
    if "H100" not in device.name.upper():
        raise MixtralMechanicsError("allocated distributed device is not H100")
    return rank, world


def _broadcast_object(value: Any, *, source: int = 0) -> Any:
    values = [value]
    dist.broadcast_object_list(values, src=source)
    return values[0]


def _synchronize_gradients(model: MixtralRevisionModel, world: int) -> None:
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise MixtralMechanicsError("distributed Shohin gradient differs")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


def _gradient_receipt(model: MixtralRevisionModel) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        finite = bool(gradient is not None and torch.isfinite(gradient).all())
        norm = float(gradient.float().norm().detach().cpu()) if finite else None
        rows.append({"name": name, "finite": finite, "norm": norm})
    if (
        len(rows) != 32
        or not all(row["finite"] for row in rows)
        or not any(float(row["norm"] or 0.0) > 0.0 for row in rows)
    ):
        raise MixtralMechanicsError("distributed Shohin gradient receipt differs")
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    return {
        "parameters": len(rows),
        "nonzero_gradients": sum(float(row["norm"] or 0.0) > 0.0 for row in rows),
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _gather_rank_receipts(value: dict[str, Any], world: int) -> list[dict[str, Any]]:
    gathered: list[dict[str, Any] | None] = [None] * world
    dist.all_gather_object(gathered, value)
    if any(item is None for item in gathered):
        raise MixtralMechanicsError("distributed rank receipt is incomplete")
    return [dict(item) for item in gathered if item is not None]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    rank, world = _require_world()
    model_root = args.model_root.resolve(strict=True)
    output = args.output.resolve(strict=False)
    checkpoint = output.with_suffix(".checkpoint.pt")
    if (
        output.exists()
        or output.is_symlink()
        or checkpoint.exists()
        or checkpoint.is_symlink()
    ):
        raise MixtralMechanicsError("refusing existing distributed mechanics output")

    model_receipt = None
    if rank == 0:
        model_receipt = verify_model_manifest(
            model_root, args.model_manifest, args.expected_model_manifest_sha256
        )
    model_receipt = _broadcast_object(model_receipt)
    if not isinstance(model_receipt, dict):
        raise MixtralMechanicsError("distributed model receipt differs")
    load_pinned_config(model_root / "config.json")
    revision_receipt = model_root / "SOURCE_REVISION"
    if (
        revision_receipt.is_symlink()
        or not revision_receipt.is_file()
        or revision_receipt.read_text(encoding="utf-8").strip() != MODEL_REVISION
    ):
        raise MixtralMechanicsError("model revision receipt differs")
    versions = _package_versions()
    if versions != EXPECTED_PACKAGES:
        raise MixtralMechanicsError("distributed mechanics package versions differ")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.distributed.configuration_utils import DistributedConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        distributed_config=DistributedConfig(tp_size=world),
        quantization_config=quantization,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    distributed_config = getattr(backbone.config, "distributed_config", None)
    if (
        distributed_config is None
        or distributed_config.tp_size != world
        or distributed_config.fsdp_size != 1
        or getattr(backbone, "_device_mesh", None) is None
        or getattr(backbone, "hf_device_map", None) is not None
    ):
        raise MixtralMechanicsError("native tensor-parallel load receipt differs")

    model = MixtralRevisionModel(backbone)
    if model.trainable_parameter_count() != TRAINABLE_PARAMETERS_PER_ROLE:
        raise MixtralMechanicsError("distributed trainable surface differs")
    initial_state = model.trainable_state()
    initial_sha256 = _state_sha256(initial_state)
    initial_hashes = _gather_rank_receipts({"sha256": initial_sha256}, world)
    if len({row["sha256"] for row in initial_hashes}) != 1:
        raise MixtralMechanicsError("replicated trainable initialization differs")

    router_before = _router_receipt(model)
    token_ids = tokenizer.encode(
        "Shohin distributed Mixtral mechanics only.", add_special_tokens=False
    )
    if not token_ids or len(token_ids) > 16:
        raise MixtralMechanicsError("distributed mechanics tokenization differs")
    input_ids = torch.tensor([token_ids], device="cuda:0", dtype=torch.long)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        foreach=False,
    )
    torch.cuda.reset_peak_memory_stats(0)
    output_payload = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
    )
    logits = output_payload.logits
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise MixtralMechanicsError("distributed full-model forward geometry differs")
    loss = logits[:, -1, :128].float().square().mean()
    if not bool(torch.isfinite(loss)):
        raise MixtralMechanicsError("distributed mechanics loss is nonfinite")
    loss.backward()
    _synchronize_gradients(model, world)
    gradients = _gradient_receipt(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if _router_receipt(model) != router_before:
        raise MixtralMechanicsError("distributed mechanics changed a native router")
    updated_state = model.trainable_state()
    updated_sha256 = _state_sha256(updated_state)
    rank_receipts = _gather_rank_receipts(
        {
            "rank": rank,
            "hostname": socket.gethostname(),
            "device_name": torch.cuda.get_device_properties(0).name,
            "device_total_memory": torch.cuda.get_device_properties(0).total_memory,
            "initial_state_sha256": initial_sha256,
            "updated_state_sha256": updated_sha256,
            "gradient_receipt_sha256": gradients["receipt_sha256"],
            "router_before_sha256": router_before,
            "router_after_sha256": _router_receipt(model),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(0),
        },
        world,
    )
    for field in (
        "initial_state_sha256",
        "updated_state_sha256",
        "gradient_receipt_sha256",
        "router_before_sha256",
        "router_after_sha256",
    ):
        if len({row[field] for row in rank_receipts}) != 1:
            raise MixtralMechanicsError(f"distributed {field} differs between ranks")
    if updated_sha256 == initial_sha256:
        raise MixtralMechanicsError("distributed Shohin update is an exact no-op")

    if rank == 0:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_name(f".{checkpoint.name}.tmp.{os.getpid()}")
        with temporary.open("xb") as handle:
            torch.save(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "trainable_state": updated_state,
                    "trainable_state_sha256": updated_sha256,
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint)
    dist.barrier()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("trainable_state_sha256") != updated_sha256
        or _state_sha256(payload.get("trainable_state", {})) != updated_sha256
    ):
        raise MixtralMechanicsError("distributed mechanics checkpoint differs")
    with torch.no_grad():
        next(
            parameter for parameter in model.parameters() if parameter.requires_grad
        ).zero_()
    _restore_trainables(model, payload["trainable_state"])
    if model.trainable_state_sha256() != updated_sha256:
        raise MixtralMechanicsError("distributed mechanics restore differs")

    report = {
        "schema": SCHEMA,
        "status": "pass",
        "seed": SEED,
        "score_rows_read": 0,
        "benchmark_rows_read": 0,
        "model_revision": MODEL_REVISION,
        "model_receipt": model_receipt,
        "package_versions": versions,
        "world_size": world,
        "backend": dist.get_backend(),
        "parallelism": "native-transformers-tensor-parallel",
        "quantization": "nf4",
        "rank_receipts": rank_receipts,
        "trainable_parameters": model.trainable_parameter_count(),
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "initial_trainable_state_sha256": initial_sha256,
        "updated_trainable_state_sha256": updated_sha256,
        "gradient_receipt": gradients,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "serialization_restore_exact": True,
        "native_router_expert_trainables": 0,
        "native_router_unchanged": True,
        "routing_receipt": model.receipt(),
        "elapsed_seconds": time.monotonic() - started,
    }
    if any(
        not math.isfinite(float(row["peak_gpu_memory_bytes"])) for row in rank_receipts
    ):
        raise MixtralMechanicsError("distributed GPU memory receipt differs")
    if rank == 0:
        _atomic_json(output, report)
    dist.barrier()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = run(parse_args())
    except (MixtralMechanicsError, MixtralRevisionError) as error:
        raise SystemExit(str(error)) from error
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    if _rank() == 0:
        print(json.dumps(result, sort_keys=True), flush=True)
