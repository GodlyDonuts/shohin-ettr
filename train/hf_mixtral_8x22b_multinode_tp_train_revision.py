#!/usr/bin/env python3
"""Train the fixed-draft Mixtral revision with four-rank native BF16 TP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from hf_mixtral_8x22b_mechanics import EXPECTED_PACKAGES, verify_model_manifest
from hf_mixtral_8x22b_multinode_tp_mechanics import (
    EXPECTED_WORLD_SIZE,
    SCHEMA as MECHANICS_SCHEMA,
    _broadcast_object,
    _gather_rank_receipts,
    _require_world,
    _synchronize_gradients,
)
from hf_mixtral_8x22b_train_revision import (
    CONSUMED_PRESENTATIONS,
    DATA_SCHEMA,
    DATA_SHA256,
    DRAFT_ORIGIN_MODEL,
    DRAFT_ORIGIN_REVISION,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    SEED,
    TRANSFER_SCOPE,
    UPDATES,
    _package_versions,
    _state_sha256,
    load_revision_rows,
    tokenize_consumed_rows,
)
from mixtral_post_mlp_revision import MixtralRevisionError, MixtralRevisionModel
from q36_upward_moe_mixtral_host import (
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    sha256_file,
)

SCHEMA = "shohin-mixtral-8x22b-bf16-tp4-fixed-draft-training-v1"
CHECKPOINT_SCHEMA = "shohin-mixtral-8x22b-bf16-tp4-revision-checkpoint-v1"


class MixtralDistributedTrainingError(RuntimeError):
    """The distributed upward-MoE training contract differed."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise MixtralDistributedTrainingError("refusing existing training report")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_mechanics(
    path: Path, expected_model_manifest_sha256: str
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MixtralDistributedTrainingError("mechanics report is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != MECHANICS_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("world_size") != EXPECTED_WORLD_SIZE
        or payload.get("parallelism") != "native-transformers-tensor-parallel"
        or payload.get("weight_dtype") != "bfloat16"
        or payload.get("quantization") != "none"
        or payload.get("score_rows_read") != 0
        or payload.get("benchmark_rows_read") != 0
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("model_receipt", {}).get("manifest_sha256")
        != expected_model_manifest_sha256
        or payload.get("trainable_parameters") != TRAINABLE_PARAMETERS_PER_ROLE
        or payload.get("native_router_expert_trainables") != 0
        or payload.get("native_router_unchanged") is not True
        or payload.get("serialization_restore_exact") is not True
        or len(payload.get("rank_receipts", [])) != EXPECTED_WORLD_SIZE
    ):
        raise MixtralDistributedTrainingError("mechanics authorization differs")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    rank, world = _require_world()
    if args.output.exists() or args.output.is_symlink():
        raise MixtralDistributedTrainingError("training output already exists")

    mechanics = None
    model_receipt = None
    if rank == 0:
        mechanics = _validate_mechanics(
            args.mechanics_report, args.expected_model_manifest_sha256
        )
        model_receipt = verify_model_manifest(
            args.model_root.resolve(strict=True),
            args.model_manifest,
            args.expected_model_manifest_sha256,
        )
    mechanics = _broadcast_object(mechanics)
    model_receipt = _broadcast_object(model_receipt)
    if not isinstance(mechanics, dict) or not isinstance(model_receipt, dict):
        raise MixtralDistributedTrainingError("distributed authorization differs")

    model_root = args.model_root.resolve(strict=True)
    if (
        sha256_file(model_root / "config.json") != MODEL_CONFIG_SHA256
        or (model_root / "SOURCE_REVISION").is_symlink()
        or not (model_root / "SOURCE_REVISION").is_file()
        or (model_root / "SOURCE_REVISION").read_text().strip() != MODEL_REVISION
    ):
        raise MixtralDistributedTrainingError("host identity differs")
    load_pinned_config(model_root / "config.json")
    if _package_versions() != EXPECTED_PACKAGES:
        raise MixtralDistributedTrainingError("training package versions differ")

    rows = load_revision_rows(args.data)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.distributed.configuration_utils import DistributedConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    examples, sequence_receipt = tokenize_consumed_rows(tokenizer, rows)
    sequence_receipts = _gather_rank_receipts(sequence_receipt, world)
    sequence_digest = hashlib.sha256(
        json.dumps(sequence_receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if len(examples) != CONSUMED_PRESENTATIONS or any(
        item != sequence_receipt for item in sequence_receipts
    ):
        raise MixtralDistributedTrainingError("distributed tokenization differs")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    backbone = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        distributed_config=DistributedConfig(tp_size=world),
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
        raise MixtralDistributedTrainingError("native tensor-parallel load differs")
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    model = MixtralRevisionModel(backbone)
    initial_state_sha256 = model.trainable_state_sha256()
    initial_receipts = _gather_rank_receipts(
        {"initial_state_sha256": initial_state_sha256}, world
    )
    if len({item["initial_state_sha256"] for item in initial_receipts}) != 1:
        raise MixtralDistributedTrainingError("trainable initialization differs")

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
    torch.cuda.reset_peak_memory_stats(0)
    trace: list[dict[str, Any]] = []
    charged_tokens = 0
    model.train()
    model.reset_receipt()
    training_started = time.monotonic()
    for microstep, (prompt, response) in enumerate(examples, start=1):
        tokens = prompt + response
        labels = [-100] * len(prompt) + response
        input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda:0")
        attention_mask = torch.ones_like(input_ids)
        label_tensor = torch.tensor([labels], dtype=torch.long, device="cuda:0")
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
                label_tensor[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            scaled_loss = loss / GRADIENT_ACCUMULATION
        if not bool(torch.isfinite(loss)):
            raise MixtralDistributedTrainingError("training loss is nonfinite")
        scaled_loss.backward()
        charged_tokens += len(tokens)
        if microstep % GRADIENT_ACCUMULATION:
            continue
        update = microstep // GRADIENT_ACCUMULATION
        _synchronize_gradients(model, world)
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
            if rank == 0:
                print(json.dumps(event, sort_keys=True), flush=True)

    final_state = model.trainable_state()
    final_state_sha256 = _state_sha256(final_state)
    rank_receipts = _gather_rank_receipts(
        {
            "rank": rank,
            "hostname": socket.gethostname(),
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "charged_tokens": charged_tokens,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        },
        world,
    )
    for field in ("initial_state_sha256", "final_state_sha256", "charged_tokens"):
        if len({item[field] for item in rank_receipts}) != 1:
            raise MixtralDistributedTrainingError(f"distributed {field} differs")
    if final_state_sha256 == initial_state_sha256:
        raise MixtralDistributedTrainingError("training produced no parameter update")

    checkpoint = args.output / "checkpoint_0000256.pt"
    metadata = {
        "schema": SCHEMA,
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": args.expected_model_manifest_sha256,
        "model_receipt": model_receipt,
        "draft_origin_model": DRAFT_ORIGIN_MODEL,
        "draft_origin_revision": DRAFT_ORIGIN_REVISION,
        "transfer_scope": TRANSFER_SCOPE,
        "standalone_mixtral_owned_draft": False,
        "data_schema": DATA_SCHEMA,
        "data_sha256": DATA_SHA256,
        "updates": UPDATES,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "seed": SEED,
        "world_size": world,
        "parallelism": "native-transformers-tensor-parallel",
        "weight_dtype": "bfloat16",
        "quantization": "none",
        "trainable_parameters": TRAINABLE_PARAMETERS_PER_ROLE,
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "native_router_expert_trainables": 0,
        "initial_trainable_state_sha256": initial_state_sha256,
        "final_trainable_state_sha256": final_state_sha256,
        "sequence_receipt": sequence_receipt,
        "sequence_receipt_sha256": sequence_digest,
        "mechanics_report_sha256": sha256_file(args.mechanics_report),
        "optimizer_state_serialized": False,
        "checkpoint_trainable_only": True,
    }
    if rank == 0:
        args.output.mkdir(parents=True)
        temporary = checkpoint.with_name(f".{checkpoint.name}.tmp.{os.getpid()}")
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "update": UPDATES,
                "trainable_state": final_state,
                "metadata": metadata,
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
    dist.barrier()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "update", "trainable_state", "metadata"}
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("update") != UPDATES
        or payload.get("metadata") != metadata
        or _state_sha256(payload.get("trainable_state", {})) != final_state_sha256
    ):
        raise MixtralDistributedTrainingError("training checkpoint restore differs")
    torch.cuda.synchronize()
    report = {
        **metadata,
        "status": "complete",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "serialization_restore_exact": True,
        "charged_tokens": charged_tokens,
        "elapsed_seconds": time.monotonic() - training_started,
        "total_elapsed_seconds": time.monotonic() - started,
        "rank_receipts": rank_receipts,
        "routing_receipt": model.receipt(),
        "trace": trace,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    if rank == 0:
        _atomic_json(args.output / "report.json", report)
    dist.barrier()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = run(parse_args())
    except (MixtralDistributedTrainingError, MixtralRevisionError) as error:
        raise SystemExit(str(error)) from error
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    if int(os.environ.get("RANK", "-1")) == 0:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                },
                sort_keys=True,
            )
        )
