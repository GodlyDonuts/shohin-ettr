#!/usr/bin/env python3
"""Train the fixed-draft Shohin revision residual on Nemotron Super."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from hf_nemotron_super_mechanics import (
    CUDA_VERSION,
    MAMBA_VERSION,
    MODELOPT_VERSION,
    OVERLAY_MANIFEST_SHA256,
    OVERLAY_RECEIPT_SHA256,
    TORCH_VERSION,
    install_triton_allocator_compatibility,
    verify_manifest,
)
from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages
from nemotron_super_post_mixer_revision import NemotronSuperRevisionModel
from q36_upward_moe_host import (
    MODEL_CONFIG_SHA256,
    MODEL_MANIFEST_SHA256,
    MODEL_REVISION,
    MODEL_SOURCE_REVISION_SHA256,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    sha256_file,
)

SCHEMA = "shohin-nemotron-super-fixed-draft-revision-training-v1"
CHECKPOINT_SCHEMA = "shohin-nemotron-super-revision-checkpoint-v1"
DATA_SCHEMA = "shohin-q36-mtr-revision-train-v1"
DATA_SHA256 = "802c85662570c5bcb72f3e4430dbd093e901081f114213831292750894c3feff"
DATA_PRESENTATIONS = 9_655
UPDATES = 256
GRADIENT_ACCUMULATION = 8
CONSUMED_PRESENTATIONS = UPDATES * GRADIENT_ACCUMULATION
MAX_SEQUENCE_LENGTH = 4_096
LEARNING_RATE = 2e-5
SEED = 2026080815


class NemotronSuperTrainingError(RuntimeError):
    """The upward-MoE revision training contract differed."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise NemotronSuperTrainingError("refusing existing training report")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_revision_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != DATA_SHA256:
        raise NemotronSuperTrainingError("revision corpus bytes differ")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != DATA_SCHEMA
            or not isinstance(identity, str)
            or len(identity) != 64
            or row.get("internal_draft_visible") is not True
            or row.get("external_candidate_text_visible") is not False
            or row.get("runtime_fields") != ["question"]
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(row.get("response"), str)
            or not row["response"].strip()
        ):
            raise NemotronSuperTrainingError("revision corpus row differs")
        rows.append(row)
    if len(rows) != DATA_PRESENTATIONS:
        raise NemotronSuperTrainingError("revision presentation count differs")
    return rows


def consumed_identity_sha256(rows: list[dict[str, Any]]) -> str:
    if len(rows) != DATA_PRESENTATIONS:
        raise NemotronSuperTrainingError("revision consumption population differs")
    preimage = "".join(
        f"{index}:{rows[index]['identity_sha256']}\n"
        for index in range(CONSUMED_PRESENTATIONS)
    ).encode()
    return hashlib.sha256(preimage).hexdigest()


def tokenize_consumed_rows(
    tokenizer: Any, rows: list[dict[str, Any]]
) -> tuple[list[tuple[list[int], list[int]]], dict[str, Any]]:
    examples: list[tuple[list[int], list[int]]] = []
    maximum = 0
    prompt_tokens = response_tokens = 0
    token_digest = hashlib.sha256()
    for index, row in enumerate(rows[:CONSUMED_PRESENTATIONS]):
        rendered = render_reasoning_messages(
            tokenizer,
            [
                {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            enable_thinking=False,
        )
        prompt = [
            int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
        ]
        response = [
            int(value)
            for value in tokenizer.encode(row["response"], add_special_tokens=False)
        ]
        response.append(int(tokenizer.eos_token_id))
        total = len(prompt) + len(response)
        if not prompt or len(response) < 2 or total > MAX_SEQUENCE_LENGTH + 1:
            raise NemotronSuperTrainingError(
                f"Nemotron tokenization row {index} requires {total} tokens"
            )
        examples.append((prompt, response))
        maximum = max(maximum, total)
        prompt_tokens += len(prompt)
        response_tokens += len(response)
        token_digest.update(
            (json.dumps([prompt, response], separators=(",", ":")) + "\n").encode()
        )
    return examples, {
        "population_presentations": DATA_PRESENTATIONS,
        "consumed_presentations": len(examples),
        "consumed_identity_sha256": consumed_identity_sha256(rows),
        "consumed_token_sha256": token_digest.hexdigest(),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "maximum_observed_tokens": maximum,
        "maximum_sequence_length": MAX_SEQUENCE_LENGTH,
        "truncated_rows": 0,
    }


def _package_versions() -> dict[str, str | None]:
    import importlib.metadata

    return {
        "mamba-ssm": importlib.metadata.version("mamba-ssm"),
        "nvidia-modelopt": importlib.metadata.version("nvidia-modelopt"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    install_triton_allocator_compatibility()
    from modelopt.torch.opt.plugins.huggingface import enable_huggingface_checkpointing
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    if args.output.exists() or args.output.is_symlink():
        raise NemotronSuperTrainingError("training output already exists")
    if args.mechanics_report.is_symlink() or not args.mechanics_report.is_file():
        raise NemotronSuperTrainingError("mechanics report is absent")
    mechanics = json.loads(args.mechanics_report.read_text(encoding="utf-8"))
    if (
        mechanics.get("schema") != "shohin-nemotron-super-two-h100-mechanics-v1"
        or mechanics.get("status") != "pass"
        or mechanics.get("score_rows_read") != 0
        or mechanics.get("benchmark_rows_read") != 0
        or mechanics.get("model_revision") != MODEL_REVISION
        or mechanics.get("trainable_parameters") != TRAINABLE_PARAMETERS_PER_ROLE
        or mechanics.get("native_router_expert_trainables") != 0
        or mechanics.get("serialization_restore_exact") is not True
    ):
        raise NemotronSuperTrainingError("mechanics authorization differs")

    model_root = args.model_root.resolve(strict=True)
    overlay_root = args.overlay_root.resolve(strict=True)
    verify_manifest(model_root, args.model_manifest, MODEL_MANIFEST_SHA256)
    verify_manifest(overlay_root, args.overlay_manifest, OVERLAY_MANIFEST_SHA256)
    if (
        sha256_file(model_root / "config.json") != MODEL_CONFIG_SHA256
        or sha256_file(model_root / "SOURCE_REVISION") != MODEL_SOURCE_REVISION_SHA256
        or (model_root / "SOURCE_REVISION").read_text().strip() != MODEL_REVISION
        or sha256_file(overlay_root / "overlay_receipt.json") != OVERLAY_RECEIPT_SHA256
    ):
        raise NemotronSuperTrainingError("host or overlay identity differs")
    load_pinned_config(model_root / "config.json")
    expected_versions = {
        "mamba-ssm": MAMBA_VERSION,
        "nvidia-modelopt": MODELOPT_VERSION,
        "torch": TORCH_VERSION,
        "cuda": CUDA_VERSION,
    }
    if _package_versions() != expected_versions:
        raise NemotronSuperTrainingError("training package versions differ")
    if torch.cuda.device_count() != 2 or any(
        "H100" not in torch.cuda.get_device_name(index).upper() for index in range(2)
    ):
        raise NemotronSuperTrainingError("training requires exactly two H100s")

    rows = load_revision_rows(args.data)
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    examples, sequence_receipt = tokenize_consumed_rows(tokenizer, rows)

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    enable_huggingface_checkpointing()
    backbone = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=True,
        device_map="balanced",
        max_memory={0: "77GiB", 1: "77GiB", "cpu": "32GiB"},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if not isinstance(getattr(backbone, "hf_device_map", None), dict) or set(
        backbone.hf_device_map.values()
    ) - {0, 1}:
        raise NemotronSuperTrainingError("training device map differs")
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    model = NemotronSuperRevisionModel(backbone)
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
    torch.cuda.reset_peak_memory_stats()
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
            labels_for_loss = label_tensor.to(logits.device)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels_for_loss[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            scaled_loss = loss / GRADIENT_ACCUMULATION
        if not bool(torch.isfinite(loss)):
            raise NemotronSuperTrainingError("training loss is nonfinite")
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
        raise NemotronSuperTrainingError("training produced no parameter update")
    args.output.mkdir(parents=True)
    checkpoint = args.output / "checkpoint_0000256.pt"
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp.{os.getpid()}")
    metadata = {
        "schema": SCHEMA,
        "model_revision": MODEL_REVISION,
        "data_sha256": DATA_SHA256,
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
    }
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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "update", "trainable_state", "metadata"}
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("update") != UPDATES
        or payload.get("metadata") != metadata
        or _state_sha256(payload.get("trainable_state", {})) != final_state_sha256
    ):
        raise NemotronSuperTrainingError("training checkpoint restore differs")
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
        "peak_gpu_memory_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in range(2)
        },
        "routing_receipt": model.receipt(),
        "trace": trace,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
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
