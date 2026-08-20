#!/usr/bin/env python3
"""Train the fixed-draft Shohin residual on GPT-OSS-120B MXFP4."""

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

from gpt_oss_harmony import tokenize_training_example
from gpt_oss_post_mlp_revision import GptOssRevisionModel
from hf_gpt_oss_120b_mechanics import (
    EXPECTED_PACKAGES,
    GptOssMechanicsError,
    SCHEMA as MECHANICS_SCHEMA,
    _native_mxfp4_load_receipt,
    _package_receipt,
    _restore_trainables,
    _state_sha256,
    verify_manifest,
)
from q36_upward_moe_gpt_oss_host import (
    MODEL_CONFIG_SHA256,
    MODEL_MANIFEST_SHA256,
    MODEL_REVISION,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    sha256_file,
)

SCHEMA = "shohin-gpt-oss-120b-fixed-draft-revision-training-v1"
CHECKPOINT_SCHEMA = "shohin-gpt-oss-120b-revision-checkpoint-v1"
DATA_SCHEMA = "shohin-q36-mtr-revision-train-v1"
DATA_SHA256 = "802c85662570c5bcb72f3e4430dbd093e901081f114213831292750894c3feff"
DATA_PRESENTATIONS = 9_655
UPDATES = 256
GRADIENT_ACCUMULATION = 8
CONSUMED_PRESENTATIONS = UPDATES * GRADIENT_ACCUMULATION
MAX_SEQUENCE_LENGTH = 4_096
LEARNING_RATE = 2e-5
SEED = 2026080815
DRAFT_ORIGIN_MODEL = "Qwen3.6-35B-A3B"
DRAFT_ORIGIN_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
TRANSFER_SCOPE = "fixed_qwen35_model_owned_draft_cross_family_screen"
GPU_MEMORY = "77GiB"


class GptOssTrainingError(RuntimeError):
    """The GPT-OSS upward-MoE revision training contract differed."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise GptOssTrainingError("refusing existing training report")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_revision_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != DATA_SHA256:
        raise GptOssTrainingError("revision corpus bytes differ")
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GptOssTrainingError("revision corpus is unreadable") from error
    for row in rows:
        identity = row.get("identity_sha256") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("schema") != DATA_SCHEMA
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
            raise GptOssTrainingError("revision corpus row differs")
    if len(rows) != DATA_PRESENTATIONS:
        raise GptOssTrainingError("revision presentation count differs")
    return rows


def consumed_identity_sha256(rows: list[dict[str, Any]]) -> str:
    if len(rows) != DATA_PRESENTATIONS:
        raise GptOssTrainingError("revision consumption population differs")
    preimage = "".join(
        f"{index}:{rows[index]['identity_sha256']}\n"
        for index in range(CONSUMED_PRESENTATIONS)
    ).encode()
    return hashlib.sha256(preimage).hexdigest()


def tokenize_consumed_rows(
    tokenizer: Any, rows: list[dict[str, Any]]
) -> tuple[list[tuple[list[int], list[int]]], dict[str, Any]]:
    examples: list[tuple[list[int], list[int]]] = []
    maximum = prompt_tokens = response_tokens = 0
    digest = hashlib.sha256()
    for index, row in enumerate(rows[:CONSUMED_PRESENTATIONS]):
        try:
            prompt, response = tokenize_training_example(
                tokenizer,
                row["question"],
                row["response"],
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )
        except Exception as error:
            raise GptOssTrainingError(
                f"GPT-OSS tokenization row {index} differs"
            ) from error
        examples.append((prompt, response))
        total = len(prompt) + len(response)
        maximum = max(maximum, total)
        prompt_tokens += len(prompt)
        response_tokens += len(response)
        digest.update(
            (json.dumps([prompt, response], separators=(",", ":")) + "\n").encode()
        )
    if len(examples) != CONSUMED_PRESENTATIONS:
        raise GptOssTrainingError("GPT-OSS consumed presentation count differs")
    return examples, {
        "population_presentations": DATA_PRESENTATIONS,
        "consumed_presentations": len(examples),
        "consumed_identity_sha256": consumed_identity_sha256(rows),
        "consumed_token_sha256": digest.hexdigest(),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "maximum_observed_tokens": maximum,
        "maximum_sequence_length": MAX_SEQUENCE_LENGTH,
        "truncated_rows": 0,
        "harmony_reasoning_effort": "low",
        "supervised_channel": "final",
    }


def validate_mechanics(
    path: Path,
    *,
    model_manifest_sha256: str,
    overlay_manifest_sha256: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GptOssTrainingError("mechanics report is absent")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GptOssTrainingError("mechanics report is unreadable") from error
    if (
        payload.get("schema") != MECHANICS_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("gpu_count") != 1
        or payload.get("model_receipt", {}).get("manifest_sha256")
        != model_manifest_sha256
        or payload.get("overlay_receipt", {}).get("manifest_sha256")
        != overlay_manifest_sha256
        or payload.get("trainable_parameters") != TRAINABLE_PARAMETERS_PER_ROLE
        or payload.get("native_router_expert_trainables") != 0
        or payload.get("checkpoint_restore_exact") is not True
        or payload.get("score_or_assessor_data_accessed") is not False
        or payload.get("scientific_result") is not False
    ):
        raise GptOssTrainingError("mechanics authorization differs")
    return payload


def _load_backbone(model_root: Path) -> tuple[Any, dict[str, Any]]:
    from kernels import get_loaded_kernels
    from transformers import AutoModelForCausalLM

    backbone = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        device_map={"": 0},
        max_memory={0: GPU_MEMORY, "cpu": "8GiB"},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    try:
        native_load_receipt = _native_mxfp4_load_receipt(backbone)
    except GptOssMechanicsError as error:
        raise GptOssTrainingError("native MXFP4 training load differs") from error
    if len(get_loaded_kernels()) != 1:
        raise GptOssTrainingError("native MXFP4 training load differs")
    return backbone, native_load_receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    started = time.monotonic()
    if args.output.exists() or args.output.is_symlink():
        raise GptOssTrainingError("training output exists")
    if args.expected_model_manifest_sha256 != MODEL_MANIFEST_SHA256:
        raise GptOssTrainingError("model manifest authorization differs")
    mechanics = validate_mechanics(
        args.mechanics_report,
        model_manifest_sha256=args.expected_model_manifest_sha256,
        overlay_manifest_sha256=args.expected_overlay_manifest_sha256,
    )
    model_root = args.model_root.resolve(strict=True)
    overlay_root = args.overlay_root.resolve(strict=True)
    model_receipt = verify_manifest(
        model_root, args.model_manifest, args.expected_model_manifest_sha256
    )
    overlay_receipt = verify_manifest(
        overlay_root, args.overlay_manifest, args.expected_overlay_manifest_sha256
    )
    if (
        sha256_file(model_root / "config.json") != MODEL_CONFIG_SHA256
        or (model_root / "SOURCE_REVISION").read_text(encoding="utf-8")
        != f"{MODEL_REVISION}\n"
    ):
        raise GptOssTrainingError("host identity differs")
    load_pinned_config(model_root / "config.json")
    packages = _package_receipt(overlay_root)
    if packages["versions"] != EXPECTED_PACKAGES:
        raise GptOssTrainingError("training package versions differ")
    if (
        torch.cuda.device_count() != 1
        or "H100" not in torch.cuda.get_device_name(0).upper()
    ):
        raise GptOssTrainingError("training requires exactly one H100")

    rows = load_revision_rows(args.data)
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    examples, sequence_receipt = tokenize_consumed_rows(tokenizer, rows)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    backbone, native_load_receipt = _load_backbone(model_root)
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    model = GptOssRevisionModel(backbone)
    trainables = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if (
        len(trainables) != 32
        or model.trainable_parameter_count() != TRAINABLE_PARAMETERS_PER_ROLE
    ):
        raise GptOssTrainingError("training parameter surface differs")
    initial_state_sha256 = model.trainable_state_sha256()
    optimizer = torch.optim.AdamW(
        trainables,
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    model.reset_receipt()
    torch.cuda.reset_peak_memory_stats(0)
    trace: list[dict[str, Any]] = []
    charged_tokens = 0
    training_started = time.monotonic()
    for microstep, (prompt, response) in enumerate(examples, start=1):
        tokens = prompt + response
        input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda:0")
        attention_mask = torch.ones_like(input_ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                logits_to_keep=len(response) + 1,
            )
            logits = outputs.logits
            if logits.shape[1] != len(response) + 1:
                raise GptOssTrainingError("training logits geometry differs")
            targets = torch.tensor(response, dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            scaled_loss = loss / GRADIENT_ACCUMULATION
        if not bool(torch.isfinite(loss)):
            raise GptOssTrainingError("training loss is nonfinite")
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
    if microstep != CONSUMED_PRESENTATIONS or update != UPDATES:
        raise GptOssTrainingError("training update count differs")
    final_state = model.trainable_state()
    final_state_sha256 = _state_sha256(final_state)
    if final_state_sha256 == initial_state_sha256:
        raise GptOssTrainingError("training is an exact no-op")

    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / f"checkpoint_{UPDATES:07d}.pt"
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp.{os.getpid()}")
    metadata = {
        "schema": SCHEMA,
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": args.expected_model_manifest_sha256,
        "overlay_manifest_sha256": args.expected_overlay_manifest_sha256,
        "draft_origin_model": DRAFT_ORIGIN_MODEL,
        "draft_origin_revision": DRAFT_ORIGIN_REVISION,
        "transfer_scope": TRANSFER_SCOPE,
        "standalone_gpt_oss_owned_draft": False,
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
        "optimizer_state_serialized": False,
        "checkpoint_trainable_only": True,
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
    with torch.no_grad():
        trainables[0].zero_()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _restore_trainables(model, payload["trainable_state"])
    if model.trainable_state_sha256() != final_state_sha256:
        raise GptOssTrainingError("checkpoint restore differs")
    torch.cuda.synchronize()
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "model_receipt": model_receipt,
        "overlay_receipt": overlay_receipt,
        "packages": packages,
        "native_mxfp4_load_receipt": native_load_receipt,
        "mechanics_report_sha256": sha256_file(args.mechanics_report),
        "mechanics_checkpoint_sha256": mechanics["checkpoint_sha256"],
        "data_sha256": DATA_SHA256,
        "sequence_receipt": sequence_receipt,
        "updates": UPDATES,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "charged_tokens": charged_tokens,
        "trainable_parameters": TRAINABLE_PARAMETERS_PER_ROLE,
        "native_router_expert_trainables": 0,
        "initial_trainable_state_sha256": initial_state_sha256,
        "final_trainable_state_sha256": final_state_sha256,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_restore_exact": True,
        "routing_receipt": model.receipt(),
        "trace": trace,
        "training_elapsed_seconds": time.monotonic() - training_started,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(0)),
        "assessor_access_count": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-model-manifest-sha256", default=MODEL_MANIFEST_SHA256
    )
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--expected-overlay-manifest-sha256", required=True)
    parser.add_argument("--mechanics-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"updates": result["updates"]}, sort_keys=True))
