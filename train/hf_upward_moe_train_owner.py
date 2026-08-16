#!/usr/bin/env python3
"""Train the source-only owner trajectory for an upward MoE host."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from hf_product_reasoning_train import (
    PRODUCT_SYSTEM_PROMPT,
    _batches,
    render_reasoning_messages,
    reservoir_rows_with_sha256,
)
from hf_q36_mtr_train_role import training_consumption_receipt
from mixtral_post_mlp_revision import MixtralRevisionModel
from nemotron_super_post_mixer_revision import NemotronSuperRevisionModel
from nemotron_ultra_post_mixer_revision import NemotronUltraRevisionModel
from upward_moe_role_lineage import (
    save_role_checkpoint,
    sha256_file,
)
from upward_moe_temporal_gate import MIXTRAL_SPEC, NEMOTRON_SPEC, ULTRA_SPEC

SCHEMA = "shohin-upward-moe-owner-training-v1"
OWNER_DATA_SHA256 = "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549"
OWNER_MAX_ROWS = 100_000
OWNER_SELECTED_ROWS = 26_387
OWNER_UPDATES = 256
OWNER_GRADIENT_ACCUMULATION = 16
OWNER_CONSUMED_PRESENTATIONS = OWNER_UPDATES * OWNER_GRADIENT_ACCUMULATION
OWNER_MAX_SEQUENCE_LENGTH = 1_024
OWNER_LEARNING_RATE = 2e-4
OWNER_SEED = 2026080711
OWNER_DATA_SEED = 20260802


class UpwardMoEOwnerTrainingError(RuntimeError):
    """The upward MoE source-only owner training contract differed."""


@dataclass(frozen=True)
class LoadedHost:
    model: Any
    tokenizer: Any
    input_device: torch.device
    spec: Any
    model_receipt: dict[str, Any]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise UpwardMoEOwnerTrainingError("owner report already exists")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def static_owner_contract() -> dict[str, Any]:
    return {
        "role": "owner",
        "data_kind": "source_only",
        "data_sha256": OWNER_DATA_SHA256,
        "selected_rows": OWNER_SELECTED_ROWS,
        "updates": OWNER_UPDATES,
        "gradient_accumulation": OWNER_GRADIENT_ACCUMULATION,
        "consumed_presentations": OWNER_CONSUMED_PRESENTATIONS,
        "max_sequence_length": OWNER_MAX_SEQUENCE_LENGTH,
        "learning_rate": OWNER_LEARNING_RATE,
        "seed": OWNER_SEED,
        "data_seed": OWNER_DATA_SEED,
        "external_proposer": False,
        "task_router": False,
        "native_router_expert_trainables": 0,
        "hosts": [
            NEMOTRON_SPEC.receipt(),
            MIXTRAL_SPEC.receipt(),
            ULTRA_SPEC.receipt(),
        ],
    }


def load_owner_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != OWNER_DATA_SHA256
    ):
        raise UpwardMoEOwnerTrainingError("owner source bytes differ")
    rows, digest = reservoir_rows_with_sha256(path, OWNER_MAX_ROWS, OWNER_DATA_SEED)
    if digest != OWNER_DATA_SHA256 or len(rows) != OWNER_SELECTED_ROWS:
        raise UpwardMoEOwnerTrainingError("owner source population differs")
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(row.get("response"), str)
            or not row["response"].strip()
        ):
            raise UpwardMoEOwnerTrainingError("owner source row differs")
    return rows, digest


def tokenize_owner_rows(
    tokenizer: Any, rows: list[dict[str, Any]]
) -> tuple[list[tuple[list[int], list[int], list[int]]], dict[str, Any]]:
    examples = []
    maximum = prompt_tokens = response_tokens = 0
    for index, row in enumerate(rows):
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
        if not prompt or len(response) < 2 or total > OWNER_MAX_SEQUENCE_LENGTH + 1:
            raise UpwardMoEOwnerTrainingError(
                f"owner row {index} requires {total} tokens"
            )
        examples.append((prompt, response, [1] * len(prompt)))
        maximum = max(maximum, total)
        prompt_tokens += len(prompt)
        response_tokens += len(response)
    return examples, {
        "selected_rows": len(examples),
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "maximum_observed_tokens": maximum,
        "maximum_sequence_length": OWNER_MAX_SEQUENCE_LENGTH,
        "eos_token_allowance": 1,
        "truncated_rows": 0,
        "source_only": True,
        "draft_masked_tokens": 0,
    }


def _validate_mechanics(
    path: Path, *, schema: str, spec: Any, expected_manifest_sha256: str | None
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise UpwardMoEOwnerTrainingError("owner mechanics report is absent")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != schema
        or report.get("status") != "pass"
        or report.get("score_rows_read") != 0
        or report.get("benchmark_rows_read") != 0
        or report.get("model_revision") != spec.model_revision
        or report.get("trainable_parameters")
        != 2 * len(spec.controlled_layer_indices) * spec.hidden_size * spec.rank
        or report.get("native_router_expert_trainables") != 0
        or report.get("serialization_restore_exact") is not True
        or (
            expected_manifest_sha256 is not None
            and report.get("model_receipt", {}).get("manifest_sha256")
            != expected_manifest_sha256
        )
        or (spec == ULTRA_SPEC and len(report.get("devices", [])) != 8)
    ):
        raise UpwardMoEOwnerTrainingError("owner mechanics authorization differs")
    return report


def _load_nemotron(
    args: argparse.Namespace, *, attach_revision: bool = True
) -> LoadedHost:
    from modelopt.torch.opt.plugins.huggingface import enable_huggingface_checkpointing
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hf_nemotron_super_mechanics import (
        OVERLAY_MANIFEST_SHA256,
        OVERLAY_RECEIPT_SHA256,
        verify_manifest,
    )
    from hf_nemotron_super_train_revision import _package_versions
    from q36_upward_moe_host import (
        MODEL_CONFIG_SHA256,
        MODEL_MANIFEST_SHA256,
        MODEL_SOURCE_REVISION_SHA256,
        load_pinned_config,
    )

    if args.overlay_root is None or args.overlay_manifest is None:
        raise UpwardMoEOwnerTrainingError("Nemotron owner overlay is absent")
    _validate_mechanics(
        args.mechanics_report,
        schema="shohin-nemotron-super-two-h100-mechanics-v1",
        spec=NEMOTRON_SPEC,
        expected_manifest_sha256=None,
    )
    verify_manifest(args.model_root, args.model_manifest, MODEL_MANIFEST_SHA256)
    verify_manifest(args.overlay_root, args.overlay_manifest, OVERLAY_MANIFEST_SHA256)
    if (
        sha256_file(args.model_root / "config.json") != MODEL_CONFIG_SHA256
        or sha256_file(args.model_root / "SOURCE_REVISION")
        != MODEL_SOURCE_REVISION_SHA256
        or (args.model_root / "SOURCE_REVISION").read_text().strip()
        != NEMOTRON_SPEC.model_revision
        or sha256_file(args.overlay_root / "overlay_receipt.json")
        != OVERLAY_RECEIPT_SHA256
    ):
        raise UpwardMoEOwnerTrainingError("Nemotron owner host identity differs")
    load_pinned_config(args.model_root / "config.json")
    if _package_versions() != {
        "mamba-ssm": "2.3.0",
        "nvidia-modelopt": "0.35.0",
        "torch": "2.8.0+cu124",
        "cuda": "12.4",
    }:
        raise UpwardMoEOwnerTrainingError("Nemotron owner packages differ")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, local_files_only=True, trust_remote_code=True
    )
    enable_huggingface_checkpointing()
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_root,
        local_files_only=True,
        trust_remote_code=True,
        device_map="balanced",
        max_memory={0: "77GiB", 1: "77GiB", "cpu": "32GiB"},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    device_map = getattr(backbone, "hf_device_map", None)
    if not isinstance(device_map, dict) or set(device_map.values()) - {0, 1}:
        raise UpwardMoEOwnerTrainingError("Nemotron owner device map differs")
    model = NemotronSuperRevisionModel(backbone) if attach_revision else backbone
    return LoadedHost(
        model=model,
        tokenizer=tokenizer,
        input_device=backbone.model.embeddings.weight.device,
        spec=NEMOTRON_SPEC,
        model_receipt={
            "model_manifest_sha256": MODEL_MANIFEST_SHA256,
            "overlay_manifest_sha256": OVERLAY_MANIFEST_SHA256,
        },
    )


def _load_mixtral(
    args: argparse.Namespace, *, attach_revision: bool = True
) -> LoadedHost:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from hf_mixtral_8x22b_mechanics import (
        EXPECTED_PACKAGES,
        SCHEMA as MECHANICS_SCHEMA,
        verify_model_manifest,
    )
    from hf_mixtral_8x22b_train_revision import _package_versions
    from q36_upward_moe_mixtral_host import load_pinned_config

    expected_manifest = args.expected_model_manifest_sha256
    if not isinstance(expected_manifest, str) or len(expected_manifest) != 64:
        raise UpwardMoEOwnerTrainingError("Mixtral owner manifest binding differs")
    _validate_mechanics(
        args.mechanics_report,
        schema=MECHANICS_SCHEMA,
        spec=MIXTRAL_SPEC,
        expected_manifest_sha256=expected_manifest,
    )
    model_receipt = verify_model_manifest(
        args.model_root, args.model_manifest, expected_manifest
    )
    if (
        sha256_file(args.model_root / "config.json") != MIXTRAL_SPEC.model_config_sha256
        or (args.model_root / "SOURCE_REVISION").is_symlink()
        or not (args.model_root / "SOURCE_REVISION").is_file()
        or (args.model_root / "SOURCE_REVISION").read_text().strip()
        != MIXTRAL_SPEC.model_revision
    ):
        raise UpwardMoEOwnerTrainingError("Mixtral owner host identity differs")
    load_pinned_config(args.model_root / "config.json")
    if _package_versions() != EXPECTED_PACKAGES:
        raise UpwardMoEOwnerTrainingError("Mixtral owner packages differ")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, local_files_only=True, trust_remote_code=False
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_root,
        local_files_only=True,
        trust_remote_code=False,
        device_map="balanced",
        max_memory={0: "77GiB", 1: "77GiB", "cpu": "64GiB"},
        quantization_config=quantization,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    device_map = getattr(backbone, "hf_device_map", None)
    if (
        not isinstance(device_map, dict)
        or set(device_map.values()) != {0, 1}
        or any(value in {"cpu", "disk"} for value in device_map.values())
    ):
        raise UpwardMoEOwnerTrainingError("Mixtral owner device map differs")
    model = MixtralRevisionModel(backbone) if attach_revision else backbone
    return LoadedHost(
        model=model,
        tokenizer=tokenizer,
        input_device=backbone.model.embed_tokens.weight.device,
        spec=MIXTRAL_SPEC,
        model_receipt=model_receipt,
    )


def _load_ultra(
    args: argparse.Namespace, *, attach_revision: bool = True
) -> LoadedHost:
    from modelopt.torch.opt.plugins.huggingface import enable_huggingface_checkpointing
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    from q36_upward_moe_ultra_host import load_pinned_config

    expected_manifest = args.expected_model_manifest_sha256
    if not isinstance(expected_manifest, str) or len(expected_manifest) != 64:
        raise UpwardMoEOwnerTrainingError("Ultra owner manifest binding differs")
    if any(
        path is None
        for path in (args.overlay_root, args.overlay_manifest, args.causal_conv_root)
    ):
        raise UpwardMoEOwnerTrainingError("Ultra owner runtime overlay is absent")
    _validate_mechanics(
        args.mechanics_report,
        schema=MECHANICS_SCHEMA,
        spec=ULTRA_SPEC,
        expected_manifest_sha256=expected_manifest,
    )
    model_receipt = verify_manifest(
        args.model_root,
        args.model_manifest,
        expected_manifest,
        exact_membership=True,
    )
    overlay_receipt = verify_manifest(
        args.overlay_root,
        args.overlay_manifest,
        OVERLAY_MANIFEST_SHA256,
        exact_membership=False,
    )
    if (
        sha256_file(args.model_root / "config.json") != ULTRA_SPEC.model_config_sha256
        or (args.model_root / "SOURCE_REVISION").is_symlink()
        or not (args.model_root / "SOURCE_REVISION").is_file()
        or (args.model_root / "SOURCE_REVISION").read_text().strip()
        != ULTRA_SPEC.model_revision
    ):
        raise UpwardMoEOwnerTrainingError("Ultra owner host identity differs")
    load_pinned_config(args.model_root / "config.json")
    if _package_versions() != {
        "mamba-ssm": MAMBA_VERSION,
        "nvidia-modelopt": MODELOPT_VERSION,
        "causal-conv1d": CAUSAL_CONV_VERSION,
        "torch": TORCH_VERSION,
        "cuda": CUDA_VERSION,
    }:
        raise UpwardMoEOwnerTrainingError("Ultra owner packages differ")
    import causal_conv1d
    import mamba_ssm
    import modelopt

    origins = {
        "causal_conv1d": Path(causal_conv1d.__file__).resolve(),
        "mamba_ssm": Path(mamba_ssm.__file__).resolve(),
        "modelopt": Path(modelopt.__file__).resolve(),
    }
    if (
        not origins["causal_conv1d"].is_relative_to(args.causal_conv_root.resolve())
        or not origins["mamba_ssm"].is_relative_to(args.overlay_root.resolve())
        or not origins["modelopt"].is_relative_to(args.overlay_root.resolve())
    ):
        raise UpwardMoEOwnerTrainingError("Ultra owner module origin differs")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root, local_files_only=True, trust_remote_code=False
    )
    enable_huggingface_checkpointing()
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_root,
        local_files_only=True,
        trust_remote_code=False,
        device_map="balanced",
        max_memory={index: "77GiB" for index in range(8)},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    device_map = getattr(backbone, "hf_device_map", None)
    if not isinstance(device_map, dict) or set(device_map.values()) != set(range(8)):
        raise UpwardMoEOwnerTrainingError("Ultra owner device map differs")
    model = NemotronUltraRevisionModel(backbone) if attach_revision else backbone
    return LoadedHost(
        model=model,
        tokenizer=tokenizer,
        input_device=backbone.model.embeddings.weight.device,
        spec=ULTRA_SPEC,
        model_receipt={
            **model_receipt,
            "overlay_manifest_sha256": overlay_receipt["manifest_sha256"],
        },
    )


def _load_host(args: argparse.Namespace, *, attach_revision: bool = True) -> LoadedHost:
    required_gpus = 8 if args.host == "nemotron-ultra" else 2
    if torch.cuda.device_count() != required_gpus or any(
        "H100" not in torch.cuda.get_device_name(index).upper()
        for index in range(required_gpus)
    ):
        raise UpwardMoEOwnerTrainingError(
            f"owner training requires exactly {required_gpus} H100s"
        )
    if args.host == "nemotron-super":
        return _load_nemotron(args, attach_revision=attach_revision)
    if args.host == "mixtral-8x22b":
        return _load_mixtral(args, attach_revision=attach_revision)
    if args.host == "nemotron-ultra":
        return _load_ultra(args, attach_revision=attach_revision)
    raise UpwardMoEOwnerTrainingError("owner host differs")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.output.exists() or args.output.is_symlink():
        raise UpwardMoEOwnerTrainingError("owner output exists")
    rows, data_sha256 = load_owner_rows(args.data)
    random.seed(OWNER_SEED)
    torch.manual_seed(OWNER_SEED)
    torch.cuda.manual_seed_all(OWNER_SEED)
    loaded = _load_host(args)
    if loaded.tokenizer.pad_token_id is None:
        loaded.tokenizer.pad_token_id = loaded.tokenizer.eos_token_id
    examples, sequence_receipt = tokenize_owner_rows(loaded.tokenizer, rows)
    consumption = training_consumption_receipt(
        examples,
        updates=OWNER_UPDATES,
        gradient_accumulation=OWNER_GRADIENT_ACCUMULATION,
        batch_size=1,
    )
    if consumption["consumed_presentations"] != OWNER_CONSUMED_PRESENTATIONS:
        raise UpwardMoEOwnerTrainingError("owner consumption differs")
    model = loaded.model
    initial_state_sha256 = model.trainable_state_sha256()
    trainables = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if model.trainable_parameter_count() != (
        2
        * len(loaded.spec.controlled_layer_indices)
        * loaded.spec.hidden_size
        * loaded.spec.rank
    ):
        raise UpwardMoEOwnerTrainingError("owner trainable surface differs")
    backbone = model.backbone
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        trainables,
        lr=OWNER_LEARNING_RATE,
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
    for microstep in range(1, OWNER_CONSUMED_PRESENTATIONS + 1):
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
            scaled_loss = loss / OWNER_GRADIENT_ACCUMULATION
        if not bool(torch.isfinite(loss)):
            raise UpwardMoEOwnerTrainingError("owner loss is nonfinite")
        scaled_loss.backward()
        charged_tokens += len(tokens)
        if microstep % OWNER_GRADIENT_ACCUMULATION:
            continue
        update = microstep // OWNER_GRADIENT_ACCUMULATION
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainables, 1.0)
        progress = (update - 1) / max(OWNER_UPDATES - 1, 1)
        learning_rate = OWNER_LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))
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
        "role": "owner",
        "host": loaded.spec.host,
        "data_sha256": data_sha256,
        "sequence_receipt": sequence_receipt,
        "consumption_receipt": consumption,
        "model_receipt": loaded.model_receipt,
        "mechanics_report_sha256": sha256_file(args.mechanics_report),
        "trace": trace,
        "charged_tokens": charged_tokens,
        "elapsed_seconds": time.monotonic() - training_started,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    checkpoint = args.output / "checkpoint_0000256.pt"
    restored = save_role_checkpoint(
        checkpoint,
        role="owner",
        state=final_state,
        spec=loaded.spec,
        initial_state_sha256=initial_state_sha256,
        training_receipt=training_receipt,
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
