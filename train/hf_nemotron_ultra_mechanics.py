#!/usr/bin/env python3
"""Run the score-free eight-H100 Nemotron Ultra load/attach/restore gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any

import torch

from nemotron_ultra_post_mixer_revision import (
    NemotronUltraRevisionError,
    NemotronUltraRevisionModel,
)
from q36_upward_moe_ultra_host import (
    MINIMUM_H100S,
    MODEL_REVISION,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    sha256_file,
)

SCHEMA = "shohin-nemotron-ultra-eight-h100-mechanics-v1"
CHECKPOINT_SCHEMA = "shohin-nemotron-ultra-mechanics-checkpoint-v1"
SEED = 2026081522
GPU_MEMORY = "77GiB"
OVERLAY_MANIFEST_SHA256 = (
    "e52a0095628e0c4eb58f69c4ebb6f4d1c7a929792eb9a55046bda670013a6ea2"
)
MAMBA_VERSION = "2.3.2.post1"
MODELOPT_VERSION = "0.43.0"
CAUSAL_CONV_VERSION = "1.6.2.post1"
TORCH_VERSION = "2.6.0+cu124"
CUDA_VERSION = "12.4"


class NemotronUltraMechanicsError(RuntimeError):
    """The score-free 550B-A55B mechanics contract failed."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise NemotronUltraMechanicsError("refusing existing mechanics output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _manifest_rows(path: Path) -> list[tuple[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise NemotronUltraMechanicsError("manifest is absent or symbolic")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        candidate = Path(relative)
        if (
            len(digest) != 64
            or separator != "  "
            or any(value not in "0123456789abcdef" for value in digest)
            or not relative
            or relative in seen
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != relative
        ):
            raise NemotronUltraMechanicsError("manifest row differs")
        seen.add(relative)
        rows.append((digest, relative))
    if not rows:
        raise NemotronUltraMechanicsError("manifest is empty")
    return rows


def verify_manifest(
    root: Path,
    manifest: Path,
    expected_sha256: str,
    *,
    exact_membership: bool,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest = manifest.resolve(strict=True)
    if root.is_symlink() or not root.is_dir() or not manifest.is_relative_to(root):
        raise NemotronUltraMechanicsError("manifest root differs")
    if sha256_file(manifest) != expected_sha256:
        raise NemotronUltraMechanicsError("manifest hash differs")
    rows = _manifest_rows(manifest)
    expected_members = {relative for _, relative in rows}
    total = 0
    for expected, relative in rows:
        candidate = root / relative
        mode = candidate.lstat().st_mode if candidate.exists() else 0
        if (
            not stat.S_ISREG(mode)
            or candidate.is_symlink()
            or sha256_file(candidate) != expected
        ):
            raise NemotronUltraMechanicsError("manifest member differs")
        total += candidate.stat().st_size
    if exact_membership:
        actual_members: set[str] = set()
        for candidate in root.rglob("*"):
            relative = candidate.relative_to(root).as_posix()
            mode = candidate.lstat().st_mode
            if stat.S_ISDIR(mode) and not candidate.is_symlink():
                continue
            if not stat.S_ISREG(mode) or candidate.is_symlink():
                raise NemotronUltraMechanicsError("model tree member differs")
            if candidate != manifest:
                actual_members.add(relative)
        if actual_members != expected_members:
            raise NemotronUltraMechanicsError("model tree membership differs")
    return {
        "manifest_sha256": expected_sha256,
        "manifest_entries": len(rows),
        "covered_bytes": total,
        "exact_membership": exact_membership,
    }


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _restore_trainables(
    model: NemotronUltraRevisionModel, state: dict[str, torch.Tensor]
) -> None:
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(state):
        raise NemotronUltraMechanicsError("serialized trainable names differ")
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise NemotronUltraMechanicsError(
                    "serialized trainable geometry differs"
                )
            parameter.copy_(value.to(parameter.device))


def _gradient_receipt(model: NemotronUltraRevisionModel) -> dict[str, Any]:
    rows = []
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
        raise NemotronUltraMechanicsError("Shohin gradient receipt differs")
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    return {
        "parameters": len(rows),
        "nonzero_gradients": sum(float(row["norm"] or 0.0) > 0.0 for row in rows),
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _package_versions() -> dict[str, str | None]:
    return {
        "mamba-ssm": importlib.metadata.version("mamba-ssm"),
        "nvidia-modelopt": importlib.metadata.version("nvidia-modelopt"),
        "causal-conv1d": importlib.metadata.version("causal-conv1d"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    model_root = args.model_root.resolve(strict=True)
    overlay_root = args.overlay_root.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise NemotronUltraMechanicsError("mechanics output already exists")
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
    load_pinned_config(model_root / "config.json")
    revision_receipt = model_root / "SOURCE_REVISION"
    if (
        revision_receipt.is_symlink()
        or not revision_receipt.is_file()
        or revision_receipt.read_text(encoding="utf-8").strip() != MODEL_REVISION
    ):
        raise NemotronUltraMechanicsError("model revision receipt differs")

    expected_versions = {
        "mamba-ssm": MAMBA_VERSION,
        "nvidia-modelopt": MODELOPT_VERSION,
        "causal-conv1d": CAUSAL_CONV_VERSION,
        "torch": TORCH_VERSION,
        "cuda": CUDA_VERSION,
    }
    package_versions = _package_versions()
    if package_versions != expected_versions:
        raise NemotronUltraMechanicsError("mechanics package versions differ")

    import mamba_ssm
    import modelopt
    from modelopt.torch.opt.plugins.huggingface import (
        enable_huggingface_checkpointing,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    module_origins = {
        "mamba_ssm": str(Path(mamba_ssm.__file__).resolve()),
        "modelopt": str(Path(modelopt.__file__).resolve()),
    }
    if any(
        not Path(origin).is_relative_to(overlay_root)
        for origin in module_origins.values()
    ):
        raise NemotronUltraMechanicsError("mechanics module origin differs")
    if torch.cuda.device_count() != MINIMUM_H100S:
        raise NemotronUltraMechanicsError("exactly eight H100 devices are required")
    devices = [
        torch.cuda.get_device_properties(index) for index in range(MINIMUM_H100S)
    ]
    if any("H100" not in value.name.upper() for value in devices):
        raise NemotronUltraMechanicsError("allocated device is not H100")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    enable_huggingface_checkpointing()
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        device_map="balanced",
        max_memory={index: GPU_MEMORY for index in range(MINIMUM_H100S)},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    device_map = getattr(backbone, "hf_device_map", None)
    if not isinstance(device_map, dict) or set(device_map.values()) != set(
        range(MINIMUM_H100S)
    ):
        raise NemotronUltraMechanicsError("model device map differs")
    model = NemotronUltraRevisionModel(backbone)
    if model.trainable_parameter_count() != TRAINABLE_PARAMETERS_PER_ROLE:
        raise NemotronUltraMechanicsError("trainable surface differs")
    initial_state = model.trainable_state()
    initial_sha256 = _state_sha256(initial_state)

    token_ids = tokenizer.encode(
        "Shohin Ultra mechanics only.", add_special_tokens=False
    )
    if not token_ids or len(token_ids) > 16:
        raise NemotronUltraMechanicsError("synthetic mechanics tokenization differs")
    input_device = backbone.model.embeddings.weight.device
    input_ids = torch.tensor([token_ids], device=input_device, dtype=torch.long)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    output_payload = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
    )
    logits = output_payload.logits
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise NemotronUltraMechanicsError("full-model forward geometry differs")
    loss = logits[:, -1, :128].float().square().mean()
    if not bool(torch.isfinite(loss)):
        raise NemotronUltraMechanicsError("mechanics loss is nonfinite")
    loss.backward()
    gradients = _gradient_receipt(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    updated_state = model.trainable_state()
    updated_sha256 = _state_sha256(updated_state)
    if updated_sha256 == initial_sha256:
        raise NemotronUltraMechanicsError("Shohin update is an exact no-op")

    checkpoint = output.with_suffix(".checkpoint.pt")
    if checkpoint.exists() or checkpoint.is_symlink():
        raise NemotronUltraMechanicsError("refusing existing mechanics checkpoint")
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
    with torch.no_grad():
        next(
            parameter for parameter in model.parameters() if parameter.requires_grad
        ).zero_()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("trainable_state_sha256") != updated_sha256
        or _state_sha256(payload.get("trainable_state", {})) != updated_sha256
    ):
        raise NemotronUltraMechanicsError("mechanics checkpoint differs")
    _restore_trainables(model, payload["trainable_state"])
    if model.trainable_state_sha256() != updated_sha256:
        raise NemotronUltraMechanicsError("mechanics restore differs")

    report = {
        "schema": SCHEMA,
        "status": "pass",
        "seed": SEED,
        "score_rows_read": 0,
        "benchmark_rows_read": 0,
        "model_revision": MODEL_REVISION,
        "model_receipt": model_receipt,
        "overlay_receipt": overlay_receipt,
        "package_versions": package_versions,
        "module_origins": module_origins,
        "devices": [
            {"index": index, "name": value.name, "total_memory": value.total_memory}
            for index, value in enumerate(devices)
        ],
        "device_map_sha256": hashlib.sha256(
            json.dumps(device_map, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "trainable_parameters": model.trainable_parameter_count(),
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "initial_trainable_state_sha256": initial_sha256,
        "updated_trainable_state_sha256": updated_sha256,
        "gradient_receipt": gradients,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "serialization_restore_exact": True,
        "native_router_expert_trainables": 0,
        "routing_receipt": model.receipt(),
        "peak_gpu_memory_bytes": {
            str(index): torch.cuda.max_memory_allocated(index)
            for index in range(MINIMUM_H100S)
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    if any(
        not math.isfinite(float(value))
        for value in report["peak_gpu_memory_bytes"].values()
    ):
        raise NemotronUltraMechanicsError("GPU memory receipt differs")
    _atomic_json(output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--overlay-root", required=True, type=Path)
    parser.add_argument("--overlay-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = run(parse_args())
    except (NemotronUltraMechanicsError, NemotronUltraRevisionError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True), flush=True)
