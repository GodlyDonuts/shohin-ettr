#!/usr/bin/env python3
"""Run the score-free two-H100 Mixtral-8x22B Shohin mechanics gate."""

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

from mixtral_post_mlp_revision import MixtralRevisionError, MixtralRevisionModel
from q36_upward_moe_mixtral_host import (
    MODEL_REVISION,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    sha256_file,
    two_h100_device_map,
)

SCHEMA = "shohin-mixtral-8x22b-two-h100-mechanics-v1"
CHECKPOINT_SCHEMA = "shohin-mixtral-8x22b-mechanics-checkpoint-v1"
SEED = 2026081539
GPU_MEMORY = "77GiB"
EXPECTED_PACKAGES = {
    "bitsandbytes": "0.50.0",
    "torch": "2.6.0+cu124",
    "transformers": "5.15.0.dev0",
}


class MixtralMechanicsError(RuntimeError):
    """The score-free Mixtral mechanics contract failed."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise MixtralMechanicsError("refusing existing mechanics output")
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
        raise MixtralMechanicsError("manifest is absent or symbolic")
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
            raise MixtralMechanicsError("manifest row differs")
        seen.add(relative)
        rows.append((digest, relative))
    if not rows:
        raise MixtralMechanicsError("manifest is empty")
    return rows


def verify_model_manifest(
    root: Path, manifest: Path, expected_sha256: str
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest = manifest.resolve(strict=True)
    if root.is_symlink() or not root.is_dir() or not manifest.is_relative_to(root):
        raise MixtralMechanicsError("manifest root differs")
    if sha256_file(manifest) != expected_sha256:
        raise MixtralMechanicsError("manifest hash differs")
    rows = _manifest_rows(manifest)
    expected_members = {relative for _, relative in rows}
    covered_bytes = 0
    for expected, relative in rows:
        candidate = root / relative
        mode = candidate.lstat().st_mode if candidate.exists() else 0
        if (
            not stat.S_ISREG(mode)
            or candidate.is_symlink()
            or sha256_file(candidate) != expected
        ):
            raise MixtralMechanicsError("manifest member differs")
        covered_bytes += candidate.stat().st_size
    actual_members: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        mode = candidate.lstat().st_mode
        if stat.S_ISDIR(mode) and not candidate.is_symlink():
            continue
        if not stat.S_ISREG(mode) or candidate.is_symlink():
            raise MixtralMechanicsError("model tree member differs")
        if candidate != manifest:
            actual_members.add(relative)
    if actual_members != expected_members:
        raise MixtralMechanicsError("model tree membership differs")
    return {
        "manifest_sha256": expected_sha256,
        "manifest_entries": len(rows),
        "covered_bytes": covered_bytes,
        "exact_membership": True,
    }


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _router_receipt(model: MixtralRevisionModel) -> str:
    digest = hashlib.sha256()
    for index, block in enumerate(model.blocks):
        value = block.base.gate.weight.detach().cpu().contiguous()
        digest.update(str(index).encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _restore_trainables(
    model: MixtralRevisionModel, state: dict[str, torch.Tensor]
) -> None:
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(state):
        raise MixtralMechanicsError("serialized trainable names differ")
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise MixtralMechanicsError("serialized trainable geometry differs")
            parameter.copy_(value.to(parameter.device))


def _gradient_receipt(model: MixtralRevisionModel) -> dict[str, Any]:
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
        raise MixtralMechanicsError("Shohin gradient receipt differs")
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    return {
        "parameters": len(rows),
        "nonzero_gradients": sum(float(row["norm"] or 0.0) > 0.0 for row in rows),
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _package_versions() -> dict[str, str]:
    return {
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    model_root = args.model_root.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise MixtralMechanicsError("mechanics output already exists")
    model_receipt = verify_model_manifest(
        model_root, args.model_manifest, args.expected_model_manifest_sha256
    )
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
        raise MixtralMechanicsError("mechanics package versions differ")
    if torch.cuda.device_count() != 2:
        raise MixtralMechanicsError("exactly two H100 devices are required")
    devices = [torch.cuda.get_device_properties(index) for index in range(2)]
    if any("H100" not in value.name.upper() for value in devices):
        raise MixtralMechanicsError("allocated device is not H100")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        device_map=two_h100_device_map(),
        max_memory={0: GPU_MEMORY, 1: GPU_MEMORY, "cpu": "64GiB"},
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
        raise MixtralMechanicsError("model device map differs")
    model = MixtralRevisionModel(backbone)
    if model.trainable_parameter_count() != TRAINABLE_PARAMETERS_PER_ROLE:
        raise MixtralMechanicsError("trainable surface differs")
    if any(
        parameter.requires_grad
        for block in model.blocks
        for parameter in block.base.parameters()
    ):
        raise MixtralMechanicsError("native router or expert is trainable")

    initial_state = model.trainable_state()
    initial_sha256 = _state_sha256(initial_state)
    router_before = _router_receipt(model)
    token_ids = tokenizer.encode(
        "Shohin Mixtral mechanics only.", add_special_tokens=False
    )
    if not token_ids or len(token_ids) > 16:
        raise MixtralMechanicsError("synthetic mechanics tokenization differs")
    input_device = backbone.model.embed_tokens.weight.device
    input_ids = torch.tensor([token_ids], device=input_device, dtype=torch.long)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        foreach=False,
    )
    for index in range(2):
        torch.cuda.reset_peak_memory_stats(index)
    output_payload = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
    )
    logits = output_payload.logits
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise MixtralMechanicsError("full-model forward geometry differs")
    loss = logits[:, -1, :128].float().square().mean()
    if not bool(torch.isfinite(loss)):
        raise MixtralMechanicsError("mechanics loss is nonfinite")
    loss.backward()
    gradients = _gradient_receipt(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if _router_receipt(model) != router_before:
        raise MixtralMechanicsError("mechanics changed a native router")
    updated_state = model.trainable_state()
    updated_sha256 = _state_sha256(updated_state)
    if updated_sha256 == initial_sha256:
        raise MixtralMechanicsError("Shohin update is an exact no-op")

    checkpoint = output.with_suffix(".checkpoint.pt")
    if checkpoint.exists() or checkpoint.is_symlink():
        raise MixtralMechanicsError("refusing existing mechanics checkpoint")
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
        raise MixtralMechanicsError("mechanics checkpoint differs")
    _restore_trainables(model, payload["trainable_state"])
    if model.trainable_state_sha256() != updated_sha256:
        raise MixtralMechanicsError("mechanics restore differs")

    report = {
        "schema": SCHEMA,
        "status": "pass",
        "seed": SEED,
        "score_rows_read": 0,
        "benchmark_rows_read": 0,
        "model_revision": MODEL_REVISION,
        "model_receipt": model_receipt,
        "package_versions": versions,
        "devices": [
            {"index": index, "name": value.name, "total_memory": value.total_memory}
            for index, value in enumerate(devices)
        ],
        "device_map_sha256": hashlib.sha256(
            json.dumps(device_map, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "quantization": "nf4",
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
        "peak_gpu_memory_bytes": {
            str(index): torch.cuda.max_memory_allocated(index) for index in range(2)
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    if any(
        not math.isfinite(float(value))
        for value in report["peak_gpu_memory_bytes"].values()
    ):
        raise MixtralMechanicsError("GPU memory receipt differs")
    _atomic_json(output, report)
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
    print(json.dumps(result, sort_keys=True), flush=True)
