#!/usr/bin/env python3
"""Run the score-free two-H100 Nemotron Super load/attach/restore gate."""

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

from nemotron_super_post_mixer_revision import (
    NemotronSuperRevisionError,
    NemotronSuperRevisionModel,
)
from q36_upward_moe_host import (
    MODEL_CONFIG_SHA256,
    MODEL_MANIFEST_SHA256,
    MODEL_REVISION,
    MODEL_SOURCE_REVISION_SHA256,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    sha256_file,
)

SCHEMA = "shohin-nemotron-super-two-h100-mechanics-v1"
SEED = 2026081521
OVERLAY_MANIFEST_SHA256 = (
    "cde0fa5b91d50d1509872cbc577cf016d0a6c6697bfb066d607f420c1b568e84"
)
OVERLAY_RECEIPT_SHA256 = (
    "a917e093a2cdba7f5ce0cd2131a5d66fedd0c3fe086dc4ff2243dd7edb332a35"
)
MAMBA_VERSION = "2.3.2.post1"
MODELOPT_VERSION = "0.43.0"
CAUSAL_CONV_VERSION = "1.6.2.post1"
TORCH_VERSION = "2.6.0+cu124"
CUDA_VERSION = "12.4"


class NemotronSuperMechanicsError(RuntimeError):
    """The score-free upward-MoE mechanics contract failed."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise NemotronSuperMechanicsError("refusing existing mechanics output")
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
        raise NemotronSuperMechanicsError("manifest is absent or symbolic")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
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
            raise NemotronSuperMechanicsError("manifest row differs")
        seen.add(relative)
        rows.append((digest, relative))
    # Ordering is already bound by the caller-pinned SHA-256 of the manifest
    # bytes.  External overlays may use installation/traversal order rather
    # than lexical path order, so integrity requires safe unique rows and
    # exact member hashes, not a second, incompatible ordering convention.
    if not rows:
        raise NemotronSuperMechanicsError("manifest is empty")
    return rows


def verify_manifest(root: Path, manifest: Path, expected_sha256: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest = manifest.resolve(strict=True)
    if root.is_symlink() or not root.is_dir() or not manifest.is_relative_to(root):
        raise NemotronSuperMechanicsError("manifest root differs")
    if sha256_file(manifest) != expected_sha256:
        raise NemotronSuperMechanicsError("manifest hash differs")
    rows = _manifest_rows(manifest)
    total = 0
    for expected, relative in rows:
        candidate = root / relative
        mode = candidate.lstat().st_mode if candidate.exists() else 0
        if (
            not stat.S_ISREG(mode)
            or candidate.is_symlink()
            or sha256_file(candidate) != expected
        ):
            raise NemotronSuperMechanicsError("manifest member differs")
        total += candidate.stat().st_size
    return {
        "manifest_sha256": expected_sha256,
        "manifest_entries": len(rows),
        "covered_bytes": total,
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
    model: NemotronSuperRevisionModel, state: dict[str, torch.Tensor]
) -> None:
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(state):
        raise NemotronSuperMechanicsError("serialized trainable names differ")
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise NemotronSuperMechanicsError(
                    "serialized trainable geometry differs"
                )
            parameter.copy_(value.to(parameter.device))


def _gradient_receipt(model: NemotronSuperRevisionModel) -> dict[str, Any]:
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
        raise NemotronSuperMechanicsError("Shohin gradient receipt differs")
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    return {
        "parameters": len(rows),
        "nonzero_gradients": sum(float(row["norm"] or 0.0) > 0.0 for row in rows),
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    model_root = args.model_root.resolve(strict=True)
    overlay_root = args.overlay_root.resolve(strict=True)
    model_manifest = args.model_manifest.resolve(strict=True)
    overlay_manifest = args.overlay_manifest.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise NemotronSuperMechanicsError("mechanics output already exists")

    model_receipt = verify_manifest(model_root, model_manifest, MODEL_MANIFEST_SHA256)
    overlay_receipt = verify_manifest(
        overlay_root, overlay_manifest, OVERLAY_MANIFEST_SHA256
    )
    if sha256_file(model_root / "config.json") != MODEL_CONFIG_SHA256:
        raise NemotronSuperMechanicsError("model config hash differs")
    if sha256_file(model_root / "SOURCE_REVISION") != MODEL_SOURCE_REVISION_SHA256:
        raise NemotronSuperMechanicsError("model source revision receipt differs")
    if (model_root / "SOURCE_REVISION").read_text().strip() != MODEL_REVISION:
        raise NemotronSuperMechanicsError("model revision differs")
    load_pinned_config(model_root / "config.json")
    if sha256_file(overlay_root / "overlay_receipt.json") != OVERLAY_RECEIPT_SHA256:
        raise NemotronSuperMechanicsError("overlay receipt hash differs")

    package_versions = {
        "mamba-ssm": importlib.metadata.version("mamba-ssm"),
        "nvidia-modelopt": importlib.metadata.version("nvidia-modelopt"),
        "causal-conv1d": importlib.metadata.version("causal-conv1d"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    expected_versions = {
        "mamba-ssm": MAMBA_VERSION,
        "nvidia-modelopt": MODELOPT_VERSION,
        "causal-conv1d": CAUSAL_CONV_VERSION,
        "torch": TORCH_VERSION,
        "cuda": CUDA_VERSION,
    }
    if package_versions != expected_versions:
        raise NemotronSuperMechanicsError("mechanics package versions differ")

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
    if not Path(module_origins["mamba_ssm"]).is_relative_to(overlay_root) or not Path(
        module_origins["modelopt"]
    ).is_relative_to(overlay_root):
        raise NemotronSuperMechanicsError("mechanics module origin differs")
    if torch.cuda.device_count() != 2:
        raise NemotronSuperMechanicsError("exactly two H100 devices are required")
    devices = [torch.cuda.get_device_properties(index) for index in range(2)]
    if any("H100" not in value.name.upper() for value in devices):
        raise NemotronSuperMechanicsError("allocated device is not H100")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    enable_huggingface_checkpointing()
    tokenizer = AutoTokenizer.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=True,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=True,
        device_map="balanced",
        max_memory={0: "77GiB", 1: "77GiB", "cpu": "32GiB"},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    device_map = getattr(backbone, "hf_device_map", None)
    if not isinstance(device_map, dict) or set(device_map.values()) - {0, 1}:
        raise NemotronSuperMechanicsError("model device map differs")
    model = NemotronSuperRevisionModel(backbone)
    if model.trainable_parameter_count() != TRAINABLE_PARAMETERS_PER_ROLE:
        raise NemotronSuperMechanicsError("trainable surface differs")
    initial_state = model.trainable_state()
    initial_sha256 = _state_sha256(initial_state)

    token_ids = tokenizer.encode("Shohin mechanics only.", add_special_tokens=False)
    if not token_ids or len(token_ids) > 16:
        raise NemotronSuperMechanicsError("synthetic mechanics tokenization differs")
    input_device = backbone.model.embeddings.weight.device
    input_ids = torch.tensor([token_ids], device=input_device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    output_payload = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = output_payload.logits
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise NemotronSuperMechanicsError("full-model forward geometry differs")
    loss = logits[:, -1, :128].float().square().mean()
    if not bool(torch.isfinite(loss)):
        raise NemotronSuperMechanicsError("mechanics loss is nonfinite")
    loss.backward()
    gradients = _gradient_receipt(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    updated_state = model.trainable_state()
    updated_sha256 = _state_sha256(updated_state)
    if updated_sha256 == initial_sha256:
        raise NemotronSuperMechanicsError("Shohin update is an exact no-op")

    checkpoint = output.with_suffix(".checkpoint.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f".{checkpoint.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "schema": "shohin-nemotron-super-mechanics-checkpoint-v1",
            "trainable_state": updated_state,
            "trainable_state_sha256": updated_sha256,
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)
    with torch.no_grad():
        next(
            parameter for parameter in model.parameters() if parameter.requires_grad
        ).zero_()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "shohin-nemotron-super-mechanics-checkpoint-v1"
        or payload.get("trainable_state_sha256") != updated_sha256
        or _state_sha256(payload.get("trainable_state", {})) != updated_sha256
    ):
        raise NemotronSuperMechanicsError("mechanics checkpoint differs")
    _restore_trainables(model, payload["trainable_state"])
    if model.trainable_state_sha256() != updated_sha256:
        raise NemotronSuperMechanicsError("mechanics restore differs")

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
            {
                "index": index,
                "name": value.name,
                "total_memory": value.total_memory,
            }
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
        "checkpoint_sha256": checkpoint_sha256,
        "serialization_restore_exact": True,
        "native_router_expert_trainables": 0,
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
        raise NemotronSuperMechanicsError("GPU memory receipt differs")
    _atomic_json(output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--overlay-root", required=True, type=Path)
    parser.add_argument("--overlay-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = run(parse_args())
    except (NemotronSuperMechanicsError, NemotronSuperRevisionError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True), flush=True)
