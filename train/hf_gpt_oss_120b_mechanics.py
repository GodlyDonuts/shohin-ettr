#!/usr/bin/env python3
"""Run the score-free one-H100 GPT-OSS-120B Shohin mechanics gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any

import torch

from gpt_oss_post_mlp_revision import GptOssRevisionModel
from q36_upward_moe_gpt_oss_host import (
    CONTROLLED_LAYER_INDICES,
    MODEL_MANIFEST_SHA256,
    MODEL_LAYERS,
    MODEL_REVISION,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    sha256_file,
)

SCHEMA = "shohin-gpt-oss-120b-one-h100-mechanics-v1"
CHECKPOINT_SCHEMA = "shohin-gpt-oss-120b-mechanics-checkpoint-v1"
SEED = 2026081843
GPU_MEMORY = "77GiB"
EXPECTED_PACKAGES = {
    "huggingface-hub": "1.22.0",
    "kernels": "0.16.0",
    "kernels-data": "0.16.0",
    "torch": "2.6.0+cu124",
    "transformers": "5.15.0.dev0",
    "triton": "3.4.0",
}
OVERLAY_MODULES = {
    "kernels": "kernels",
    "kernels-data": "kernels_data",
    "triton": "triton",
}
KERNEL_COMPATIBILITY_RELATIVE = Path(
    "kernel-repo/build/torch-cuda/matmul_ogs_details/opt_flags_details/"
    "opt_flags_nvidia.py"
)
KERNEL_COMPATIBILITY_PATCHED_SHA256 = (
    "6cc30325a3df036fd56b535e0246a1a36150c7a7d66871df68a740d121bcd0ff"
)
EXPECTED_H100_MAX_SHARED_MEMORY = 232448


class GptOssMechanicsError(RuntimeError):
    """The score-free GPT-OSS mechanics contract failed."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise GptOssMechanicsError("refusing existing mechanics output")
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
        raise GptOssMechanicsError("manifest is absent or symbolic")
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
            raise GptOssMechanicsError("manifest row differs")
        seen.add(relative)
        rows.append((digest, relative))
    if not rows:
        raise GptOssMechanicsError("manifest is empty")
    return rows


def verify_manifest(root: Path, manifest: Path, expected_sha256: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    manifest = manifest.resolve(strict=True)
    if root.is_symlink() or not root.is_dir() or not manifest.is_relative_to(root):
        raise GptOssMechanicsError("manifest root differs")
    if sha256_file(manifest) != expected_sha256:
        raise GptOssMechanicsError("manifest hash differs")
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
            raise GptOssMechanicsError("manifest member differs")
        covered_bytes += candidate.stat().st_size
    actual_members: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        mode = candidate.lstat().st_mode
        if stat.S_ISDIR(mode) and not candidate.is_symlink():
            continue
        if not stat.S_ISREG(mode) or candidate.is_symlink():
            raise GptOssMechanicsError("manifest tree member differs")
        if candidate != manifest:
            actual_members.add(relative)
    if actual_members != expected_members:
        raise GptOssMechanicsError("manifest tree membership differs")
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


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _packed_sample_sha256(value: Any) -> str:
    storage = getattr(value, "storage", None)
    data = getattr(storage, "data", None)
    if not isinstance(data, torch.Tensor) or not data.numel():
        raise GptOssMechanicsError("MXFP4 packed expert storage differs")
    flat = data.detach().reshape(-1)
    width = min(4096, flat.numel())
    sample = torch.cat((flat[:width], flat[-width:])).cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(type(value).__name__.encode())
    digest.update(str(data.dtype).encode())
    digest.update(json.dumps(list(data.shape), separators=(",", ":")).encode())
    digest.update(sample.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _native_surface_receipt(model: GptOssRevisionModel) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for offset, block in enumerate(model.blocks):
        layer_index = CONTROLLED_LAYER_INDICES[offset]
        base = block.base
        experts = base.experts
        row = {
            "layer": layer_index,
            "router_weight_sha256": _tensor_sha256(base.router.weight),
            "router_bias_sha256": _tensor_sha256(base.router.bias),
            "gate_up_bias_sha256": _tensor_sha256(experts.gate_up_proj_bias),
            "down_bias_sha256": _tensor_sha256(experts.down_proj_bias),
        }
        if offset in (0, len(model.blocks) - 1):
            row.update(
                {
                    "gate_up_packed_sample_sha256": _packed_sample_sha256(
                        experts.gate_up_proj
                    ),
                    "down_packed_sample_sha256": _packed_sample_sha256(
                        experts.down_proj
                    ),
                }
            )
        rows.append(row)
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    return {
        "layers": len(rows),
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
        "packed_samples": 4,
    }


def _gradient_receipt(model: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        finite = bool(gradient is not None and torch.isfinite(gradient).all())
        norm = float(gradient.float().norm().detach().cpu()) if finite else None
        rows.append({"name": name, "finite": finite, "norm": norm})
    adapter_b = [row for row in rows if row["name"].endswith("adapter_b.weight")]
    if (
        len(rows) != 32
        or len(adapter_b) != 16
        or not all(row["finite"] for row in rows)
        or not all(float(row["norm"] or 0.0) > 0.0 for row in adapter_b)
    ):
        raise GptOssMechanicsError("Shohin MXFP4 gradient receipt differs")
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    return {
        "parameters": len(rows),
        "nonzero_gradients": sum(float(row["norm"] or 0.0) > 0.0 for row in rows),
        "adapter_b_nonzero_gradients": len(adapter_b),
        "earliest_controlled_layer_nonzero": float(adapter_b[0]["norm"] or 0.0) > 0.0,
        "latest_controlled_layer_nonzero": float(adapter_b[-1]["norm"] or 0.0) > 0.0,
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _restore_trainables(
    model: GptOssRevisionModel, state: dict[str, torch.Tensor]
) -> None:
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(state):
        raise GptOssMechanicsError("serialized trainable names differ")
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise GptOssMechanicsError("serialized trainable geometry differs")
            parameter.copy_(value.to(parameter.device))


def _package_receipt(overlay_root: Path) -> dict[str, Any]:
    versions = {
        name: (
            torch.__version__ if name == "torch" else importlib.metadata.version(name)
        )
        for name in EXPECTED_PACKAGES
    }
    if versions != EXPECTED_PACKAGES:
        raise GptOssMechanicsError(
            f"mechanics package versions differ: expected={EXPECTED_PACKAGES!r} observed={versions!r}"
        )
    origins: dict[str, str] = {}
    for distribution, module in OVERLAY_MODULES.items():
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise GptOssMechanicsError("overlay module origin is absent")
        origin = Path(spec.origin).resolve(strict=True)
        if not origin.is_relative_to(overlay_root):
            raise GptOssMechanicsError("overlay module origin differs")
        origins[distribution] = origin.relative_to(overlay_root).as_posix()
    return {"versions": versions, "overlay_module_origins": origins}


def _kernel_compatibility_receipt(
    overlay_root: Path,
    max_shared_memory: int,
    *,
    torch_property_present: bool,
) -> dict[str, Any]:
    target = overlay_root / KERNEL_COMPATIBILITY_RELATIVE
    mode = target.lstat().st_mode if target.exists() else 0
    observed_sha256 = sha256_file(target) if stat.S_ISREG(mode) else None
    if (
        target.is_symlink()
        or not stat.S_ISREG(mode)
        or observed_sha256 != KERNEL_COMPATIBILITY_PATCHED_SHA256
        or isinstance(max_shared_memory, bool)
        or max_shared_memory != EXPECTED_H100_MAX_SHARED_MEMORY
        or torch_property_present is not False
    ):
        raise GptOssMechanicsError("MXFP4 kernel compatibility receipt differs")
    return {
        "relative_path": KERNEL_COMPATIBILITY_RELATIVE.as_posix(),
        "patched_sha256": observed_sha256,
        "shared_memory_source": "triton.compiler.compiler.max_shared_mem(0)",
        "max_shared_memory_bytes": max_shared_memory,
        "torch_device_property_absent": True,
    }


def _is_cuda_zero(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == 0
    if isinstance(value, str):
        return value in {"cuda", "cuda:0"}
    if isinstance(value, torch.device):
        return value.type == "cuda" and value.index in {None, 0}
    return False


def _tensor_is_cuda_zero(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and _is_cuda_zero(value.device)


def _cuda_residency_receipt(backbone: Any) -> dict[str, Any]:
    parameters = list(backbone.named_parameters())
    buffers = list(backbone.named_buffers())
    layers = getattr(getattr(backbone, "model", None), "layers", None)
    packed: list[torch.Tensor] = []
    if layers is not None and len(layers) == MODEL_LAYERS:
        for layer in layers:
            experts = getattr(getattr(layer, "mlp", None), "experts", None)
            for name in ("gate_up_proj", "down_proj"):
                storage = getattr(getattr(experts, name, None), "storage", None)
                data = getattr(storage, "data", None)
                if isinstance(data, torch.Tensor):
                    packed.append(data)
    parameter_devices = sorted({str(value.device) for _, value in parameters})
    buffer_devices = sorted({str(value.device) for _, value in buffers})
    packed_devices = sorted({str(value.device) for value in packed})
    receipt = {
        "parameter_tensors": len(parameters),
        "parameter_devices": parameter_devices,
        "buffer_tensors": len(buffers),
        "buffer_devices": buffer_devices,
        "packed_expert_tensors": len(packed),
        "packed_expert_devices": packed_devices,
        "expected_packed_expert_tensors": 2 * MODEL_LAYERS,
        "all_parameters_cuda_zero": bool(
            parameters and all(_tensor_is_cuda_zero(value) for _, value in parameters)
        ),
        "all_buffers_cuda_zero": all(
            _tensor_is_cuda_zero(value) for _, value in buffers
        ),
        "all_packed_experts_cuda_zero": bool(
            len(packed) == 2 * MODEL_LAYERS
            and all(_tensor_is_cuda_zero(value) for value in packed)
        ),
    }
    if not all(
        receipt[name] is True
        for name in (
            "all_parameters_cuda_zero",
            "all_buffers_cuda_zero",
            "all_packed_experts_cuda_zero",
        )
    ):
        raise GptOssMechanicsError(
            "native MXFP4 CUDA residency differs: "
            + json.dumps(receipt, sort_keys=True, separators=(",", ":"))
        )
    return receipt


def _native_mxfp4_load_receipt(backbone: Any) -> dict[str, Any]:
    """Normalize equivalent CUDA:0 spellings while rejecting offload/dequantize."""

    device_map = getattr(backbone, "hf_device_map", None)
    quantizer = getattr(backbone, "hf_quantizer", None)
    quantization_config = getattr(quantizer, "quantization_config", None)
    explicit_device_map_cuda_zero = bool(
        isinstance(device_map, dict)
        and device_map
        and all(_is_cuda_zero(value) for value in device_map.values())
    )
    receipt = {
        "device_map_type": type(device_map).__name__,
        "device_map": (
            {str(name): str(device) for name, device in sorted(device_map.items())}
            if isinstance(device_map, dict)
            else None
        ),
        "device_map_mode": (
            "absent_single_device_load"
            if device_map is None
            else "explicit_cuda_zero" if explicit_device_map_cuda_zero else "invalid"
        ),
        "quantizer_class": type(quantizer).__name__,
        "quantization_config_class": type(quantization_config).__name__,
        "dequantize": getattr(quantization_config, "dequantize", None),
    }
    if (
        receipt["device_map_mode"]
        not in {"absent_single_device_load", "explicit_cuda_zero"}
        or receipt["quantizer_class"] != "Mxfp4HfQuantizer"
        or receipt["quantization_config_class"] != "Mxfp4Config"
        or receipt["dequantize"] is not False
    ):
        raise GptOssMechanicsError(
            "native MXFP4 load differs: "
            + json.dumps(receipt, sort_keys=True, separators=(",", ":"))
        )
    receipt["cuda_residency"] = _cuda_residency_receipt(backbone)
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    model_root = args.model_root.resolve(strict=True)
    overlay_root = args.overlay_root.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise GptOssMechanicsError("mechanics output already exists")
    model_receipt = verify_manifest(
        model_root, args.model_manifest, args.expected_model_manifest_sha256
    )
    overlay_receipt = verify_manifest(
        overlay_root, args.overlay_manifest, args.expected_overlay_manifest_sha256
    )
    if args.expected_model_manifest_sha256 != MODEL_MANIFEST_SHA256:
        raise GptOssMechanicsError("model manifest authorization differs")
    load_pinned_config(model_root / "config.json")
    revision = model_root / "SOURCE_REVISION"
    if (
        revision.is_symlink()
        or not revision.is_file()
        or revision.read_text(encoding="utf-8") != f"{MODEL_REVISION}\n"
    ):
        raise GptOssMechanicsError("model source revision differs")
    expected_kernel_root = overlay_root / "kernel-repo"
    if (
        os.environ.get("LOCAL_KERNELS")
        != (f"kernels-community/gpt-oss-triton-kernels={expected_kernel_root}")
        or os.environ.get("HF_HUB_OFFLINE") != "1"
    ):
        raise GptOssMechanicsError("offline kernel environment differs")
    packages = _package_receipt(overlay_root)
    if torch.cuda.device_count() != 1:
        raise GptOssMechanicsError("exactly one H100 device is required")
    device = torch.cuda.get_device_properties(0)
    if "H100" not in device.name.upper():
        raise GptOssMechanicsError("allocated device is not H100")

    from kernels import get_loaded_kernels
    from triton.compiler.compiler import max_shared_mem
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kernel_compatibility = _kernel_compatibility_receipt(
        overlay_root,
        int(max_shared_mem(0)),
        torch_property_present=hasattr(device, "shared_memory_per_block_optin"),
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
        device_map={"": 0},
        max_memory={0: GPU_MEMORY, "cpu": "8GiB"},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    native_load = _native_mxfp4_load_receipt(backbone)
    loaded_kernels = get_loaded_kernels()
    if len(loaded_kernels) != 1:
        raise GptOssMechanicsError("loaded kernel count differs")

    model = GptOssRevisionModel(backbone)
    if model.trainable_parameter_count() != TRAINABLE_PARAMETERS_PER_ROLE:
        raise GptOssMechanicsError("trainable surface differs")
    optimizer_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(optimizer_parameters) != 32:
        raise GptOssMechanicsError("optimizer parameter surface differs")
    initial_state = model.trainable_state()
    initial_sha256 = _state_sha256(initial_state)
    native_before = _native_surface_receipt(model)

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Shohin GPT-OSS mechanics only."}],
        add_generation_prompt=True,
        tokenize=False,
    )
    tokenized = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized["input_ids"].to("cuda:0")
    attention_mask = tokenized["attention_mask"].to("cuda:0")
    if input_ids.numel() < 1 or input_ids.numel() > 128:
        raise GptOssMechanicsError("synthetic mechanics tokenization differs")
    model.train()
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=1e-4, foreach=False)
    torch.cuda.reset_peak_memory_stats(0)
    output_payload = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    logits = output_payload.logits
    if logits.ndim != 3 or logits.shape[:2] != (1, 1):
        raise GptOssMechanicsError("full-model forward geometry differs")
    loss = logits[:, -1, :128].float().square().mean()
    if not bool(torch.isfinite(loss)):
        raise GptOssMechanicsError("mechanics loss is nonfinite")
    loss.backward()
    gradients = _gradient_receipt(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    native_after = _native_surface_receipt(model)
    if native_after != native_before:
        raise GptOssMechanicsError("mechanics changed the native MXFP4 surface")
    updated_state = model.trainable_state()
    updated_sha256 = _state_sha256(updated_state)
    if updated_sha256 == initial_sha256:
        raise GptOssMechanicsError("Shohin update is an exact no-op")

    checkpoint = output.with_suffix(".checkpoint.pt")
    if checkpoint.exists() or checkpoint.is_symlink():
        raise GptOssMechanicsError("refusing existing mechanics checkpoint")
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
        optimizer_parameters[0].zero_()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("trainable_state_sha256") != updated_sha256
        or _state_sha256(payload.get("trainable_state", {})) != updated_sha256
    ):
        raise GptOssMechanicsError("mechanics checkpoint differs")
    _restore_trainables(model, payload["trainable_state"])
    if model.trainable_state_sha256() != updated_sha256:
        raise GptOssMechanicsError("mechanics restore differs")

    report = {
        "schema": SCHEMA,
        "status": "pass",
        "model_id": "openai/gpt-oss-120b",
        "model_revision": MODEL_REVISION,
        "host_total_parameters": 117_000_000_000,
        "host_active_parameters": 5_100_000_000,
        "gpu": device.name,
        "gpu_count": 1,
        "packages": packages,
        "model_receipt": model_receipt,
        "overlay_receipt": overlay_receipt,
        "kernel_compatibility": kernel_compatibility,
        "native_quantization": "mxfp4",
        "native_load": native_load,
        "loaded_kernel_count": len(loaded_kernels),
        "trainable_parameters": model.trainable_parameter_count(),
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "native_router_expert_trainables": 0,
        "native_surface_before": native_before,
        "native_surface_after": native_after,
        "initial_trainable_state_sha256": initial_sha256,
        "updated_trainable_state_sha256": updated_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_restore_exact": True,
        "gradients": gradients,
        "loss": float(loss.detach().cpu()),
        "synthetic_prompt_tokens": int(input_ids.numel()),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(0)),
        "elapsed_seconds": time.monotonic() - started,
        "score_or_assessor_data_accessed": False,
        "scientific_result": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }
    if not math.isfinite(report["loss"]):
        raise GptOssMechanicsError("mechanics loss receipt is nonfinite")
    _atomic_json(output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-model-manifest-sha256",
        default=MODEL_MANIFEST_SHA256,
    )
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path, required=True)
    parser.add_argument("--expected-overlay-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
