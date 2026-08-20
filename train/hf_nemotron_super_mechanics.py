#!/usr/bin/env python3
"""Run the score-free two-H100 Nemotron Super load/attach/restore gate."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from nemotron_super_post_mixer_revision import (
    NemotronSuperRevisionError,
    NemotronSuperRevisionModel,
)
from q36_upward_moe_host import (
    LAYER_TYPES,
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
TRAINING_GRADIENT_ACCUMULATION = 8
TRAINING_LEARNING_RATE = 2e-5
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
TRITON_VERSION = "3.2.0"
HF_QUANT_CONFIG_SHA256 = (
    "827209265a15cc7161e96773c4538da60f0980288dc0b86dd5dc2f906a5cfb4f"
)
MODEL_INDEX_SHA256 = "126f3105feb375f4f0390aa2b339d5d27d37d1dc720f797bafb4263252c12628"
MODELOPT_EXPORT_VERSION = "0.41.0"
FP8_LINEAR_COUNT = 41_120
MODEL_WEIGHT_MAP_ENTRIES = 124_941
MODEL_WEIGHT_SHARDS = 26
BACKBONE_WEIGHT_MAP_ENTRIES = 123_898
MTP_WEIGHT_MAP_ENTRIES = 1_042
LM_HEAD_WEIGHT_MAP_ENTRIES = 1
FP8_SCALE_MULTIPLIER = 448.0
KV_CACHE_SCALE_COUNT = 16
KV_CACHE_AMAX_NAMES_SHA256 = (
    "d4b7afa82f4f1ceef8ce45e8e03bbfec03eb4237c7a5cda043b86af24f5c9d27"
)
FP8_MODULE_NAMES_SHA256 = (
    "cd608a21448741388b00a63bfd25cf38040029f6196685e0e8838247de307912"
)
MODELOPT_IGNORE_PATTERNS = 130
MODELOPT_BACKBONE_IGNORE_PATTERNS = 128
MODELOPT_SOURCE_IGNORE_SHA256 = (
    "a3ab4871ac4c811c37fa01d964c3364972ddc959e6f96b4ebcb23ffcd799266c"
)
MODELOPT_TARGET_IGNORE_SHA256 = (
    "8180f3be29deacecaeb02701156cb5c80e688c3bddd125d5a9f1aa574660b832"
)
REMOTE_CONFIGURATION_SHA256 = (
    "0fc818c10506c91bd02df5a605f49cb0704b5498954f46dbde2d63999ae36c3d"
)
REMOTE_MODELING_SHA256 = (
    "e1cb5fc02e887983f0a445bf4c1a2604453b2cb2db4624c7004dcf663bbb1b6e"
)
MODELOPT_FP8_BACKEND_SHA256 = (
    "4a68f8dfd2df4ec3ff472b701816c8a2d32a71fa1ee0e8691e8804fe28780cb2"
)
MAMBA_LAYER_INDICES = tuple(
    index for index, layer_type in enumerate(LAYER_TYPES) if layer_type == "mamba"
)
MAMBA_OUTPUT_PROJECTION_NAMES = tuple(
    f"model.layers.{index}.mixer.out_proj" for index in MAMBA_LAYER_INDICES
)
MAMBA_OUTPUT_PROJECTION_NAMES_SHA256 = hashlib.sha256(
    json.dumps(list(MAMBA_OUTPUT_PROJECTION_NAMES), separators=(",", ":")).encode()
).hexdigest()
MOE_LAYER_INDICES = tuple(
    index for index, layer_type in enumerate(LAYER_TYPES) if layer_type == "moe"
)
MOE_MIXER_NAMES = tuple(f"model.layers.{index}.mixer" for index in MOE_LAYER_INDICES)
MOE_MIXER_NAMES_SHA256 = hashlib.sha256(
    json.dumps(list(MOE_MIXER_NAMES), separators=(",", ":")).encode()
).hexdigest()
ROUTED_EXPERTS_PER_LAYER = 512


class NemotronSuperMechanicsError(RuntimeError):
    """The score-free upward-MoE mechanics contract failed."""


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def training_objective_receipt_is_exact(payload: Any) -> bool:
    """Validate that mechanics exercised the frozen trainer's actual objective."""

    if not isinstance(payload, dict) or set(payload) != {
        "objective",
        "prompt_tokens",
        "response_tokens",
        "ignore_index",
        "gradient_accumulation_scale",
        "learning_rate",
        "autocast_dtype",
    }:
        return False
    prompt_tokens = payload.get("prompt_tokens")
    response_tokens = payload.get("response_tokens")
    ignore_index = payload.get("ignore_index")
    accumulation_scale = payload.get("gradient_accumulation_scale")
    learning_rate = payload.get("learning_rate")
    return bool(
        payload.get("objective") == "response_only_next_token_cross_entropy"
        and isinstance(prompt_tokens, int)
        and not isinstance(prompt_tokens, bool)
        and prompt_tokens > 0
        and isinstance(response_tokens, int)
        and not isinstance(response_tokens, bool)
        and response_tokens > 0
        and isinstance(ignore_index, int)
        and not isinstance(ignore_index, bool)
        and ignore_index == -100
        and isinstance(accumulation_scale, int)
        and not isinstance(accumulation_scale, bool)
        and accumulation_scale == TRAINING_GRADIENT_ACCUMULATION
        and isinstance(learning_rate, float)
        and learning_rate == TRAINING_LEARNING_RATE
        and payload.get("autocast_dtype") == "torch.bfloat16"
    )


def _modelopt_fp8_quantization_config(
    model_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct the pinned ModelOpt FP8 loader config without mutating the host."""

    config_path = model_root / "config.json"
    legacy_path = model_root / "hf_quant_config.json"
    index_path = model_root / "model.safetensors.index.json"
    if (
        sha256_file(legacy_path) != HF_QUANT_CONFIG_SHA256
        or sha256_file(index_path) != MODEL_INDEX_SHA256
    ):
        raise NemotronSuperMechanicsError("ModelOpt export receipts differ")
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    exported = config_payload.get("quantization_config")
    if not isinstance(exported, dict) or not isinstance(legacy_payload, dict):
        raise NemotronSuperMechanicsError("ModelOpt export configuration differs")

    from modelopt.torch.export.convert_hf_config import convert_hf_quant_config_format

    converted = convert_hf_quant_config_format(legacy_payload)
    if converted != exported:
        raise NemotronSuperMechanicsError("ModelOpt converted configuration differs")
    import modelopt.torch.quantization as mtq

    if (
        exported.get("quant_method") != "modelopt"
        or exported.get("quant_algo") != "FP8"
        or exported.get("producer")
        != {"name": "modelopt", "version": MODELOPT_EXPORT_VERSION}
        or exported.get("kv_cache_scheme")
        != {"dynamic": False, "num_bits": 8, "type": "float"}
        or list(exported.get("config_groups", {})) != ["group_0"]
        or exported["config_groups"]["group_0"].get("targets") != ["Linear"]
        or not isinstance(exported.get("ignore"), list)
        or len(exported["ignore"]) != MODELOPT_IGNORE_PATTERNS
    ):
        raise NemotronSuperMechanicsError("ModelOpt FP8 export contract differs")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise NemotronSuperMechanicsError("ModelOpt weight index differs")
    weight_scales = {name for name in weight_map if name.endswith(".weight_scale")}
    input_scales = {name for name in weight_map if name.endswith(".input_scale")}
    fp8_weights = {
        name
        for name in weight_map
        if name.endswith(".weight") and f"{name[:-7]}.weight_scale" in weight_map
    }
    kv_cache_amax_names = frozenset(
        f"model.{name[len('backbone.') : -len(source_suffix)]}{target_suffix}"
        for name in weight_map
        for source_suffix, target_suffix in (
            (".k_proj.k_scale", ".k_bmm_quantizer._amax"),
            (".v_proj.v_scale", ".v_bmm_quantizer._amax"),
        )
        if name.startswith("backbone.") and name.endswith(source_suffix)
    )
    if (
        len(weight_map) != MODEL_WEIGHT_MAP_ENTRIES
        or len(set(weight_map.values())) != MODEL_WEIGHT_SHARDS
        or len(weight_scales) != FP8_LINEAR_COUNT
        or len(input_scales) != FP8_LINEAR_COUNT
        or len(fp8_weights) != FP8_LINEAR_COUNT
        or len(kv_cache_amax_names) != KV_CACHE_SCALE_COUNT
        or _canonical_sha256(sorted(kv_cache_amax_names)) != KV_CACHE_AMAX_NAMES_SHA256
    ):
        raise NemotronSuperMechanicsError("ModelOpt FP8 tensor geometry differs")

    quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    if (
        quant_cfg.get("algorithm") != "max"
        or quant_cfg.get("quant_cfg", {}).get("*weight_quantizer")
        != {"num_bits": (4, 3), "axis": None}
        or quant_cfg.get("quant_cfg", {}).get("*input_quantizer")
        != {"num_bits": (4, 3), "axis": None}
    ):
        raise NemotronSuperMechanicsError("pinned ModelOpt FP8 defaults differ")
    source_disabled_patterns: list[str] = []
    disabled_patterns: list[str] = []
    renamed_disabled_patterns = 0
    for value in exported["ignore"]:
        if (
            not isinstance(value, str)
            or not value
            or value.startswith("/")
            or ".." in value
        ):
            raise NemotronSuperMechanicsError("ModelOpt ignore pattern differs")
        source_pattern = value if any(mark in value for mark in "*?[") else f"{value}*"
        # The immutable export names the causal stack ``backbone`` whereas the
        # pinned Transformers implementation names the same stack ``model``.
        # ModelOpt matches these patterns against the instantiated module
        # namespace, so replay the same namespace translation used for streamed
        # checkpoint tensors. Leaving the export spelling untouched silently
        # quantizes modules that the producer explicitly excluded.
        if source_pattern.startswith("backbone."):
            pattern = f"model.{source_pattern[len('backbone.') :]}"
            renamed_disabled_patterns += 1
        else:
            pattern = source_pattern
        quant_cfg["quant_cfg"][pattern] = {"enable": False}
        source_disabled_patterns.append(source_pattern)
        disabled_patterns.append(pattern)
    # KV-cache quantization is a distinct producer-declared surface. The
    # checkpoint stores its calibrated amax values as k_proj/v_proj scales,
    # including on attention blocks whose Linear weights are intentionally
    # excluded. Re-enable only the exact hash-bound quantizers after applying
    # the broader Linear exclusion patterns.
    for amax_name in sorted(kv_cache_amax_names):
        quant_cfg["quant_cfg"][amax_name.removesuffix("._amax")] = {
            "num_bits": (4, 3),
            "axis": None,
        }
    receipt = {
        "hf_quant_config_sha256": HF_QUANT_CONFIG_SHA256,
        "model_index_sha256": MODEL_INDEX_SHA256,
        "exported_quantization_config_sha256": _canonical_sha256(exported),
        "converted_quantization_config_sha256": _canonical_sha256(converted),
        "modelopt_loader_config_sha256": _canonical_sha256(quant_cfg),
        "modelopt_export_version": MODELOPT_EXPORT_VERSION,
        "fp8_linear_count": FP8_LINEAR_COUNT,
        "kv_cache_scale_count": len(kv_cache_amax_names),
        "kv_cache_amax_names_sha256": _canonical_sha256(sorted(kv_cache_amax_names)),
        "kv_cache_scheme": exported["kv_cache_scheme"],
        "weight_map_entries": len(weight_map),
        "weight_shards": len(set(weight_map.values())),
        "disabled_patterns": len(disabled_patterns),
        "renamed_disabled_patterns": renamed_disabled_patterns,
        "disabled_pattern_source_prefix": "backbone.",
        "disabled_pattern_target_prefix": "model.",
        "source_disabled_pattern_sha256": _canonical_sha256(source_disabled_patterns),
        "disabled_pattern_sha256": _canonical_sha256(disabled_patterns),
        "quant_gemm": True,
    }
    return quant_cfg, receipt


def _cuda_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in {0, 1} else None
    if isinstance(value, torch.device):
        return value.index if value.type == "cuda" and value.index in {0, 1} else None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "cuda", "cuda:0"}:
            return 0
        if normalized in {"1", "cuda:1"}:
            return 1
    return None


def _expected_fp8_module_names(model_root: Path) -> frozenset[str]:
    index = json.loads(
        (model_root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise NemotronSuperMechanicsError("ModelOpt weight index differs")
    names = frozenset(
        f"model.{name[len('backbone.') : -len('.weight_scale')]}"
        for name in weight_map
        if name.startswith("backbone.") and name.endswith(".weight_scale")
    )
    if (
        len(names) != FP8_LINEAR_COUNT
        or _canonical_sha256(sorted(names)) != FP8_MODULE_NAMES_SHA256
    ):
        raise NemotronSuperMechanicsError("ModelOpt FP8 module identities differ")
    return names


def _expected_kv_cache_amax_names(model_root: Path) -> frozenset[str]:
    index = json.loads(
        (model_root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise NemotronSuperMechanicsError("ModelOpt weight index differs")
    names = frozenset(
        f"model.{name[len('backbone.') : -len(source_suffix)]}{target_suffix}"
        for name in weight_map
        for source_suffix, target_suffix in (
            (".k_proj.k_scale", ".k_bmm_quantizer._amax"),
            (".v_proj.v_scale", ".v_bmm_quantizer._amax"),
        )
        if name.startswith("backbone.") and name.endswith(source_suffix)
    )
    if (
        len(names) != KV_CACHE_SCALE_COUNT
        or _canonical_sha256(sorted(names)) != KV_CACHE_AMAX_NAMES_SHA256
    ):
        raise NemotronSuperMechanicsError("ModelOpt KV-cache identities differ")
    return names


@contextmanager
def _translate_export_checkpoint_keys(
    expected_fp8_modules: frozenset[str],
    expected_kv_cache_amax: frozenset[str],
) -> Any:
    """Translate the pinned Megatron export namespace during streamed loading.

    The immutable checkpoint names the causal backbone ``backbone`` and also
    carries a one-layer MTP head. The pinned remote Transformers class names
    the same causal backbone ``model`` and intentionally has no MTP module.
    Accelerate loads one safetensors shard at a time, so translate those keys
    in memory without rewriting or duplicating the 120B checkpoint.
    """

    import accelerate.utils.modeling as accelerate_modeling

    original = accelerate_modeling.load_state_dict
    modelopt_accelerate = sys.modules.get(
        "modelopt.torch.quantization.plugins.accelerate"
    )
    if modelopt_accelerate is None or not hasattr(
        modelopt_accelerate, "load_checkpoint_and_dispatch"
    ):
        raise NemotronSuperMechanicsError("ModelOpt Accelerate loader differs")
    original_dispatch = modelopt_accelerate.load_checkpoint_and_dispatch
    counts: Counter[str] = Counter()

    def translated_load_state_dict(*args: Any, **kwargs: Any) -> dict[str, Any]:
        state = original(*args, **kwargs)
        if not isinstance(state, dict):
            raise NemotronSuperMechanicsError("checkpoint shard state differs")
        translated: dict[str, Any] = {}
        for name, value in state.items():
            if name.startswith("backbone."):
                translated_name = f"model.{name[len('backbone.'):]}"
                counts["backbone_to_model"] += 1
            elif name.startswith("mtp."):
                counts["mtp_ignored"] += 1
                continue
            elif name == "lm_head.weight":
                translated_name = name
                counts["lm_head_unchanged"] += 1
            else:
                raise NemotronSuperMechanicsError("checkpoint export namespace differs")
            if translated_name.endswith(".input_scale"):
                translated_name = (
                    f"{translated_name[:-len('.input_scale')]}" ".input_quantizer._amax"
                )
                value = value.to(dtype=torch.float32) * FP8_SCALE_MULTIPLIER
                counts["input_scale_to_amax"] += 1
            elif translated_name.endswith(".weight_scale"):
                translated_name = (
                    f"{translated_name[:-len('.weight_scale')]}"
                    ".weight_quantizer._amax"
                )
                value = value.to(dtype=torch.float32) * FP8_SCALE_MULTIPLIER
                counts["weight_scale_to_amax"] += 1
            elif translated_name.endswith(".k_proj.k_scale"):
                translated_name = (
                    f"{translated_name[:-len('.k_proj.k_scale')]}"
                    ".k_bmm_quantizer._amax"
                )
                value = value.to(dtype=torch.float32) * FP8_SCALE_MULTIPLIER
                counts["k_scale_to_amax"] += 1
            elif translated_name.endswith(".v_proj.v_scale"):
                translated_name = (
                    f"{translated_name[:-len('.v_proj.v_scale')]}"
                    ".v_bmm_quantizer._amax"
                )
                value = value.to(dtype=torch.float32) * FP8_SCALE_MULTIPLIER
                counts["v_scale_to_amax"] += 1
            if translated_name in translated:
                raise NemotronSuperMechanicsError(
                    "checkpoint export translation collides"
                )
            translated[translated_name] = value
        return translated

    def prepared_load_checkpoint_and_dispatch(
        model: Any, *args: Any, **kwargs: Any
    ) -> Any:
        _prepare_fp8_scale_buffers(
            model, counts, expected_fp8_modules, expected_kv_cache_amax
        )
        return original_dispatch(model, *args, **kwargs)

    accelerate_modeling.load_state_dict = translated_load_state_dict
    modelopt_accelerate.load_checkpoint_and_dispatch = (
        prepared_load_checkpoint_and_dispatch
    )
    try:
        yield counts
    finally:
        accelerate_modeling.load_state_dict = original
        modelopt_accelerate.load_checkpoint_and_dispatch = original_dispatch


def _enabled_fp8_linears(model: Any) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        weight_quantizer = getattr(module, "weight_quantizer", None)
        input_quantizer = getattr(module, "input_quantizer", None)
        if (
            not isinstance(weight, torch.Tensor)
            or weight.dim() != 2
            or weight_quantizer is None
            or input_quantizer is None
            or getattr(weight_quantizer, "is_enabled", None) is not True
            or getattr(input_quantizer, "is_enabled", None) is not True
        ):
            continue
        observed[name] = module
    return observed


def _prepare_fp8_scale_buffers(
    model: Any,
    counts: Counter[str],
    expected_fp8_modules: frozenset[str],
    expected_kv_cache_amax: frozenset[str],
) -> None:
    quantized_linears = _enabled_fp8_linears(model)
    if set(quantized_linears) != expected_fp8_modules:
        raise NemotronSuperMechanicsError("ModelOpt scale-buffer geometry differs")
    counts["enabled_fp8_module_identities"] = len(quantized_linears)
    for module in quantized_linears.values():
        weight_quantizer = module.weight_quantizer
        input_quantizer = module.input_quantizer
        weight_amax = getattr(weight_quantizer, "_amax", None)
        if (
            not isinstance(weight_amax, torch.Tensor)
            or not weight_amax.is_meta
            or weight_amax.dtype != torch.bfloat16
            or weight_amax.shape != torch.Size([])
        ):
            raise NemotronSuperMechanicsError("ModelOpt weight amax pre-state differs")
        weight_quantizer._buffers["_amax"] = weight_amax.to(dtype=torch.float32)
        counts["weight_amax_placeholders"] += 1
        if hasattr(input_quantizer, "_amax"):
            raise NemotronSuperMechanicsError("ModelOpt input amax pre-state differs")
        input_quantizer.register_buffer(
            "_amax", torch.empty((), dtype=torch.float32, device="meta")
        )
        counts["input_amax_placeholders"] += 1
    modules = dict(model.named_modules())
    for amax_name in expected_kv_cache_amax:
        quantizer_name = amax_name.removesuffix("._amax")
        quantizer = modules.get(quantizer_name)
        if (
            quantizer is None
            or getattr(quantizer, "is_enabled", None) is not True
            or getattr(quantizer, "num_bits", None) != (4, 3)
            or hasattr(quantizer, "_amax")
        ):
            raise NemotronSuperMechanicsError(
                "ModelOpt KV-cache amax pre-state differs"
            )
        quantizer.register_buffer(
            "_amax", torch.empty((), dtype=torch.float32, device="meta")
        )
        counts["kv_cache_amax_placeholders"] += 1
    counts["kv_cache_amax_identities"] = len(expected_kv_cache_amax)


def _checkpoint_translation_receipt(counts: Counter[str]) -> dict[str, Any]:
    expected = {
        "backbone_to_model": BACKBONE_WEIGHT_MAP_ENTRIES,
        "mtp_ignored": MTP_WEIGHT_MAP_ENTRIES,
        "lm_head_unchanged": LM_HEAD_WEIGHT_MAP_ENTRIES,
        "input_scale_to_amax": FP8_LINEAR_COUNT,
        "weight_scale_to_amax": FP8_LINEAR_COUNT,
        "input_amax_placeholders": FP8_LINEAR_COUNT,
        "weight_amax_placeholders": FP8_LINEAR_COUNT,
        "enabled_fp8_module_identities": FP8_LINEAR_COUNT,
        "k_scale_to_amax": KV_CACHE_SCALE_COUNT // 2,
        "v_scale_to_amax": KV_CACHE_SCALE_COUNT // 2,
        "kv_cache_amax_placeholders": KV_CACHE_SCALE_COUNT,
        "kv_cache_amax_identities": KV_CACHE_SCALE_COUNT,
    }
    observed = {name: counts.get(name, 0) for name in expected}
    namespace_total = sum(
        observed[name]
        for name in ("backbone_to_model", "mtp_ignored", "lm_head_unchanged")
    )
    if observed != expected or namespace_total != MODEL_WEIGHT_MAP_ENTRIES:
        raise NemotronSuperMechanicsError("checkpoint export translation differs")
    return {
        **observed,
        "source_prefix": "backbone.",
        "target_prefix": "model.",
        "mtp_policy": "ignored_not_implemented_by_remote_causal_lm",
        "scale_to_amax_multiplier": FP8_SCALE_MULTIPLIER,
        "enabled_fp8_module_names_sha256": FP8_MODULE_NAMES_SHA256,
        "weight_amax_pre_dtype": "torch.bfloat16",
        "weight_amax_dtype": "torch.float32",
        "weight_amax_shape": [],
        "input_amax_shape": [],
        "kv_cache_amax_shape": [],
        "kv_cache_amax_names_sha256": KV_CACHE_AMAX_NAMES_SHA256,
        "translation_sha256": _canonical_sha256(observed),
    }


def _modelopt_fp8_runtime_receipt(
    backbone: Any,
    expected_fp8_modules: frozenset[str],
    expected_kv_cache_amax: frozenset[str],
) -> dict[str, Any]:
    device_map = getattr(backbone, "hf_device_map", None)
    if not isinstance(device_map, dict) or not device_map:
        raise NemotronSuperMechanicsError("ModelOpt device map differs")
    normalized_map = {name: _cuda_index(value) for name, value in device_map.items()}
    if None in normalized_map.values() or set(normalized_map.values()) != {0, 1}:
        raise NemotronSuperMechanicsError("ModelOpt device placement differs")

    parameter_devices = Counter(str(value.device) for value in backbone.parameters())
    buffer_devices = Counter(str(value.device) for value in backbone.buffers())
    if any(not name.startswith("cuda:") for name in parameter_devices | buffer_devices):
        raise NemotronSuperMechanicsError("ModelOpt tensor residency differs")
    observed_devices = {
        _cuda_index(name) for name in set(parameter_devices) | set(buffer_devices)
    }
    if None in observed_devices or observed_devices != {0, 1}:
        raise NemotronSuperMechanicsError("ModelOpt tensor device coverage differs")

    from modelopt.torch.quantization.backends.gemm_registry import (
        is_real_quant_gemm_enabled,
    )

    quantized_linears = _enabled_fp8_linears(backbone)
    if (
        set(quantized_linears) != expected_fp8_modules
        or not is_real_quant_gemm_enabled(backbone)
        or any(
            getattr(module.weight_quantizer, "fake_quant", True)
            for module in quantized_linears.values()
        )
    ):
        raise NemotronSuperMechanicsError("ModelOpt real FP8 execution differs")
    modules = dict(backbone.named_modules())
    kv_cache_devices: Counter[str] = Counter()
    for amax_name in expected_kv_cache_amax:
        quantizer = modules.get(amax_name.removesuffix("._amax"))
        amax = getattr(quantizer, "_amax", None)
        if (
            quantizer is None
            or getattr(quantizer, "is_enabled", None) is not True
            or not isinstance(amax, torch.Tensor)
            or amax.shape != torch.Size([])
            or amax.dtype != torch.float32
            or amax.is_meta
            or not bool(torch.isfinite(amax))
            or not bool(amax > 0)
            or _cuda_index(amax.device) is None
        ):
            raise NemotronSuperMechanicsError("ModelOpt KV-cache FP8 runtime differs")
        kv_cache_devices[str(amax.device)] += 1
    return {
        "device_map_sha256": _canonical_sha256(normalized_map),
        "device_map_entries": len(normalized_map),
        "parameter_devices": dict(sorted(parameter_devices.items())),
        "buffer_devices": dict(sorted(buffer_devices.items())),
        "real_quant_gemm_enabled": True,
        "real_fp8_linear_count": len(quantized_linears),
        "enabled_fp8_module_names_sha256": _canonical_sha256(sorted(quantized_linears)),
        "kv_cache_amax_count": len(expected_kv_cache_amax),
        "kv_cache_amax_names_sha256": _canonical_sha256(sorted(expected_kv_cache_amax)),
        "kv_cache_amax_devices": dict(sorted(kv_cache_devices.items())),
        "cpu_tensors": 0,
        "disk_tensors": 0,
        "meta_tensors": 0,
    }


def modelopt_fp8_receipt_is_exact(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    export = payload.get("export")
    checkpoint_translation = payload.get("checkpoint_translation")
    remote_model = payload.get("remote_model")
    runtime = payload.get("runtime")
    fp8_backend = payload.get("fp8_per_tensor_backend")
    empty_experts = payload.get("frozen_empty_expert_compatibility")
    mamba_projection = payload.get("mamba_output_projection_compatibility")
    return bool(
        isinstance(export, dict)
        and isinstance(checkpoint_translation, dict)
        and isinstance(remote_model, dict)
        and isinstance(runtime, dict)
        and isinstance(fp8_backend, dict)
        and isinstance(empty_experts, dict)
        and isinstance(mamba_projection, dict)
        and export.get("hf_quant_config_sha256") == HF_QUANT_CONFIG_SHA256
        and export.get("model_index_sha256") == MODEL_INDEX_SHA256
        and export.get("fp8_linear_count") == FP8_LINEAR_COUNT
        and export.get("kv_cache_scale_count") == KV_CACHE_SCALE_COUNT
        and export.get("kv_cache_amax_names_sha256") == KV_CACHE_AMAX_NAMES_SHA256
        and export.get("kv_cache_scheme")
        == {"dynamic": False, "num_bits": 8, "type": "float"}
        and export.get("weight_map_entries") == MODEL_WEIGHT_MAP_ENTRIES
        and export.get("weight_shards") == MODEL_WEIGHT_SHARDS
        and export.get("disabled_patterns") == MODELOPT_IGNORE_PATTERNS
        and export.get("renamed_disabled_patterns") == MODELOPT_BACKBONE_IGNORE_PATTERNS
        and export.get("disabled_pattern_source_prefix") == "backbone."
        and export.get("disabled_pattern_target_prefix") == "model."
        and export.get("source_disabled_pattern_sha256")
        == MODELOPT_SOURCE_IGNORE_SHA256
        and export.get("disabled_pattern_sha256") == MODELOPT_TARGET_IGNORE_SHA256
        and export.get("quant_gemm") is True
        and checkpoint_translation.get("backbone_to_model")
        == BACKBONE_WEIGHT_MAP_ENTRIES
        and checkpoint_translation.get("mtp_ignored") == MTP_WEIGHT_MAP_ENTRIES
        and checkpoint_translation.get("lm_head_unchanged")
        == LM_HEAD_WEIGHT_MAP_ENTRIES
        and checkpoint_translation.get("source_prefix") == "backbone."
        and checkpoint_translation.get("target_prefix") == "model."
        and checkpoint_translation.get("mtp_policy")
        == "ignored_not_implemented_by_remote_causal_lm"
        and checkpoint_translation.get("input_scale_to_amax") == FP8_LINEAR_COUNT
        and checkpoint_translation.get("weight_scale_to_amax") == FP8_LINEAR_COUNT
        and checkpoint_translation.get("input_amax_placeholders") == FP8_LINEAR_COUNT
        and checkpoint_translation.get("weight_amax_placeholders") == FP8_LINEAR_COUNT
        and checkpoint_translation.get("enabled_fp8_module_identities")
        == FP8_LINEAR_COUNT
        and checkpoint_translation.get("enabled_fp8_module_names_sha256")
        == FP8_MODULE_NAMES_SHA256
        and checkpoint_translation.get("k_scale_to_amax") == KV_CACHE_SCALE_COUNT // 2
        and checkpoint_translation.get("v_scale_to_amax") == KV_CACHE_SCALE_COUNT // 2
        and checkpoint_translation.get("kv_cache_amax_placeholders")
        == KV_CACHE_SCALE_COUNT
        and checkpoint_translation.get("kv_cache_amax_identities")
        == KV_CACHE_SCALE_COUNT
        and checkpoint_translation.get("kv_cache_amax_names_sha256")
        == KV_CACHE_AMAX_NAMES_SHA256
        and checkpoint_translation.get("weight_amax_pre_dtype") == "torch.bfloat16"
        and checkpoint_translation.get("weight_amax_dtype") == "torch.float32"
        and checkpoint_translation.get("weight_amax_shape") == []
        and checkpoint_translation.get("input_amax_shape") == []
        and checkpoint_translation.get("kv_cache_amax_shape") == []
        and checkpoint_translation.get("scale_to_amax_multiplier")
        == FP8_SCALE_MULTIPLIER
        and remote_model.get("configuration_sha256") == REMOTE_CONFIGURATION_SHA256
        and remote_model.get("modeling_sha256") == REMOTE_MODELING_SHA256
        and str(remote_model.get("model_class", "")).endswith(".NemotronHForCausalLM")
        and runtime.get("real_quant_gemm_enabled") is True
        and runtime.get("real_fp8_linear_count") == FP8_LINEAR_COUNT
        and runtime.get("enabled_fp8_module_names_sha256") == FP8_MODULE_NAMES_SHA256
        and runtime.get("kv_cache_amax_count") == KV_CACHE_SCALE_COUNT
        and runtime.get("kv_cache_amax_names_sha256") == KV_CACHE_AMAX_NAMES_SHA256
        and isinstance(runtime.get("kv_cache_amax_devices"), dict)
        and sum(runtime.get("kv_cache_amax_devices", {}).values())
        == KV_CACHE_SCALE_COUNT
        and set(runtime.get("kv_cache_amax_devices", {})) <= {"cuda:0", "cuda:1"}
        and runtime.get("cpu_tensors") == 0
        and runtime.get("disk_tensors") == 0
        and runtime.get("meta_tensors") == 0
        and set(runtime.get("parameter_devices", {})) == {"cuda:0", "cuda:1"}
        and set(runtime.get("buffer_devices", {})) <= {"cuda:0", "cuda:1"}
        and fp8_backend.get("mode") == "modelopt-fp8-per-tensor-scaled-mm"
        and fp8_backend.get("source_sha256") == MODELOPT_FP8_BACKEND_SHA256
        and fp8_backend.get("registration_count") == 1
        and fp8_backend.get("gemm_function") == "Fp8PerTensorLinear.apply"
        and fp8_backend.get("availability_check") == "_fp8_availability_check"
        and empty_experts.get("mode")
        == "skip-mathematically-zero-frozen-empty-expert-compute"
        and empty_experts.get("moe_layers") == len(MOE_LAYER_INDICES)
        and empty_experts.get("experts_per_layer") == ROUTED_EXPERTS_PER_LAYER
        and empty_experts.get("expert_modules")
        == len(MOE_LAYER_INDICES) * ROUTED_EXPERTS_PER_LAYER
        and empty_experts.get("mixer_names_sha256") == MOE_MIXER_NAMES_SHA256
        and empty_experts.get("expert_biases") is False
        and empty_experts.get("active_expert_path") == "unchanged"
        and empty_experts.get("native_router_expert_trainables") == 0
        and mamba_projection.get("mode") == "quant-aware-projection-after-fused-ssm"
        and mamba_projection.get("mamba_layers") == len(MAMBA_LAYER_INDICES)
        and mamba_projection.get("projection_names_sha256")
        == MAMBA_OUTPUT_PROJECTION_NAMES_SHA256
        and str(mamba_projection.get("remote_module", "")).endswith(
            ".modeling_nemotron_h"
        )
        and mamba_projection.get("fused_outproj_weight") is None
        and mamba_projection.get("fused_outproj_bias") is None
        and mamba_projection.get("final_states_preserved") is True
    )


def install_modelopt_fp8_per_tensor_backend() -> dict[str, Any]:
    """Register the pinned FP8 scaled-matmul backend omitted by ModelOpt's package init."""

    from modelopt.torch.quantization.backends.gemm_registry import gemm_registry
    from modelopt.torch.quantization.backends import fp8_per_tensor_gemm as backend

    source = Path(inspect.getfile(backend)).resolve(strict=True)
    registrations = [
        entry
        for entry in getattr(gemm_registry, "_registry", ())
        if getattr(entry.get("gemm_func"), "__self__", None)
        is backend.Fp8PerTensorLinear
    ]
    if (
        sha256_file(source) != MODELOPT_FP8_BACKEND_SHA256
        or not str(source).endswith(
            "/modelopt/torch/quantization/backends/fp8_per_tensor_gemm.py"
        )
        or len(registrations) != 1
        or registrations[0].get("availability_check")
        is not backend._fp8_availability_check
    ):
        raise NemotronSuperMechanicsError("ModelOpt FP8 GEMM backend differs")
    return {
        "mode": "modelopt-fp8-per-tensor-scaled-mm",
        "source_sha256": MODELOPT_FP8_BACKEND_SHA256,
        "registration_count": 1,
        "gemm_function": "Fp8PerTensorLinear.apply",
        "availability_check": "_fp8_availability_check",
    }


def install_frozen_empty_expert_compatibility(backbone: Any) -> dict[str, Any]:
    """Skip only the pinned remote model's mathematically zero empty-expert calls.

    The upstream single-process forward calls every unselected expert on an all-zero
    tensor solely to mark distributed parameters as used.  Shohin freezes every
    native router/expert parameter, so those calls have exactly zero value and zero
    relevant gradient while forcing unsupported FP8 fallback GEMMs.
    """

    layers = getattr(getattr(backbone, "model", None), "layers", None)
    if not isinstance(layers, (list, tuple)) and type(layers).__name__ != "ModuleList":
        raise NemotronSuperMechanicsError("MoE empty-expert layer surface differs")

    observed_names: list[str] = []
    expert_modules = 0
    for index in MOE_LAYER_INDICES:
        mixer = getattr(layers[index], "mixer", None)
        experts = getattr(mixer, "experts", None)
        original = getattr(mixer, "moe", None)
        if (
            type(mixer).__name__ != "QuantNemotronHMoE"
            or not callable(original)
            or getattr(original, "_shohin_frozen_empty_expert_compatibility", False)
            or type(experts).__name__ != "ModuleList"
            or len(experts) != ROUTED_EXPERTS_PER_LAYER
        ):
            raise NemotronSuperMechanicsError("MoE empty-expert geometry differs")
        for expert in experts:
            if (
                type(expert).__name__ != "NemotronHMLP"
                or getattr(getattr(expert, "up_proj", None), "bias", object())
                is not None
                or getattr(getattr(expert, "down_proj", None), "bias", object())
                is not None
            ):
                raise NemotronSuperMechanicsError("MoE empty-expert surface differs")
        observed_names.append(f"model.layers.{index}.mixer")
        expert_modules += len(experts)

        def _moe_without_empty_expert_compute(
            self: Any,
            hidden_states: torch.Tensor,
            topk_indices: torch.Tensor,
            topk_weights: torch.Tensor,
        ) -> torch.Tensor:
            final_hidden_states = torch.zeros_like(
                hidden_states, dtype=topk_weights.dtype
            )
            expert_mask = torch.nn.functional.one_hot(
                topk_indices, num_classes=len(self.experts)
            ).permute(2, 0, 1)
            for expert_idx, expert in enumerate(self.experts):
                token_indices, weight_indices = torch.where(expert_mask[expert_idx])
                if token_indices.numel() == 0:
                    continue
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)
            return final_hidden_states.type(hidden_states.dtype)

        _moe_without_empty_expert_compute._shohin_frozen_empty_expert_compatibility = (  # type: ignore[attr-defined]
            True
        )
        mixer.moe = MethodType(_moe_without_empty_expert_compute, mixer)

    if tuple(observed_names) != MOE_MIXER_NAMES:
        raise NemotronSuperMechanicsError("MoE empty-expert identities differ")
    return {
        "mode": "skip-mathematically-zero-frozen-empty-expert-compute",
        "moe_layers": len(MOE_LAYER_INDICES),
        "experts_per_layer": ROUTED_EXPERTS_PER_LAYER,
        "expert_modules": expert_modules,
        "mixer_names_sha256": MOE_MIXER_NAMES_SHA256,
        "expert_biases": False,
        "active_expert_path": "unchanged",
        "native_router_expert_trainables": 0,
    }


def install_modelopt_mamba_output_projection_compatibility(
    backbone: Any,
) -> dict[str, Any]:
    """Keep fused SSM execution while routing FP8 out-projections through ModelOpt."""

    layers = getattr(getattr(backbone, "model", None), "layers", None)
    if not isinstance(layers, (list, tuple)) and type(layers).__name__ != "ModuleList":
        raise NemotronSuperMechanicsError("Mamba projection layer surface differs")
    projections: dict[int, Any] = {}
    module_names: set[str] = set()
    observed_names: list[str] = []
    for index in MAMBA_LAYER_INDICES:
        layer = layers[index]
        mixer = getattr(layer, "mixer", None)
        projection = getattr(mixer, "out_proj", None)
        weight = getattr(projection, "weight", None)
        if (
            type(mixer).__name__ != "NemotronHMamba2Mixer"
            or not callable(projection)
            or not isinstance(weight, torch.Tensor)
            or id(weight) in projections
        ):
            raise NemotronSuperMechanicsError("Mamba output projection differs")
        projections[id(weight)] = projection
        module_names.add(type(mixer).__module__)
        observed_names.append(f"model.layers.{index}.mixer.out_proj")
    if (
        tuple(observed_names) != MAMBA_OUTPUT_PROJECTION_NAMES
        or len(projections) != len(MAMBA_LAYER_INDICES)
        or len(module_names) != 1
    ):
        raise NemotronSuperMechanicsError("Mamba output projection geometry differs")
    remote_module = sys.modules.get(next(iter(module_names)))
    original = getattr(remote_module, "mamba_split_conv1d_scan_combined", None)
    if (
        remote_module is None
        or not callable(original)
        or getattr(original, "_shohin_modelopt_projection_compatibility", False)
    ):
        raise NemotronSuperMechanicsError("Mamba fused SSM implementation differs")

    def _compatible_fused_ssm(*args: Any, **kwargs: Any) -> Any:
        weight = kwargs.get("outproj_weight")
        projection = projections.get(id(weight))
        if projection is None:
            return original(*args, **kwargs)
        if (
            "outproj_weight" not in kwargs
            or "outproj_bias" not in kwargs
            or kwargs["outproj_bias"] is not getattr(projection, "bias", None)
            or kwargs.get("return_final_states") is not True
        ):
            raise NemotronSuperMechanicsError(
                "Mamba fused output projection call differs"
            )
        fused_kwargs = dict(kwargs)
        fused_kwargs["outproj_weight"] = None
        fused_kwargs["outproj_bias"] = None
        result = original(*args, **fused_kwargs)
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or not isinstance(result[0], torch.Tensor)
        ):
            raise NemotronSuperMechanicsError("Mamba fused SSM result differs")
        return projection(result[0]), result[1]

    _compatible_fused_ssm._shohin_modelopt_projection_compatibility = True  # type: ignore[attr-defined]
    remote_module.mamba_split_conv1d_scan_combined = _compatible_fused_ssm
    return {
        "mode": "quant-aware-projection-after-fused-ssm",
        "mamba_layers": len(MAMBA_LAYER_INDICES),
        "projection_names_sha256": MAMBA_OUTPUT_PROJECTION_NAMES_SHA256,
        "remote_module": next(iter(module_names)),
        "fused_outproj_weight": None,
        "fused_outproj_bias": None,
        "final_states_preserved": True,
    }


def load_modelopt_fp8_backbone(model_root: Path) -> tuple[Any, dict[str, Any]]:
    """Load the immutable ModelOpt FP8 export across exactly two local H100s."""

    fp8_backend_receipt = install_modelopt_fp8_per_tensor_backend()
    quant_cfg, export_receipt = _modelopt_fp8_quantization_config(model_root)
    expected_fp8_modules = _expected_fp8_module_names(model_root)
    expected_kv_cache_amax = _expected_kv_cache_amax_names(model_root)
    from modelopt.torch.quantization.plugins.accelerate import init_quantized_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=True,
    )
    observed_quantization = getattr(config, "quantization_config", None)
    expected_quantization = json.loads(
        (model_root / "config.json").read_text(encoding="utf-8")
    )["quantization_config"]
    if observed_quantization != expected_quantization:
        raise NemotronSuperMechanicsError("Transformers ModelOpt configuration differs")
    if (
        sha256_file(model_root / "configuration_nemotron_h.py")
        != REMOTE_CONFIGURATION_SHA256
        or sha256_file(model_root / "modeling_nemotron_h.py") != REMOTE_MODELING_SHA256
        or config.auto_map.get("AutoModelForCausalLM")
        != "modeling_nemotron_h.NemotronHForCausalLM"
    ):
        raise NemotronSuperMechanicsError("pinned remote model implementation differs")
    model_class = get_class_from_dynamic_module(
        config.auto_map["AutoModelForCausalLM"],
        model_root,
        local_files_only=True,
    )
    resolved_modeling = Path(inspect.getfile(model_class)).resolve(strict=True)
    if sha256_file(resolved_modeling) != REMOTE_MODELING_SHA256:
        raise NemotronSuperMechanicsError("loaded remote model implementation differs")
    AutoModelForCausalLM.register(type(config), model_class, exist_ok=True)
    delattr(config, "quantization_config")
    config.torch_dtype = torch.bfloat16
    with _translate_export_checkpoint_keys(
        expected_fp8_modules, expected_kv_cache_amax
    ) as translation_counts:
        with init_quantized_weights(
            quant_cfg, gpu_mem_percentage=0.95, quant_gemm=True
        ):
            backbone = AutoModelForCausalLM.from_pretrained(
                model_root,
                config=config,
                trust_remote_code=True,
                strict=True,
            )
    translation_receipt = _checkpoint_translation_receipt(translation_counts)
    runtime_receipt = _modelopt_fp8_runtime_receipt(
        backbone, expected_fp8_modules, expected_kv_cache_amax
    )
    empty_expert_receipt = install_frozen_empty_expert_compatibility(backbone)
    mamba_projection_receipt = install_modelopt_mamba_output_projection_compatibility(
        backbone
    )
    return backbone, {
        "export": export_receipt,
        "checkpoint_translation": translation_receipt,
        "remote_model": {
            "configuration_sha256": REMOTE_CONFIGURATION_SHA256,
            "modeling_sha256": REMOTE_MODELING_SHA256,
            "model_class": f"{model_class.__module__}.{model_class.__name__}",
        },
        "runtime": runtime_receipt,
        "fp8_per_tensor_backend": fp8_backend_receipt,
        "frozen_empty_expert_compatibility": empty_expert_receipt,
        "mamba_output_projection_compatibility": mamba_projection_receipt,
    }


def install_triton_allocator_compatibility() -> dict[str, Any]:
    """Bridge Mamba-SSM allocator registration on the pinned Triton 3.2."""

    import triton

    observed = importlib.metadata.version("triton")
    if observed != TRITON_VERSION:
        raise NemotronSuperMechanicsError("Triton version differs")
    if hasattr(triton, "set_allocator"):
        return {"triton_version": observed, "mode": "native-set-allocator"}

    def _set_allocator_compatibility(allocator: Any) -> None:
        if not callable(allocator):
            raise TypeError("allocator must be callable")

    triton.set_allocator = _set_allocator_compatibility
    return {"triton_version": observed, "mode": "triton-3.2-internal-descriptor"}


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
    diagnostics = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        finite = bool(gradient is not None and torch.isfinite(gradient).all())
        norm = float(gradient.float().norm().detach().cpu()) if finite else None
        rows.append({"name": name, "finite": finite, "norm": norm})
        diagnostics.append(
            {
                "name": name,
                "present": gradient is not None,
                "finite": finite,
                "nan_values": (
                    int(torch.isnan(gradient).sum().detach().cpu())
                    if gradient is not None
                    else None
                ),
                "positive_infinite_values": (
                    int(torch.isposinf(gradient).sum().detach().cpu())
                    if gradient is not None
                    else None
                ),
                "negative_infinite_values": (
                    int(torch.isneginf(gradient).sum().detach().cpu())
                    if gradient is not None
                    else None
                ),
                "norm": norm,
            }
        )
    if (
        len(rows) != 32
        or not all(row["finite"] for row in rows)
        or not any(float(row["norm"] or 0.0) > 0.0 for row in rows)
    ):
        raise NemotronSuperMechanicsError(
            "Shohin gradient receipt differs: "
            + json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))
        )
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    return {
        "parameters": len(rows),
        "nonzero_gradients": sum(float(row["norm"] or 0.0) > 0.0 for row in rows),
        "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _mechanics_next_token_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    if (
        logits.ndim != 3
        or labels.ndim != 2
        or logits.shape[:2] != labels.shape
        or labels.shape[1] < 2
        or labels.dtype != torch.long
    ):
        raise NemotronSuperMechanicsError("mechanics loss geometry differs")
    shifted_labels = labels[:, 1:].to(logits.device)
    supervised_tokens = int((shifted_labels != -100).sum().detach().cpu())
    if supervised_tokens < 1:
        raise NemotronSuperMechanicsError("mechanics supervised tokens differ")
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )
    if not bool(torch.isfinite(loss)):
        raise NemotronSuperMechanicsError("mechanics loss is nonfinite")
    return loss


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

    triton_allocator = install_triton_allocator_compatibility()
    import mamba_ssm
    import modelopt
    from transformers import AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=True,
    )
    backbone, modelopt_fp8 = load_modelopt_fp8_backbone(model_root)
    device_map = getattr(backbone, "hf_device_map", None)
    model = NemotronSuperRevisionModel(backbone, modelopt_quantized=True)
    if model.trainable_parameter_count() != TRAINABLE_PARAMETERS_PER_ROLE:
        raise NemotronSuperMechanicsError("trainable surface differs")
    initial_state = model.trainable_state()
    initial_sha256 = _state_sha256(initial_state)

    prompt_ids = tokenizer.encode("Shohin mechanics:", add_special_tokens=False)
    response_ids = tokenizer.encode(" verified.", add_special_tokens=False)
    token_ids = prompt_ids + response_ids
    label_ids = [-100] * len(prompt_ids) + response_ids
    if not prompt_ids or not response_ids or len(token_ids) < 2 or len(token_ids) > 16:
        raise NemotronSuperMechanicsError("synthetic mechanics tokenization differs")
    input_device = backbone.model.embeddings.weight.device
    input_ids = torch.tensor([token_ids], device=input_device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([label_ids], device=input_device, dtype=torch.long)
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=TRAINING_LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_payload = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = output_payload.logits
        if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
            raise NemotronSuperMechanicsError("full-model forward geometry differs")
        loss = _mechanics_next_token_loss(logits, labels)
        scaled_loss = loss / TRAINING_GRADIENT_ACCUMULATION
    scaled_loss.backward()
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
        "triton_allocator_compatibility": triton_allocator,
        "modelopt_fp8": modelopt_fp8,
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
        "training_objective_receipt": {
            "objective": "response_only_next_token_cross_entropy",
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
            "ignore_index": -100,
            "gradient_accumulation_scale": TRAINING_GRADIENT_ACCUMULATION,
            "learning_rate": TRAINING_LEARNING_RATE,
            "autocast_dtype": "torch.bfloat16",
        },
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
