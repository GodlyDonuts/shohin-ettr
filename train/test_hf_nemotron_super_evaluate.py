"""CPU tests for the fixed-draft Nemotron Super screen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import hf_nemotron_super_evaluate as evaluation
import hf_nemotron_super_mechanics as mechanics


def _modelopt_fp8() -> dict[str, object]:
    return {
        "export": {
            "hf_quant_config_sha256": mechanics.HF_QUANT_CONFIG_SHA256,
            "model_index_sha256": mechanics.MODEL_INDEX_SHA256,
            "fp8_linear_count": mechanics.FP8_LINEAR_COUNT,
            "kv_cache_scale_count": mechanics.KV_CACHE_SCALE_COUNT,
            "kv_cache_amax_names_sha256": mechanics.KV_CACHE_AMAX_NAMES_SHA256,
            "kv_cache_scheme": {
                "dynamic": False,
                "num_bits": 8,
                "type": "float",
            },
            "weight_map_entries": mechanics.MODEL_WEIGHT_MAP_ENTRIES,
            "weight_shards": mechanics.MODEL_WEIGHT_SHARDS,
            "disabled_patterns": mechanics.MODELOPT_IGNORE_PATTERNS,
            "renamed_disabled_patterns": mechanics.MODELOPT_BACKBONE_IGNORE_PATTERNS,
            "disabled_pattern_source_prefix": "backbone.",
            "disabled_pattern_target_prefix": "model.",
            "source_disabled_pattern_sha256": mechanics.MODELOPT_SOURCE_IGNORE_SHA256,
            "disabled_pattern_sha256": mechanics.MODELOPT_TARGET_IGNORE_SHA256,
            "quant_gemm": True,
        },
        "remote_model": {
            "configuration_sha256": mechanics.REMOTE_CONFIGURATION_SHA256,
            "modeling_sha256": mechanics.REMOTE_MODELING_SHA256,
            "model_class": "frozen.NemotronHForCausalLM",
        },
        "checkpoint_translation": {
            "backbone_to_model": mechanics.BACKBONE_WEIGHT_MAP_ENTRIES,
            "mtp_ignored": mechanics.MTP_WEIGHT_MAP_ENTRIES,
            "lm_head_unchanged": mechanics.LM_HEAD_WEIGHT_MAP_ENTRIES,
            "source_prefix": "backbone.",
            "target_prefix": "model.",
            "mtp_policy": "ignored_not_implemented_by_remote_causal_lm",
            "input_scale_to_amax": mechanics.FP8_LINEAR_COUNT,
            "weight_scale_to_amax": mechanics.FP8_LINEAR_COUNT,
            "input_amax_placeholders": mechanics.FP8_LINEAR_COUNT,
            "weight_amax_placeholders": mechanics.FP8_LINEAR_COUNT,
            "enabled_fp8_module_identities": mechanics.FP8_LINEAR_COUNT,
            "enabled_fp8_module_names_sha256": mechanics.FP8_MODULE_NAMES_SHA256,
            "k_scale_to_amax": mechanics.KV_CACHE_SCALE_COUNT // 2,
            "v_scale_to_amax": mechanics.KV_CACHE_SCALE_COUNT // 2,
            "kv_cache_amax_placeholders": mechanics.KV_CACHE_SCALE_COUNT,
            "kv_cache_amax_identities": mechanics.KV_CACHE_SCALE_COUNT,
            "kv_cache_amax_names_sha256": mechanics.KV_CACHE_AMAX_NAMES_SHA256,
            "weight_amax_pre_dtype": "torch.bfloat16",
            "weight_amax_dtype": "torch.float32",
            "weight_amax_shape": [],
            "input_amax_shape": [],
            "kv_cache_amax_shape": [],
            "scale_to_amax_multiplier": mechanics.FP8_SCALE_MULTIPLIER,
        },
        "runtime": {
            "real_quant_gemm_enabled": True,
            "real_fp8_linear_count": mechanics.FP8_LINEAR_COUNT,
            "enabled_fp8_module_names_sha256": mechanics.FP8_MODULE_NAMES_SHA256,
            "kv_cache_amax_count": mechanics.KV_CACHE_SCALE_COUNT,
            "kv_cache_amax_names_sha256": mechanics.KV_CACHE_AMAX_NAMES_SHA256,
            "kv_cache_amax_devices": {
                "cuda:0": mechanics.KV_CACHE_SCALE_COUNT // 2,
                "cuda:1": mechanics.KV_CACHE_SCALE_COUNT // 2,
            },
            "cpu_tensors": 0,
            "disk_tensors": 0,
            "meta_tensors": 0,
            "parameter_devices": {"cuda:0": 1, "cuda:1": 1},
            "buffer_devices": {"cuda:0": 1},
        },
        "fp8_per_tensor_backend": {
            "mode": "modelopt-fp8-per-tensor-scaled-mm",
            "source_sha256": mechanics.MODELOPT_FP8_BACKEND_SHA256,
            "registration_count": 1,
            "gemm_function": "Fp8PerTensorLinear.apply",
            "availability_check": "_fp8_availability_check",
        },
        "frozen_empty_expert_compatibility": {
            "mode": "skip-mathematically-zero-frozen-empty-expert-compute",
            "moe_layers": len(mechanics.MOE_LAYER_INDICES),
            "experts_per_layer": mechanics.ROUTED_EXPERTS_PER_LAYER,
            "expert_modules": len(mechanics.MOE_LAYER_INDICES)
            * mechanics.ROUTED_EXPERTS_PER_LAYER,
            "mixer_names_sha256": mechanics.MOE_MIXER_NAMES_SHA256,
            "expert_biases": False,
            "active_expert_path": "unchanged",
            "native_router_expert_trainables": 0,
        },
        "mamba_output_projection_compatibility": {
            "mode": "quant-aware-projection-after-fused-ssm",
            "mamba_layers": len(mechanics.MAMBA_LAYER_INDICES),
            "projection_names_sha256": mechanics.MAMBA_OUTPUT_PROJECTION_NAMES_SHA256,
            "remote_module": "frozen.modeling_nemotron_h",
            "fused_outproj_weight": None,
            "fused_outproj_bias": None,
            "final_states_preserved": True,
        },
    }


def _mechanics() -> dict[str, object]:
    return {
        "schema": evaluation.MECHANICS_SCHEMA,
        "status": "pass",
        "model_revision": evaluation.MODEL_REVISION,
        "score_rows_read": 0,
        "benchmark_rows_read": 0,
        "trainable_parameters": evaluation.TRAINABLE_PARAMETERS_PER_ROLE,
        "native_router_expert_trainables": 0,
        "serialization_restore_exact": True,
        "devices": [{"index": 0}, {"index": 1}],
        "training_objective_receipt": {
            "objective": "response_only_next_token_cross_entropy",
            "prompt_tokens": 3,
            "response_tokens": 2,
            "ignore_index": -100,
            "gradient_accumulation_scale": mechanics.TRAINING_GRADIENT_ACCUMULATION,
            "learning_rate": mechanics.TRAINING_LEARNING_RATE,
            "autocast_dtype": "torch.bfloat16",
        },
        "modelopt_fp8": _modelopt_fp8(),
    }


def test_mechanics_report_is_score_free_and_exact(tmp_path: Path) -> None:
    path = tmp_path / "mechanics.json"
    path.write_text(json.dumps(_mechanics()))
    assert evaluation.validate_mechanics_report(path)["status"] == "pass"
    payload = _mechanics()
    payload["benchmark_rows_read"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(evaluation.NemotronSuperEvaluationError):
        evaluation.validate_mechanics_report(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("objective", "next_token_cross_entropy"),
        ("prompt_tokens", 0),
        ("response_tokens", True),
        ("ignore_index", 0),
        ("ignore_index", -100.0),
        ("gradient_accumulation_scale", 1),
        ("gradient_accumulation_scale", 8.0),
        ("learning_rate", 1e-5),
        ("learning_rate", True),
        ("autocast_dtype", "torch.float32"),
    ],
)
def test_mechanics_report_binds_training_objective(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _mechanics()
    receipt = payload["training_objective_receipt"]
    assert isinstance(receipt, dict)
    receipt[field] = value
    path = tmp_path / "mechanics.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(evaluation.NemotronSuperEvaluationError):
        evaluation.validate_mechanics_report(path)


class _Model:
    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def named_parameters(self):
        return [("block.adapter_a.weight", self.parameter)]

    def trainable_parameter_name_sha256(self) -> str:
        import hashlib

        return hashlib.sha256(b"block.adapter_a.weight").hexdigest()

    def trainable_state_sha256(self) -> str:
        return evaluation._state_sha256(
            {"block.adapter_a.weight": self.parameter.detach()}
        )


def test_revision_checkpoint_restore_binds_schedule_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _Model()
    state = {"block.adapter_a.weight": torch.ones(2, dtype=torch.float32)}
    monkeypatch.setattr(evaluation, "TRAINABLE_PARAMETERS_PER_ROLE", 2)
    metadata = {
        "schema": evaluation.TRAINING_SCHEMA,
        "model_revision": evaluation.MODEL_REVISION,
        "data_sha256": evaluation.DATA_SHA256,
        "updates": evaluation.UPDATES,
        "gradient_accumulation": evaluation.GRADIENT_ACCUMULATION,
        "learning_rate": evaluation.LEARNING_RATE,
        "max_sequence_length": evaluation.MAX_SEQUENCE_LENGTH,
        "seed": evaluation.TRAINING_SEED,
        "trainable_parameters": evaluation.TRAINABLE_PARAMETERS_PER_ROLE,
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "native_router_expert_trainables": 0,
        "optimizer_state_serialized": False,
        "checkpoint_trainable_only": True,
        "final_trainable_state_sha256": evaluation._state_sha256(state),
        "modelopt_fp8": _modelopt_fp8(),
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema": evaluation.CHECKPOINT_SCHEMA,
            "update": evaluation.UPDATES,
            "trainable_state": state,
            "metadata": metadata,
        },
        path,
    )
    restored = evaluation.load_revision_checkpoint(path, model)
    assert restored == metadata
    assert torch.equal(model.parameter, torch.ones(2))
    metadata["updates"] -= 1
    torch.save(
        {
            "schema": evaluation.CHECKPOINT_SCHEMA,
            "update": evaluation.UPDATES,
            "trainable_state": state,
            "metadata": metadata,
        },
        path,
    )
    with pytest.raises(evaluation.NemotronSuperEvaluationError):
        evaluation.load_revision_checkpoint(path, model)


def test_all_arms_share_the_training_prompt_envelope() -> None:
    source = Path(evaluation.__file__).read_text()
    assert "_render_prompt(tokenizer, question, True, False)" in source
    assert "_render_prompt(tokenizer, question, False, False)" not in source
