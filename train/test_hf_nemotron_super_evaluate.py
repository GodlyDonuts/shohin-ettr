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
            "weight_map_entries": mechanics.MODEL_WEIGHT_MAP_ENTRIES,
            "weight_shards": mechanics.MODEL_WEIGHT_SHARDS,
            "disabled_patterns": mechanics.MODELOPT_IGNORE_PATTERNS,
            "quant_gemm": True,
        },
        "runtime": {
            "real_quant_gemm_enabled": True,
            "real_fp8_linear_count": mechanics.FP8_LINEAR_COUNT,
            "cpu_tensors": 0,
            "disk_tensors": 0,
            "meta_tensors": 0,
            "parameter_devices": {"cuda:0": 1, "cuda:1": 1},
            "buffer_devices": {"cuda:0": 1},
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
