from copy import deepcopy
import json
from pathlib import Path

import pytest

from capability_floor_interface import (
    CapabilityFloorInterfaceError,
    build_interface_contract,
    validate_interface_contract,
    validate_pinned_config,
)


def _qwen_config() -> dict[str, object]:
    return {
        "text_config": {
            "hidden_size": 1024,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "num_attention_heads": 8,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "vocab_size": 248320,
        }
    }


def _smollm3_config() -> dict[str, object]:
    return {
        "hidden_size": 2048,
        "max_position_embeddings": 65536,
        "model_type": "smollm3",
        "num_attention_heads": 16,
        "num_hidden_layers": 36,
        "num_key_value_heads": 4,
        "vocab_size": 128256,
    }


def test_interface_is_exact_and_launch_blocked() -> None:
    payload = build_interface_contract()
    validate_interface_contract(payload)
    assert payload["launch_authorized"] is False
    assert payload["optimizer"]["seed_pairs"] == [[31, 11], [32, 12]]
    assert payload["optimizer"]["semantic_microbatches_per_update"] == 4
    assert payload["optimizer"]["stratification_receipt_required"] is True
    assert payload["interface_sufficiency"]["strict_threshold"] == 0.95
    assert (
        payload["interface_sufficiency"]["assessor_features_available_at_inference"]
        is False
    )
    assert payload["input_envelope"]["chat_template"] == "forbidden"
    assert payload["data"]["token_truncation"].startswith("forbidden")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chat_template", "native"),
        ("semantic_bytes_identical_before-tokenization", False),
    ],
)
def test_interface_rejects_prompt_drift(field: str, value: object) -> None:
    payload = deepcopy(build_interface_contract())
    payload["input_envelope"][field] = value
    with pytest.raises(CapabilityFloorInterfaceError, match="protocol differs"):
        validate_interface_contract(payload)


def test_interface_rejects_mean_seed_promotion() -> None:
    payload = deepcopy(build_interface_contract())
    payload["evaluation"]["aggregation"] = "mean-across-seeds"
    with pytest.raises(CapabilityFloorInterfaceError, match="protocol differs"):
        validate_interface_contract(payload)


def test_accessible_pinned_configs_are_exact() -> None:
    validate_pinned_config("qwen3.5-0.8b-text-backbone", _qwen_config())
    validate_pinned_config("smollm3-3b", _smollm3_config())


def test_qwen_text_submodel_geometry_cannot_drift() -> None:
    config = _qwen_config()
    config["text_config"]["hidden_size"] = 1000
    with pytest.raises(CapabilityFloorInterfaceError, match="Qwen"):
        validate_pinned_config("qwen3.5-0.8b-text-backbone", config)


def test_mobilellm_cannot_launch_from_api_metadata_only() -> None:
    with pytest.raises(CapabilityFloorInterfaceError, match="gated config"):
        validate_pinned_config(
            "facebook-mobilellm-r1-360m",
            {"model_type": "llama4_text"},
        )


def test_checked_in_interface_receipt_matches_builder() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/r12/ettr_capability_floor_interface_v1.json"
    )
    assert json.loads(path.read_text(encoding="ascii")) == build_interface_contract()
