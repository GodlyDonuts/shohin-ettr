#!/usr/bin/env python3
"""Freeze the cross-backbone interface before capability-floor GPU work."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

from capability_floor_campaign import (
    ETTR_RELEASE_SHA256,
    PROTECTED_SHOHIN_SHA256,
    build_preregistration,
    validate_preregistration,
)
from train_ettr_component_island import _canonical_bytes, _write_no_replace


SCHEMA = "shohin-ettr-capability-floor-interface-v1"
QWEN_CONFIG_SHA256 = (
    "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
)
SMOLLM3_CONFIG_SHA256 = (
    "c72b1031274ff4626e434d0019e88e95a767460135db9ee492eb80652b786af1"
)


class CapabilityFloorInterfaceError(RuntimeError):
    """The cross-backbone interface is incomplete or has drifted."""


def _backbone_contracts() -> list[dict[str, object]]:
    return [
        {
            "candidate": "protected-shohin-125m-step300k",
            "config_receipt": "repository-model-config-and-checkpoint-contract",
            "config_sha256": None,
            "context_limit": 2048,
            "hidden_width": 576,
            "model_type": "shohin-gpt",
            "num_hidden_layers": 30,
            "source_access": "available",
            "source_revision": PROTECTED_SHOHIN_SHA256,
        },
        {
            "candidate": "facebook-mobilellm-r1-360m",
            "config_receipt": "required-after-manual-license-acceptance",
            "config_sha256": None,
            "context_limit": None,
            "hidden_width": None,
            "model_type": "llama4_text",
            "num_hidden_layers": None,
            "source_access": "blocked-manual-gated-license",
            "source_revision": "ac72186c210d932d27eb63c1bd2d103d82ca2ed1",
        },
        {
            "candidate": "qwen3.5-0.8b-text-backbone",
            "config_receipt": "pinned-official-config-json",
            "config_sha256": QWEN_CONFIG_SHA256,
            "context_limit": 262144,
            "hidden_width": 1024,
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 24,
            "source_access": "available",
            "source_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        },
        {
            "candidate": "smollm3-3b",
            "config_receipt": "pinned-official-config-json",
            "config_sha256": SMOLLM3_CONFIG_SHA256,
            "context_limit": 65536,
            "hidden_width": 2048,
            "model_type": "smollm3",
            "num_hidden_layers": 36,
            "source_access": "available",
            "source_revision": "a07cc9a04f16550a088caea529712d1d335b0ac1",
        },
    ]


def build_interface_contract() -> dict[str, object]:
    preregistration = build_preregistration()
    validate_preregistration(preregistration)
    return {
        "adapter": {
            "backbone_mode": "eval-frozen-no-gradient",
            "common_ettr_width": 512,
            "feature": "final-post-norm-hidden-state-for-every-source-token",
            "normalization": "learned-projection-then-rmsnorm",
            "projection": "candidate-specific-bias-free-linear-counted-as-treatment",
            "role_pooling": "mean-of-token-residuals-overlapping-canonical-ascii-span",
            "special_token_features": False,
        },
        "backbones": _backbone_contracts(),
        "comparison_accounting": {
            "backbone_inference_flops_logged_separately": True,
            "cache_policy": "same-within-backbone-for-ettr-and-dense",
            "capacity_floor_is_not_cross-backbone-compute-matched": True,
            "dense_control_parameter_tolerance": 0.01,
            "dense_control_sidecar_training_flop_tolerance": 0.05,
            "end_to_end_inference_flops_reported": True,
        },
        "data": {
            "canonical_source_encoding": "ascii",
            "development_episodes": 5000,
            "episode_release_sha256": ETTR_RELEASE_SHA256,
            "rectangle_atomic_split": True,
            "semantic_cohort": "four-backbone-tokenization-intersection",
            "stream_rows": 180000,
            "stream_sha256": (
                "8f205de26b4c6ad4aa10d85d7765a3c0640255b6a1972c3941ee236fbe020f87"
            ),
            "token_truncation": "forbidden-reject-row-from-all-candidates",
            "train_episodes": 40000,
        },
        "evaluation": {
            "aggregation": "minimum-across-seeds-not-mean",
            "confirmation_open_condition": "both-development-seeds-pass-all-gates",
            "negative_controls": [
                "binding-deranged",
                "state-reset",
                "query-only",
                "shuffled-label",
            ],
            "source_deleted": True,
            "strict_component_threshold": 0.95,
            "strict_composition_threshold": 0.90,
        },
        "input_envelope": {
            "add_native_required_bos_only": True,
            "chat_template": "forbidden",
            "padding": "right-padding-masked-out",
            "renderer": "shared-canonical-ettr-ascii-v1",
            "semantic_bytes_identical_before-tokenization": True,
            "token_offsets_required": True,
        },
        "launch_authorized": False,
        "launch_blockers": [
            "final-v15-v21-mechanism-hash-unset",
            "mobilellm-r1-manual-license-not-accepted",
            "four-tokenizer-semantic-intersection-not-receipted",
            "dense-control-parameter-and-flop-receipts-not-built",
        ],
        "optimizer": {
            "betas": [0.9, 0.95],
            "component_updates_per_seed": 2000,
            "composition_updates_per_seed": 5000,
            "gradient_clip": 1.0,
            "learning_rate": 0.0003,
            "optimizer": "fused-adamw",
            "semantic_batch_size": 16,
            "seed_pairs": [[31, 11], [32, 12]],
            "weight_decay": 0.01,
        },
        "ownership_boundary": {
            "command_recurrence": "tied-model-owned-state-core",
            "query_available_during_world_or_command": False,
            "query_readout": "late-only-after-adaptive-stop",
            "teacher_program_available_during_autonomous_evaluation": False,
            "teacher_state_available_during_autonomous_evaluation": False,
        },
        "preregistration_schema": preregistration["schema"],
        "schema": SCHEMA,
        "status": "interface-frozen-launch-blocked",
    }


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CapabilityFloorInterfaceError(f"{name} differs")
    return value


def validate_interface_contract(payload: Mapping[str, object]) -> None:
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "interface-frozen-launch-blocked"
        or payload.get("launch_authorized") is not False
    ):
        raise CapabilityFloorInterfaceError("capability-floor interface custody differs")
    backbones = payload.get("backbones")
    if not isinstance(backbones, list) or backbones != _backbone_contracts():
        raise CapabilityFloorInterfaceError("capability-floor backbone interface differs")
    adapter = _require_mapping(payload.get("adapter"), "adapter")
    envelope = _require_mapping(payload.get("input_envelope"), "input envelope")
    data = _require_mapping(payload.get("data"), "data")
    optimizer = _require_mapping(payload.get("optimizer"), "optimizer")
    evaluation = _require_mapping(payload.get("evaluation"), "evaluation")
    boundary = _require_mapping(payload.get("ownership_boundary"), "ownership")
    if (
        adapter.get("common_ettr_width") != 512
        or adapter.get("backbone_mode") != "eval-frozen-no-gradient"
        or envelope.get("chat_template") != "forbidden"
        or envelope.get("semantic_bytes_identical_before-tokenization") is not True
        or data.get("token_truncation")
        != "forbidden-reject-row-from-all-candidates"
        or data.get("train_episodes") != 40000
        or data.get("development_episodes") != 5000
        or optimizer.get("seed_pairs") != [[31, 11], [32, 12]]
        or optimizer.get("semantic_batch_size") != 16
        or evaluation.get("aggregation") != "minimum-across-seeds-not-mean"
        or boundary.get("query_available_during_world_or_command") is not False
    ):
        raise CapabilityFloorInterfaceError("capability-floor protocol differs")
    blockers = payload.get("launch_blockers")
    if (
        not isinstance(blockers, list)
        or "final-v15-v21-mechanism-hash-unset" not in blockers
        or "mobilellm-r1-manual-license-not-accepted" not in blockers
    ):
        raise CapabilityFloorInterfaceError("capability-floor blockers differ")
    if dict(payload) != build_interface_contract():
        raise CapabilityFloorInterfaceError("capability-floor interface differs")


def validate_pinned_config(candidate: str, config: Mapping[str, object]) -> None:
    """Validate accessible official configs without accepting approximate aliases."""

    if candidate == "qwen3.5-0.8b-text-backbone":
        text_config = _require_mapping(config.get("text_config"), "Qwen text config")
        expected = {
            "hidden_size": 1024,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_5_text",
            "num_attention_heads": 8,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "vocab_size": 248320,
        }
        if any(text_config.get(key) != value for key, value in expected.items()):
            raise CapabilityFloorInterfaceError("Qwen text configuration differs")
        return
    if candidate == "smollm3-3b":
        expected = {
            "hidden_size": 2048,
            "max_position_embeddings": 65536,
            "model_type": "smollm3",
            "num_attention_heads": 16,
            "num_hidden_layers": 36,
            "num_key_value_heads": 4,
            "vocab_size": 128256,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise CapabilityFloorInterfaceError("SmolLM3 configuration differs")
        return
    if candidate == "facebook-mobilellm-r1-360m":
        raise CapabilityFloorInterfaceError(
            "MobileLLM-R1 requires exact gated config admission"
        )
    raise CapabilityFloorInterfaceError("unsupported external backbone config")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_interface_contract()
    validate_interface_contract(payload)
    _write_no_replace(args.output, _canonical_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
