#!/usr/bin/env python3
"""Machine-readable preregistration for the frozen-backbone capability floor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

from capability_floor_trajectory import (
    MECHANISM_SCHEMA,
    mechanism_architecture_sha256,
    mechanism_source_sha256,
)
from train_ettr_component_island import _canonical_bytes, _write_no_replace


SCHEMA = "shohin-ettr-capability-floor-preregistration-v1"
PROTECTED_SHOHIN_SHA256 = (
    "211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6"
)
ETTR_RELEASE_SHA256 = (
    "8c6d7d80603e29e92f14027929ae4ef7e848094a44a154ef37b2bcbf726d4462"
)


class CapabilityFloorContractError(RuntimeError):
    """The capability-floor preregistration is incomplete or inconsistent."""


def build_preregistration() -> dict[str, object]:
    """Return the frozen experiment design; it deliberately cannot launch yet."""

    return {
        "backbones": [
            {
                "candidate": "protected-shohin-125m-step300k",
                "frozen": True,
                "license": "private-research-artifact",
                "parameter_class": "125m",
                "source": "local-protected-checkpoint",
                "source_revision": PROTECTED_SHOHIN_SHA256,
                "training_stage": "raw-pretrain",
            },
            {
                "candidate": "facebook-mobilellm-r1-360m",
                "frozen": True,
                "license": "fair-noncommercial-research",
                "parameter_class": "360m",
                "source": "facebook/MobileLLM-R1-360M",
                "source_revision": "ac72186c210d932d27eb63c1bd2d103d82ca2ed1",
                "training_stage": "post-trained-reasoner",
            },
            {
                "candidate": "qwen3.5-0.8b-text-backbone",
                "frozen": True,
                "license": "apache-2.0",
                "parameter_class": "0.8b",
                "source": "Qwen/Qwen3.5-0.8B",
                "source_revision": "2fc06364715b967f1860aea9cf38778875588b17",
                "training_stage": "post-trained-multimodal-reasoner-text-path",
            },
            {
                "candidate": "smollm3-3b",
                "frozen": True,
                "license": "apache-2.0",
                "parameter_class": "3b",
                "source": "HuggingFaceTB/SmolLM3-3B",
                "source_revision": "a07cc9a04f16550a088caea529712d1d335b0ac1",
                "training_stage": "post-trained-reasoner",
            },
        ],
        "component_gates": {
            "oracle_program_executor_exact": 0.95,
            "oracle_state_query_reader_strict_command": 0.95,
            "oracle_state_query_reader_strict_world": 0.95,
            "world_compiler_effect_binding_exact": 0.95,
        },
        "composition_gates": {
            "autonomous_strict_command": 0.90,
            "autonomous_strict_world": 0.90,
            "autonomous_terminal_packet_exact": 0.90,
        },
        "controls": {
            "negative": [
                "binding-deranged",
                "state-reset",
                "query-only",
                "shuffled-label",
            ],
            "negative_max_above_empirical_chance": 0.02,
            "parameter_match_relative_tolerance": 0.01,
            "positive": "favorable-dense-recurrent-control",
            "training_flop_relative_tolerance": 0.05,
        },
        "data": {
            "confirmation_visible_to_optimizer": False,
            "release_sha256": ETTR_RELEASE_SHA256,
            "same_examples_across_backbones": True,
            "source_deleted_evaluation": True,
            "tokenizer_specific_transcoding": "semantic-byte-equivalent-and-receipted",
        },
        "decision_rules": [
            "all-backbones-fail-same-oracle-component=>redesign-interface",
            "0.8b-or-3b-pass-and-smaller-fail=>record-capacity-floor",
            "dense-control-equals-or-beats-ettr=>reject-ettr-inclusion",
            "3b-fails-autonomous-composition=>retire-current-ettr",
            "joint-gate-fails=>replace-separate-fit-composition-with-one-model-owned-trajectory",
        ],
        "launch_authorized": False,
        "mechanism_admission": {
            "architecture_hash": mechanism_architecture_sha256(),
            "closed_current_family_endpoint": "v20-failed-stop-no-v21",
            "mechanism_schema": MECHANISM_SCHEMA,
            "source_sha256": mechanism_source_sha256(),
            "status": "unified-source-frozen-preflight-blocked",
            "successor": "tied-world-state-command-terminal-query-trajectory",
        },
        "optimizer_budget": {
            "accumulation_normalization": "global-over-replay-window",
            "component_updates_per_seed": 2000,
            "composition_updates_per_seed": 5000,
            "evaluation": "full-frozen-development-then-sealed-confirmation",
            "matched_charged_positions_required": True,
            "semantic_microbatch_size": 16,
            "semantic_microbatches_per_update": 4,
            "stratified_replay_receipt_required": True,
            "seeds": [31, 32],
        },
        "schema": SCHEMA,
        "status": "preregistered-not-launchable",
    }


def validate_preregistration(payload: Mapping[str, object]) -> None:
    if (
        payload.get("schema") != SCHEMA
        or payload.get("launch_authorized") is not False
        or payload.get("status") != "preregistered-not-launchable"
    ):
        raise CapabilityFloorContractError("capability-floor custody differs")
    backbones = payload.get("backbones")
    if not isinstance(backbones, list) or len(backbones) != 4:
        raise CapabilityFloorContractError("capability-floor backbones differ")
    candidates = set()
    for backbone in backbones:
        if (
            not isinstance(backbone, Mapping)
            or backbone.get("frozen") is not True
            or not isinstance(backbone.get("source_revision"), str)
            or len(str(backbone["source_revision"])) not in {40, 64}
            or not isinstance(backbone.get("candidate"), str)
        ):
            raise CapabilityFloorContractError("capability-floor backbone differs")
        candidates.add(str(backbone["candidate"]))
    if len(candidates) != 4:
        raise CapabilityFloorContractError("capability-floor backbone identity differs")
    component = payload.get("component_gates")
    composition = payload.get("composition_gates")
    controls = payload.get("controls")
    optimizer = payload.get("optimizer_budget")
    mechanism = payload.get("mechanism_admission")
    if (
        not isinstance(component, Mapping)
        or set(component.values()) != {0.95}
        or not isinstance(composition, Mapping)
        or set(composition.values()) != {0.90}
        or not isinstance(controls, Mapping)
        or controls.get("positive") != "favorable-dense-recurrent-control"
        or not isinstance(optimizer, Mapping)
        or optimizer.get("seeds") != [31, 32]
        or not isinstance(mechanism, Mapping)
        or not isinstance(mechanism.get("architecture_hash"), str)
        or len(str(mechanism["architecture_hash"])) != 64
        or mechanism.get("architecture_hash") != mechanism_architecture_sha256()
        or mechanism.get("source_sha256") != mechanism_source_sha256()
        or mechanism.get("mechanism_schema") != MECHANISM_SCHEMA
    ):
        raise CapabilityFloorContractError("capability-floor gates differ")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_preregistration()
    validate_preregistration(payload)
    _write_no_replace(args.output, _canonical_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
