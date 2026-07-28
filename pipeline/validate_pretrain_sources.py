#!/usr/bin/env python3
"""Fail-closed validation for the Phase 2 pretraining source registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class RegistryError(ValueError):
    """The source registry violates a launch-blocking invariant."""


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_registry(registry: dict[str, Any]) -> dict[str, Any]:
    if registry.get("schema_version") != 2:
        raise RegistryError("schema_version must be 2")
    if registry.get("status") != "candidate_registry_not_training_admission":
        raise RegistryError("registry status must explicitly deny training admission")
    if registry.get("owner_repo") != "GodlyDonuts/shohin-ettr":
        raise RegistryError("owner_repo must be the private Shohin research repository")

    semantics = registry.get("decision_semantics", {})
    required_receipts = semantics.get("training_admission_requires", [])
    if not isinstance(required_receipts, list) or len(required_receipts) < 8:
        raise RegistryError("training_admission_requires is incomplete")

    requirements = registry.get("global_requirements", {})
    mandatory_true = (
        "pin_upstream_revision",
        "retain_per_document_provenance",
        "retain_per_document_license_for_code",
        "within_source_exact_and_near_dedup",
        "cross_source_dedup",
        "evaluation_decontamination_before_tokenization",
        "cross_corpus_decontamination_after_selection",
        "pii_and_secret_scan",
        "malware_and_unsafe_code_scan",
        "human_sample_audit",
        "factual_and_extraction_spot_checks",
        "equal_token_model_utility_ablation",
        "fail_on_unknown_code_license",
        "admission_receipt_required",
    )
    missing_requirements = [name for name in mandatory_true if requirements.get(name) is not True]
    if missing_requirements:
        raise RegistryError(
            "required global gates are not fail-closed: " + ", ".join(missing_requirements)
        )

    candidate_mix = registry.get("phase2_quality_first_mix_candidate_pct", {})
    if not isinstance(candidate_mix, dict) or not candidate_mix:
        raise RegistryError("phase2 candidate mix is missing")
    candidate_total = sum(float(value) for value in candidate_mix.values())
    if abs(candidate_total - 100.0) > 1e-9:
        raise RegistryError(f"phase2 candidate mix must total 100, got {candidate_total}")
    if any(float(value) <= 0.0 for value in candidate_mix.values()):
        raise RegistryError("phase2 candidate mix weights must be positive")

    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RegistryError("sources must be a nonempty list")
    seen: set[str] = set()
    active_sources = 0
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise RegistryError(f"source {index} is not an object")
        source_id = source.get("id")
        if not _nonblank(source_id):
            raise RegistryError(f"source {index} has no id")
        if source_id in seen:
            raise RegistryError(f"duplicate source id: {source_id}")
        seen.add(source_id)

        for field in (
            "dataset",
            "url",
            "domain",
            "priority",
            "decision",
            "known_scale",
            "license_summary",
            "mirror_policy",
        ):
            if not _nonblank(source.get(field)):
                raise RegistryError(f"{source_id} has blank {field}")
        if not source["url"].startswith("https://"):
            raise RegistryError(f"{source_id} URL is not HTTPS")
        if not isinstance(source.get("selection"), dict):
            raise RegistryError(f"{source_id} selection must be an object")
        if not isinstance(source.get("risks"), list) or not source["risks"]:
            raise RegistryError(f"{source_id} must enumerate risks")

        weight = float(source.get("target_mix_pct", 0))
        if weight < 0:
            raise RegistryError(f"{source_id} has a negative target_mix_pct")
        if weight > 0:
            active_sources += 1
            decision = source["decision"].lower()
            if "reject" in decision or "hold" in decision:
                raise RegistryError(f"{source_id} has positive weight but decision={decision}")
        phase2_weight = float(source.get("phase2_candidate_mix_pct", 0))
        if phase2_weight < 0:
            raise RegistryError(f"{source_id} has a negative phase2_candidate_mix_pct")

    if active_sources < 4:
        raise RegistryError("registry has too few active source candidates")

    return {
        "schema": "shohin-pretrain-source-registry-validation-v1",
        "sources": len(sources),
        "active_legacy_source_candidates": active_sources,
        "phase2_candidate_mix_total_pct": candidate_total,
        "status": "valid_candidate_registry_not_training_admission",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default=str(Path(__file__).with_name("pretrain_sources.json")),
    )
    args = parser.parse_args()
    registry = json.loads(Path(args.registry).read_text())
    print(json.dumps(validate_registry(registry), sort_keys=True))


if __name__ == "__main__":
    main()
