from copy import deepcopy
import json
from pathlib import Path

import pytest

from capability_floor_campaign import (
    CapabilityFloorContractError,
    build_preregistration,
    validate_preregistration,
)
from capability_floor_trajectory import mechanism_architecture_sha256


def test_capability_floor_is_frozen_and_not_launchable() -> None:
    payload = build_preregistration()
    validate_preregistration(payload)
    assert payload["launch_authorized"] is False
    assert [row["parameter_class"] for row in payload["backbones"]] == [
        "125m",
        "360m",
        "0.8b",
        "3b",
    ]
    assert payload["mechanism_admission"]["architecture_hash"] == (
        mechanism_architecture_sha256()
    )
    assert payload["mechanism_admission"]["status"] == (
        "unified-source-frozen-preflight-blocked"
    )
    assert payload["mechanism_admission"]["closed_current_family_endpoint"] == (
        "v20-failed-stop-no-v21"
    )
    assert payload["optimizer_budget"]["semantic_microbatches_per_update"] == 4


def test_capability_floor_rejects_premature_launch() -> None:
    payload = deepcopy(build_preregistration())
    payload["launch_authorized"] = True
    with pytest.raises(CapabilityFloorContractError, match="custody differs"):
        validate_preregistration(payload)


def test_capability_floor_rejects_weakened_component_gate() -> None:
    payload = deepcopy(build_preregistration())
    payload["component_gates"]["oracle_program_executor_exact"] = 0.90
    with pytest.raises(CapabilityFloorContractError, match="gates differ"):
        validate_preregistration(payload)


def test_checked_in_preregistration_matches_builder() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/r12/ettr_capability_floor_preregistration_v1.json"
    )
    assert json.loads(path.read_text(encoding="ascii")) == build_preregistration()
