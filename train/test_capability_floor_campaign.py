from copy import deepcopy

import pytest

from capability_floor_campaign import (
    CapabilityFloorContractError,
    build_preregistration,
    validate_preregistration,
)


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
    assert payload["mechanism_admission"]["architecture_hash"] is None


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
