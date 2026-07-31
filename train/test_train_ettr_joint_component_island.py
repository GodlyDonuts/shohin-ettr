import pytest

from train_ettr_joint_component_island import (
    ETTRJointComponentIslandError,
    _validate_parent_lineage,
)
from train_ettr_joint_instruction_canary import (
    MODEL_SCHEMA,
    RUN_SCHEMA,
)


def _valid_parent() -> tuple[dict[str, object], dict[str, object]]:
    config = {"slots": 64}
    contract = {
        "ettr_release_sha256": "1" * 64,
        "model_config": config,
        "schema": RUN_SCHEMA,
    }
    payload = {
        "ettr_config": config,
        "run_contract_sha256": "2" * 64,
        "schema": MODEL_SCHEMA,
    }
    return contract, payload


def test_joint_component_parent_lineage_accepts_exact_parent() -> None:
    contract, payload = _valid_parent()
    _validate_parent_lineage(
        contract,
        payload,
        release_sha256="1" * 64,
        parent_run_contract_sha256="2" * 64,
    )


def test_joint_component_parent_lineage_rejects_config_drift() -> None:
    contract, payload = _valid_parent()
    payload["ettr_config"] = {"slots": 32}
    with pytest.raises(
        ETTRJointComponentIslandError,
        match="lineage differs",
    ):
        _validate_parent_lineage(
            contract,
            payload,
            release_sha256="1" * 64,
            parent_run_contract_sha256="2" * 64,
        )
