from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from audit_ettr_public_operation_role_capacity import (
    EFFECT_CAPACITY_SCHEMA,
    EFFECT_SLOTS,
    MAXIMUM_ROLES,
    MOTORS_PER_ROLE,
    OperationRoleCapacityAuditError,
    _load_effect_capacity,
    operation_arity,
)
from ettr_il_v3_protocol import canonical_json_bytes


def test_operation_arity_counts_operator_and_arguments() -> None:
    assert operation_arity(["call", 4, [["operator"], ["left"], ["right"]]]) == 3


def test_operation_arity_rejects_non_operation_and_empty_call() -> None:
    with pytest.raises(OperationRoleCapacityAuditError):
        operation_arity(["call", 3, [["operator"]]])
    with pytest.raises(OperationRoleCapacityAuditError):
        operation_arity(["call", 4, []])


def test_role_geometry_covers_maximum_effects_for_smallest_operation() -> None:
    assert MAXIMUM_ROLES == 4
    assert MOTORS_PER_ROLE == 5
    assert EFFECT_SLOTS == 20
    assert 2 * MOTORS_PER_ROLE >= 10


def test_effect_capacity_receipt_is_hash_and_payload_bound(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    tokenizer_sha256 = "a" * 64
    report = {
        "data_root": str(data_root),
        "effect_set_capacity": {
            split: {"histogram": {"10": 1}, "instances": 1, "maximum": 10}
            for split in ("train", "development")
        },
        "schema": EFFECT_CAPACITY_SCHEMA,
        "status": "pass",
        "tokenizer": {"sha256": tokenizer_sha256},
    }
    report["report_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    payload = canonical_json_bytes(report)
    path = tmp_path / "effect-report.json"
    path.write_bytes(payload)
    loaded = _load_effect_capacity(
        path,
        hashlib.sha256(payload).hexdigest(),
        data_root=data_root,
        tokenizer_sha256=tokenizer_sha256,
    )
    assert loaded["train"]["maximum"] == 10
    with pytest.raises(OperationRoleCapacityAuditError):
        _load_effect_capacity(
            path,
            "0" * 64,
            data_root=data_root,
            tokenizer_sha256=tokenizer_sha256,
        )
