from __future__ import annotations

import pytest

from audit_ettr_public_operation_role_capacity import (
    OperationRoleCapacityAuditError,
    operation_arity,
)


def test_operation_arity_counts_operator_and_arguments() -> None:
    assert operation_arity(["call", 4, [["operator"], ["left"], ["right"]]]) == 3


def test_operation_arity_rejects_non_operation_and_empty_call() -> None:
    with pytest.raises(OperationRoleCapacityAuditError):
        operation_arity(["call", 3, [["operator"]]])
    with pytest.raises(OperationRoleCapacityAuditError):
        operation_arity(["call", 4, []])
