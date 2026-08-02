from __future__ import annotations

import pytest

from audit_ettr_operation_effect_family_identifiability import (
    OperationEffectFamilyIdentifiabilityError,
    abstract_resolved_operation,
)
from audit_ettr_operation_effect_kind_balance import (
    EffectKindBalanceAuditError,
    operation_effect_family,
)


def test_operation_effect_family_is_exact_and_mutually_exclusive() -> None:
    assert operation_effect_family(()) == "none"
    assert operation_effect_family(("write", "write")) == "write"
    assert operation_effect_family(("link", "link", "link")) == "link"
    with pytest.raises(EffectKindBalanceAuditError, match="family differs"):
        operation_effect_family(("write", "link"))
    with pytest.raises(EffectKindBalanceAuditError, match="family differs"):
        operation_effect_family(("commit",))


def test_abstract_resolved_operation_removes_only_literal_payloads() -> None:
    value = [
        "call",
        4,
        [
            ["integer", 17],
            ["declared-symbol", 7, ["call", 2, [["integer", 99]]]],
        ],
    ]
    assert abstract_resolved_operation(value) == [
        "call",
        4,
        [
            ["integer"],
            ["declared-symbol", 7, ["call", 2, [["integer"]]]],
        ],
    ]
    with pytest.raises(
        OperationEffectFamilyIdentifiabilityError,
        match="structure differs",
    ):
        abstract_resolved_operation(("call", 4, ()))
