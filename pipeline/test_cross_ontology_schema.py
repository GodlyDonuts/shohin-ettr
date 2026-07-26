from __future__ import annotations

import pytest

from cross_ontology_schema import (
    CrossOntologySchemaError,
    ReactorState,
    RelationEdge,
    RelationSpec,
    Transaction,
    TransactionOpcode,
    apply_transaction,
    apply_transactions,
)


def _empty() -> ReactorState:
    return ReactorState(
        capacity=6,
        type_count=2,
        relation_specs=(
            RelationSpec(0, (0, 1)),
            RelationSpec(1, (1,)),
        ),
    )


def test_generic_transactions_round_trip() -> None:
    state = apply_transactions(
        _empty(),
        (
            Transaction(TransactionOpcode.ALLOC, (0, 0)),
            Transaction(TransactionOpcode.ALLOC, (1, 1)),
            Transaction(TransactionOpcode.WRITE, (1, 17)),
            Transaction(TransactionOpcode.LINK, (0, 0, 1)),
            Transaction(TransactionOpcode.LINK, (1, 1)),
            Transaction(TransactionOpcode.SET_ROOT, (0,)),
            Transaction(TransactionOpcode.COMMIT),
            Transaction(TransactionOpcode.HALT),
        ),
    )
    assert state.halted
    assert state.root == 0
    assert state.committed_steps == 8
    assert state.edges == (
        RelationEdge(0, (0, 1)),
        RelationEdge(1, (1,)),
    )
    assert ReactorState.from_deployed_wire(
        state.deployed_wire()
    ) == state


def test_type_invalid_link_is_atomic() -> None:
    state = apply_transactions(
        _empty(),
        (
            Transaction(TransactionOpcode.ALLOC, (0, 0)),
            Transaction(TransactionOpcode.ALLOC, (1, 0)),
        ),
    )
    with pytest.raises(
        CrossOntologySchemaError,
        match="typing differs",
    ):
        apply_transaction(
            state,
            Transaction(TransactionOpcode.LINK, (0, 0, 1)),
        )
    assert state.edges == ()
    assert state.committed_steps == 2


def test_clear_rejects_live_references() -> None:
    state = apply_transactions(
        _empty(),
        (
            Transaction(TransactionOpcode.ALLOC, (0, 0)),
            Transaction(TransactionOpcode.ALLOC, (1, 1)),
            Transaction(TransactionOpcode.LINK, (0, 0, 1)),
        ),
    )
    with pytest.raises(
        CrossOntologySchemaError,
        match="live structure",
    ):
        apply_transaction(
            state,
            Transaction(TransactionOpcode.CLEAR, (1,)),
        )


def test_halt_is_terminal() -> None:
    halted = apply_transaction(
        _empty(),
        Transaction(TransactionOpcode.HALT),
    )
    with pytest.raises(
        CrossOntologySchemaError,
        match="follows halt",
    ):
        apply_transaction(
            halted,
            Transaction(TransactionOpcode.COMMIT),
        )
