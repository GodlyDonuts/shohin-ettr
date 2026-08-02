from types import SimpleNamespace

from audit_ettr_public_operation_state_delta import (
    runtime_state_value,
    state_delta_factor_values,
    state_delta_value,
)


def _state(*, value: int, edge: bool = False) -> SimpleNamespace:
    values = [0] * 64
    values[32] = value
    types = [0] * 64
    active = [False] * 64
    active[32] = True
    root = [False] * 64
    return SimpleNamespace(
        active=tuple(active),
        committed=False,
        halted=False,
        relations=frozenset({(2, 32, 32)} if edge else set()),
        root=tuple(root),
        type_index=tuple(types),
        value_code=tuple(values),
    )


def test_state_delta_is_order_independent_final_state_quotient() -> None:
    before = _state(value=3)
    after = _state(value=7, edge=True)
    assert state_delta_value(before, after) == {
        "edges_added": [(2, 32, 32)],
        "edges_removed": [],
        "nodes": [[32, [True, 0, 3, False], [True, 0, 7, False]]],
        "status": [False, False, False, False],
    }


def test_runtime_state_excludes_materializer_control_slots() -> None:
    state = _state(value=7, edge=True)
    assert runtime_state_value(state) == {
        "edges": [(2, 32, 32)],
        "nodes": [[32, True, 0, 7, False]],
        "status": [False, False],
    }


def test_state_delta_factors_separate_shape_addresses_and_payloads() -> None:
    delta = state_delta_value(_state(value=3), _state(value=7, edge=True))
    assert state_delta_factor_values(delta) == {
        "delta_shape": {
            "edge_additions": 1,
            "edge_removals": 0,
            "node_field_changes": [[False, False, True, False]],
            "status_changes": [False, False],
        },
        "delta_addresses": {
            "edges_added": [(2, 32, 32)],
            "edges_removed": [],
            "nodes": [32],
        },
        "delta_payloads": {
            "edge_relations_added": [2],
            "edge_relations_removed": [],
            "nodes": [
                [
                    [True, 0, 3, False],
                    [True, 0, 7, False],
                ]
            ],
            "status_after": [False, False],
        },
    }
