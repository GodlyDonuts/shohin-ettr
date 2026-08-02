"""Tests for exact ETTR operation-effect kind balance."""

from __future__ import annotations

from audit_ettr_operation_effect_kind_balance import (
    effect_kinds,
    write_link_dependency_counts,
)


def test_effect_kinds_match_typed_target_contract() -> None:
    delta = {
        "nodes": [
            [0, [False, 0, 0, False], [True, 2, 3, True]],
            [1, [True, 2, 3, False], [True, 2, 4, False]],
            [2, [True, 2, 3, False], [True, 4, 3, False]],
            [3, [True, 2, 3, False], [False, 2, 3, False]],
        ],
        "edges_added": [[1, 0, 1]],
        "edges_removed": [[2, 1, 2]],
        "status": [False, False, True, False],
    }
    assert effect_kinds(delta) == (
        "allocate",
        "write",
        "replace",
        "clear",
        "link",
        "unlink",
        "root_set",
        "commit",
    )


def test_effect_kinds_ignore_status_reset_like_training_target() -> None:
    delta = {
        "nodes": [],
        "edges_added": [],
        "edges_removed": [],
        "status": [True, False, False, False],
    }
    assert effect_kinds(delta) == ()


def test_write_link_dependency_counts_same_operation_endpoints() -> None:
    delta = {
        "nodes": [
            [0, [True, 2, 3, False], [True, 2, 4, False]],
            [2, [True, 2, 5, False], [True, 2, 6, False]],
        ],
        "edges_added": [[1, 0, 1], [2, 1, 2], [3, 0, 2]],
        "edges_removed": [],
        "status": [False, False, False, False],
    }
    assert write_link_dependency_counts(delta) == {
        "link_source_endpoints_written": 2,
        "link_target_endpoints_written": 2,
        "links_added": 3,
        "links_touching_written_slot": 3,
        "operations_with_link": 1,
        "operations_with_write": 1,
        "operations_with_write_and_link": 1,
        "operations_with_write_and_link_touching": 1,
        "written_slots": 2,
    }
