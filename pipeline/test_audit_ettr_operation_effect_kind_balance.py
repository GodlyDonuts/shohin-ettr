"""Tests for exact ETTR operation-effect kind balance."""

from __future__ import annotations

from audit_ettr_operation_effect_kind_balance import effect_kinds


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
