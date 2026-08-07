from __future__ import annotations

import random

from diverge_ewc1_data import TRAIN_PAIRS, _record, scan_symbol_occurrences
from eval_diverge_ewc1 import (
    ALIAS_PERMUTATION,
    _alias_permuted,
    _register_swapped,
    _renamed,
    _scrubbed,
)


def _row():
    return _record(
        split="train",
        seed=41,
        serial=2,
        pair=TRAIN_PAIRS[0],
        rng=random.Random(43),
    )


def test_register_and_alias_table_actions_do_not_change_source():
    row = _row()
    assert _register_swapped(row)["source_text"] == row["source_text"]
    permuted = _alias_permuted(row)
    assert permuted["source_text"] == row["source_text"]
    assert permuted["aliases"] == [row["aliases"][index] for index in ALIAS_PERMUTATION]


def test_entity_rename_preserves_candidate_geometry():
    row = _row()
    renamed = _renamed(row)
    assert len(scan_symbol_occurrences(renamed["source_text"], renamed["aliases"])) == len(
        scan_symbol_occurrences(row["source_text"], row["aliases"])
    )
    assert set(row["aliases"]).isdisjoint(renamed["aliases"])


def test_scrub_preserves_declared_symbols_but_removes_context():
    row = _row()
    scrubbed = _scrubbed(row)
    for symbol in (*row["aliases"], *row["registers"]):
        if symbol in row["source_text"]:
            assert symbol in scrubbed["source_text"]
    assert "Begin" not in scrubbed["source_text"]
    assert "Execute" not in scrubbed["source_text"]
