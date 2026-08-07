from __future__ import annotations

import random

import torch

from diverge_cwc1_data import TRAIN_PAIRS, _record, counterfactual_source
from eval_diverge_cwc1 import (
    _block_swapped,
    _matched,
    _renamed,
    _score,
    _scrubbed,
)


def _row(serial: int = 2):
    return _record(
        split="train",
        seed=83,
        serial=serial,
        pair=TRAIN_PAIRS[serial % len(TRAIN_PAIRS)],
        rng=random.Random(serial + 89),
    )


def test_rename_preserves_geometry_and_counterfactual_action():
    row = _row()
    renamed = _renamed(row)
    assert len(renamed["source_text"]) == len(row["source_text"])
    assert renamed["candidate_bounds"] == row["candidate_bounds"]
    assert renamed["directive_bounds"] == row["directive_bounds"]
    assert set(renamed["candidate_labels"]).isdisjoint(row["candidate_labels"])
    assert counterfactual_source(renamed) != renamed["source_text"]


def test_directive_scrub_removes_the_only_counterfactual_difference():
    row = _row()
    scrubbed = _scrubbed(row)
    left, right = scrubbed["directive_bounds"]
    assert scrubbed["source_text"][left:right] == "#" * (right - left)
    assert counterfactual_source(scrubbed) == scrubbed["source_text"]
    for block_left, block_right in row["candidate_bounds"]:
        assert (
            scrubbed["source_text"][block_left:block_right]
            == row["source_text"][block_left:block_right]
        )


def test_whole_block_swap_is_an_exact_involution():
    row = _row()
    swapped = _block_swapped(row)
    restored = _block_swapped(swapped)
    assert swapped["target_position"] == 1 - row["target_position"]
    assert swapped["candidate_labels"] == list(reversed(row["candidate_labels"]))
    assert restored["source_text"] == row["source_text"]
    assert restored["candidate_bounds"] == row["candidate_bounds"]
    assert restored["candidate_labels"] == row["candidate_labels"]
    assert restored["target_position"] == row["target_position"]


def test_score_uses_mapped_counterfactual_targets():
    rows = [_row(0), _row(1)]
    normal = torch.tensor([[3.0, -1.0], [-2.0, 4.0]])
    partner = normal.flip(dims=(-1,))
    assert _score(rows, normal)["exact"] == 2
    assert _score(rows, partner, flip_target=True)["exact"] == 2


def test_matched_receipts_include_compute_and_initialization():
    common = {
        "seed": 1,
        "updates": 2,
        "batch_size": 3,
        "learning_rate": 0.1,
        "data_sha256": "data",
        "trainable_parameters": 4,
        "initial_model_sha256": "initial",
        "source_bytes_seen": 5,
        "forwards_per_update": 2,
    }
    assert all(_matched([common, dict(common)]).values())
    changed = dict(common)
    changed["forwards_per_update"] = 1
    assert _matched([common, changed])["forwards_per_update"] is False
