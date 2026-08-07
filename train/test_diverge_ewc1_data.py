from __future__ import annotations

import random

from diverge_ewc1_data import (
    CONFIRMATION_PAIRS,
    DEVELOPMENT_PAIRS,
    TRAIN_PAIRS,
    _record,
    scan_integer_spans,
    scan_symbol_occurrences,
    validate_record,
)


def _row(split: str = "train"):
    pairs = {
        "train": TRAIN_PAIRS,
        "development": DEVELOPMENT_PAIRS,
        "confirmation": CONFIRMATION_PAIRS,
    }[split]
    return _record(
        split=split,
        seed={"train": 17, "development": 18, "confirmation": 19}[split],
        serial=3,
        pair=pairs[0],
        rng=random.Random(11),
    )


def test_renderer_buckets_are_disjoint_and_cover_all_pairs():
    groups = (set(TRAIN_PAIRS), set(DEVELOPMENT_PAIRS), set(CONFIRMATION_PAIRS))
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert set.union(*groups) == {
        (initial, sequence) for initial in range(8) for sequence in range(8)
    }


def test_generated_row_reconstructs_typed_world():
    row = _row()
    validate_record(row)
    text = row["source_text"]
    numeric = scan_integer_spans(text)
    initial = tuple(
        int(text[numeric[index][0] : numeric[index][1]])
        for index in row["numeric_targets"]
    )
    occurrences = scan_symbol_occurrences(text, row["aliases"])
    symbols = tuple(
        occurrence[2]
        for occurrence, target in zip(
            occurrences, row["operation_targets"], strict=True
        )
        if target
    )
    assert initial == tuple(row["initial_state"])
    assert symbols == tuple(row["symbols"])


def test_source_disjoint_names_change_with_split_seed():
    train = _row("train")
    development = _row("development")
    assert set(train["aliases"]).isdisjoint(development["aliases"])
    assert set(train["registers"]).isdisjoint(development["registers"])
