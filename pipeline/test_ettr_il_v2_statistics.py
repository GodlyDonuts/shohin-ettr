from __future__ import annotations

import hashlib

import pytest

import ettr_il_v2_statistics as statistics
from ettr_il_v2_statistics import (
    BOOTSTRAP_REPLICATES,
    BootstrapCell,
    CounterStream,
    StatisticsError,
    build_bootstrap_plan,
    replicate_root,
    simultaneous_lower_bounds,
)


SPLIT_SHA256 = hashlib.sha256(b"development-plaintext").hexdigest()


def test_replicate_root_and_counter_words_match_frozen_golden_vectors() -> None:
    root = replicate_root(SPLIT_SHA256, 17)
    assert root.hex() == "768b05d5290cbc6c214306de8aede09340cc648b38bc35aa697af2f3e9c92936"
    stream = CounterStream(root, "model-seeds")
    assert tuple(stream.uint64() for _ in range(5)) == (
        18369454481433380417,
        6108995717740777198,
        9831080199493775058,
        13736006538865524672,
        12601320168380833682,
    )
    assert stream.words_consumed == 5


def test_draws_use_domain_separation_and_are_deterministic() -> None:
    root = replicate_root(SPLIT_SHA256, 9)
    first = CounterStream(root, "model-seeds").draws(n=5, count=5)
    replay = CounterStream(root, "model-seeds").draws(n=5, count=5)
    cell = CounterStream(root, "cell|0|horn|seen_id").draws(n=32, count=32)
    assert first == replay == (1, 1, 2, 1, 0)
    assert cell[:8] == (17, 11, 1, 25, 20, 2, 18, 17)
    assert first != cell[:5]


def test_rejection_sampling_discards_out_of_range_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = CounterStream(b"\x00" * 32, "test")
    values = iter(((1 << 64) - 1, 7))
    monkeypatch.setattr(CounterStream, "uint64", lambda _self: next(values))
    assert stream.draw_below(10) == 7


def test_build_plan_sorts_cells_and_resamples_exact_counts() -> None:
    cells = (
        BootstrapCell(1, "resource", "all_axes", 24),
        BootstrapCell(0, "horn", "seen_id", 32),
    )
    first = build_bootstrap_plan(SPLIT_SHA256, 0, cells)
    replay = build_bootstrap_plan(SPLIT_SHA256, 0, reversed(cells))
    assert first == replay
    assert len(first.model_seed_indices) == 5
    assert all(0 <= value < 5 for value in first.model_seed_indices)
    assert [len(indices) for _, indices in first.cell_indices] == [32, 24]
    assert tuple(cell for cell, _ in first.cell_indices) == tuple(sorted(cells))


def test_protocol_rejects_malformed_inputs() -> None:
    with pytest.raises(StatisticsError, match="SHA"):
        replicate_root("bad", 0)
    with pytest.raises(StatisticsError, match="index"):
        replicate_root(SPLIT_SHA256, BOOTSTRAP_REPLICATES)
    with pytest.raises(StatisticsError, match="ASCII"):
        CounterStream(b"\x00" * 32, "non-ascii-\N{SNOWMAN}")
    with pytest.raises(StatisticsError, match="upper bound"):
        CounterStream(b"\x00" * 32, "test").draw_below(0)
    with pytest.raises(StatisticsError, match="core count"):
        build_bootstrap_plan(
            SPLIT_SHA256,
            0,
            (BootstrapCell(0, "horn", "all_axes", 32),),
        )


def test_simultaneous_lower_bound_uses_exact_one_based_95000th_element() -> None:
    observed = (0.50, 0.25)
    replicates = tuple(
        (
            0.50 - index / 1_000_000,
            0.25 - index / 2_000_000,
        )
        for index in range(BOOTSTRAP_REPLICATES)
    )
    bounds = simultaneous_lower_bounds(observed, replicates)
    assert bounds == pytest.approx((0.405001, 0.155001))


def test_simultaneous_lower_bound_requires_exact_geometry() -> None:
    with pytest.raises(StatisticsError, match="population"):
        simultaneous_lower_bounds((), ())
    with pytest.raises(StatisticsError, match="endpoint"):
        simultaneous_lower_bounds(
            (0.0, 0.0),
            ((0.0,),) * BOOTSTRAP_REPLICATES,
        )
    with pytest.raises(StatisticsError, match="replicate count"):
        simultaneous_lower_bounds((0.0,), ((0.0,),))


def test_statistics_module_imports_no_runtime_prng() -> None:
    source = open(statistics.__file__, encoding="ascii").read()
    assert "import random" not in source
    assert "numpy.random" not in source
