from __future__ import annotations

import random

import torch

from diverge_cwc1_data import (
    CONFIRMATION_PAIRS,
    DEVELOPMENT_PAIRS,
    TRAIN_PAIRS,
    _record,
    counterfactual_source,
    generate_records,
    overlap_report,
    validate_record,
)
from diverge_cwc1_runtime import (
    CWC1Config,
    CounterfactualWorldCommitter,
    tensorize_records,
)


def _rows(count: int = 4):
    return [
        _record(
            split="train",
            seed=61,
            serial=index,
            pair=TRAIN_PAIRS[index % len(TRAIN_PAIRS)],
            rng=random.Random(index + 67),
        )
        for index in range(count)
    ]


def test_split_pairs_are_disjoint_and_complete():
    groups = (set(TRAIN_PAIRS), set(DEVELOPMENT_PAIRS), set(CONFIRMATION_PAIRS))
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert set.union(*groups) == {(left, right) for left in range(8) for right in range(8)}


def test_counterfactual_changes_only_directive_and_flips_semantics():
    row = _rows(1)[0]
    validate_record(row)
    partner = counterfactual_source(row)
    left, right = row["directive_bounds"]
    assert partner[:left] == row["source_text"][:left]
    assert partner[right:] == row["source_text"][right:]
    assert partner[left:right] != row["source_text"][left:right]


def test_projected_scores_are_exactly_role_involutive():
    rows = _rows()
    device = torch.device("cpu")
    torch.manual_seed(71)
    model = CounterfactualWorldCommitter(CWC1Config()).eval()
    normal = tensorize_records(rows, device)
    partner = tensorize_records(rows, device, counterfactual=True)
    first = model.projected_scores(normal, partner)
    second = model.projected_scores(partner, normal)
    assert torch.equal(first, second.flip(dims=(-1,)))


def test_duplicate_control_is_unchanged_by_partner_argument():
    rows = _rows()
    device = torch.device("cpu")
    torch.manual_seed(73)
    model = CounterfactualWorldCommitter(CWC1Config(projection_mode="duplicate")).eval()
    normal = tensorize_records(rows, device)
    partner = tensorize_records(rows, device, counterfactual=True)
    assert torch.equal(model.projected_scores(normal, partner), model.raw_scores(normal))


def test_every_frozen_split_has_exact_target_balance():
    groups = (
        ("train", generate_records(split="train", seed=2026080731, count=50_000)),
        (
            "development",
            generate_records(split="development", seed=2026080732, count=4_096),
        ),
        (
            "confirmation",
            generate_records(split="confirmation", seed=2026080733, count=4_096),
        ),
    )
    report = overlap_report(*groups)
    assert report["exact_target_balance"] is True
    assert report["all_zero"] is True
