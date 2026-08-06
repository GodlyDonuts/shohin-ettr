#!/usr/bin/env python3
"""Contract tests for the frozen DIVERGE-MQB1 interface."""

from __future__ import annotations

from dataclasses import replace

import torch

from diverge_mei1_data import EVIDENCE_COHORTS, generate_probe_evidence
from diverge_mqb1_data import FIELD_COUNT, generate_mention_evidence
from diverge_mqb1_runtime import (
    MAX_CANDIDATES,
    NONE_ADDRESS,
    NONE_PHASE,
    NONE_VALUE,
    REGISTER_COUNT,
    VALUE_COUNT,
    MQB1Config,
    MQB1ContractError,
    MentionBinderLogits,
    MentionEvidenceBinder,
    exact_field_assignment,
    source_audit,
)


def _perfect_logits(batch: int = 1) -> tuple[MentionBinderLogits, torch.Tensor]:
    words = MAX_CANDIDATES
    value = torch.full((batch, words, VALUE_COUNT + 1), -20.0)
    phase = torch.full((batch, words, 3), -20.0)
    address = torch.full((batch, words, 6), -20.0)
    field = torch.full((batch, words, FIELD_COUNT), -20.0)
    pair = torch.full((batch, words, words), -20.0)
    value[..., NONE_VALUE] = 5.0
    phase[..., NONE_PHASE] = 5.0
    address[..., NONE_ADDRESS] = 5.0
    for typed_field in range(FIELD_COUNT):
        value[:, typed_field, NONE_VALUE] = -20.0
        value[:, typed_field, 20 + typed_field] = 20.0
        phase[:, typed_field, NONE_PHASE] = -20.0
        phase[:, typed_field, typed_field // REGISTER_COUNT] = 20.0
        address[:, typed_field, NONE_ADDRESS] = -20.0
        address[:, typed_field, typed_field % REGISTER_COUNT] = 20.0
        field[:, typed_field, typed_field] = 20.0
    for address_index in range(REGISTER_COUNT):
        pair[:, address_index, REGISTER_COUNT + address_index] = 20.0
        pair[:, REGISTER_COUNT + address_index, address_index] = 20.0
    return MentionBinderLogits(value, phase, address, field, pair), torch.ones(
        batch, words, dtype=torch.bool
    )


def test_renderer_parity() -> None:
    for cohort_index, cohort in enumerate(EVIDENCE_COHORTS):
        for index in range(100):
            seed = 800_000 * (cohort_index + 1) + index
            old = generate_probe_evidence(seed=seed, cohort=cohort)
            new = generate_mention_evidence(seed=seed, cohort=cohort)
            assert new.words == old.words
            assert new.before == old.before and new.after == old.after
            assert len(new.mentions) == FIELD_COUNT
            assert [mention.field for mention in new.mentions] == list(range(FIELD_COUNT))
            assert len({mention.word_index for mention in new.mentions}) == FIELD_COUNT


def test_exact_assignment_is_one_to_one() -> None:
    scores = torch.full((2, MAX_CANDIDATES, FIELD_COUNT), -10.0)
    for field in range(FIELD_COUNT):
        scores[:, field, field] = 10.0
    scores[:, 0, 1] = 100.0
    mask = torch.ones(2, MAX_CANDIDATES, dtype=torch.bool)
    assignment = exact_field_assignment(scores, mask)
    assert bool(assignment.valid.all())
    for row in assignment.candidate_for_field:
        assert len(set(row.tolist())) == FIELD_COUNT


def test_complete_binding_and_atomic_value_swap() -> None:
    binder = MentionEvidenceBinder(
        MQB1Config(input_width=8, width=16, heads=4, layers=1, pair_width=8)
    )
    logits, mask = _perfect_logits()
    binding = binder.decode(logits, mask)
    assert bool(binding.valid.item())
    assert binding.before.tolist() == [[20, 21, 22, 23, 24]]
    assert binding.after.tolist() == [[25, 26, 27, 28, 29]]
    assert len(set(binding.provenance[0].tolist())) == FIELD_COUNT

    shifted_value = logits.value.clone()
    shifted_value[:, :FIELD_COUNT] = logits.value[:, :FIELD_COUNT].roll(1, dims=1)
    shifted = binder.decode(replace(logits, value=shifted_value), mask)
    assert bool(shifted.valid.item())
    assert shifted.provenance.equal(binding.provenance)
    assert not shifted.selected_values.equal(binding.selected_values)


def test_pair_certificate_fails_closed() -> None:
    binder = MentionEvidenceBinder(
        MQB1Config(input_width=8, width=16, heads=4, layers=1, pair_width=8)
    )
    logits, mask = _perfect_logits()
    pair = logits.pair.clone()
    pair[:, :REGISTER_COUNT, REGISTER_COUNT:FIELD_COUNT] = -20.0
    pair[:, REGISTER_COUNT:FIELD_COUNT, :REGISTER_COUNT] = -20.0
    rejected = binder.decode(replace(logits, pair=pair), mask)
    assert not bool(rejected.valid.item())


def test_short_source_rejected_and_audit_passes() -> None:
    binder = MentionEvidenceBinder(
        MQB1Config(input_width=8, width=16, heads=4, layers=1, pair_width=8)
    )
    logits, mask = _perfect_logits()
    short = MentionBinderLogits(
        logits.value[:, :9],
        logits.phase[:, :9],
        logits.address[:, :9],
        logits.field[:, :9],
        logits.pair[:, :9, :9],
    )
    try:
        binder.decode(short, mask[:, :9])
    except MQB1ContractError:
        pass
    else:
        raise AssertionError("short source did not fail closed")
    assert source_audit()["pass"]


if __name__ == "__main__":
    test_renderer_parity()
    test_exact_assignment_is_one_to_one()
    test_complete_binding_and_atomic_value_swap()
    test_pair_certificate_fails_closed()
    test_short_source_rejected_and_audit_passes()
    print("DIVERGE-MQB1 tests passed")
