from __future__ import annotations

import hashlib
import itertools

import pytest

import ettr_il_v2_controls as controls
from ettr_il_v2_controls import (
    BindingKey,
    ControlError,
    TargetBundleDescriptor,
    build_binding_derangement,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _key(label: str = "base") -> BindingKey:
    return BindingKey(
        ontology="horn",
        depth=2,
        renderer=0,
        presentation=label,
        query_semantic_pair_signature=_digest("queries"),
        paraphrase_pair_signature=_digest("paraphrases"),
        initial_support_shape=_digest("initial-support"),
        terminal_support_shape=_digest("terminal-support"),
        transaction_mask=_digest("transaction-mask"),
    )


def _record(index: int, answer: int, key: BindingKey | None = None):
    return TargetBundleDescriptor(
        semantic_rectangle_id=_digest(f"rectangle|{index}"),
        key=_key() if key is None else key,
        terminal_packet_sha256s=tuple(
            _digest(f"packet|{index}|{corner}")
            for corner in range(4)
        ),
        transaction_sha256s=tuple(
            _digest(f"trace|{index}|{corner}")
            for corner in range(4)
        ),
        answer_labels=(answer,) * 16,
    )


def test_derangement_is_deterministic_and_lexicographically_minimal() -> None:
    records = tuple(
        _record(index, index % 2)
        for index in range(4)
    )
    first = build_binding_derangement(records, fold=0)
    replay = build_binding_derangement(reversed(records), fold=0)
    assert first == replay
    assert len(first.assignments) == 4
    assert first.receipt()["fixed_points"] == 0
    assert len({value.donor_id for value in first.assignments}) == 4

    recipients = tuple(sorted(records, key=lambda value: value.semantic_rectangle_id))
    rank_vectors: list[tuple[int, ...]] = []
    donor_vectors: list[tuple[str, ...]] = []
    for donors in itertools.permutations(recipients):
        if all(
            controls._admissible(recipient, donor)
            for recipient, donor in zip(recipients, donors, strict=True)
        ):
            ordered_candidates = {
                recipient.semantic_rectangle_id: sorted(
                    (
                        controls._donor_digest(
                            0,
                            recipient.semantic_rectangle_id,
                            donor.semantic_rectangle_id,
                        ),
                        donor.semantic_rectangle_id,
                    )
                    for donor in recipients
                    if controls._admissible(recipient, donor)
                )
                for recipient in recipients
            }
            vector = tuple(
                tuple(value[1] for value in ordered_candidates[recipient.semantic_rectangle_id]).index(
                    donor.semantic_rectangle_id
                )
                for recipient, donor in zip(recipients, donors, strict=True)
            )
            rank_vectors.append(vector)
            donor_vectors.append(
                tuple(donor.semantic_rectangle_id for donor in donors)
            )
    best_index = min(
        range(len(rank_vectors)),
        key=lambda index: (rank_vectors[index], donor_vectors[index]),
    )
    assert tuple(value.donor_rank for value in first.assignments) == rank_vectors[
        best_index
    ]
    assert tuple(value.donor_id for value in first.assignments) == donor_vectors[
        best_index
    ]


def test_rejects_group_without_perfect_matching() -> None:
    records = (
        _record(0, 0),
        _record(1, 0),
        _record(2, 1),
    )
    with pytest.raises(ControlError, match="perfect matching"):
        build_binding_derangement(records, fold=0)


def test_groups_are_matched_independently() -> None:
    first_key = _key("base")
    second_key = _key("alpha")
    records = (
        _record(0, 0, first_key),
        _record(1, 1, first_key),
        _record(2, 0, second_key),
        _record(3, 1, second_key),
    )
    result = build_binding_derangement(records, fold=1)
    by_id = {value.semantic_rectangle_id: value for value in records}
    assert all(
        by_id[value.recipient_id].key == by_id[value.donor_id].key
        for value in result.assignments
    )


def test_rejects_malformed_bundle() -> None:
    malformed = TargetBundleDescriptor(
        semantic_rectangle_id=_digest("bad"),
        key=_key(),
        terminal_packet_sha256s=(_digest("one"),) * 4,
        transaction_sha256s=(_digest("two"),) * 4,
        answer_labels=(0,) * 15,
    )
    with pytest.raises(ControlError, match="answer label vector"):
        build_binding_derangement((malformed,), fold=2)
