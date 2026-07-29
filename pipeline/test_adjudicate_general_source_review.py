import pytest

from pipeline.adjudicate_general_source_review import (
    AdjudicationError,
    SCORE_FIELDS,
    build_receipt,
)


def reviews(count=100):
    return [
        {
            "dataset": "test/source",
            "config": "default",
            "stable_identity_sha256": f"{index:064x}",
            "document_sha256": f"{index + 1000:064x}",
        }
        for index in range(count)
    ]


def labels(count=100, authority="human"):
    return [
        {
            "stable_identity_sha256": f"{index:064x}",
            "decision": "accept_core" if index % 2 else "reject",
            "adjudication_authority": authority,
            **{field: 4 for field in SCORE_FIELDS},
            "reason_codes": ["clear_explanation"],
        }
        for index in range(count)
    ]


def test_complete_human_review_can_satisfy_semantic_gate():
    receipt = build_receipt(
        reviews(),
        labels(),
        review_sha256="a" * 64,
        labels_sha256="b" * 64,
        required_rows=100,
    )
    assert receipt["training_admission_eligible"]
    assert receipt["contains_document_text"] is False
    assert receipt["decision_counts"] == {"reject": 50, "accept_core": 50}


def test_model_review_remains_preliminary():
    receipt = build_receipt(
        reviews(),
        labels(authority="model_preliminary"),
        review_sha256="a" * 64,
        labels_sha256="b" * 64,
        required_rows=100,
    )
    assert not receipt["training_admission_eligible"]
    assert receipt["admission_status"] == "preliminary_not_training_admission"


def test_metadata_only_review_cannot_satisfy_semantic_admission():
    packet = reviews()
    packet[0]["document_sha256"] = None
    with pytest.raises(AdjudicationError, match="nonempty text document"):
        build_receipt(
            packet,
            labels(),
            review_sha256="a" * 64,
            labels_sha256="b" * 64,
            required_rows=100,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[:-1],
        lambda rows: [*rows[:-1], {**rows[-1], "correctness": 6}],
        lambda rows: [*rows[:-1], {**rows[-1], "decision": "maybe"}],
        lambda rows: [*rows[:-1], {**rows[-1], "reason_codes": []}],
    ],
)
def test_invalid_or_incomplete_labels_fail_closed(mutate):
    with pytest.raises(AdjudicationError):
        build_receipt(
            reviews(),
            mutate(labels()),
            review_sha256="a" * 64,
            labels_sha256="b" * 64,
            required_rows=100,
        )
