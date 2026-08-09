import pytest

from build_idr1_revision_data import TRAIN_SCHEMA
from build_vfr1_teacher_requests import (
    VFR1RequestError,
    collect_unique_requests,
    teacher_prompt,
)


def _source(identity: str, task: str = "math500") -> dict[str, str]:
    return {"identity_sha256": identity, "task": task, "question": "What is 1+1?", "answer": "\\boxed{2}"}


def _row(identity: str, presentation: int = 0, response: str = "\\boxed{2}") -> dict[str, object]:
    return {
        "schema": TRAIN_SCHEMA,
        "source_identity_sha256": identity,
        "outcome_class": "both_wrong",
        "target_kind": "source_verified_repair",
        "question": "source plus draft",
        "response": response,
        "presentation": presentation,
    }


def test_teacher_prompt_has_strict_two_block_contract() -> None:
    prompt = teacher_prompt("question", "\\boxed{2}", "math500")
    assert "<FAULT>" in prompt and "<REVISION>" in prompt
    assert "VERIFIED REFERENCE" in prompt
    assert "\\boxed{}" in prompt


def test_collect_requests_deduplicates_identical_presentations() -> None:
    identity = "a" * 64
    rows = [_row(identity, 0), _row(identity, 1)]
    requests = collect_unique_requests(rows, {identity: _source(identity)})
    assert len(requests) == 1
    assert requests[0]["identity_sha256"] == identity


def test_collect_requests_rejects_disagreeing_presentations() -> None:
    identity = "b" * 64
    rows = [_row(identity, 0), _row(identity, 1, "\\boxed{3}")]
    with pytest.raises(VFR1RequestError, match="presentations disagree"):
        collect_unique_requests(rows, {identity: _source(identity)})


def test_collect_requests_accepts_train_subset_of_full_source_bank() -> None:
    train_identity = "c" * 64
    heldout_identity = "d" * 64
    requests = collect_unique_requests(
        [_row(train_identity)],
        {
            train_identity: _source(train_identity),
            heldout_identity: _source(heldout_identity),
        },
    )
    assert [row["identity_sha256"] for row in requests] == [train_identity]
