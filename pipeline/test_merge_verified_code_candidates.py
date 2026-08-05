import pytest

from merge_verified_code_candidates import (
    VerifiedCodeCandidateMergeError,
    merge_arms,
)


def _row(identity: str, sample: int, task: str = "humaneval") -> dict:
    return {
        "identity_sha256": identity,
        "task": task,
        "sample_index": sample,
        "completion": "pass",
        "correct": False,
    }


def test_merge_arms_preserves_first_arm_as_anchor() -> None:
    merged, report = merge_arms(
        [
            ("greedy", [_row("a", 0), _row("b", 0)]),
            (
                "sampled",
                [_row("a", 0), _row("a", 1), _row("b", 0), _row("b", 1)],
            ),
        ]
    )
    assert report["samples_per_identity"] == 3
    assert [row["sample_index"] for row in merged if row["identity_sha256"] == "a"] == [
        0,
        1,
        2,
    ]
    assert merged[0]["candidate_arm"] == "greedy"
    assert merged[1]["candidate_arm"] == "sampled"


def test_merge_arms_rejects_identity_mismatch() -> None:
    with pytest.raises(VerifiedCodeCandidateMergeError, match="identities differ"):
        merge_arms([("one", [_row("a", 0)]), ("two", [_row("b", 0)])])
