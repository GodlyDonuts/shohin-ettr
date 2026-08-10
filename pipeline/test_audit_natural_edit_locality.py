import json

import pytest

from audit_natural_edit_locality import (
    EditLocalityAuditError,
    audit_pairs,
    edit_locality,
)


def _row(split, left, left_correct, right, right_correct, task="math500"):
    return {
        "split": split,
        "task": task,
        "outcome_class": "base-only" if left_correct else "expert-only",
        "candidates": [
            {"completion": left, "correct": left_correct},
            {"completion": right, "correct": right_correct},
        ],
    }


def test_edit_locality_counts_nonoverlapping_prefix_and_suffix():
    result = edit_locality("abcOLDxyz", "abcNEWxyz")
    assert result["common_prefix_characters"] == 3
    assert result["common_suffix_characters"] == 3
    assert result["single_splice_replacement_characters"] == 3
    assert result["single_splice_copy_fraction"] == pytest.approx(6 / 9)


def test_audit_scores_only_allowed_exactly_one_correct_pairs(tmp_path):
    path = tmp_path / "pairs.jsonl"
    rows = [
        _row("train", "bad", False, "good", True),
        _row("development", "same prefix bad", False, "same prefix good", True),
        _row("holdout", "secret bad", False, "secret good", True),
        _row("train", "bad one", False, "bad two", False),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = audit_pairs(path)

    assert report["holdout_scored"] is False
    assert report["row_counts_metadata_only"] == {
        "development": 1,
        "holdout": 1,
        "train": 2,
    }
    assert report["splits"]["train"]["scored_exactly_one_correct_pairs"] == 1
    assert report["splits"]["development"]["scored_exactly_one_correct_pairs"] == 1
    assert report["ignored_scored_split_outcomes"] == {"expert-only": 1}


def test_audit_rejects_holdout_as_allowed_split(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(_row("holdout", "bad", False, "good", True)) + "\n")
    with pytest.raises(EditLocalityAuditError, match="holdout scoring is forbidden"):
        audit_pairs(path, allowed_splits=("holdout",))
