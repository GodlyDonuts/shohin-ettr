import hashlib
import json
from pathlib import Path
import random

import pytest

from pipeline.profile_general_source import (
    ProfileError,
    buffered_shuffle,
    flatten_nested_files,
    iter_local_jsonl,
    profile_rows,
    review_excerpt,
    text_metrics,
    verify_local_profile_file,
)


def empty_eval_index():
    return {"exact": set(), "ngrams": set(), "files": []}


def test_text_metrics_detect_repetition_and_boilerplate():
    metrics = text_metrics("Accept cookies\nsame line\nsame line\n")
    assert metrics["boilerplate_markers"] == 1
    assert metrics["max_line_repeat_fraction"] == 2 / 3
    assert metrics["unique_line_fraction"] == 2 / 3
    assert metrics["control_fraction"] == 0


def test_review_excerpt_preserves_both_document_ends():
    excerpt, truncated = review_excerpt("A" * 100 + "B" * 100, 100)
    assert truncated
    assert excerpt.startswith("A")
    assert excerpt.endswith("B")
    short, short_truncated = review_excerpt("complete", 100)
    assert short == "complete"
    assert not short_truncated


def test_profile_is_deterministic_and_reports_duplicates_and_overlap():
    eval_text = "this evaluation phrase contains thirteen distinct words for a precise overlap test now"
    rows = [
        {
            "id": f"id-{index}",
            "url": f"https://domain{index % 2}.example/doc/{index}",
            "language": "en",
            "int_score": 4 + index % 2,
            "text": eval_text if index == 0 else f"educational document number {index}",
        }
        for index in range(20)
    ]
    rows[3]["text"] = rows[2]["text"]
    eval_index = {
        "exact": {eval_text},
        "ngrams": {
            "this evaluation phrase contains thirteen distinct words for a precise overlap test now"
        },
        "files": ["eval.jsonl"],
    }
    report, reviews = profile_rows(
        rows,
        dataset="test/source",
        config="default",
        text_field="text",
        scan_rows=20,
        review_rows=5,
        max_review_chars=100,
        eval_index=eval_index,
    )
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    shuffled_report, shuffled_reviews = profile_rows(
        shuffled,
        dataset="test/source",
        config="default",
        text_field="text",
        scan_rows=20,
        review_rows=5,
        max_review_chars=100,
        eval_index=eval_index,
    )
    assert report["scanned_rows"] == 20
    assert report["exact_duplicate_text_rows"] == 1
    assert report["sample_eval_overlap"]["exact_prompt_rows"] == 1
    assert report["sample_eval_overlap"]["eval_13gram_rows_bounded_head_tail"] == 1
    receipts = report["sample_eval_overlap"]["hashed_overlap_receipts"]
    assert len(receipts) == 1
    assert receipts[0]["document_sha256"]
    assert receipts[0]["matched_13gram_sha256"]
    assert {row["stable_identity_sha256"] for row in reviews} == {
        row["stable_identity_sha256"] for row in shuffled_reviews
    }
    assert shuffled_report["exact_duplicate_text_rows"] == 1


def test_profile_supports_metadata_only_sources():
    report, reviews = profile_rows(
        [{"blob_id": "abc", "detected_licenses": ["MIT"], "language": "Python"}],
        dataset="test/code",
        config="Python",
        text_field="text",
        scan_rows=1,
        review_rows=1,
        max_review_chars=100,
        eval_index=empty_eval_index(),
    )
    assert report["text_rows"] == 0
    assert report["metadata_only_rows"] == 1
    assert report["license_values"] == {'["MIT"]': 1}
    assert reviews[0]["review_text"] == ""


def test_flatten_nested_files_preserves_repo_and_file_provenance():
    flattened = list(
        flatten_nested_files(
            [
                {
                    "repo_path": "owner/repo",
                    "commit_id": "deadbeef",
                    "files": [
                        {
                            "content_id": "abc",
                            "content": "print('hello')",
                            "file_path": "main.py",
                            "license_type": "permissive",
                        }
                    ],
                }
            ],
            files_field="files",
            nested_text_field="content",
            max_files_per_record=8,
        )
    )
    assert flattened == [
        {
            "repo_path": "owner/repo",
            "repo_name": "owner/repo",
            "commit_id": "deadbeef",
            "content_id": "abc",
            "text": "print('hello')",
            "file_path": "main.py",
            "license_type": "permissive",
        }
    ]


def test_nested_review_context_is_private_and_hash_bound():
    flattened = list(
        flatten_nested_files(
            [
                {
                    "id": "source-1",
                    "text": "Original source facts.",
                    "rollout_results": [
                        {"text": "Generated transformation.", "finish_reason": "stop"}
                    ],
                }
            ],
            files_field="rollout_results",
            nested_text_field="text",
            max_files_per_record=1,
            parent_review_context_field="text",
        )
    )
    report, reviews = profile_rows(
        flattened,
        dataset="test/synthetic",
        config="faq",
        text_field="text",
        scan_rows=1,
        review_rows=1,
        max_review_chars=100,
        eval_index=empty_eval_index(),
        review_context_field="_review_context_text",
    )
    assert reviews[0]["review_text"] == "Generated transformation."
    assert reviews[0]["review_context_text"] == "Original source facts."
    assert len(reviews[0]["review_context_sha256"]) == 64
    assert report["review_packet_contains_context_text"]
    assert "review_context_text" not in report
    assert report["categorical_values"]["finish_reason"] == {"stop": 1}


def test_flatten_nested_files_hash_samples_each_repository():
    rows = [
        {
            "repo_path": "owner/repo",
            "files": [
                {"content_id": str(index), "content": str(index)}
                for index in range(20)
            ],
        }
    ]
    first = list(
        flatten_nested_files(
            rows,
            files_field="files",
            nested_text_field="content",
            max_files_per_record=4,
        )
    )
    second = list(
        flatten_nested_files(
            rows,
            files_field="files",
            nested_text_field="content",
            max_files_per_record=4,
        )
    )
    assert len(first) == 4
    assert [row["content_id"] for row in first] == [
        row["content_id"] for row in second
    ]


def test_profile_reports_numeric_quality_quantiles():
    report, _ = profile_rows(
        [
            {"id": "a", "text": "alpha beta gamma", "score": 1.5},
            {"id": "b", "text": "delta epsilon zeta", "score": 3.5},
        ],
        dataset="test/source",
        config="default",
        text_field="text",
        scan_rows=2,
        review_rows=1,
        max_review_chars=100,
        eval_index=empty_eval_index(),
    )
    assert report["quality_metric_quantiles"]["score"]["min"] == 1.5
    assert report["quality_metric_quantiles"]["score"]["max"] == 3.5


def test_profile_rejects_nonpositive_limits():
    try:
        profile_rows(
            [],
            dataset="test/source",
            config="default",
            text_field="text",
            scan_rows=0,
            review_rows=1,
            max_review_chars=100,
            eval_index=empty_eval_index(),
        )
    except ProfileError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("expected ProfileError")


def test_local_profile_input_is_hash_bound_read_only_and_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps({"id": index, "text": f"document {index}"}) + "\n"
            for index in range(20)
        )
    )
    source.chmod(0o444)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt = verify_local_profile_file(
        source,
        expected_sha256=expected,
    )
    assert receipt == {
        "bytes": source.stat().st_size,
        "name": "source.jsonl",
        "sha256": expected,
    }
    first = list(
        buffered_shuffle(
            iter_local_jsonl((source,)),
            seed=17,
            buffer_size=5,
        )
    )
    second = list(
        buffered_shuffle(
            iter_local_jsonl((source,)),
            seed=17,
            buffer_size=5,
        )
    )
    assert first == second
    assert sorted(row["id"] for row in first) == list(range(20))
    assert [row["id"] for row in first] != list(range(20))


def test_local_profile_input_rejects_mutable_linked_and_wrong_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"text":"document"}\n')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ProfileError, match="read-only"):
        verify_local_profile_file(
            source,
            expected_sha256=expected,
        )
    source.chmod(0o444)
    with pytest.raises(ProfileError, match="hash"):
        verify_local_profile_file(
            source,
            expected_sha256="0" * 64,
        )
    link = tmp_path / "link.jsonl"
    link.symlink_to(source)
    with pytest.raises(ProfileError, match="physical"):
        verify_local_profile_file(
            link,
            expected_sha256=expected,
        )
    hardlink = tmp_path / "hardlink.jsonl"
    hardlink.hardlink_to(source)
    with pytest.raises(ProfileError, match="single-link"):
        verify_local_profile_file(
            source,
            expected_sha256=expected,
        )
