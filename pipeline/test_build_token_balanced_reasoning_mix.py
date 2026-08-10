from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.build_token_balanced_reasoning_mix import (
    _quality_rank,
    TokenBalancedMixError,
    build_token_balanced_mix,
    parse_weights,
)


def test_source_test_replay_has_verified_priority() -> None:
    assert _quality_rank({"verification": "execution_verified_source_tests"}) == 4
    assert _quality_rank({"verification": "execution_verified"}) == 3


class FakeTokenizer:
    name_or_path = "fake-tokenizer"

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return " ".join(message["content"] for message in messages)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


class FakeTokenizerWithoutTemplate:
    name_or_path = "fake-tokenizer-without-template"
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def _write(path: Path, group: str, count: int, response_words: int = 4) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "question": f"{group} question {index}",
                        "response": " ".join([group] * response_words),
                        "training_group": group,
                    }
                )
                + "\n"
            )


def test_token_balanced_mix_is_reproducible_and_near_exact(tmp_path: Path) -> None:
    math = tmp_path / "math.jsonl"
    code = tmp_path / "code.jsonl"
    _write(math, "math", 30, 4)
    _write(code, "code", 30, 9)
    weights = parse_weights("math=.5,code=.5")
    kwargs = dict(
        tokenizer=FakeTokenizer(),
        model_revision="test",
        weights=weights,
        total_target_tokens=100,
        max_sequence_length=64,
        workspace_slots=0,
        seed=31,
    )
    first = build_token_balanced_mix(
        [math, code],
        tmp_path / "first.jsonl",
        tmp_path / "first.report.json",
        **kwargs,
    )
    second = build_token_balanced_mix(
        [math, code],
        tmp_path / "second.jsonl",
        tmp_path / "second.report.json",
        **kwargs,
    )
    assert first["output_sha256"] == second["output_sha256"]
    assert first["duplicate_questions"] == 0
    assert first["response_truncated_rows"] == 0
    assert first["prompt_truncated_rows"] == 0
    assert "eval_overlap_filter" not in first
    for metrics in first["selected_groups"].values():
        assert metrics["charged_target_tokens"] >= metrics["target_charged_tokens"]


def test_token_balanced_mix_rejects_truncated_capacity(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write(source, "math", 4, 100)
    with pytest.raises(TokenBalancedMixError, match="below required"):
        build_token_balanced_mix(
            [source],
            tmp_path / "bad.jsonl",
            tmp_path / "bad.report.json",
            tokenizer=FakeTokenizer(),
            model_revision="test",
            weights=parse_weights("math=1"),
            total_target_tokens=10,
            max_sequence_length=32,
            workspace_slots=0,
            seed=31,
        )


def test_token_balanced_mix_prefers_verified_duplicate(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        json.dumps(
            {
                "question": "same question",
                "response": "unverified response",
                "training_group": "math",
            }
        )
        + "\n"
    )
    second.write_text(
        json.dumps(
            {
                "question": "same question",
                "response": "verified response",
                "training_group": "math",
                "verification": "expected_answer_match_v1",
            }
        )
        + "\n"
        + json.dumps(
            {
                "question": "another question",
                "response": "verified response",
                "training_group": "math",
                "verification": "expected_answer_match_v1",
            }
        )
        + "\n"
    )
    output = tmp_path / "out.jsonl"
    report = build_token_balanced_mix(
        [first, second],
        output,
        tmp_path / "report.json",
        tokenizer=FakeTokenizer(),
        model_revision="test",
        weights=parse_weights("math=1"),
        total_target_tokens=3,
        max_sequence_length=64,
        workspace_slots=0,
        seed=31,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    duplicate = next(row for row in rows if row["question"] == "same question")
    assert duplicate["response"] == "verified response"
    assert report["scan_counters"]["duplicate_replacements"] == 1


def test_token_balanced_mix_selects_verified_rows_before_unverified(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question": "unverified question",
                        "response": "code code code code",
                        "training_group": "code",
                    }
                ),
                json.dumps(
                    {
                        "question": "verified question",
                        "response": "code code code code",
                        "training_group": "code",
                        "verification": "execution_verified",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl"
    build_token_balanced_mix(
        [source],
        output,
        tmp_path / "report.json",
        tokenizer=FakeTokenizer(),
        model_revision="test",
        weights=parse_weights("code=1"),
        total_target_tokens=4,
        max_sequence_length=64,
        workspace_slots=0,
        seed=31,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["question"] for row in rows] == ["verified question"]


def test_token_balanced_mix_supports_tokenizer_without_chat_template(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    _write(source, "math", 4, 4)
    report = build_token_balanced_mix(
        [source],
        tmp_path / "output.jsonl",
        tmp_path / "report.json",
        tokenizer=FakeTokenizerWithoutTemplate(),
        model_revision="test",
        weights=parse_weights("math=1"),
        total_target_tokens=8,
        max_sequence_length=64,
        workspace_slots=0,
        seed=31,
    )
    assert report["selected_groups"]["math"]["charged_target_tokens"] >= 8


def _reference_row(question: str, *, assessor: bool = True) -> dict:
    if assessor:
        return {"assessor": {"question": question}}
    return {"internal_draft": {"question": question}}


def test_eval_filter_removes_exact_and_unique_ngram_and_backfills(
    tmp_path: Path,
) -> None:
    exact = "An exact held-out source question!"
    gram = "one two three four five six seven eight nine ten eleven twelve thirteen"
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "question": exact,
                    "response": "math math math math",
                    "training_group": "math",
                },
                {
                    "question": f"prefix {gram} suffix",
                    "response": "math math math math",
                    "training_group": "math",
                },
                {
                    "question": "distinct backfill question one",
                    "response": "math math math math",
                    "training_group": "math",
                },
                {
                    "question": "distinct backfill question two",
                    "response": "math math math math",
                    "training_group": "math",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reference = tmp_path / "development.jsonl"
    reference.write_text(
        "\n".join(
            [
                json.dumps(_reference_row("AN EXACT HELD OUT SOURCE QUESTION")),
                json.dumps(_reference_row(gram, assessor=False)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl"
    report = build_token_balanced_mix(
        [source],
        output,
        tmp_path / "report.json",
        tokenizer=FakeTokenizer(),
        model_revision="test",
        weights=parse_weights("math=1"),
        total_target_tokens=8,
        max_sequence_length=64,
        workspace_slots=0,
        seed=31,
        eval_references=[reference],
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["question"] for row in rows} == {
        "distinct backfill question one",
        "distinct backfill question two",
    }
    assert report["scan_counters"]["eval_exact_overlap_rejected"] == 1
    assert report["scan_counters"]["eval_unique_ngram_overlap_rejected"] == 1
    overlap = report["eval_overlap_filter"]
    assert overlap["ngram_size"] == 13
    assert overlap["references"][0]["sha256"]
    assert overlap["references"][0]["rows"] == 2
    assert report["selected_groups"]["math"]["charged_target_tokens"] >= 8


def test_repeated_eval_boilerplate_ngram_is_not_removed(tmp_path: Path) -> None:
    common = "one two three four five six seven eight nine ten eleven twelve thirteen"
    reference = tmp_path / "development.jsonl"
    reference.write_text(
        "\n".join(
            [
                json.dumps(_reference_row(f"{common} alpha")),
                json.dumps(_reference_row(f"{common} beta")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "question": f"prefix {common}",
                "response": "math math math math",
                "training_group": "math",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl"
    report = build_token_balanced_mix(
        [source],
        output,
        tmp_path / "report.json",
        tokenizer=FakeTokenizer(),
        model_revision="test",
        weights=parse_weights("math=1"),
        total_target_tokens=4,
        max_sequence_length=64,
        workspace_slots=0,
        seed=31,
        eval_references=[reference],
    )
    assert report["scan_counters"].get("eval_unique_ngram_overlap_rejected", 0) == 0
    assert report["selected_rows"] == 1
