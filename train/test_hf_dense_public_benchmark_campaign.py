from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_dense_public_benchmark_campaign import (
    batch_can_accept,
    batch_token_cost,
    DenseBenchmarkGenerationError,
    LEDGER_SCHEMA,
    MANIFEST_SCHEMA,
    QUESTION_SCHEMA,
    load_ledger,
    load_manifest,
    stage_prompt,
)


def write_lines(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def question(identity: str, benchmark: str = "mmlu_pro") -> dict:
    return {
        "schema": QUESTION_SCHEMA,
        "id": identity,
        "benchmark": benchmark,
        "upstream_id": identity[:4],
        "question": "Question?",
        "response_mode": "general",
    }


def test_manifest_preserves_global_order_and_variable_token_limits(tmp_path: Path) -> None:
    first = "a" * 64
    second = "b" * 64
    board_a = tmp_path / "a.jsonl"
    board_b = tmp_path / "b.jsonl"
    write_lines(board_a, [question(first)])
    write_lines(board_b, [question(second, "ifeval")])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "benchmarks": [
                    {"name": "mmlu_pro", "questions": str(board_a), "rows": 1, "max_new_tokens": 8},
                    {"name": "ifeval", "questions": str(board_b), "rows": 1, "max_new_tokens": 16},
                ],
            }
        )
    )
    _, rows = load_manifest(manifest)
    assert [row["id"] for row in rows] == [first, second]
    assert [row["max_new_tokens"] for row in rows] == [8, 16]


def test_ledger_must_be_an_exact_prefix(tmp_path: Path) -> None:
    path = tmp_path / "draft.jsonl"
    write_lines(
        path,
        [
            {
                "schema": LEDGER_SCHEMA,
                "stage": "draft",
                "id": "b" * 64,
                "prompt_sha256": "x",
                "completion": "answer",
            }
        ],
    )
    with pytest.raises(DenseBenchmarkGenerationError, match="resume ledger differs"):
        load_ledger(path, "draft", ["a" * 64, "b" * 64])


def test_second_pass_prompt_is_bound_to_its_own_draft() -> None:
    identity = "a" * 64
    row = {
        "id": identity,
        "question": "Original",
        "response_mode": "general",
    }
    prompt = stage_prompt(
        "trained_revision", row, {identity: {"completion": "Draft answer"}}
    )
    assert "Original request:\nOriginal" in prompt
    assert "Internal draft:\nDraft answer" in prompt


def test_direct_base_and_draft_use_the_identical_source_prompt() -> None:
    row = {"id": "a" * 64, "question": "Original", "response_mode": "general"}
    assert stage_prompt("direct_base", row, {}) == stage_prompt("draft", row, {})


def test_padded_batch_token_cost_uses_longest_prompt() -> None:
    assert batch_token_cost([100, 200, 150], 50) == 750


def test_long_context_batch_budget_shrinks_without_dropping_first_row() -> None:
    assert batch_can_accept([], 120_000, 1024, 65_536)
    assert not batch_can_accept([20_000], 50_000, 1024, 65_536)
    assert not batch_can_accept([20_000], 32_000, 1024, 65_536)
    assert batch_can_accept([1_000], 1_100, 1024, 65_536)
