"""CPU tests for the first upward-MoE revision training boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import hf_nemotron_super_train_revision as training


def _row(index: int) -> dict[str, object]:
    identity = hashlib.sha256(str(index).encode()).hexdigest()
    return {
        "schema": training.DATA_SCHEMA,
        "identity_sha256": identity,
        "internal_draft_visible": True,
        "external_candidate_text_visible": False,
        "runtime_fields": ["question"],
        "question": f"question {index}",
        "response": f"response {index}",
    }


def test_load_revision_rows_binds_hash_schema_and_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "revision.jsonl"
    rows = [_row(index) for index in range(4)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(training, "DATA_PRESENTATIONS", 4)
    monkeypatch.setattr(training, "DATA_SHA256", training.sha256_file(path))
    assert training.load_revision_rows(path) == rows
    rows[0]["internal_draft_visible"] = False
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(training, "DATA_SHA256", training.sha256_file(path))
    with pytest.raises(training.NemotronSuperTrainingError):
        training.load_revision_rows(path)


def test_consumption_digest_binds_order_and_exact_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_row(index) for index in range(6)]
    monkeypatch.setattr(training, "DATA_PRESENTATIONS", 6)
    monkeypatch.setattr(training, "CONSUMED_PRESENTATIONS", 4)
    first = training.consumed_identity_sha256(rows)
    rows[0], rows[1] = rows[1], rows[0]
    assert training.consumed_identity_sha256(rows) != first
    rows[4], rows[5] = rows[5], rows[4]
    assert training.consumed_identity_sha256(rows) != first


class _Tokenizer:
    eos_token_id = 9

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [len(value), 1]


def test_tokenization_receipt_is_complete_and_untruncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_row(index) for index in range(4)]
    monkeypatch.setattr(training, "DATA_PRESENTATIONS", 4)
    monkeypatch.setattr(training, "CONSUMED_PRESENTATIONS", 4)
    examples, receipt = training.tokenize_consumed_rows(_Tokenizer(), rows)
    assert len(examples) == 4
    assert receipt["consumed_presentations"] == 4
    assert receipt["truncated_rows"] == 0
    assert receipt["maximum_observed_tokens"] == 5
