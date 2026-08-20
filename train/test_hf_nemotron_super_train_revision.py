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


def _mechanics_authorization() -> dict[str, object]:
    return {
        "schema": "shohin-nemotron-super-two-h100-mechanics-v1",
        "status": "pass",
        "score_rows_read": 0,
        "benchmark_rows_read": 0,
        "model_revision": training.MODEL_REVISION,
        "trainable_parameters": training.TRAINABLE_PARAMETERS_PER_ROLE,
        "native_router_expert_trainables": 0,
        "serialization_restore_exact": True,
        "modelopt_fp8": {"valid": True},
        "training_objective_receipt": {
            "objective": "response_only_next_token_cross_entropy",
            "prompt_tokens": 3,
            "response_tokens": 2,
            "ignore_index": -100,
            "gradient_accumulation_scale": training.GRADIENT_ACCUMULATION,
            "learning_rate": training.LEARNING_RATE,
            "autocast_dtype": "torch.bfloat16",
        },
    }


def test_training_authorization_binds_mechanics_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training,
        "modelopt_fp8_receipt_is_exact",
        lambda value: value == {"valid": True},
    )
    payload = _mechanics_authorization()
    training.validate_mechanics_authorization(payload)
    receipt = payload["training_objective_receipt"]
    assert isinstance(receipt, dict)
    receipt["gradient_accumulation_scale"] = 1
    with pytest.raises(training.NemotronSuperTrainingError):
        training.validate_mechanics_authorization(payload)
    payload = _mechanics_authorization()
    payload.pop("training_objective_receipt")
    with pytest.raises(training.NemotronSuperTrainingError):
        training.validate_mechanics_authorization(payload)


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
