from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import hf_upward_moe_train_owner as module
from upward_moe_temporal_gate import NEMOTRON_SPEC


def test_static_owner_contract_matches_surviving_recipe() -> None:
    contract = module.static_owner_contract()
    assert contract["role"] == "owner"
    assert contract["data_kind"] == "source_only"
    assert contract["selected_rows"] == 26387
    assert contract["updates"] == 256
    assert contract["gradient_accumulation"] == 16
    assert contract["consumed_presentations"] == 4096
    assert contract["max_sequence_length"] == 1024
    assert contract["learning_rate"] == 2e-4
    assert contract["external_proposer"] is False
    assert contract["native_router_expert_trainables"] == 0
    assert [row["host"] for row in contract["hosts"]] == [
        "Nemotron-Super-120B-A12B",
        "Mixtral-8x22B-141B-A39B",
    ]


def test_mechanics_must_be_score_free_exact_host_and_serializable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mechanics.json"
    payload = {
        "schema": "mechanics",
        "status": "pass",
        "score_rows_read": 0,
        "benchmark_rows_read": 0,
        "model_revision": NEMOTRON_SPEC.model_revision,
        "trainable_parameters": 2
        * len(NEMOTRON_SPEC.controlled_layer_indices)
        * NEMOTRON_SPEC.hidden_size
        * NEMOTRON_SPEC.rank,
        "native_router_expert_trainables": 0,
        "serialization_restore_exact": True,
    }
    path.write_text(json.dumps(payload))
    assert (
        module._validate_mechanics(
            path,
            schema="mechanics",
            spec=NEMOTRON_SPEC,
            expected_manifest_sha256=None,
        )["status"]
        == "pass"
    )
    for key, bad in (
        ("score_rows_read", 1),
        ("benchmark_rows_read", 1),
        ("native_router_expert_trainables", 1),
        ("serialization_restore_exact", False),
    ):
        changed = dict(payload)
        changed[key] = bad
        path.write_text(json.dumps(changed))
        with pytest.raises(module.UpwardMoEOwnerTrainingError):
            module._validate_mechanics(
                path,
                schema="mechanics",
                spec=NEMOTRON_SPEC,
                expected_manifest_sha256=None,
            )


def test_parse_args_requires_explicit_host_and_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "owner",
            "--host",
            "mixtral-8x22b",
            "--model-root",
            "/model",
            "--model-manifest",
            "/manifest",
            "--mechanics-report",
            "/mechanics",
            "--data",
            "/data",
            "--output",
            "/output",
        ],
    )
    args = module.parse_args()
    assert isinstance(args, argparse.Namespace)
    assert args.host == "mixtral-8x22b"
    assert args.output == Path("/output")


def test_owner_tokenization_is_source_only_untruncated_and_draft_free(
    monkeypatch,
) -> None:
    class Tokenizer:
        eos_token_id = 99

        @staticmethod
        def encode(text, add_special_tokens=False):
            assert add_special_tokens is False
            return [len(text), 7]

    monkeypatch.setattr(
        module,
        "render_reasoning_messages",
        lambda _tokenizer, messages, enable_thinking: messages[-1]["content"],
    )
    examples, receipt = module.tokenize_owner_rows(
        Tokenizer(),
        [{"question": "source question", "response": "source response"}],
    )
    assert examples == [([15, 7], [15, 7, 99], [1, 1])]
    assert receipt["source_only"] is True
    assert receipt["draft_masked_tokens"] == 0
    assert receipt["truncated_rows"] == 0


def test_owner_tokenization_rejects_sequence_over_budget(monkeypatch) -> None:
    class Tokenizer:
        eos_token_id = 99

        @staticmethod
        def encode(text, add_special_tokens=False):
            return [1] * (module.OWNER_MAX_SEQUENCE_LENGTH if text == "question" else 2)

    monkeypatch.setattr(
        module,
        "render_reasoning_messages",
        lambda _tokenizer, messages, enable_thinking: messages[-1]["content"],
    )
    with pytest.raises(module.UpwardMoEOwnerTrainingError):
        module.tokenize_owner_rows(
            Tokenizer(), [{"question": "question", "response": "response"}]
        )
