#!/usr/bin/env python3
"""Focused tests for deterministic native role-state SFT mechanics."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from model import GPT, GPTConfig
from native_role_lora import ROLE_ADAPTER_SCHEMA
from native_role_sft import (
    TRAINING_REPORT_SCHEMA,
    atomic_json,
    atomic_torch_save,
    cosine_multiplier,
    deterministic_batch_indices,
    main,
)


def test_deterministic_batches_are_seeded_full_and_in_range():
    first = list(deterministic_batch_indices(11, 4, 7, 123))
    second = list(deterministic_batch_indices(11, 4, 7, 123))
    assert len(first) == 7
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
        assert left.shape == (4,)
        assert len(set(left.tolist())) == 4
        assert int(left.min()) >= 0
        assert int(left.max()) < 11


def test_deterministic_batches_reject_invalid_geometry():
    with pytest.raises(ValueError, match="smaller"):
        list(deterministic_batch_indices(3, 4, 1, 1))
    with pytest.raises(ValueError, match="positive"):
        list(deterministic_batch_indices(8, 4, 0, 1))


def test_cosine_schedule_has_expected_boundaries():
    assert cosine_multiplier(0, 100, 10) == 0.0
    assert cosine_multiplier(10, 100, 10) == 1.0
    assert cosine_multiplier(100, 100, 10) == pytest.approx(0.1)
    assert 0.1 < cosine_multiplier(50, 100, 10) < 1.0


def test_atomic_publishers_replace_only_their_temporary_file(tmp_path: Path):
    tensor_path = tmp_path / "payload.pt"
    json_path = tmp_path / "report.json"
    atomic_torch_save(tensor_path, {"value": torch.tensor([1, 2, 3])})
    atomic_json(json_path, {"status": "complete"})
    loaded = torch.load(tensor_path, weights_only=True)
    torch.testing.assert_close(loaded["value"], torch.tensor([1, 2, 3]))
    assert json_path.read_text() == '{\n  "status": "complete"\n}\n'
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "payload.pt",
        "report.json",
    ]


def test_tiny_end_to_end_role_training_publishes_bound_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab={
                "<unk>": 0,
                "<|endoftext|>": 1,
                "Question": 2,
                ":": 3,
                "Answer": 4,
                "add": 5,
                "one": 6,
                "two": 7,
                "three": 8,
                "four": 9,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    config = GPTConfig(
        vocab_size=10,
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        d_model=16,
        d_ff=32,
        seq_len=16,
        zloss=0.0,
    )
    checkpoint_path = tmp_path / "base.pt"
    torch.save(
        {"cfg": vars(config), "model": GPT(config).state_dict(), "step": 12},
        checkpoint_path,
    )
    data_path = tmp_path / "data.jsonl"
    rows = [
        {"question": "add one two", "response": "three"},
        {"question": "add two two", "response": "four"},
        {"question": "add one one", "response": "two"},
        {"question": "add two one", "response": "three"},
    ]
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "trained"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_role_sft.py",
            "--init",
            str(checkpoint_path),
            "--data",
            str(data_path),
            "--tokenizer",
            str(tokenizer_path),
            "--out",
            str(output),
            "--role",
            "revision",
            "--adapter-layers",
            "1",
            "--adapter-rank",
            "2",
            "--adapter-alpha",
            "4",
            "--pack-len",
            "16",
            "--batch-size",
            "1",
            "--exact-updates",
            "2",
            "--warmup",
            "0",
            "--log-every",
            "1",
        ],
    )
    main()

    report = json.loads((output / "training_report.json").read_text())
    payload = torch.load(output / "revision_adapter.pt", weights_only=True)
    assert report["schema"] == TRAINING_REPORT_SCHEMA
    assert report["status"] == "complete"
    assert report["optimization"]["updates"] == 2
    assert report["adapter"]["trainable_parameters"] > 0
    assert payload["schema"] == ROLE_ADAPTER_SCHEMA
    assert payload["role"] == "revision"
    assert payload["base_checkpoint_sha256"] == report["base"]["sha256"]
