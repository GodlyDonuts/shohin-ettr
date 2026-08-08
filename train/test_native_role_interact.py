#!/usr/bin/env python3
"""Tests for native Shohin draft/revision interaction mechanics."""

import json
from pathlib import Path
import sys

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from model import GPT, GPTConfig
from native_role_lora import (
    NativeRoleLoRAConfig,
    attach_role_lora,
    export_role_adapter,
)
from native_role_interact import (
    INTERACTION_REPORT_SCHEMA,
    load_jsonl,
    main,
    render_draft_prompt,
    render_revision_prompt,
)
from native_role_sft import sha256_file


def test_default_prompts_preserve_question_and_draft():
    row = {"id": "x", "question": "What is 6 times 7?"}
    draft_prompt = render_draft_prompt(row)
    revision_prompt = render_revision_prompt(row, "The answer may be 41.")
    assert "What is 6 times 7?" in draft_prompt
    assert "What is 6 times 7?" in revision_prompt
    assert "The answer may be 41." in revision_prompt
    assert revision_prompt.endswith("Final answer:")


def test_explicit_revision_prompt_requires_exactly_one_draft_slot():
    with pytest.raises(ValueError, match=r"one \{draft\}"):
        render_revision_prompt({"revision_prompt": "no slot"}, "x")
    with pytest.raises(ValueError, match=r"one \{draft\}"):
        render_revision_prompt(
            {"revision_prompt": "{draft} and {draft}"}, "x"
        )
    assert (
        render_revision_prompt({"revision_prompt": "Review: {draft}"}, "candidate")
        == "Review: candidate"
    )


def test_prompt_board_loader_is_ordered_and_nonempty(tmp_path: Path):
    board = tmp_path / "board.jsonl"
    board.write_text(
        json.dumps({"id": "a", "question": "one"})
        + "\n\n"
        + json.dumps({"id": "b", "question": "two"})
        + "\n"
    )
    assert [row["id"] for row in load_jsonl(board)] == ["a", "b"]
    board.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        load_jsonl(board)


def test_tiny_end_to_end_two_role_interaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab={
                "<unk>": 0,
                "<|endoftext|>": 1,
                "Question": 2,
                ":": 3,
                "one": 4,
                "Answer": 5,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config = GPTConfig(
        vocab_size=6,
        n_layer=1,
        n_head=2,
        n_kv_head=1,
        d_model=16,
        d_ff=32,
        seq_len=32,
        zloss=0.0,
    )
    model = GPT(config)
    checkpoint_path = tmp_path / "base.pt"
    torch.save(
        {"cfg": vars(config), "model": model.state_dict(), "step": 4},
        checkpoint_path,
    )
    base_sha256 = sha256_file(checkpoint_path)
    adapter_config = NativeRoleLoRAConfig(layers=1, rank=2, alpha=4)
    attach_role_lora(model, adapter_config)
    draft_path = tmp_path / "draft.pt"
    revision_path = tmp_path / "revision.pt"
    torch.save(
        export_role_adapter(
            model,
            adapter_config,
            "draft",
            base_checkpoint_sha256=base_sha256,
        ),
        draft_path,
    )
    torch.save(
        export_role_adapter(
            model,
            adapter_config,
            "revision",
            base_checkpoint_sha256=base_sha256,
        ),
        revision_path,
    )
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps(
            {
                "id": "tiny",
                "question": "one",
                "draft_prompt": "Question : one Answer :",
                "revision_prompt": "Question : one {draft} Answer :",
            }
        )
        + "\n"
    )
    output = tmp_path / "interaction.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_role_interact.py",
            "--init",
            str(checkpoint_path),
            "--draft-adapter",
            str(draft_path),
            "--revision-adapter",
            str(revision_path),
            "--tokenizer",
            str(tokenizer_path),
            "--prompts",
            str(prompts),
            "--out",
            str(output),
            "--max-new-draft",
            "1",
            "--max-new-revision",
            "1",
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert report["schema"] == INTERACTION_REPORT_SCHEMA
    assert report["status"] == "complete"
    assert len(report["transcripts"]) == 1
    assert report["transcripts"][0]["id"] == "tiny"
