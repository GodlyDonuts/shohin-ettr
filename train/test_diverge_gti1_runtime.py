#!/usr/bin/env python3
"""Focused structural tests for DIVERGE-GTI1."""

from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from diverge_gti1_runtime import (
    GTI1Config,
    GenerativeTransactionInterpreter,
    adapter_state_sha256,
    canonical_query_text,
    expected_transaction,
    frozen_backbone_state_sha256,
    render_prompt,
    transaction_token_ids,
)
from model import GPT, GPTConfig


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "READ": 1, "alpha": 2, "beta": 3},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def _record() -> dict[str, object]:
    return {
        "source_text": "Return cedar's value; oak's value is not requested.",
        "symbols": ["cedar", "oak", "unused"],
        "symbol_role_ids": [0, 1],
        "mode": "sensitive",
        "renderer": 5,
    }


def _model() -> GenerativeTransactionInterpreter:
    backbone = GPT(
        GPTConfig(
            vocab_size=16,
            n_layer=4,
            n_head=2,
            n_kv_head=1,
            d_model=16,
            d_ff=32,
            seq_len=128,
            qk_norm=False,
        )
    )
    return GenerativeTransactionInterpreter(backbone, _tokenizer(), GTI1Config())


def test_canonical_transaction_protocol() -> None:
    record = _record()
    assert canonical_query_text(record) == (
        "Return alpha's value; beta's value is not requested."
    )
    assert canonical_query_text(record, control="swap_mentions") == (
        "Return beta's value; alpha's value is not requested."
    )
    assert canonical_query_text(record, control="scrub_context") == "alpha then beta"
    assert render_prompt(record).endswith("\nTransaction:")
    assert expected_transaction(record) == 0
    assert len(transaction_token_ids(_tokenizer())[0]) == len(
        transaction_token_ids(_tokenizer())[1]
    )


def test_lora_preserves_frozen_backbone_and_receives_gradients() -> None:
    torch.manual_seed(7)
    model = _model()
    before = frozen_backbone_state_sha256(model.backbone)
    adapter_before = adapter_state_sha256(model)
    loss = model.supervised_loss([_record()], [0], device=torch.device("cpu"))
    loss.backward()
    parameters = list(model.adapter_parameters())
    assert parameters
    assert any(parameter.grad is not None for parameter in parameters)
    with torch.no_grad():
        parameters[0].add_(0.5)
    assert frozen_backbone_state_sha256(model.backbone) == before
    assert adapter_state_sha256(model) != adapter_before


def test_candidate_decoder_emits_one_complete_legal_transaction() -> None:
    torch.manual_seed(11)
    model = _model().eval()
    scores = model.candidate_scores(
        [_record()], device=torch.device("cpu"), batch_size=2
    )
    assert scores.shape == (1, 2)
    assert torch.isfinite(scores).all()
    assert int(scores.argmax(-1)) in (0, 1)


if __name__ == "__main__":
    test_canonical_transaction_protocol()
    test_lora_preserves_frozen_backbone_and_receives_gradients()
    test_candidate_decoder_emits_one_complete_legal_transaction()
    print("DIVERGE-GTI1 runtime tests passed")
