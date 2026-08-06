from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from diverge_vcr1_product import (
    VCR1ProductModel,
    load_vcr1_checkpoint,
    save_vcr1_checkpoint,
)
from hf_product_reasoning_train import ProductReasoningModel, _save_checkpoint


class FakeBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return states + torch.tanh(self.projection(states))


class FakeTextModel(nn.Module):
    def __init__(self, vocab: int, width: int, layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, width)
        self.layers = nn.ModuleList(FakeBlock(width) for _ in range(layers))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ):
        states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        for layer in self.layers:
            states = layer(states)
        return SimpleNamespace(last_hidden_state=states)


class FakeBackbone(nn.Module):
    def __init__(self, vocab: int = 32, width: int = 12, layers: int = 4) -> None:
        super().__init__()
        self.model = FakeTextModel(vocab, width, layers)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.config = SimpleNamespace(hidden_size=width)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_checkpoint(path: Path) -> None:
    torch.manual_seed(3)
    source = ProductReasoningModel(
        FakeBackbone(),
        arm="baseline",
        lora_layers=4,
        lora_rank=8,
        lora_alpha=16.0,
        workspace_width=16,
        workspace_slots=4,
        recurrent_steps=2,
        unfreeze_layers=2,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in source.parameters() if parameter.requires_grad]
    )
    metadata = {
        "arm": "baseline",
        "model_revision": "fake-revision",
        "lora_layers": 4,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "unfreeze_layers": 2,
    }
    _save_checkpoint(path, source, optimizer, 400, metadata)


def _model(tmp_path: Path, *, role_blind: bool = False) -> VCR1ProductModel:
    checkpoint = tmp_path / "source.pt"
    if not checkpoint.exists():
        _source_checkpoint(checkpoint)
    return VCR1ProductModel(
        FakeBackbone(),
        checkpoint,
        source_checkpoint_sha256=_sha256(checkpoint),
        source_revision="fake-revision",
        role_blind=role_blind,
        workspace_width=16,
        workspace_slots=3,
        recurrent_steps=2,
        attention_heads=4,
        ff_multiplier=2,
    )


def test_vcr1_forward_updates_only_reactor(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = _model(tmp_path)
    assert all(not parameter.requires_grad for parameter in model.source.parameters())
    before = model.frozen_source_sha256()
    loss, metrics = model.forward_batch(
        [[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]],
        [[8, 9, 0], [8, 9, 0]],
        [[True, True, False, False, False]] * 2,
        [[False, False, True, True, True]] * 2,
        [False, True],
        0,
    )
    loss.backward()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    optimizer.step()
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["validity_accuracy"] <= 1.0
    assert before == model.frozen_source_sha256()
    assert all(parameter.grad is None for parameter in model.source.parameters())


def test_vcr1_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = _model(tmp_path)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    checkpoint = tmp_path / "vcr.pt"
    save_vcr1_checkpoint(checkpoint, model, optimizer, 17, {"arm": "vcr1"})
    expected = {
        name: parameter.detach().clone()
        for name, parameter in model.reactor.named_parameters()
    }
    with torch.no_grad():
        for parameter in model.reactor.parameters():
            parameter.zero_()
    update, metadata = load_vcr1_checkpoint(checkpoint, model)
    assert update == 17 and metadata == {"arm": "vcr1"}
    assert all(
        torch.equal(expected[name], parameter)
        for name, parameter in model.reactor.named_parameters()
    )


def test_role_blind_model_has_identical_parameter_contract(tmp_path: Path) -> None:
    torch.manual_seed(3)
    treatment = _model(tmp_path, role_blind=False)
    torch.manual_seed(3)
    control = _model(tmp_path, role_blind=True)
    assert treatment.trainable_parameter_count() == control.trainable_parameter_count()
    assert set(treatment.reactor.state_dict()) == set(control.reactor.state_dict())
