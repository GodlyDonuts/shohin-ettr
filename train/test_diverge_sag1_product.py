"""CPU integration tests for counterfactual advantage-gated reasoning."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from diverge_sag1_product import SAG1ProductModel, frozen_parameter_sha256
from hf_product_reasoning_train import ProductReasoningModel


class _TinyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 12)
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(12, 12), nn.SiLU()) for _ in range(2)]
        )

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        use_cache=False,
    ):
        del attention_mask, use_cache
        hidden = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def _backbone() -> nn.Module:
    backbone = nn.Module()
    backbone.model = _TinyTextModel()
    backbone.lm_head = nn.Linear(12, 32, bias=False)
    backbone.config = SimpleNamespace(hidden_size=12)
    return backbone


def _base_checkpoint(path: Path) -> str:
    model = ProductReasoningModel(
        _backbone(),
        "baseline",
        lora_layers=1,
        lora_rank=2,
        lora_alpha=4.0,
        workspace_width=8,
        workspace_slots=2,
        recurrent_steps=2,
        dense_width=8,
        unfreeze_layers=0,
    )
    trainable = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(
        {
            "schema": "shohin-hf-product-reasoning-checkpoint-v1",
            "update": 1,
            "trainable_state": trainable,
            "optimizer": {},
            "metadata": {
                "arm": "baseline",
                "model_revision": "tiny-revision",
                "lora_layers": 1,
                "lora_rank": 2,
                "lora_alpha": 4.0,
            },
        },
        path,
    )
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(checkpoint: Path, checkpoint_sha256: str) -> SAG1ProductModel:
    return SAG1ProductModel(
        _backbone(),
        base_checkpoint=checkpoint,
        base_checkpoint_sha256=checkpoint_sha256,
        model_revision="tiny-revision",
        lora_layers=1,
        lora_rank=2,
        lora_alpha=4.0,
        workspace_width=8,
        source_slots=2,
        query_slots=2,
        recurrent_steps=2,
        attention_heads=2,
        ff_multiplier=2,
        pointer_temperature=0.5,
        binding_weight=0.05,
        coverage_weight=0.02,
        reset_weight=0.02,
        router_hidden=4,
        advantage_margin=0.02,
        router_threshold=0.5,
        router_weight=0.2,
        risk_weight=0.5,
        sparsity_weight=0.01,
    )


class DivergeSAG1ProductTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026080711)
        self.temporary = TemporaryDirectory()
        self.checkpoint = Path(self.temporary.name) / "base.pt"
        self.checkpoint_sha256 = _base_checkpoint(self.checkpoint)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_base_adapter_is_loaded_and_frozen(self) -> None:
        model = _model(self.checkpoint, self.checkpoint_sha256)
        lora_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if ".lora_" in name
        ]
        self.assertTrue(lora_parameters)
        self.assertTrue(all(not parameter.requires_grad for parameter in lora_parameters))
        self.assertTrue(any(parameter.requires_grad for parameter in model.router.parameters()))

    def test_initial_router_abstention_is_exact_base_identity(self) -> None:
        model = _model(self.checkpoint, self.checkpoint_sha256).eval()
        ids = torch.tensor([[1, 2, 3], [0, 4, 5]])
        mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
        original = model.text_model.embed_tokens(ids)
        selected, selected_mask = model.generation_embeddings(ids, mask)
        self.assertTrue(torch.equal(selected_mask, mask))
        self.assertTrue(torch.equal(selected, original))

    def test_training_reaches_expert_and_router_but_not_base(self) -> None:
        model = _model(self.checkpoint, self.checkpoint_sha256)
        before = frozen_parameter_sha256(model)
        loss, metrics = model.forward_batch(
            [[1, 2, 3], [4, 5]],
            [[6, 7, 8], [9, 10]],
            pad_token_id=0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(metrics["base_language_loss"], metrics["language_loss"])
        loss.backward()
        self.assertIsNotNone(model.workspace.output_projection.weight.grad)
        self.assertIsNotNone(model.router[-1].bias.grad)
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if ".lora_" in name
            )
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        optimizer.step()
        self.assertEqual(before, frozen_parameter_sha256(model))

    def test_forced_expert_commit_changes_embeddings(self) -> None:
        model = _model(self.checkpoint, self.checkpoint_sha256).eval()
        nn.init.normal_(model.workspace.output_projection.weight, std=0.05)
        nn.init.constant_(model.router[-1].bias, 5.0)
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)
        original = model.text_model.embed_tokens(ids)
        selected, _ = model.generation_embeddings(ids, mask)
        self.assertFalse(torch.equal(selected, original))


if __name__ == "__main__":
    unittest.main()
