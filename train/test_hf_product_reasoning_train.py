"""CPU mechanics tests for matched product-reasoning training."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
import torch.nn as nn

from hf_product_reasoning_train import (
    LoRALinear,
    ProductReasoningModel,
    ProductReasoningTrainError,
    install_lora,
    load_trainable_checkpoint,
    pack_training_embeddings,
    product_generation_embeddings,
    render_reasoning_messages,
    resolve_product_backbone_layout,
    reservoir_rows,
    reservoir_rows_with_sha256,
)


class ProductReasoningTrainTests(unittest.TestCase):
    def test_plain_reasoning_envelope_supports_base_tokenizer(self) -> None:
        tokenizer = SimpleNamespace(chat_template=None)
        rendered = render_reasoning_messages(
            tokenizer,
            [
                {"role": "system", "content": "be precise"},
                {"role": "user", "content": "2+2?"},
            ],
            enable_thinking=False,
        )
        self.assertEqual(
            rendered,
            "System: be precise\n\nUser: 2+2?\n\nAssistant:",
        )

    def test_native_reasoning_template_remains_authoritative(self) -> None:
        class Tokenizer:
            chat_template = "native"

            def apply_chat_template(self, messages, **kwargs):
                self.call = (messages, kwargs)
                return "native-render"

        tokenizer = Tokenizer()
        messages = [{"role": "user", "content": "question"}]
        rendered = render_reasoning_messages(
            tokenizer,
            messages,
            enable_thinking=True,
        )
        self.assertEqual(rendered, "native-render")
        self.assertEqual(tokenizer.call[0], messages)
        self.assertTrue(tokenizer.call[1]["enable_thinking"])

    @staticmethod
    def _text_backbone(multimodal: bool) -> nn.Module:
        text = nn.Module()
        text.embed_tokens = nn.Embedding(20, 8)
        text.layers = nn.ModuleList([nn.Sequential(nn.Linear(8, 8))])
        backbone = nn.Module()
        backbone.lm_head = nn.Linear(8, 20, bias=False)
        if multimodal:
            wrapper = nn.Module()
            wrapper.language_model = text
            backbone.model = wrapper
            backbone.config = SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=8)
            )
        else:
            backbone.model = text
            backbone.config = SimpleNamespace(hidden_size=8)
        return backbone

    def test_backbone_layout_supports_multimodal_qwen_text_path(self) -> None:
        text, head, width, layout = resolve_product_backbone_layout(
            self._text_backbone(multimodal=True)
        )
        self.assertTrue(hasattr(text, "embed_tokens"))
        self.assertEqual(head.in_features, 8)
        self.assertEqual(width, 8)
        self.assertEqual(layout, "multimodal-language-model")

    def test_backbone_layout_supports_standard_causal_lm(self) -> None:
        text, head, width, layout = resolve_product_backbone_layout(
            self._text_backbone(multimodal=False)
        )
        self.assertTrue(hasattr(text, "layers"))
        self.assertEqual(head.out_features, 20)
        self.assertEqual(width, 8)
        self.assertEqual(layout, "causal-language-model")

    def test_lora_starts_as_exact_frozen_projection(self) -> None:
        torch.manual_seed(31)
        base = nn.Linear(5, 7, bias=False)
        inputs = torch.randn(3, 5)
        expected = base(inputs).detach()
        layer = LoRALinear(base, rank=2, alpha=4.0)
        self.assertTrue(torch.equal(layer(inputs), expected))
        layer(inputs).sum().backward()
        self.assertIsNone(layer.base.weight.grad)
        self.assertIsNotNone(layer.lora_b.weight.grad)

    def test_install_lora_replaces_nested_linears(self) -> None:
        module = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.Linear(4, 4)))
        self.assertEqual(install_lora(module, rank=2, alpha=2.0), 2)
        self.assertIsInstance(module[0], LoRALinear)
        self.assertIsInstance(module[1][0], LoRALinear)

    def test_last_layer_unfreeze_is_explicit_and_local(self) -> None:
        frozen = ProductReasoningModel(
            self._text_backbone(multimodal=False),
            "baseline",
            1,
            2,
            4.0,
            8,
            2,
            1,
            unfreeze_layers=0,
        )
        unfrozen = ProductReasoningModel(
            self._text_backbone(multimodal=False),
            "baseline",
            1,
            2,
            4.0,
            8,
            2,
            1,
            unfreeze_layers=1,
        )

        frozen_projection = frozen.text_model.layers[0][0]
        unfrozen_projection = unfrozen.text_model.layers[0][0]
        self.assertIsInstance(frozen_projection, LoRALinear)
        self.assertIsInstance(unfrozen_projection, LoRALinear)
        self.assertFalse(frozen_projection.base.weight.requires_grad)
        self.assertTrue(unfrozen_projection.base.weight.requires_grad)
        self.assertGreater(
            unfrozen.trainable_parameter_count(), frozen.trainable_parameter_count()
        )

    def test_pack_masks_prompt_and_workspace(self) -> None:
        embedding = nn.Embedding(20, 6)
        prefix = torch.randn(2, 3, 6)
        packed, attention, labels, charged = pack_training_embeddings(
            embedding,
            [[1, 2], [3]],
            [[4, 5], [6, 7, 8]],
            prefix,
            pad_token_id=0,
        )
        self.assertEqual(packed.shape, (2, 7, 6))
        self.assertEqual(attention.sum(dim=1).tolist(), [7, 7])
        self.assertEqual(labels[0].tolist(), [-100, -100, -100, -100, -100, 4, 5])
        self.assertEqual(charged, 5)

    def test_pack_adds_workspace_residual_without_extending_sequence(self) -> None:
        embedding = nn.Embedding(20, 6)
        residual = torch.ones(2, 3, 6)
        packed, attention, labels, charged = pack_training_embeddings(
            embedding,
            [[1, 2], [3]],
            [[4, 5], [6, 7, 8]],
            None,
            pad_token_id=0,
            prompt_residuals=residual,
        )
        self.assertEqual(packed.shape, (2, 4, 6))
        self.assertEqual(attention.sum(dim=1).tolist(), [4, 4])
        self.assertEqual(labels[0].tolist(), [-100, -100, 4, 5])
        self.assertTrue(torch.allclose(packed[0, :2], embedding(torch.tensor([1, 2])) + 1))
        self.assertEqual(charged, 5)

    def test_reservoir_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                "".join(
                    f'{{"question":"q{index}","response":"r{index}"}}\n'
                    for index in range(20)
                ),
                encoding="utf-8",
            )
            self.assertEqual(reservoir_rows(path, 5, 31), reservoir_rows(path, 5, 31))
            _, digest = reservoir_rows_with_sha256(path, 5, 31)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_pack_rejects_batch_mismatch(self) -> None:
        with self.assertRaises(ProductReasoningTrainError):
            pack_training_embeddings(nn.Embedding(5, 2), [[1]], [], None, 0)

    def test_generation_embeddings_preserve_baseline_prompt(self) -> None:
        model = object.__new__(ProductReasoningModel)
        nn.Module.__init__(model)
        model.text_model = nn.Module()
        model.text_model.embed_tokens = nn.Embedding(20, 6)
        model.workspace = None
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)
        embeddings, attention = product_generation_embeddings(model, ids, mask)
        self.assertEqual(embeddings.shape, (1, 3, 6))
        self.assertTrue(torch.equal(attention, mask))

    def test_checkpoint_loader_rejects_parameter_mismatch(self) -> None:
        model = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "schema": "shohin-hf-product-reasoning-checkpoint-v1",
                    "update": 1,
                    "trainable_state": {"wrong": torch.zeros(1)},
                    "metadata": {},
                },
                path,
            )
            with self.assertRaises(ProductReasoningTrainError):
                load_trainable_checkpoint(path, model)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
