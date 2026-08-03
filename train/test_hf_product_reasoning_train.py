"""CPU mechanics tests for matched product-reasoning training."""

from __future__ import annotations

import hashlib
from pathlib import Path
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
    reservoir_rows,
    reservoir_rows_with_sha256,
)


class ProductReasoningTrainTests(unittest.TestCase):
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
