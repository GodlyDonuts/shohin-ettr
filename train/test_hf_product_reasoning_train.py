"""CPU mechanics tests for matched product-reasoning training."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn

from hf_product_reasoning_train import (
    LoRALinear,
    ProductReasoningTrainError,
    install_lora,
    pack_training_embeddings,
    reservoir_rows,
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

    def test_pack_rejects_batch_mismatch(self) -> None:
        with self.assertRaises(ProductReasoningTrainError):
            pack_training_embeddings(nn.Embedding(5, 2), [[1]], [], None, 0)


if __name__ == "__main__":
    unittest.main()
