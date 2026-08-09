import unittest

import torch
import torch.nn as nn

from shared_post_mlp_revision import (
    SharedPostMLPConfig,
    SharedPostMLPResidual,
)


class SharedPostMLPResidualTest(unittest.TestCase):
    def test_zero_initialized_residual_preserves_base(self) -> None:
        base = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        config = SharedPostMLPConfig(hidden_size=8, controlled_layers=1, rank=2, alpha=2)
        block = SharedPostMLPResidual(base, config)
        inputs = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        expected = base(inputs)
        actual = block(inputs)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(sum(p.numel() for p in block.parameters() if p.requires_grad), 32)
        self.assertFalse(block.base.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
