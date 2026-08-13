import unittest

import torch
import torch.nn as nn

from shared_post_mlp_revision import (
    SharedPostMLPConfig,
    SharedPostMLPResidual,
    trainable_state,
    trainable_state_sha256,
)


class SharedPostMLPResidualTest(unittest.TestCase):
    def test_zero_initialized_residual_preserves_base(self) -> None:
        base = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        config = SharedPostMLPConfig(
            hidden_size=8, controlled_layers=1, rank=2, alpha=2
        )
        block = SharedPostMLPResidual(base, config)
        inputs = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        expected = base(inputs)
        actual = block(inputs)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            sum(p.numel() for p in block.parameters() if p.requires_grad), 32
        )
        self.assertEqual(block.adapter_a.weight.dtype, torch.float32)
        self.assertEqual(block.adapter_b.weight.dtype, torch.float32)
        self.assertFalse(block.base.weight.requires_grad)

    def test_commit_scale_update_is_not_lost_to_bfloat16_rounding(self) -> None:
        value = torch.tensor(0.1, dtype=torch.float32)
        updated = value - 2e-6
        self.assertNotEqual(updated.item(), value.item())
        rounded = torch.tensor(0.1, dtype=torch.bfloat16)
        self.assertEqual((rounded - 2e-6).to(torch.bfloat16).item(), rounded.item())

    def test_trainable_state_hash_binds_names_dtypes_shapes_and_bytes(self) -> None:
        base = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        block = SharedPostMLPResidual(
            base,
            SharedPostMLPConfig(hidden_size=8, controlled_layers=1, rank=2, alpha=2),
        )
        state = trainable_state(block)
        digest = trainable_state_sha256(state)
        changed = {name: value.clone() for name, value in state.items()}
        changed["adapter_a.weight"][0, 0] += 1e-6
        self.assertNotEqual(trainable_state_sha256(changed), digest)


if __name__ == "__main__":
    unittest.main()
