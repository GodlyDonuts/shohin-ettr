import unittest

import torch

from byte_tape_compiler import ByteTapeCompiler, byte_batch, byte_loss, load_byte_program


class ByteTapeCompilerTest(unittest.TestCase):
    def test_forward(self):
        row = {
            "schema": "shohin-btt1-byte-supervision-v1",
            "identity_sha256": "a" * 64,
            "family": "chain_sum",
            "question": "2 + 3",
            "byte_roles": ["NUM_BEGIN", "IGNORE", "ADD", "IGNORE", "NUM_BEGIN"],
            "gold_actions": [],
        }
        batch = byte_batch([load_byte_program(row)], torch.device("cpu"))
        model = ByteTapeCompiler(width=32, encoder_layers=1, heads=4)
        output = model(batch["byte_ids"], batch["mask"])
        self.assertEqual(tuple(output.role_logits.shape), (1, 5, 10))
        self.assertTrue(torch.isfinite(byte_loss(output, batch["role"])))


if __name__ == "__main__":
    unittest.main()
