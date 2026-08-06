"""Focused integration checks for RSM1 loss and source-pass accounting."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from diverge_rsm1_product import RSM1ProductModel
from diverge_rsm1_workspace import PersistentReplayConfig, PersistentStateReplay


class StubRSM1Product(RSM1ProductModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.text_model = nn.Module()
        self.text_model.embed_tokens = nn.Embedding(8, 32)
        self.replay_config = PersistentReplayConfig(
            backbone_width=32,
            state_width=16,
            state_slots=4,
            packet_slots=2,
            max_trace_steps=3,
            attention_heads=4,
            ff_multiplier=2,
        )
        self.replay = PersistentStateReplay(self.replay_config)
        self.context_calls = 0

    def _frozen_context(self, *args, **kwargs):
        del kwargs
        self.context_calls += 1
        batch = len(args[0])
        device = self.text_model.embed_tokens.weight.device
        chosen = torch.tensor([2, 0], device=device, dtype=torch.long)[:batch]
        return (
            torch.randn(batch, 2, 32, device=device),
            torch.randn(batch, 5, 32, device=device),
            torch.ones(batch, 5, dtype=torch.bool, device=device),
            torch.ones(batch, 5, dtype=torch.bool, device=device),
            torch.ones(batch, 3, 5, dtype=torch.bool, device=device),
            torch.zeros(batch, 4, device=device),
            chosen,
        )


class RSM1ProductTest(unittest.TestCase):
    def test_forward_reuses_one_frozen_context(self) -> None:
        model = StubRSM1Product()
        initial = torch.full((2, 4), 2, dtype=torch.long)
        free_targets = torch.full((2, 3, 4), 2, dtype=torch.long)
        free_active = torch.tensor(
            [[False, True, True], [False, False, False]], dtype=torch.bool
        )
        oracle_active = torch.ones(2, 3, dtype=torch.bool)
        loss, metrics = model.forward_batch(
            prompt_rows=[[1, 2], [2, 3]],
            problem_masks=[[True, True], [True, True]],
            packet_step_masks=[[[True, True]] * 3] * 2,
            operation_masks=[[[True, True]] * 3] * 2,
            final_masks=[[True, True], [True, True]],
            selection_targets=[2, 0],
            initial_targets=initial,
            free_targets=free_targets,
            free_active=free_active,
            oracle_predecessors=free_targets,
            oracle_targets=free_targets,
            oracle_active=oracle_active,
            terminal_targets=initial,
            pad_token_id=0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.context_calls, 1)
        self.assertEqual(metrics["source_tokens"], 4)
        loss.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.replay.parameters())
        )


if __name__ == "__main__":
    unittest.main()
