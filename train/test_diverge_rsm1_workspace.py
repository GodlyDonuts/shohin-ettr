"""Focused shape and discrete-feedback tests for RSM1."""

from __future__ import annotations

import unittest

import torch

from diverge_rsm1_workspace import (
    PersistentReplayConfig,
    PersistentStateReplay,
    RSM1WorkspaceError,
)


class RSM1WorkspaceTest(unittest.TestCase):
    def _inputs(self):
        config = PersistentReplayConfig(
            backbone_width=16,
            state_width=16,
            state_slots=8,
            packet_slots=3,
            max_trace_steps=4,
            attention_heads=4,
            ff_multiplier=2,
        )
        packet = torch.randn(2, 3, 16)
        memory = torch.randn(2, 12, 16)
        attention = torch.ones(2, 12, dtype=torch.bool)
        problem = torch.zeros(2, 12, dtype=torch.bool)
        problem[:, :2] = True
        steps = torch.zeros(2, 4, 12, dtype=torch.bool)
        steps[0, 0, 2:4] = True
        steps[0, 1, 4:6] = True
        steps[0, 2, 6:8] = True
        steps[1, 0, 2:4] = True
        steps[1, 1, 4:6] = True
        selected = torch.tensor([2, 0], dtype=torch.long)
        return config, packet, memory, attention, problem, steps, selected

    def test_forward_uses_hard_bounded_state(self) -> None:
        config, packet, memory, attention, problem, steps, selected = self._inputs()
        model = PersistentStateReplay(config)
        output = model(packet, memory, attention, problem, steps, selected)
        self.assertEqual(output.initial_logits.shape, (2, 8, config.state_vocab_size))
        self.assertEqual(
            output.transition_logits.shape,
            (2, 4, 8, config.state_vocab_size),
        )
        self.assertEqual(output.state_trace_tokens.shape, (2, 5, 8))
        self.assertTrue(torch.equal(output.replay_active[0], torch.tensor([False, True, True, False])))
        self.assertFalse(output.replay_active[1].any())
        self.assertTrue(
            torch.equal(
                output.state_trace_tokens[1, 0],
                output.state_trace_tokens[1, -1],
            )
        )
        self.assertGreaterEqual(int(output.terminal_tokens.min()), 0)
        self.assertLess(int(output.terminal_tokens.max()), config.state_vocab_size)

        predecessor = torch.randint(0, config.state_vocab_size, (2, 4, 8))
        oracle = model.oracle_transition_logits(
            memory, attention, problem, steps, predecessor
        )
        self.assertEqual(oracle.shape, (2, 4, 8, config.state_vocab_size))

    def test_selection_outside_depth_fails_closed(self) -> None:
        config, packet, memory, attention, problem, steps, _ = self._inputs()
        model = PersistentStateReplay(config)
        with self.assertRaises(RSM1WorkspaceError):
            model(
                packet,
                memory,
                attention,
                problem,
                steps,
                torch.tensor([4, 3], dtype=torch.long),
            )

    def test_nondense_steps_fail_closed(self) -> None:
        config, packet, memory, attention, problem, steps, selected = self._inputs()
        steps[0, 1] = False
        model = PersistentStateReplay(config)
        with self.assertRaises(RSM1WorkspaceError):
            model(packet, memory, attention, problem, steps, selected)


if __name__ == "__main__":
    unittest.main()
