"""CPU integration tests for the DIVERGE-QST1 product path."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from diverge_qst1_product import QST1ProductModel, frozen_parameter_sha256


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


def _model() -> QST1ProductModel:
    return QST1ProductModel(
        _backbone(),
        lora_layers=1,
        lora_rank=2,
        lora_alpha=4.0,
        workspace_width=8,
        source_slots=2,
        state_slots=2,
        query_slots=1,
        recurrent_steps=3,
        attention_heads=2,
        ff_multiplier=2,
        binding_temperature=0.10,
        binding_weight=0.05,
        reset_weight=0.02,
        halting_weight=0.01,
    )


class DivergeQST1ProductTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026080701)

    def test_batch_is_finite_and_reaches_lora_and_every_owner(self) -> None:
        model = _model()
        loss, metrics = model.forward_batch(
            [[1, 2, 3], [4, 5]],
            [[6, 7, 8], [9, 10]],
            pad_token_id=0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["logical_charged_tokens"], 5.0)
        loss.backward()
        for parameter in (
            model.workspace.source_queries,
            model.workspace.state_seed,
            model.workspace.query_seed,
        ):
            self.assertIsNotNone(parameter.grad)
        lora_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "lora_" in name
        ]
        self.assertTrue(any(gradient is not None for gradient in lora_gradients))

    def test_generation_appends_one_complete_transaction_packet(self) -> None:
        model = _model().eval()
        ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        normal, normal_mask = model.generation_embeddings(ids, mask)
        self.assertEqual(normal.shape, (2, 8, 12))
        self.assertEqual(normal_mask.shape, (2, 8))
        model.set_control("packet_swap")
        swapped, _ = model.generation_embeddings(ids, mask)
        self.assertFalse(torch.equal(normal[:, 3:], swapped[:, 3:]))
        model.set_control("state_reset")
        reset, _ = model.generation_embeddings(ids, mask)
        self.assertFalse(torch.equal(normal[:, 3:], reset[:, 3:]))

    def test_source_packet_is_not_overwritten_by_transition(self) -> None:
        model = _model()
        output = model._workspace_output([[1, 2, 3]], pad_token_id=0)
        source_before = output.source_packet.detach().clone()
        model.workspace._transition(output.source_packet, output.initial_state)
        self.assertTrue(torch.equal(source_before, output.source_packet.detach()))
        self.assertEqual(output.cumulative_halt.shape, (1, 3))
        self.assertTrue(
            torch.all(output.cumulative_halt[:, 1:] >= output.cumulative_halt[:, :-1])
        )

    def test_optimizer_cannot_change_protected_parameter_hash(self) -> None:
        model = _model()
        before = frozen_parameter_sha256(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        loss, _ = model.forward_batch([[1, 2]], [[3, 4]], pad_token_id=0)
        loss.backward()
        optimizer.step()
        self.assertEqual(before, frozen_parameter_sha256(model))


if __name__ == "__main__":
    unittest.main()
