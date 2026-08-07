"""CPU integration tests for the DIVERGE-QPT1 product path."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from diverge_qpt1_product import QPT1ProductModel, frozen_parameter_sha256


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


def _model() -> QPT1ProductModel:
    return QPT1ProductModel(
        _backbone(),
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
    )


class DivergeQPT1ProductTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026080702)

    def test_batch_is_finite_and_reaches_pointer_transaction_and_lora(self) -> None:
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
            model.workspace.source_seeds,
            model.workspace.step_identities,
            model.workspace.query_seeds,
        ):
            self.assertIsNotNone(parameter.grad)
        lora_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "lora_" in name
        ]
        self.assertTrue(any(gradient is not None for gradient in lora_gradients))

    def test_pointers_are_hard_and_transactions_preserve_slot_geometry(self) -> None:
        model = _model().eval()
        output = model._workspace_output([[1, 2, 3, 4]], pad_token_id=0)
        self.assertEqual(output.source_assignments.shape, (1, 2, 4))
        self.assertEqual(output.transaction_reads.shape, (1, 2, 2))
        self.assertEqual(output.transaction_writes.shape, (1, 2, 2))
        for assignments in (
            output.source_assignments,
            output.query_assignments,
            output.transaction_reads,
            output.transaction_writes,
        ):
            self.assertTrue(
                torch.equal(
                    assignments.sum(dim=-1), torch.ones_like(assignments[..., 0])
                )
            )
            self.assertTrue(torch.all((assignments == 0) | (assignments == 1)))

    def test_generation_keeps_sequence_geometry_and_release_off_is_identity(
        self,
    ) -> None:
        model = _model().eval()
        ids = torch.tensor([[1, 2, 3], [0, 4, 5]])
        mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
        original = model.text_model.embed_tokens(ids)
        normal, normal_mask = model.generation_embeddings(ids, mask)
        self.assertEqual(normal.shape, original.shape)
        self.assertTrue(torch.equal(normal_mask, mask))
        self.assertTrue(torch.equal(normal, original))

        nn.init.normal_(model.workspace.output_projection.weight, std=0.05)
        normal, _ = model.generation_embeddings(ids, mask)
        model.set_control("release_off")
        released_off, _ = model.generation_embeddings(ids, mask)
        self.assertFalse(torch.equal(normal, original))
        self.assertTrue(torch.equal(released_off, original))

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
