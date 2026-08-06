"""CPU integration tests for the DIVERGE-VMT1 product path."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from diverge_vmt1_product import VMT1ProductModel, frozen_parameter_sha256


class _TinyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(48, 12)
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
    backbone.lm_head = nn.Linear(12, 48, bias=False)
    backbone.config = SimpleNamespace(hidden_size=12)
    return backbone


def _model() -> VMT1ProductModel:
    return VMT1ProductModel(
        _backbone(),
        lora_layers=1,
        lora_rank=2,
        lora_alpha=4.0,
        latent_width=8,
        trajectory_slots=2,
        recurrent_steps=2,
        attention_heads=2,
        ff_multiplier=2,
        assignment_temperature=0.1,
        validity_margin=1.0,
        trace_weight=1.0,
        validity_weight=0.25,
        halting_weight=0.01,
    )


class DivergeVMT1ProductTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026080602)

    def test_full_batch_is_finite_and_reaches_all_trainable_paths(self) -> None:
        model = _model()
        loss, metrics = model.forward_batch(
            [[1, 2, 3], [4, 5]],
            [[[6, 7, 8], [9, 10, 11]], [[12, 13], [14, 15, 16]]],
            [[True, False], [False, True]],
            pad_token_id=0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(metrics["selected_correct_response_nll_rows"]), 2)
        self.assertEqual(metrics["logical_charged_tokens"], 6.0)
        self.assertEqual(metrics["candidate_charged_tokens"], 12.0)
        self.assertEqual(metrics["trace_target_tokens"], 11.0)
        loss.backward()
        self.assertIsNotNone(model.workspace.factor_slots.grad)
        self.assertIsNotNone(model.validity_head.weight.grad)
        lora_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "lora_" in name
        ]
        self.assertTrue(any(gradient is not None for gradient in lora_gradients))

    def test_generation_uses_one_complete_validity_selected_prefix(self) -> None:
        model = _model().eval()
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)
        selected, selected_mask = model.generation_embeddings(ids, mask)
        self.assertEqual(selected.shape, (1, 5, 12))
        self.assertEqual(selected_mask.shape, (1, 5))
        model.set_selection_strategy("reset")
        reset, _ = model.generation_embeddings(ids, mask)
        prompt = model.text_model.embed_tokens(ids)
        self.assertTrue(torch.equal(reset[:, :3], prompt))
        self.assertEqual(int(torch.count_nonzero(reset[:, 3:])), 0)

    def test_optimizer_cannot_change_frozen_parameter_hash(self) -> None:
        model = _model()
        before = frozen_parameter_sha256(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        loss, _ = model.forward_batch(
            [[1, 2]],
            [[[3, 4], [5, 6]]],
            [[True, False]],
            pad_token_id=0,
        )
        loss.backward()
        optimizer.step()
        self.assertEqual(before, frozen_parameter_sha256(model))

    def test_pair_contract_fails_closed(self) -> None:
        model = _model()
        with self.assertRaisesRegex(RuntimeError, "one verified-correct"):
            model.forward_batch(
                [[1, 2]],
                [[[3, 4], [5, 6]]],
                [[True, True]],
                pad_token_id=0,
            )


if __name__ == "__main__":
    unittest.main()
