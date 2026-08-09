"""CPU mechanics tests for matched product-reasoning training."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
import torch.nn as nn

from hf_product_reasoning_train import (
    BoundedLoRATopKRouter,
    LoRALinear,
    ProductReasoningModel,
    ProductReasoningTrainError,
    error_syndrome_residual_loss,
    install_lora,
    install_scoped_lora,
    load_trainable_checkpoint,
    pack_training_embeddings,
    product_generation_embeddings,
    render_reasoning_messages,
    resolve_product_backbone_layout,
    reservoir_rows,
    reservoir_rows_with_sha256,
    validate_warm_start_metadata,
)


class ProductReasoningTrainTests(unittest.TestCase):
    @staticmethod
    def _topk_router() -> nn.Module:
        class Router(nn.Module):
            hidden_dim = 4
            num_experts = 3
            top_k = 2
            norm_topk_prob = True

            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 4))

            def forward(self, hidden_states: torch.Tensor):
                flattened = hidden_states.reshape(-1, self.hidden_dim)
                logits = torch.nn.functional.linear(flattened, self.weight)
                probabilities = torch.softmax(logits.float(), dim=-1)
                scores, indices = torch.topk(probabilities, self.top_k, dim=-1)
                scores = scores / scores.sum(dim=-1, keepdim=True)
                return logits, scores.to(logits.dtype), indices

        return Router()

    def test_plain_reasoning_envelope_supports_base_tokenizer(self) -> None:
        tokenizer = SimpleNamespace(chat_template=None)
        rendered = render_reasoning_messages(
            tokenizer,
            [
                {"role": "system", "content": "be precise"},
                {"role": "user", "content": "2+2?"},
            ],
            enable_thinking=False,
        )
        self.assertEqual(
            rendered,
            "System: be precise\n\nUser: 2+2?\n\nAssistant:",
        )

    def test_native_reasoning_template_remains_authoritative(self) -> None:
        class Tokenizer:
            chat_template = "native"

            def apply_chat_template(self, messages, **kwargs):
                self.call = (messages, kwargs)
                return "native-render"

        tokenizer = Tokenizer()
        messages = [{"role": "user", "content": "question"}]
        rendered = render_reasoning_messages(
            tokenizer,
            messages,
            enable_thinking=True,
        )
        self.assertEqual(rendered, "native-render")
        self.assertEqual(tokenizer.call[0], messages)
        self.assertTrue(tokenizer.call[1]["enable_thinking"])

    @staticmethod
    def _text_backbone(multimodal: bool) -> nn.Module:
        text = nn.Module()
        text.embed_tokens = nn.Embedding(20, 8)
        text.layers = nn.ModuleList([nn.Sequential(nn.Linear(8, 8))])
        backbone = nn.Module()
        backbone.lm_head = nn.Linear(8, 20, bias=False)
        if multimodal:
            wrapper = nn.Module()
            wrapper.language_model = text
            backbone.model = wrapper
            backbone.config = SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=8)
            )
        else:
            backbone.model = text
            backbone.config = SimpleNamespace(hidden_size=8)
        return backbone

    def test_backbone_layout_supports_multimodal_qwen_text_path(self) -> None:
        text, head, width, layout = resolve_product_backbone_layout(
            self._text_backbone(multimodal=True)
        )
        self.assertTrue(hasattr(text, "embed_tokens"))
        self.assertEqual(head.in_features, 8)
        self.assertEqual(width, 8)
        self.assertEqual(layout, "multimodal-language-model")

    def test_backbone_layout_supports_standard_causal_lm(self) -> None:
        text, head, width, layout = resolve_product_backbone_layout(
            self._text_backbone(multimodal=False)
        )
        self.assertTrue(hasattr(text, "layers"))
        self.assertEqual(head.out_features, 20)
        self.assertEqual(width, 8)
        self.assertEqual(layout, "causal-language-model")

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

    def test_token_mixer_scope_leaves_moe_path_frozen(self) -> None:
        layer = nn.Module()
        layer.self_attn = nn.Sequential(nn.Linear(4, 4))
        layer.mlp = nn.Sequential(nn.Linear(4, 4))
        self.assertEqual(
            install_scoped_lora(layer, "token_mixer", rank=2, alpha=2.0), 1
        )
        self.assertIsInstance(layer.self_attn[0], LoRALinear)
        self.assertIsInstance(layer.mlp[0], nn.Linear)

    def test_router_lora_starts_as_exact_frozen_router(self) -> None:
        torch.manual_seed(41)
        base = self._topk_router()
        inputs = torch.randn(2, 3, 4)
        expected = base(inputs)
        router = BoundedLoRATopKRouter(base, rank=2, alpha=4.0)
        observed = router(inputs)
        for actual, reference in zip(observed, expected, strict=True):
            self.assertTrue(torch.equal(actual, reference))

        observed[1].sum().backward()
        self.assertIsNone(router.base.weight.grad)
        self.assertIsNotNone(router.lora_b.weight.grad)
        self.assertEqual(
            [name for name, parameter in router.named_parameters() if parameter.requires_grad],
            ["lora_a.weight", "lora_b.weight"],
        )

    def test_router_lora_residual_is_bounded(self) -> None:
        base = self._topk_router()
        router = BoundedLoRATopKRouter(base, rank=2, alpha=4.0)
        with torch.no_grad():
            router.lora_a.weight.fill_(100.0)
            router.lora_b.weight.fill_(100.0)
        inputs = torch.ones(5, 4)
        base_logits = base(inputs)[0]
        router_logits = router(inputs)[0]
        self.assertLessEqual(
            float((router_logits - base_logits).abs().max().detach()),
            router.scale + 1e-6,
        )

    def test_router_scope_leaves_experts_and_attention_frozen(self) -> None:
        layer = nn.Module()
        layer.self_attn = nn.Sequential(nn.Linear(4, 4))
        layer.mlp = nn.Module()
        layer.mlp.gate = self._topk_router()
        layer.mlp.experts = nn.Sequential(nn.Linear(4, 4))
        layer.requires_grad_(False)
        self.assertEqual(install_scoped_lora(layer, "router", 2, 4.0), 1)
        self.assertIsInstance(layer.mlp.gate, BoundedLoRATopKRouter)
        self.assertFalse(layer.self_attn[0].weight.requires_grad)
        self.assertFalse(layer.mlp.experts[0].weight.requires_grad)

    def test_last_layer_unfreeze_is_explicit_and_local(self) -> None:
        frozen = ProductReasoningModel(
            self._text_backbone(multimodal=False),
            "baseline",
            1,
            2,
            4.0,
            8,
            2,
            1,
            unfreeze_layers=0,
        )
        unfrozen = ProductReasoningModel(
            self._text_backbone(multimodal=False),
            "baseline",
            1,
            2,
            4.0,
            8,
            2,
            1,
            unfreeze_layers=1,
        )

        frozen_projection = frozen.text_model.layers[0][0]
        unfrozen_projection = unfrozen.text_model.layers[0][0]
        self.assertIsInstance(frozen_projection, LoRALinear)
        self.assertIsInstance(unfrozen_projection, LoRALinear)
        self.assertFalse(frozen_projection.base.weight.requires_grad)
        self.assertTrue(unfrozen_projection.base.weight.requires_grad)
        self.assertGreater(
            unfrozen.trainable_parameter_count(), frozen.trainable_parameter_count()
        )

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

    def test_pack_adds_workspace_residual_without_extending_sequence(self) -> None:
        embedding = nn.Embedding(20, 6)
        residual = torch.ones(2, 3, 6)
        packed, attention, labels, charged = pack_training_embeddings(
            embedding,
            [[1, 2], [3]],
            [[4, 5], [6, 7, 8]],
            None,
            pad_token_id=0,
            prompt_residuals=residual,
        )
        self.assertEqual(packed.shape, (2, 4, 6))
        self.assertEqual(attention.sum(dim=1).tolist(), [4, 4])
        self.assertEqual(labels[0].tolist(), [-100, -100, 4, 5])
        self.assertTrue(torch.allclose(packed[0, :2], embedding(torch.tensor([1, 2])) + 1))
        self.assertEqual(charged, 5)

    def test_pack_preserves_geometry_while_masking_draft_keys(self) -> None:
        embedding = nn.Embedding(20, 6)
        packed, attention, labels, charged = pack_training_embeddings(
            embedding,
            [[1, 2, 3, 4]],
            [[5, 6]],
            None,
            pad_token_id=0,
            prompt_attention_rows=[[1, 0, 0, 1]],
        )
        self.assertEqual(packed.shape, (1, 6, 6))
        self.assertEqual(attention.tolist(), [[1, 0, 0, 1, 1, 1]])
        self.assertEqual(labels.tolist(), [[-100, -100, -100, -100, 5, 6]])
        self.assertEqual(charged, 2)

    def test_syndrome_loss_accepts_exact_verified_residual_direction(self) -> None:
        embedding = nn.Embedding(4, 2)
        with torch.no_grad():
            embedding.weight.zero_()
            embedding.weight[1] = torch.tensor([1.0, 0.0])
            embedding.weight[2] = torch.tensor([0.0, 1.0])
        workspace = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
        loss = error_syndrome_residual_loss(
            embedding,
            [[0, 1]],
            [[2, 3]],
            [[0, 1]],
            workspace,
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_syndrome_loss_rejects_missing_draft(self) -> None:
        with self.assertRaises(ProductReasoningTrainError):
            error_syndrome_residual_loss(
                nn.Embedding(4, 2),
                [[0, 1]],
                [[2, 3]],
                [[0, 0]],
                torch.zeros(1, 2, 2),
            )

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

    def test_warm_start_metadata_requires_same_trainable_geometry(self) -> None:
        args = SimpleNamespace(
            arm="baseline",
            model_root=Path("/model"),
            model_source_root=Path("/canonical-model"),
            model_revision="revision",
            lora_layers=4,
            lora_rank=8,
            lora_alpha=16.0,
            unfreeze_layers=2,
            lora_scope="all",
            quantization="none",
        )
        metadata = {
            "arm": "baseline",
            "model_root": "/canonical-model",
            "model_revision": "revision",
            "lora_layers": 4,
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "unfreeze_layers": 2,
            "lora_scope": "all",
            "quantization": "none",
        }
        validate_warm_start_metadata(metadata, args)
        metadata["unfreeze_layers"] = 1
        with self.assertRaises(ProductReasoningTrainError):
            validate_warm_start_metadata(metadata, args)


if __name__ == "__main__":
    unittest.main()
