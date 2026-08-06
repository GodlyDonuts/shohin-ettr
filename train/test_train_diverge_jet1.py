#!/usr/bin/env python3
"""Focused trainer-contract tests for DIVERGE-JET1."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from diverge_jet1_data import generate_jet1_episode
from diverge_jet1_runtime import JET1Config
from diverge_mei1_data import EVIDENCE_COHORTS
from train_diverge_jet1 import (
    EVAL_DEPTHS,
    QwenJET1,
    build_gate,
    cosine_scale,
    tensorize_episodes,
    training_loss,
)


class _DummyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(8, 8), nn.GELU()) for _ in range(2)]
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _DummyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _DummyTextModel()
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.config = SimpleNamespace(hidden_size=8)


class _DummyTokenizer:
    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        del kwargs
        width = 6
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        mask = torch.ones_like(ids)
        for row, text in enumerate(texts):
            ids[row] = torch.tensor(
                [(len(text) + row + offset) % 31 + 1 for offset in range(width)]
            )
        return {"input_ids": ids, "attention_mask": mask}


def test_tensorization_and_joint_backward() -> None:
    episodes = [
        generate_jet1_episode(seed=100 + index, cohort="train", depth=2)
        for index in range(2)
    ]
    batch = tensorize_episodes(episodes, torch.device("cpu"))
    assert batch.program_actions.shape == (2, 2, 2, 2)
    assert batch.evidence_targets.shape == (2, 2, 10)
    assert len(batch.words) == 4
    model = QwenJET1(
        _DummyBackbone(),
        lora_layers=1,
        lora_rank=2,
        lora_alpha=4.0,
        trajectory_config=JET1Config(
            input_width=8,
            reader_width=8,
            reader_heads=2,
        ),
    )
    loss, metrics, tokens = training_loss(
        model,
        _DummyTokenizer(),
        batch,
        maximum_tokens=16,
        operator_seed=17,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert tokens == 24
    assert metrics["loss"] > 0
    assert any(parameter.grad is not None for parameter in model.lora_parameters())
    assert model.trajectory.evidence.input_projection.weight.grad is not None
    assert model.trajectory.executor.route_logits.grad is not None
    assert model.trajectory.query_route_logits.grad is not None


def test_schedule_and_gate_contract() -> None:
    assert cosine_scale(1, 100, 10) == 0.1
    assert cosine_scale(10, 100, 10) == 1.0
    assert abs(cosine_scale(100, 100, 10) - 0.1) < 1e-12
    passing = {
        f"{cohort}/depth{depth}": {
            "evidence_pair_exact": 0.96,
            "choice_sequence_exact": 0.95,
            "terminal_state_exact": 0.95,
            "answer_exact": 0.95,
            "wrong_prior_recovery": 0.95,
            "evidence_shuffle_answer_exact": 0.50,
            "state_reset_answer_exact": 0.50,
            "invalid_episodes": 0,
        }
        for cohort in EVIDENCE_COHORTS
        for depth in EVAL_DEPTHS
    }
    primitive = {
        "complete_state_exact": 0.999,
        "invalid_examples": 0,
    }
    gate = build_gate(
        passing,
        primitive,
        frozen_before="same",
        frozen_after="same",
        source_audit_pass=True,
    )
    assert gate["pass"]
    passing["renderer_shift/depth24"]["evidence_pair_exact"] = 0.94
    assert not build_gate(
        passing,
        primitive,
        frozen_before="same",
        frozen_after="same",
        source_audit_pass=True,
    )["pass"]


def main() -> None:
    test_tensorization_and_joint_backward()
    test_schedule_and_gate_contract()
    print("DIVERGE-JET1 trainer tests passed", flush=True)


if __name__ == "__main__":
    main()
