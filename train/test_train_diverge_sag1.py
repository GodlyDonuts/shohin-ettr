from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from hf_product_reasoning_train import _save_checkpoint
from train_diverge_sag1 import load_sag1_training_checkpoint


class _TinySAG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))


def _metadata() -> dict[str, object]:
    return {
        "architecture": "diverge-sag1",
        "arm": "diverge_sag1",
        "model_root": "/model",
        "model_revision": "revision",
        "data_sha256": "data",
        "selected_rows": 10,
        "seed": 11,
        "data_seed": 12,
        "lora_layers": 4,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "workspace_config": {"width": 8},
        "workspace_architecture_sha256": "architecture",
        "base_checkpoint_sha256": "base",
    }


def test_resume_restores_trainable_parameters_and_optimizer(tmp_path: Path) -> None:
    source = _TinySAG()
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    source.weight.grad = torch.tensor([0.5, -0.5])
    optimizer.step()
    expected = source.weight.detach().clone()
    checkpoint = tmp_path / "checkpoint_0000064.pt"
    _save_checkpoint(checkpoint, source, optimizer, 64, _metadata())

    restored = _TinySAG()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    update, metadata = load_sag1_training_checkpoint(
        checkpoint, restored, restored_optimizer, _metadata()
    )
    assert update == 64
    assert metadata == _metadata()
    assert torch.equal(restored.weight, expected)
    assert restored_optimizer.state
