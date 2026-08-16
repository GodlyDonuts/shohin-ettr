import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import lift_nemotron_super_adapter_to_ultra as lift


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    monkeypatch.setattr(lift, "SUPER_HIDDEN_SIZE", 4)
    monkeypatch.setattr(lift, "ULTRA_HIDDEN_SIZE", 6)
    monkeypatch.setattr(lift, "RANK", 2)
    monkeypatch.setattr(lift, "SUPER_LAYERS", (1, 3))
    monkeypatch.setattr(lift, "ULTRA_LAYERS", (2, 4))
    monkeypatch.setattr(lift, "CONTROLLED_LAYERS", 2)
    monkeypatch.setattr(lift, "ULTRA_MODEL_LAYERS", 5)

    generator = torch.Generator().manual_seed(17)
    super_anchor = F.normalize(torch.randn(16, 4, generator=generator), dim=-1)
    ultra_anchor = F.normalize(torch.randn(16, 6, generator=generator), dim=-1)
    kernel = ultra_anchor @ ultra_anchor.T
    kernel.diagonal().add_(0.1)
    factor = tmp_path / "factor.pt"
    torch.save(
        {
            "schema": "shohin-nemotron-super-ultra-transfer-basis-v1",
            "factor": {
                "anchor_ids": torch.arange(16),
                "super_anchor": super_anchor,
                "ultra_anchor": ultra_anchor,
                "ultra_kernel_cholesky": torch.linalg.cholesky(kernel),
            },
        },
        factor,
    )

    state = {}
    for layer in lift.SUPER_LAYERS:
        prefix = f"backbone.model.layers.{layer}.mixer"
        state[f"{prefix}.adapter_a.weight"] = torch.randn(2, 4, generator=generator)
        state[f"{prefix}.adapter_b.weight"] = torch.randn(4, 2, generator=generator)
    checkpoint = tmp_path / "super.pt"
    torch.save(
        {
            "schema": lift.SUPER_CHECKPOINT_SCHEMA,
            "update": 256,
            "trainable_state": state,
            "metadata": {
                "model_revision": lift.SUPER_MODEL_REVISION,
                "data_sha256": lift.DATA_SHA256,
                "native_router_expert_trainables": 0,
            },
        },
        checkpoint,
    )

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "architectures": ["NemotronHForCausalLM"],
                "model_type": "nemotron_h",
                "hidden_size": 6,
                "layers_block_type": ["mamba", "attention", "moe", "mamba", "moe"],
                "n_routed_experts": 512,
                "num_experts_per_tok": 22,
            }
        )
    )
    monkeypatch.setattr(lift, "ULTRA_CONFIG_SHA256", _sha(config))
    return argparse.Namespace(
        super_checkpoint=checkpoint,
        expected_super_checkpoint_sha256=_sha(checkpoint),
        factor=factor,
        expected_factor_sha256=_sha(factor),
        ultra_config=config,
        output_checkpoint=tmp_path / "out" / "checkpoint.pt",
        output_report=tmp_path / "out" / "report.json",
    )


def test_transfer_directions_preserves_norms():
    generator = torch.Generator().manual_seed(3)
    source = torch.randn(4, 5, generator=generator)
    super_anchor = F.normalize(torch.randn(16, 4, generator=generator), dim=-1)
    ultra_anchor = F.normalize(torch.randn(16, 6, generator=generator), dim=-1)
    kernel = ultra_anchor @ ultra_anchor.T
    kernel.diagonal().add_(0.1)
    mapped, correlations = lift.transfer_directions(
        source, super_anchor, ultra_anchor, torch.linalg.cholesky(kernel)
    )
    assert mapped.shape == (6, 5)
    assert torch.allclose(mapped.norm(dim=0), source.norm(dim=0), atol=1e-5)
    assert correlations.shape == (5,)
    assert torch.isfinite(correlations).all()


def test_run_emits_zero_label_ultra_checkpoint(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    report = lift.run(args)
    assert report["status"] == "complete"
    assert report["label_rows_read"] == 0
    assert report["benchmark_rows_read"] == 0
    assert report["optimizer_updates"] == 0
    assert report["model_weight_mutations"] == 0
    assert report["native_router_expert_trainables"] == 0
    assert report["trainable_parameters"] == 48
    payload = torch.load(args.output_checkpoint, weights_only=True)
    assert payload["schema"] == lift.CHECKPOINT_SCHEMA
    assert set(payload["trainable_state"]) == {
        f"backbone.model.layers.{layer}.mixer.adapter_{kind}.weight"
        for layer in (2, 4)
        for kind in ("a", "b")
    }
    assert tuple(
        payload["trainable_state"][
            "backbone.model.layers.2.mixer.adapter_a.weight"
        ].shape
    ) == (2, 6)


def test_run_rejects_tampered_factor(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    with args.factor.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(lift.UltraAdapterTransferError, match="factor hash"):
        lift.run(args)


def test_run_refuses_existing_output(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    args.output_checkpoint.parent.mkdir()
    args.output_checkpoint.write_bytes(b"occupied")
    with pytest.raises(lift.UltraAdapterTransferError, match="existing checkpoint"):
        lift.run(args)
