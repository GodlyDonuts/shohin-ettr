from pathlib import Path

import torch

from frozen_pointer_backbone import load_frozen_pointer_backbone
from model import GPT, GPTConfig


def tiny_model() -> GPT:
    return GPT(GPTConfig(
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=1,
        d_model=16,
        d_ff=32,
        seq_len=16,
        tie_embeddings=True,
    ))


def test_load_plain_shohin_checkpoint(tmp_path: Path) -> None:
    model = tiny_model()
    path = tmp_path / "plain.pt"
    torch.save({"cfg": vars(model.cfg), "model": model.state_dict(), "step": 17}, path)
    loaded, config, receipt = load_frozen_pointer_backbone(path, device="cpu")
    assert config == model.cfg
    assert receipt.checkpoint_format == "plain-shohin"
    assert receipt.base_step == 17
    assert torch.equal(loaded.tok.weight, model.tok.weight)


def test_load_prefixed_ettr_parent_checkpoint(tmp_path: Path) -> None:
    model = tiny_model()
    path = tmp_path / "parent.pt"
    torch.save({
        "base_config": vars(model.cfg),
        "base_rms_norm_eps": 1e-5,
        "base_import": {"model_id": "test"},
        "initialization": {"mode": "external-control"},
        "model": {
            **{"base." + key: value for key, value in model.state_dict().items()},
            "ettr.extra": torch.ones(1),
        },
    }, path)
    loaded, config, receipt = load_frozen_pointer_backbone(path, device="cpu")
    assert config == model.cfg
    assert receipt.checkpoint_format == "ettr-parent-base"
    assert receipt.initialization == "external-control"
    assert receipt.base_import == {"model_id": "test"}
    assert torch.equal(loaded.tok.weight, model.tok.weight)
