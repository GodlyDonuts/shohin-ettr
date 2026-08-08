from pathlib import Path

import torch

from pipeline.normalize_lora_checkpoint_metadata import normalize_checkpoint
from pipeline.tokenize_shards import sha256_file


def test_normalization_changes_only_legacy_unfreeze_metadata(tmp_path: Path):
    source = tmp_path / "source.pt"
    output = tmp_path / "output.pt"
    report = tmp_path / "report.json"
    torch.save(
        {
            "schema": "fixture",
            "update": 10,
            "metadata": {"arm": "baseline"},
            "trainable_state": {
                "model.layer.lora_a.weight": torch.arange(6).reshape(2, 3),
                "model.layer.lora_b.weight": torch.arange(6).reshape(3, 2),
            },
            "optimizer": {"state": [torch.tensor([1.0, 2.0])]},
        },
        source,
    )
    receipt = normalize_checkpoint(
        source=source,
        source_sha256=sha256_file(source),
        output=output,
        report=report,
    )
    migrated = torch.load(output, map_location="cpu", weights_only=False)
    original = torch.load(source, map_location="cpu", weights_only=False)
    assert migrated["metadata"] == {"arm": "baseline", "unfreeze_layers": 0}
    assert torch.equal(
        migrated["trainable_state"]["model.layer.lora_a.weight"],
        original["trainable_state"]["model.layer.lora_a.weight"],
    )
    assert receipt["all_non_metadata_state_bitwise_equal"] is True
    assert receipt["trainable_state_tensors"] == 2
