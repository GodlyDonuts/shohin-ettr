from pathlib import Path

import torch

from gset1_fault_gate import (
    GSET1Config,
    GSET1FaultGate,
    load_gate_checkpoint,
    save_gate_checkpoint,
)


def test_gate_geometry_and_checkpoint(tmp_path: Path) -> None:
    gate = GSET1FaultGate(GSET1Config(hidden_size=16, gate_width=8))
    logits = gate(torch.randn(4, 16))
    assert logits.shape == (4, 2)
    assert gate.trainable_parameter_count() == 16 * 2 + 16 * 8 + 8 + 8 * 2 + 2
    optimizer = torch.optim.AdamW(gate.parameters())
    path = tmp_path / "gate.pt"
    save_gate_checkpoint(path, gate, optimizer, 3, {"arm": "aligned"})
    restored, metadata = load_gate_checkpoint(path)
    assert metadata == {"arm": "aligned"}
    assert torch.equal(restored.out.weight, gate.out.weight)
