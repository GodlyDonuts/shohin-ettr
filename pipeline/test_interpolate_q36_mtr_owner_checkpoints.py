from pathlib import Path

import pytest
import torch

import interpolate_q36_mtr_owner_checkpoints as module
from q36_mtr_roles import OWNER_UPDATES, ROLE_CHECKPOINT_SCHEMA, role_contract
from shared_post_mlp_revision import trainable_state_sha256


def _checkpoint(path: Path, value: float, monkeypatch, *, role: str = "owner") -> None:
    state = {"weight": torch.full((2, 2), value, dtype=torch.float32)}
    metadata = role_contract(role)
    metadata["final_trainable_state_sha256"] = trainable_state_sha256(state)
    torch.save(
        {
            "schema": ROLE_CHECKPOINT_SCHEMA,
            "update": OWNER_UPDATES,
            "trainable_state": state,
            "metadata": metadata,
        },
        path,
    )
    monkeypatch.setattr(module, "TRAINABLE_PARAMETERS", 4)


def test_interpolates_exact_owner_state(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "output.pt"
    report = tmp_path / "report.json"
    _checkpoint(first, 1.0, monkeypatch)
    _checkpoint(second, 3.0, monkeypatch)
    result = module.interpolate(first, second, output, report, second_weight=0.25)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert torch.equal(payload["trainable_state"]["weight"], torch.full((2, 2), 1.5))
    assert result["parameters"] == 4
    assert result["second_weight"] == 0.25
    assert payload["metadata"]["interpolation"][
        "second_checkpoint_sha256"
    ] == module.sha256_file(second)


def test_rejects_endpoint_weight(tmp_path: Path) -> None:
    with pytest.raises(module.Q36MTROwnerInterpolationError, match="settings"):
        module.interpolate(
            tmp_path / "a",
            tmp_path / "b",
            tmp_path / "out",
            tmp_path / "report",
            second_weight=1.0,
        )


def test_interpolates_aligned_reviser_state(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "output.pt"
    report = tmp_path / "report.json"
    _checkpoint(first, 2.0, monkeypatch, role="aligned")
    _checkpoint(second, 4.0, monkeypatch, role="aligned")
    result = module.interpolate(
        first,
        second,
        output,
        report,
        second_weight=0.1,
        role="aligned",
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert torch.allclose(payload["trainable_state"]["weight"], torch.full((2, 2), 2.2))
    assert payload["metadata"]["role"] == "aligned"
    assert payload["metadata"]["interpolation"]["role"] == "aligned"
    assert result["role"] == "aligned"
