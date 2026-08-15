from __future__ import annotations

from pathlib import Path

import pytest
import torch

from upward_moe_role_lineage import (
    UpwardMoERoleLineageError,
    load_role_checkpoint,
    load_role_pair,
    save_role_checkpoint,
    trainable_state_sha256,
)
from upward_moe_temporal_gate import UpwardMoETemporalGateSpec

SPEC = UpwardMoETemporalGateSpec(
    host="test-moe",
    model_revision="1" * 40,
    model_config_sha256="2" * 64,
    architecture="test-temporal-gate",
    attachment_surface="post-mixer-residual",
    module_attribute="mixer",
    hidden_size=2,
    rank=1,
    alpha=1.0,
    controlled_layer_indices=(1, 3),
    require_final_contiguous=False,
)


def _state(offset: float):
    state = {}
    for index in SPEC.controlled_layer_indices:
        prefix = f"backbone.model.layers.{index}.mixer"
        state[f"{prefix}.adapter_a.weight"] = torch.tensor(
            [[1.0 + offset, 2.0 + offset]], dtype=torch.float32
        )
        state[f"{prefix}.adapter_b.weight"] = torch.tensor(
            [[3.0 + offset], [4.0 + offset]], dtype=torch.float32
        )
    return state


def _save_pair(root: Path):
    owner_path = root / "owner.pt"
    owner = save_role_checkpoint(
        owner_path,
        role="owner",
        state=_state(0.0),
        spec=SPEC,
        initial_state_sha256="a" * 64,
        training_receipt={"data_sha256": "b" * 64, "updates": 256},
    )
    revision_path = root / "aligned.pt"
    save_role_checkpoint(
        revision_path,
        role="aligned",
        state=_state(0.5),
        spec=SPEC,
        initial_state_sha256=owner["metadata"]["final_trainable_state_sha256"],
        training_receipt={"data_sha256": "c" * 64, "updates": 256},
        warm_start_checkpoint=owner_path,
    )
    return owner_path, revision_path


def test_exact_owner_revision_warm_start_round_trip(tmp_path: Path) -> None:
    owner_path, revision_path = _save_pair(tmp_path)
    owner_state, revision_state, receipt = load_role_pair(
        owner_path, revision_path, SPEC
    )
    assert receipt["warm_start_exact"] is True
    assert receipt["native_router_expert_trainables"] == 0
    assert receipt["owner_state_sha256"] == trainable_state_sha256(owner_state)
    assert receipt["revision_state_sha256"] == trainable_state_sha256(revision_state)
    assert receipt["owner_state_sha256"] != receipt["revision_state_sha256"]


def test_aligned_cannot_start_from_unbound_or_wrong_owner(tmp_path: Path) -> None:
    owner_path = tmp_path / "owner.pt"
    owner = save_role_checkpoint(
        owner_path,
        role="owner",
        state=_state(0.0),
        spec=SPEC,
        initial_state_sha256="a" * 64,
        training_receipt={"data_sha256": "b" * 64},
    )
    with pytest.raises(UpwardMoERoleLineageError):
        save_role_checkpoint(
            tmp_path / "missing.pt",
            role="aligned",
            state=_state(0.5),
            spec=SPEC,
            initial_state_sha256=owner["metadata"]["final_trainable_state_sha256"],
            training_receipt={"data_sha256": "c" * 64},
        )
    with pytest.raises(UpwardMoERoleLineageError):
        save_role_checkpoint(
            tmp_path / "wrong.pt",
            role="aligned",
            state=_state(0.5),
            spec=SPEC,
            initial_state_sha256="d" * 64,
            training_receipt={"data_sha256": "c" * 64},
            warm_start_checkpoint=owner_path,
        )


def test_cross_host_or_surface_checkpoint_is_rejected(tmp_path: Path) -> None:
    owner_path, _ = _save_pair(tmp_path)
    other = UpwardMoETemporalGateSpec(
        **{**SPEC.__dict__, "host": "other-moe", "module_attribute": "mlp"}
    )
    with pytest.raises(UpwardMoERoleLineageError):
        load_role_checkpoint(owner_path, other)


def test_state_names_dtype_finiteness_and_nonzero_update_are_required(
    tmp_path: Path,
) -> None:
    bad = _state(0.0)
    bad.pop(next(iter(bad)))
    with pytest.raises(UpwardMoERoleLineageError):
        save_role_checkpoint(
            tmp_path / "missing-name.pt",
            role="owner",
            state=bad,
            spec=SPEC,
            initial_state_sha256="a" * 64,
            training_receipt={"data_sha256": "b" * 64},
        )
    bad = _state(0.0)
    next(iter(bad.values()))[0, 0] = float("nan")
    with pytest.raises(UpwardMoERoleLineageError):
        save_role_checkpoint(
            tmp_path / "nonfinite.pt",
            role="owner",
            state=bad,
            spec=SPEC,
            initial_state_sha256="a" * 64,
            training_receipt={"data_sha256": "b" * 64},
        )
    state = _state(0.0)
    with pytest.raises(UpwardMoERoleLineageError):
        save_role_checkpoint(
            tmp_path / "unchanged.pt",
            role="owner",
            state=state,
            spec=SPEC,
            initial_state_sha256=trainable_state_sha256(state),
            training_receipt={"data_sha256": "b" * 64},
        )
