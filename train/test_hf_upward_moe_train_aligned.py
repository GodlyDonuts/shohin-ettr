from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

import hf_upward_moe_train_aligned as module
from upward_moe_role_lineage import save_role_checkpoint
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


class _Model:
    def __init__(self):
        self.parameters = {
            name: torch.nn.Parameter(torch.zeros_like(value))
            for name, value in _state(0.0).items()
        }

    def named_parameters(self):
        return self.parameters.items()

    def trainable_state_sha256(self):
        from upward_moe_role_lineage import trainable_state_sha256

        return trainable_state_sha256(self.parameters)


def test_static_aligned_contract_matches_causal_transfer() -> None:
    contract = module.static_aligned_contract()
    assert contract["role"] == "aligned"
    assert contract["warm_start_role"] == "owner"
    assert contract["presentations"] == 9655
    assert contract["updates"] == 256
    assert contract["gradient_accumulation"] == 8
    assert contract["consumed_presentations"] == 2048
    assert contract["internal_draft_visible"] is True
    assert contract["external_proposer"] is False
    assert contract["native_router_expert_trainables"] == 0


def test_restore_exact_owner_copies_bound_trainable_state(tmp_path: Path) -> None:
    owner = tmp_path / "owner.pt"
    payload = save_role_checkpoint(
        owner,
        role="owner",
        state=_state(0.0),
        spec=SPEC,
        initial_state_sha256="a" * 64,
        training_receipt={"data_sha256": "b" * 64},
    )
    model = _Model()
    receipt = module.restore_exact_owner(model, owner, SPEC)
    assert receipt["owner_restore_exact"] is True
    assert (
        receipt["owner_state_sha256"]
        == payload["metadata"]["final_trainable_state_sha256"]
    )
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, payload["trainable_state"][name])


def test_aligned_rows_require_unique_draft_visible_internal_trajectories() -> None:
    rows = [
        {
            "schema": module.DATA_SCHEMA,
            "identity_sha256": f"{index:064x}",
            "internal_draft_visible": True,
            "external_candidate_text_visible": False,
            "runtime_fields": ["question"],
            "question": "Problem\n\nInternal model-owned draft:\nDraft",
            "response": "Answer",
        }
        for index in range(module.ALIGNED_PRESENTATIONS)
    ]
    assert module.validate_aligned_rows(rows) is rows
    rows[0]["internal_draft_visible"] = False
    with pytest.raises(module.UpwardMoEAlignedTrainingError):
        module.validate_aligned_rows(rows)


def test_parse_args_requires_owner_data_hash_and_host(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "aligned",
            "--host",
            "nemotron-super",
            "--model-root",
            "/model",
            "--model-manifest",
            "/manifest",
            "--mechanics-report",
            "/mechanics",
            "--data",
            "/data",
            "--expected-data-sha256",
            "a" * 64,
            "--owner-checkpoint",
            "/owner",
            "--output",
            "/output",
        ],
    )
    args = module.parse_args()
    assert isinstance(args, argparse.Namespace)
    assert args.host == "nemotron-super"
    assert args.owner_checkpoint == Path("/owner")
