from dataclasses import replace

import pytest
import torch

from capability_floor_sufficiency import (
    OperationFamilyTensorProbe,
    SufficiencyScores,
    TensorSufficiencyError,
    build_sufficiency_receipt,
    sufficiency_decision,
    tensor_sha256,
    validate_sufficiency_receipt,
)
from capability_floor_trajectory import UnifiedTrajectoryConfig, empty_unified_state


def _config() -> UnifiedTrajectoryConfig:
    return UnifiedTrajectoryConfig(
        input_width=12,
        state_width=12,
        num_slots=3,
        num_types=2,
        num_relations=2,
        num_value_codes=5,
        num_heads=3,
        core_layers=1,
        reader_layers=1,
        ff_multiplier=2,
        max_world_steps=2,
        max_command_steps=2,
        max_edges=4,
    )


def _scores() -> SufficiencyScores:
    return SufficiencyScores(
        symbolic_reference_accuracy=0.99,
        tensor_probe_accuracy=0.98,
        renderer_orbit_accuracy=0.97,
        renderer_orbit_prediction_agreement=0.98,
        binding_deranged_accuracy=0.34,
        state_value_permuted_accuracy=0.35,
        empirical_chance_accuracy=1.0 / 3.0,
    )


def test_probe_accepts_only_exact_tensor_geometry() -> None:
    config = _config()
    probe = OperationFamilyTensorProbe(config, max_roles=4)
    source = torch.randn(2, 6, config.input_width)
    source_mask = torch.ones(2, 6, dtype=torch.bool)
    role_masks = torch.zeros(2, 4, 6, dtype=torch.bool)
    role_masks[:, 0, 0:2] = True
    role_masks[:, 1, 2:4] = True
    role_masks[:, 2, 4] = True
    state = empty_unified_state(
        2,
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    logits = probe(source, source_mask, role_masks, state)
    assert logits.shape == (2, 4, 3)
    assert logits[:, 3].eq(0).all()
    bad_roles = role_masks.clone()
    bad_roles[:, 0, 5] = True
    bad_mask = source_mask.clone()
    bad_mask[:, 5] = False
    with pytest.raises(TensorSufficiencyError, match="padded"):
        probe(source, bad_mask, bad_roles, state)


def test_tensor_hash_binds_dtype_shape_and_content() -> None:
    value = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    digest = tensor_sha256(value)
    assert len(digest) == 64
    assert digest != tensor_sha256(value.float())
    changed = value.clone()
    changed[0, 0] = 1
    assert digest != tensor_sha256(changed)


@pytest.mark.parametrize(
    ("scores", "decision"),
    [
        (
            replace(_scores(), symbolic_reference_accuracy=0.90),
            "reject-symbolic-reference-or-corpus",
        ),
        (
            replace(_scores(), tensor_probe_accuracy=0.90),
            "redesign-neural-interface",
        ),
        (
            replace(_scores(), renderer_orbit_accuracy=0.90),
            "reject-renderer-instability",
        ),
        (
            replace(_scores(), binding_deranged_accuracy=0.50),
            "reject-binding-control-leakage",
        ),
        (_scores(), "pass-interface-sufficiency"),
    ],
)
def test_sufficiency_decisions_are_fail_closed(
    scores: SufficiencyScores,
    decision: str,
) -> None:
    assert sufficiency_decision(scores) == decision


def test_receipt_binds_all_model_inputs_and_revalidates() -> None:
    config = _config()
    source = torch.randn(2, 4, config.input_width)
    source_mask = torch.ones(2, 4, dtype=torch.bool)
    role_masks = torch.ones(2, 2, 4, dtype=torch.bool)
    labels = torch.tensor([[0, 1], [2, 0]])
    state = empty_unified_state(
        2,
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    receipt = build_sufficiency_receipt(
        candidate="test-backbone",
        component="operation-family",
        split_sha256="a" * 64,
        source_features=source,
        source_mask=source_mask,
        role_masks=role_masks,
        state=state,
        labels=labels,
        scores=_scores(),
    )
    validate_sufficiency_receipt(receipt)
    assert receipt["decision"] == "pass-interface-sufficiency"
    receipt["assessor_features_available_at_inference"] = True
    with pytest.raises(TensorSufficiencyError, match="custody"):
        validate_sufficiency_receipt(receipt)
