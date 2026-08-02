import json
from pathlib import Path

import torch

from capability_floor_dense_control import (
    DenseControlConfig,
    FavorableDenseRecurrentControl,
    build_dense_control_descriptor,
    find_parameter_matched_dense_width,
)
from capability_floor_trajectory import UnifiedETTRTrajectory, UnifiedTrajectoryConfig


def _treatment_config() -> UnifiedTrajectoryConfig:
    return UnifiedTrajectoryConfig(
        input_width=16,
        state_width=16,
        num_slots=4,
        num_types=3,
        num_relations=2,
        num_value_codes=7,
        num_heads=4,
        core_layers=1,
        reader_layers=1,
        ff_multiplier=2,
        max_world_steps=3,
        max_command_steps=3,
        min_world_steps=1,
        min_command_steps=1,
        max_edges=8,
    )


def _inputs(config: UnifiedTrajectoryConfig):
    generator = torch.Generator().manual_seed(9)
    return (
        torch.randn(2, 5, config.input_width, generator=generator),
        torch.ones(2, 5, dtype=torch.bool),
        torch.randn(2, 4, config.input_width, generator=generator),
        torch.ones(2, 4, dtype=torch.bool),
        torch.randn(2, 3, config.input_width, generator=generator),
        torch.ones(2, 3, dtype=torch.bool),
    )


def test_dense_control_is_favorable_untied_and_source_deleted() -> None:
    treatment = _treatment_config()
    config = DenseControlConfig.from_treatment(treatment, hidden_width=16)
    model = FavorableDenseRecurrentControl(config)
    assert model.world_cell is not model.command_cell
    inputs = _inputs(treatment)
    first = model(*inputs, hard=True)
    changed = list(inputs)
    changed[4] = inputs[4].roll(1, 0)
    second = model(*changed, hard=True)
    assert first.query_delta.shape == (2, 3, treatment.input_width)
    assert torch.equal(first.dense_terminal, second.dense_terminal)
    assert first.terminal_state.committed.eq(1).all()
    assert first.world_trace.stop_step.min().item() >= treatment.min_world_steps
    assert first.command_trace.stop_step.min().item() >= treatment.min_command_steps


def test_dense_control_soft_path_is_end_to_end_differentiable() -> None:
    treatment = _treatment_config()
    model = FavorableDenseRecurrentControl(
        DenseControlConfig.from_treatment(treatment, hidden_width=16)
    )
    model.query_gate.data.fill_(0.5)
    inputs = list(_inputs(treatment))
    inputs[0].requires_grad_(True)
    inputs[2].requires_grad_(True)
    output = model(*inputs, hard=False)
    output.query_delta.square().mean().backward()
    assert inputs[0].grad is not None and inputs[0].grad.abs().sum().item() > 0.0
    assert inputs[2].grad is not None and inputs[2].grad.abs().sum().item() > 0.0


def test_parameter_matcher_can_recover_an_exact_dense_width() -> None:
    treatment = _treatment_config()
    target_config = DenseControlConfig.from_treatment(treatment, hidden_width=24)
    target_parameters = FavorableDenseRecurrentControl(
        target_config
    ).architecture_parameters()
    config, parameters, relative = find_parameter_matched_dense_width(
        target_parameters,
        treatment,
        minimum_width=16,
        maximum_width=32,
        tolerance=0.0001,
    )
    assert config.hidden_width == 24
    assert parameters == target_parameters
    assert relative == 0.0


def test_default_treatment_has_a_parameter_matched_dense_descriptor() -> None:
    treatment = _treatment_config()
    treatment_parameters = UnifiedETTRTrajectory(treatment).architecture_parameters()
    descriptor = build_dense_control_descriptor(treatment_parameters, treatment)
    assert descriptor["parameter_match_relative_error"] <= 0.01
    assert descriptor["flop_receipt"] is None
    assert descriptor["untied_world_and_command_cells"] is True


def test_checked_in_dense_descriptor_matches_source() -> None:
    treatment = UnifiedTrajectoryConfig()
    treatment_parameters = UnifiedETTRTrajectory(treatment).architecture_parameters()
    expected = build_dense_control_descriptor(treatment_parameters, treatment)
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/r12/ettr_favorable_dense_control_v1.json"
    )
    assert json.loads(path.read_text(encoding="ascii")) == expected
