from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from capability_floor_trajectory import (
    MECHANISM_SCHEMA,
    OPERATION_NAMES,
    UnifiedActionPolicy,
    UnifiedETTRTrajectory,
    UnifiedTrajectoryConfig,
    UnifiedTrajectoryError,
    apply_unified_action,
    build_mechanism_receipt,
    empty_unified_state,
    mechanism_architecture_sha256,
    validate_unified_state,
)


def _config() -> UnifiedTrajectoryConfig:
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
        max_command_steps=4,
        min_world_steps=2,
        min_command_steps=2,
        max_edges=8,
    )


def _inputs(config: UnifiedTrajectoryConfig, batch: int = 2):
    generator = torch.Generator().manual_seed(42)
    world = torch.randn(batch, 5, config.input_width, generator=generator)
    command = torch.randn(batch, 4, config.input_width, generator=generator)
    query = torch.randn(batch, 3, config.input_width, generator=generator)
    return (
        world,
        torch.ones(batch, 5, dtype=torch.bool),
        command,
        torch.ones(batch, 4, dtype=torch.bool),
        query,
        torch.ones(batch, 3, dtype=torch.bool),
    )


def _policy(
    config: UnifiedTrajectoryConfig,
    operation: str,
    *,
    source: int = 0,
    target: int = 0,
    relation: int = 0,
    type_index: int = 0,
    value: int = 0,
) -> UnifiedActionPolicy:
    operation_tensor = F.one_hot(
        torch.tensor([OPERATION_NAMES.index(operation)]),
        len(OPERATION_NAMES),
    ).float()
    source_tensor = F.one_hot(torch.tensor([source]), config.num_slots).float()
    target_tensor = F.one_hot(torch.tensor([target]), config.num_slots).float()
    relation_tensor = F.one_hot(
        torch.tensor([relation]), config.num_relations
    ).float()
    type_tensor = F.one_hot(torch.tensor([type_index]), config.num_types).float()
    value_tensor = F.one_hot(torch.tensor([value]), config.num_value_codes).float()
    stop = torch.zeros(1)
    return UnifiedActionPolicy(
        operation_probabilities=operation_tensor,
        source_probabilities=source_tensor,
        target_probabilities=target_tensor,
        relation_probabilities=relation_tensor,
        type_probabilities=type_tensor,
        value_probabilities=value_tensor,
        stop_probabilities=stop,
        applied_operation=operation_tensor,
        applied_source=source_tensor,
        applied_target=target_tensor,
        applied_relation=relation_tensor,
        applied_type=type_tensor,
        applied_value=value_tensor,
        applied_stop=stop,
    )


def test_config_rejects_invalid_stop_bounds() -> None:
    config = _config()
    with pytest.raises(UnifiedTrajectoryError, match="WORLD minimum"):
        replace(config, min_world_steps=config.max_world_steps + 1).validate()


def test_fixed_state_algebra_allocates_writes_and_links() -> None:
    config = _config()
    state = empty_unified_state(1, config, device=torch.device("cpu"), dtype=torch.float32)
    state = apply_unified_action(
        state,
        _policy(config, "ALLOCATE", source=0, type_index=2, value=3),
        config,
        hard=True,
    )
    state = apply_unified_action(
        state,
        _policy(config, "ALLOCATE", source=1, type_index=1, value=4),
        config,
        hard=True,
    )
    state = apply_unified_action(
        state,
        _policy(config, "WRITE", source=0, value=6),
        config,
        hard=True,
    )
    state = apply_unified_action(
        state,
        _policy(config, "LINK", source=0, target=1, relation=1),
        config,
        hard=True,
    )
    validate_unified_state(state, config)
    assert state.active.tolist() == [[1.0, 1.0, 0.0, 0.0]]
    assert state.type_probabilities[0, 0].argmax().item() == 2
    assert state.value_probabilities[0, 0].argmax().item() == 6
    assert state.relations[0, 1, 0, 1].item() == 1.0


def test_world_and_command_use_one_shared_cell_and_terminate() -> None:
    config = _config()
    model = UnifiedETTRTrajectory(config)
    calls: list[int] = []
    hook = model.cell.register_forward_hook(lambda *_: calls.append(1))
    output = model(*_inputs(config), hard=True)
    hook.remove()
    assert len(calls) == config.max_world_steps + config.max_command_steps
    assert output.world_trace.stop_step.min().item() >= config.min_world_steps
    assert output.command_trace.stop_step.min().item() >= config.min_command_steps
    assert output.terminal_state.committed.eq(1).all()
    assert output.query_delta.shape == (2, 3, config.input_width)
    assert not hasattr(model, "world_cell")
    assert not hasattr(model, "command_cell")


def test_query_is_late_only_and_cannot_change_terminal_state() -> None:
    config = _config()
    torch.manual_seed(1)
    model = UnifiedETTRTrajectory(config)
    world, world_mask, command, command_mask, query, query_mask = _inputs(config)
    initial = model.initial_state(world)
    with pytest.raises(UnifiedTrajectoryError, match="COMMAND termination"):
        model.read_query(initial, query, query_mask)
    first = model(
        world,
        world_mask,
        command,
        command_mask,
        query,
        query_mask,
        hard=True,
    )
    second = model(
        world,
        world_mask,
        command,
        command_mask,
        query.roll(1, 0),
        query_mask,
        hard=True,
    )
    assert torch.equal(
        first.terminal_state.value_probabilities,
        second.terminal_state.value_probabilities,
    )
    assert torch.equal(first.terminal_state.relations, second.terminal_state.relations)
    assert "query" not in inspect.signature(model.run_phase).parameters


def test_soft_end_to_end_path_reaches_world_and_command_inputs() -> None:
    config = _config()
    torch.manual_seed(3)
    model = UnifiedETTRTrajectory(config)
    model.query_reader.gate.data.fill_(0.5)
    inputs = list(_inputs(config))
    inputs[0].requires_grad_(True)
    inputs[2].requires_grad_(True)
    output = model(*inputs, hard=False)
    loss = output.query_delta.square().mean()
    loss.backward()
    assert inputs[0].grad is not None and inputs[0].grad.abs().sum().item() > 0.0
    assert inputs[2].grad is not None and inputs[2].grad.abs().sum().item() > 0.0
    assert model.cell.operation_head.weight.grad is not None


def test_mechanism_receipt_binds_source_and_ownership() -> None:
    receipt = build_mechanism_receipt()
    assert receipt["schema"] == MECHANISM_SCHEMA
    assert receipt["phase_order"] == ["WORLD", "COMMAND", "QUERY"]
    assert "query-features" in receipt["forbidden_in_transition"]
    assert len(receipt["source_sha256"]) == 64
    assert len(mechanism_architecture_sha256()) == 64


def test_checked_in_mechanism_receipt_matches_source() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/r12/ettr_unified_trajectory_mechanism_v1.json"
    )
    payload = json.loads(path.read_text(encoding="ascii"))
    expected = {
        **build_mechanism_receipt(),
        "architecture_sha256": mechanism_architecture_sha256(),
        "architecture_parameters": UnifiedETTRTrajectory(
            UnifiedTrajectoryConfig()
        ).architecture_parameters(),
    }
    assert payload == expected
