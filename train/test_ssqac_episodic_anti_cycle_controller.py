"""Focused tests for full-trajectory episodic anti-cycle control."""

from __future__ import annotations

from pathlib import Path
import inspect

import torch

import ssqac_episodic_anti_cycle_controller as episodic


def _tiny_config() -> episodic.EpisodicConfig:
    return episodic.EpisodicConfig(
        base=episodic.proof.ControllerConfig(
            field_width=8,
            width=16,
            cell_hidden=24,
            matrix_layers=1,
            contract_hidden=20,
            coordinate_harmonics=1,
        ),
        state_width=12,
        state_hidden=18,
        state_layers=1,
        memory_slots=4,
    )


def _trajectory() -> episodic.ExpertTrajectory:
    rows = episodic.canonical_matrix(((2, 0), (0, 1)))
    normalize = next(
        action
        for action in episodic.enumerate_legal_actions(rows)
        if action.kind == "NORMALIZE"
    )
    next_rows = episodic.apply_action(rows, normalize)
    halt = next(
        action
        for action in episodic.enumerate_legal_actions(next_rows)
        if action.kind == "HALT"
    )
    return episodic.ExpertTrajectory(
        states=(rows, next_rows),
        actions=(normalize, halt),
    )


def test_preparation_round_trip(tmp_path: Path) -> None:
    artifact = episodic.prepare_artifact(
        seed=17,
        train_matrices=2,
        evaluation_matrices=2,
        train_maximum_rows=2,
        train_maximum_columns=2,
        evaluation_minimum_rows=3,
        evaluation_minimum_columns=3,
        evaluation_maximum_rows=3,
        evaluation_maximum_columns=3,
        maximum_preparation_steps=32,
    )
    path = tmp_path / "preparation.json"
    path.write_bytes(artifact.canonical_bytes())
    assert episodic._load_preparation(path) == artifact


def test_memory_modes_are_exact_causal_ablations() -> None:
    torch.manual_seed(7)
    model = episodic.EpisodicAntiCycleController(_tiny_config())
    trajectory = _trajectory()
    empty = episodic.EpisodicState()
    first = model.score(
        trajectory.states[0],
        empty,
        mode=episodic.MODE_REAL,
    )
    selected = episodic._target_index(first.actions, trajectory.actions[0])
    memory = model.advance(empty, first, selected)
    real = model.score(
        trajectory.states[1],
        memory,
        mode=episodic.MODE_REAL,
    )
    zero = model.score(
        trajectory.states[1],
        memory,
        mode=episodic.MODE_ZERO,
    )
    shuffled = model.score(
        trajectory.states[1],
        memory,
        mode=episodic.MODE_SHUFFLED,
    )
    assert torch.equal(zero.logits, model.score(
        trajectory.states[1],
        episodic.EpisodicState(recurrent=memory.recurrent),
        mode=episodic.MODE_REAL,
    ).logits)
    assert not torch.equal(real.maximum_similarity, shuffled.maximum_similarity)
    assert real.logits.shape == zero.logits.shape


def test_semantic_barrier_uses_feature_identity() -> None:
    torch.manual_seed(71)
    model = episodic.EpisodicAntiCycleController(_tiny_config())
    trajectory = _trajectory()
    key = model.encode_states(
        torch.tensor(trajectory.states[0], dtype=torch.long)
    )
    memory = episodic.EpisodicState(keys=(key,))
    barrier = model.score(
        trajectory.states[0],
        memory,
        mode=episodic.MODE_BARRIER,
    )
    shuffled = model.score(
        trajectory.states[0],
        memory,
        mode=episodic.MODE_BARRIER_SHUFFLED,
    )
    assert torch.any(barrier.cycle_evidence > shuffled.cycle_evidence)
    assert not torch.equal(barrier.logits, shuffled.logits)


def test_exact_barrier_distinguishes_raw_state_from_control() -> None:
    torch.manual_seed(72)
    model = episodic.EpisodicAntiCycleController(_tiny_config())
    trajectory = _trajectory()
    memory = episodic.EpisodicState(raw_states=(trajectory.states[0],))
    exact = model.score(
        trajectory.states[0],
        memory,
        mode=episodic.MODE_EXACT_BARRIER,
    )
    shuffled = model.score(
        trajectory.states[0],
        memory,
        mode=episodic.MODE_EXACT_BARRIER_SHUFFLED,
    )
    assert exact.exact_cycle_evidence.sum() > 0
    assert shuffled.exact_cycle_evidence.sum() == 0
    assert not torch.equal(exact.logits, shuffled.logits)


def test_action_renderer_is_permutation_equivariant() -> None:
    torch.manual_seed(8)
    model = episodic.EpisodicAntiCycleController(_tiny_config()).eval()
    rows = _trajectory().states[0]
    actions = episodic.enumerate_legal_actions(rows)
    canonical = model.score(rows, episodic.EpisodicState(), mode=episodic.MODE_ZERO)
    reverse = model.score(
        rows,
        episodic.EpisodicState(),
        mode=episodic.MODE_ZERO,
        actions=tuple(reversed(actions)),
    )
    mapped = {
        action: float(logit.detach())
        for action, logit in zip(reverse.actions, reverse.logits, strict=True)
    }
    for action, logit in zip(canonical.actions, canonical.logits, strict=True):
        assert torch.allclose(logit, torch.tensor(mapped[action]), atol=1e-6)


def test_full_trajectory_training_smoke() -> None:
    torch.manual_seed(9)
    model = episodic.EpisodicAntiCycleController(_tiny_config())
    receipt = episodic.train_full_trajectories(
        model,
        (_trajectory(),),
        mode=episodic.MODE_REAL,
        optimizer_updates=2,
        batch_size=1,
        learning_rate=1e-3,
        seed=9,
        amp_bfloat16=False,
    )
    assert receipt.optimizer_updates == 2
    assert receipt.trajectory_presentations == 2
    assert receipt.state_presentations == 4
    assert len(receipt.batch_schedule_sha256) == 64


def test_exact_barrier_full_trajectory_training_smoke() -> None:
    torch.manual_seed(10)
    model = episodic.EpisodicAntiCycleController(_tiny_config())
    receipt = episodic.train_full_trajectories(
        model,
        (_trajectory(),),
        mode=episodic.MODE_EXACT_BARRIER,
        optimizer_updates=1,
        batch_size=1,
        learning_rate=1e-3,
        seed=10,
        amp_bfloat16=False,
    )
    assert receipt.optimizer_updates == 1
    assert receipt.state_presentations == 2


def test_complete_system_respects_budget() -> None:
    model = episodic.EpisodicAntiCycleController(_tiny_config())
    assert model.parameter_count > 0
    assert model.complete_system_parameters < episodic.TOTAL_PARAMETER_BUDGET


def test_candidate_rollout_source_has_no_privileged_calls() -> None:
    source = inspect.getsource(episodic.autonomous_rollout)
    assert "oracle" not in source
    assert "search" not in source
    assert "verifier" not in source
    assert "assess" not in source


def test_module_does_not_name_protected_checkpoint() -> None:
    source = Path(episodic.__file__).read_text(encoding="utf-8")
    assert "ckpt_0300000.pt" not in source
