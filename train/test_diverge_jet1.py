#!/usr/bin/env python3
"""Focused mechanics tests for DIVERGE-JET1."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_jet1_data import PROGRAM_ACTIONS, apply_program, generate_jet1_episode
from diverge_jet1_runtime import (
    DELTAS,
    FIELD_COUNT,
    REGISTER_COUNT,
    VALUE_COUNT,
    JET1Config,
    JointEpistemicTrajectory,
    architecture_receipt,
)


def _install_exact_algebra(model: JointEpistemicTrajectory) -> None:
    routes = (
        (0, 1, 2, 3, 4),
        (1, 0, 2, 3, 4),
        (0, 1, 3, 2, 4),
        (0, 1, 2, 4, 3),
    )
    with torch.no_grad():
        model.executor.route_logits.fill_(-20.0)
        model.executor.delta_logits.fill_(-20.0)
        for action, sources in enumerate(routes):
            for output, source in enumerate(sources):
                model.executor.route_logits[action, output, source] = 20.0
                delta = 3 if action == 0 and output == 0 else 0
                model.executor.delta_logits[action, output, DELTAS.index(delta)] = 20.0
        model.query_route_logits.fill_(-20.0)
        for slot in range(REGISTER_COUNT):
            model.query_route_logits[slot, slot] = 20.0


class _FixedEvidence(nn.Module):
    def __init__(self, before: tuple[int, ...], after: tuple[int, ...]):
        super().__init__()
        values = (*before, *after)
        logits = torch.full((FIELD_COUNT, VALUE_COUNT), -20.0)
        for field, value in enumerate(values):
            logits[field, value] = 20.0
        self.register_buffer("logits", logits)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.logits[None].expand(features.shape[0], -1, -1)


def _program_tensor(programs: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.zeros(1, 1, 2, 2, dtype=torch.long)
    mask = torch.zeros_like(actions, dtype=torch.bool)
    for candidate, program in enumerate(programs):
        sequence = PROGRAM_ACTIONS[program]
        actions[0, 0, candidate, : len(sequence)] = torch.tensor(sequence)
        mask[0, 0, candidate, : len(sequence)] = True
    return actions, mask


def test_data_is_deterministic_and_wrong_prior_is_forced() -> None:
    first = generate_jet1_episode(seed=41, cohort="train", depth=8)
    second = generate_jet1_episode(seed=41, cohort="train", depth=8)
    assert first == second
    terminal = first.initial_state
    for step in first.steps:
        terminal = apply_program(
            terminal, step.candidate_programs[step.gold_candidate]
        )
    assert first.terminal_state == terminal
    for step in first.steps:
        assert step.prior_logits[step.gold_candidate] < step.prior_logits[1 - step.gold_candidate]


def test_exact_joint_trajectory_recovers_against_wrong_prior() -> None:
    config = JET1Config(input_width=16, reader_width=16, reader_heads=4)
    model = JointEpistemicTrajectory(config)
    _install_exact_algebra(model)
    initial = (11, 29, 37, 43, 53)
    probe_before = (7, 19, 31, 47, 59)
    probe_after = apply_program(probe_before, 0)
    model.evidence = _FixedEvidence(probe_before, probe_after)
    actions, action_mask = _program_tensor((0, 2))
    output = model(
        torch.randn(1, 1, 6, 16),
        torch.ones(1, 1, 6, dtype=torch.bool),
        torch.tensor([initial]),
        actions,
        action_mask,
        torch.tensor([[[-1.3862944, -0.2876821]]]),
        torch.tensor([0]),
        hard_forward=True,
    )
    expected = apply_program(initial, 0)
    assert int(output.selected_candidates[0, 0]) == 0
    assert tuple(output.terminal_probabilities[0].argmax(-1).tolist()) == expected
    assert int(output.answer_probabilities[0].argmax()) == expected[0]
    assert float(output.invalid_mass.detach().max()) == 0.0


def test_joint_loss_reaches_reader_executor_and_query() -> None:
    torch.manual_seed(3)
    config = JET1Config(input_width=16, reader_width=16, reader_heads=4)
    model = JointEpistemicTrajectory(config)
    actions, action_mask = _program_tensor((0, 3))
    output = model(
        torch.randn(2, 1, 9, 16),
        torch.ones(2, 1, 9, dtype=torch.bool),
        torch.tensor(((9, 13, 21, 25, 33), (11, 17, 23, 31, 41))),
        actions.expand(2, -1, -1, -1).clone(),
        action_mask.expand(2, -1, -1, -1).clone(),
        torch.tensor([[[-1.3862944, -0.2876821]], [[-1.3862944, -0.2876821]]]),
        torch.tensor([0, 1]),
        hard_forward=True,
    )
    targets = torch.tensor([12, 20])
    loss = F.nll_loss(output.answer_probabilities.clamp_min(1e-8).log(), targets)
    loss = loss + F.cross_entropy(
        output.choice_logits.reshape(-1, 2), torch.zeros(2, dtype=torch.long)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.evidence.input_projection.weight.grad is not None
    assert model.executor.route_logits.grad is not None
    assert model.query_route_logits.grad is not None


def test_candidate_source_audit() -> None:
    receipt = architecture_receipt(
        JointEpistemicTrajectory(
            JET1Config(input_width=16, reader_width=16, reader_heads=4)
        )
    )
    assert receipt["source_audit"]["pass"]
    assert receipt["whole_program_hard_forward"]
    assert not receipt["fieldwise_hypothesis_averaging"]


def main() -> None:
    test_data_is_deterministic_and_wrong_prior_is_forced()
    test_exact_joint_trajectory_recovers_against_wrong_prior()
    test_joint_loss_reaches_reader_executor_and_query()
    test_candidate_source_audit()
    print("DIVERGE-JET1 tests passed", flush=True)


if __name__ == "__main__":
    main()
