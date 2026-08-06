#!/usr/bin/env python3
"""Assessor-side deterministic episodes for DIVERGE-JET1."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from diverge_mei1_data import EVIDENCE_COHORTS, exact_action_batch, generate_probe_evidence
from diverge_mei1_runtime import REGISTER_COUNT, VALUE_COUNT


PROGRAM_ACTIONS = (
    (0, 1),
    (1, 0),
    (2,),
    (3,),
)
MAX_PROGRAM_ACTIONS = max(len(actions) for actions in PROGRAM_ACTIONS)


@dataclass(frozen=True, slots=True)
class JET1Step:
    words: tuple[str, ...]
    before: tuple[int, ...]
    after: tuple[int, ...]
    candidate_programs: tuple[int, int]
    candidate_actions: tuple[tuple[int, ...], tuple[int, ...]]
    prior_logits: tuple[float, float]
    gold_candidate: int


@dataclass(frozen=True, slots=True)
class JET1Episode:
    episode_id: str
    cohort: str
    initial_state: tuple[int, ...]
    steps: tuple[JET1Step, ...]
    terminal_state: tuple[int, ...]
    query_slot: int
    answer: int

    @property
    def depth(self) -> int:
        return len(self.steps)


def apply_program(values: tuple[int, ...], program: int) -> tuple[int, ...]:
    """Apply one assessor program through the independent primitive oracle."""

    if not 0 <= program < len(PROGRAM_ACTIONS):
        raise ValueError("JET1 program leaves the fixed vocabulary")
    state = values
    for action in PROGRAM_ACTIONS[program]:
        state = exact_action_batch((state,), (action,))[0]
    if len(state) != REGISTER_COUNT or any(not 0 <= value < VALUE_COUNT for value in state):
        raise ValueError("JET1 supervisor program leaves the state domain")
    return state


def generate_jet1_episode(*, seed: int, cohort: str, depth: int) -> JET1Episode:
    if cohort not in EVIDENCE_COHORTS:
        raise ValueError("unknown JET1 evidence cohort")
    if not 1 <= depth <= 24:
        raise ValueError("JET1 depth leaves the frozen board")
    rng = random.Random(seed)
    initial = tuple(rng.randrange(5, 31) for _ in range(REGISTER_COUNT))
    persistent = initial
    steps = []
    for step_index in range(depth):
        gold_program = rng.randrange(len(PROGRAM_ACTIONS))
        false_program = rng.randrange(len(PROGRAM_ACTIONS) - 1)
        if false_program >= gold_program:
            false_program += 1
        if rng.randrange(2):
            candidates = (gold_program, false_program)
            gold_candidate = 0
        else:
            candidates = (false_program, gold_program)
            gold_candidate = 1
        evidence = generate_probe_evidence(
            seed=seed * 104729 + step_index * 7919 + 17,
            cohort=cohort,
            program=gold_program,
            sample_program=False,
        )
        prior_logits = tuple(
            math.log(0.25 if index == gold_candidate else 0.75)
            for index in range(2)
        )
        steps.append(
            JET1Step(
                words=evidence.words,
                before=evidence.before,
                after=evidence.after,
                candidate_programs=candidates,
                candidate_actions=tuple(PROGRAM_ACTIONS[value] for value in candidates),
                prior_logits=prior_logits,
                gold_candidate=gold_candidate,
            )
        )
        persistent = apply_program(persistent, gold_program)
    query_slot = rng.randrange(REGISTER_COUNT)
    return JET1Episode(
        episode_id=f"jet1-{cohort}-{seed}-{depth}",
        cohort=cohort,
        initial_state=initial,
        steps=tuple(steps),
        terminal_state=persistent,
        query_slot=query_slot,
        answer=persistent[query_slot],
    )


def renderer_parity(*, seed: int, count: int) -> dict[str, int | bool]:
    """Prove that JET1 consumes the unchanged MEI1 evidence bytes."""

    if count <= 0:
        raise ValueError("JET1 parity count must be positive")
    mismatches = 0
    records = 0
    for index in range(count):
        depth = index % 8 + 1
        episode = generate_jet1_episode(seed=seed + index, cohort="train", depth=depth)
        for step_index, step in enumerate(episode.steps):
            gold_program = step.candidate_programs[step.gold_candidate]
            expected = generate_probe_evidence(
                seed=(seed + index) * 104729 + step_index * 7919 + 17,
                cohort="train",
                program=gold_program,
                sample_program=False,
            )
            mismatches += int(step.words != expected.words)
            records += 1
    return {
        "episodes": count,
        "records": records,
        "word_mismatches": mismatches,
        "pass": mismatches == 0,
    }
