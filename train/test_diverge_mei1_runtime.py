#!/usr/bin/env python3
"""Standalone contract tests for the candidate-only DIVERGE-MEI1 runtime."""

from __future__ import annotations

import torch

from diverge_mei1_runtime import (
    ACTION_NAMES,
    DELTAS,
    DIVERGEMEI1,
    MEI1Config,
    MEI1ContractError,
    ModelChoice,
    ModelState,
    action_id,
    architecture_receipt,
    derive_model_allowed,
    execute_model_mdd,
    query_model_mdd,
    source_audit,
    support_contains,
)


def _install_exact_test_weights(model: DIVERGEMEI1) -> None:
    routes = (
        (0, 1, 2, 3, 4),
        (1, 0, 2, 3, 4),
        (0, 1, 3, 2, 4),
        (0, 1, 2, 4, 3),
    )
    deltas = (
        (3, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    with torch.no_grad():
        model.executor.route_logits.fill_(-20)
        model.executor.delta_logits.fill_(-20)
        for action, row in enumerate(routes):
            for output, source in enumerate(row):
                model.executor.route_logits[action, output, source] = 20
                model.executor.delta_logits[
                    action, output, DELTAS.index(deltas[action][output])
                ] = 20
        model.query.route_logits.fill_(-20)
        for slot in range(5):
            model.query.route_logits[slot, slot] = 20


def _choice(record: int, domain: int, actions: tuple[int, ...], key: str) -> ModelChoice:
    return ModelChoice(record, domain, 1, actions, key, f"p-{record}-{domain}")


def test_action_tokens() -> None:
    assert action_id("ADD_VALUE", (0, 3)) == 0
    assert action_id("SWAP_VALUE", (0, 1)) == 1
    assert action_id("SWAP_VALUE", (2, 3)) == 2
    assert action_id("SWAP_VALUE", (3, 4)) == 3
    try:
        action_id("ADD_VALUE", (1, 3))
    except MEI1ContractError:
        pass
    else:
        raise AssertionError("unknown packet action did not fail closed")


def test_learned_executor_and_query() -> None:
    model = DIVERGEMEI1(MEI1Config())
    _install_exact_test_weights(model)
    state = torch.tensor([[1, 10, 20, 30, 40]], dtype=torch.long)
    actions = torch.arange(len(ACTION_NAMES), dtype=torch.long)
    states = state.expand(len(ACTION_NAMES), -1).clone()
    observed = model.executor.hard_step(states, actions)
    assert observed.tolist() == [
        [4, 10, 20, 30, 40],
        [10, 1, 20, 30, 40],
        [1, 10, 30, 20, 40],
        [1, 10, 20, 40, 30],
    ]
    slots = torch.arange(5, dtype=torch.long)
    answers = model.query.hard_read(state.expand(5, -1), slots)
    assert answers.tolist() == [1, 10, 20, 30, 40]


def test_evidence_interface() -> None:
    model = DIVERGEMEI1(MEI1Config())
    features = torch.randn(3, 17, 192)
    mask = torch.ones(3, 17, dtype=torch.bool)
    logits = model.evidence(features, mask)
    assert logits.shape == (3, 10, 128)
    before, after = model.evidence.hard_states(features, mask)
    assert before.shape == after.shape == (3, 5)


def test_factorized_model_execution_and_guards() -> None:
    model = DIVERGEMEI1(MEI1Config())
    _install_exact_test_weights(model)
    choices = (
        (
            _choice(0, 0, (), "background"),
            _choice(0, 1, (0, 1), "add-then-swap"),
            _choice(0, 2, (1, 0), "swap-then-add"),
        ),
        (
            _choice(1, 0, (2,), "swap23"),
            _choice(1, 1, (3,), "swap34"),
        ),
    )
    initial = ModelState((1, 10, 20, 30, 40))
    execution = execute_model_mdd(initial, choices, model.executor)
    assert not execution.overflow
    assert execution.represented_worlds == 6
    assert support_contains(execution, (1, 0))
    probe_before = (
        ModelState((2, 11, 21, 31, 41)),
        ModelState((3, 12, 22, 32, 42)),
    )
    probe_after = (
        ModelState((11, 5, 21, 31, 41)),
        ModelState((3, 12, 32, 22, 42)),
    )
    allowed = derive_model_allowed(
        choices, probe_before, probe_after, model.executor
    )
    assert allowed == {0: frozenset({1}), 1: frozenset({0})}
    decision = query_model_mdd(execution, 0, model.query, allowed=allowed)
    assert decision.disposition == "ANSWER"
    assert decision.answer == 10
    open_decision = query_model_mdd(execution, 0, model.query)
    assert open_decision.disposition == "ABSTAIN"


def test_source_and_receipt() -> None:
    assert source_audit()["pass"]
    receipt = architecture_receipt(MEI1Config())
    assert receipt["whole_hypothesis_only"]
    assert receipt["trainable_parameters"] > 0


def run_standalone_tests() -> None:
    test_action_tokens()
    test_learned_executor_and_query()
    test_evidence_interface()
    test_factorized_model_execution_and_guards()
    test_source_and_receipt()
    print("DIVERGE-MEI1 runtime tests passed", flush=True)


if __name__ == "__main__":
    run_standalone_tests()
