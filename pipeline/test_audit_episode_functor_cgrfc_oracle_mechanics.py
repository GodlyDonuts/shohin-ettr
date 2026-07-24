from __future__ import annotations

import torch

from pipeline.audit_episode_functor_cgrfc_oracle_mechanics import (
    _metrics,
    _oracle_board,
)
from episode_functor_conflict_reentrant_revision import (
    ConflictGatedReentrantRevision,
)


def test_oracle_board_is_deterministic_local_and_faulted() -> None:
    first, transitions, observers = _oracle_board(
        3,
        seed=19,
        record_width=32,
        fault_margin=0.5,
    )
    second, transitions_two, observers_two = _oracle_board(
        3,
        seed=19,
        record_width=32,
        fault_margin=0.5,
    )
    assert torch.equal(first.transition_logits, second.transition_logits)
    assert torch.equal(first.claim_logits, second.claim_logits)
    assert torch.equal(transitions, transitions_two)
    assert torch.equal(observers, observers_two)
    module = ConflictGatedReentrantRevision(
        record_width=32,
        controller_width=128,
        cycles=2,
        max_step=1.0,
    )
    metrics = _metrics(
        module,
        first,
        transitions,
        observers,
        routing_mode="causal",
    )
    assert metrics.exact_machines == 0
    assert metrics.transition_cells < metrics.transition_total
    assert metrics.observer_cells < metrics.observer_total
