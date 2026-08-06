"""Autonomous contradiction-guided replay over FTA1 typed packets."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from diverge_ats1_data import supervisor_states
from diverge_ats1_runtime import (
    ATS1RuntimeError,
    CompiledSegment,
    TypedState,
    execute_step,
    render_typed_state,
    shifted_operation,
)


class FTA1AutonomousError(RuntimeError):
    """An autonomous typed replay receipt violates the frozen contract."""


ABLATIONS = (
    "normal",
    "trust_source",
    "ignore_first_conflict",
    "initial_swap",
    "operation_shift",
)


def _initial_states(
    rows: Sequence[dict[str, Any]],
    compiled: dict[tuple[str, int, str], CompiledSegment],
    *,
    swap: bool,
) -> dict[str, TypedState]:
    starts: dict[str, TypedState] = {}
    by_family: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        identity = str(row["identity_sha256"])
        packet = compiled.get((identity, 0, "wrong"))
        if packet is None:
            continue
        starts[identity] = packet.lhs
        by_family[str(row["family"])].append(identity)
    if not swap:
        return starts
    output = dict(starts)
    for identities in by_family.values():
        if len(identities) < 2:
            raise FTA1AutonomousError("initial swap needs two rows per family")
        values = [starts[identity] for identity in identities]
        output.update(zip(identities, values[-1:] + values[:-1], strict=True))
    return output


def _rates(counts: Counter[str]) -> dict[str, float]:
    rows = max(1, counts["rows"])
    steps = max(1, counts["active_steps"])
    return {
        "initial_exact": counts["initial_exact"] / rows,
        "selection_exact": counts["selection_exact"] / rows,
        "step_exact": counts["exact_steps"] / steps,
        "trajectory_exact": counts["trajectory_exact"] / rows,
        "terminal_exact": counts["terminal_exact"] / rows,
        "invalid": counts["invalid"] / rows,
    }


def evaluate_autonomous_replay(
    rows: Sequence[dict[str, Any]],
    compiled: dict[tuple[str, int, str], CompiledSegment],
    *,
    ablation: str = "normal",
) -> dict[str, Any]:
    """Find the first typed contradiction, commit once, and replay to terminal."""

    if ablation not in ABLATIONS:
        raise FTA1AutonomousError("autonomous replay ablation differs")
    starts = _initial_states(rows, compiled, swap=ablation == "initial_swap")
    totals: Counter[str] = Counter()
    per_family: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []

    for row in rows:
        identity = str(row["identity_sha256"])
        family = str(row["family"])
        depth = int(row["depth"])
        target_selection = int(row["error_index"])
        expected = supervisor_states(row)
        counts = per_family[family]
        totals["rows"] += 1
        counts["rows"] += 1
        totals["active_steps"] += depth
        counts["active_steps"] += depth
        state = starts.get(identity)
        selection = 0
        repairing = False
        ignored_conflict = False
        exact_steps = 0
        invalid = state is None
        if state is not None:
            initial_exact = render_typed_state(state) == expected[0]
            totals["initial_exact"] += initial_exact
            counts["initial_exact"] += initial_exact
        try:
            if state is None:
                raise ATS1RuntimeError("initial typed packet is missing")
            for step_index in range(depth):
                packet = compiled[(identity, step_index, "wrong")]
                if ablation == "trust_source":
                    state = packet.rhs_claim
                else:
                    operation = packet.operation_id
                    if ablation == "operation_shift":
                        operation = shifted_operation(operation)
                    computed = execute_step(state, operation, packet.arguments)
                    if repairing:
                        state = computed
                    else:
                        conflict = state != packet.lhs or computed != packet.rhs_claim
                        if conflict and ablation == "ignore_first_conflict" and not ignored_conflict:
                            ignored_conflict = True
                            state = packet.rhs_claim
                        elif conflict:
                            selection = step_index + 1
                            repairing = True
                            state = computed
                        else:
                            state = packet.rhs_claim
                exact_steps += render_typed_state(state) == expected[step_index + 1]
            terminal = render_typed_state(state)
        except (ATS1RuntimeError, KeyError):
            invalid = True
            terminal = "<INVALID>"

        selection_exact = selection == target_selection
        trajectory_exact = not invalid and exact_steps == depth
        terminal_exact = not invalid and terminal == str(row["answer"])
        for destination in (totals, counts):
            destination["invalid"] += invalid
            destination["selection_exact"] += selection_exact
            destination["exact_steps"] += exact_steps
            destination["trajectory_exact"] += trajectory_exact
            destination["terminal_exact"] += terminal_exact
            destination["selected_none"] += selection == 0
            destination["selected_early"] += 0 < selection < target_selection
            destination["selected_late"] += selection > target_selection
        if len(examples) < 24:
            examples.append(
                {
                    "identity_sha256": identity,
                    "family": family,
                    "selection": selection,
                    "target_selection": target_selection,
                    "prediction": terminal,
                    "target": str(row["answer"]),
                    "selection_exact": selection_exact,
                    "terminal_exact": terminal_exact,
                    "trajectory_exact": trajectory_exact,
                    "invalid": invalid,
                }
            )

    return {
        "ablation": ablation,
        "counts": dict(totals),
        "rates": _rates(totals),
        "per_family": {
            family: {"counts": dict(values), "rates": _rates(values)}
            for family, values in sorted(per_family.items())
        },
        "examples": examples,
    }


__all__ = [
    "ABLATIONS",
    "FTA1AutonomousError",
    "evaluate_autonomous_replay",
]
